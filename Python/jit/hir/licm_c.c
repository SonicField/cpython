/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Pure C LICM — Loop-Invariant Code Motion for guard hoisting.
 */

#include "cinderx/Jit/hir/dominator_c.h"
#include "cinderx/Jit/hir/hir_c_api.h"
#include "cinderx/Jit/hir/hir_cfg_rpo_c.h"
#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/hir/hir_basic_block_c.h"
#include "cinderx/Jit/hir/phx_ptr_queue.h"  /* X2a: PhxPtrQueue (was file-scoped PhxQueue) */
#include "cinderx/Common/jit_log_c.h"
#include "Python.h"

/* hir_reg_instr canonical decl in hir_c_api.h (post-W25b). */

#include <string.h>

/* ---- Lightweight block set (bit-array indexed by block ID) ---- */

typedef struct {
    uint8_t *bits;
    size_t cap;
} PhxBlockSet;

static void bset_init(PhxBlockSet *s, int max_id) {
    s->cap = (size_t)(max_id + 1);
    s->bits = (uint8_t *)PyMem_RawCalloc(s->cap, 1);
}

static void bset_destroy(PhxBlockSet *s) {
    PyMem_RawFree(s->bits);
}

static int bset_contains(const PhxBlockSet *s, int id) {
    return id >= 0 && (size_t)id < s->cap && s->bits[id];
}

static void bset_insert(PhxBlockSet *s, int id) {
    if (id >= 0 && (size_t)id < s->cap) s->bits[id] = 1;
}

/* PhxQueue → PhxPtrQueue promoted to phx_ptr_queue.h shared header
 * (X2a, supervisor 03:06:27Z disposition). Realloc OOM now loud-fails
 * via JIT_CHECK_C (was silent-skip W2-A1 violation in original). */

/* ---- Pointer array ---- */

typedef struct {
    void **data;
    size_t count, cap;
} PhxArr;

static void arr_init(PhxArr *a) { memset(a, 0, sizeof(*a)); }
static void arr_destroy(PhxArr *a) { PyMem_RawFree(a->data); }

static void arr_push(PhxArr *a, void *v) {
    if (a->count >= a->cap) {
        a->cap = a->cap ? a->cap * 2 : 8;
        a->data = (void **)PyMem_RawRealloc(a->data, a->cap * sizeof(void *));
    }
    a->data[a->count++] = v;
}

/* ---- Loop info ---- */

typedef struct {
    HirBasicBlock *header;
    PhxBlockSet body;
    HirBasicBlock *preheader;
} LoopInfo;

/* ---- Check if instruction is a hoistable guard ---- */

static int is_hoistable_guard(const void *instr) {
    int op = hir_c_opcode(instr);
    return op == HIR_OP_GuardType || op == HIR_OP_GuardIs;
}

/* ---- Check if register is defined outside loop ---- */

static int is_defined_outside_loop(void *reg, const PhxBlockSet *loop_body) {
    if (reg == NULL) return 1;
    void *def = hir_reg_instr(reg);
    if (def == NULL) return 1;
    HirBasicBlock *def_block = (HirBasicBlock *)((HirInstrLayout *)def)->block;
    if (def_block == NULL) return 1;
    return !bset_contains(loop_body, def_block->id);
}

/* ---- Visit-uses callback for checking all uses outside loop ---- */

typedef struct {
    const PhxBlockSet *loop_body;
    int all_outside;
} LoopCheckCtx;

static int check_use_outside_loop(void **reg_slot, void *ctx_raw) {
    LoopCheckCtx *ctx = (LoopCheckCtx *)ctx_raw;
    if (!is_defined_outside_loop(*reg_slot, ctx->loop_body)) {
        ctx->all_outside = 0;
        return 0;
    }
    return 1;
}

/* ---- Find all natural loops ---- */

static size_t find_loops(HirFunction func, PhxDominatorState *doms,
                         void **rpo, size_t rpo_count,
                         LoopInfo *loops, size_t loops_cap, int max_bid) {
    size_t n_loops = 0;

    for (size_t ri = 0; ri < rpo_count; ri++) {
        HirBasicBlock *block = (HirBasicBlock *)rpo[ri];
        size_t n_out = hir_bb_out_edges_count(block);

        for (size_t e = 0; e < n_out; e++) {
            const HirEdge *edge = hir_bb_out_edge(block, e);
            HirBasicBlock *target = (HirBasicBlock *)edge->to;
            if (target == NULL) continue;

            if (!phx_dom_dominates(doms, target->id, block->id)) continue;

            if (n_loops >= loops_cap) goto done;

            LoopInfo *loop = &loops[n_loops];
            loop->header = target;
            bset_init(&loop->body, max_bid);
            bset_insert(&loop->body, target->id);
            loop->preheader = NULL;

            if (block != target) {
                PhxPtrQueue worklist;
                phx_ptr_queue_init(&worklist);
                bset_insert(&loop->body, block->id);
                phx_ptr_queue_push(&worklist, block);

                while (!phx_ptr_queue_empty(&worklist)) {
                    HirBasicBlock *cur = (HirBasicBlock *)phx_ptr_queue_pop(&worklist);
                    size_t n_in = hir_bb_in_edges_count(cur);
                    for (size_t pi = 0; pi < n_in; pi++) {
                        const HirEdge *pred_edge = hir_bb_in_edge(cur, pi);
                        HirBasicBlock *pred = (HirBasicBlock *)pred_edge->from;
                        if (pred != NULL && !bset_contains(&loop->body, pred->id)) {
                            bset_insert(&loop->body, pred->id);
                            phx_ptr_queue_push(&worklist, pred);
                        }
                    }
                }
                phx_ptr_queue_destroy(&worklist);
            }

            HirBasicBlock *preheader_candidate = NULL;
            int non_loop_preds = 0;
            size_t n_in = hir_bb_in_edges_count(target);
            for (size_t pi = 0; pi < n_in; pi++) {
                const HirEdge *pred_edge = hir_bb_in_edge(target, pi);
                HirBasicBlock *pred = (HirBasicBlock *)pred_edge->from;
                if (pred != NULL && !bset_contains(&loop->body, pred->id)) {
                    preheader_candidate = pred;
                    non_loop_preds++;
                }
            }
            if (non_loop_preds == 1) {
                loop->preheader = preheader_candidate;
            }

            if (loop->preheader != NULL) {
                n_loops++;
            } else {
                bset_destroy(&loop->body);
            }
        }
    }
done:
    return n_loops;
}

/* ---- Hoist invariant guards from a single loop ---- */

static int hoist_invariant_guards(LoopInfo *loop, HirCFG *cfg) {
    int hoisted = 0;
    PhxArr to_hoist;
    arr_init(&to_hoist);

    for (HirBasicBlock *block = hir_cfg_first_block(cfg); block;
         block = hir_cfg_next_block(cfg, block)) {
        if (!bset_contains(&loop->body, block->id)) continue;
        if (block == loop->preheader) continue;

        void *instr = hir_bb_first_instr(block);
        while (instr) {
            void *next = hir_bb_next_instr(block, instr);
            if (!is_hoistable_guard(instr) || hir_c_is_phi(instr)) {
                instr = next;
                continue;
            }
            LoopCheckCtx ctx;
            ctx.loop_body = &loop->body;
            ctx.all_outside = 1;
            hir_c_visit_uses(instr, check_use_outside_loop, &ctx);
            if (ctx.all_outside) {
                arr_push(&to_hoist, instr);
            }
            instr = next;
        }
    }

    for (size_t i = 0; i < to_hoist.count; i++) {
        void *instr = to_hoist.data[i];
        hir_instr_unlink(instr);
        void *terminator = hir_bb_last_instr(loop->preheader);
        if (terminator) {
            hir_c_insert_before(instr, terminator);
        } else {
            hir_bb_append_instr(loop->preheader, instr);
        }
        hoisted++;
    }

    arr_destroy(&to_hoist);
    return hoisted;
}

/* ---- Public API ---- */

void hir_licm_run(HirFunction func) {
    HirCFG *cfg = (HirCFG *)hir_func_cfg_ptr(func);

    size_t n_blocks = 0;
    int max_bid = 0;
    for (HirBasicBlock *b = hir_cfg_first_block(cfg); b; b = hir_cfg_next_block(cfg, b)) {
        n_blocks++;
        if (b->id > max_bid) max_bid = b->id;
    }

    if (n_blocks == 0) return;

    void **rpo = (void **)PyMem_RawMalloc(n_blocks * sizeof(void *));
    size_t rpo_count = hir_cfg_get_rpo_c(cfg, rpo, n_blocks);

    PhxDominatorState *doms = phx_dom_create(func);

    LoopInfo *loops = (LoopInfo *)PyMem_RawMalloc(64 * sizeof(LoopInfo));
    size_t n_loops = find_loops(func, doms, rpo, rpo_count, loops, 64, max_bid);

    for (size_t i = 0; i < n_loops; i++) {
        hoist_invariant_guards(&loops[i], cfg);
    }

    for (size_t i = 0; i < n_loops; i++) {
        bset_destroy(&loops[i].body);
    }
    PyMem_RawFree(loops);
    phx_dom_destroy(doms);
    PyMem_RawFree(rpo);
}
