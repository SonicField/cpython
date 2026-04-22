/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Pure C SSAify — Braun et al. SSA construction algorithm.
 * "Simple and Efficient Construction of Static Single Assignment Form"
 */

#include "cinderx/Jit/hir/ssaify_c.h"
#include "cinderx/Jit/hir/hir_cfg_rpo_c.h"
#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/hir/hir_basic_block_c.h"
#include "cinderx/Common/jit_log_c.h"
#include "Python.h"

#include <string.h>

/* Forward declarations for opaque API functions (avoid hir_c_api.h conflict) */
extern void *hir_func_alloc_register(void *func);
extern void hir_reflow_types(void *func);
extern void hir_phi_elimination_run(void *func);

/* ---- Simple hash table: void* → void* ---- */

typedef struct {
    void *key;
    void *val;
} PhxKV;

typedef struct {
    PhxKV *entries;
    size_t capacity;
    size_t count;
} PhxMap;

static void phx_map_init(PhxMap *m) {
    m->capacity = 16;
    m->count = 0;
    m->entries = (PhxKV *)PyMem_RawCalloc(m->capacity, sizeof(PhxKV));
}

static void phx_map_destroy(PhxMap *m) {
    PyMem_RawFree(m->entries);
}

static size_t phx_map_hash(const void *p) {
    uintptr_t v = (uintptr_t)p;
    return (size_t)((v >> 4) ^ (v >> 16));
}

static void phx_map_grow(PhxMap *m);

static void *phx_map_get(const PhxMap *m, const void *key) {
    size_t mask = m->capacity - 1;
    size_t idx = phx_map_hash(key) & mask;
    for (;;) {
        if (m->entries[idx].key == key) return m->entries[idx].val;
        if (m->entries[idx].key == NULL) return NULL;
        idx = (idx + 1) & mask;
    }
}

static int phx_map_contains(const PhxMap *m, const void *key) {
    size_t mask = m->capacity - 1;
    size_t idx = phx_map_hash(key) & mask;
    for (;;) {
        if (m->entries[idx].key == key) return 1;
        if (m->entries[idx].key == NULL) return 0;
        idx = (idx + 1) & mask;
    }
}

static void phx_map_set(PhxMap *m, void *key, void *val) {
    if (m->count * 2 >= m->capacity) phx_map_grow(m);
    size_t mask = m->capacity - 1;
    size_t idx = phx_map_hash(key) & mask;
    for (;;) {
        if (m->entries[idx].key == key) { m->entries[idx].val = val; return; }
        if (m->entries[idx].key == NULL) {
            m->entries[idx].key = key;
            m->entries[idx].val = val;
            m->count++;
            return;
        }
        idx = (idx + 1) & mask;
    }
}

static void phx_map_grow(PhxMap *m) {
    size_t old_cap = m->capacity;
    PhxKV *old = m->entries;
    m->capacity = old_cap * 2;
    m->entries = (PhxKV *)PyMem_RawCalloc(m->capacity, sizeof(PhxKV));
    m->count = 0;
    for (size_t i = 0; i < old_cap; i++) {
        if (old[i].key) phx_map_set(m, old[i].key, old[i].val);
    }
    PyMem_RawFree(old);
}

/* ---- Dynamic pointer array ---- */

typedef struct {
    void **data;
    size_t count;
    size_t cap;
} PhxPtrArr;

static void phx_arr_init(PhxPtrArr *a) { memset(a, 0, sizeof(*a)); }
static void phx_arr_destroy(PhxPtrArr *a) { PyMem_RawFree(a->data); }

static void phx_arr_push(PhxPtrArr *a, void *p) {
    if (a->count >= a->cap) {
        a->cap = a->cap ? a->cap * 2 : 4;
        a->data = (void **)PyMem_RawRealloc(a->data, a->cap * sizeof(void *));
    }
    a->data[a->count++] = p;
}

static int phx_arr_contains(const PhxPtrArr *a, const void *p) {
    for (size_t i = 0; i < a->count; i++)
        if (a->data[i] == p) return 1;
    return 0;
}

/* ---- SSABasicBlock ---- */

typedef struct PhxSSABlock {
    void *block;            /* HirBasicBlock* */
    int unsealed_preds;
    PhxPtrArr preds;        /* PhxSSABlock* */
    PhxPtrArr succs;        /* PhxSSABlock* */
    PhxMap local_defs;      /* Register* → Register* */
    PhxMap phi_nodes;       /* Register* → Phi* */
    PhxPtrArr inc_phi_regs; /* incomplete phi: original register */
    PhxPtrArr inc_phi_outs; /* incomplete phi: output register */
} PhxSSABlock;

/* ---- SSAify state ---- */

typedef struct {
    void *func;           /* HirFunction */
    void *null_reg;       /* singleton null register */
    PhxMap block_map;     /* BasicBlock* → PhxSSABlock* */
    PhxPtrArr allocs;     /* individually allocated PhxSSABlock* for cleanup */
} PhxSSAState;

static PhxSSABlock *ssa_alloc_block(PhxSSAState *st, void *bb) {
    PhxSSABlock *s = (PhxSSABlock *)PyMem_RawMalloc(sizeof(PhxSSABlock));
    memset(s, 0, sizeof(*s));
    s->block = bb;
    phx_arr_init(&s->preds);
    phx_arr_init(&s->succs);
    phx_map_init(&s->local_defs);
    phx_map_init(&s->phi_nodes);
    phx_arr_init(&s->inc_phi_regs);
    phx_arr_init(&s->inc_phi_outs);
    phx_arr_push(&st->allocs, s);
    return s;
}

static PhxSSABlock *ssa_get_or_create(PhxSSAState *st, void *bb) {
    void *existing = phx_map_get(&st->block_map, bb);
    if (existing) return (PhxSSABlock *)existing;
    PhxSSABlock *s = ssa_alloc_block(st, bb);
    phx_map_set(&st->block_map, bb, s);
    return s;
}

/* Forward declaration */
static void *ssa_get_define(PhxSSAState *st, PhxSSABlock *ssablock, void *reg);

static void *ssa_create_phi(PhxSSAState *st, PhxSSABlock *ssa_block,
                            void *out_reg, void **pred_blocks, void **pred_regs,
                            size_t n_preds) {
    void *phi = hir_c_alloc_instr(sizeof(HirPhi), n_preds);
    hir_c_init_instr(phi, HIR_OP_Phi);
    hir_c_set_output(phi, out_reg);
    for (size_t i = 0; i < n_preds; i++) {
        hir_c_set_operand(phi, i, pred_regs[i]);
    }
    HirPhi *p = (HirPhi *)phi;
    /* libc malloc to match hir_instr_c.h destroy paths (cannot include
     * pymem.h there, see W8 root-cause). */
    p->bb_data = (void **)malloc(n_preds * sizeof(void *));
    p->bb_count = n_preds;
    p->bb_cap = n_preds;
    memcpy(p->bb_data, pred_blocks, n_preds * sizeof(void *));

    /* Sort by block ID (hir_phi_block_index uses binary search) */
    for (size_t i = 1; i < n_preds; i++) {
        void *tmp_bb = p->bb_data[i];
        void *tmp_reg = hir_c_get_operand(phi, i);
        size_t j = i;
        while (j > 0 && ((HirBasicBlock *)p->bb_data[j-1])->id >
                         ((HirBasicBlock *)tmp_bb)->id) {
            p->bb_data[j] = p->bb_data[j-1];
            hir_c_set_operand(phi, j, hir_c_get_operand(phi, j-1));
            j--;
        }
        p->bb_data[j] = tmp_bb;
        hir_c_set_operand(phi, j, tmp_reg);
    }

    return phi;
}

static void ssa_maybe_add_phi(PhxSSAState *st, PhxSSABlock *ssa_block,
                              void *reg, void *out) {
    size_t n = ssa_block->preds.count;
    void **pred_blocks = (void **)PyMem_RawMalloc(n * sizeof(void *));
    void **pred_regs = (void **)PyMem_RawMalloc(n * sizeof(void *));

    for (size_t i = 0; i < n; i++) {
        PhxSSABlock *pred = (PhxSSABlock *)ssa_block->preds.data[i];
        pred_blocks[i] = pred->block;
        pred_regs[i] = ssa_get_define(st, pred, reg);
    }

    void *first_instr = hir_bb_first_instr((HirBasicBlock *)ssa_block->block);
    void *phi = ssa_create_phi(st, ssa_block, out, pred_blocks, pred_regs, n);
    if (first_instr) {
        hir_c_copy_bytecode_offset(phi, first_instr);
    }
    phx_map_set(&ssa_block->phi_nodes, out, phi);

    PyMem_RawFree(pred_blocks);
    PyMem_RawFree(pred_regs);
}

static void *ssa_get_define(PhxSSAState *st, PhxSSABlock *ssablock, void *reg) {
    void *local = phx_map_get(&ssablock->local_defs, reg);
    if (local) return local;

    if (ssablock->preds.count == 0) {
        if (st->null_reg == NULL) {
            HirBasicBlock *bb = (HirBasicBlock *)ssablock->block;
            void *it = hir_bb_first_instr(bb);
            while (it) {
                int op = hir_c_opcode(it);
                if (op != HIR_OP_LoadArg && op != HIR_OP_LoadCurrentFunc &&
                    op != HIR_OP_LoadFrame) break;
                it = hir_bb_next_instr(bb, it);
            }
            st->null_reg = hir_func_alloc_register(st->func);
            HirType nullptr_type = HIR_TYPE_NULLPTR;
            void *loadnull = hir_c_create_load_const(st->null_reg, nullptr_type);
            if (it) {
                hir_c_copy_bytecode_offset(loadnull, it);
                hir_c_insert_before_pure(loadnull, it, bb);
            } else {
                hir_bb_append_instr(bb, loadnull);
            }
        }
        phx_map_set(&ssablock->local_defs, reg, st->null_reg);
        return st->null_reg;
    }

    if (ssablock->unsealed_preds > 0) {
        void *phi_output = hir_func_alloc_register(st->func);
        phx_arr_push(&ssablock->inc_phi_regs, reg);
        phx_arr_push(&ssablock->inc_phi_outs, phi_output);
        phx_map_set(&ssablock->local_defs, reg, phi_output);
        return phi_output;
    }

    if (ssablock->preds.count == 1) {
        void *new_reg = ssa_get_define(st, (PhxSSABlock *)ssablock->preds.data[0], reg);
        phx_map_set(&ssablock->local_defs, reg, new_reg);
        return new_reg;
    }

    void *new_reg = hir_func_alloc_register(st->func);
    phx_map_set(&ssablock->local_defs, reg, new_reg);
    ssa_maybe_add_phi(st, ssablock, reg, new_reg);
    return phx_map_get(&ssablock->local_defs, reg);
}

static void ssa_fix_incomplete_phis(PhxSSAState *st, PhxSSABlock *ssa_block) {
    for (size_t i = 0; i < ssa_block->inc_phi_regs.count; i++) {
        void *reg = ssa_block->inc_phi_regs.data[i];
        void *out = ssa_block->inc_phi_outs.data[i];
        ssa_maybe_add_phi(st, ssa_block, reg, out);
    }
}

/* ---- visitUses callback for SSAify ---- */

typedef struct {
    PhxSSAState *st;
    PhxSSABlock *ssablock;
} SSAVisitCtx;

static int ssa_visit_use_cb(void **reg_slot, void *ctx_raw) {
    SSAVisitCtx *vc = (SSAVisitCtx *)ctx_raw;
    JIT_CHECK_C(*reg_slot != NULL, "Instructions should not have nullptr operands");
    *reg_slot = ssa_get_define(vc->st, vc->ssablock, *reg_slot);
    return 1;
}

/* Compare phis by output register ID (descending for push_front ordering) */
static int phi_cmp_desc(const void *a, const void *b) {
    const void *pa = *(const void **)a;
    const void *pb = *(const void **)b;
    int id_a = hir_reg_id(hir_c_output(pa));
    int id_b = hir_reg_id(hir_c_output(pb));
    return (id_b > id_a) - (id_b < id_a);
}

/* ---- Public API ---- */

void hir_ssaify_run_c(HirFunction func) {
    PhxSSAState st;
    memset(&st, 0, sizeof(st));
    st.func = func;
    phx_map_init(&st.block_map);
    phx_arr_init(&st.allocs);

    HirCFG *cfg = (HirCFG *)hir_func_cfg_ptr(func);

    size_t n_blocks = 0;
    int max_bid = 0;
    for (HirBasicBlock *b = hir_cfg_first_block(cfg); b; b = hir_cfg_next_block(cfg, b)) {
        n_blocks++;
        if (b->id > max_bid) max_bid = b->id;
    }

    void **rpo = (void **)PyMem_RawMalloc(n_blocks * sizeof(void *));
    size_t rpo_count = hir_cfg_get_rpo_c(cfg, rpo, n_blocks);

    /* Build SSABasicBlock graph */
    for (size_t ri = 0; ri < rpo_count; ri++) {
        HirBasicBlock *block = (HirBasicBlock *)rpo[ri];
        PhxSSABlock *ssablock = ssa_get_or_create(&st, block);

        size_t n_out = hir_bb_out_edges_count(block);
        for (size_t e = 0; e < n_out; e++) {
            const HirEdge *edge = hir_bb_out_edge(block, e);
            HirBasicBlock *succ = (HirBasicBlock *)edge->to;
            PhxSSABlock *succ_ssa = ssa_get_or_create(&st, succ);
            if (!phx_arr_contains(&succ_ssa->preds, ssablock)) {
                phx_arr_push(&succ_ssa->preds, ssablock);
                succ_ssa->unsealed_preds++;
                phx_arr_push(&ssablock->succs, succ_ssa);
            }
        }
    }

    /* Main SSA construction */
    for (size_t ri = 0; ri < rpo_count; ri++) {
        HirBasicBlock *block = (HirBasicBlock *)rpo[ri];
        PhxSSABlock *ssablock = (PhxSSABlock *)phx_map_get(&st.block_map, block);

        void *instr = hir_bb_first_instr(block);
        while (instr) {
            JIT_CHECK_C(!hir_c_is_phi(instr), "SSAify does not support Phis in input");

            SSAVisitCtx vc;
            vc.st = &st;
            vc.ssablock = ssablock;
            hir_c_visit_uses(instr, ssa_visit_use_cb, &vc);

            void *out_reg = hir_c_output(instr);
            if (out_reg) {
                void *new_reg = hir_func_alloc_register(st.func);
                hir_c_set_output(instr, new_reg);
                phx_map_set(&ssablock->local_defs, out_reg, new_reg);
            }

            instr = hir_bb_next_instr(block, instr);
        }

        for (size_t si = 0; si < ssablock->succs.count; si++) {
            PhxSSABlock *succ = (PhxSSABlock *)ssablock->succs.data[si];
            succ->unsealed_preds--;
            if (succ->unsealed_preds <= 0) {
                ssa_fix_incomplete_phis(&st, succ);
            }
        }
    }

    /* Realize phi functions */
    for (size_t ri = 0; ri < rpo_count; ri++) {
        HirBasicBlock *block = (HirBasicBlock *)rpo[ri];
        PhxSSABlock *ssablock = (PhxSSABlock *)phx_map_get(&st.block_map, block);

        /* Collect phis from phi_nodes map */
        PhxPtrArr phis;
        phx_arr_init(&phis);
        for (size_t i = 0; i < ssablock->phi_nodes.capacity; i++) {
            if (ssablock->phi_nodes.entries[i].key) {
                phx_arr_push(&phis, ssablock->phi_nodes.entries[i].val);
            }
        }

        /* Sort by output register ID (descending) for push_front */
        if (phis.count > 1) {
            qsort(phis.data, phis.count, sizeof(void *), phi_cmp_desc);
        }

        for (size_t i = 0; i < phis.count; i++) {
            hir_bb_push_front_instr(block, phis.data[i]);
        }
        phx_arr_destroy(&phis);
    }

    /* Cleanup */
    for (size_t i = 0; i < st.allocs.count; i++) {
        PhxSSABlock *s = (PhxSSABlock *)st.allocs.data[i];
        phx_arr_destroy(&s->preds);
        phx_arr_destroy(&s->succs);
        phx_map_destroy(&s->local_defs);
        phx_map_destroy(&s->phi_nodes);
        phx_arr_destroy(&s->inc_phi_regs);
        phx_arr_destroy(&s->inc_phi_outs);
        PyMem_RawFree(s);
    }
    phx_arr_destroy(&st.allocs);
    phx_map_destroy(&st.block_map);
    PyMem_RawFree(rpo);

    /* Post-SSA passes */
    hir_reflow_types(func);
    hir_phi_elimination_run(func);
}
