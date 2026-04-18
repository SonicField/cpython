/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Pure C LivenessAnalysis — replaces liveness_c.cpp.
 * Uses PhxDataFlowAnalyzer from dataflow_c.h for backward liveness analysis.
 */

/* Avoid including liveness_c.h here — it pulls in hir_c_api.h which
 * defines HirBasicBlock/HirCFG as void*, conflicting with the struct
 * definitions in hir_basic_block_c.h that we need for direct access. */
#include "cinderx/Jit/dataflow_c.h"
#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/hir/hir_basic_block_c.h"
#include "cinderx/Common/jit_log_c.h"
#include "Python.h"

typedef void* HirFunction;
typedef void* HirInstr;
typedef void* HirRegister;

typedef struct HirLivenessState HirLivenessState;
int hir_liveness_verify(HirFunction func, const HirLivenessState *c_state);

#include <string.h>

/* ---- Last-use hash table ---- */

typedef struct {
    const void *instr;
    PhxBitVector regs;
} PhxLastUseEntry;

typedef struct {
    PhxLastUseEntry *entries;
    size_t capacity;
    size_t count;
    size_t num_bits;
} PhxLastUseTable;

static void lu_init(PhxLastUseTable *t, size_t num_bits) {
    t->capacity = 64;
    t->count = 0;
    t->num_bits = num_bits;
    t->entries = (PhxLastUseEntry *)PyMem_RawCalloc(t->capacity, sizeof(PhxLastUseEntry));
}

static void lu_destroy(PhxLastUseTable *t) {
    for (size_t i = 0; i < t->capacity; i++) {
        if (t->entries[i].instr) {
            phx_bv_destroy(&t->entries[i].regs);
        }
    }
    PyMem_RawFree(t->entries);
}

static size_t lu_hash(const void *ptr) {
    uintptr_t v = (uintptr_t)ptr;
    v = (v >> 4) ^ (v >> 16);
    return (size_t)v;
}

static void lu_grow(PhxLastUseTable *t);

static PhxLastUseEntry *lu_find_or_create(PhxLastUseTable *t, const void *instr) {
    if (t->count * 2 >= t->capacity) {
        lu_grow(t);
    }
    size_t mask = t->capacity - 1;
    size_t idx = lu_hash(instr) & mask;
    for (;;) {
        if (t->entries[idx].instr == instr) {
            return &t->entries[idx];
        }
        if (t->entries[idx].instr == NULL) {
            t->entries[idx].instr = instr;
            phx_bv_init(&t->entries[idx].regs, t->num_bits);
            t->count++;
            return &t->entries[idx];
        }
        idx = (idx + 1) & mask;
    }
}

static const PhxLastUseEntry *lu_find(const PhxLastUseTable *t, const void *instr) {
    size_t mask = t->capacity - 1;
    size_t idx = lu_hash(instr) & mask;
    for (;;) {
        if (t->entries[idx].instr == instr) {
            return &t->entries[idx];
        }
        if (t->entries[idx].instr == NULL) {
            return NULL;
        }
        idx = (idx + 1) & mask;
    }
}

static void lu_grow(PhxLastUseTable *t) {
    size_t old_cap = t->capacity;
    PhxLastUseEntry *old = t->entries;

    t->capacity = old_cap * 2;
    t->entries = (PhxLastUseEntry *)PyMem_RawCalloc(t->capacity, sizeof(PhxLastUseEntry));
    t->count = 0;

    for (size_t i = 0; i < old_cap; i++) {
        if (old[i].instr) {
            PhxLastUseEntry *e = lu_find_or_create(t, old[i].instr);
            phx_bv_destroy(&e->regs);
            e->regs = old[i].regs;
        }
    }
    PyMem_RawFree(old);
}

/* ---- Liveness state ---- */

struct HirLivenessState {
    PhxDataFlowAnalyzer analyzer;
    PhxDataFlowBlock *df_blocks;
    PhxDataFlowBlock df_entry;
    PhxDataFlowBlock df_exit;
    size_t n_cfg_blocks;

    int max_block_id;
    size_t *block_id_to_idx;
    void **cfg_blocks;

    PhxLastUseTable last_uses;
};

/* ---- analyzeInstrLiveness in C ----
 * Mirrors the C++ template: processes output, skips Phi uses,
 * visits regular uses, then visits Phi inputs at successors. */

typedef void (*LivenessDefineFunc)(void *reg, void *ctx);
typedef void (*LivenessUseFunc)(void *reg, void *ctx);

typedef struct {
    LivenessUseFunc use_fn;
    void *ctx;
} UseVisitorCtx;

static int use_visitor_cb(void **reg_slot, void *ctx_raw) {
    UseVisitorCtx *vc = (UseVisitorCtx *)ctx_raw;
    vc->use_fn(*reg_slot, vc->ctx);
    return 1;
}

static void analyze_instr_liveness(
    void *instr,
    LivenessDefineFunc define_output,
    LivenessUseFunc use_fn,
    void *ctx)
{
    void *output = hir_c_output(instr);
    if (output) {
        define_output(output, ctx);
    }

    if (hir_c_is_phi(instr)) {
        return;
    }

    UseVisitorCtx vc;
    vc.use_fn = use_fn;
    vc.ctx = ctx;
    hir_c_visit_uses(instr, use_visitor_cb, &vc);

    size_t n_edges = hir_c_num_edges(instr);
    if (n_edges > 0) {
        void *my_block = ((HirInstrLayout *)instr)->block;
        for (size_t ei = 0; ei < n_edges; ei++) {
            HirBasicBlock *succ = (HirBasicBlock *)hir_c_successor(instr, ei);
            int phi_idx = -1;
            void *si = hir_bb_first_instr(succ);
            while (si) {
                if (!hir_c_is_phi(si)) break;
                if (phi_idx == -1) {
                    phi_idx = (int)hir_phi_block_index(si, (const HirBasicBlock *)my_block);
                }
                void *phi_op = hir_c_get_operand(si, (size_t)phi_idx);
                use_fn(phi_op, ctx);
                si = hir_bb_next_instr(succ, si);
            }
        }
    }
}

/* ---- gen/kill computation context ---- */

typedef struct {
    PhxDataFlowAnalyzer *analyzer;
    PhxDataFlowBlock *df_block;
    PhxBitVector gen_bits;
    PhxBitVector kill_bits;
} GenKillCtx;

static void gk_define(void *reg, void *ctx_raw) {
    GenKillCtx *gk = (GenKillCtx *)ctx_raw;
    int rid = hir_reg_id(reg);
    size_t idx = gk->analyzer->obj_id_to_index[rid];
    phx_bv_set_bit(&gk->kill_bits, idx, 1);
    phx_bv_set_bit(&gk->gen_bits, idx, 0);
}

static void gk_use(void *reg, void *ctx_raw) {
    GenKillCtx *gk = (GenKillCtx *)ctx_raw;
    int rid = hir_reg_id(reg);
    size_t idx = gk->analyzer->obj_id_to_index[rid];
    phx_bv_set_bit(&gk->gen_bits, idx, 1);
}

/* ---- last-uses computation context ---- */

typedef struct {
    PhxBitVector *live;
    PhxLastUseTable *last_uses;
    PhxDataFlowAnalyzer *analyzer;
    const void *instr;
} LastUseCtx;

static void lu_define(void *reg, void *ctx_raw) {
    LastUseCtx *lu = (LastUseCtx *)ctx_raw;
    int rid = hir_reg_id(reg);
    size_t idx = lu->analyzer->obj_id_to_index[rid];
    if (!phx_bv_get_bit(lu->live, idx)) {
        PhxLastUseEntry *e = lu_find_or_create(lu->last_uses, lu->instr);
        phx_bv_set_bit(&e->regs, idx, 1);
    } else {
        phx_bv_set_bit(lu->live, idx, 0);
    }
}

static void lu_use(void *reg, void *ctx_raw) {
    LastUseCtx *lu = (LastUseCtx *)ctx_raw;
    int rid = hir_reg_id(reg);
    size_t idx = lu->analyzer->obj_id_to_index[rid];
    if (!phx_bv_get_bit(lu->live, idx)) {
        PhxLastUseEntry *e = lu_find_or_create(lu->last_uses, lu->instr);
        phx_bv_set_bit(&e->regs, idx, 1);
        phx_bv_set_bit(lu->live, idx, 1);
    }
}

/* ---- Public API ---- */

HirLivenessState *hir_liveness_create(HirFunction func) {
    HirLivenessState *state = (HirLivenessState *)PyMem_RawCalloc(1, sizeof(HirLivenessState));
    HirEnvironment *env = hir_func_env(func);
    size_t reg_count = hir_env_reg_count(env);
    void **reg_data = hir_env_reg_data(env);

    int max_reg_id = hir_env_next_register_id(env);
    phx_df_init(&state->analyzer, (size_t)max_reg_id);

    for (size_t i = 0; i < reg_count; i++) {
        if (reg_data[i]) {
            phx_df_add_object(&state->analyzer, reg_data[i], (size_t)hir_reg_id(reg_data[i]));
        }
    }

    HirCFG *cfg = (HirCFG *)hir_func_cfg_ptr(func);

    size_t n_blocks = 0;
    int max_bid = 0;
    for (HirBasicBlock *bb = hir_cfg_first_block(cfg); bb; bb = hir_cfg_next_block(cfg, bb)) {
        n_blocks++;
        if (bb->id > max_bid) max_bid = bb->id;
    }
    state->n_cfg_blocks = n_blocks;
    state->max_block_id = max_bid;
    state->df_blocks = (PhxDataFlowBlock *)PyMem_RawCalloc(n_blocks, sizeof(PhxDataFlowBlock));
    state->block_id_to_idx = (size_t *)PyMem_RawCalloc((size_t)(max_bid + 1), sizeof(size_t));
    state->cfg_blocks = (void **)PyMem_RawMalloc(n_blocks * sizeof(void *));

    size_t bi = 0;
    for (HirBasicBlock *bb = hir_cfg_first_block(cfg); bb; bb = hir_cfg_next_block(cfg, bb)) {
        state->block_id_to_idx[bb->id] = bi;
        state->cfg_blocks[bi] = bb;

        PhxDataFlowBlock *dfb = &state->df_blocks[bi];
        phx_df_block_init(dfb);
        phx_df_add_block(&state->analyzer, dfb);

        GenKillCtx gk;
        gk.analyzer = &state->analyzer;
        gk.df_block = dfb;
        phx_bv_init(&gk.gen_bits, state->analyzer.num_bits);
        phx_bv_init(&gk.kill_bits, state->analyzer.num_bits);

        void *instr = hir_bb_last_instr(bb);
        while (instr) {
            analyze_instr_liveness(instr, gk_define, gk_use, &gk);
            instr = hir_bb_prev_instr(bb, instr);
        }

        phx_bv_copy(&dfb->gen, &gk.gen_bits);
        phx_bv_copy(&dfb->kill, &gk.kill_bits);
        phx_bv_destroy(&gk.gen_bits);
        phx_bv_destroy(&gk.kill_bits);

        bi++;
    }

    phx_df_block_init(&state->df_entry);
    phx_df_add_block(&state->analyzer, &state->df_entry);
    phx_df_set_entry(&state->analyzer, &state->df_entry);

    phx_df_block_init(&state->df_exit);
    phx_df_add_block(&state->analyzer, &state->df_exit);
    phx_df_set_exit(&state->analyzer, &state->df_exit);

    for (bi = 0; bi < n_blocks; bi++) {
        HirBasicBlock *bb = (HirBasicBlock *)state->cfg_blocks[bi];
        PhxDataFlowBlock *dfb = &state->df_blocks[bi];

        if (bb == (HirBasicBlock *)cfg->entry_block) {
            phx_df_block_connect(&state->df_entry, dfb);
        }

        size_t n_out = hir_bb_out_edges_count(bb);
        if (n_out == 0) {
            phx_df_block_connect(dfb, &state->df_exit);
        } else {
            for (size_t e = 0; e < n_out; e++) {
                const HirEdge *edge = hir_bb_out_edge(bb, e);
                HirBasicBlock *succ = (HirBasicBlock *)edge->to;
                size_t succ_idx = state->block_id_to_idx[succ->id];
                phx_df_block_connect(dfb, &state->df_blocks[succ_idx]);
            }
        }
    }

    phx_df_run(&state->analyzer, 0);

    lu_init(&state->last_uses, state->analyzer.num_bits);

    for (bi = 0; bi < n_blocks; bi++) {
        HirBasicBlock *bb = (HirBasicBlock *)state->cfg_blocks[bi];
        PhxDataFlowBlock *dfb = &state->df_blocks[bi];

        PhxBitVector live;
        phx_bv_init(&live, state->analyzer.num_bits);
        phx_bv_copy(&live, &dfb->out);

        void *instr = hir_bb_last_instr(bb);
        while (instr) {
            LastUseCtx lu;
            lu.live = &live;
            lu.last_uses = &state->last_uses;
            lu.analyzer = &state->analyzer;
            lu.instr = instr;
            analyze_instr_liveness(instr, lu_define, lu_use, &lu);
            instr = hir_bb_prev_instr(bb, instr);
        }
        phx_bv_destroy(&live);
    }

    JIT_DCHECK_C(hir_liveness_verify(func, state),
                 "C LivenessAnalysis diverges from C++");

    return state;
}

int hir_liveness_is_last_use(
    const HirLivenessState *state, HirInstr instr, HirRegister reg)
{
    const PhxLastUseEntry *e = lu_find(&state->last_uses, instr);
    if (!e) return 0;
    int rid = hir_reg_id(reg);
    if (rid < 0 || (size_t)rid > state->analyzer.max_obj_id) return 0;
    size_t idx = state->analyzer.obj_id_to_index[rid];
    return phx_bv_get_bit(&e->regs, idx);
}

void hir_liveness_destroy(HirLivenessState *state) {
    lu_destroy(&state->last_uses);
    for (size_t i = 0; i < state->n_cfg_blocks; i++) {
        phx_df_block_destroy(&state->df_blocks[i]);
    }
    phx_df_block_destroy(&state->df_entry);
    phx_df_block_destroy(&state->df_exit);
    phx_df_destroy(&state->analyzer);
    PyMem_RawFree(state->df_blocks);
    PyMem_RawFree(state->block_id_to_idx);
    PyMem_RawFree(state->cfg_blocks);
    PyMem_RawFree(state);
}
