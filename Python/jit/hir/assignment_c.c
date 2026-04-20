/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Pure C AssignmentAnalysis — forward dataflow for definite/maybe assignment.
 */

#include "cinderx/Jit/hir/assignment_c.h"
#include "cinderx/Jit/dataflow_c.h"
#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/hir/hir_basic_block_c.h"
#include "cinderx/Common/jit_log_c.h"
#include "Python.h"

struct PhxAssignmentState {
    PhxDataFlowAnalyzer analyzer;
    PhxDataFlowBlock *df_blocks;
    PhxDataFlowBlock df_entry;
    PhxDataFlowBlock df_exit;
    size_t n_cfg_blocks;
    int max_block_id;
    size_t *block_id_to_idx;
    int is_definite;
};

PhxAssignmentState *phx_assign_create(HirFunction func, int is_definite) {
    PhxAssignmentState *state = (PhxAssignmentState *)PyMem_RawCalloc(1, sizeof(PhxAssignmentState));
    state->is_definite = is_definite;
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

    HirBasicBlock *entry_block = (HirBasicBlock *)cfg->entry_block;

    size_t bi = 0;
    for (HirBasicBlock *bb = hir_cfg_first_block(cfg); bb; bb = hir_cfg_next_block(cfg, bb)) {
        state->block_id_to_idx[bb->id] = bi;
        PhxDataFlowBlock *dfb = &state->df_blocks[bi];
        phx_df_block_init(dfb);
        phx_df_add_block(&state->analyzer, dfb);

        /* gen = args (if entry block) + all outputs in this block.
         * kill = empty (no DEL_FAST support). */
        if (bb == entry_block) {
            void *instr = hir_bb_first_instr(bb);
            while (instr) {
                if (hir_c_opcode(instr) == HIR_OP_LoadArg) {
                    void *out = hir_c_output(instr);
                    if (out) {
                        size_t idx = state->analyzer.obj_id_to_index[hir_reg_id(out)];
                        phx_bv_set_bit(&dfb->gen, idx, 1);
                    }
                }
                instr = hir_bb_next_instr(bb, instr);
            }
        }

        void *instr = hir_bb_first_instr(bb);
        while (instr) {
            void *out = hir_c_output(instr);
            if (out) {
                size_t idx = state->analyzer.obj_id_to_index[hir_reg_id(out)];
                phx_bv_set_bit(&dfb->gen, idx, 1);
            }
            instr = hir_bb_next_instr(bb, instr);
        }

        bi++;
    }

    phx_df_block_init(&state->df_entry);
    phx_df_add_block(&state->analyzer, &state->df_entry);
    phx_df_set_entry(&state->analyzer, &state->df_entry);

    phx_df_block_init(&state->df_exit);
    phx_df_add_block(&state->analyzer, &state->df_exit);
    phx_df_set_exit(&state->analyzer, &state->df_exit);

    for (bi = 0; bi < n_blocks; bi++) {
        HirBasicBlock *bb = hir_cfg_first_block(cfg);
        for (size_t j = 0; j < bi; j++) bb = hir_cfg_next_block(cfg, bb);
        PhxDataFlowBlock *dfb = &state->df_blocks[bi];

        if (bb == entry_block) {
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

    phx_df_run_ex(&state->analyzer, 1, is_definite, is_definite);

    return state;
}

int phx_assign_is_assigned_in(const PhxAssignmentState *state,
                              int block_id, HirRegister reg) {
    if (block_id < 0 || block_id > state->max_block_id) return 0;
    size_t bi = state->block_id_to_idx[block_id];
    const PhxDataFlowBlock *dfb = &state->df_blocks[bi];
    int rid = hir_reg_id(reg);
    if (rid < 0 || (size_t)rid > state->analyzer.max_obj_id) return 0;
    size_t idx = state->analyzer.obj_id_to_index[rid];
    return phx_bv_get_bit(&dfb->in, idx);
}

int phx_assign_is_assigned_out(const PhxAssignmentState *state,
                               int block_id, HirRegister reg) {
    if (block_id < 0 || block_id > state->max_block_id) return 0;
    size_t bi = state->block_id_to_idx[block_id];
    const PhxDataFlowBlock *dfb = &state->df_blocks[bi];
    int rid = hir_reg_id(reg);
    if (rid < 0 || (size_t)rid > state->analyzer.max_obj_id) return 0;
    size_t idx = state->analyzer.obj_id_to_index[rid];
    return phx_bv_get_bit(&dfb->out, idx);
}

void phx_assign_destroy(PhxAssignmentState *state) {
    for (size_t i = 0; i < state->n_cfg_blocks; i++) {
        phx_df_block_destroy(&state->df_blocks[i]);
    }
    phx_df_block_destroy(&state->df_entry);
    phx_df_block_destroy(&state->df_exit);
    phx_df_destroy(&state->analyzer);
    PyMem_RawFree(state->df_blocks);
    PyMem_RawFree(state->block_id_to_idx);
    PyMem_RawFree(state);
}
