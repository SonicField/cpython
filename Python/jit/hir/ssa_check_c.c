/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Pure C checkFunc — SSA verification for HIR functions.
 * Verifies CFG well-formedness and SSA register definitions.
 */

#include "cinderx/Jit/hir/ssa_check_c.h"
#include "cinderx/Jit/hir/assignment_c.h"
#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/hir/hir_basic_block_c.h"
#include "Python.h"

#include <stdio.h>
#include <string.h>

static int cfg_node_is_linked(const HirBasicBlock *bb) {
    return bb->cfg_node.prev_ != &bb->cfg_node;
}

static int check_cfg(HirCFG *cfg, int max_bid) {
    size_t arr_size = (size_t)(max_bid + 1);
    char *reachable = (char *)PyMem_RawCalloc(arr_size, 1);
    int *queue = (int *)PyMem_RawMalloc(arr_size * sizeof(int));
    size_t qh = 0, qt = 0;

    HirBasicBlock *entry = (HirBasicBlock *)cfg->entry_block;
    reachable[entry->id] = 1;
    queue[qt++] = entry->id;

    /* Map block IDs to block pointers */
    void **id_to_block = (void **)PyMem_RawCalloc(arr_size, sizeof(void *));
    for (HirBasicBlock *b = hir_cfg_first_block(cfg); b; b = hir_cfg_next_block(cfg, b))
        id_to_block[b->id] = b;

    while (qh < qt) {
        int bid = queue[qh++];
        HirBasicBlock *block = (HirBasicBlock *)id_to_block[bid];
        if (!cfg_node_is_linked(block)) {
            fprintf(stderr, "ERROR: Reachable bb %d isn't part of CFG\n", bid);
            PyMem_RawFree(reachable); PyMem_RawFree(queue); PyMem_RawFree(id_to_block);
            return 0;
        }
        size_t n_out = hir_bb_out_edges_count(block);
        for (size_t e = 0; e < n_out; e++) {
            const HirEdge *edge = hir_bb_out_edge(block, e);
            HirBasicBlock *succ = (HirBasicBlock *)edge->to;
            if (!reachable[succ->id]) {
                reachable[succ->id] = 1;
                queue[qt++] = succ->id;
            }
        }
    }

    int ok = 1;
    for (HirBasicBlock *block = hir_cfg_first_block(cfg); block; block = hir_cfg_next_block(cfg, block)) {
        if (!reachable[block->id]) {
            fprintf(stderr, "ERROR: CFG contains unreachable bb %d\n", block->id);
            ok = 0; goto cleanup;
        }

        char *seen = (char *)PyMem_RawCalloc(arr_size, 1);
        size_t n_in = hir_bb_in_edges_count(block);
        for (size_t e = 0; e < n_in; e++) {
            const HirEdge *edge = hir_bb_in_edge(block, e);
            HirBasicBlock *pred = (HirBasicBlock *)edge->from;
            if (!reachable[pred->id]) {
                fprintf(stderr, "ERROR: bb %d has unreachable predecessor bb %d\n",
                        block->id, pred->id);
                PyMem_RawFree(seen); ok = 0; goto cleanup;
            }
            if (seen[pred->id]) {
                fprintf(stderr, "ERROR: bb %d has > 1 edge from predecessor bb %d\n",
                        block->id, pred->id);
                PyMem_RawFree(seen); ok = 0; goto cleanup;
            }
            seen[pred->id] = 1;
        }
        PyMem_RawFree(seen);
    }

cleanup:
    PyMem_RawFree(reachable);
    PyMem_RawFree(queue);
    PyMem_RawFree(id_to_block);
    return ok;
}

int hir_check_func_c(HirFunction func) {
    HirCFG *cfg = (HirCFG *)hir_func_cfg_ptr(func);
    HirEnvironment *env = hir_func_env(func);
    int max_reg_id = hir_env_next_register_id(env);

    int max_bid = 0;
    for (HirBasicBlock *b = hir_cfg_first_block(cfg); b; b = hir_cfg_next_block(cfg, b))
        if (b->id > max_bid) max_bid = b->id;

    if (!check_cfg(cfg, max_bid))
        return 0;

    PhxAssignmentState *assign = phx_assign_create(func, 1);

    size_t arr_size = (size_t)(max_reg_id + 1);
    int *def_block = (int *)PyMem_RawMalloc(arr_size * sizeof(int));
    memset(def_block, -1, arr_size * sizeof(int));

    int ok = 1;
    HirBasicBlock *entry_block = (HirBasicBlock *)cfg->entry_block;

    for (HirBasicBlock *block = hir_cfg_first_block(cfg); block; block = hir_cfg_next_block(cfg, block)) {
        if (hir_bb_empty(block)) {
            fprintf(stderr, "ERROR: bb %d has no instructions\n", block->id);
            ok = 0; continue;
        }

        int phi_section = 1;
        int allow_prologue = (block == entry_block);

        void *instr = hir_bb_first_instr(block);
        while (instr) {
            int op = hir_c_opcode(instr);

            if (hir_c_is_phi(instr)) {
                if (!phi_section) {
                    fprintf(stderr, "ERROR: Phi in bb %d comes after non-Phi\n", block->id);
                    ok = 0;
                    instr = hir_bb_next_instr(block, instr);
                    continue;
                }

                HirPhi *phi = (HirPhi *)instr;
                size_t n_ops = hir_c_num_operands(instr);
                for (size_t i = 0; i < n_ops && i < phi->bb_count; i++) {
                    void *operand = hir_c_get_operand(instr, i);
                    HirBasicBlock *phi_bb = (HirBasicBlock *)phi->bb_data[i];
                    if (operand && !phx_assign_is_assigned_out(assign, phi_bb->id, operand)) {
                        fprintf(stderr, "ERROR: Phi input r%d not defined at end of bb %d\n",
                                hir_reg_id(operand), phi_bb->id);
                        ok = 0;
                    }
                }
            } else {
                phi_section = 0;

                size_t n_ops = hir_c_num_operands(instr);
                for (size_t i = 0; i < n_ops; i++) {
                    void *operand = hir_c_get_operand(instr, i);
                    if (operand && !phx_assign_is_assigned_in(assign, block->id, operand)) {
                        int rid = hir_reg_id(operand);
                        if (def_block[rid] < 0) {
                            fprintf(stderr, "ERROR: Operand r%d not defined at use in bb %d\n",
                                    rid, block->id);
                            ok = 0;
                        }
                    }
                }
            }

            if (op == HIR_OP_LoadArg || op == HIR_OP_LoadCurrentFunc ||
                op == HIR_OP_LoadFrame) {
                if (!allow_prologue) {
                    fprintf(stderr, "ERROR: %s in bb %d comes after non-Load* instruction\n",
                            hir_instr_info_name(op), block->id);
                    ok = 0;
                }
            } else {
                allow_prologue = 0;
            }

            /* Check terminator position */
            int is_last = (hir_bb_next_instr(block, instr) == NULL);
            if (hir_instr_info_is_terminator(op) && !is_last) {
                fprintf(stderr, "ERROR: bb %d has terminator in non-terminal position\n", block->id);
                ok = 0;
            }
            if (is_last && !hir_instr_info_is_terminator(op)) {
                fprintf(stderr, "ERROR: bb %d has no terminator at end\n", block->id);
                ok = 0;
            }

            /* Track definitions */
            void *output = hir_c_output(instr);
            if (output) {
                int rid = hir_reg_id(output);
                void *defining_instr = ((HirRegLayout *)output)->instr;
                if (defining_instr != instr) {
                    fprintf(stderr, "ERROR: r%d's instr doesn't match defining instruction in bb %d\n",
                            rid, block->id);
                    ok = 0;
                }
                if (def_block[rid] >= 0) {
                    fprintf(stderr, "ERROR: r%d redefined in bb %d; previous in bb %d\n",
                            rid, block->id, def_block[rid]);
                    ok = 0;
                }
                def_block[rid] = block->id;
            }

            instr = hir_bb_next_instr(block, instr);
        }
    }

    PyMem_RawFree(def_block);
    phx_assign_destroy(assign);
    return ok;
}
