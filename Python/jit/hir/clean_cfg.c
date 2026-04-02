/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Clean CFG pass — combines block absorption, phi elimination,
 * and unreachable code removal to simplify the control flow graph.
 */

#include "cinderx/Jit/hir/clean_cfg_c.h"
#include "cinderx/Jit/hir/phi_elimination_c.h"
#include "cinderx/Jit/hir/hir_c_api.h"

#include <assert.h>
#include <stdlib.h>

/* Try to absorb the successor block into block when:
 * - block ends with an unconditional Branch
 * - the target has exactly 1 predecessor
 * - the target is not block itself (no self-loops)
 * Returns 1 if absorption happened, 0 otherwise. */
static int absorb_dst_block(HirBasicBlock block) {
    HirInstr term = hir_block_terminator(block);
    if (!hir_instr_is_branch(term)) {
        return 0;
    }

    HirBasicBlock target = hir_branch_target(term);
    if (target == block) {
        return 0;
    }
    if (hir_block_in_edges_count(target) != 1) {
        return 0;
    }

    /* Unlink the branch instruction */
    hir_instr_unlink(term);

    /* Move all instructions from target into block */
    while (!hir_block_empty(target)) {
        HirInstr instr = hir_block_pop_front(target);
        assert(!hir_instr_is_phi(instr));
        hir_block_append(block, instr);
    }

    /* Successors of the new terminator may have Phis referencing target.
     * Retarget them to refer to block. */
    HirInstr new_term = hir_block_terminator(block);
    assert(new_term != NULL);
    size_t num_edges = hir_instr_num_edges(new_term);
    for (size_t i = 0; i < num_edges; i++) {
        HirBasicBlock succ = hir_instr_successor(new_term, i);
        hir_block_fixup_phis(succ, target, block);
    }

    /* Delete the old branch instruction */
    hir_instr_delete(term);
    return 1;
}

void hir_clean_cfg_run(HirFunction func) {
    HirCFG cfg = hir_func_cfg(func);
    int changed = 0;

    do {
        hir_remove_unreachable_instructions(func);
        hir_phi_elimination_run(func);

        /* Get RPO traversal */
        size_t rpo_cap = 256;
        HirBasicBlock *rpo_blocks = (HirBasicBlock *)malloc(
            rpo_cap * sizeof(HirBasicBlock));
        if (!rpo_blocks) return;
        size_t num_blocks = hir_cfg_get_rpo(cfg, rpo_blocks, rpo_cap);
        if (num_blocks > rpo_cap) {
            rpo_cap = num_blocks;
            rpo_blocks = (HirBasicBlock *)realloc(
                rpo_blocks, rpo_cap * sizeof(HirBasicBlock));
            if (!rpo_blocks) return;
            hir_cfg_get_rpo(cfg, rpo_blocks, rpo_cap);
        }

        for (size_t i = 0; i < num_blocks; i++) {
            HirBasicBlock block = rpo_blocks[i];
            if (hir_block_empty(block)) {
                continue;
            }
            /* Keep absorbing successors until no more changes */
            while (absorb_dst_block(block)) {
                changed = 1;
            }
        }

        free(rpo_blocks);
    } while (hir_remove_unreachable_blocks(func));

    if (changed) {
        hir_reflow_types(func);
    }
}
