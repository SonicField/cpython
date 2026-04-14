/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Dynamic comparison elimination pass — C port.
 * Fuses Compare + IsTruthy + CondBranch into CompareBool + CondBranch.
 */

#include "cinderx/Jit/hir/dynamic_comparison_elimination_c.h"
#include "cinderx/Jit/hir/hir_c_api.h"
#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/hir/liveness_c.h"

#include <assert.h>
#include <stdlib.h>

void hir_dynamic_comparison_elimination_run(HirFunction func) {
    HirLivenessState *liveness = hir_liveness_create(func);

    HirCFG cfg = hir_func_cfg(func);
    HirBasicBlock block = hir_cfg_blocks_first(cfg);

    while (block != NULL) {
        HirBasicBlock next_block = hir_cfg_blocks_next(cfg, block);

        /* Get last instruction in block */
        HirInstr instr = hir_block_back(block);
        if (instr == NULL || !hir_c_is_condbranch(instr)) {
            block = next_block;
            continue;
        }

        /* Looking for: truthy = IsTruthy(compare); CondBranch truthy */
        HirRegister truthy_reg = hir_c_get_operand(instr, 0);
        assert(truthy_reg == hir_instr_get_operand(instr, 0));
        HirInstr truthy = hir_reg_instr(truthy_reg);
        if (!hir_c_is_istruthy(truthy) ||
            hir_c_block(truthy) != block) {
            block = next_block;
            continue;
        }

        HirRegister truthy_input_reg = hir_c_get_operand(truthy, 0);
        assert(truthy_input_reg == hir_instr_get_operand(truthy, 0));
        HirInstr truthy_target = hir_reg_instr(truthy_input_reg);
        if (hir_c_block(truthy_target) != block ||
            (!hir_c_is_compare(truthy_target) &&
             !hir_c_is_vectorcall(truthy_target))) {
            block = next_block;
            continue;
        }

        /* Check if the compare output is a last use at truthy */
        if (!hir_liveness_is_last_use(liveness, truthy, truthy_input_reg)) {
            block = next_block;
            continue;
        }

        /* Scan between truthy_target and the branch for interfering uses */
        /* Collect snapshots that use the compare output */
        HirInstr snapshots[64];
        int num_snapshots = 0;
        int can_optimize = 1;

        /* Walk backwards from the CondBranch */
        HirInstr it = hir_block_back(block);
        /* Skip the CondBranch itself */
        /* We need to walk backwards — use prev via block_next in reverse */
        /* Actually, iterate forward and collect */
        it = hir_block_first(block);
        int found_target = 0;
        int found_truthy = 0;
        while (it != NULL) {
            HirInstr next = hir_block_next(block, it);
            assert(hir_c_is_replayable(it) == hir_instr_is_replayable(it));
            if (it == truthy_target) {
                found_target = 1;
            } else if (found_target && it != truthy && it != instr) {
                if (hir_c_is_snapshot(it)) {
                    if (hir_instr_uses_reg(it, hir_c_output(truthy_target))) {
                        if (num_snapshots < 64) {
                            snapshots[num_snapshots++] = it;
                        }
                    }
                } else if (!hir_c_is_replayable(it)) {
                    can_optimize = 0;
                    break;
                } else if (hir_instr_uses_reg(it, truthy_input_reg)) {
                    can_optimize = 0;
                    break;
                }
            }
            it = next;
        }

        if (!can_optimize) {
            block = next_block;
            continue;
        }

        /* Create replacement: CompareBool */
        HirInstr replacement = NULL;
        if (hir_c_is_compare(truthy_target)) {
            int op = hir_c_compare_op(truthy_target);
            /* Assert: C struct accessor matches C++ bridge */
            assert(op == hir_instr_compare_op(truthy_target));
            HirRegister left = hir_c_get_operand(truthy_target, 0);
            HirRegister right = hir_c_get_operand(truthy_target, 1);
            replacement = hir_compare_bool_create(
                hir_c_output(truthy), op, left, right, truthy);
        }

        if (replacement != NULL) {
            hir_c_copy_bytecode_offset(replacement, instr);
            hir_instr_replace_with(truthy, replacement);

            hir_instr_unlink(truthy_target);
            hir_instr_delete(truthy_target);
            hir_instr_delete(truthy);

            /* Delete collected snapshots */
            for (int i = 0; i < num_snapshots; i++) {
                hir_instr_unlink(snapshots[i]);
                hir_instr_delete(snapshots[i]);
            }
        }

        block = next_block;
    }

    hir_liveness_destroy(liveness);
    hir_reflow_types(func);
}
