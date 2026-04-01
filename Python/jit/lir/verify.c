/*
 * verify.c -- Post-register-allocation LIR verification (pure C)
 *
 * Phase 3D conversion: verify.cpp -> verify.c
 *
 * Verifies that each basic block has branches to all successors
 * unless a successor is the next block in the code layout and
 * in the same code section.
 */

#include "cinderx/Jit/lir/lir_c_api.h"

#include <stdio.h>

int
jit_lir_verify_post_regalloc(JitLirFunc func, FILE *err) {
    if (err == NULL) {
        err = stderr;
    }
    size_t num_blocks = jit_lir_func_num_blocks(func);

    for (size_t i = 0; i < num_blocks; i++) {
        JitLirBlock block = jit_lir_func_get_block(func, i);
        JitLirBlock next_block =
            (i + 1 < num_blocks)
                ? jit_lir_func_get_block(func, i + 1)
                : NULL;

        /* Collect branch targets from this block's instructions.
         * Typical blocks have 0-3 branches; 256 is far beyond any
         * realistic block size. */
        JitLirBlock branched[256];
        size_t num_branched = 0;

        size_t num_instrs = jit_lir_block_num_instrs(block);
        for (size_t j = 0; j < num_instrs; j++) {
            JitLirInstr instr = jit_lir_block_get_instr_at(block, j);
            if (jit_lir_instr_is_branch(instr) ||
                jit_lir_instr_is_branch_cc(instr)) {
                JitLirOperand operand =
                    jit_lir_instr_get_input(instr, 0);
                if (num_branched < 256) {
                    branched[num_branched++] =
                        jit_lir_operand_get_basic_block(operand);
                }
            }
        }

        /* Check that each successor has a matching branch (or is
         * the physically next block in the same section). */
        size_t num_succs = jit_lir_block_num_succs(block);
        for (size_t s = 0; s < num_succs; s++) {
            JitLirBlock succ = jit_lir_block_get_succ(block, s);

            /* Fall-through to next block in same section needs no branch. */
            if (succ == next_block && next_block != NULL &&
                jit_lir_block_get_section(next_block) ==
                    jit_lir_block_get_section(block)) {
                continue;
            }

            /* Linear search for the successor in branched targets. */
            int found = 0;
            for (size_t b = 0; b < num_branched; b++) {
                if (branched[b] == succ) {
                    found = 1;
                    break;
                }
            }

            if (!found) {
                fprintf(err,
                    "ERROR: Basic block %d does not contain a jump to "
                    "non-immediate successor %d.\n",
                    jit_lir_block_get_id(block),
                    jit_lir_block_get_id(succ));
                return 0;
            }
        }
    }

    return 1;
}
