/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Phi elimination pass — removes trivial Phi instructions by replacing
 * them with Assign or LoadConst<Bottom>.
 */

#include "cinderx/Jit/hir/phi_elimination_c.h"
#include "cinderx/Jit/hir/copy_propagation_c.h"
#include "cinderx/Jit/hir/hir_c_api.h"
#include "cinderx/Jit/hir/hir_instr_c.h"

#include <stdlib.h>

void hir_phi_elimination_run(HirFunction func) {
    HirCFG cfg = hir_func_cfg(func);
    int changed = 1;

    while (changed) {
        changed = 0;

        HirBasicBlock block = hir_cfg_blocks_first(cfg);
        while (block != NULL) {
            HirBasicBlock next_block = hir_cfg_blocks_next(cfg, block);

            /* Collect replacement instructions for trivial Phis */
            size_t repl_cap = 16;
            size_t repl_len = 0;
            HirInstr *replacements = malloc(repl_cap * sizeof(HirInstr));
            if (!replacements) return;

            HirInstr instr = hir_block_first(block);
            while (instr != NULL) {
                HirInstr next = hir_block_next(block, instr);

                if (!hir_c_is_phi(instr)) {
                    /* First non-Phi: insert all replacements before it */
                    for (size_t i = 0; i < repl_len; i++) {
                        hir_instr_insert_before(replacements[i], instr);
                    }
                    break;
                }

                HirRegister trivial_value = hir_phi_is_trivial(instr);
                if (trivial_value != NULL) {
                    HirRegister model = hir_chase_assign(trivial_value);
                    HirRegister output = hir_c_output(instr);
                    HirInstr new_instr;

                    if (model == output) {
                        /* Self-referential trivial Phi — unreachable value */
                        new_instr = hir_load_const_bottom_create(output);
                    } else {
                        new_instr = hir_assign_create(output, trivial_value);
                    }
                    hir_c_copy_bytecode_offset(new_instr, instr);

                    /* Grow replacements array if needed */
                    if (repl_len >= repl_cap) {
                        repl_cap *= 2;
                        HirInstr *tmp = realloc(replacements,
                                                repl_cap * sizeof(HirInstr));
                        if (!tmp) { free(replacements); return; }
                        replacements = tmp;
                    }
                    replacements[repl_len++] = new_instr;

                    hir_instr_unlink(instr);
                    hir_instr_delete(instr);
                    changed = 1;
                }

                instr = next;
            }

            free(replacements);
            block = next_block;
        }

        hir_copy_propagation_run(func);
    }

    hir_remove_trampoline_blocks(cfg);
}
