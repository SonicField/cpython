/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Copy propagation pass — eliminates Assign instructions by propagating
 * copies through the HIR.
 */

#include "cinderx/Jit/hir/copy_propagation_c.h"
#include "cinderx/Jit/hir/hir_c_api.h"
#include "cinderx/Jit/hir/hir_instr_c.h"

#include <stdlib.h>

/* Callback for hir_c_visit_uses: chase each register operand through
 * Assign chains to the original value. */
static int chase_assign_cb(void **reg_slot, void *ctx) {
    (void)ctx;
    *reg_slot = hir_chase_assign(*reg_slot);
    return 1; /* continue */
}

void hir_copy_propagation_run(HirFunction func) {
    HirCFG cfg = hir_func_cfg(func);

    /* Get RPO traversal */
    size_t rpo_cap = 256;
    HirBasicBlock *rpo_blocks = malloc(rpo_cap * sizeof(HirBasicBlock));
    if (!rpo_blocks) return;
    size_t num_blocks = hir_cfg_get_rpo(cfg, rpo_blocks, rpo_cap);
    if (num_blocks > rpo_cap) {
        rpo_cap = num_blocks;
        rpo_blocks = realloc(rpo_blocks, rpo_cap * sizeof(HirBasicBlock));
        if (!rpo_blocks) return;
        hir_cfg_get_rpo(cfg, rpo_blocks, rpo_cap);
    }

    /* Collect Assign instructions while propagating copies */
    size_t assigns_cap = 32;
    size_t assigns_len = 0;
    HirInstr *assigns = malloc(assigns_cap * sizeof(HirInstr));
    if (!assigns) {
        free(rpo_blocks);
        return;
    }

    for (size_t i = 0; i < num_blocks; i++) {
        HirBasicBlock block = rpo_blocks[i];
        HirInstr instr = hir_block_first(block);

        while (instr != NULL) {
            HirInstr next = hir_block_next(block, instr);

            hir_c_visit_uses(instr, chase_assign_cb, NULL);

            if (hir_c_is_assign(instr)) {
                if (assigns_len >= assigns_cap) {
                    assigns_cap *= 2;
                    HirInstr *tmp = realloc(assigns, assigns_cap * sizeof(HirInstr));
                    if (!tmp) break;
                    assigns = tmp;
                }
                assigns[assigns_len++] = instr;
            }

            instr = next;
        }
    }

    /* Delete all Assign instructions */
    for (size_t i = 0; i < assigns_len; i++) {
        hir_instr_unlink(assigns[i]);
        hir_instr_delete(assigns[i]);
    }

    free(assigns);
    free(rpo_blocks);
}
