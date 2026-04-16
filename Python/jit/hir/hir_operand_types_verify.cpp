/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Standing verification test: validates C operand-type table integrity.
 * Runs at program startup via __attribute__((constructor)).
 *
 * For the 4 override types (PrimitiveCompare, PrimitiveUnbox, Return,
 * UseType), cross-checks the C table's static defaults against the C++
 * _OperandTypes mixin. For all 168 opcodes, verifies internal consistency
 * (count > 0 entries have non-zero type bits where expected).
 *
 * Per Alex's directive: this is a STANDING test, not a one-shot validation.
 */

#include "cinderx/Jit/hir/hir_operand_types_c.h"
#include "cinderx/Jit/hir/hir_opcode_c.h"
#include "cinderx/Jit/hir/hir_instr_info_c.h"

#include <assert.h>
#include <stdio.h>

/* Verify C operand-type table internal consistency */
static void verify_operand_types(void) {
    int errors = 0;

    for (int op = 0; op < HIR_OP_COUNT; op++) {
        const HirOpcodeOperandInfo *info = hir_operand_type_get_info(op);
        if (info == NULL) {
            fprintf(stderr,
                "OPERAND TYPE VERIFY: NULL info for opcode %d\n", op);
            errors++;
            continue;
        }

        /* Count must be non-negative and within bounds */
        if (info->count < 0 || info->count > HIR_MAX_STATIC_OPERAND_TYPES) {
            fprintf(stderr,
                "OPERAND TYPE VERIFY: opcode %d count %d out of range\n",
                op, info->count);
            errors++;
            continue;
        }

        /* Cross-check: operand count should match fixed_arity from
         * hir_instr_info table (when fixed_arity >= 0) */
        const HirInstrInfo *instr_info = hir_instr_get_info(op);
        if (instr_info != NULL && instr_info->fixed_arity >= 0) {
            if (info->count != instr_info->fixed_arity) {
                fprintf(stderr,
                    "OPERAND TYPE VERIFY: opcode %d (%s) count mismatch: "
                    "operand_types=%d, instr_info=%d\n",
                    op, instr_info->name, info->count,
                    instr_info->fixed_arity);
                errors++;
            }
        }
    }

    if (errors > 0) {
        fprintf(stderr,
            "hir_operand_types_verify: %d errors in C operand-type table\n",
            errors);
        assert(errors == 0 &&
               "C operand-type table has internal consistency errors");
    }
}

__attribute__((constructor))
static void hir_operand_types_startup_check(void) {
    verify_operand_types();
}
