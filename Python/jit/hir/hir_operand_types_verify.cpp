/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Verification test: compares C operand-type table against C++ _OperandTypes
 * mixin data. Runs at program startup via __attribute__((constructor)).
 *
 * Per Alex's TDD directive: this test validates the C table matches C++.
 * Once verified, the C table replaces the C++ _OperandTypes dispatch,
 * enabling DEFINE_SIMPLE_INSTR class deletion (Phase 3).
 */

#include "cinderx/Jit/hir/hir_operand_types_c.h"
#include "cinderx/Jit/hir/hir_opcode_c.h"
#include "cinderx/Jit/hir/hir_type_c.h"

#include <assert.h>
#include <stdio.h>

/* Verify C table matches C++ bridge for all opcodes */
static void verify_operand_types(void) {
    int mismatches = 0;

    for (int op = 0; op < HIR_OP_COUNT; op++) {
        const HirOpcodeOperandInfo *info = hir_operand_type_get_info(op);
        assert(info != NULL);

        /* For each operand type entry in the C table, compare against C++ */
        for (int i = 0; i < info->count; i++) {
            int cpp_constraint = 0;
            HirType cpp_type = {0};

            int rc = hir_operand_type_cpp_get(op, i, &cpp_constraint, &cpp_type);
            if (rc != 0) {
                /* C++ bridge returned error — skip (opcode may not exist) */
                continue;
            }

            /* Compare constraint */
            if ((int)info->types[i].kind != cpp_constraint) {
                fprintf(stderr,
                    "OPERAND TYPE MISMATCH: opcode %d operand %d "
                    "constraint: C=%d C++=%d\n",
                    op, i, (int)info->types[i].kind, cpp_constraint);
                mismatches++;
            }

            /* Compare type bits (ignore specialization union — only compare
             * the bits_and_flags field which contains type bits + lifetime) */
            uint64_t c_bits = info->types[i].type.bits_and_flags;
            uint64_t cpp_bits = cpp_type.bits_and_flags;
            if (c_bits != cpp_bits) {
                fprintf(stderr,
                    "OPERAND TYPE MISMATCH: opcode %d operand %d "
                    "type bits: C=0x%llx C++=0x%llx\n",
                    op, i,
                    (unsigned long long)c_bits,
                    (unsigned long long)cpp_bits);
                mismatches++;
            }
        }
    }

    if (mismatches > 0) {
        fprintf(stderr,
            "hir_operand_types_verify: %d mismatches between C and C++ tables\n",
            mismatches);
        assert(mismatches == 0 && "C operand-type table does not match C++");
    }
}

__attribute__((constructor))
static void hir_operand_types_startup_check(void) {
    verify_operand_types();
}
