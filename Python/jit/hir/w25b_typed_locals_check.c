/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * W25b §5.3 narrower-falsifier compile check (typed-locals path).
 *
 * Per docs/w25b-typedef-promotion.md §5 + supervisor L2418 + theologian L2417:
 *
 * The §5.1 PRIMARY type-only mutation on hir_c_insert_before (testkeeper L2415)
 * established that W25b typedef promotion CLOSES drift only on the call-site
 * pattern that uses HirInstr-typed locals. When callers use raw `void *`
 * locals, C99 §6.3.2.3 implicit void-to-object-pointer conversion bypasses
 * -Werror=incompatible-pointer-types — drift surface remains open for that
 * call-pattern (accepted-residual; tracked separately as W25c future work
 * for systematic void*-locals→typed-handles conversion).
 *
 * This TU is the NARROWER falsifier per option (a):
 *   - Positive: compile-clean. Demonstrates that going-forward C TUs that
 *     use the typed-handle pattern enjoy W25b drift protection on the
 *     entire W25b-promoted API surface.
 *   - Negative (manual): mutate hir_c_api.h+.cpp's hir_c_insert_before
 *     2nd arg from `HirInstr` to `struct HirBasicBlock *`. Build. Expect
 *     BUILD_EXIT=2 with [-Werror,-Wincompatible-pointer-types] at the
 *     hir_c_insert_before call below. Restore mutation after.
 *
 * Naming: w25b_typed_locals_check.c (not test_*) parallels
 * w25_dual_include_check.c + w25b_distinct_handles_check.c — JIT_SOURCES
 * glob picks it up automatically.
 */

#include "cinderx/Jit/hir/hir_c_api.h"

/* Force the typed-handle calling discipline for each W25b-promoted API
 * function. Each call exercises the path Werror is supposed to gate. None
 * of these run at execution time; their existence is the test.
 *
 * If hir_c_api.h's signature for any of these functions drifts to take a
 * different handle type than the one the typed local declares here, the
 * compiler MUST emit a [-Werror,-Wincompatible-pointer-types] hard error,
 * blocking the build. */
int w25b_typed_locals_check(HirFunction func,
                             HirInstr instr_a,
                             HirInstr instr_b,
                             HirRegister reg) {
    if (func == NULL || instr_a == NULL || instr_b == NULL || reg == NULL) {
        return -1;
    }

    /* HirInstr -> hir_c_insert_before(HirInstr, HirInstr) — the §5.1 PRIMARY
     * acceptance test exercises this. With HirInstr-typed locals, drift on
     * either parameter typedef will be caught at compile time. */
    hir_c_insert_before(instr_a, instr_b);

    /* HirInstr -> hir_instr_unlink(HirInstr) */
    hir_instr_unlink(instr_a);

    /* HirRegister -> hir_reg_instr(HirRegister) */
    HirInstr def = hir_reg_instr(reg);
    (void)def;

    /* HirFunction -> hir_func_alloc_register(HirFunction) returns HirRegister */
    HirRegister new_reg = hir_func_alloc_register(func);
    (void)new_reg;

    return 0;
}
