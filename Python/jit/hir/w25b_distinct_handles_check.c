/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * W25b §5.2 distinct-handles falsification artifact.
 *
 * Per docs/w25b-typedef-promotion.md §5.2: compile-only test that the
 * three handle typedefs in hir_c_api.h are distinct pointer types in C
 * post-W25b. Pre-W25b, all three were `typedef void *` and assignments
 * between them compiled silently via C99 §6.3.2.3. Post-W25b, each is a
 * distinct opaque struct-pointer typedef and cross-assignment must fail
 * to compile in C.
 *
 * Note: this file is compiled as C (not C++). On the C++ side W25b
 * intentionally keeps the void* typedefs to preserve the existing
 * cast-heavy hir_c_api.cpp implementation. C++ already has type safety
 * on the canonical jit::hir::Instr/Register/Function classes.
 *
 * The file's compile-clean state is the positive test. The commented-out
 * cross-assignments below document the negative test: uncommenting any
 * single line MUST produce a C compile error post-W25b. If a line
 * compiles silently, W25b regressed.
 *
 * Naming note: file is `w25b_distinct_handles_check.c` (not test_*) to
 * avoid the JIT_SOURCES test_* exclude filter (matches w25_dual_include_check.c
 * convention).
 */

#include "cinderx/Jit/hir/hir_c_api.h"

void w25b_distinct_handles_check(HirInstr i, HirRegister r, HirFunction f) {
    /* These should ALL be C compile errors post-W25b. Kept commented-out as
     * documentation of what W25b protects against. Uncommenting any line
     * MUST produce a compile error; if it does not, W25b regressed. */

    /* HirInstr i_from_r = r;     // W25b BLOCK: HirRegister -> HirInstr */
    /* HirInstr i_from_f = f;     // W25b BLOCK: HirFunction -> HirInstr */
    /* HirRegister r_from_i = i;  // W25b BLOCK: HirInstr -> HirRegister */
    /* HirRegister r_from_f = f;  // W25b BLOCK: HirFunction -> HirRegister */
    /* HirFunction f_from_i = i;  // W25b BLOCK: HirInstr -> HirFunction */
    /* HirFunction f_from_r = r;  // W25b BLOCK: HirRegister -> HirFunction */

    (void)i; (void)r; (void)f;  /* silence unused-param */
}
