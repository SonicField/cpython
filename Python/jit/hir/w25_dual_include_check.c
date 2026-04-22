/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * W25 §5.1 dual-include compile check.
 *
 * Per W25 spec docs/w25-hbb-canonicalization.md §5.1: this TU includes
 * BOTH hir_c_api.h AND hir_basic_block_c.h to verify the dual-typedef
 * collision that blocked W25 Step B is fully resolved post-Step-A
 * (commit e6a8a2d0fb).
 *
 * Pre-W25: hir_c_api.h had `typedef void* HirBasicBlock` while
 * hir_basic_block_c.h had `typedef struct HirBasicBlock {...} HirBasicBlock`.
 * A single TU including both would fail with redefinition of HirBasicBlock
 * to a different type.
 *
 * Post-W25 Step A: hir_c_api.h forward-declares `struct HirBasicBlock`
 * and `struct HirCFG` instead. hir_basic_block_c.h's full struct definitions
 * satisfy the forward decls. Both headers coexist without conflict.
 *
 * If this file FAILS to compile, the W25 canonicalization is incomplete —
 * the typedef collision regression has been reintroduced. Step C lint gate
 * (when shipped) will catch the related class of regressions (local extern
 * decls of API functions).
 *
 * Naming note: file is named `w25_dual_include_check.c` (not test_*) to
 * avoid the JIT_SOURCES test_* exclude filter at jit_build/CMakeLists.txt:131.
 * This way the dual-include check runs at every build, not as a separate
 * gated test artifact.
 */

#include "cinderx/Jit/hir/hir_c_api.h"
#include "cinderx/Jit/hir/hir_basic_block_c.h"

/* Reference both API and struct-field access. The function never gets
 * called at runtime — its existence is the test. The compiler must
 * accept both hir_block_id (declared in hir_c_api.h with struct ptr
 * signature) AND direct struct-field access via hir_basic_block_c.h's
 * full layout. */
int w25_dual_include_check(struct HirBasicBlock *bb) {
    if (bb == NULL) {
        return -1;
    }
    /* API call: hir_block_id signature is `int hir_block_id(struct HirBasicBlock *)` */
    int api_id = hir_block_id(bb);
    /* Direct struct field access: hir_basic_block_c.h's full layout exposes id_ */
    int direct_id = bb->id;
    /* Both should agree (runtime invariant — but the test is compile-time only). */
    return (api_id == direct_id) ? 0 : 1;
}
