/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C-compatible dead code elimination pass.
 */
#pragma once

#include "cinderx/Jit/hir/hir_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Eliminate instructions whose outputs are not used by any instruction
 * with side effects or control flow. Uses a worklist algorithm to
 * compute the transitive closure of useful instructions. */
void hir_dead_code_elimination_run(HirFunction func);

#ifdef __cplusplus
} /* extern "C" */
#endif
