/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C checkFunc — SSA verification for HIR functions.
 * Replaces the C++ checkFunc from ssa.cpp for C callers.
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef void* HirFunction;

/* Verify that func's CFG is well-formed and HIR is valid SSA.
 * Returns 1 if no errors, 0 if errors found. Prints details to stderr. */
int hir_check_func_c(HirFunction func);

#ifdef __cplusplus
} /* extern "C" */
#endif
