/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C-compatible phi elimination pass.
 */
#pragma once

#include "cinderx/Jit/hir/hir_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Remove trivial Phi instructions (all inputs are the same value).
 * Replaces them with Assign or LoadConst<Bottom>, then runs
 * copy propagation to clean up. Repeats until no more trivial Phis.
 * Finally removes trampoline blocks. */
void hir_phi_elimination_run(HirFunction func);

#ifdef __cplusplus
} /* extern "C" */
#endif
