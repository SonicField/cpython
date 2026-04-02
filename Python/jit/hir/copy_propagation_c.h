/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C-compatible copy propagation pass.
 */
#pragma once

#include "cinderx/Jit/hir/hir_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Eliminate Assign instructions by propagating copies through the HIR.
 * For each instruction, all operand registers are chased through
 * Assign chains to their original values. Then all Assign instructions
 * are removed. */
void hir_copy_propagation_run(HirFunction func);

#ifdef __cplusplus
} /* extern "C" */
#endif
