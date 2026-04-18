/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C SSAify — SSA construction using Braun et al. algorithm.
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef void* HirFunction;

/* Run SSA construction on the function's HIR.
 * Transforms non-SSA HIR to SSA form, then runs phi elimination. */
void hir_ssaify_run_c(HirFunction func);

#ifdef __cplusplus
} /* extern "C" */
#endif
