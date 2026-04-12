/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C interface for DynamicComparisonElimination pass.
 */
#pragma once

#include "cinderx/Jit/hir/hir_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Run the dynamic comparison elimination pass on function. */
void hir_dynamic_comparison_elimination_run(HirFunction func);

#ifdef __cplusplus
} /* extern "C" */
#endif
