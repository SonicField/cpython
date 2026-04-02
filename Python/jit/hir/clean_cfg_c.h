/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C-compatible clean CFG pass.
 */
#pragma once

#include "cinderx/Jit/hir/hir_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Clean up the CFG by combining block absorption, phi elimination,
 * unreachable code removal, and type reflow. Runs iteratively until
 * no more changes are made. */
void hir_clean_cfg_run(HirFunction func);

#ifdef __cplusplus
} /* extern "C" */
#endif
