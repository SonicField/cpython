/* Copyright (c) Meta Platforms, Inc. and affiliates.
 * C interface for GuardTypeRemoval pass.
 */
#pragma once

#include "cinderx/Jit/hir/hir_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

void hir_guard_type_removal_run(HirFunction func);

#ifdef __cplusplus
}
#endif
