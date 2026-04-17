/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C implementation of funcTypeChecks — SSA type constraint validation.
 */
#pragma once

#include "cinderx/Jit/hir/hir_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

int hir_func_type_checks(HirFunction func);

#ifdef __cplusplus
}
#endif
