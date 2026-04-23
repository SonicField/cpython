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

/* Constraint-check helpers (Batch 2-A: promoted from func_type_checks_c.c
 * statics, replaces analysis.cpp's registerTypeMatches + operandsMustMatch
 * shadow that was retained only for hir_c_api.cpp legacy bridges). */
int hir_register_type_matches(HirType op_hir, HirOperandType expected);
int hir_operands_must_match_op(HirOperandType op_type);

#ifdef __cplusplus
}
#endif
