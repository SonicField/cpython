/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C wrapper for LivenessAnalysis — enables C port files to perform
 * liveness analysis without C++ dependencies.
 */
#pragma once

#include "cinderx/Jit/hir/hir_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to a LivenessAnalysis + LastUses result */
typedef struct HirLivenessState HirLivenessState;

/* Create a liveness analysis, run it, and compute last uses.
 * Caller must free with hir_liveness_destroy(). */
HirLivenessState *hir_liveness_create(HirFunction func);

/* Check if register reg is a last use at instruction instr.
 * Returns 1 if reg dies after instr, 0 otherwise. */
int hir_liveness_is_last_use(
    const HirLivenessState *state, HirInstr instr, HirRegister reg);

/* Free the liveness analysis state. */
void hir_liveness_destroy(HirLivenessState *state);

#ifdef __cplusplus
} /* extern "C" */
#endif
