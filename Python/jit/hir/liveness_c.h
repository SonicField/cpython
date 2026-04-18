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

/* Get all registers that die after instruction instr.
 * Writes Register* pointers into out_regs, up to capacity.
 * Returns number of dying registers. */
size_t hir_liveness_get_dying_regs(
    const HirLivenessState *state, HirInstr instr,
    void **out_regs, size_t capacity);

/* Check if a register is live-in to a basic block.
 * block must be a BasicBlock* from the function used to create state. */
int hir_liveness_is_live_in(
    const HirLivenessState *state, const void *block, HirRegister reg);

/* Iterate all registers live-in to a basic block.
 * Calls func(reg, ctx) for each live-in register. */
typedef void (*HirLivenessPerRegFunc)(void *reg, void *ctx);
void hir_liveness_foreach_live_in(
    const HirLivenessState *state, const void *block,
    HirLivenessPerRegFunc func, void *ctx);

/* Differential verification: compare C liveness results against C++.
 * Returns 1 if all last-use results match, 0 if any mismatch.
 * Logs mismatches via JIT_LOG. Safe to call in release builds. */
int hir_liveness_verify(HirFunction func, const HirLivenessState *c_state);

#ifdef __cplusplus
} /* extern "C" */
#endif
