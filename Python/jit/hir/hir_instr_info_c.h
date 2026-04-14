/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Static metadata table for HIR instruction opcodes.
 * T2-A: Phase 3D Tier 2 — replaces InstrT<> template metadata with
 * compile-time-equivalent C tables.
 */
#pragma once

#include "cinderx/Jit/hir/hir_opcode_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Per-opcode metadata (one entry per opcode in HIR_OPCODE enum) */
typedef struct {
    const char *name;
    int fixed_arity;      /* -1 for variadic (Phi, CallEx, etc.) */
    int has_output;       /* 0 or 1 */
    int is_deopt_base;    /* 0 or 1 */
    int is_terminator;    /* 0 or 1 */
    int is_replayable;    /* 0 or 1 — from Instr::isReplayable() */
} HirInstrInfo;

/* Accessor: get metadata for an opcode */
const HirInstrInfo *hir_instr_get_info(int opcode);

/* Convenience accessors */
static inline int hir_instr_info_is_deopt_base(int opcode) {
    return hir_instr_get_info(opcode)->is_deopt_base;
}

static inline int hir_instr_info_is_terminator(int opcode) {
    return hir_instr_get_info(opcode)->is_terminator;
}

static inline int hir_instr_info_fixed_arity(int opcode) {
    return hir_instr_get_info(opcode)->fixed_arity;
}

static inline int hir_instr_info_has_output(int opcode) {
    return hir_instr_get_info(opcode)->has_output;
}

static inline const char *hir_instr_info_name(int opcode) {
    return hir_instr_get_info(opcode)->name;
}

static inline int hir_instr_info_is_replayable(int opcode) {
    return hir_instr_get_info(opcode)->is_replayable;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
