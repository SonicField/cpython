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

/* ---- Compare op metadata (Batch 28) ----
 * Lookup tables + parse/convert helpers for CompareOp + PrimitiveCompareOp.
 * Index = enum value (kLessThan=0, kLessThanEqual=1, ...). Defined in
 * hir_instr_info_c.c so they live in one TU rather than per-include. */

extern const char *const kCompareOpNames_c[];
extern const size_t kNumCompareOps_c;
extern const char *const kPrimitiveCompareOpNames_c[];
extern const size_t kNumPrimitiveCompareOps_c;

static inline const char *hir_c_get_compare_op_name(int op) {
    return kCompareOpNames_c[op];
}

static inline const char *hir_c_get_primitive_compare_op_name(int op) {
    return kPrimitiveCompareOpNames_c[op];
}

/* Parse: returns enum value (>= 0) on match. JIT_ABORT_C on miss. */
int hir_c_parse_compare_op_name(const char *name, size_t len);
int hir_c_parse_primitive_compare_op_name(const char *name, size_t len);

/* Convert CompareOp → PrimitiveCompareOp.
 * Returns -1 for the 3 CompareOp values without a primitive equivalent
 * (kIn, kNotIn, kExcMatch); valid PrimitiveCompareOp int otherwise. */
int hir_c_to_primitive_compare_op(int op);

/* ---- Binary/Unary op metadata (Batch 29) ----
 * Same pattern as Batch 28. Index = enum value (kAdd=0, kAnd=1, ...). */

extern const char *const kBinaryOpNames_c[];
extern const size_t kNumBinaryOpKinds_c;
extern const char *const kUnaryOpNames_c[];
extern const size_t kNumUnaryOpKinds_c;

static inline const char *hir_c_get_binary_op_name(int op) {
    return kBinaryOpNames_c[op];
}

static inline const char *hir_c_get_unary_op_name(int op) {
    return kUnaryOpNames_c[op];
}

int hir_c_parse_binary_op_name(const char *name, size_t len);
int hir_c_parse_unary_op_name(const char *name, size_t len);

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
