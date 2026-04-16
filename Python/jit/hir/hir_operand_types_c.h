/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C operand-type table for HIR instructions.
 * Replaces C++ _OperandTypes mixin dispatch with static C lookup.
 * Phase 3: enables DEFINE_SIMPLE_INSTR class deletion.
 */
#pragma once

#include "cinderx/Jit/hir/hir_opcode_c.h"
#include "cinderx/Jit/hir/hir_type_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Lightweight operand type for the C table. Uses int for constraint
 * (matches HirConstraint enum values from hir_instr_c.h) to avoid
 * pulling in hir_instr_c.h → frame_state.h → CPython _Atomic headers
 * that break ARM64 C++ TUs. */
typedef struct {
    int kind;       /* HirConstraint enum value (0=kType, etc.) */
    HirType type;
} HirOperandTypeEntry;

/* Per-opcode static operand type info */
#define HIR_MAX_STATIC_OPERAND_TYPES 8

typedef struct {
    int count;
    HirOperandTypeEntry types[HIR_MAX_STATIC_OPERAND_TYPES];
} HirOpcodeOperandInfo;

/* Get static operand type info for an opcode.
 * Returns NULL if opcode is out of range.
 * Note: 4 opcodes (PrimitiveCompare, PrimitiveUnbox, Return, UseType) have
 * instance-dependent operand types at runtime. This table returns their
 * STATIC defaults (from the type list in the INSTR_CLASS macro). */
const HirOpcodeOperandInfo *hir_operand_type_get_info(int opcode);


#ifdef __cplusplus
}
#endif
