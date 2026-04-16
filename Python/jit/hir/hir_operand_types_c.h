/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C operand-type table for HIR instructions.
 * Replaces C++ _OperandTypes mixin dispatch with static C lookup.
 * Phase 3: enables DEFINE_SIMPLE_INSTR class deletion.
 */
#pragma once

#include "cinderx/Jit/hir/hir_instr_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Reuse HirConstraint and HirOperandType from hir_instr_c.h */

/* Per-opcode static operand type info */
#define HIR_MAX_STATIC_OPERAND_TYPES 8

typedef struct {
    int count;                                      /* number of operand types */
    HirOperandType types[HIR_MAX_STATIC_OPERAND_TYPES];
} HirOpcodeOperandInfo;

/* Get static operand type info for an opcode.
 * Returns NULL if opcode is out of range.
 * Note: 4 opcodes (PrimitiveCompare, PrimitiveUnbox, Return, UseType) have
 * instance-dependent operand types at runtime. This table returns their
 * STATIC defaults (from the type list in the INSTR_CLASS macro). */
const HirOpcodeOperandInfo *hir_operand_type_get_info(int opcode);

/* C++ bridge: get operand type from the C++ _OperandTypes mixin table.
 * Used by the verification test to cross-check C table against C++.
 * Returns 0 on success, -1 if opcode/index out of range. */
int hir_operand_type_cpp_get(int opcode, int index,
                             int *out_constraint, HirType *out_type);

#ifdef __cplusplus
}
#endif
