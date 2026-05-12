// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Phase 5.B c5: layout-pin static_asserts cross-validating C struct
// sizes/offsets against C++ counterparts (LirPhyLocation/LirOperand/
// LirInstruction/LirBasicBlock/LirFunction). Migrated out of
// instruction.cpp per separation-of-concerns; follows the
// hir_instr_c_verify.cpp pattern (feedback_verifier_pattern).

#include "cinderx/Jit/lir/instruction.h"

#include "cinderx/Jit/lir/block.h"
#include "cinderx/Jit/lir/function.h"
#include "cinderx/Jit/lir/lir_types_c.h"

namespace jit::lir {

// ---- LirPhyLocation vs PhyLocation ----
static_assert(sizeof(LirPhyLocation) == sizeof(PhyLocation),
    "LirPhyLocation and PhyLocation size mismatch");
static_assert(offsetof(LirPhyLocation, loc) == offsetof(PhyLocation, loc),
    "LirPhyLocation.loc offset mismatch");
static_assert(offsetof(LirPhyLocation, bit_size) == offsetof(PhyLocation, bitSize),
    "LirPhyLocation.bit_size offset mismatch");

// ---- LirOperand vs OperandBase ----
static_assert(sizeof(LirOperand) == sizeof(OperandBase),
    "LirOperand and OperandBase size mismatch");
static_assert(offsetof(LirOperand, parent_instr_) == offsetof(OperandBase, parent_instr_),
    "LirOperand.parent_instr_ offset mismatch");
static_assert(offsetof(LirOperand, last_use_) == offsetof(OperandBase, last_use_),
    "LirOperand.last_use_ offset mismatch");
static_assert(offsetof(LirOperand, is_linked_) == offsetof(OperandBase, is_linked_),
    "LirOperand.is_linked_ offset mismatch");
static_assert(offsetof(LirOperand, type_) == offsetof(OperandBase, type_),
    "LirOperand.type_ offset mismatch");
static_assert(offsetof(LirOperand, data_type_) == offsetof(OperandBase, data_type_),
    "LirOperand.data_type_ offset mismatch");
static_assert(offsetof(LirOperand, value_) == offsetof(OperandBase, value_),
    "LirOperand.value_ offset mismatch");
static_assert(offsetof(LirOperand, def_opnd_) == offsetof(OperandBase, def_opnd_),
    "LirOperand.def_opnd_ offset mismatch");

// ---- LirInstruction vs Instruction ----
static_assert(sizeof(LirInstruction) == sizeof(Instruction),
    "LirInstruction and Instruction size mismatch");
static_assert(offsetof(LirInstruction, id_) == offsetof(Instruction, id_),
    "LirInstruction.id_ offset mismatch");
static_assert(offsetof(LirInstruction, opcode_) == offsetof(Instruction, opcode_),
    "LirInstruction.opcode_ offset mismatch");
static_assert(offsetof(LirInstruction, output_) == offsetof(Instruction, output_),
    "LirInstruction.output_ offset mismatch");
static_assert(offsetof(LirInstruction, basic_block_) == offsetof(Instruction, basic_block_),
    "LirInstruction.basic_block_ offset mismatch");
static_assert(offsetof(LirInstruction, inputs_) == offsetof(Instruction, inputs_),
    "LirInstruction.inputs_ offset mismatch");
static_assert(offsetof(LirInstruction, num_inputs_) == offsetof(Instruction, num_inputs_),
    "LirInstruction.num_inputs_ offset mismatch");
static_assert(offsetof(LirInstruction, inputs_capacity_) == offsetof(Instruction, inputs_capacity_),
    "LirInstruction.inputs_capacity_ offset mismatch");
static_assert(offsetof(LirInstruction, prev_) == offsetof(Instruction, prev_),
    "LirInstruction.prev_ offset mismatch");
static_assert(offsetof(LirInstruction, next_) == offsetof(Instruction, next_),
    "LirInstruction.next_ offset mismatch");

// ---- LirBasicBlock vs BasicBlock ----
static_assert(sizeof(LirBasicBlock) == sizeof(BasicBlock),
    "LirBasicBlock and BasicBlock size mismatch");
static_assert(offsetof(LirBasicBlock, id_) == offsetof(BasicBlock, id_),
    "LirBasicBlock.id_ offset mismatch");
static_assert(offsetof(LirBasicBlock, successors_) == offsetof(BasicBlock, successors_),
    "LirBasicBlock.successors_ offset mismatch");
static_assert(offsetof(LirBasicBlock, instr_head_) == offsetof(BasicBlock, instr_head_),
    "LirBasicBlock.instr_head_ offset mismatch");
static_assert(offsetof(LirBasicBlock, section_) == offsetof(BasicBlock, section_),
    "LirBasicBlock.section_ offset mismatch");

// ---- LirFunction vs Function ----
static_assert(sizeof(LirFunction) == sizeof(Function),
    "LirFunction and Function size mismatch");

} // namespace jit::lir
