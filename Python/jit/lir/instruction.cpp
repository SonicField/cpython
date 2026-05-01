// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Phase 3D: Minimal stub — only methods that cannot be inlined
// (destructor, input management, InstrProperty static data).
// Simple accessors moved inline to instruction.h.

#include "cinderx/Jit/lir/instruction.h"

#include "cinderx/Jit/lir/block.h"
#include "cinderx/Jit/lir/function.h"
#include "cinderx/Jit/lir/lir_impl_internal.h"

// Phase B5: Cross-validate C struct sizes against C++ struct sizes.
#include "cinderx/Jit/lir/lir_types_c.h"

#include <array>
#include <utility>

namespace jit::lir {

// Phase 3D: Cross-validate ALL C struct sizes AND field offsets against C++.
static_assert(sizeof(LirPhyLocation) == sizeof(PhyLocation),
    "LirPhyLocation and PhyLocation size mismatch");
static_assert(offsetof(LirPhyLocation, loc) == offsetof(PhyLocation, loc),
    "LirPhyLocation.loc offset mismatch");
static_assert(offsetof(LirPhyLocation, bit_size) == offsetof(PhyLocation, bitSize),
    "LirPhyLocation.bit_size offset mismatch");

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
static_assert(sizeof(LirFunction) == sizeof(Function),
    "LirFunction and Function size mismatch");

// ---- Destructor + Input management + getOperand*: MOVED to instruction.h (Phase 5.A2 C2-C4) ----

// ---- InstrProperty (static data — stays in .cpp until C equivalent exists) ----

bool Instruction::getOutputPhyRegUse() const {
  return InstrProperty::getProperties(opcode_).output_phy_use;
}

bool Instruction::getInputPhyRegUse(size_t i) const {
  if ((isMove() || isMoveRelaxed()) && output_.isInd()) {
    return true;
  }
  auto& uses = InstrProperty::getProperties(opcode_).input_phy_uses;
  if (i >= uses.size()) {
    return false;
  }
  return uses.at(i);
}

bool Instruction::inputsLiveAcross() const {
  return InstrProperty::getProperties(opcode_).inputs_live_across;
}

InstrProperty::InstrInfo& InstrProperty::getProperties(
    Instruction::Opcode opcode) {
  return prop_map_.at(opcode);
}

#define BEGIN_INSTR_PROPERTY \
  std::vector<InstrProperty::InstrInfo> InstrProperty::prop_map_ = {
#define END_INSTR_PROPERTY \
  }                        \
  ;

#define PROPERTY(__t, __p...) {#__t, __p},

// clang-format off
BEGIN_INSTR_PROPERTY
  FOREACH_INSTR_TYPE(PROPERTY)
END_INSTR_PROPERTY
// clang-format on

} // namespace jit::lir

// ---- Extern C wrappers for InstrProperty queries ----
// These query the authoritative C++ prop_map_ (generated from
// FOREACH_INSTR_TYPE). No data duplication — correct by construction.

extern "C" int lir_instr_get_output_phy_reg_use(int opcode) {
  using IP = jit::lir::InstrProperty;
  using Op = jit::lir::Instruction::Opcode;
  if (opcode < 0 || opcode > jit::lir::Instruction::kYieldValue) {
    return 1; // default: output uses phy reg
  }
  return IP::getProperties(static_cast<Op>(opcode)).output_phy_use;
}

extern "C" int lir_instr_get_input_phy_reg_use(int opcode, size_t i) {
  using IP = jit::lir::InstrProperty;
  using Op = jit::lir::Instruction::Opcode;
  if (opcode < 0 || opcode > jit::lir::Instruction::kYieldValue) {
    return 0;
  }
  auto& uses = IP::getProperties(static_cast<Op>(opcode)).input_phy_uses;
  if (i >= uses.size()) return 0;
  return uses[i];
}

extern "C" int lir_instr_inputs_live_across(int opcode) {
  using IP = jit::lir::InstrProperty;
  using Op = jit::lir::Instruction::Opcode;
  if (opcode < 0 || opcode > jit::lir::Instruction::kYieldValue) {
    return 0;
  }
  return IP::getProperties(static_cast<Op>(opcode)).inputs_live_across;
}
