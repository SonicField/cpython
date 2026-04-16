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

// ---- Destructor ----

Instruction::~Instruction() {
  lir_instruction_destroy(reinterpret_cast<LirInstruction*>(this));
}

// ---- Input management (wired to lir_instruction.c) ----

Operand* Instruction::allocateImmediateInput(uint64_t n, DataType data_type) {
  return reinterpret_cast<Operand*>(
      lir_instruction_alloc_imm_input(
          reinterpret_cast<LirInstruction*>(this), n, static_cast<int>(data_type)));
}

Operand* Instruction::allocateFPImmediateInput(double n) {
  return reinterpret_cast<Operand*>(
      lir_instruction_alloc_fp_imm_input(
          reinterpret_cast<LirInstruction*>(this), n));
}

LinkedOperand* Instruction::allocateLinkedInput(Instruction* def_instr) {
  return reinterpret_cast<LinkedOperand*>(
      lir_instruction_alloc_linked_input(
          reinterpret_cast<LirInstruction*>(this),
          reinterpret_cast<LirInstruction*>(def_instr)));
}

void Instruction::ensureInputCapacity(size_t needed) {
  lir_instruction_ensure_input_capacity(
      reinterpret_cast<LirInstruction*>(this), needed);
}

Operand* Instruction::allocatePhyRegisterInput(PhyLocation loc) {
  return reinterpret_cast<Operand*>(
      lir_instruction_alloc_phyreg_input(
          reinterpret_cast<LirInstruction*>(this),
          *reinterpret_cast<LirPhyLocation*>(&loc)));
}

Operand* Instruction::allocateStackInput(PhyLocation stack) {
  return reinterpret_cast<Operand*>(
      lir_instruction_alloc_stack_input(
          reinterpret_cast<LirInstruction*>(this),
          *reinterpret_cast<LirPhyLocation*>(&stack)));
}

Operand* Instruction::allocatePhyRegOrStackInput(PhyLocation loc) {
  return reinterpret_cast<Operand*>(
      lir_instruction_alloc_phyreg_or_stack_input(
          reinterpret_cast<LirInstruction*>(this),
          *reinterpret_cast<LirPhyLocation*>(&loc)));
}

Operand* Instruction::allocateAddressInput(void* address) {
  return reinterpret_cast<Operand*>(
      lir_instruction_alloc_addr_input(
          reinterpret_cast<LirInstruction*>(this), address));
}

Operand* Instruction::allocateLabelInput(BasicBlock* block) {
  return reinterpret_cast<Operand*>(
      lir_instruction_alloc_label_input(
          reinterpret_cast<LirInstruction*>(this),
          reinterpret_cast<LirBasicBlock*>(block)));
}

void Instruction::setInput(size_t i, OperandBase* input) {
  lir_instruction_set_input(
      reinterpret_cast<LirInstruction*>(this), i,
      reinterpret_cast<LirOperand*>(input));
}

OperandBase* Instruction::removeInput(size_t index) {
  return reinterpret_cast<OperandBase*>(
      lir_instruction_remove_input(
          reinterpret_cast<LirInstruction*>(this), index));
}

OperandBase* Instruction::releaseInput(size_t index) {
  return reinterpret_cast<OperandBase*>(
      lir_instruction_release_input(
          reinterpret_cast<LirInstruction*>(this), index));
}

OperandBase* Instruction::appendInput(OperandBase* operand) {
  return reinterpret_cast<OperandBase*>(
      lir_instruction_append_input(
          reinterpret_cast<LirInstruction*>(this),
          reinterpret_cast<LirOperand*>(operand)));
}

OperandBase* Instruction::prependInput(OperandBase* operand) {
  return reinterpret_cast<OperandBase*>(
      lir_instruction_prepend_input(
          reinterpret_cast<LirInstruction*>(this),
          reinterpret_cast<LirOperand*>(operand)));
}

OperandBase* Instruction::getOperandByPredecessor(const BasicBlock* pred) {
  return reinterpret_cast<OperandBase*>(
      lir_instruction_get_operand_by_predecessor(
          reinterpret_cast<const LirInstruction*>(this),
          reinterpret_cast<const LirBasicBlock*>(pred)));
}

int Instruction::getOperandIndexByPredecessor(const BasicBlock* pred) const {
  auto* result = lir_instruction_get_operand_by_predecessor(
      reinterpret_cast<const LirInstruction*>(this),
      reinterpret_cast<const LirBasicBlock*>(pred));
  if (result == nullptr) return -1;
  // Find the index of this operand — it's the value after the label
  for (size_t i = 0; i < num_inputs_; i += 2) {
    if (inputs_[i]->getBasicBlock() == pred) {
      return i + 1;
    }
  }
  return -1;
}

const OperandBase* Instruction::getOperandByPredecessor(
    const BasicBlock* pred) const {
  return const_cast<Instruction*>(this)->getOperandByPredecessor(pred);
}

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
