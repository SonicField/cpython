// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Phase 3D: Reduced — simple accessors moved inline to function.h.
// Remaining: destructor, block management, deep copy, sort.

#include "cinderx/Jit/lir/function.h"

#include "cinderx/Jit/containers.h"
#include "cinderx/Jit/lir/blocksorter.h"

#include "pymem.h"

#include <cstring>

namespace jit::lir {

namespace {

void copyIndirect(
    UnorderedMap<LinkedOperand*, int>& instr_refs,
    Operand* dest_op,
    MemoryIndirect* source_op) {
  auto base = source_op->getBaseRegOperand();
  auto index = source_op->getIndexRegOperand();
  std::variant<Instruction*, PhyLocation> dest_base;
  std::variant<Instruction*, PhyLocation> dest_index;
  if (base->isLinked()) {
    dest_base = dest_op->instr();
  } else {
    dest_base = base->getPhyRegister();
  }
  if (index != nullptr) {
    if (index->isLinked()) {
      dest_index = dest_op->instr();
    } else {
      dest_index = index->getPhyRegister();
    }
  }

  dest_op->setMemoryIndirect(
      dest_base, dest_index,
      source_op->getMultipiler(), source_op->getOffset());

  auto memInd = dest_op->getMemoryIndirect();
  if (base->isLinked()) {
    auto base_linked = static_cast<const LinkedOperand*>(base);
    instr_refs.emplace(
        static_cast<LinkedOperand*>(memInd->getBaseRegOperand()),
        base_linked->getLinkedOperand()->instr()->id());
  }
  if (index != nullptr && index->isLinked()) {
    auto index_linked = static_cast<const LinkedOperand*>(index);
    instr_refs.emplace(
        static_cast<LinkedOperand*>(memInd->getIndexRegOperand()),
        index_linked->getLinkedOperand()->instr()->id());
  }
}

void copyOperand(
    UnorderedMap<int, BasicBlock*>& block_index_map,
    UnorderedMap<LinkedOperand*, int>& instr_refs,
    Operand* operand,
    Operand* operand_copy) {
  switch (operand->type()) {
    case OperandBase::kReg:
      operand_copy->setPhyRegister(operand->getPhyRegister());
      operand_copy->setDataType(operand->dataType());
      break;
    case OperandBase::kStack:
      operand_copy->setStackSlot(operand->getStackSlot());
      operand_copy->setDataType(operand->dataType());
      break;
    case OperandBase::kMem:
      operand_copy->setMemoryAddress(operand->getMemoryAddress());
      break;
    case OperandBase::kImm:
      operand_copy->setConstant(operand->getConstant(), operand->dataType());
      break;
    case OperandBase::kLabel:
      operand_copy->setBasicBlock(
          map_get_strict(block_index_map, operand->getBasicBlock()->id_));
      break;
    case OperandBase::kInd:
      copyIndirect(instr_refs, operand_copy, operand->getMemoryIndirect());
      break;
    case OperandBase::kNone:
    case OperandBase::kVreg:
      break;
  }
}

void copyInput(
    UnorderedMap<int, BasicBlock*>& block_index_map,
    UnorderedMap<LinkedOperand*, int>& instr_refs,
    OperandBase* input,
    Instruction* instr_copy) {
  if (input->isLinked()) {
    LinkedOperand* linked_opnd = instr_copy->allocateLinkedInput(nullptr);
    instr_refs.emplace(
        linked_opnd,
        static_cast<LinkedOperand*>(input)->getDefine()->instr()->id());
  } else {
    Operand* input_copy = instr_copy->allocateImmediateInput(0);
    copyOperand(
        block_index_map, instr_refs, static_cast<Operand*>(input), input_copy);
    input_copy->setDataType(input->dataType());
  }
}

void connectLinkedOperands(
    UnorderedMap<int, Instruction*>& output_index_map,
    UnorderedMap<LinkedOperand*, int>& instr_refs) {
  for (auto& [operand, instr_index] : instr_refs) {
    operand->setLinkedInstr(map_get_strict(output_index_map, instr_index));
  }
}

void deepCopyBasicBlocks(
    ConstBlockSpan src_blocks,
    UnorderedMap<int, BasicBlock*>& block_index_map,
    const hir::Instr* origin) {
  UnorderedMap<int, Instruction*> output_index_map;
  UnorderedMap<LinkedOperand*, int> instr_refs;

  for (auto bb : src_blocks) {
    BasicBlock* bb_copy = map_get_strict(block_index_map, bb->id_);
    for (auto succ : bb->successors()) {
      bb_copy->addSuccessor(map_get_strict(block_index_map, succ->id_));
    }
    for (auto& instr : bb->instructions()) {
      auto* instr_copy = new Instruction(bb_copy, &instr, origin);
      bb_copy->appendInstr(instr_copy);
      output_index_map.emplace(instr.id_, instr_copy);
      copyOperand(block_index_map, instr_refs, &instr.output_, &instr_copy->output_);
      for (size_t i = 0, n = instr.num_inputs_; i < n; ++i) {
        copyInput(block_index_map, instr_refs, instr.inputs_[i], instr_copy);
      }
    }
  }
  connectLinkedOperands(output_index_map, instr_refs);
}

} // namespace

Function::~Function() {
  for (size_t i = 0; i < num_blocks_; i++) {
    delete blocks_[i];
  }
  PyMem_RawFree(blocks_);
}

void Function::ensureBlockCapacity(size_t needed) {
  if (needed <= blocks_capacity_) return;
  size_t new_cap = blocks_capacity_ ? blocks_capacity_ * 2 : 8;
  while (new_cap < needed) new_cap *= 2;
  BasicBlock** new_blocks;
  if (blocks_ == nullptr) {
    new_blocks = static_cast<BasicBlock**>(
        PyMem_RawMalloc(new_cap * sizeof(BasicBlock*)));
  } else {
    new_blocks = static_cast<BasicBlock**>(
        PyMem_RawRealloc(blocks_, new_cap * sizeof(BasicBlock*)));
  }
  JIT_CHECK(new_blocks != nullptr, "Failed to allocate block array");
  blocks_ = new_blocks;
  blocks_capacity_ = new_cap;
}

Function::CopyResult Function::copyFrom(
    const Function* src_func,
    BasicBlock* prev_bb,
    BasicBlock* next_bb,
    const hir::Instr* origin) {
  JIT_CHECK(
      prev_bb->successors().size() == 1 && prev_bb->successors()[0] == next_bb,
      "prev_bb should only have 1 successor which should be next_bb.");

  UnorderedMap<int, BasicBlock*> block_index_map;
  size_t src_count = src_func->num_blocks_;

  for (auto bb : src_func->basicblocks()) {
    auto* bb_copy = new BasicBlock(this);
    block_index_map.emplace(bb->id_, bb_copy);
    ensureBlockCapacity(num_blocks_ + 1);
    blocks_[num_blocks_] = blocks_[num_blocks_ - 1];
    blocks_[num_blocks_ - 1] = bb_copy;
    num_blocks_++;
  }

  deepCopyBasicBlocks(src_func->basicblocks(), block_index_map, origin);

  int end = num_blocks_ - 1;
  int start = end - src_count;
  prev_bb->setSuccessor(0, blocks_[start]);
  JIT_CHECK(
      blocks_[end - 1]->successors().empty(),
      "Last block of function should have no successors.");
  blocks_[end - 1]->addSuccessor(next_bb);
  return CopyResult{start, end};
}

BasicBlock* Function::allocateBasicBlock() {
  auto* new_block = new BasicBlock(this);
  ensureBlockCapacity(num_blocks_ + 1);
  blocks_[num_blocks_++] = new_block;
  return new_block;
}

BasicBlock* Function::allocateBasicBlockAfter(BasicBlock* block) {
  size_t pos = 0;
  while (pos < num_blocks_ && blocks_[pos] != block) pos++;
  pos++;

  auto* new_block = new BasicBlock(this);
  ensureBlockCapacity(num_blocks_ + 1);
  memmove(&blocks_[pos + 1], &blocks_[pos],
          (num_blocks_ - pos) * sizeof(BasicBlock*));
  blocks_[pos] = new_block;
  num_blocks_++;
  return new_block;
}

void Function::sortBasicBlocks() {
  size_t out_count = 0;
  JitLirBlock* sorted = jit_lir_sort_blocks_rpo(
      reinterpret_cast<JitLirBlock*>(blocks_), num_blocks_, &out_count);
  memcpy(blocks_, sorted, out_count * sizeof(BasicBlock*));
  num_blocks_ = out_count;
  PyMem_RawFree(sorted);
}

} // namespace jit::lir
