// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Phase 3D: Reduced — simple accessors moved inline to function.h.
// Remaining: destructor, block management, deep copy, sort.

#include "cinderx/Jit/lir/function.h"

#include "cinderx/Jit/containers.h"
#include "cinderx/Jit/lir/lir_impl_internal.h"

#include "pymem.h"

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
  lir_function_destroy(reinterpret_cast<LirFunction*>(this));
}

void Function::ensureBlockCapacity(size_t needed) {
  lir_function_ensure_block_capacity(
      reinterpret_cast<LirFunction*>(this), needed);
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
  return reinterpret_cast<BasicBlock*>(
      lir_function_alloc_block(
          reinterpret_cast<LirFunction*>(this)));
}

BasicBlock* Function::allocateBasicBlockAfter(BasicBlock* block) {
  return reinterpret_cast<BasicBlock*>(
      lir_function_alloc_block_after(
          reinterpret_cast<LirFunction*>(this),
          reinterpret_cast<LirBasicBlock*>(block)));
}

void Function::sortBasicBlocks() {
  lir_function_sort_blocks(
      reinterpret_cast<LirFunction*>(this));
}

} // namespace jit::lir

/* C-callable wrapper for Function::copyFrom */
extern "C" int lir_function_copy_from(
    void* caller, const void* callee,
    void* prev_bb, void* next_bb,
    const void* origin,
    int* out_begin, int* out_end) {
  auto* c = static_cast<jit::lir::Function*>(caller);
  auto result = c->copyFrom(
      static_cast<const jit::lir::Function*>(callee),
      static_cast<jit::lir::BasicBlock*>(prev_bb),
      static_cast<jit::lir::BasicBlock*>(next_bb),
      static_cast<const jit::hir::Instr*>(origin));
  *out_begin = result.begin_bb;
  *out_end = result.end_bb;
  return 0;
}
