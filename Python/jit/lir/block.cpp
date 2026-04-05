// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cinderx/Jit/lir/block.h"

#include "cinderx/Common/log.h"
#include "cinderx/Jit/codegen/code_section.h"
#include "cinderx/Jit/lir/function.h"

namespace jit::lir {

BasicBlock::BasicBlock(Function* func) : id_(func->allocateId()), func_(func) {}

BasicBlock::~BasicBlock() {
  // Free all owned instructions.
  Instruction* cur = instr_head_;
  while (cur) {
    Instruction* next = cur->next_;
    delete cur;
    cur = next;
  }
  PyMem_RawFree(successors_);
  PyMem_RawFree(predecessors_);
}

static void appendToBlockArray(
    BasicBlock**& arr, size_t& count, size_t& capacity, BasicBlock* bb) {
  if (count >= capacity) {
    size_t new_cap = capacity == 0 ? 2 : capacity * 2;
    auto** new_arr = static_cast<BasicBlock**>(
        PyMem_RawCalloc(new_cap, sizeof(BasicBlock*)));
    if (arr) {
      std::memcpy(new_arr, arr, count * sizeof(BasicBlock*));
      PyMem_RawFree(arr);
    }
    arr = new_arr;
    capacity = new_cap;
  }
  arr[count++] = bb;
}

static void eraseFromBlockArray(
    BasicBlock**& arr, size_t& count, BasicBlock* bb) {
  for (size_t i = 0; i < count; i++) {
    if (arr[i] == bb) {
      for (size_t j = i; j + 1 < count; j++) {
        arr[j] = arr[j + 1];
      }
      count--;
      return;
    }
  }
}

int BasicBlock::id() const {
  return id_;
}

void BasicBlock::setId(int id) {
  id_ = id;
}

Function* BasicBlock::function() {
  return func_;
}

const Function* BasicBlock::function() const {
  return func_;
}

void BasicBlock::addSuccessor(BasicBlock* bb) {
  appendToBlockArray(successors_, num_succs_, succs_capacity_, bb);
  appendToBlockArray(bb->predecessors_, bb->num_preds_, bb->preds_capacity_, this);
}

void BasicBlock::setSuccessor(size_t index, BasicBlock* bb) {
  JIT_CHECK(index < num_succs_, "Index out of range");
  BasicBlock* old_bb = successors_[index];
  eraseFromBlockArray(old_bb->predecessors_, old_bb->num_preds_, this);

  successors_[index] = bb;
  appendToBlockArray(bb->predecessors_, bb->num_preds_, bb->preds_capacity_, this);
}

BlockSpan BasicBlock::successors() {
  return {successors_, num_succs_};
}

ConstBlockSpan BasicBlock::successors() const {
  return {successors_, num_succs_};
}

void BasicBlock::swapSuccessors() {
  if (num_succs_ < 2) {
    return;
  }

  JIT_DCHECK(num_succs_ == 2, "Should at most have two successors.");
  std::swap(successors_[0], successors_[1]);
}

BasicBlock* BasicBlock::getTrueSuccessor() const {
  return successors_[0];
}

BasicBlock* BasicBlock::getFalseSuccessor() const {
  return successors_[1];
}

BlockSpan BasicBlock::predecessors() {
  return {predecessors_, num_preds_};
}

ConstBlockSpan BasicBlock::predecessors() const {
  return {predecessors_, num_preds_};
}

void BasicBlock::appendInstr(Instruction* instr) {
  instr->prev_ = instr_tail_;
  instr->next_ = nullptr;
  if (instr_tail_) {
    instr_tail_->next_ = instr;
  } else {
    instr_head_ = instr;
  }
  instr_tail_ = instr;
  num_instrs_++;
}

void BasicBlock::insertInstrBefore(Instruction* pos, Instruction* instr) {
  if (pos == nullptr) {
    // Insert at end.
    appendInstr(instr);
    return;
  }
  instr->next_ = pos;
  instr->prev_ = pos->prev_;
  if (pos->prev_) {
    pos->prev_->next_ = instr;
  } else {
    instr_head_ = instr;
  }
  pos->prev_ = instr;
  num_instrs_++;
}

Instruction* BasicBlock::removeInstr(instr_iter_t pos) {
  Instruction* instr = pos;
  if (instr->prev_) {
    instr->prev_->next_ = instr->next_;
  } else {
    instr_head_ = instr->next_;
  }
  if (instr->next_) {
    instr->next_->prev_ = instr->prev_;
  } else {
    instr_tail_ = instr->prev_;
  }
  instr->prev_ = nullptr;
  instr->next_ = nullptr;
  num_instrs_--;
  return instr;
}

InstrRange BasicBlock::instructions() {
  return {instr_head_, instr_tail_, num_instrs_};
}

InstrRange BasicBlock::instructions() const {
  return {instr_head_, instr_tail_, num_instrs_};
}

bool BasicBlock::isEmpty() const {
  return instr_head_ == nullptr;
}

size_t BasicBlock::getNumInstrs() const {
  return num_instrs_;
}

Instruction* BasicBlock::getFirstInstr() {
  return instr_head_;
}

const Instruction* BasicBlock::getFirstInstr() const {
  return instr_head_;
}

Instruction* BasicBlock::getLastInstr() {
  return instr_tail_;
}

const Instruction* BasicBlock::getLastInstr() const {
  return instr_tail_;
}

instr_iter_t BasicBlock::getLastInstrIter() {
  return instr_tail_;
}

BasicBlock* BasicBlock::insertBasicBlockBetween(BasicBlock* block) {
  // Find block in successors.
  size_t idx = num_succs_;
  for (size_t i = 0; i < num_succs_; i++) {
    if (successors_[i] == block) { idx = i; break; }
  }
  JIT_DCHECK(idx < num_succs_, "block must be one of the successors.");

  auto new_block = func_->allocateBasicBlockAfter(this);
  successors_[idx] = new_block;
  appendToBlockArray(
      new_block->predecessors_, new_block->num_preds_,
      new_block->preds_capacity_, this);

  eraseFromBlockArray(block->predecessors_, block->num_preds_, this);

  new_block->addSuccessor(block);

  return new_block;
}

BasicBlock* BasicBlock::splitBefore(Instruction* instr) {
  JIT_CHECK(
      func_ != nullptr, "cannot split block that doesn't belong to a function");
  JIT_CHECK(
      instr->opcode_ != Instruction::kPhi, "cannot split block at a phi node");

  // find the instruction in this block
  bool found = false;
  for (Instruction* i = instr_head_; i; i = i->next_) {
    if (i == instr) { found = true; break; }
  }
  if (!found) {
    return nullptr;
  }

  auto second_block = func_->allocateBasicBlockAfter(this);
  // move all instructions from instr onward to second_block
  Instruction* cur = instr;
  while (cur) {
    Instruction* next = cur->next_;
    removeInstr(cur);
    cur->setbasicblock(second_block);
    second_block->appendInstr(cur);
    cur = next;
  }

  // fix up successors
  for (size_t i = 0; i < num_succs_; i++) {
    auto* bb = successors_[i];
    // fix up phis in successors
    bb->fixupPhis(this, second_block);
    // update successors of second block
    appendToBlockArray(
        second_block->successors_, second_block->num_succs_,
        second_block->succs_capacity_, bb);
    // replace this with second_block in bb's predecessors
    for (size_t j = 0; j < bb->num_preds_; j++) {
      if (bb->predecessors_[j] == this) {
        bb->predecessors_[j] = second_block;
      }
    }
  }

  // update successors of first block
  num_succs_ = 0;
  // addSuccessor also fixes predecessors of second block
  addSuccessor(second_block);
  return second_block;
}

void BasicBlock::fixupPhis(BasicBlock* old_pred, BasicBlock* new_pred) {
  foreachPhiInstr([&](Instruction* instr) {
    for (size_t i = 0, n = instr->getNumInputs(); i < n; ++i) {
      auto block = instr->getInput(i);
      if (block->type() == Operand::kLabel) {
        if (block->getBasicBlock() == old_pred) {
          static_cast<Operand*>(block)->setBasicBlock(new_pred);
        }
      }
    }
  });
}

codegen::CodeSection BasicBlock::section() const {
  return section_;
}

void BasicBlock::setSection(codegen::CodeSection section) {
  section_ = section;
}

BasicBlock::instr_iter_t BasicBlock::iterator_to(Instruction* instr) {
  // With intrusive linked list, the instruction IS the iterator.
  return instr;
}

} // namespace jit::lir
