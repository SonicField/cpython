// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Phase 3D: Minimal stub — only methods with complex logic or
// circular deps. Simple accessors moved inline to block.h.

#include "cinderx/Jit/lir/block.h"

#include "cinderx/Common/log.h"
#include "cinderx/Jit/lir/function.h"
#include "cinderx/Jit/lir/lir_impl_internal.h"

#include <cstring>

namespace jit::lir {

BasicBlock::~BasicBlock() {
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

void BasicBlock::addSuccessor(BasicBlock* bb) {
  lir_block_add_successor(
      reinterpret_cast<LirBasicBlock*>(this),
      reinterpret_cast<LirBasicBlock*>(bb));
}

void BasicBlock::setSuccessor(size_t index, BasicBlock* bb) {
  lir_block_set_successor(
      reinterpret_cast<LirBasicBlock*>(this), index,
      reinterpret_cast<LirBasicBlock*>(bb));
}

void BasicBlock::appendInstr(Instruction* instr) {
  lir_block_append_instr(
      reinterpret_cast<LirBasicBlock*>(this),
      reinterpret_cast<LirInstruction*>(instr));
}

void BasicBlock::insertInstrBefore(Instruction* pos, Instruction* instr) {
  if (pos == nullptr) {
    appendInstr(instr);
    return;
  }
  lir_block_insert_instr_before(
      reinterpret_cast<LirBasicBlock*>(this),
      reinterpret_cast<LirInstruction*>(pos),
      reinterpret_cast<LirInstruction*>(instr));
}

Instruction* BasicBlock::removeInstr(instr_iter_t pos) {
  return reinterpret_cast<Instruction*>(
      lir_block_remove_instr(
          reinterpret_cast<LirBasicBlock*>(this),
          reinterpret_cast<LirInstruction*>(pos)));
}

BasicBlock* BasicBlock::insertBasicBlockBetween(BasicBlock* block) {
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
  JIT_CHECK(func_ != nullptr, "cannot split block that doesn't belong to a function");
  JIT_CHECK(instr->opcode_ != Instruction::kPhi, "cannot split block at a phi node");

  bool found = false;
  for (Instruction* i = instr_head_; i; i = i->next_) {
    if (i == instr) { found = true; break; }
  }
  if (!found) return nullptr;

  auto second_block = func_->allocateBasicBlockAfter(this);
  Instruction* cur = instr;
  while (cur) {
    Instruction* next = cur->next_;
    removeInstr(cur);
    cur->setbasicblock(second_block);
    second_block->appendInstr(cur);
    cur = next;
  }

  for (size_t i = 0; i < num_succs_; i++) {
    auto* bb = successors_[i];
    bb->fixupPhis(this, second_block);
    appendToBlockArray(
        second_block->successors_, second_block->num_succs_,
        second_block->succs_capacity_, bb);
    for (size_t j = 0; j < bb->num_preds_; j++) {
      if (bb->predecessors_[j] == this) {
        bb->predecessors_[j] = second_block;
      }
    }
  }

  num_succs_ = 0;
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

} // namespace jit::lir
