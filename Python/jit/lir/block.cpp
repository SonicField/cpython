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
  return reinterpret_cast<BasicBlock*>(
      lir_block_insert_between(
          reinterpret_cast<LirBasicBlock*>(this),
          reinterpret_cast<LirBasicBlock*>(block)));
}

BasicBlock* BasicBlock::splitBefore(Instruction* instr) {
  return reinterpret_cast<BasicBlock*>(
      lir_block_split_before(
          reinterpret_cast<LirBasicBlock*>(this),
          reinterpret_cast<LirInstruction*>(instr)));
}

void BasicBlock::fixupPhis(BasicBlock* old_pred, BasicBlock* new_pred) {
  lir_block_fixup_phis(
      reinterpret_cast<LirBasicBlock*>(this),
      reinterpret_cast<LirBasicBlock*>(old_pred),
      reinterpret_cast<LirBasicBlock*>(new_pred));
}

} // namespace jit::lir
