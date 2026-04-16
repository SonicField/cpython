// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Phase 3D: Minimal stub — only methods with complex logic or
// circular deps. Simple accessors moved inline to block.h.

#include "cinderx/Jit/lir/block.h"

#include "cinderx/Jit/lir/lir_impl_internal.h"

namespace jit::lir {

BasicBlock::~BasicBlock() {
  lir_block_destroy(reinterpret_cast<LirBasicBlock*>(this));
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
