// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/Jit/lir/block.h"

namespace jit::hir {
class Function;
}

namespace jit::lir {

struct Function {
  struct CopyResult {
    int begin_bb;
    int end_bb;
  };

  explicit Function(const hir::Function* hir_func = nullptr);
  ~Function();

  // Allocate a new ID for a basic block or an instruction.
  int allocateId();

  // Set the next ID to return from allocateId().  Only meant to be used by the
  // LIR parser.
  void setNextId(int id);

  // Deep copy function into dest_func.
  // Insert the blocks between prev_bb and next_bb.
  // Assumes that prev_bb and next_bb appear consecutively
  // in dest_func->basic_blocks_.
  // Returns the range of inserted blocks in dest_func->basic_blocks_.
  // The inserted blocks start at (inclusive) dest_func->basic_blocks_[begin_bb]
  // and end right before (exclusive) dest_func->basic_blocks_[begin_bb].
  CopyResult copyFrom(
      const Function* src_func,
      BasicBlock* prev_bb,
      BasicBlock* next_bb,
      const hir::Instr* origin);

  // Create a new block and insert it as the last block in the CFG.
  BasicBlock* allocateBasicBlock();

  // Create a new block and insert it in a given spot in the CFG.
  BasicBlock* allocateBasicBlockAfter(BasicBlock* block);

  // Returns the list of all the basic blocks.
  // The basic blocks will be in RPO as long as the CFG has not been
  // modified since the last call to SortRPO().
  BlockSpan basicblocks();
  ConstBlockSpan basicblocks() const;

  BasicBlock* entryBlock() const;

  size_t getNumBasicBlocks() const;

  void sortBasicBlocks();

  const hir::Function* hirFunc() const;

  // Phase B4c: all fields public.
  const hir::Function* hir_func_;

  void ensureBlockCapacity(size_t needed);

  // Phase B3d: individually allocated BasicBlocks, pointer array for ordering.
  // The first block is always the entry block; the last is the exit block.
  BasicBlock** blocks_{nullptr};
  size_t num_blocks_{0};
  size_t blocks_capacity_{0};

  // The next id to assign to a BasicBlock or Instruction.
  int next_id_{0};
};

} // namespace jit::lir
