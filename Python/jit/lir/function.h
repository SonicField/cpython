// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/Jit/lir/block.h"

namespace jit::hir {
class Function;
}

namespace jit::lir {

struct Function {
  // Phase 5.A3 commit 7: copyFrom + CopyResult removed; deep-copy-for-
  // inlining lives entirely in C as lir_function_copy_from_impl
  // (function_impl.c), declared in lir_impl_internal.h. inliner_c.c is
  // the sole consumer (Phase 5.B c4 removed the void* extern C wrapper).

  explicit Function(const hir::Function* hir_func = nullptr)
      : hir_func_{hir_func} {}
  // Phase 5.B c4: dtor inlined per destroy-vs-free pattern (D-1776379922).
  ~Function() {
    lir_function_destroy(reinterpret_cast<LirFunction*>(this));
  }

  int allocateId() { return next_id_++; }
  void setNextId(int id) { next_id_ = id; }

  // Create a new block and insert it as the last block in the CFG.
  // Phase 5.B c4: inlined.
  BasicBlock* allocateBasicBlock() {
    return reinterpret_cast<BasicBlock*>(
        lir_function_alloc_block(reinterpret_cast<LirFunction*>(this)));
  }

  // Returns the list of all the basic blocks.
  // The basic blocks will be in RPO as long as the CFG has not been
  // modified since the last call to SortRPO().
  BlockSpan basicblocks() { return {blocks_, num_blocks_}; }
  ConstBlockSpan basicblocks() const { return {blocks_, num_blocks_}; }

  BasicBlock* entryBlock() const { return num_blocks_ > 0 ? blocks_[0] : nullptr; }
  size_t getNumBasicBlocks() const { return num_blocks_; }

  // Phase 5.B c4: inlined; allocateBasicBlockAfter + ensureBlockCapacity
  // C++ wrappers deleted (unused from C++; C-side lir_function_*
  // remain, called from block_impl.c + function_impl.c internals).
  void sortBasicBlocks() {
    lir_function_sort_blocks(reinterpret_cast<LirFunction*>(this));
  }

  const hir::Function* hirFunc() const { return hir_func_; }

  // Phase B4c: all fields public.
  const hir::Function* hir_func_;

  // Phase B3d: individually allocated BasicBlocks, pointer array for ordering.
  // The first block is always the entry block; the last is the exit block.
  BasicBlock** blocks_{nullptr};
  size_t num_blocks_{0};
  size_t blocks_capacity_{0};

  // The next id to assign to a BasicBlock or Instruction.
  int next_id_{0};
};

} // namespace jit::lir
