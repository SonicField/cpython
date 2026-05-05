// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Phase 3D: Reduced — simple accessors moved inline to function.h.
// Phase 5.A3 commit 7: anonymous-namespace deep-copy helpers
// (copyIndirect / copyOperand / copyInput / connectLinkedOperands /
// deepCopyBasicBlocks) and the Function::copyFrom body removed —
// all replaced by the C path in function_impl.c (commits 0-5).
// Remaining: destructor, block-management delegations, sort.

#include "cinderx/Jit/lir/function.h"

#include "cinderx/Jit/lir/lir_impl_internal.h"

#include "pymem.h"

namespace jit::lir {

Function::~Function() {
  lir_function_destroy(reinterpret_cast<LirFunction*>(this));
}

void Function::ensureBlockCapacity(size_t needed) {
  lir_function_ensure_block_capacity(
      reinterpret_cast<LirFunction*>(this), needed);
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

/* C-callable wrapper for the deep-copy-for-inlining path.
 *
 * Phase 5.A3 commit 5 sole-path flip: forwards to
 * lir_function_copy_from_impl (function_impl.c). The C++
 * Function::copyFrom body it previously dispatched through has been
 * deleted in commit 7; this wrapper is the sole entry point for the
 * deep-copy path, called from inliner_c.c.
 */
extern "C" int lir_function_copy_from_impl(
    LirFunction *caller, const LirFunction *callee,
    LirBasicBlock *prev_bb, LirBasicBlock *next_bb,
    const void *origin,
    int *out_begin, int *out_end);

extern "C" int lir_function_copy_from(
    void* caller, const void* callee,
    void* prev_bb, void* next_bb,
    const void* origin,
    int* out_begin, int* out_end) {
  return lir_function_copy_from_impl(
      reinterpret_cast<LirFunction*>(caller),
      reinterpret_cast<const LirFunction*>(callee),
      reinterpret_cast<LirBasicBlock*>(prev_bb),
      reinterpret_cast<LirBasicBlock*>(next_bb),
      origin,
      out_begin, out_end);
}
