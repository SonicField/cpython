/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Differential verification: compares C DominatorAnalysis (dominator_c.c)
 * against C++ DominatorAnalysis (analysis.cpp).
 */

#include "cinderx/Jit/hir/dominator_c.h"
#include "cinderx/Jit/hir/function.h"
#include "cinderx/Jit/hir/analysis.h"
#include "cinderx/Common/log.h"

using namespace jit::hir;

extern "C" {

int phx_dom_verify(void *func_handle, const PhxDominatorState *c_state) {
  Function* func = static_cast<Function*>(func_handle);
  DominatorAnalysis cpp_dom(*func);

  int mismatches = 0;

  for (auto& block : func->cfg.blocks) {
    const BasicBlock* cpp_idom = cpp_dom.immediateDominator(&block);
    void* c_idom = phx_dom_idom(c_state, block.id);

    if (static_cast<const void*>(cpp_idom) != c_idom) {
      JIT_LOG("Dominator idom mismatch bb {}: C={} C++={} in {}",
              block.id,
              c_idom ? static_cast<const BasicBlock*>(c_idom)->id : -1,
              cpp_idom ? cpp_idom->id : -1,
              func->fullname);
      mismatches++;
    }
  }

  return mismatches == 0 ? 1 : 0;
}

} /* extern "C" */
