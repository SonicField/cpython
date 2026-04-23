// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/Jit/hir/function.h"

#include <string>
#include <string_view>

namespace jit::hir {

// An abstract compiler pass over an HIR function.
class Pass {
 public:
  explicit Pass(std::string_view name) : name_{name} {}
  virtual ~Pass() = default;

  virtual void Run(Function& irfunc) = 0;

  constexpr std::string_view name() const {
    return name_;
  }

 protected:
  std::string name_;
};

// Free-function pass utilities (chaseAssignOperand, collectDirectRegUses,
// reflowTypes, removeTrampolineBlocks, removeUnreachableBlocks,
// removeUnreachableInstructions, simplifyRedundantCondBranches) are
// implemented in pass_output_type_c.c (Batch 2-C: pass.cpp eliminated).
// C++ callers in hir/ now invoke the hir_*_c entry points directly.
// The RegUses heap container used by hir_collect_reg_uses + friends lives
// inside hir_c_api.cpp (the C/C++ bridge boundary).

} // namespace jit::hir
