// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/Jit/hir/dead_code_elimination_c.h"
#include "cinderx/Jit/hir/pass.h"

namespace jit::hir {

// Eliminate instructions whose outputs are not used in a return or by
// other instructions with side-effects
class DeadCodeElimination : public Pass {
 public:
  DeadCodeElimination() : Pass("DeadCodeElimination") {}

  void Run(Function& irfunc) override {
    hir_dead_code_elimination_run(static_cast<HirFunction>(&irfunc));
  }

  static std::unique_ptr<DeadCodeElimination> Factory() {
    return std::make_unique<DeadCodeElimination>();
  }
};

} // namespace jit::hir
