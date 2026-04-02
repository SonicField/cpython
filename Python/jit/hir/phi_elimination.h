// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/Jit/hir/pass.h"
#include "cinderx/Jit/hir/phi_elimination_c.h"

namespace jit::hir {

// Remove Phis that only have one unique input value (other than their output).
class PhiElimination : public Pass {
 public:
  PhiElimination() : Pass("PhiElimination") {}

  void Run(Function& irfunc) override {
    hir_phi_elimination_run(static_cast<HirFunction>(&irfunc));
  }

  static std::unique_ptr<PhiElimination> Factory() {
    return std::make_unique<PhiElimination>();
  }
};

} // namespace jit::hir
