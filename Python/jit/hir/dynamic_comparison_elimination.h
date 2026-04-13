// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/Jit/hir/dynamic_comparison_elimination_c.h"
#include "cinderx/Jit/hir/pass.h"

namespace jit::hir {

class DynamicComparisonElimination : public Pass {
 public:
  DynamicComparisonElimination() : Pass("DynamicComparisonElimination") {}

  void Run(Function& irfunc) override {
    hir_dynamic_comparison_elimination_run(static_cast<HirFunction>(&irfunc));
  }

  static std::unique_ptr<DynamicComparisonElimination> Factory() {
    return std::make_unique<DynamicComparisonElimination>();
  }
};

} // namespace jit::hir
