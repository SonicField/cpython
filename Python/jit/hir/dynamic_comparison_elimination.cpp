// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// C++ wrapper — delegates to C implementation.

#include "cinderx/Jit/hir/dynamic_comparison_elimination.h"
#include "cinderx/Jit/hir/dynamic_comparison_elimination_c.h"

namespace jit::hir {

void DynamicComparisonElimination::Run(Function& irfunc) {
  hir_dynamic_comparison_elimination_run(static_cast<HirFunction>(&irfunc));
}

} // namespace jit::hir
