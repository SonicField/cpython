// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cinderx/Jit/hir/simplify.h"
#include "cinderx/Jit/hir/simplify_c.h"

namespace jit::hir {

void Simplify::Run(Function& irfunc) {
  hir_simplify_run_c(static_cast<void*>(&irfunc));
}

} // namespace jit::hir
