// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cinderx/Jit/hir/licm_c.h"
#include "cinderx/Jit/hir/licm.h"

#include "cinderx/Jit/hir/hir.h"

namespace jit::hir {

void LICM::Run(Function& irfunc) {
  hir_licm_run(static_cast<void*>(&irfunc));
}

} // namespace jit::hir
