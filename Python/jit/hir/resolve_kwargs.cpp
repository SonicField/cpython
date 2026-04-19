// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cinderx/Jit/hir/resolve_kwargs_c.h"
#include "cinderx/Jit/hir/resolve_kwargs.h"

#include "cinderx/Jit/hir/hir.h"

namespace jit::hir {

void ResolveKwargs::Run(Function& irfunc) {
  hir_resolve_kwargs_run(static_cast<void*>(&irfunc));
}

}  // namespace jit::hir
