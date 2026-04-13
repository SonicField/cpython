// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// C++ Pass wrapper — delegates to pure C implementation in guard_removal.c.

#include "cinderx/Jit/hir/guard_removal.h"

namespace jit::hir {

void GuardTypeRemoval::Run(Function& func) {
  hir_guard_type_removal_run(static_cast<HirFunction>(&func));
}

} // namespace jit::hir
