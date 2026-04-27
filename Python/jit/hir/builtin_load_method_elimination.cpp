// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Phase 4.A: pure-C body lives in builtin_load_method_elimination_c.c.
// This file is a thin C++ shell so jit::hir::BuiltinLoadMethodElimination
// remains an instantiable Pass for compiler.cpp's runPassIf chain
// (compiler.cpp:127). The Run override forwards to the C entry; the
// extern "C" hir_builtin_load_method_elimination_run is defined on the
// C side.

#include "cinderx/Jit/hir/builtin_load_method_elimination.h"

#include "cinderx/Jit/hir/builtin_load_method_elimination_c.h"
#include "cinderx/Jit/hir/hir_c_api.h"

namespace jit::hir {

void BuiltinLoadMethodElimination::Run(Function& irfunc) {
  hir_builtin_load_method_elimination_run(static_cast<HirFunction>(&irfunc));
}

} // namespace jit::hir
