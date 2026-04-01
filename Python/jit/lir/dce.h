// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/Jit/lir/lir_c_api.h"

/* ---- C API (implemented in dce.c) ---- */
#ifdef __cplusplus
extern "C" {
#endif

void jit_lir_eliminate_dead_code(JitLirFunc func);

#ifdef __cplusplus
} /* extern "C" */
#endif

#ifdef __cplusplus
#include "cinderx/Jit/lir/function.h"

namespace jit::lir {

inline void eliminateDeadCode(Function* func) {
  jit_lir_eliminate_dead_code(static_cast<JitLirFunc>(func));
}

} // namespace jit::lir
#endif
