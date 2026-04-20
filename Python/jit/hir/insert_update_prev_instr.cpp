// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cinderx/Jit/hir/insert_update_prev_instr.h"
#include "cinderx/Jit/hir/insert_update_prev_instr_c.h"

namespace jit::hir {

void InsertUpdatePrevInstr::Run(Function& func) {
  hir_insert_update_prev_instr_run(static_cast<void*>(&func));
}

} // namespace jit::hir
