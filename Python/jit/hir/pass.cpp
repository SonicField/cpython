// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cinderx/Jit/hir/pass.h"

extern "C" void hir_simplify_redundant_cond_branches_c(void *cfg);
extern "C" void hir_reflow_types_c(void *func, void *start_block);
extern "C" int hir_remove_trampoline_blocks_c(void *cfg);
extern "C" int hir_remove_unreachable_blocks_c(void *func);
extern "C" int hir_remove_unreachable_instructions_c(void *func);

namespace jit::hir {

RegUses collectDirectRegUses(Function& func) {
  RegUses uses;
  for (auto& block : func.cfg.blocks) {
    for (Instr& instr : block) {
      for (size_t i = 0; i < instr.NumOperands(); ++i) {
        uses[instr.GetOperand(i)].insert(&instr);
      }
    }
  }
  return uses;
}

void reflowTypes(Function& func) {
  hir_reflow_types_c(&func, func.cfg.entry_block);
}

void reflowTypes(Function& func, BasicBlock* start) {
  hir_reflow_types_c(&func, start);
}

bool removeTrampolineBlocks(CFG* cfg) {
  return hir_remove_trampoline_blocks_c(cfg) != 0;
}

bool removeUnreachableBlocks(Function& func) {
  return hir_remove_unreachable_blocks_c(&func) != 0;
}

bool removeUnreachableInstructions(Function& func) {
  return hir_remove_unreachable_instructions_c(&func) != 0;
}


} // namespace jit::hir
