/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C wrapper for LivenessAnalysis.
 */

#include "cinderx/Jit/hir/liveness_c.h"
#include "cinderx/Jit/hir/analysis.h"
#include "cinderx/Jit/hir/hir.h"

using namespace jit::hir;

struct HirLivenessState {
  LivenessAnalysis analysis;
  LivenessAnalysis::LastUses last_uses;

  explicit HirLivenessState(const Function& func)
      : analysis(func) {
    analysis.Run();
    last_uses = analysis.GetLastUses();
  }
};

extern "C" {

HirLivenessState *hir_liveness_create(HirFunction func) {
  auto* f = static_cast<Function*>(func);
  return new HirLivenessState(*f);
}

int hir_liveness_is_last_use(
    const HirLivenessState *state, HirInstr instr, HirRegister reg) {
  auto* i = static_cast<const Instr*>(instr);
  auto* r = static_cast<Register*>(reg);
  auto it = state->last_uses.find(i);
  if (it == state->last_uses.end()) {
    return 0;
  }
  return it->second.contains(r) ? 1 : 0;
}

void hir_liveness_destroy(HirLivenessState *state) {
  delete state;
}

} /* extern "C" */
