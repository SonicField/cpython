/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Differential verification: compares C LivenessAnalysis (liveness_c.c)
 * against C++ LivenessAnalysis (analysis.cpp). Used by JIT_CHECK to
 * validate port fidelity in release builds.
 */

#include "cinderx/Jit/hir/liveness_c.h"
#include "cinderx/Jit/hir/analysis.h"
#include "cinderx/Jit/hir/hir.h"
#include "cinderx/Common/log.h"

using namespace jit::hir;

extern "C" {

int hir_liveness_verify(HirFunction func_handle, const HirLivenessState *c_state) {
  auto* func = static_cast<Function*>(func_handle);

  LivenessAnalysis cpp_analysis(*func);
  cpp_analysis.Run();
  auto cpp_last_uses = cpp_analysis.GetLastUses();

  int mismatches = 0;

  for (auto& block : func->cfg.blocks) {
    for (auto it = block.rbegin(); it != block.rend(); ++it) {
      auto& instr = *it;
      auto cpp_it = cpp_last_uses.find(&instr);

      for (size_t i = 0; i < func->env.reg_count(); i++) {
        auto* reg = func->env.reg_data()[i];
        if (!reg) continue;

        int c_result = hir_liveness_is_last_use(c_state, (HirInstr)&instr, (HirRegister)reg);
        int cpp_result = 0;
        if (cpp_it != cpp_last_uses.end()) {
          cpp_result = cpp_it->second.contains(reg) ? 1 : 0;
        }

        if (c_result != cpp_result) {
          JIT_LOG(
              "Liveness mismatch: instr {} reg {} in bb {} of {}: C={} C++={}",
              instr, reg->name(), block.id, func->fullname,
              c_result, cpp_result);
          mismatches++;
        }
      }
    }
  }

  return mismatches == 0 ? 1 : 0;
}

} /* extern "C" */
