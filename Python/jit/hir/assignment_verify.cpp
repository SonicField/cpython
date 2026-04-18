/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Differential verification for C AssignmentAnalysis.
 */

#include "cinderx/Jit/hir/assignment_c.h"
#include "cinderx/Jit/hir/function.h"
#include "cinderx/Jit/hir/analysis.h"
#include "cinderx/Common/log.h"

using namespace jit::hir;

extern "C" {

int phx_assign_verify(void *func_handle, const PhxAssignmentState *c_state,
                      int is_definite) {
  Function* func = static_cast<Function*>(func_handle);
  AssignmentAnalysis cpp_assign(*func, is_definite != 0);
  cpp_assign.Run();

  int mismatches = 0;

  for (auto& block : func->cfg.blocks) {
    for (size_t i = 0; i < func->env.reg_count(); i++) {
      Register* reg = func->env.reg_data()[i];
      if (!reg) continue;

      int c_in = phx_assign_is_assigned_in(c_state, block.id, reg);
      int cpp_in = cpp_assign.IsAssignedIn(&block, reg) ? 1 : 0;
      if (c_in != cpp_in) {
        JIT_LOG("Assignment IN mismatch: reg {} bb {} of {}: C={} C++={}",
                reg->name(), block.id, func->fullname, c_in, cpp_in);
        mismatches++;
      }

      int c_out = phx_assign_is_assigned_out(c_state, block.id, reg);
      int cpp_out = cpp_assign.IsAssignedOut(&block, reg) ? 1 : 0;
      if (c_out != cpp_out) {
        JIT_LOG("Assignment OUT mismatch: reg {} bb {} of {}: C={} C++={}",
                reg->name(), block.id, func->fullname, c_out, cpp_out);
        mismatches++;
      }
    }
  }

  return mismatches == 0 ? 1 : 0;
}

} /* extern "C" */
