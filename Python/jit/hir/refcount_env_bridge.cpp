/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C++ bridge for refcount_env_c functions that need C++ type constants.
 */

#include "cinderx/Jit/hir/refcount_env_c.h"
#include "cinderx/Jit/hir/hir.h"
#include "cinderx/Jit/hir/type.h"

using namespace jit::hir;

extern "C" {

int phx_rc_is_uncounted(void *reg) {
  auto* r = static_cast<Register*>(reg);
  auto reg_type = r->type();
  return !(reg_type <= TMortalObject);
}

int phx_rc_reg_is_object(void *reg) {
  auto* r = static_cast<Register*>(reg);
  return r->type() <= TObject ? 1 : 0;
}

int phx_rc_condbranch_check_type_is_wait_handle(void *instr) {
  auto* cond = static_cast<CondBranchCheckType*>(static_cast<Instr*>(instr));
  return cond->type() == TWaitHandle ? 1 : 0;
}

int phx_rc_is_passthrough(void *instr) {
  return isPassthrough(*static_cast<Instr*>(instr)) ? 1 : 0;
}

int phx_rc_is_guard_is(void *instr) {
  return static_cast<Instr*>(instr)->IsGuardIs() ? 1 : 0;
}

} /* extern "C" */
