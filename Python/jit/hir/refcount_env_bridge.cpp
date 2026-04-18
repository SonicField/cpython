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

} /* extern "C" */
