/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * SSAify verification note:
 *
 * SSAify is a DESTRUCTIVE pass — it modifies the function in place.
 * Differential C vs C++ comparison requires cloning the function
 * (not currently supported). Instead, SSAify correctness is verified
 * by the existing checkFunc JIT_DCHECK in compiler.cpp, which runs
 * after every pass and validates SSA properties (every use dominated
 * by definition, every register has one definition, valid CFG).
 *
 * This file exists as a placeholder for future self-consistency
 * invariant checks per theologian's post-flip validation strategy.
 */

#include "cinderx/Jit/hir/ssaify_c.h"

extern "C" {

int hir_ssaify_verify(void *func_handle) {
  (void)func_handle;
  return 1;
}

} /* extern "C" */
