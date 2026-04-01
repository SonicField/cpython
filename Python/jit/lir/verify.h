// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <stdio.h>

/* ---- C API (implemented in verify.c) ---- */
#ifdef __cplusplus
extern "C" {
#endif

/* Returns 1 if function passes post-regalloc invariants, 0 otherwise.
 * Errors are printed to err (NULL defaults to stderr). */
int jit_lir_verify_post_regalloc(void* func, FILE *err);

#ifdef __cplusplus
} /* extern "C" */
#endif

#ifdef __cplusplus
#include <iosfwd>

namespace jit::lir {

class Function;

// Verifies the following properties of a LIR function:
//
// - Each block has branches to all successors unless a successor is the next
// block
//   in the code layout post register allocation.
//
// Returns true if the function passes all LIR invariants we wish to uphold post
// register allocation.
inline bool verifyPostRegAllocInvariants(Function* func, std::ostream& /*err*/) {
  return jit_lir_verify_post_regalloc(static_cast<void*>(func), stderr) != 0;
}

} // namespace jit::lir
#endif
