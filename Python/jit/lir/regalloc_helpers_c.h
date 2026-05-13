/*
 * regalloc_helpers_c.h -- C header for regalloc_helpers_c.c
 *
 * Phase 5.B c15: pure-C ports of small predicate helpers from
 * regalloc.cpp anonymous namespace, parallel to inliner_helpers_c.c
 * + generator_helpers_c.{h,c} precedents.
 */

#ifndef JIT_LIR_REGALLOC_HELPERS_C_H
#define JIT_LIR_REGALLOC_HELPERS_C_H

#include "cinderx/Jit/lir/lir_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Phase 5.B c15: predicate for "operand should be replaced with a new
 * one by the register allocator". True for vreg or linked operands.
 * Original C++ at regalloc.cpp:46-50 (DELETED in c15). */
int phx_should_replace_operand(JitLirOperand op);

#ifdef __cplusplus
}
#endif

#endif /* JIT_LIR_REGALLOC_HELPERS_C_H */
