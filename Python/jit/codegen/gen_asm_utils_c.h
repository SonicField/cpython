/*
 * gen_asm_utils_c.h -- Pure C interface for emitCall utilities
 *
 * Replaces the C++ emitCall functions from gen_asm_utils.h.
 * Emits a function call instruction and optionally binds a debug label
 * at the return point.
 *
 * On x86_64: simple CALL instruction.
 * On ARM64:  saves the return address (via ADR) to either the GenDataFooter
 *            savedIP slot (generators) or a stack slot (regular functions),
 *            then issues BL/BLR.
 *
 * This header requires PHOENIX_ASM -- the PhxBuilder/PhxLabel types are
 * provided by phoenix_asm.h.  Without PHOENIX_ASM, gen_asm_utils.h
 * provides inline C++ implementations using raw asmjit instead.
 */

#ifndef GEN_ASM_UTILS_C_H
#define GEN_ASM_UTILS_C_H

#if defined(PHOENIX_ASM) || defined(__aarch64__)

#include "jit/phoenix_asm/phoenix_asm.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Context for emitCall -- holds the subset of Environ state needed
 * to emit a call instruction.  On x86_64 only `builder` is used.
 * On ARM64 all fields are used.
 */
typedef struct {
    PhxBuilder *builder;

    /* ARM64 only: whether the function being compiled is a generator.
     * Determines where the return address is saved (GenDataFooter vs stack). */
    int is_generator;

    /* ARM64 only: FP-relative offset of the saved-IP slot for non-generator
     * functions.  Ignored when is_generator is nonzero. */
    int saved_ip_fp_offset;

    /* ARM64 only: offset of savedIP within GenDataFooter.
     * Typically offsetof(jit::GenDataFooter, savedIP).
     * Ignored when is_generator is zero. */
    int gen_footer_saved_ip_offset;
} PhxEmitCallCtx;

/*
 * Emit a call to a label (for intra-function calls, e.g. deopt exits).
 *
 * If out_debug_label is non-NULL and the caller wants a debug label bound
 * at the instruction following the call, set *out_debug_label to the
 * newly-bound label.  Set it to a label with id == UINT32_MAX if no
 * label was bound (i.e. origin is NULL -- caller should check).
 *
 * has_origin: nonzero if the call site has a debug origin (the caller
 *             passes 0 when instr->origin() == nullptr).
 */
void phx_emit_call_label(
    PhxEmitCallCtx *ctx,
    PhxLabel target,
    int has_origin,
    PhxLabel *out_debug_label);

/*
 * Emit a call to an absolute function address (for runtime helper calls).
 *
 * Same out_debug_label semantics as phx_emit_call_label.
 */
void phx_emit_call_func(
    PhxEmitCallCtx *ctx,
    uint64_t func_addr,
    int has_origin,
    PhxLabel *out_debug_label);

#ifdef __cplusplus
}
#endif

#endif /* PHOENIX_ASM || __aarch64__ */

#endif /* GEN_ASM_UTILS_C_H */
