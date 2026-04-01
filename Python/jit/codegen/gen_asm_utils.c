/*
 * gen_asm_utils.c -- Pure C implementation of emitCall utilities
 *
 * Phase 3D conversion: gen_asm_utils.cpp -> gen_asm_utils.c
 * Uses the phoenix-asm C API instead of asmjit.
 *
 * C11, no C++ dependencies.
 */

#include "gen_asm_utils_c.h"
#include "arch/detection.h"

#if defined(CINDER_X86_64)
#include "jit/phoenix_asm/x86_64.h"
#elif defined(CINDER_AARCH64)
#include "jit/phoenix_asm/arm64.h"
#else
#error "Unsupported architecture"
#endif

/*
 * Bind a debug label at the current position if has_origin is true.
 * Writes the label to *out_debug_label if non-NULL.
 */
static void record_debug_label(
    PhxBuilder *builder,
    int has_origin,
    PhxLabel *out_debug_label)
{
    if (!has_origin) {
        if (out_debug_label) {
            /* Sentinel: no label was bound */
            PhxLabel invalid = { UINT32_MAX };
            *out_debug_label = invalid;
        }
        return;
    }

    PhxLabel lbl = phx_builder_new_label(builder);
    phx_builder_bind(builder, lbl);

    if (out_debug_label) {
        *out_debug_label = lbl;
    }
}

#if defined(CINDER_AARCH64)

/* Forward declaration — implemented in arch.c */
extern PhxMem jit_arch_ptr_resolve(PhxBuilder *as, PhxGp base, int32_t offset,
                                    PhxGp scratch, int32_t access_size);

/*
 * ARM64 helper: save the return address to the appropriate location
 * (GenDataFooter.savedIP for generators, stack slot for regular functions)
 * and emit BL to a label target.
 */
static void emit_call_arm64_label(PhxEmitCallCtx *ctx, PhxLabel target)
{
    PhxBuilder *b = ctx->builder;
    PhxLabel after_call = phx_builder_new_label(b);

    /* ADR scratch0, after_call */
    phx_a64_adr(b, PHX_X12, after_call);

    if (ctx->is_generator) {
        /* STR x12, [x29, #savedIP_offset] */
        phx_a64_str(b, PHX_X12,
                     phx_ptr(PHX_X29, ctx->gen_footer_saved_ip_offset));
    } else {
        /* STR x12, [fp, #saved_ip_fp_offset] (via ptr_resolve from arch.c) */
        PhxMem mem = jit_arch_ptr_resolve(
            b, PHX_X29, ctx->saved_ip_fp_offset, PHX_X13, 8);
        phx_a64_str(b, PHX_X12, mem);
    }

    /* BL target */
    phx_a64_bl(b, target);
    phx_builder_bind(b, after_call);
}

/*
 * ARM64 helper: load the function address into a scratch register,
 * save the return address, and emit BLR.
 */
static void emit_call_arm64_func(PhxEmitCallCtx *ctx, uint64_t func_addr)
{
    PhxBuilder *b = ctx->builder;

    /* MOV x16, func_addr */
    phx_a64_mov_ri(b, PHX_X16, func_addr);

    PhxLabel after_call = phx_builder_new_label(b);

    /* ADR scratch0, after_call */
    phx_a64_adr(b, PHX_X12, after_call);

    if (ctx->is_generator) {
        /* STR x12, [x29, #savedIP_offset] */
        phx_a64_str(b, PHX_X12,
                     phx_ptr(PHX_X29, ctx->gen_footer_saved_ip_offset));
    } else {
        /* STR x12, [fp, #saved_ip_fp_offset] (via ptr_resolve from arch.c) */
        PhxMem mem = jit_arch_ptr_resolve(
            b, PHX_X29, ctx->saved_ip_fp_offset, PHX_X13, 8);
        phx_a64_str(b, PHX_X12, mem);
    }

    /* BLR x16 */
    phx_a64_blr(b, PHX_X16);
    phx_builder_bind(b, after_call);
}

#endif /* CINDER_AARCH64 */


void phx_emit_call_label(
    PhxEmitCallCtx *ctx,
    PhxLabel target,
    int has_origin,
    PhxLabel *out_debug_label)
{
#if defined(CINDER_X86_64)
    phx_x86_call_label(ctx->builder, target);
#elif defined(CINDER_AARCH64)
    emit_call_arm64_label(ctx, target);
#else
    (void)ctx;
    (void)target;
#endif

    record_debug_label(ctx->builder, has_origin, out_debug_label);
}


void phx_emit_call_func(
    PhxEmitCallCtx *ctx,
    uint64_t func_addr,
    int has_origin,
    PhxLabel *out_debug_label)
{
#if defined(CINDER_X86_64)
    /* x86_64: call <absolute address> via R11 scratch register */
    phx_x86_mov_ri(ctx->builder, PHX_R11, (int64_t)func_addr);
    phx_x86_call_r(ctx->builder, PHX_R11);
#elif defined(CINDER_AARCH64)
    emit_call_arm64_func(ctx, func_addr);
#else
    (void)ctx;
    (void)func_addr;
#endif

    record_debug_label(ctx->builder, has_origin, out_debug_label);
}
