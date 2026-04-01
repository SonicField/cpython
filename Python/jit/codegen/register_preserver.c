/*
 * register_preserver.c -- Pure C implementation of register preservation
 *
 * Phase 3D conversion: register_preserver.cpp -> register_preserver.c
 * Handles saving and restoring GP and XMM/VecD registers during code
 * generation. Replaces the C++ RegisterPreserver class.
 *
 * C11, no C++ dependencies.
 */

#include "register_preserver_c.h"
#include "arch/detection.h"

#if defined(CINDER_X86_64)
#include "jit/phoenix_asm/x86_64.h"
#elif defined(CINDER_AARCH64)
#include "jit/phoenix_asm/arm64.h"
#else
#error "Unsupported architecture"
#endif

/* Stack alignment requirement (both x86_64 and ARM64 use 16) */
#define STACK_ALIGNMENT 16

/* ------------------------------------------------------------------ */
/*  Helper: register type classification                               */
/* ------------------------------------------------------------------ */

#if defined(CINDER_X86_64)

/* On x86_64, PhxGp.size == 8 means 64-bit GP, size == 16 means XMM */
static inline int phx_reg_is_gp64(PhxGp reg) {
    return reg.size == 8;
}

static inline int phx_reg_is_xmm(PhxGp reg) {
    return reg.size == 16;
}

#elif defined(CINDER_AARCH64)

/* On ARM64, GP registers have id < PHX_FP_FLAG (0x40) */
static inline int phx_reg_is_gp_x(PhxGp reg) {
    return (reg.id & PHX_FP_FLAG) == 0 && reg.size == 8;
}

static inline int phx_reg_is_vec_d(PhxGp reg) {
    return (reg.id & PHX_FP_FLAG) != 0 && reg.size == 8;
}

#endif

/* ------------------------------------------------------------------ */
/*  Helper: register equality                                          */
/* ------------------------------------------------------------------ */

static inline int phx_reg_eq(PhxGp a, PhxGp b) {
    return a.id == b.id && a.size == b.size;
}

/* ------------------------------------------------------------------ */
/*  ARM64: register grouping for STP/LDP pairing                       */
/* ------------------------------------------------------------------ */

#if defined(CINDER_AARCH64)

typedef enum {
    REG_GROUP_GP = 0,
    REG_GROUP_VEC_D,
    REG_GROUP_GP_PAIR,
    REG_GROUP_VEC_D_PAIR
} RegGroupKind;

typedef struct {
    int idx;         /* index into the regs array */
    RegGroupKind kind;
} RegGroup;

/* Maximum register groups (one per register, conservatively) */
#define MAX_REG_GROUPS 64

/*
 * Compute register groupings for STP/LDP pairing.
 * Consecutive GP pairs -> kGpPair, consecutive VecD pairs -> kVecDPair,
 * singles stay as kGp or kVecD.
 */
static int compute_register_groups(const PhxRegPair *regs, int num_regs,
                                   RegGroup *groups) {
    int group_count = 0;
    int idx = 0;

    while (idx < num_regs) {
        if ((idx + 1 < num_regs) &&
            phx_reg_is_gp_x(regs[idx].src) &&
            phx_reg_is_gp_x(regs[idx + 1].src)) {
            groups[group_count].idx = idx;
            groups[group_count].kind = REG_GROUP_GP_PAIR;
            group_count++;
            idx += 2;
        } else if ((idx + 1 < num_regs) &&
                   phx_reg_is_vec_d(regs[idx].src) &&
                   phx_reg_is_vec_d(regs[idx + 1].src)) {
            groups[group_count].idx = idx;
            groups[group_count].kind = REG_GROUP_VEC_D_PAIR;
            group_count++;
            idx += 2;
        } else if (phx_reg_is_gp_x(regs[idx].src)) {
            groups[group_count].idx = idx;
            groups[group_count].kind = REG_GROUP_GP;
            group_count++;
            idx += 1;
        } else {
            groups[group_count].idx = idx;
            groups[group_count].kind = REG_GROUP_VEC_D;
            group_count++;
            idx += 1;
        }
    }

    return group_count;
}

#endif /* CINDER_AARCH64 */

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */

void phx_reg_preserver_init(PhxRegPreserver *rp, PhxBuilder *builder,
                            const PhxRegPair *regs, int num_regs) {
    rp->builder = builder;
    rp->regs = regs;
    rp->num_regs = num_regs;
    rp->align_stack = 0;
}

void phx_reg_preserver_preserve(PhxRegPreserver *rp) {
    PhxBuilder *b = rp->builder;

#if defined(CINDER_X86_64)

    size_t rsp_offset = 0;
    for (int i = 0; i < rp->num_regs; i++) {
        PhxGp reg = rp->regs[i].src;
        if (phx_reg_is_gp64(reg)) {
            phx_x86_push_r(b, reg);
        } else if (phx_reg_is_xmm(reg)) {
            phx_x86_sub_ri(b, PHX_RSP, (int32_t)reg.size);
            phx_x86_movdqu_mr(b, phx_dqword_ptr(PHX_RSP, 0), reg);
        } else {
            /* Unsupported register type -- should not happen */
            phx_x86_ud2(b);
            return;
        }
        rsp_offset += reg.size;
    }

    rp->align_stack = (rsp_offset % STACK_ALIGNMENT) != 0;
    if (rp->align_stack) {
        phx_x86_push_r(b, PHX_RAX);
    }

#elif defined(CINDER_AARCH64)

    RegGroup groups[MAX_REG_GROUPS];
    int group_count = compute_register_groups(rp->regs, rp->num_regs, groups);

    for (int i = 0; i < group_count; i++) {
        int idx = groups[i].idx;
        switch (groups[i].kind) {
            case REG_GROUP_GP_PAIR:
                phx_a64_stp_pre(b,
                    rp->regs[idx].src,
                    rp->regs[idx + 1].src,
                    PHX_SP, -STACK_ALIGNMENT);
                break;
            case REG_GROUP_VEC_D_PAIR:
                phx_a64_stp_pre(b,
                    rp->regs[idx].src,
                    rp->regs[idx + 1].src,
                    PHX_SP, -STACK_ALIGNMENT);
                break;
            case REG_GROUP_GP:
                /* No str_pre in phoenix-asm; decompose into sub + str */
                phx_a64_sub_rri(b, PHX_SP, PHX_SP, STACK_ALIGNMENT);
                phx_a64_str(b, rp->regs[idx].src, phx_ptr(PHX_SP, 0));
                break;
            case REG_GROUP_VEC_D:
                phx_a64_sub_rri(b, PHX_SP, PHX_SP, STACK_ALIGNMENT);
                phx_a64_str(b, rp->regs[idx].src, phx_ptr(PHX_SP, 0));
                break;
        }
    }

#else
#error "Unsupported architecture"
#endif
}

void phx_reg_preserver_remap(PhxRegPreserver *rp) {
    PhxBuilder *b = rp->builder;

#if defined(CINDER_X86_64)

    for (int i = 0; i < rp->num_regs; i++) {
        PhxGp src = rp->regs[i].src;
        PhxGp dst = rp->regs[i].dst;
        if (!phx_reg_eq(src, dst)) {
            if (phx_reg_is_gp64(src)) {
                phx_x86_mov_rr(b, dst, src);
            } else if (phx_reg_is_xmm(src)) {
                phx_x86_movsd_rr(b, dst, src);
            }
        }
    }

#elif defined(CINDER_AARCH64)

    for (int i = 0; i < rp->num_regs; i++) {
        PhxGp src = rp->regs[i].src;
        PhxGp dst = rp->regs[i].dst;
        if (!phx_reg_eq(src, dst)) {
            if (phx_reg_is_gp_x(src)) {
                phx_a64_mov_rr(b, dst, src);
            } else if (phx_reg_is_vec_d(src)) {
                phx_a64_fmov(b, dst, src);
            }
        }
    }

#else
#error "Unsupported architecture"
#endif
}

void phx_reg_preserver_restore(PhxRegPreserver *rp) {
    PhxBuilder *b = rp->builder;

#if defined(CINDER_X86_64)

    if (rp->align_stack) {
        phx_x86_add_ri(b, PHX_RSP, 8);
    }

    /* Restore in reverse order */
    for (int i = rp->num_regs - 1; i >= 0; i--) {
        PhxGp dst = rp->regs[i].dst;
        if (phx_reg_is_gp64(dst)) {
            phx_x86_pop_r(b, dst);
        } else if (phx_reg_is_xmm(dst)) {
            phx_x86_movdqu_rm(b, dst, phx_dqword_ptr(PHX_RSP, 0));
            phx_x86_add_ri(b, PHX_RSP, 16);
        } else {
            /* Unsupported register type -- should not happen */
            phx_x86_ud2(b);
            return;
        }
    }

#elif defined(CINDER_AARCH64)

    RegGroup groups[MAX_REG_GROUPS];
    int group_count = compute_register_groups(rp->regs, rp->num_regs, groups);

    /* Restore in reverse group order */
    for (int i = group_count - 1; i >= 0; i--) {
        int idx = groups[i].idx;
        switch (groups[i].kind) {
            case REG_GROUP_GP_PAIR:
                phx_a64_ldp_post(b,
                    rp->regs[idx].dst,
                    rp->regs[idx + 1].dst,
                    PHX_SP, STACK_ALIGNMENT);
                break;
            case REG_GROUP_VEC_D_PAIR:
                phx_a64_ldp_post(b,
                    rp->regs[idx].dst,
                    rp->regs[idx + 1].dst,
                    PHX_SP, STACK_ALIGNMENT);
                break;
            case REG_GROUP_GP:
                /* No ldr_post in phoenix-asm; decompose into ldr + add */
                phx_a64_ldr(b, rp->regs[idx].dst, phx_ptr(PHX_SP, 0));
                phx_a64_add_rri(b, PHX_SP, PHX_SP, STACK_ALIGNMENT);
                break;
            case REG_GROUP_VEC_D:
                phx_a64_ldr(b, rp->regs[idx].dst, phx_ptr(PHX_SP, 0));
                phx_a64_add_rri(b, PHX_SP, PHX_SP, STACK_ALIGNMENT);
                break;
        }
    }

#else
#error "Unsupported architecture"
#endif
}
