/*
 * arm64.h -- ARM64 (AArch64) instruction encoding backend for phoenix-asm
 *
 * Declares all ARM64 instruction emission functions, register constants,
 * and condition codes.  All functions emit into a PhxBuilder node list.
 *
 * C11, no C++ dependencies.
 */

#ifndef PHX_ARM64_H
#define PHX_ARM64_H

#include "phoenix_asm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  ARM64 condition codes                                              */
/* ------------------------------------------------------------------ */

typedef enum {
    PHX_COND_EQ = 0,   /* Equal (Z==1)                              */
    PHX_COND_NE = 1,   /* Not equal (Z==0)                          */
    PHX_COND_CS = 2,   /* Carry set / unsigned higher or same       */
    PHX_COND_HS = 2,   /* Alias for CS                              */
    PHX_COND_CC = 3,   /* Carry clear / unsigned lower              */
    PHX_COND_LO = 3,   /* Alias for CC                              */
    PHX_COND_MI = 4,   /* Minus / negative (N==1)                   */
    PHX_COND_PL = 5,   /* Plus / positive or zero (N==0)            */
    PHX_COND_VS = 6,   /* Overflow (V==1)                           */
    PHX_COND_VC = 7,   /* No overflow (V==0)                        */
    PHX_COND_HI = 8,   /* Unsigned higher (C==1 && Z==0)            */
    PHX_COND_LS = 9,   /* Unsigned lower or same (C==0 || Z==1)     */
    PHX_COND_GE = 10,  /* Signed greater or equal (N==V)            */
    PHX_COND_LT = 11,  /* Signed less than (N!=V)                   */
    PHX_COND_GT = 12,  /* Signed greater than (Z==0 && N==V)        */
    PHX_COND_LE = 13,  /* Signed less or equal (Z==1 || N!=V)       */
    PHX_COND_AL = 14,  /* Always (unconditional)                    */
    PHX_COND_NV = 15   /* Never (architecturally reserved)          */
} PhxArm64Cond;

/* ------------------------------------------------------------------ */
/*  ARM64 general-purpose register constants (64-bit / X registers)    */
/*                                                                     */
/*  Use size=8 for Xn (64-bit), size=4 for Wn (32-bit).               */
/*  Register 31 is SP or XZR depending on instruction context.         */
/* ------------------------------------------------------------------ */

#define PHX_REG_GP(n, sz) ((PhxGp){ (uint8_t)(n), (uint8_t)(sz) })

/* 64-bit general-purpose registers (X0..X30) */
#define PHX_X0   PHX_REG_GP(0, 8)
#define PHX_X1   PHX_REG_GP(1, 8)
#define PHX_X2   PHX_REG_GP(2, 8)
#define PHX_X3   PHX_REG_GP(3, 8)
#define PHX_X4   PHX_REG_GP(4, 8)
#define PHX_X5   PHX_REG_GP(5, 8)
#define PHX_X6   PHX_REG_GP(6, 8)
#define PHX_X7   PHX_REG_GP(7, 8)
#define PHX_X8   PHX_REG_GP(8, 8)
#define PHX_X9   PHX_REG_GP(9, 8)
#define PHX_X10  PHX_REG_GP(10, 8)
#define PHX_X11  PHX_REG_GP(11, 8)
#define PHX_X12  PHX_REG_GP(12, 8)
#define PHX_X13  PHX_REG_GP(13, 8)
#define PHX_X14  PHX_REG_GP(14, 8)
#define PHX_X15  PHX_REG_GP(15, 8)
#define PHX_X16  PHX_REG_GP(16, 8)
#define PHX_X17  PHX_REG_GP(17, 8)
#define PHX_X18  PHX_REG_GP(18, 8)
#define PHX_X19  PHX_REG_GP(19, 8)
#define PHX_X20  PHX_REG_GP(20, 8)
#define PHX_X21  PHX_REG_GP(21, 8)
#define PHX_X22  PHX_REG_GP(22, 8)
#define PHX_X23  PHX_REG_GP(23, 8)
#define PHX_X24  PHX_REG_GP(24, 8)
#define PHX_X25  PHX_REG_GP(25, 8)
#define PHX_X26  PHX_REG_GP(26, 8)
#define PHX_X27  PHX_REG_GP(27, 8)
#define PHX_X28  PHX_REG_GP(28, 8)
#define PHX_X29  PHX_REG_GP(29, 8)  /* Frame pointer (FP) */
#define PHX_X30  PHX_REG_GP(30, 8)  /* Link register (LR) */
#define PHX_SP   PHX_REG_GP(31, 8)  /* Stack pointer */
#define PHX_XZR  PHX_REG_GP(31, 8)  /* Zero register (same encoding as SP) */
#define PHX_FP   PHX_X29
#define PHX_LR   PHX_X30

/* 32-bit general-purpose registers (W0..W30) */
#define PHX_W0   PHX_REG_GP(0, 4)
#define PHX_W1   PHX_REG_GP(1, 4)
#define PHX_W2   PHX_REG_GP(2, 4)
#define PHX_W3   PHX_REG_GP(3, 4)
#define PHX_W4   PHX_REG_GP(4, 4)
#define PHX_W5   PHX_REG_GP(5, 4)
#define PHX_W6   PHX_REG_GP(6, 4)
#define PHX_W7   PHX_REG_GP(7, 4)
#define PHX_W8   PHX_REG_GP(8, 4)
#define PHX_W9   PHX_REG_GP(9, 4)
#define PHX_W10  PHX_REG_GP(10, 4)
#define PHX_W11  PHX_REG_GP(11, 4)
#define PHX_W12  PHX_REG_GP(12, 4)
#define PHX_W13  PHX_REG_GP(13, 4)
#define PHX_W14  PHX_REG_GP(14, 4)
#define PHX_W15  PHX_REG_GP(15, 4)
#define PHX_W16  PHX_REG_GP(16, 4)
#define PHX_W17  PHX_REG_GP(17, 4)
#define PHX_W18  PHX_REG_GP(18, 4)
#define PHX_W19  PHX_REG_GP(19, 4)
#define PHX_W20  PHX_REG_GP(20, 4)
#define PHX_W21  PHX_REG_GP(21, 4)
#define PHX_W22  PHX_REG_GP(22, 4)
#define PHX_W23  PHX_REG_GP(23, 4)
#define PHX_W24  PHX_REG_GP(24, 4)
#define PHX_W25  PHX_REG_GP(25, 4)
#define PHX_W26  PHX_REG_GP(26, 4)
#define PHX_W27  PHX_REG_GP(27, 4)
#define PHX_W28  PHX_REG_GP(28, 4)
#define PHX_W29  PHX_REG_GP(29, 4)
#define PHX_W30  PHX_REG_GP(30, 4)
#define PHX_WZR  PHX_REG_GP(31, 4)

/* ------------------------------------------------------------------ */
/*  ARM64 FP/SIMD register constants (D0..D31 for double-precision)    */
/* ------------------------------------------------------------------ */

/* FP registers share the same PhxGp type but use IDs 0..31.
 * We distinguish them by the size field:
 *   size=4  -> S register (single-precision float)
 *   size=8  -> D register (double-precision float)
 * The caller / instruction function knows which register file to use
 * based on context (GP vs FP instruction).
 *
 * We mark FP registers with id | 0x40 (bit 6) to distinguish from GP
 * registers in the operand, since both D0 and X0 would otherwise have
 * id=0. The encoding functions mask this bit off for the actual hw field.
 */

#define PHX_FP_FLAG  0x40

#define PHX_REG_FP(n, sz) ((PhxGp){ (uint8_t)((n) | PHX_FP_FLAG), (uint8_t)(sz) })

#define PHX_D0   PHX_REG_FP(0, 8)
#define PHX_D1   PHX_REG_FP(1, 8)
#define PHX_D2   PHX_REG_FP(2, 8)
#define PHX_D3   PHX_REG_FP(3, 8)
#define PHX_D4   PHX_REG_FP(4, 8)
#define PHX_D5   PHX_REG_FP(5, 8)
#define PHX_D6   PHX_REG_FP(6, 8)
#define PHX_D7   PHX_REG_FP(7, 8)
#define PHX_D8   PHX_REG_FP(8, 8)
#define PHX_D9   PHX_REG_FP(9, 8)
#define PHX_D10  PHX_REG_FP(10, 8)
#define PHX_D11  PHX_REG_FP(11, 8)
#define PHX_D12  PHX_REG_FP(12, 8)
#define PHX_D13  PHX_REG_FP(13, 8)
#define PHX_D14  PHX_REG_FP(14, 8)
#define PHX_D15  PHX_REG_FP(15, 8)
#define PHX_D16  PHX_REG_FP(16, 8)
#define PHX_D17  PHX_REG_FP(17, 8)
#define PHX_D18  PHX_REG_FP(18, 8)
#define PHX_D19  PHX_REG_FP(19, 8)
#define PHX_D20  PHX_REG_FP(20, 8)
#define PHX_D21  PHX_REG_FP(21, 8)
#define PHX_D22  PHX_REG_FP(22, 8)
#define PHX_D23  PHX_REG_FP(23, 8)
#define PHX_D24  PHX_REG_FP(24, 8)
#define PHX_D25  PHX_REG_FP(25, 8)
#define PHX_D26  PHX_REG_FP(26, 8)
#define PHX_D27  PHX_REG_FP(27, 8)
#define PHX_D28  PHX_REG_FP(28, 8)
#define PHX_D29  PHX_REG_FP(29, 8)
#define PHX_D30  PHX_REG_FP(30, 8)
#define PHX_D31  PHX_REG_FP(31, 8)

/* Single-precision */
#define PHX_S0   PHX_REG_FP(0, 4)
#define PHX_S1   PHX_REG_FP(1, 4)
#define PHX_S2   PHX_REG_FP(2, 4)
#define PHX_S3   PHX_REG_FP(3, 4)
#define PHX_S4   PHX_REG_FP(4, 4)
#define PHX_S5   PHX_REG_FP(5, 4)
#define PHX_S6   PHX_REG_FP(6, 4)
#define PHX_S7   PHX_REG_FP(7, 4)

/* ------------------------------------------------------------------ */
/*  ARM64 system register IDs (for MRS instruction)                    */
/* ------------------------------------------------------------------ */

/* System register encoding: op0:op1:CRn:CRm:op2 packed into 16 bits.
 * Common registers used by the JIT: */
#define PHX_SYSREG_TPIDR_EL0  0xDE82u  /* Thread pointer (user-space TLS) */
#define PHX_SYSREG_NZCV       0xDA10u  /* Condition flags */

/* ------------------------------------------------------------------ */
/*  Immediate validation utilities                                     */
/* ------------------------------------------------------------------ */

/* Check if imm fits in the 12-bit ADD/SUB immediate form.
 * Returns nonzero if imm can be encoded as imm12 or imm12<<12. */
int phx_arm64_is_add_sub_imm(uint64_t imm);

/* Check if val is a valid logical immediate for the given register width.
 * width must be 32 or 64. Returns nonzero if valid. */
int phx_arm64_is_logical_imm(uint64_t val, uint32_t width);

/* Encode a logical immediate value into the N:immr:imms fields.
 * Returns the encoded 13-bit field (N << 12 | immr << 6 | imms)
 * positioned for OR-ing into the instruction word at bits [22:10].
 * Caller must verify the value is encodable first. */
uint32_t phx_arm64_encode_logical_imm(uint64_t val, uint32_t width);

/* ------------------------------------------------------------------ */
/*  ARM64 internal opcodes (for PhxNode.opcode field)                   */
/*                                                                     */
/*  These are phoenix-asm internal identifiers, not ARM64 hw opcodes.   */
/*  They allow the finalize pass to know how to patch label fixups.     */
/* ------------------------------------------------------------------ */

typedef enum {
    /* Data movement */
    PHX_A64_MOV = 0x100,
    PHX_A64_LDR,
    PHX_A64_LDRB,
    PHX_A64_LDRH,
    PHX_A64_LDRSB,
    PHX_A64_LDRSH,
    PHX_A64_LDRSW,
    PHX_A64_LDP,
    PHX_A64_STR,
    PHX_A64_STRB,
    PHX_A64_STRH,
    PHX_A64_STP,
    PHX_A64_FMOV,
    PHX_A64_ADR,

    /* Arithmetic */
    PHX_A64_ADD,
    PHX_A64_ADDS,
    PHX_A64_SUB,
    PHX_A64_SUBS,
    PHX_A64_MUL,
    PHX_A64_MADD,
    PHX_A64_SDIV,
    PHX_A64_UDIV,

    /* Logic */
    PHX_A64_AND,
    PHX_A64_EOR,
    PHX_A64_ORR,
    PHX_A64_MVN,

    /* Comparison / test */
    PHX_A64_CMP,
    PHX_A64_TST,
    PHX_A64_FCMP,

    /* Conditional */
    PHX_A64_CSEL,
    PHX_A64_CSET,

    /* Branches */
    PHX_A64_B,
    PHX_A64_BL,
    PHX_A64_BLR,
    PHX_A64_BR,
    PHX_A64_B_COND,
    PHX_A64_CBZ,
    PHX_A64_CBNZ,

    /* Extensions */
    PHX_A64_SXTB,
    PHX_A64_SXTH,
    PHX_A64_SXTW,
    PHX_A64_UXTB,
    PHX_A64_UXTH,

    /* Shift */
    PHX_A64_LSL,

    /* FP arithmetic */
    PHX_A64_FADD,
    PHX_A64_FSUB,
    PHX_A64_FMUL,
    PHX_A64_FDIV,

    /* Return / trap */
    PHX_A64_RET,
    PHX_A64_UDF,

    /* Exclusive / atomic */
    PHX_A64_LDXR,
    PHX_A64_STXR,

    /* System */
    PHX_A64_MRS,

    PHX_A64_OPCODE_COUNT
} PhxArm64Opcode;

/* ------------------------------------------------------------------ */
/*  Instruction emission functions                                     */
/*                                                                     */
/*  Naming convention:                                                 */
/*    phx_a64_<mnemonic>[_<suffix>]                                    */
/*  Suffix indicates operand form when ambiguous:                      */
/*    _rr  = register, register                                        */
/*    _ri  = register, immediate                                       */
/*    _rm  = register, memory                                          */
/*    _mr  = memory, register                                          */
/* ------------------------------------------------------------------ */

/* ---- Data Movement ---- */

/* MOV Xd, Xm  (register-to-register) */
void phx_a64_mov_rr(PhxBuilder *b, PhxGp dst, PhxGp src);

/* MOV Xd, #imm  (immediate; uses MOVZ/MOVK sequence for wide values) */
void phx_a64_mov_ri(PhxBuilder *b, PhxGp dst, uint64_t imm);

/* LDR Xt, [Xn, #offset]  -- 64-bit or 32-bit depending on dst.size */
void phx_a64_ldr(PhxBuilder *b, PhxGp dst, PhxMem mem);

/* LDRB Wt, [Xn, #offset] */
void phx_a64_ldrb(PhxBuilder *b, PhxGp dst, PhxMem mem);

/* LDRH Wt, [Xn, #offset] */
void phx_a64_ldrh(PhxBuilder *b, PhxGp dst, PhxMem mem);

/* LDRSB Xt, [Xn, #offset] -- sign-extend byte */
void phx_a64_ldrsb(PhxBuilder *b, PhxGp dst, PhxMem mem);

/* LDRSH Xt, [Xn, #offset] -- sign-extend halfword */
void phx_a64_ldrsh(PhxBuilder *b, PhxGp dst, PhxMem mem);

/* LDRSW Xt, [Xn, #offset] -- sign-extend word (always 64-bit dest) */
void phx_a64_ldrsw(PhxBuilder *b, PhxGp dst, PhxMem mem);

/* LDP Xt1, Xt2, [Xn, #offset] */
void phx_a64_ldp(PhxBuilder *b, PhxGp rt1, PhxGp rt2, PhxMem mem);
/* LDP pre-indexed: ldp rt1, rt2, [rn, #offset]! */
void phx_a64_ldp_pre(PhxBuilder *b, PhxGp rt1, PhxGp rt2, PhxGp base, int32_t offset);
/* LDP post-indexed: ldp rt1, rt2, [rn], #offset */
void phx_a64_ldp_post(PhxBuilder *b, PhxGp rt1, PhxGp rt2, PhxGp base, int32_t offset);

/* STR Xt, [Xn, #offset] */
void phx_a64_str(PhxBuilder *b, PhxGp src, PhxMem mem);

/* STRB Wt, [Xn, #offset] */
void phx_a64_strb(PhxBuilder *b, PhxGp src, PhxMem mem);

/* STRH Wt, [Xn, #offset] */
void phx_a64_strh(PhxBuilder *b, PhxGp src, PhxMem mem);

/* STP Xt1, Xt2, [Xn, #offset] */
void phx_a64_stp(PhxBuilder *b, PhxGp rt1, PhxGp rt2, PhxMem mem);
/* STP pre-indexed: stp rt1, rt2, [rn, #offset]! */
void phx_a64_stp_pre(PhxBuilder *b, PhxGp rt1, PhxGp rt2, PhxGp base, int32_t offset);
/* STP post-indexed: stp rt1, rt2, [rn], #offset */
void phx_a64_stp_post(PhxBuilder *b, PhxGp rt1, PhxGp rt2, PhxGp base, int32_t offset);

/* FMOV Dd, Xn  (GP to FP) / FMOV Xd, Dn  (FP to GP) /
 * FMOV Dd, Dm  (FP to FP) */
void phx_a64_fmov(PhxBuilder *b, PhxGp dst, PhxGp src);

/* ADR Xd, label  -- PC-relative address (21-bit range) */
void phx_a64_adr(PhxBuilder *b, PhxGp dst, PhxLabel label);

/* ---- Arithmetic ---- */

/* ADD Xd, Xn, Xm [, shift #amount] */
void phx_a64_add_rrr(PhxBuilder *b, PhxGp dst, PhxGp src1, PhxGp src2);
void phx_a64_add_rrr_shifted(PhxBuilder *b, PhxGp dst, PhxGp src1, PhxGp src2,
                              uint32_t shift_type, uint32_t shift_amount);
/* SUB Xd, Xn, Xm [, shift #amount] */
void phx_a64_sub_rrr_shifted(PhxBuilder *b, PhxGp dst, PhxGp src1, PhxGp src2,
                              uint32_t shift_type, uint32_t shift_amount);

/* ADD Xd, Xn, #imm */
void phx_a64_add_rri(PhxBuilder *b, PhxGp dst, PhxGp src, int64_t imm);

/* ADDS Xd, Xn, Xm  (sets flags) */
void phx_a64_adds_rrr(PhxBuilder *b, PhxGp dst, PhxGp src1, PhxGp src2);

/* ADDS Xd, Xn, #imm  (sets flags) */
void phx_a64_adds_rri(PhxBuilder *b, PhxGp dst, PhxGp src, int64_t imm);

/* SUB Xd, Xn, Xm */
void phx_a64_sub_rrr(PhxBuilder *b, PhxGp dst, PhxGp src1, PhxGp src2);

/* SUB Xd, Xn, #imm */
void phx_a64_sub_rri(PhxBuilder *b, PhxGp dst, PhxGp src, int64_t imm);

/* SUBS Xd, Xn, Xm  (sets flags) */
void phx_a64_subs_rrr(PhxBuilder *b, PhxGp dst, PhxGp src1, PhxGp src2);

/* SUBS Xd, Xn, #imm  (sets flags) */
void phx_a64_subs_rri(PhxBuilder *b, PhxGp dst, PhxGp src, int64_t imm);

/* MUL Xd, Xn, Xm  (alias for MADD Xd, Xn, Xm, XZR) */
void phx_a64_mul(PhxBuilder *b, PhxGp dst, PhxGp src1, PhxGp src2);

/* MADD Xd, Xn, Xm, Xa   (Xd = Xa + Xn * Xm) */
void phx_a64_madd(PhxBuilder *b, PhxGp dst, PhxGp src1, PhxGp src2,
                  PhxGp addend);

/* SDIV Xd, Xn, Xm */
void phx_a64_sdiv(PhxBuilder *b, PhxGp dst, PhxGp src1, PhxGp src2);

/* UDIV Xd, Xn, Xm */
void phx_a64_udiv(PhxBuilder *b, PhxGp dst, PhxGp src1, PhxGp src2);

/* ---- Logic ---- */

/* AND Xd, Xn, Xm */
void phx_a64_and_rrr(PhxBuilder *b, PhxGp dst, PhxGp src1, PhxGp src2);

/* AND Xd, Xn, #imm  (logical immediate) */
void phx_a64_and_rri(PhxBuilder *b, PhxGp dst, PhxGp src, uint64_t imm);

/* EOR Xd, Xn, Xm */
void phx_a64_eor_rrr(PhxBuilder *b, PhxGp dst, PhxGp src1, PhxGp src2);

/* EOR Xd, Xn, #imm */
void phx_a64_eor_rri(PhxBuilder *b, PhxGp dst, PhxGp src, uint64_t imm);

/* ORR Xd, Xn, Xm */
void phx_a64_orr_rrr(PhxBuilder *b, PhxGp dst, PhxGp src1, PhxGp src2);

/* ORR Xd, Xn, #imm */
void phx_a64_orr_rri(PhxBuilder *b, PhxGp dst, PhxGp src, uint64_t imm);

/* MVN Xd, Xm  (alias for ORN Xd, XZR, Xm) */
void phx_a64_mvn(PhxBuilder *b, PhxGp dst, PhxGp src);

/* ---- Comparison / Test ---- */

/* CMP Xn, Xm  (alias for SUBS XZR, Xn, Xm) */
void phx_a64_cmp_rr(PhxBuilder *b, PhxGp src1, PhxGp src2);

/* CMP Xn, #imm  (alias for SUBS XZR, Xn, #imm) */
void phx_a64_cmp_ri(PhxBuilder *b, PhxGp src, int64_t imm);

/* TST Xn, Xm  (alias for ANDS XZR, Xn, Xm) */
void phx_a64_tst_rr(PhxBuilder *b, PhxGp src1, PhxGp src2);

/* TST Xn, #imm  (alias for ANDS XZR, Xn, #imm) */
void phx_a64_tst_ri(PhxBuilder *b, PhxGp src, uint64_t imm);

/* FCMP Dn, Dm */
void phx_a64_fcmp(PhxBuilder *b, PhxGp src1, PhxGp src2);

/* ---- Conditional ---- */

/* CSEL Xd, Xn, Xm, cond */
void phx_a64_csel(PhxBuilder *b, PhxGp dst, PhxGp src1, PhxGp src2,
                  PhxArm64Cond cond);

/* CSET Xd, cond  (alias for CSINC Xd, XZR, XZR, invert(cond)) */
void phx_a64_cset(PhxBuilder *b, PhxGp dst, PhxArm64Cond cond);

/* ---- Branches: unconditional ---- */

/* B label  (26-bit PC-relative offset) */
void phx_a64_b(PhxBuilder *b, PhxLabel label);

/* BL label  (26-bit PC-relative offset, sets LR) */
void phx_a64_bl(PhxBuilder *b, PhxLabel label);

/* BLR Xn  (branch with link to register) */
void phx_a64_blr(PhxBuilder *b, PhxGp target);

/* BR Xn  (branch to register) */
void phx_a64_br(PhxBuilder *b, PhxGp target);

/* ---- Branches: conditional ---- */

/* B.cond label  (19-bit PC-relative offset)
 * Covers: b_eq, b_ne, b_mi, b_pl, b_gt, b_ge, b_lt, b_le,
 *         b_hi, b_hs, b_lo, b_ls, b_cs, b_cc, b_vs, b_vc */
void phx_a64_b_cond(PhxBuilder *b, PhxArm64Cond cond, PhxLabel label);

/* Convenience wrappers for common conditional branches */
static inline void phx_a64_b_eq(PhxBuilder *b, PhxLabel l) { phx_a64_b_cond(b, PHX_COND_EQ, l); }
static inline void phx_a64_b_ne(PhxBuilder *b, PhxLabel l) { phx_a64_b_cond(b, PHX_COND_NE, l); }
static inline void phx_a64_b_mi(PhxBuilder *b, PhxLabel l) { phx_a64_b_cond(b, PHX_COND_MI, l); }
static inline void phx_a64_b_pl(PhxBuilder *b, PhxLabel l) { phx_a64_b_cond(b, PHX_COND_PL, l); }
static inline void phx_a64_b_gt(PhxBuilder *b, PhxLabel l) { phx_a64_b_cond(b, PHX_COND_GT, l); }
static inline void phx_a64_b_ge(PhxBuilder *b, PhxLabel l) { phx_a64_b_cond(b, PHX_COND_GE, l); }
static inline void phx_a64_b_lt(PhxBuilder *b, PhxLabel l) { phx_a64_b_cond(b, PHX_COND_LT, l); }
static inline void phx_a64_b_le(PhxBuilder *b, PhxLabel l) { phx_a64_b_cond(b, PHX_COND_LE, l); }
static inline void phx_a64_b_hi(PhxBuilder *b, PhxLabel l) { phx_a64_b_cond(b, PHX_COND_HI, l); }
static inline void phx_a64_b_hs(PhxBuilder *b, PhxLabel l) { phx_a64_b_cond(b, PHX_COND_HS, l); }
static inline void phx_a64_b_lo(PhxBuilder *b, PhxLabel l) { phx_a64_b_cond(b, PHX_COND_LO, l); }
static inline void phx_a64_b_ls(PhxBuilder *b, PhxLabel l) { phx_a64_b_cond(b, PHX_COND_LS, l); }
static inline void phx_a64_b_cs(PhxBuilder *b, PhxLabel l) { phx_a64_b_cond(b, PHX_COND_CS, l); }
static inline void phx_a64_b_cc(PhxBuilder *b, PhxLabel l) { phx_a64_b_cond(b, PHX_COND_CC, l); }
static inline void phx_a64_b_vs(PhxBuilder *b, PhxLabel l) { phx_a64_b_cond(b, PHX_COND_VS, l); }
static inline void phx_a64_b_vc(PhxBuilder *b, PhxLabel l) { phx_a64_b_cond(b, PHX_COND_VC, l); }

/* ---- Compare-and-branch ---- */

/* CBZ Xt, label  (19-bit offset) */
void phx_a64_cbz(PhxBuilder *b, PhxGp src, PhxLabel label);

/* CBNZ Xt, label  (19-bit offset) */
void phx_a64_cbnz(PhxBuilder *b, PhxGp src, PhxLabel label);

/* ---- Sign/zero extension ---- */

/* SXTB Xd, Wn  (sign-extend byte; alias for SBFM) */
void phx_a64_sxtb(PhxBuilder *b, PhxGp dst, PhxGp src);

/* SXTH Xd, Wn  (sign-extend halfword; alias for SBFM) */
void phx_a64_sxth(PhxBuilder *b, PhxGp dst, PhxGp src);

/* SXTW Xd, Wn  (sign-extend word; alias for SBFM) */
void phx_a64_sxtw(PhxBuilder *b, PhxGp dst, PhxGp src);

/* UXTB Wd, Wn  (zero-extend byte; alias for UBFM) */
void phx_a64_uxtb(PhxBuilder *b, PhxGp dst, PhxGp src);

/* UXTH Wd, Wn  (zero-extend halfword; alias for UBFM) */
void phx_a64_uxth(PhxBuilder *b, PhxGp dst, PhxGp src);

/* ---- Shift ---- */

/* LSL Xd, Xn, #shift  (alias for UBFM) */
void phx_a64_lsl(PhxBuilder *b, PhxGp dst, PhxGp src, uint32_t shift);

/* ---- FP arithmetic ---- */

/* FADD Dd, Dn, Dm */
void phx_a64_fadd(PhxBuilder *b, PhxGp dst, PhxGp src1, PhxGp src2);

/* FSUB Dd, Dn, Dm */
void phx_a64_fsub(PhxBuilder *b, PhxGp dst, PhxGp src1, PhxGp src2);

/* FMUL Dd, Dn, Dm */
void phx_a64_fmul(PhxBuilder *b, PhxGp dst, PhxGp src1, PhxGp src2);

/* FDIV Dd, Dn, Dm */
void phx_a64_fdiv(PhxBuilder *b, PhxGp dst, PhxGp src1, PhxGp src2);

/* ---- Return / trap ---- */

/* RET {Xn}  (default: X30 / LR) */
void phx_a64_ret(PhxBuilder *b);

/* RET Xn  (return to specific register) */
void phx_a64_ret_reg(PhxBuilder *b, PhxGp target);

/* UDF #imm16  (permanently undefined -- trap) */
void phx_a64_udf(PhxBuilder *b, uint16_t imm);

/* ---- Exclusive / atomic ---- */

/* LDXR Xt, [Xn] */
void phx_a64_ldxr(PhxBuilder *b, PhxGp dst, PhxGp base);

/* STXR Ws, Xt, [Xn]  -- Ws = status (0=success) */
void phx_a64_stxr(PhxBuilder *b, PhxGp status, PhxGp src, PhxGp base);

/* ---- System ---- */

/* MRS Xt, <sysreg>  (sysreg is a 16-bit encoded system register ID) */
void phx_a64_mrs(PhxBuilder *b, PhxGp dst, uint16_t sysreg);

/* ------------------------------------------------------------------ */
/*  Finalize: resolve fixups and linearize into code buffer             */
/* ------------------------------------------------------------------ */

/* Resolve all ARM64 label fixups and linearize the node list into
 * the code holder's buffer.  Returns 0 on success, nonzero on error
 * (e.g. unresolved label, branch offset out of range). */
int phx_a64_finalize(PhxBuilder *b);

#ifdef __cplusplus
}
#endif

#endif /* PHX_ARM64_H */
