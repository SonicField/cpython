/*
 * x86_64.h -- x86_64 instruction encoding backend for phoenix-asm
 *
 * Register constants, condition codes, and instruction emission functions.
 * All functions emit into a PhxBuilder node list.
 *
 * C11, no C++ dependencies.
 */

#ifndef PHX_X86_64_H
#define PHX_X86_64_H

#include "phoenix_asm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  x86_64 General Purpose Register Constants                         */
/*                                                                     */
/*  PhxGp values:  id = hardware register number, size = 8 (64-bit)   */
/*  For 32/16/8-bit variants, use phx_gp32/phx_gp16/phx_gp8 helpers  */
/* ------------------------------------------------------------------ */

#define PHX_RAX  ((PhxGp){  0, 8 })
#define PHX_RCX  ((PhxGp){  1, 8 })
#define PHX_RDX  ((PhxGp){  2, 8 })
#define PHX_RBX  ((PhxGp){  3, 8 })
#define PHX_RSP  ((PhxGp){  4, 8 })
#define PHX_RBP  ((PhxGp){  5, 8 })
#define PHX_RSI  ((PhxGp){  6, 8 })
#define PHX_RDI  ((PhxGp){  7, 8 })
#define PHX_R8   ((PhxGp){  8, 8 })
#define PHX_R9   ((PhxGp){  9, 8 })
#define PHX_R10  ((PhxGp){ 10, 8 })
#define PHX_R11  ((PhxGp){ 11, 8 })
#define PHX_R12  ((PhxGp){ 12, 8 })
#define PHX_R13  ((PhxGp){ 13, 8 })
#define PHX_R14  ((PhxGp){ 14, 8 })
#define PHX_R15  ((PhxGp){ 15, 8 })

/* 32-bit aliases */
#define PHX_EAX  ((PhxGp){  0, 4 })
#define PHX_ECX  ((PhxGp){  1, 4 })
#define PHX_EDX  ((PhxGp){  2, 4 })
#define PHX_EBX  ((PhxGp){  3, 4 })
#define PHX_ESP  ((PhxGp){  4, 4 })
#define PHX_EBP  ((PhxGp){  5, 4 })
#define PHX_ESI  ((PhxGp){  6, 4 })
#define PHX_EDI  ((PhxGp){  7, 4 })
#define PHX_R8D  ((PhxGp){  8, 4 })
#define PHX_R9D  ((PhxGp){  9, 4 })
#define PHX_R10D ((PhxGp){ 10, 4 })
#define PHX_R11D ((PhxGp){ 11, 4 })
#define PHX_R12D ((PhxGp){ 12, 4 })
#define PHX_R13D ((PhxGp){ 13, 4 })
#define PHX_R14D ((PhxGp){ 14, 4 })
#define PHX_R15D ((PhxGp){ 15, 4 })

/* ------------------------------------------------------------------ */
/*  XMM Register Constants                                             */
/*  Reuse PhxGp with size=16 to distinguish from GP registers          */
/* ------------------------------------------------------------------ */

#define PHX_XMM0   ((PhxGp){  0, 16 })
#define PHX_XMM1   ((PhxGp){  1, 16 })
#define PHX_XMM2   ((PhxGp){  2, 16 })
#define PHX_XMM3   ((PhxGp){  3, 16 })
#define PHX_XMM4   ((PhxGp){  4, 16 })
#define PHX_XMM5   ((PhxGp){  5, 16 })
#define PHX_XMM6   ((PhxGp){  6, 16 })
#define PHX_XMM7   ((PhxGp){  7, 16 })
#define PHX_XMM8   ((PhxGp){  8, 16 })
#define PHX_XMM9   ((PhxGp){  9, 16 })
#define PHX_XMM10  ((PhxGp){ 10, 16 })
#define PHX_XMM11  ((PhxGp){ 11, 16 })
#define PHX_XMM12  ((PhxGp){ 12, 16 })
#define PHX_XMM13  ((PhxGp){ 13, 16 })
#define PHX_XMM14  ((PhxGp){ 14, 16 })
#define PHX_XMM15  ((PhxGp){ 15, 16 })

/* ------------------------------------------------------------------ */
/*  Register size conversion helpers                                   */
/* ------------------------------------------------------------------ */

static inline PhxGp phx_gp64(PhxGp r) { return (PhxGp){ r.id, 8 }; }
static inline PhxGp phx_gp32(PhxGp r) { return (PhxGp){ r.id, 4 }; }
static inline PhxGp phx_gp16(PhxGp r) { return (PhxGp){ r.id, 2 }; }
static inline PhxGp phx_gp8(PhxGp r)  { return (PhxGp){ r.id, 1 }; }

/* ------------------------------------------------------------------ */
/*  Condition Codes (for Jcc and SETcc)                                */
/* ------------------------------------------------------------------ */

typedef enum {
    PHX_CC_O   = 0x0,   /* overflow */
    PHX_CC_NO  = 0x1,   /* not overflow */
    PHX_CC_B   = 0x2,   /* below (carry) */
    PHX_CC_AE  = 0x3,   /* above or equal (not carry) */
    PHX_CC_E   = 0x4,   /* equal (zero) */
    PHX_CC_NE  = 0x5,   /* not equal (not zero) */
    PHX_CC_BE  = 0x6,   /* below or equal */
    PHX_CC_A   = 0x7,   /* above */
    PHX_CC_S   = 0x8,   /* sign */
    PHX_CC_NS  = 0x9,   /* not sign */
    PHX_CC_P   = 0xA,   /* parity */
    PHX_CC_NP  = 0xB,   /* not parity */
    PHX_CC_L   = 0xC,   /* less */
    PHX_CC_GE  = 0xD,   /* greater or equal */
    PHX_CC_LE  = 0xE,   /* less or equal */
    PHX_CC_G   = 0xF    /* greater */
} PhxCondCode;

/* Aliases for readability */
#define PHX_CC_Z   PHX_CC_E
#define PHX_CC_NZ  PHX_CC_NE
#define PHX_CC_C   PHX_CC_B
#define PHX_CC_NC  PHX_CC_AE

/* ------------------------------------------------------------------ */
/*  Internal instruction opcodes (phoenix-asm internal enumeration)    */
/* ------------------------------------------------------------------ */

typedef enum {
    /* Data movement */
    PHX_OP_MOV = 1,
    PHX_OP_LEA,
    PHX_OP_MOVSD,
    PHX_OP_MOVSX,
    PHX_OP_MOVSXD,
    PHX_OP_MOVZX,
    PHX_OP_MOVQ,
    PHX_OP_PUSH,
    PHX_OP_POP,
    PHX_OP_XCHG,
    PHX_OP_MOVDQU,
    PHX_OP_EMBED_DATA,
    PHX_OP_LEAVE,
    PHX_OP_CMOVNZ,

    /* Arithmetic */
    PHX_OP_ADD,
    PHX_OP_SUB,
    PHX_OP_NEG,
    PHX_OP_INC,
    PHX_OP_DEC,
    PHX_OP_IMUL,
    PHX_OP_IDIV,
    PHX_OP_DIV,

    /* Logic */
    PHX_OP_XOR,
    PHX_OP_AND,
    PHX_OP_OR,
    PHX_OP_NOT,

    /* Comparison/Test */
    PHX_OP_CMP,
    PHX_OP_TEST,
    PHX_OP_BT,
    PHX_OP_COMISD,

    /* Setcc (single opcode, cc stored in operand) */
    PHX_OP_SETCC,

    /* Branches */
    PHX_OP_JMP,
    PHX_OP_JCC,   /* conditional jump, cc stored in operand */
    PHX_OP_CALL,
    PHX_OP_RET,
    PHX_OP_UD2,

    /* FP Arithmetic */
    PHX_OP_ADDSD,
    PHX_OP_SUBSD,
    PHX_OP_MULSD,
    PHX_OP_DIVSD,

    /* SSE Misc */
    PHX_OP_PTEST,
    PHX_OP_PCMPEQW,
    PHX_OP_PSRLQ,
    PHX_OP_PXOR,

    /* Sign Extend */
    PHX_OP_CDQ,
    PHX_OP_CQO
} PhxX86Opcode;

/* ------------------------------------------------------------------ */
/*  Memory operand with explicit access size                           */
/* ------------------------------------------------------------------ */

static inline PhxMem phx_byte_ptr(PhxGp base, int32_t offset) {
    PhxMem m = phx_ptr(base, offset);
    m.size = 1;
    return m;
}

static inline PhxMem phx_word_ptr(PhxGp base, int32_t offset) {
    PhxMem m = phx_ptr(base, offset);
    m.size = 2;
    return m;
}

static inline PhxMem phx_dword_ptr(PhxGp base, int32_t offset) {
    PhxMem m = phx_ptr(base, offset);
    m.size = 4;
    return m;
}

static inline PhxMem phx_qword_ptr(PhxGp base, int32_t offset) {
    PhxMem m = phx_ptr(base, offset);
    m.size = 8;
    return m;
}

static inline PhxMem phx_dqword_ptr(PhxGp base, int32_t offset) {
    PhxMem m = phx_ptr(base, offset);
    m.size = 16;
    return m;
}

/* ------------------------------------------------------------------ */
/*  Data Movement Instructions                                         */
/* ------------------------------------------------------------------ */

/* MOV variants */
void phx_x86_mov_rr(PhxBuilder *b, PhxGp dst, PhxGp src);
void phx_x86_mov_rm(PhxBuilder *b, PhxGp dst, PhxMem src);
void phx_x86_mov_mr(PhxBuilder *b, PhxMem dst, PhxGp src);
void phx_x86_mov_ri(PhxBuilder *b, PhxGp dst, int64_t imm);
void phx_x86_mov_mi(PhxBuilder *b, PhxMem dst, int32_t imm);

/* LEA */
void phx_x86_lea(PhxBuilder *b, PhxGp dst, PhxMem src);

/* MOVSD (scalar double) */
void phx_x86_movsd_rr(PhxBuilder *b, PhxGp dst, PhxGp src);
void phx_x86_movsd_rm(PhxBuilder *b, PhxGp dst, PhxMem src);
void phx_x86_movsd_mr(PhxBuilder *b, PhxMem dst, PhxGp src);

/* MOVSX (sign-extend byte/word to dword) */
void phx_x86_movsx_rr(PhxBuilder *b, PhxGp dst, PhxGp src);
void phx_x86_movsx_rm(PhxBuilder *b, PhxGp dst, PhxMem src);

/* MOVSXD (sign-extend dword to qword) */
void phx_x86_movsxd_rr(PhxBuilder *b, PhxGp dst, PhxGp src);
void phx_x86_movsxd_rm(PhxBuilder *b, PhxGp dst, PhxMem src);

/* MOVZX (zero-extend byte/word to dword/qword) */
void phx_x86_movzx_rr(PhxBuilder *b, PhxGp dst, PhxGp src);
void phx_x86_movzx_rm(PhxBuilder *b, PhxGp dst, PhxMem src);

/* MOVQ (move quadword to/from XMM) */
void phx_x86_movq_rr(PhxBuilder *b, PhxGp dst, PhxGp src);

/* PUSH / POP */
void phx_x86_push_r(PhxBuilder *b, PhxGp src);
void phx_x86_push_m(PhxBuilder *b, PhxMem src);
void phx_x86_push_i(PhxBuilder *b, int32_t imm);
void phx_x86_pop_r(PhxBuilder *b, PhxGp dst);
void phx_x86_pop_m(PhxBuilder *b, PhxMem dst);

/* XCHG */
void phx_x86_xchg_rr(PhxBuilder *b, PhxGp a, PhxGp c);

/* MOVDQU (move unaligned double quadword) */
void phx_x86_movdqu_rm(PhxBuilder *b, PhxGp dst, PhxMem src);
void phx_x86_movdqu_mr(PhxBuilder *b, PhxMem dst, PhxGp src);

/* LEAVE */
void phx_x86_leave(PhxBuilder *b);

/* CMOVNZ (conditional move if not zero) */
void phx_x86_cmovnz_rr(PhxBuilder *b, PhxGp dst, PhxGp src);
void phx_x86_cmovnz_rm(PhxBuilder *b, PhxGp dst, PhxMem src);

/* ------------------------------------------------------------------ */
/*  Arithmetic Instructions                                            */
/* ------------------------------------------------------------------ */

/* ADD variants */
void phx_x86_add_rr(PhxBuilder *b, PhxGp dst, PhxGp src);
void phx_x86_add_rm(PhxBuilder *b, PhxGp dst, PhxMem src);
void phx_x86_add_mr(PhxBuilder *b, PhxMem dst, PhxGp src);
void phx_x86_add_ri(PhxBuilder *b, PhxGp dst, int32_t imm);
void phx_x86_add_mi(PhxBuilder *b, PhxMem dst, int32_t imm);

/* SUB variants */
void phx_x86_sub_rr(PhxBuilder *b, PhxGp dst, PhxGp src);
void phx_x86_sub_rm(PhxBuilder *b, PhxGp dst, PhxMem src);
void phx_x86_sub_mr(PhxBuilder *b, PhxMem dst, PhxGp src);
void phx_x86_sub_ri(PhxBuilder *b, PhxGp dst, int32_t imm);
void phx_x86_sub_mi(PhxBuilder *b, PhxMem dst, int32_t imm);

/* NEG */
void phx_x86_neg_r(PhxBuilder *b, PhxGp dst);
void phx_x86_neg_m(PhxBuilder *b, PhxMem dst);

/* INC / DEC */
void phx_x86_inc_r(PhxBuilder *b, PhxGp dst);
void phx_x86_inc_m(PhxBuilder *b, PhxMem dst);
void phx_x86_dec_r(PhxBuilder *b, PhxGp dst);
void phx_x86_dec_m(PhxBuilder *b, PhxMem dst);

/* IMUL */
void phx_x86_imul_rr(PhxBuilder *b, PhxGp dst, PhxGp src);
void phx_x86_imul_rm(PhxBuilder *b, PhxGp dst, PhxMem src);
void phx_x86_imul_rri(PhxBuilder *b, PhxGp dst, PhxGp src, int32_t imm);

/* IDIV / DIV */
void phx_x86_idiv_r(PhxBuilder *b, PhxGp src);
void phx_x86_idiv_m(PhxBuilder *b, PhxMem src);
void phx_x86_div_r(PhxBuilder *b, PhxGp src);
void phx_x86_div_m(PhxBuilder *b, PhxMem src);

/* ------------------------------------------------------------------ */
/*  Logic Instructions                                                 */
/* ------------------------------------------------------------------ */

void phx_x86_xor_rr(PhxBuilder *b, PhxGp dst, PhxGp src);
void phx_x86_xor_rm(PhxBuilder *b, PhxGp dst, PhxMem src);
void phx_x86_xor_ri(PhxBuilder *b, PhxGp dst, int32_t imm);

void phx_x86_and_rr(PhxBuilder *b, PhxGp dst, PhxGp src);
void phx_x86_and_rm(PhxBuilder *b, PhxGp dst, PhxMem src);
void phx_x86_and_ri(PhxBuilder *b, PhxGp dst, int32_t imm);
void phx_x86_and_mi(PhxBuilder *b, PhxMem dst, int32_t imm);

void phx_x86_or_rr(PhxBuilder *b, PhxGp dst, PhxGp src);
void phx_x86_or_rm(PhxBuilder *b, PhxGp dst, PhxMem src);
void phx_x86_or_ri(PhxBuilder *b, PhxGp dst, int32_t imm);
void phx_x86_or_mi(PhxBuilder *b, PhxMem dst, int32_t imm);

void phx_x86_not_r(PhxBuilder *b, PhxGp dst);
void phx_x86_not_m(PhxBuilder *b, PhxMem dst);

/* ------------------------------------------------------------------ */
/*  Comparison / Test                                                  */
/* ------------------------------------------------------------------ */

void phx_x86_cmp_rr(PhxBuilder *b, PhxGp a, PhxGp c);
void phx_x86_cmp_rm(PhxBuilder *b, PhxGp a, PhxMem c);
void phx_x86_cmp_mr(PhxBuilder *b, PhxMem a, PhxGp c);
void phx_x86_cmp_ri(PhxBuilder *b, PhxGp a, int32_t imm);
void phx_x86_cmp_mi(PhxBuilder *b, PhxMem a, int32_t imm);

void phx_x86_test_rr(PhxBuilder *b, PhxGp a, PhxGp c);
void phx_x86_test_ri(PhxBuilder *b, PhxGp a, int32_t imm);
void phx_x86_test_mi(PhxBuilder *b, PhxMem a, int32_t imm);

void phx_x86_bt_rr(PhxBuilder *b, PhxGp a, PhxGp bit);
void phx_x86_bt_ri(PhxBuilder *b, PhxGp a, uint8_t bit);

void phx_x86_comisd(PhxBuilder *b, PhxGp a, PhxGp c);

/* ------------------------------------------------------------------ */
/*  SETcc                                                              */
/* ------------------------------------------------------------------ */

void phx_x86_setcc_r(PhxBuilder *b, PhxCondCode cc, PhxGp dst);
void phx_x86_setcc_m(PhxBuilder *b, PhxCondCode cc, PhxMem dst);

/* Convenience wrappers */
static inline void phx_x86_sete(PhxBuilder *b, PhxGp d)  { phx_x86_setcc_r(b, PHX_CC_E, d);  }
static inline void phx_x86_setne(PhxBuilder *b, PhxGp d) { phx_x86_setcc_r(b, PHX_CC_NE, d); }
static inline void phx_x86_setg(PhxBuilder *b, PhxGp d)  { phx_x86_setcc_r(b, PHX_CC_G, d);  }
static inline void phx_x86_setge(PhxBuilder *b, PhxGp d) { phx_x86_setcc_r(b, PHX_CC_GE, d); }
static inline void phx_x86_setl(PhxBuilder *b, PhxGp d)  { phx_x86_setcc_r(b, PHX_CC_L, d);  }
static inline void phx_x86_setle(PhxBuilder *b, PhxGp d) { phx_x86_setcc_r(b, PHX_CC_LE, d); }
static inline void phx_x86_seta(PhxBuilder *b, PhxGp d)  { phx_x86_setcc_r(b, PHX_CC_A, d);  }
static inline void phx_x86_setae(PhxBuilder *b, PhxGp d) { phx_x86_setcc_r(b, PHX_CC_AE, d); }
static inline void phx_x86_setb(PhxBuilder *b, PhxGp d)  { phx_x86_setcc_r(b, PHX_CC_B, d);  }
static inline void phx_x86_setbe(PhxBuilder *b, PhxGp d) { phx_x86_setcc_r(b, PHX_CC_BE, d); }

/* ------------------------------------------------------------------ */
/*  Branches                                                           */
/* ------------------------------------------------------------------ */

/* Unconditional jump */
void phx_x86_jmp_label(PhxBuilder *b, PhxLabel target);
void phx_x86_jmp_r(PhxBuilder *b, PhxGp target);
void phx_x86_jmp_m(PhxBuilder *b, PhxMem target);

/* Conditional jump (generic, with condition code) */
void phx_x86_jcc(PhxBuilder *b, PhxCondCode cc, PhxLabel target);

/* Convenience wrappers for conditional jumps */
static inline void phx_x86_je(PhxBuilder *b, PhxLabel t)  { phx_x86_jcc(b, PHX_CC_E, t);  }
static inline void phx_x86_jne(PhxBuilder *b, PhxLabel t) { phx_x86_jcc(b, PHX_CC_NE, t); }
static inline void phx_x86_jz(PhxBuilder *b, PhxLabel t)  { phx_x86_jcc(b, PHX_CC_Z, t);  }
static inline void phx_x86_jnz(PhxBuilder *b, PhxLabel t) { phx_x86_jcc(b, PHX_CC_NZ, t); }
static inline void phx_x86_ja(PhxBuilder *b, PhxLabel t)  { phx_x86_jcc(b, PHX_CC_A, t);  }
static inline void phx_x86_jae(PhxBuilder *b, PhxLabel t) { phx_x86_jcc(b, PHX_CC_AE, t); }
static inline void phx_x86_jb(PhxBuilder *b, PhxLabel t)  { phx_x86_jcc(b, PHX_CC_B, t);  }
static inline void phx_x86_jbe(PhxBuilder *b, PhxLabel t) { phx_x86_jcc(b, PHX_CC_BE, t); }
static inline void phx_x86_jg(PhxBuilder *b, PhxLabel t)  { phx_x86_jcc(b, PHX_CC_G, t);  }
static inline void phx_x86_jge(PhxBuilder *b, PhxLabel t) { phx_x86_jcc(b, PHX_CC_GE, t); }
static inline void phx_x86_jl(PhxBuilder *b, PhxLabel t)  { phx_x86_jcc(b, PHX_CC_L, t);  }
static inline void phx_x86_jle(PhxBuilder *b, PhxLabel t) { phx_x86_jcc(b, PHX_CC_LE, t); }
static inline void phx_x86_jc(PhxBuilder *b, PhxLabel t)  { phx_x86_jcc(b, PHX_CC_C, t);  }
static inline void phx_x86_jnc(PhxBuilder *b, PhxLabel t) { phx_x86_jcc(b, PHX_CC_NC, t); }
static inline void phx_x86_jo(PhxBuilder *b, PhxLabel t)  { phx_x86_jcc(b, PHX_CC_O, t);  }
static inline void phx_x86_jno(PhxBuilder *b, PhxLabel t) { phx_x86_jcc(b, PHX_CC_NO, t); }
static inline void phx_x86_js(PhxBuilder *b, PhxLabel t)  { phx_x86_jcc(b, PHX_CC_S, t);  }
static inline void phx_x86_jns(PhxBuilder *b, PhxLabel t) { phx_x86_jcc(b, PHX_CC_NS, t); }

/* ------------------------------------------------------------------ */
/*  Call / Return                                                      */
/* ------------------------------------------------------------------ */

void phx_x86_call_label(PhxBuilder *b, PhxLabel target);
void phx_x86_call_r(PhxBuilder *b, PhxGp target);
void phx_x86_call_m(PhxBuilder *b, PhxMem target);

void phx_x86_ret(PhxBuilder *b);
void phx_x86_ud2(PhxBuilder *b);

/* ------------------------------------------------------------------ */
/*  FP Arithmetic (scalar double)                                      */
/* ------------------------------------------------------------------ */

void phx_x86_addsd_rr(PhxBuilder *b, PhxGp dst, PhxGp src);
void phx_x86_addsd_rm(PhxBuilder *b, PhxGp dst, PhxMem src);
void phx_x86_subsd_rr(PhxBuilder *b, PhxGp dst, PhxGp src);
void phx_x86_subsd_rm(PhxBuilder *b, PhxGp dst, PhxMem src);
void phx_x86_mulsd_rr(PhxBuilder *b, PhxGp dst, PhxGp src);
void phx_x86_mulsd_rm(PhxBuilder *b, PhxGp dst, PhxMem src);
void phx_x86_divsd_rr(PhxBuilder *b, PhxGp dst, PhxGp src);
void phx_x86_divsd_rm(PhxBuilder *b, PhxGp dst, PhxMem src);

/* ------------------------------------------------------------------ */
/*  SSE Misc                                                           */
/* ------------------------------------------------------------------ */

void phx_x86_ptest_rr(PhxBuilder *b, PhxGp a, PhxGp c);
void phx_x86_pcmpeqw_rr(PhxBuilder *b, PhxGp dst, PhxGp src);
void phx_x86_psrlq_ri(PhxBuilder *b, PhxGp dst, uint8_t imm);
void phx_x86_pxor_rr(PhxBuilder *b, PhxGp dst, PhxGp src);

/* ------------------------------------------------------------------ */
/*  Sign Extend                                                        */
/* ------------------------------------------------------------------ */

void phx_x86_cdq(PhxBuilder *b);
void phx_x86_cqo(PhxBuilder *b);

/* ------------------------------------------------------------------ */
/*  Finalize                                                           */
/* ------------------------------------------------------------------ */

/* Resolve all x86_64 label fixups and linearize the node list into
 * the code holder's buffer.  Returns 0 on success, nonzero on error
 * (e.g. unresolved label, buffer allocation failure). */
int phx_x86_finalize(PhxBuilder *b);

#ifdef __cplusplus
}
#endif

#endif /* PHX_X86_64_H */
