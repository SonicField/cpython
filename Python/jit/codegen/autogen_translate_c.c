/*
 * autogen_translate_c.c -- C implementations of autogen translate* functions
 *
 * Phase 3D: Converts autogen.cpp translate* functions to pure C.
 * Each function emits machine code for a specific LIR instruction type
 * using phoenix-asm C API directly (no asmjit wrapper).
 *
 * Functions are registered in autogen.cpp via CALL_C() macro — no
 * changes to the registration code needed until the DSL is replaced.
 */

#include "cinderx/Jit/lir/lir_c_api.h"
#include "cinderx/Jit/lir/lir_types_c.h"
#include "cinderx/Jit/codegen/phylocation.h"

#include "jit/phoenix_asm/phoenix_asm.h"

#if defined(CINDER_X86_64)
#include "jit/phoenix_asm/x86_64.h"
#elif defined(CINDER_AARCH64)
#include "jit/phoenix_asm/arm64.h"
#endif

#include "Python.h"

#include <assert.h>
#include <limits.h>
#include <stdint.h>

/* Forward declarations for C functions defined elsewhere */
#if defined(CINDER_AARCH64)
/* From arch.c — ARM64 memory addressing helpers */
PhxMem jit_arch_ptr_offset(PhxGp base, int32_t offset, int32_t access_size);
PhxMem jit_arch_ptr_resolve(PhxBuilder *as, PhxGp base, int32_t offset,
                            PhxGp scratch, int32_t access_size);
#endif

/* ---- Helper: get PhxBuilder* from opaque Environ ---- */

static inline PhxBuilder *
get_builder(void *env) {
    return (PhxBuilder *)jit_environ_get_phx_builder(env);
}

/* ---- Helper: convert LIR operand to PhxGp register ---- */

static inline PhxGp
operand_to_gp(const LirOperand *op) {
    int reg = lir_operand_get_phy_register(op).loc;
    uint8_t dt = op->data_type_;
#if defined(CINDER_X86_64)
    PhxGp base = {(uint8_t)reg, 8};
    switch (dt) {
        case JIT_LIR_DT_8BIT:   return phx_gp8(base);
        case JIT_LIR_DT_16BIT:  return phx_gp16(base);
        case JIT_LIR_DT_32BIT:  return phx_gp32(base);
        case JIT_LIR_DT_OBJECT:
        case JIT_LIR_DT_64BIT:  return base;
        default: assert(0 && "bad GP data type"); return base;
    }
#elif defined(CINDER_AARCH64)
    switch (dt) {
        case JIT_LIR_DT_8BIT:
        case JIT_LIR_DT_16BIT:
        case JIT_LIR_DT_32BIT:
            return PHX_REG_GP(reg, 4); /* Wn (32-bit) */
        case JIT_LIR_DT_OBJECT:
        case JIT_LIR_DT_64BIT:
            return PHX_REG_GP(reg, 8); /* Xn (64-bit) */
        default: assert(0 && "bad GP data type");
            return PHX_REG_GP(reg, 8);
    }
#endif
}

/* ---- Helper: convert LIR operand to PhxGp for output (ARM64 8/16 → W) ---- */

static inline PhxGp
operand_to_gp_output(const LirOperand *op) {
#if defined(CINDER_AARCH64)
    int reg = lir_operand_get_phy_register(op).loc;
    uint8_t dt = op->data_type_;
    if (dt == JIT_LIR_DT_8BIT || dt == JIT_LIR_DT_16BIT) {
        return PHX_REG_GP(reg, 4);
    }
    return operand_to_gp(op);
#else
    return operand_to_gp(op);
#endif
}

#if defined(CINDER_X86_64)

/* ---- Helper: convert LIR operand to FP register (x86_64 XMM) ---- */

static inline PhxGp
operand_to_fp(const LirOperand *op) {
    int reg = lir_operand_get_phy_register(op).loc;
    return (PhxGp){(uint8_t)reg, 8};
}

#elif defined(CINDER_AARCH64)

/* ---- Helper: convert LIR operand to PhxVecD register (ARM64) ---- */

static inline PhxGp
operand_to_vecd(const LirOperand *op) {
    int reg = lir_operand_get_phy_register(op).loc - PHYLOC_VECD_REG_BASE;
    return PHX_REG_FP(reg, 8); /* Dn (64-bit FP) */
}

#endif

/* ---- Helper: ARM64 isAddSubImm check ---- */
#if defined(CINDER_AARCH64)

static inline int
is_add_sub_imm(uint64_t val) {
    return val <= 0xFFF || (val <= 0xFFF000 && (val & 0xFFF) == 0);
}

/* ARM64 scratch registers */
#define A64_SCRATCH_0   PHX_REG_GP(12, 8)  /* X12 */
#define A64_SCRATCH_0_W PHX_REG_GP(12, 4)  /* W12 */
#define A64_SCRATCH_1   PHX_REG_GP(13, 8)  /* X13 */
#define A64_SCRATCH_1_W PHX_REG_GP(13, 4)  /* W13 */
#define A64_FP          PHX_REG_GP(29, 8)  /* X29 (FP) */
#define A64_SP          PHX_REG_GP(31, 8)  /* SP */

/* ================================================================
 * ARM64 translate* functions — C implementations
 *
 * These replace the C++ versions in autogen.cpp.
 * Each has extern "C" linkage and can be called via CALL_C() macro.
 * ================================================================ */

void
autogen_c_translateUnreachable(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    phx_a64_udf(pb, 0);
}

/* ---- Add/Sub ---- */

static void
translate_add_sub_op(void *env, const LirInstruction *instr, int is_sub) {
    PhxBuilder *pb = get_builder(env);

    const LirOperand *output =
        (instr->output_.type_ != JIT_LIR_OPTYPE_NONE)
        ? &instr->output_ : instr->inputs_[0];
    const LirOperand *opnd0 = instr->inputs_[0];
    const LirOperand *opnd1 = instr->inputs_[1];

    assert(output->type_ == JIT_LIR_OPTYPE_REG);
    assert(opnd0->type_ == JIT_LIR_OPTYPE_REG);

    PhxGp output_reg = operand_to_gp(output);
    PhxGp opnd0_reg = operand_to_gp(opnd0);

    if (opnd1->type_ == JIT_LIR_OPTYPE_IMM) {
        uint64_t constant = lir_operand_get_constant(opnd1);
        assert(is_add_sub_imm(constant));
        if (is_sub)
            phx_a64_sub_rri(pb, output_reg, opnd0_reg, constant);
        else
            phx_a64_add_rri(pb, output_reg, opnd0_reg, constant);
    } else if (opnd1->type_ == JIT_LIR_OPTYPE_REG) {
        if (is_sub)
            phx_a64_sub_rrr(pb, output_reg, opnd0_reg, operand_to_gp(opnd1));
        else
            phx_a64_add_rrr(pb, output_reg, opnd0_reg, operand_to_gp(opnd1));
    } else if (opnd1->type_ == JIT_LIR_OPTYPE_STACK) {
        int32_t loc = lir_operand_get_stack_slot(opnd1).loc;
        PhxMem ptr = jit_arch_ptr_resolve(pb, A64_FP, loc, A64_SCRATCH_0, 8);
        phx_a64_ldr(pb, A64_SCRATCH_0, ptr);
        if (is_sub)
            phx_a64_sub_rrr(pb, output_reg, opnd0_reg, A64_SCRATCH_0);
        else
            phx_a64_add_rrr(pb, output_reg, opnd0_reg, A64_SCRATCH_0);
    } else {
        assert(0 && "Unsupported operand type for add/sub");
    }
}

void
autogen_c_translateAdd(void *env, const LirInstruction *instr) {
    translate_add_sub_op(env, instr, 0);
}

void
autogen_c_translateSub(void *env, const LirInstruction *instr) {
    translate_add_sub_op(env, instr, 1);
}

/* ---- Inc/Dec ---- */

static void
translate_inc_dec_op(void *env, const LirInstruction *instr, int is_dec) {
    PhxBuilder *pb = get_builder(env);

    const LirOperand *opnd = instr->inputs_[0];

    if (opnd->type_ == JIT_LIR_OPTYPE_REG) {
        PhxGp reg = operand_to_gp(opnd);
        if (is_dec)
            phx_a64_sub_rri(pb, reg, reg, 1);
        else
            phx_a64_add_rri(pb, reg, reg, 1);
    } else if (opnd->type_ == JIT_LIR_OPTYPE_STACK) {
        int32_t loc = lir_operand_get_stack_slot(opnd).loc;
        PhxMem ptr = jit_arch_ptr_resolve(pb, A64_FP, loc, A64_SCRATCH_1, 8);
        phx_a64_ldr(pb, A64_SCRATCH_0, ptr);
        if (is_dec)
            phx_a64_sub_rri(pb, A64_SCRATCH_0, A64_SCRATCH_0, 1);
        else
            phx_a64_add_rri(pb, A64_SCRATCH_0, A64_SCRATCH_0, 1);
        phx_a64_str(pb, A64_SCRATCH_0, ptr);
    } else {
        assert(0 && "Unsupported operand type for inc/dec");
    }
}

void
autogen_c_translateInc(void *env, const LirInstruction *instr) {
    translate_inc_dec_op(env, instr, 0);
}

void
autogen_c_translateDec(void *env, const LirInstruction *instr) {
    translate_inc_dec_op(env, instr, 1);
}

/* ---- Logical ops (And/Or/Xor) ---- */

enum { LOGICAL_AND = 0, LOGICAL_ORR = 1, LOGICAL_EOR = 2 };

static inline int
is_logical_imm(uint64_t val, uint32_t width) {
    if (val == 0) return 0;
    if (width == 32) val = (val & 0xFFFFFFFF) | (val << 32);
    if (val == ~(uint64_t)0) return 0;
    for (uint32_t size = 2; size <= 64; size <<= 1) {
        uint64_t mask = (~(uint64_t)0) >> (64 - size);
        uint64_t elem = val & mask;
        uint64_t rep = 0;
        for (uint32_t i = 0; i < 64; i += size) rep |= (elem << i);
        if (rep != val) continue;
        if (elem == 0 || elem == mask) continue;
        uint64_t transitions = elem ^ ((elem >> 1) | ((elem & 1) << (size - 1)));
        if (__builtin_popcountll(transitions) == 2) return 1;
    }
    return 0;
}

static void
translate_logical_op(void *env, const LirInstruction *instr, int op) {
    PhxBuilder *pb = get_builder(env);

    const LirOperand *output =
        (instr->output_.type_ != JIT_LIR_OPTYPE_NONE)
        ? &instr->output_ : instr->inputs_[0];
    const LirOperand *opnd0 = instr->inputs_[0];
    const LirOperand *opnd1 = instr->inputs_[1];

    assert(output->type_ == JIT_LIR_OPTYPE_REG);
    assert(opnd0->type_ == JIT_LIR_OPTYPE_REG);

    PhxGp out_reg = operand_to_gp(output);
    PhxGp r0 = operand_to_gp(opnd0);

    if (opnd1->type_ == JIT_LIR_OPTYPE_IMM) {
        uint64_t c = lir_operand_get_constant(opnd1);
        switch (op) {
            case LOGICAL_AND: phx_a64_and_rri(pb, out_reg, r0, c); break;
            case LOGICAL_ORR: phx_a64_orr_rri(pb, out_reg, r0, c); break;
            case LOGICAL_EOR: phx_a64_eor_rri(pb, out_reg, r0, c); break;
        }
    } else if (opnd1->type_ == JIT_LIR_OPTYPE_REG) {
        PhxGp r1 = operand_to_gp(opnd1);
        switch (op) {
            case LOGICAL_AND: phx_a64_and_rrr(pb, out_reg, r0, r1); break;
            case LOGICAL_ORR: phx_a64_orr_rrr(pb, out_reg, r0, r1); break;
            case LOGICAL_EOR: phx_a64_eor_rrr(pb, out_reg, r0, r1); break;
        }
    } else if (opnd1->type_ == JIT_LIR_OPTYPE_STACK) {
        int32_t loc = lir_operand_get_stack_slot(opnd1).loc;
        PhxMem ptr = jit_arch_ptr_resolve(pb, A64_FP, loc, A64_SCRATCH_0, 8);
        phx_a64_ldr(pb, A64_SCRATCH_0, ptr);
        switch (op) {
            case LOGICAL_AND: phx_a64_and_rrr(pb, out_reg, r0, A64_SCRATCH_0); break;
            case LOGICAL_ORR: phx_a64_orr_rrr(pb, out_reg, r0, A64_SCRATCH_0); break;
            case LOGICAL_EOR: phx_a64_eor_rrr(pb, out_reg, r0, A64_SCRATCH_0); break;
        }
    } else {
        assert(0 && "Unsupported operand type for logical op");
    }
}

void autogen_c_translateAnd(void *env, const LirInstruction *instr) {
    translate_logical_op(env, instr, LOGICAL_AND);
}
void autogen_c_translateOr(void *env, const LirInstruction *instr) {
    translate_logical_op(env, instr, LOGICAL_ORR);
}
void autogen_c_translateXor(void *env, const LirInstruction *instr) {
    translate_logical_op(env, instr, LOGICAL_EOR);
}

/* ---- Mul ---- */

void
autogen_c_translateMul(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);

    const LirOperand *output =
        (instr->output_.type_ != JIT_LIR_OPTYPE_NONE)
        ? &instr->output_ : instr->inputs_[0];
    const LirOperand *opnd0 = instr->inputs_[0];
    const LirOperand *opnd1 = instr->inputs_[1];

    assert(output->type_ == JIT_LIR_OPTYPE_REG);
    assert(opnd0->type_ == JIT_LIR_OPTYPE_REG);

    PhxGp out_reg = operand_to_gp(output);
    PhxGp r0 = operand_to_gp(opnd0);

    if (opnd1->type_ == JIT_LIR_OPTYPE_IMM) {
        phx_a64_mov_ri(pb, A64_SCRATCH_0, lir_operand_get_constant(opnd1));
        phx_a64_mul(pb, out_reg, r0, A64_SCRATCH_0);
    } else if (opnd1->type_ == JIT_LIR_OPTYPE_REG) {
        phx_a64_mul(pb, out_reg, r0, operand_to_gp(opnd1));
    } else if (opnd1->type_ == JIT_LIR_OPTYPE_STACK) {
        int32_t loc = lir_operand_get_stack_slot(opnd1).loc;
        PhxMem ptr = jit_arch_ptr_resolve(pb, A64_FP, loc, A64_SCRATCH_0, 8);
        phx_a64_ldr(pb, A64_SCRATCH_0, ptr);
        phx_a64_mul(pb, out_reg, r0, A64_SCRATCH_0);
    } else {
        assert(0 && "Unsupported operand type for Mul");
    }
}

/* ---- Div/DivUn ---- */

static void
translate_div_op(void *env, const LirInstruction *instr, int is_unsigned) {
    PhxBuilder *pb = get_builder(env);

    const LirOperand *output =
        (instr->output_.type_ != JIT_LIR_OPTYPE_NONE)
        ? &instr->output_ : instr->inputs_[0];
    const LirOperand *opnd0 = instr->inputs_[0];
    const LirOperand *opnd1 = instr->inputs_[1];

    assert(output->type_ == JIT_LIR_OPTYPE_REG);
    assert(opnd0->type_ == JIT_LIR_OPTYPE_REG);

    PhxGp out_reg = operand_to_gp(output);
    PhxGp r0 = operand_to_gp(opnd0);

    PhxGp divisor;
    if (opnd1->type_ == JIT_LIR_OPTYPE_REG) {
        divisor = operand_to_gp(opnd1);
    } else if (opnd1->type_ == JIT_LIR_OPTYPE_STACK) {
        int32_t loc = lir_operand_get_stack_slot(opnd1).loc;
        PhxMem ptr = jit_arch_ptr_resolve(pb, A64_FP, loc, A64_SCRATCH_0, 8);
        phx_a64_ldr(pb, A64_SCRATCH_0, ptr);
        divisor = A64_SCRATCH_0;
    } else {
        assert(0 && "Unsupported operand type for Div");
        divisor = A64_SCRATCH_0;
    }

    if (is_unsigned)
        phx_a64_udiv(pb, out_reg, r0, divisor);
    else
        phx_a64_sdiv(pb, out_reg, r0, divisor);
}

void autogen_c_translateDiv(void *env, const LirInstruction *instr) {
    translate_div_op(env, instr, 0);
}
void autogen_c_translateDivUn(void *env, const LirInstruction *instr) {
    translate_div_op(env, instr, 1);
}

/* ---- Push/Pop ---- */

void
autogen_c_translatePush(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    const LirOperand *operand = instr->inputs_[0];

    if (operand->type_ == JIT_LIR_OPTYPE_IMM) {
        phx_a64_mov_ri(pb, A64_SCRATCH_0, lir_operand_get_constant(operand));
        phx_a64_str(pb, A64_SCRATCH_0,
            jit_arch_ptr_offset(A64_SP, -16, 8)); /* pre-index */
    } else if (operand->type_ == JIT_LIR_OPTYPE_REG) {
        phx_a64_str(pb, operand_to_gp(operand),
            jit_arch_ptr_offset(A64_SP, -16, 8));
    } else if (operand->type_ == JIT_LIR_OPTYPE_STACK) {
        int32_t loc = lir_operand_get_stack_slot(operand).loc;
        PhxMem ptr = jit_arch_ptr_resolve(pb, A64_FP, loc, A64_SCRATCH_1, 8);
        phx_a64_ldr(pb, A64_SCRATCH_0, ptr);
        phx_a64_str(pb, A64_SCRATCH_0,
            jit_arch_ptr_offset(A64_SP, -16, 8));
    } else {
        assert(0 && "Unsupported operand type for push");
    }
}

void
autogen_c_translatePop(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    const LirOperand *operand = &instr->output_;

    if (operand->type_ == JIT_LIR_OPTYPE_REG) {
        PhxMem post = jit_arch_ptr_offset(A64_SP, 16, 8); /* post-index */
        phx_a64_ldr(pb, operand_to_gp(operand), post);
    } else if (operand->type_ == JIT_LIR_OPTYPE_STACK) {
        int32_t loc = lir_operand_get_stack_slot(operand).loc;
        PhxMem ptr = jit_arch_ptr_resolve(pb, A64_FP, loc, A64_SCRATCH_1, 8);
        PhxMem post = jit_arch_ptr_offset(A64_SP, 16, 8);
        phx_a64_ldr(pb, A64_SCRATCH_0, post);
        phx_a64_str(pb, A64_SCRATCH_0, ptr);
    } else {
        assert(0 && "Unsupported operand type for pop");
    }
}

/* ---- Exchange ---- */

void
autogen_c_translateExchange(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    const LirOperand *opnd0 = &instr->output_;
    const LirOperand *opnd1 = instr->inputs_[0];

    assert(opnd0->type_ == JIT_LIR_OPTYPE_REG);
    assert(opnd1->type_ == JIT_LIR_OPTYPE_REG);

    if (lir_operand_is_fp(opnd0) && lir_operand_is_fp(opnd1)) {
        PhxGp vec0 = operand_to_vecd(opnd0);
        PhxGp vec1 = operand_to_vecd(opnd1);
        phx_a64_fmov(pb, A64_SCRATCH_0, vec0);
        phx_a64_fmov(pb, vec0, vec1);
        phx_a64_fmov(pb, vec1, A64_SCRATCH_0);
    } else {
        PhxGp reg0 = operand_to_gp(opnd0);
        PhxGp reg1 = operand_to_gp(opnd1);
        phx_a64_mov_rr(pb, A64_SCRATCH_0, reg0);
        phx_a64_mov_rr(pb, reg0, reg1);
        phx_a64_mov_rr(pb, reg1, A64_SCRATCH_0);
    }
}

/* ---- Cmp ---- */

void
autogen_c_translateCmp(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    const LirOperand *inp0 = instr->inputs_[0];
    const LirOperand *inp1 = instr->inputs_[1];

    assert(inp0->type_ == JIT_LIR_OPTYPE_REG);

    if (inp1->type_ == JIT_LIR_OPTYPE_REG) {
        if (lir_operand_is_fp(inp0) && lir_operand_is_fp(inp1)) {
            phx_a64_fcmp(pb, operand_to_vecd(inp0), operand_to_vecd(inp1));
        } else {
            phx_a64_cmp_rr(pb, operand_to_gp(inp0), operand_to_gp(inp1));
        }
    } else if (inp1->type_ == JIT_LIR_OPTYPE_IMM) {
        uint64_t constant = lir_operand_get_constant(inp1);
        if (is_add_sub_imm(constant)) {
            phx_a64_cmp_ri(pb, operand_to_gp(inp0), constant);
        } else {
            phx_a64_mov_ri(pb, A64_SCRATCH_0, constant);
            phx_a64_cmp_rr(pb, operand_to_gp(inp0), A64_SCRATCH_0);
        }
    } else {
        assert(0 && "Unsupported operand types for cmp");
    }
}

/* ---- BitTest ---- */

void
autogen_c_translateBitTest(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    PhxGp test_reg = operand_to_gp(instr->inputs_[0]);
    uint64_t bit_pos = lir_operand_get_constant(instr->inputs_[1]);
    uint64_t mask = 1ULL << bit_pos;
    phx_a64_tst_ri(pb, test_reg, mask);
}

/* ---- MovZX/MovSX/MovSXD ---- */

static void
translate_mov_ext_op(void *env, const LirInstruction *instr, int is_signed) {
    PhxBuilder *pb = get_builder(env);

    PhxGp output = operand_to_gp_output(&instr->output_);
    const LirOperand *input = instr->inputs_[0];
    size_t input_size = lir_operand_size_in_bits(input);

    if (input->type_ == JIT_LIR_OPTYPE_REG) {
        PhxGp input_reg = operand_to_gp(input);
        switch (input_size) {
            case 8:
                if (is_signed) phx_a64_sxtb(pb, output, input_reg);
                else           phx_a64_uxtb(pb, output, input_reg);
                break;
            case 16:
                if (is_signed) phx_a64_sxth(pb, output, input_reg);
                else           phx_a64_uxth(pb, output, input_reg);
                break;
            case 32:
                phx_a64_mov_rr(pb, PHX_REG_GP(output.id, 4),
                               PHX_REG_GP(input_reg.id, 4));
                break;
            default:
                assert(0 && "Unsupported input size for mov ext");
        }
    } else if (input->type_ == JIT_LIR_OPTYPE_STACK) {
        int32_t loc = lir_operand_get_stack_slot(input).loc;
        int32_t access_sz = (int32_t)(input_size / 8);
        PhxMem ptr = jit_arch_ptr_resolve(pb, A64_FP, loc, A64_SCRATCH_0,
                                          access_sz);
        switch (input_size) {
            case 8:
                if (is_signed) phx_a64_ldrsb(pb, output, ptr);
                else           phx_a64_ldrb(pb, output, ptr);
                break;
            case 16:
                if (is_signed) phx_a64_ldrsh(pb, output, ptr);
                else           phx_a64_ldrh(pb, output, ptr);
                break;
            case 32: {
                PhxGp w_out = PHX_REG_GP(output.id, 4);
                PhxMem ptr32 = jit_arch_ptr_resolve(pb, A64_FP, loc,
                                                    A64_SCRATCH_0, 4);
                phx_a64_ldr(pb, w_out, ptr32);
                break;
            }
            default:
                assert(0 && "Unsupported input size for mov ext");
        }
    } else {
        assert(0 && "Unsupported operand type for mov ext");
    }
}

void autogen_c_translateMovZX(void *env, const LirInstruction *instr) {
    translate_mov_ext_op(env, instr, 0);
}
void autogen_c_translateMovSX(void *env, const LirInstruction *instr) {
    translate_mov_ext_op(env, instr, 1);
}

void
autogen_c_translateMovSXD(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);

    PhxGp output = operand_to_gp_output(&instr->output_);
    const LirOperand *input = instr->inputs_[0];

    if (input->type_ == JIT_LIR_OPTYPE_REG) {
        phx_a64_sxtw(pb, output, operand_to_gp(input));
    } else if (input->type_ == JIT_LIR_OPTYPE_STACK) {
        int32_t loc = lir_operand_get_stack_slot(input).loc;
        PhxMem ptr = jit_arch_ptr_resolve(pb, A64_FP, loc, A64_SCRATCH_0, 4);
        phx_a64_ldrsw(pb, output, ptr);
    } else {
        assert(0 && "Unsupported operand type for MovSXD");
    }
}

/* ---- Tst ---- */

void
autogen_c_translateTst(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);

    const LirOperand *opnd0 = instr->inputs_[0];
    const LirOperand *opnd1 = instr->inputs_[1];
    uint8_t dt = opnd0->data_type_;

    int shift = 0;
    if (dt == JIT_LIR_DT_8BIT) shift = 24;
    else if (dt == JIT_LIR_DT_16BIT) shift = 16;

    if (shift) {
        PhxGp w0 = PHX_REG_GP(lir_operand_get_phy_register(opnd0).loc, 4);
        PhxGp w1 = PHX_REG_GP(lir_operand_get_phy_register(opnd1).loc, 4);
        phx_a64_lsl(pb, A64_SCRATCH_0_W, w0, shift);
        phx_a64_lsl(pb, A64_SCRATCH_1_W, w1, shift);
        phx_a64_tst_rr(pb, A64_SCRATCH_0_W, A64_SCRATCH_1_W);
    } else {
        phx_a64_tst_rr(pb, operand_to_gp(opnd0), operand_to_gp(opnd1));
    }
}

/* ---- IntToBool ---- */

void
autogen_c_translateIntToBool(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    const LirOperand *input = instr->inputs_[0];
    PhxGp output = operand_to_gp_output(&instr->output_);

    assert(instr->output_.data_type_ == JIT_LIR_DT_8BIT);

    if (input->type_ == JIT_LIR_OPTYPE_IMM) {
        phx_a64_mov_ri(pb, output,
            lir_operand_get_constant(input) ? 1 : 0);
    } else {
        phx_a64_cmp_ri(pb, operand_to_gp(input), 0);
        phx_a64_cset(pb, output, PHX_COND_NE);
    }
}

/* ---- Select ---- */

void
autogen_c_translateSelect(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);

    PhxGp output = operand_to_gp_output(&instr->output_);
    const LirOperand *condition_op = instr->inputs_[0];
    uint8_t cond_dt = condition_op->data_type_;

    PhxGp condition_reg;
    if (cond_dt == JIT_LIR_DT_8BIT || cond_dt == JIT_LIR_DT_16BIT) {
        int reg_id = lir_operand_get_phy_register(condition_op).loc;
        condition_reg = PHX_REG_GP(reg_id, 4);
        uint64_t mask = (1ULL << jit_lir_bit_size(cond_dt)) - 1;
        phx_a64_and_rri(pb, condition_reg, condition_reg, mask);
    } else {
        condition_reg = operand_to_gp(condition_op);
    }

    PhxGp true_val_reg = operand_to_gp(instr->inputs_[1]);
    uint64_t false_val = lir_operand_get_constant(instr->inputs_[2]);

    phx_a64_mov_ri(pb, A64_SCRATCH_0, false_val);
    phx_a64_cmp_ri(pb, condition_reg, 0);
    phx_a64_csel(pb, output, true_val_reg, A64_SCRATCH_0, PHX_COND_NE);
}

/* ---- Lea helpers ---- */

static PhxGp
get_gp_or_sp(const LirOperand *op) {
    int reg = lir_operand_get_phy_register(op).loc;
    /* Register 31 on ARM64 is SP in this context */
    return PHX_REG_GP(reg, 8);
}

static void
lea_index(PhxBuilder *pb, PhxGp output, PhxGp base, PhxGp index,
          uint8_t multiplier) {
    switch (multiplier) {
        case 0:
            phx_a64_add_rrr(pb, output, base, index);
            break;
        case 1:
            phx_a64_add_rrr_shifted(pb, output, base, index, 0/*LSL*/, 1);
            break;
        case 2:
            phx_a64_add_rrr_shifted(pb, output, base, index, 0/*LSL*/, 2);
            break;
        case 3:
            phx_a64_add_rrr_shifted(pb, output, base, index, 0/*LSL*/, 3);
            break;
        default: {
            phx_a64_mov_ri(pb, A64_SCRATCH_0, (uint64_t)1 << multiplier);
            phx_a64_madd(pb, output, index, A64_SCRATCH_0, base);
            break;
        }
    }
}

static void
lea_indirect(PhxBuilder *pb, PhxGp output, PhxGp scratch0,
             const LirMemoryIndirect *indirect) {
    LirOperand *base_op = lir_memind_base_reg(indirect);
    PhxGp base = get_gp_or_sp(base_op);
    LirOperand *index_op = lir_memind_index_reg(indirect);
    int32_t offset = lir_memind_offset(indirect);

    if (index_op != NULL) {
        lea_index(pb, output, base, operand_to_gp(index_op),
                  lir_memind_multiplier(indirect));
        base = output;
    }

    if (offset > 0) {
        if (is_add_sub_imm((uint64_t)offset)) {
            phx_a64_add_rri(pb, output, base, offset);
        } else {
            phx_a64_mov_ri(pb, scratch0, offset);
            phx_a64_add_rrr(pb, output, base, scratch0);
        }
    } else if (offset < 0) {
        if (is_add_sub_imm((uint64_t)(-offset))) {
            phx_a64_sub_rri(pb, output, base, -offset);
        } else {
            phx_a64_mov_ri(pb, scratch0, -offset);
            phx_a64_sub_rrr(pb, output, base, scratch0);
        }
    } else if (index_op == NULL) {
        phx_a64_mov_rr(pb, output, base);
    }
}

/* ---- Lea ---- */

void
autogen_c_translateLea(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);

    const LirOperand *output = &instr->output_;
    const LirOperand *input = instr->inputs_[0];

    assert(output->type_ == JIT_LIR_OPTYPE_REG);

    PhxGp out_reg = operand_to_gp(output);

    if (input->type_ == JIT_LIR_OPTYPE_STACK) {
        int32_t loc = lir_operand_get_stack_slot(input).loc;
        if (loc >= 0) {
            phx_a64_add_rri(pb, out_reg, A64_FP, loc);
        } else {
            phx_a64_sub_rri(pb, out_reg, A64_FP, -loc);
        }
    } else if (input->type_ == JIT_LIR_OPTYPE_MEM) {
        uint64_t address = (uint64_t)(uintptr_t)lir_operand_get_mem_address(input);
        phx_a64_mov_ri(pb, out_reg, address);
    } else if (input->type_ == JIT_LIR_OPTYPE_IND) {
        LirMemoryIndirect *ind = lir_operand_get_indirect(input);
        lea_indirect(pb, out_reg, A64_SCRATCH_0, ind);
    } else {
        assert(0 && "Unsupported operand type for Lea");
    }
}

/* ---- ptrIndirect: resolve MemoryIndirect to PhxMem ---- */

static PhxMem
ptr_indirect(PhxBuilder *pb, PhxGp scratch0, PhxGp scratch1,
             const LirMemoryIndirect *indirect) {
    LirOperand *base_op = lir_memind_base_reg(indirect);
    PhxGp base = get_gp_or_sp(base_op);
    LirOperand *index_op = lir_memind_index_reg(indirect);
    int32_t offset = lir_memind_offset(indirect);

    if (index_op != NULL) {
        lea_index(pb, scratch1, base, operand_to_gp(index_op),
                  lir_memind_multiplier(indirect));
        base = scratch1;
    }

    return jit_arch_ptr_resolve(pb, base, offset, scratch0, 8);
}

/* ---- loadToReg: load from PhxMem into register operand ---- */

static void
load_to_reg(PhxBuilder *pb, const LirOperand *output, PhxMem input) {
    if (lir_operand_is_fp(output)) {
        phx_a64_ldr_fp(pb, operand_to_vecd(output), input);
    } else {
        switch (output->data_type_) {
            case JIT_LIR_DT_8BIT:
                phx_a64_ldrb(pb, operand_to_gp_output(output), input);
                break;
            case JIT_LIR_DT_16BIT:
                phx_a64_ldrh(pb, operand_to_gp_output(output), input);
                break;
            default:
                phx_a64_ldr(pb, operand_to_gp(output), input);
                break;
        }
    }
}

/* ---- storeFromReg: store from register operand to PhxMem ---- */

static void
store_from_reg(PhxBuilder *pb, const LirOperand *input, PhxMem output) {
    if (lir_operand_is_fp(input)) {
        phx_a64_str_fp(pb, operand_to_vecd(input), output);
    } else {
        int reg = lir_operand_get_phy_register(input).loc;
        switch (input->data_type_) {
            case JIT_LIR_DT_8BIT:
                phx_a64_strb(pb, PHX_REG_GP(reg, 4), output);
                break;
            case JIT_LIR_DT_16BIT:
                phx_a64_strh(pb, PHX_REG_GP(reg, 4), output);
                break;
            default:
                phx_a64_str(pb, operand_to_gp(input), output);
                break;
        }
    }
}

/* ---- Move ---- */

void
autogen_c_translateMove(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);

    const LirOperand *output = &instr->output_;
    const LirOperand *input = instr->inputs_[0];

    switch (output->type_) {
    case JIT_LIR_OPTYPE_REG:
        switch (input->type_) {
        case JIT_LIR_OPTYPE_REG:
            if (lir_operand_is_fp(output)) {
                if (lir_operand_is_fp(input)) {
                    phx_a64_fmov(pb, operand_to_vecd(output),
                                 operand_to_vecd(input));
                } else {
                    phx_a64_fmov(pb, operand_to_vecd(output),
                                 operand_to_gp(input));
                }
            } else {
                if (lir_operand_is_fp(input)) {
                    phx_a64_fmov(pb, operand_to_gp(output),
                                 operand_to_vecd(input));
                } else {
                    phx_a64_mov_rr(pb, operand_to_gp(output),
                                   operand_to_gp(input));
                }
            }
            break;
        case JIT_LIR_OPTYPE_STACK: {
            int32_t loc = lir_operand_get_stack_slot(input).loc;
            PhxMem ptr = jit_arch_ptr_resolve(pb, A64_FP, loc,
                                              A64_SCRATCH_0, 8);
            if (lir_operand_is_fp(output)) {
                phx_a64_ldr_fp(pb, operand_to_vecd(output), ptr);
            } else {
                switch (output->data_type_) {
                    case JIT_LIR_DT_8BIT:
                        phx_a64_ldrb(pb, operand_to_gp_output(output), ptr);
                        break;
                    case JIT_LIR_DT_16BIT:
                        phx_a64_ldrh(pb, operand_to_gp_output(output), ptr);
                        break;
                    default:
                        phx_a64_ldr(pb, operand_to_gp(output), ptr);
                        break;
                }
            }
            break;
        }
        case JIT_LIR_OPTYPE_MEM:
            phx_a64_mov_ri(pb, A64_SCRATCH_0,
                (uint64_t)(uintptr_t)lir_operand_get_mem_address(input));
            load_to_reg(pb, output, phx_ptr(A64_SCRATCH_0, 0));
            break;
        case JIT_LIR_OPTYPE_IND: {
            LirMemoryIndirect *ind = lir_operand_get_indirect(input);
            PhxMem ptr = ptr_indirect(pb, A64_SCRATCH_0, A64_SCRATCH_1, ind);
            load_to_reg(pb, output, ptr);
            break;
        }
        case JIT_LIR_OPTYPE_IMM:
            if (lir_operand_is_fp(output)) {
                phx_a64_mov_ri(pb, A64_SCRATCH_0,
                    lir_operand_get_constant(input));
                phx_a64_fmov(pb, operand_to_vecd(output), A64_SCRATCH_0);
            } else {
                phx_a64_mov_ri(pb, operand_to_gp(output),
                    lir_operand_get_constant(input));
            }
            break;
        default:
            assert(0 && "Unsupported input type for Move:Reg");
        }
        break;

    case JIT_LIR_OPTYPE_STACK: {
        int32_t loc = lir_operand_get_stack_slot(output).loc;
        PhxMem ptr = jit_arch_ptr_resolve(pb, A64_FP, loc, A64_SCRATCH_0, 8);
        if (input->type_ == JIT_LIR_OPTYPE_REG) {
            store_from_reg(pb, input, ptr);
        } else if (input->type_ == JIT_LIR_OPTYPE_IMM) {
            phx_a64_mov_ri(pb, A64_SCRATCH_0,
                lir_operand_get_constant(input));
            phx_a64_str(pb, A64_SCRATCH_0, ptr);
        } else {
            assert(0 && "Unsupported input type for Move:Stk");
        }
        break;
    }

    case JIT_LIR_OPTYPE_MEM: {
        phx_a64_mov_ri(pb, A64_SCRATCH_0,
            (uint64_t)(uintptr_t)lir_operand_get_mem_address(output));
        if (input->type_ == JIT_LIR_OPTYPE_REG) {
            if (lir_operand_is_fp(input)) {
                phx_a64_str_fp(pb, operand_to_vecd(input),
                    phx_ptr(A64_SCRATCH_0, 0));
            } else {
                phx_a64_str(pb, operand_to_gp(input),
                    phx_ptr(A64_SCRATCH_0, 0));
            }
        } else if (input->type_ == JIT_LIR_OPTYPE_IMM) {
            phx_a64_mov_ri(pb, A64_SCRATCH_1,
                lir_operand_get_constant(input));
            phx_a64_str(pb, A64_SCRATCH_1, phx_ptr(A64_SCRATCH_0, 0));
        } else {
            assert(0 && "Unsupported input type for Move:Mem");
        }
        break;
    }

    case JIT_LIR_OPTYPE_IND: {
        LirMemoryIndirect *ind = lir_operand_get_indirect(output);
        PhxMem ptr = ptr_indirect(pb, A64_SCRATCH_0, A64_SCRATCH_1, ind);
        if (input->type_ == JIT_LIR_OPTYPE_REG) {
            store_from_reg(pb, input, ptr);
        } else if (input->type_ == JIT_LIR_OPTYPE_IMM) {
            int reg_id = A64_SCRATCH_1.id;
            switch (output->data_type_) {
                case JIT_LIR_DT_8BIT:
                    phx_a64_mov_ri(pb, PHX_REG_GP(reg_id, 4),
                        lir_operand_get_constant(input));
                    phx_a64_strb(pb, PHX_REG_GP(reg_id, 4), ptr);
                    break;
                case JIT_LIR_DT_16BIT:
                    phx_a64_mov_ri(pb, PHX_REG_GP(reg_id, 4),
                        lir_operand_get_constant(input));
                    phx_a64_strh(pb, PHX_REG_GP(reg_id, 4), ptr);
                    break;
                default:
                    phx_a64_mov_ri(pb, A64_SCRATCH_1,
                        lir_operand_get_constant(input));
                    phx_a64_str(pb, A64_SCRATCH_1, ptr);
                    break;
            }
        } else {
            assert(0 && "Unsupported input type for Move:Ind");
        }
        break;
    }

    default:
        assert(0 && "Unsupported output type for Move");
    }
}

/* ================================================================
 * TranslateDeoptPatchpoint — cross-architecture
 * ================================================================ */

void
autogen_c_TranslateDeoptPatchpoint(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);

    void *patcher = lir_operand_get_mem_address(instr->inputs_[0]);

#if defined(CINDER_X86_64) && defined(Py_GIL_DISABLED)
    phx_builder_align(pb, 8);
#endif

    PhxLabel patchpoint_label = phx_builder_new_label(pb);
    phx_builder_bind(pb, patchpoint_label);

    const uint8_t *bytes_data;
    size_t bytes_size;
    jit_jump_patcher_stored_bytes(patcher, &bytes_data, &bytes_size);
    phx_builder_embed(pb, bytes_data, bytes_size);

    uint64_t index = lir_operand_get_constant(instr->inputs_[1]);
    void *code_rt = jit_environ_get_code_rt(env);
    jit_fill_live_value_locations(code_rt, index, instr, 2,
                                  instr->num_inputs_);

    PhxLabel deopt_label = phx_builder_new_label(pb);
    jit_environ_add_deopt_exit(env, index, deopt_label, instr);
    jit_environ_add_pending_deopt_patcher(env, patcher,
                                          patchpoint_label, deopt_label);
}

/* ================================================================
 * TranslateGuard — cross-architecture (x86_64 + ARM64)
 * ================================================================ */

void
autogen_c_TranslateGuard(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);

    PhxLabel deopt_label = phx_builder_new_label(pb);
    uint64_t kind = lir_operand_get_constant(instr->inputs_[0]);

#if defined(CINDER_X86_64)
    PhxGp reg = PHX_RAX;
    int is_double = 0;

    if (kind != JIT_GUARD_ALWAYS_FAIL) {
        if (instr->inputs_[2]->data_type_ == JIT_LIR_DT_DOUBLE) {
            assert(kind == JIT_GUARD_NOT_ZERO);
            PhxGp vecd_reg = operand_to_fp(instr->inputs_[2]);
            phx_x86_ptest_rr(pb, vecd_reg, vecd_reg);
            phx_x86_jz(pb, deopt_label);
            is_double = 1;
        } else {
            reg = operand_to_gp(instr->inputs_[2]);
        }
    }

    if (!is_double) {
        switch (kind) {
            case JIT_GUARD_NOT_ZERO:
                phx_x86_test_rr(pb, reg, reg);
                phx_x86_jz(pb, deopt_label);
                break;
            case JIT_GUARD_NOT_NEGATIVE:
                phx_x86_test_rr(pb, reg, reg);
                phx_x86_js(pb, deopt_label);
                break;
            case JIT_GUARD_ZERO:
                phx_x86_test_rr(pb, reg, reg);
                phx_x86_jnz(pb, deopt_label);
                break;
            case JIT_GUARD_ALWAYS_FAIL:
                phx_x86_jmp_label(pb, deopt_label);
                break;
            case JIT_GUARD_IS: {
                const LirOperand *target_opnd = instr->inputs_[3];
                if (target_opnd->type_ == JIT_LIR_OPTYPE_IMM ||
                    target_opnd->type_ == JIT_LIR_OPTYPE_MEM) {
                    uint64_t target = lir_operand_get_constant_or_address(target_opnd);
                    phx_x86_cmp_ri(pb, reg, target);
                } else {
                    phx_x86_cmp_rr(pb, reg, operand_to_gp(target_opnd));
                }
                phx_x86_jne(pb, deopt_label);
                break;
            }
            case JIT_GUARD_HAS_TYPE: {
                const LirOperand *target_opnd = instr->inputs_[3];
                if (target_opnd->type_ == JIT_LIR_OPTYPE_IMM ||
                    target_opnd->type_ == JIT_LIR_OPTYPE_MEM) {
                    uint64_t target = lir_operand_get_constant_or_address(target_opnd);
                    phx_x86_cmp_mi(pb,
                        phx_qword_ptr(reg, offsetof(PyObject, ob_type)),
                        target);
                } else {
                    phx_x86_cmp_mr(pb,
                        phx_qword_ptr(reg, offsetof(PyObject, ob_type)),
                        operand_to_gp(target_opnd));
                }
                phx_x86_jne(pb, deopt_label);
                break;
            }
        }
    }

#elif defined(CINDER_AARCH64)
    PhxGp reg = A64_SCRATCH_0;
    int is_double = 0;
    uint64_t mask = 0;
    size_t sign_bit = 0;

    if (kind != JIT_GUARD_ALWAYS_FAIL) {
        if (instr->inputs_[2]->data_type_ == JIT_LIR_DT_DOUBLE) {
            assert(kind == JIT_GUARD_NOT_ZERO);
            PhxGp vecd_reg = operand_to_vecd(instr->inputs_[2]);
            phx_a64_fmov(pb, reg, vecd_reg);
            phx_a64_cbz(pb, reg, deopt_label);
            is_double = 1;
        } else {
            uint8_t dt = instr->inputs_[2]->data_type_;
            int rloc = lir_operand_get_phy_register(instr->inputs_[2]).loc;
            if (dt == JIT_LIR_DT_8BIT) {
                mask = 0xFF;
                sign_bit = 7;
                reg = PHX_REG_GP(rloc, 4);
            } else if (dt == JIT_LIR_DT_16BIT) {
                mask = 0xFFFF;
                sign_bit = 15;
                reg = PHX_REG_GP(rloc, 4);
            } else {
                reg = operand_to_gp(instr->inputs_[2]);
                sign_bit = (size_t)reg.size * CHAR_BIT - 1;
            }
        }
    }

    if (!is_double) {
        switch (kind) {
            case JIT_GUARD_NOT_ZERO:
                if (mask) {
                    phx_a64_tst_ri(pb, reg, mask);
                    phx_a64_b_eq(pb, deopt_label);
                } else {
                    phx_a64_cbz(pb, reg, deopt_label);
                }
                break;
            case JIT_GUARD_NOT_NEGATIVE:
                phx_a64_tbnz(pb, reg, sign_bit, deopt_label);
                break;
            case JIT_GUARD_ZERO:
                if (mask) {
                    phx_a64_tst_ri(pb, reg, mask);
                    phx_a64_b_ne(pb, deopt_label);
                } else {
                    phx_a64_cbnz(pb, reg, deopt_label);
                }
                break;
            case JIT_GUARD_ALWAYS_FAIL:
                phx_a64_b(pb, deopt_label);
                break;
            case JIT_GUARD_IS: {
                const LirOperand *target_opnd = instr->inputs_[3];
                if (target_opnd->type_ == JIT_LIR_OPTYPE_IMM ||
                    target_opnd->type_ == JIT_LIR_OPTYPE_MEM) {
                    uint64_t target = lir_operand_get_constant_or_address(target_opnd);
                    phx_a64_cmp_ri(pb, reg, target);
                } else {
                    phx_a64_cmp_rr(pb, reg, operand_to_gp(target_opnd));
                }
                phx_a64_b_ne(pb, deopt_label);
                break;
            }
            case JIT_GUARD_HAS_TYPE: {
                PhxMem ob_type_ptr = jit_arch_ptr_offset(
                    reg, offsetof(PyObject, ob_type), 8);
                phx_a64_ldr(pb, A64_SCRATCH_0, ob_type_ptr);

                const LirOperand *target_opnd = instr->inputs_[3];
                if (target_opnd->type_ == JIT_LIR_OPTYPE_IMM ||
                    target_opnd->type_ == JIT_LIR_OPTYPE_MEM) {
                    uint64_t target = lir_operand_get_constant_or_address(target_opnd);
                    phx_a64_cmp_ri(pb, A64_SCRATCH_0, target);
                } else {
                    phx_a64_cmp_rr(pb, A64_SCRATCH_0,
                                   operand_to_gp(target_opnd));
                }
                phx_a64_b_ne(pb, deopt_label);
                break;
            }
        }
    }
#endif

    /* Common: fill live value locations and record deopt exit */
    uint64_t index = lir_operand_get_constant(instr->inputs_[1]);
    void *code_rt = jit_environ_get_code_rt(env);
    jit_fill_live_value_locations(code_rt, index, instr, 4,
                                  instr->num_inputs_);
    jit_environ_add_deopt_exit(env, index, deopt_label, instr);
}

/* ---- Call ---- */

#define A64_SCRATCH_BR  PHX_REG_GP(16, 8)  /* X16 — branch scratch */
#define A64_X0          PHX_REG_GP(0, 8)
#define A64_W0          PHX_REG_GP(0, 4)
#define A64_D0          PHX_REG_FP(0, 8)
#define A64_X29         PHX_REG_GP(29, 8)

void
autogen_c_translateCall(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);

    const LirOperand *output = &instr->output_;
    const LirOperand *input = instr->inputs_[0];

    PhxLabel after_call = phx_builder_new_label(pb);

    int is_gen = jit_environ_is_generator(env);
    int saved_ip_off = jit_environ_saved_ip_fp_offset(env);
    int gen_footer_off = jit_gen_data_footer_saved_ip_offset();

    /* Helper: emit saved-IP store + BLR pattern */
    #define EMIT_SAVE_IP_AND_BLR()                                          \
        phx_a64_adr(pb, A64_SCRATCH_0, after_call);                         \
        if (is_gen) {                                                       \
            phx_a64_str(pb, A64_SCRATCH_0,                                  \
                phx_ptr(A64_X29, gen_footer_off));                          \
        } else {                                                            \
            phx_a64_str(pb, A64_SCRATCH_0,                                  \
                jit_arch_ptr_resolve(pb, A64_FP, saved_ip_off,              \
                                    A64_SCRATCH_1, 8));                     \
        }                                                                   \
        phx_a64_blr(pb, A64_SCRATCH_BR)

    if (input->type_ == JIT_LIR_OPTYPE_REG) {
        PhxGp target = operand_to_gp(input);
        if (target.id != A64_SCRATCH_BR.id) {
            phx_a64_mov_rr(pb, A64_SCRATCH_BR, target);
        }
        EMIT_SAVE_IP_AND_BLR();
    } else if (input->type_ == JIT_LIR_OPTYPE_IMM) {
        phx_a64_mov_ri(pb, A64_SCRATCH_BR, lir_operand_get_constant(input));
        EMIT_SAVE_IP_AND_BLR();
    } else if (input->type_ == JIT_LIR_OPTYPE_STACK) {
        int32_t loc = lir_operand_get_stack_slot(input).loc;
        phx_a64_ldr(pb, A64_SCRATCH_BR,
            jit_arch_ptr_resolve(pb, A64_FP, loc, A64_SCRATCH_0, 8));
        EMIT_SAVE_IP_AND_BLR();
    } else {
        assert(0 && "Unsupported operand type for Call");
    }

    #undef EMIT_SAVE_IP_AND_BLR

    phx_builder_bind(pb, after_call);

    /* Debug location */
    if (instr->origin_) {
        PhxLabel label = phx_builder_new_label(pb);
        phx_builder_bind(pb, label);
        jit_environ_add_pending_debug_loc(env, label, instr->origin_);
    }

    /* Move return value to output register */
    if (output->type_ != JIT_LIR_OPTYPE_NONE) {
        if (lir_operand_is_fp(output)) {
            phx_a64_fmov(pb, operand_to_vecd(output), A64_D0);
        } else {
            PhxGp out_reg = operand_to_gp(output);
            if (out_reg.size <= 4) {
                phx_a64_mov_rr(pb, out_reg, A64_W0);
            } else {
                phx_a64_mov_rr(pb, out_reg, A64_X0);
            }
        }
    }
}

#endif /* CINDER_AARCH64 */
