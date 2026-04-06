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
#include "cinderx/Jit/lir/lir_impl_internal.h"
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
    uint8_t dt = lir_operand_data_type(op);  /* resolve linked operands */
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
    int reg = lir_operand_get_phy_register(op).loc - PHYLOC_VECD_REG_BASE;
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

    assert(lir_operand_type(output) == JIT_LIR_OPTYPE_REG);
    assert(lir_operand_type(opnd0) == JIT_LIR_OPTYPE_REG);

    PhxGp output_reg = operand_to_gp(output);
    PhxGp opnd0_reg = operand_to_gp(opnd0);

    if (lir_operand_type(opnd1) == JIT_LIR_OPTYPE_IMM) {
        uint64_t constant = lir_operand_get_constant(opnd1);
        assert(is_add_sub_imm(constant));
        if (is_sub)
            phx_a64_sub_rri(pb, output_reg, opnd0_reg, constant);
        else
            phx_a64_add_rri(pb, output_reg, opnd0_reg, constant);
    } else if (lir_operand_type(opnd1) == JIT_LIR_OPTYPE_REG) {
        if (is_sub)
            phx_a64_sub_rrr(pb, output_reg, opnd0_reg, operand_to_gp(opnd1));
        else
            phx_a64_add_rrr(pb, output_reg, opnd0_reg, operand_to_gp(opnd1));
    } else if (lir_operand_type(opnd1) == JIT_LIR_OPTYPE_STACK) {
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

    if (lir_operand_type(opnd) == JIT_LIR_OPTYPE_REG) {
        PhxGp reg = operand_to_gp(opnd);
        if (is_dec)
            phx_a64_sub_rri(pb, reg, reg, 1);
        else
            phx_a64_add_rri(pb, reg, reg, 1);
    } else if (lir_operand_type(opnd) == JIT_LIR_OPTYPE_STACK) {
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

    assert(lir_operand_type(output) == JIT_LIR_OPTYPE_REG);
    assert(lir_operand_type(opnd0) == JIT_LIR_OPTYPE_REG);

    PhxGp out_reg = operand_to_gp(output);
    PhxGp r0 = operand_to_gp(opnd0);

    if (lir_operand_type(opnd1) == JIT_LIR_OPTYPE_IMM) {
        uint64_t c = lir_operand_get_constant(opnd1);
        switch (op) {
            case LOGICAL_AND: phx_a64_and_rri(pb, out_reg, r0, c); break;
            case LOGICAL_ORR: phx_a64_orr_rri(pb, out_reg, r0, c); break;
            case LOGICAL_EOR: phx_a64_eor_rri(pb, out_reg, r0, c); break;
        }
    } else if (lir_operand_type(opnd1) == JIT_LIR_OPTYPE_REG) {
        PhxGp r1 = operand_to_gp(opnd1);
        switch (op) {
            case LOGICAL_AND: phx_a64_and_rrr(pb, out_reg, r0, r1); break;
            case LOGICAL_ORR: phx_a64_orr_rrr(pb, out_reg, r0, r1); break;
            case LOGICAL_EOR: phx_a64_eor_rrr(pb, out_reg, r0, r1); break;
        }
    } else if (lir_operand_type(opnd1) == JIT_LIR_OPTYPE_STACK) {
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

    assert(lir_operand_type(output) == JIT_LIR_OPTYPE_REG);
    assert(lir_operand_type(opnd0) == JIT_LIR_OPTYPE_REG);

    PhxGp out_reg = operand_to_gp(output);
    PhxGp r0 = operand_to_gp(opnd0);

    if (lir_operand_type(opnd1) == JIT_LIR_OPTYPE_IMM) {
        phx_a64_mov_ri(pb, A64_SCRATCH_0, lir_operand_get_constant(opnd1));
        phx_a64_mul(pb, out_reg, r0, A64_SCRATCH_0);
    } else if (lir_operand_type(opnd1) == JIT_LIR_OPTYPE_REG) {
        phx_a64_mul(pb, out_reg, r0, operand_to_gp(opnd1));
    } else if (lir_operand_type(opnd1) == JIT_LIR_OPTYPE_STACK) {
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

    assert(lir_operand_type(output) == JIT_LIR_OPTYPE_REG);
    assert(lir_operand_type(opnd0) == JIT_LIR_OPTYPE_REG);

    PhxGp out_reg = operand_to_gp(output);
    PhxGp r0 = operand_to_gp(opnd0);

    PhxGp divisor;
    if (lir_operand_type(opnd1) == JIT_LIR_OPTYPE_REG) {
        divisor = operand_to_gp(opnd1);
    } else if (lir_operand_type(opnd1) == JIT_LIR_OPTYPE_STACK) {
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

    if (lir_operand_type(operand) == JIT_LIR_OPTYPE_IMM) {
        phx_a64_mov_ri(pb, A64_SCRATCH_0, lir_operand_get_constant(operand));
        phx_a64_str(pb, A64_SCRATCH_0,
            jit_arch_ptr_offset(A64_SP, -16, 8)); /* pre-index */
    } else if (lir_operand_type(operand) == JIT_LIR_OPTYPE_REG) {
        phx_a64_str(pb, operand_to_gp(operand),
            jit_arch_ptr_offset(A64_SP, -16, 8));
    } else if (lir_operand_type(operand) == JIT_LIR_OPTYPE_STACK) {
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

    if (lir_operand_type(operand) == JIT_LIR_OPTYPE_REG) {
        PhxMem post = jit_arch_ptr_offset(A64_SP, 16, 8); /* post-index */
        phx_a64_ldr(pb, operand_to_gp(operand), post);
    } else if (lir_operand_type(operand) == JIT_LIR_OPTYPE_STACK) {
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

    assert(lir_operand_type(opnd0) == JIT_LIR_OPTYPE_REG);
    assert(lir_operand_type(opnd1) == JIT_LIR_OPTYPE_REG);

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

    assert(lir_operand_type(inp0) == JIT_LIR_OPTYPE_REG);

    if (lir_operand_type(inp1) == JIT_LIR_OPTYPE_REG) {
        if (lir_operand_is_fp(inp0) && lir_operand_is_fp(inp1)) {
            phx_a64_fcmp(pb, operand_to_vecd(inp0), operand_to_vecd(inp1));
        } else {
            phx_a64_cmp_rr(pb, operand_to_gp(inp0), operand_to_gp(inp1));
        }
    } else if (lir_operand_type(inp1) == JIT_LIR_OPTYPE_IMM) {
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

    if (lir_operand_type(input) == JIT_LIR_OPTYPE_REG) {
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
    } else if (lir_operand_type(input) == JIT_LIR_OPTYPE_STACK) {
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

    if (lir_operand_type(input) == JIT_LIR_OPTYPE_REG) {
        phx_a64_sxtw(pb, output, operand_to_gp(input));
    } else if (lir_operand_type(input) == JIT_LIR_OPTYPE_STACK) {
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

/* ---- FP arithmetic (FADD, FSUB, FMUL, FDIV) ---- */

void
autogen_c_translateFadd(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    const LirOperand *output =
        (instr->output_.type_ != JIT_LIR_OPTYPE_NONE)
        ? &instr->output_ : instr->inputs_[0];
    PhxGp out = operand_to_vecd(output);
    PhxGp r0 = operand_to_vecd(instr->inputs_[0]);
    PhxGp r1 = operand_to_vecd(instr->inputs_[1]);
    phx_a64_fadd(pb, out, r0, r1);
}

void
autogen_c_translateFsub(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    const LirOperand *output =
        (instr->output_.type_ != JIT_LIR_OPTYPE_NONE)
        ? &instr->output_ : instr->inputs_[0];
    PhxGp out = operand_to_vecd(output);
    PhxGp r0 = operand_to_vecd(instr->inputs_[0]);
    PhxGp r1 = operand_to_vecd(instr->inputs_[1]);
    phx_a64_fsub(pb, out, r0, r1);
}

void
autogen_c_translateFmul(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    const LirOperand *output =
        (instr->output_.type_ != JIT_LIR_OPTYPE_NONE)
        ? &instr->output_ : instr->inputs_[0];
    PhxGp out = operand_to_vecd(output);
    PhxGp r0 = operand_to_vecd(instr->inputs_[0]);
    PhxGp r1 = operand_to_vecd(instr->inputs_[1]);
    phx_a64_fmul(pb, out, r0, r1);
}

void
autogen_c_translateFdiv(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    const LirOperand *output =
        (instr->output_.type_ != JIT_LIR_OPTYPE_NONE)
        ? &instr->output_ : instr->inputs_[0];
    PhxGp out = operand_to_vecd(output);
    PhxGp r0 = operand_to_vecd(instr->inputs_[0]);
    PhxGp r1 = operand_to_vecd(instr->inputs_[1]);
    phx_a64_fdiv(pb, out, r0, r1);
}

/* ---- Negate/Invert ---- */

void
autogen_c_translateNegate(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    const LirOperand *output =
        (instr->output_.type_ != JIT_LIR_OPTYPE_NONE)
        ? &instr->output_ : instr->inputs_[0];
    PhxGp out = operand_to_gp(output);
    PhxGp r0 = operand_to_gp(instr->inputs_[0]);
    /* ARM64 neg is SUB from zero register */
    PhxGp xzr = PHX_REG_GP(31, out.size);
    phx_a64_sub_rrr(pb, out, xzr, r0);
}

void
autogen_c_translateInvert(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    const LirOperand *output =
        (instr->output_.type_ != JIT_LIR_OPTYPE_NONE)
        ? &instr->output_ : instr->inputs_[0];
    PhxGp out = operand_to_gp(output);
    PhxGp r0 = operand_to_gp(instr->inputs_[0]);
    phx_a64_mvn(pb, out, r0);
}

/* ---- Branch opcodes ---- */

static int
opcode_to_arm64_cond(int opcode) {
    switch (opcode) {
        case JIT_LIR_OP_BRANCHNZ: return PHX_COND_NE;
        case JIT_LIR_OP_BRANCHZ:  return PHX_COND_EQ;
        case JIT_LIR_OP_BRANCHA:  return PHX_COND_HI;
        case JIT_LIR_OP_BRANCHB:  return PHX_COND_LO;
        case JIT_LIR_OP_BRANCHAE: return PHX_COND_HS;
        case JIT_LIR_OP_BRANCHBE: return PHX_COND_LS;
        case JIT_LIR_OP_BRANCHG:  return PHX_COND_GT;
        case JIT_LIR_OP_BRANCHL:  return PHX_COND_LT;
        case JIT_LIR_OP_BRANCHGE: return PHX_COND_GE;
        case JIT_LIR_OP_BRANCHLE: return PHX_COND_LE;
        case JIT_LIR_OP_BRANCHE:  return PHX_COND_EQ;
        case JIT_LIR_OP_BRANCHNE: return PHX_COND_NE;
        case JIT_LIR_OP_BRANCHC:  return PHX_COND_CS;
        case JIT_LIR_OP_BRANCHNC: return PHX_COND_CC;
        case JIT_LIR_OP_BRANCHO:  return PHX_COND_VS;
        case JIT_LIR_OP_BRANCHNO: return PHX_COND_VC;
        case JIT_LIR_OP_BRANCHS:  return PHX_COND_MI;
        case JIT_LIR_OP_BRANCHNS: return PHX_COND_PL;
        default: return -1;
    }
}

void
autogen_c_translateBranch(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    LirBasicBlock *target = (LirBasicBlock *)lir_operand_get_basic_block(
        instr->inputs_[0]);
    PhxLabel label = jit_environ_get_block_label(env, target);
    phx_a64_b(pb, label);
}

void
autogen_c_translateCondBranch(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    int cond = opcode_to_arm64_cond(instr->opcode_);
    LirBasicBlock *target = (LirBasicBlock *)lir_operand_get_basic_block(
        instr->inputs_[0]);
    PhxLabel label = jit_environ_get_block_label(env, target);
    phx_a64_b_cond(pb, cond, label);
}

/* ---- IntToBool ---- */

void
autogen_c_translateIntToBool(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    const LirOperand *input = instr->inputs_[0];
    PhxGp output = operand_to_gp_output(&instr->output_);

    assert(instr->output_.data_type_ == JIT_LIR_DT_8BIT);

    if (lir_operand_type(input) == JIT_LIR_OPTYPE_IMM) {
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

    assert(lir_operand_type(output) == JIT_LIR_OPTYPE_REG);

    PhxGp out_reg = operand_to_gp(output);

    if (lir_operand_type(input) == JIT_LIR_OPTYPE_STACK) {
        int32_t loc = lir_operand_get_stack_slot(input).loc;
        if (loc >= 0) {
            phx_a64_add_rri(pb, out_reg, A64_FP, loc);
        } else {
            phx_a64_sub_rri(pb, out_reg, A64_FP, -loc);
        }
    } else if (lir_operand_type(input) == JIT_LIR_OPTYPE_MEM) {
        uint64_t address = (uint64_t)(uintptr_t)lir_operand_get_mem_address(input);
        phx_a64_mov_ri(pb, out_reg, address);
    } else if (lir_operand_type(input) == JIT_LIR_OPTYPE_IND) {
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

    switch (lir_operand_type(output)) {
    case JIT_LIR_OPTYPE_REG:
        switch (lir_operand_type(input)) {
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
        if (lir_operand_type(input) == JIT_LIR_OPTYPE_REG) {
            store_from_reg(pb, input, ptr);
        } else if (lir_operand_type(input) == JIT_LIR_OPTYPE_IMM) {
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
        if (lir_operand_type(input) == JIT_LIR_OPTYPE_REG) {
            if (lir_operand_is_fp(input)) {
                phx_a64_str_fp(pb, operand_to_vecd(input),
                    phx_ptr(A64_SCRATCH_0, 0));
            } else {
                phx_a64_str(pb, operand_to_gp(input),
                    phx_ptr(A64_SCRATCH_0, 0));
            }
        } else if (lir_operand_type(input) == JIT_LIR_OPTYPE_IMM) {
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
        if (lir_operand_type(input) == JIT_LIR_OPTYPE_REG) {
            store_from_reg(pb, input, ptr);
        } else if (lir_operand_type(input) == JIT_LIR_OPTYPE_IMM) {
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

#endif /* CINDER_AARCH64 — ARM64-only translate* functions end here */

/* ================================================================
 * Cross-architecture translate* functions (compiled on both arches)
 * ================================================================ */

/* ================================================================
 * TranslateCompare — cross-architecture
 * ================================================================ */

void
autogen_c_TranslateCompare(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);

    const LirOperand *inp0 = instr->inputs_[0];
    const LirOperand *inp1 = instr->inputs_[1];

#if defined(CINDER_X86_64)
    if (lir_operand_type(inp1) == JIT_LIR_OPTYPE_IMM ||
        lir_operand_type(inp1) == JIT_LIR_OPTYPE_MEM) {
        phx_x86_cmp_ri(pb, operand_to_gp(inp0),
                        lir_operand_get_constant_or_address(inp1));
    } else if (!lir_operand_is_fp(inp1)) {
        phx_x86_cmp_rr(pb, operand_to_gp(inp0), operand_to_gp(inp1));
    } else {
        phx_x86_comisd(pb, operand_to_fp(inp0), operand_to_fp(inp1));
    }

    PhxGp output = operand_to_gp(&instr->output_);
    switch (instr->opcode_) {
        case JIT_LIR_OP_EQUAL:                  phx_x86_sete(pb, output); break;
        case JIT_LIR_OP_NOTEQUAL:               phx_x86_setne(pb, output); break;
        case JIT_LIR_OP_GREATERTHANSIGNED:       phx_x86_setg(pb, output); break;
        case JIT_LIR_OP_GREATERTHANEQUALSIGNED:  phx_x86_setge(pb, output); break;
        case JIT_LIR_OP_LESSTHANSIGNED:          phx_x86_setl(pb, output); break;
        case JIT_LIR_OP_LESSTHANEQUALSIGNED:     phx_x86_setle(pb, output); break;
        case JIT_LIR_OP_GREATERTHANUNSIGNED:     phx_x86_seta(pb, output); break;
        case JIT_LIR_OP_GREATERTHANEQUALUNSIGNED:phx_x86_setae(pb, output); break;
        case JIT_LIR_OP_LESSTHANUNSIGNED:        phx_x86_setb(pb, output); break;
        case JIT_LIR_OP_LESSTHANEQUALUNSIGNED:   phx_x86_setbe(pb, output); break;
        default: assert(0 && "bad instruction for TranslateCompare");
    }

    if (instr->output_.data_type_ != JIT_LIR_DT_8BIT) {
        int reg_id = lir_operand_get_phy_register(&instr->output_).loc;
        PhxGp full = operand_to_gp(&instr->output_);
        PhxGp byte_reg = {(uint8_t)reg_id, 1};
        phx_x86_movzx_rr(pb, full, byte_reg);
    }

#elif defined(CINDER_AARCH64)
    if (lir_operand_type(inp1) == JIT_LIR_OPTYPE_MEM) {
        uint64_t address = lir_operand_get_constant_or_address(inp1);
        phx_a64_mov_ri(pb, A64_SCRATCH_0, address);
        phx_a64_ldr(pb, A64_SCRATCH_0, phx_ptr(A64_SCRATCH_0, 0));
        phx_a64_cmp_rr(pb, operand_to_gp(inp0), A64_SCRATCH_0);
    } else if (lir_operand_type(inp1) == JIT_LIR_OPTYPE_IMM) {
        uint64_t constant = lir_operand_get_constant_or_address(inp1);
        if (is_add_sub_imm(constant)) {
            phx_a64_cmp_ri(pb, operand_to_gp(inp0), constant);
        } else {
            phx_a64_mov_ri(pb, A64_SCRATCH_0, constant);
            phx_a64_cmp_rr(pb, operand_to_gp(inp0), A64_SCRATCH_0);
        }
    } else if (!lir_operand_is_fp(inp1)) {
        phx_a64_cmp_rr(pb, operand_to_gp(inp0), operand_to_gp(inp1));
    } else {
        phx_a64_fcmp(pb, operand_to_vecd(inp0), operand_to_vecd(inp1));
    }

    PhxGp output = operand_to_gp_output(&instr->output_);
    switch (instr->opcode_) {
        case JIT_LIR_OP_EQUAL:                  phx_a64_cset(pb, output, PHX_COND_EQ); break;
        case JIT_LIR_OP_NOTEQUAL:               phx_a64_cset(pb, output, PHX_COND_NE); break;
        case JIT_LIR_OP_GREATERTHANSIGNED:       phx_a64_cset(pb, output, PHX_COND_GT); break;
        case JIT_LIR_OP_GREATERTHANEQUALSIGNED:  phx_a64_cset(pb, output, PHX_COND_GE); break;
        case JIT_LIR_OP_LESSTHANSIGNED:          phx_a64_cset(pb, output, PHX_COND_LT); break;
        case JIT_LIR_OP_LESSTHANEQUALSIGNED:     phx_a64_cset(pb, output, PHX_COND_LE); break;
        case JIT_LIR_OP_GREATERTHANUNSIGNED:     phx_a64_cset(pb, output, PHX_COND_HI); break;
        case JIT_LIR_OP_GREATERTHANEQUALUNSIGNED:phx_a64_cset(pb, output, PHX_COND_HS); break;
        case JIT_LIR_OP_LESSTHANUNSIGNED:        phx_a64_cset(pb, output, PHX_COND_LO); break;
        case JIT_LIR_OP_LESSTHANEQUALUNSIGNED:   phx_a64_cset(pb, output, PHX_COND_LS); break;
        default: assert(0 && "bad instruction for TranslateCompare");
    }
#endif
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
                if (lir_operand_type(target_opnd) == JIT_LIR_OPTYPE_IMM ||
                    lir_operand_type(target_opnd) == JIT_LIR_OPTYPE_MEM) {
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
                if (lir_operand_type(target_opnd) == JIT_LIR_OPTYPE_IMM ||
                    lir_operand_type(target_opnd) == JIT_LIR_OPTYPE_MEM) {
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
                if (lir_operand_type(target_opnd) == JIT_LIR_OPTYPE_IMM ||
                    lir_operand_type(target_opnd) == JIT_LIR_OPTYPE_MEM) {
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
                if (lir_operand_type(target_opnd) == JIT_LIR_OPTYPE_IMM ||
                    lir_operand_type(target_opnd) == JIT_LIR_OPTYPE_MEM) {
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

#if defined(CINDER_AARCH64)
/* ---- Call (ARM64 only) ---- */

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

    if (lir_operand_type(input) == JIT_LIR_OPTYPE_REG) {
        PhxGp target = operand_to_gp(input);
        if (target.id != A64_SCRATCH_BR.id) {
            phx_a64_mov_rr(pb, A64_SCRATCH_BR, target);
        }
        EMIT_SAVE_IP_AND_BLR();
    } else if (lir_operand_type(input) == JIT_LIR_OPTYPE_IMM) {
        phx_a64_mov_ri(pb, A64_SCRATCH_BR, lir_operand_get_constant(input));
        EMIT_SAVE_IP_AND_BLR();
    } else if (lir_operand_type(input) == JIT_LIR_OPTYPE_STACK) {
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
    if (lir_operand_type(output) != JIT_LIR_OPTYPE_NONE) {
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

/* ================================================================
 * C dispatch table — replaces C++ trie/DSL for converted opcodes.
 * Returns 1 if handled, 0 to fall through to C++ AutoTranslator.
 *
 * ARM64: handles ALL non-yield opcodes (full C codegen path).
 * x86_64: handles cross-arch CALL_C opcodes (Guard, Compare,
 *          DeoptPatchpoint, IntToBool). ASM() opcodes stay in trie.
 * ================================================================ */

int
autogen_c_dispatch(void *env, const LirInstruction *instr) {
    switch (instr->opcode_) {
    case JIT_LIR_OP_BIND:
        return 1; /* no-op */

    /* Cross-arch opcodes — handled on both x86_64 and ARM64 */
    case JIT_LIR_OP_GUARD:
        autogen_c_TranslateGuard(env, instr);
        return 1;
    case JIT_LIR_OP_DEOPTPATCHPOINT:
        autogen_c_TranslateDeoptPatchpoint(env, instr);
        return 1;
    case JIT_LIR_OP_EQUAL:
    case JIT_LIR_OP_NOTEQUAL:
    case JIT_LIR_OP_GREATERTHANSIGNED:
    case JIT_LIR_OP_LESSTHANSIGNED:
    case JIT_LIR_OP_GREATERTHANEQUALSIGNED:
    case JIT_LIR_OP_LESSTHANEQUALSIGNED:
    case JIT_LIR_OP_GREATERTHANUNSIGNED:
    case JIT_LIR_OP_LESSTHANUNSIGNED:
    case JIT_LIR_OP_GREATERTHANEQUALUNSIGNED:
    case JIT_LIR_OP_LESSTHANEQUALUNSIGNED:
        autogen_c_TranslateCompare(env, instr);
        return 1;
    case JIT_LIR_OP_UNREACHABLE:
        autogen_c_translateUnreachable(env, instr);
        return 1;
    case JIT_LIR_OP_ADD:
        autogen_c_translateAdd(env, instr);
        return 1;
    case JIT_LIR_OP_SUB:
        autogen_c_translateSub(env, instr);
        return 1;
    case JIT_LIR_OP_AND:
        autogen_c_translateAnd(env, instr);
        return 1;
    case JIT_LIR_OP_OR:
        autogen_c_translateOr(env, instr);
        return 1;
    case JIT_LIR_OP_XOR:
        autogen_c_translateXor(env, instr);
        return 1;
    case JIT_LIR_OP_MUL:
        autogen_c_translateMul(env, instr);
        return 1;
    case JIT_LIR_OP_DIV:
        autogen_c_translateDiv(env, instr);
        return 1;
    case JIT_LIR_OP_DIVUN:
        autogen_c_translateDivUn(env, instr);
        return 1;
    case JIT_LIR_OP_FADD:
        autogen_c_translateFadd(env, instr);
        return 1;
    case JIT_LIR_OP_FSUB:
        autogen_c_translateFsub(env, instr);
        return 1;
    case JIT_LIR_OP_FMUL:
        autogen_c_translateFmul(env, instr);
        return 1;
    case JIT_LIR_OP_FDIV:
        autogen_c_translateFdiv(env, instr);
        return 1;
    case JIT_LIR_OP_NEGATE:
        autogen_c_translateNegate(env, instr);
        return 1;
    case JIT_LIR_OP_INVERT:
        autogen_c_translateInvert(env, instr);
        return 1;
    case JIT_LIR_OP_BRANCH:
        autogen_c_translateBranch(env, instr);
        return 1;
    case JIT_LIR_OP_BRANCHNZ:
    case JIT_LIR_OP_BRANCHZ:
    case JIT_LIR_OP_BRANCHA:
    case JIT_LIR_OP_BRANCHB:
    case JIT_LIR_OP_BRANCHAE:
    case JIT_LIR_OP_BRANCHBE:
    case JIT_LIR_OP_BRANCHG:
    case JIT_LIR_OP_BRANCHL:
    case JIT_LIR_OP_BRANCHGE:
    case JIT_LIR_OP_BRANCHLE:
    case JIT_LIR_OP_BRANCHC:
    case JIT_LIR_OP_BRANCHNC:
    case JIT_LIR_OP_BRANCHO:
    case JIT_LIR_OP_BRANCHNO:
    case JIT_LIR_OP_BRANCHS:
    case JIT_LIR_OP_BRANCHNS:
    case JIT_LIR_OP_BRANCHE:
    case JIT_LIR_OP_BRANCHNE:
        autogen_c_translateCondBranch(env, instr);
        return 1;
    case JIT_LIR_OP_LSHIFT:
    case JIT_LIR_OP_RSHIFT:
    case JIT_LIR_OP_RSHIFTUN:
    case JIT_LIR_OP_NOP:
    case JIT_LIR_OP_LOADARG:
    case JIT_LIR_OP_LOADSECONDCALLRESULT:
    case JIT_LIR_OP_RETURN:
        /* These opcodes should not appear in autogen — they're handled
         * earlier in the pipeline or are no-ops. Return 0 to let C++
         * trie handle them (which will also fail — they have no rules). */
        return 0;
    case JIT_LIR_OP_INC:
        autogen_c_translateInc(env, instr);
        return 1;
    case JIT_LIR_OP_DEC:
        autogen_c_translateDec(env, instr);
        return 1;
    case JIT_LIR_OP_PUSH:
        autogen_c_translatePush(env, instr);
        return 1;
    case JIT_LIR_OP_POP:
        autogen_c_translatePop(env, instr);
        return 1;
    case JIT_LIR_OP_EXCHANGE:
        autogen_c_translateExchange(env, instr);
        return 1;
    case JIT_LIR_OP_CMP:
        autogen_c_translateCmp(env, instr);
        return 1;
    case JIT_LIR_OP_BITTEST:
        autogen_c_translateBitTest(env, instr);
        return 1;
    case JIT_LIR_OP_TEST:
    case JIT_LIR_OP_TEST32:
        autogen_c_translateTst(env, instr);
        return 1;
    case JIT_LIR_OP_MOVZX:
        autogen_c_translateMovZX(env, instr);
        return 1;
    case JIT_LIR_OP_MOVSX:
        autogen_c_translateMovSX(env, instr);
        return 1;
    case JIT_LIR_OP_MOVSXD:
        autogen_c_translateMovSXD(env, instr);
        return 1;
    case JIT_LIR_OP_INTTOBOOL:
        autogen_c_translateIntToBool(env, instr);
        return 1;
    case JIT_LIR_OP_SELECT:
        autogen_c_translateSelect(env, instr);
        return 1;
    case JIT_LIR_OP_LEA:
        autogen_c_translateLea(env, instr);
        return 1;
    case JIT_LIR_OP_CALL:
        autogen_c_translateCall(env, instr);
        return 1;
    case JIT_LIR_OP_MOVE:
    case JIT_LIR_OP_MOVERELAXED:
        autogen_c_translateMove(env, instr);
        return 1;
    default:
        return 0; /* Yield*, Nop, etc. — fall through to C++ */
    }
}

#endif /* CINDER_AARCH64 */

/* ================================================================
 * x86_64 translate functions — C implementations replacing the
 * C++ trie/DSL-based autogen for all non-Yield opcodes.
 * ================================================================ */

#if defined(CINDER_X86_64)

/* ---- x86_64 helper: LIR operand → PhxMem ---- */
static PhxMem
x86_operand_to_mem(const LirOperand *op) {
    PhxMem m = {0};
    uint32_t size_bits = lir_operand_size_in_bits(op);
    m.size = (uint8_t)(size_bits / 8);
    if (m.size == 0) m.size = 8;

    if (lir_operand_type(op) == JIT_LIR_OPTYPE_STACK) {
        PhxGp rbp = {5, 8};
        m = phx_ptr(rbp, lir_operand_get_stack_slot(op).loc);
        m.size = (uint8_t)(size_bits / 8);
        if (m.size == 0) m.size = 8;
    } else if (lir_operand_type(op) == JIT_LIR_OPTYPE_MEM) {
        /* Absolute address — use is_abs_addr flag for SIB disp32 encoding */
        uint64_t addr = (uint64_t)(uintptr_t)lir_operand_get_mem_address(op);
        m.is_abs_addr = 1;
        m.abs_addr = addr;
        m.offset = (int32_t)addr;
        m.size = (uint8_t)(size_bits / 8);
        if (m.size == 0) m.size = 8;
    } else if (lir_operand_type(op) == JIT_LIR_OPTYPE_IND) {
        LirMemoryIndirect *ind = lir_operand_get_indirect(op);
        LirOperand *base_op = lir_memind_base_reg(ind);
        LirOperand *idx_op = lir_memind_index_reg(ind);
        int32_t offset = lir_memind_offset(ind);
        PhxGp base = {(uint8_t)lir_operand_get_phy_register(base_op).loc, 8};
        if (idx_op == NULL || lir_operand_type(idx_op) == JIT_LIR_OPTYPE_NONE) {
            m = phx_ptr(base, offset);
        } else {
            PhxGp idx = {(uint8_t)lir_operand_get_phy_register(idx_op).loc, 8};
            m = phx_ptr_index(base, idx, lir_memind_multiplier(ind), offset);
        }
        m.size = (uint8_t)(size_bits / 8);
        if (m.size == 0) m.size = 8;
    } else {
        assert(0 && "bad operand type for x86 memory");
    }
    return m;
}

/* ---- x86_64 Move / MoveRelaxed ---- */

/* ---- x86_64 Move / MoveRelaxed ----
 * Move uses kOut sizing: ALL operands use the OUTPUT operand's size.
 * This matches C++ LIROperandSizeMapper behavior. */

/* Convert operand to GP register using output's data type (kOut sizing) */
static inline PhxGp
operand_to_gp_kout(const LirOperand *op, uint8_t out_data_type) {
    int reg = lir_operand_get_phy_register(op).loc;
    PhxGp base = {(uint8_t)reg, 8};
    switch (out_data_type) {
        case JIT_LIR_DT_8BIT:   return phx_gp8(base);
        case JIT_LIR_DT_16BIT:  return phx_gp16(base);
        case JIT_LIR_DT_32BIT:  return phx_gp32(base);
        case JIT_LIR_DT_OBJECT:
        case JIT_LIR_DT_64BIT:  return base;
        default: return base;
    }
}

/* Check if operand is a VecD (FP) register by physical location.
 * Matches C++ isVecD() which checks getPhyRegister().loc >= VECD_REG_BASE.
 * This differs from lir_operand_is_fp() which checks data_type_ == DOUBLE. */
static inline int
x86_is_vecd(const LirOperand *op) {
    if (lir_operand_type(op) != JIT_LIR_OPTYPE_REG) return 0;
    return lir_operand_get_phy_register(op).loc >= PHYLOC_VECD_REG_BASE;
}

static void
x86_translateMove(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    const LirOperand *output = &instr->output_;
    const LirOperand *input = instr->inputs_[0];

    /* DEBUG: log Move patterns to find 5th bug */
    if (getenv("JIT_MOVE_DEBUG")) {
        uint8_t ot = lir_operand_type(output);
        uint8_t it = lir_operand_type(input);
        uint8_t odt = lir_operand_data_type(output);
        uint8_t idt = lir_operand_data_type(input);
        fprintf(stderr, "MOVE: out_type=%d in_type=%d out_dt=%d in_dt=%d "
                "out_fp=%d in_fp=%d\n",
                ot, it, odt, idt,
                x86_is_vecd(output), x86_is_vecd(input));
    }
    int out_fp = x86_is_vecd(output);
    int in_fp = x86_is_vecd(input);
    uint8_t out_dt = lir_operand_data_type(output);
    uint32_t out_bits = lir_operand_size_in_bits(output);
    uint8_t out_bytes = (uint8_t)(out_bits / 8);
    if (out_bytes == 0) out_bytes = 8;
    uint8_t in_type = lir_operand_type(input);
    uint8_t out_type = lir_operand_type(output);

    switch (out_type) {
    case JIT_LIR_OPTYPE_REG:
        switch (in_type) {
        case JIT_LIR_OPTYPE_REG:
            if (out_fp && in_fp)       phx_x86_movsd_rr(pb, operand_to_fp(output), operand_to_fp(input));
            else if (out_fp && !in_fp) phx_x86_movq_rr(pb, operand_to_fp(output), operand_to_gp_kout(input, out_dt));
            else if (!out_fp && in_fp) phx_x86_movq_rr(pb, operand_to_gp(output), operand_to_fp(input));
            else                       phx_x86_mov_rr(pb, operand_to_gp(output), operand_to_gp_kout(input, out_dt));
            break;
        case JIT_LIR_OPTYPE_IMM:
            phx_x86_mov_ri(pb, operand_to_gp(output),
                (int64_t)lir_operand_get_constant(input));
            break;
        case JIT_LIR_OPTYPE_STACK:
        case JIT_LIR_OPTYPE_MEM:
        case JIT_LIR_OPTYPE_IND: {
            PhxMem mem = x86_operand_to_mem(input);
            mem.size = out_bytes;
            if (out_fp) phx_x86_movsd_rm(pb, operand_to_fp(output), mem);
            else        phx_x86_mov_rm(pb, operand_to_gp(output), mem);
            break;
        }
        default: assert(0 && "bad Move input");
        }
        break;
    case JIT_LIR_OPTYPE_STACK:
    case JIT_LIR_OPTYPE_MEM:
    case JIT_LIR_OPTYPE_IND: {
        PhxMem mem = x86_operand_to_mem(output);
        mem.size = out_bytes;
        if (in_type == JIT_LIR_OPTYPE_REG) {
            if (in_fp) phx_x86_movsd_mr(pb, mem, operand_to_fp(input));
            else       phx_x86_mov_mr(pb, mem, operand_to_gp_kout(input, out_dt));
        } else {
            phx_x86_mov_mi(pb, mem, (int32_t)lir_operand_get_constant(input));
        }
        break;
    }
    default: assert(0 && "bad Move output");
    }
}

/* ---- x86_64 binary op helper ---- */
typedef void (*x86_binop_rr_fn)(PhxBuilder *, PhxGp, PhxGp);
typedef void (*x86_binop_ri_fn)(PhxBuilder *, PhxGp, int32_t);
typedef void (*x86_binop_rm_fn)(PhxBuilder *, PhxGp, PhxMem);

static void
x86_translateBinaryOp(void *env, const LirInstruction *instr,
                      x86_binop_rr_fn rr, x86_binop_ri_fn ri, x86_binop_rm_fn rm) {
    PhxBuilder *pb = get_builder(env);
    const LirOperand *output = &instr->output_;
    if (lir_operand_type(output) == JIT_LIR_OPTYPE_NONE) {
        /* 2-operand: dst = input[0], src = input[last] */
        PhxGp dst = operand_to_gp(instr->inputs_[0]);
        const LirOperand *src = instr->inputs_[instr->num_inputs_ - 1];
        if (lir_operand_type(src) == JIT_LIR_OPTYPE_IMM) ri(pb, dst, (int32_t)lir_operand_get_constant(src));
        else if (lir_operand_type(src) == JIT_LIR_OPTYPE_REG) rr(pb, dst, operand_to_gp(src));
        else rm(pb, dst, x86_operand_to_mem(src));
    } else {
        /* 3-operand: output != input[0] */
        PhxGp dst = operand_to_gp(output);
        phx_x86_mov_rr(pb, dst, operand_to_gp(instr->inputs_[0]));
        const LirOperand *src = instr->inputs_[1];
        if (lir_operand_type(src) == JIT_LIR_OPTYPE_IMM) ri(pb, dst, (int32_t)lir_operand_get_constant(src));
        else if (lir_operand_type(src) == JIT_LIR_OPTYPE_REG) rr(pb, dst, operand_to_gp(src));
        else rm(pb, dst, x86_operand_to_mem(src));
    }
}

static void x86_translateAdd(void *e, const LirInstruction *i) { x86_translateBinaryOp(e, i, phx_x86_add_rr, phx_x86_add_ri, phx_x86_add_rm); }
static void x86_translateSub(void *e, const LirInstruction *i) { x86_translateBinaryOp(e, i, phx_x86_sub_rr, phx_x86_sub_ri, phx_x86_sub_rm); }
static void x86_translateAnd(void *e, const LirInstruction *i) { x86_translateBinaryOp(e, i, phx_x86_and_rr, phx_x86_and_ri, phx_x86_and_rm); }
static void x86_translateOr(void *e, const LirInstruction *i)  { x86_translateBinaryOp(e, i, phx_x86_or_rr, phx_x86_or_ri, phx_x86_or_rm); }
static void x86_translateXor(void *e, const LirInstruction *i) { x86_translateBinaryOp(e, i, phx_x86_xor_rr, phx_x86_xor_ri, phx_x86_xor_rm); }

static void x86_translateMul(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    const LirOperand *output = &instr->output_;
    if (lir_operand_type(output) == JIT_LIR_OPTYPE_NONE) {
        PhxGp dst = operand_to_gp(instr->inputs_[0]);
        const LirOperand *src = instr->inputs_[instr->num_inputs_ - 1];
        if (lir_operand_type(src) == JIT_LIR_OPTYPE_IMM) phx_x86_imul_rri(pb, dst, dst, (int32_t)lir_operand_get_constant(src));
        else if (lir_operand_type(src) == JIT_LIR_OPTYPE_REG) phx_x86_imul_rr(pb, dst, operand_to_gp(src));
        else phx_x86_imul_rm(pb, dst, x86_operand_to_mem(src));
    } else {
        PhxGp dst = operand_to_gp(output);
        phx_x86_mov_rr(pb, dst, operand_to_gp(instr->inputs_[0]));
        const LirOperand *src = instr->inputs_[1];
        if (lir_operand_type(src) == JIT_LIR_OPTYPE_IMM) phx_x86_imul_rri(pb, dst, dst, (int32_t)lir_operand_get_constant(src));
        else if (lir_operand_type(src) == JIT_LIR_OPTYPE_REG) phx_x86_imul_rr(pb, dst, operand_to_gp(src));
        else phx_x86_imul_rm(pb, dst, x86_operand_to_mem(src));
    }
}

/* ---- x86_64 Div / DivUn ---- */
static void x86_translateDiv(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    const LirOperand *divisor = instr->inputs_[instr->num_inputs_ - 1];
    if (lir_operand_type(divisor) == JIT_LIR_OPTYPE_STACK || lir_operand_type(divisor) == JIT_LIR_OPTYPE_MEM || lir_operand_type(divisor) == JIT_LIR_OPTYPE_IND)
        phx_x86_idiv_m(pb, x86_operand_to_mem(divisor));
    else
        phx_x86_idiv_r(pb, operand_to_gp(divisor));
}

static void x86_translateDivUn(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    const LirOperand *divisor = instr->inputs_[instr->num_inputs_ - 1];
    if (lir_operand_type(divisor) == JIT_LIR_OPTYPE_STACK || lir_operand_type(divisor) == JIT_LIR_OPTYPE_MEM || lir_operand_type(divisor) == JIT_LIR_OPTYPE_IND)
        phx_x86_div_m(pb, x86_operand_to_mem(divisor));
    else
        phx_x86_div_r(pb, operand_to_gp(divisor));
}

/* ---- x86_64 FP binary ops ---- */
static void x86_translateFadd(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    if (instr->output_.type_ != JIT_LIR_OPTYPE_NONE) {
        phx_x86_movsd_rr(pb, operand_to_fp(&instr->output_), operand_to_fp(instr->inputs_[0]));
        phx_x86_addsd_rr(pb, operand_to_fp(&instr->output_), operand_to_fp(instr->inputs_[1]));
    } else {
        phx_x86_addsd_rr(pb, operand_to_fp(instr->inputs_[0]), operand_to_fp(instr->inputs_[1]));
    }
}
static void x86_translateFsub(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    if (instr->output_.type_ != JIT_LIR_OPTYPE_NONE) {
        phx_x86_movsd_rr(pb, operand_to_fp(&instr->output_), operand_to_fp(instr->inputs_[0]));
        phx_x86_subsd_rr(pb, operand_to_fp(&instr->output_), operand_to_fp(instr->inputs_[1]));
    } else {
        phx_x86_subsd_rr(pb, operand_to_fp(instr->inputs_[0]), operand_to_fp(instr->inputs_[1]));
    }
}
static void x86_translateFmul(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    if (instr->output_.type_ != JIT_LIR_OPTYPE_NONE) {
        phx_x86_movsd_rr(pb, operand_to_fp(&instr->output_), operand_to_fp(instr->inputs_[0]));
        phx_x86_mulsd_rr(pb, operand_to_fp(&instr->output_), operand_to_fp(instr->inputs_[1]));
    } else {
        phx_x86_mulsd_rr(pb, operand_to_fp(instr->inputs_[0]), operand_to_fp(instr->inputs_[1]));
    }
}
static void x86_translateFdiv(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    if (instr->output_.type_ != JIT_LIR_OPTYPE_NONE) {
        phx_x86_movsd_rr(pb, operand_to_fp(&instr->output_), operand_to_fp(instr->inputs_[0]));
        phx_x86_divsd_rr(pb, operand_to_fp(&instr->output_), operand_to_fp(instr->inputs_[1]));
    } else {
        phx_x86_divsd_rr(pb, operand_to_fp(instr->inputs_[0]), operand_to_fp(instr->inputs_[1]));
    }
}

/* ---- x86_64 Lea ---- */
static void x86_translateLea(void *env, const LirInstruction *instr) {
    phx_x86_lea(get_builder(env), operand_to_gp(&instr->output_), x86_operand_to_mem(instr->inputs_[0]));
}

/* ---- x86_64 Negate / Invert ---- */
static void x86_translateNegate(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    const LirOperand *out = &instr->output_, *in = instr->inputs_[0];
    if (lir_operand_type(out) == JIT_LIR_OPTYPE_NONE) { phx_x86_neg_r(pb, operand_to_gp(in)); }
    else if (lir_operand_type(in) == JIT_LIR_OPTYPE_IMM) { phx_x86_mov_ri(pb, operand_to_gp(out), -(int64_t)lir_operand_get_constant(in)); }
    else if (lir_operand_type(in) == JIT_LIR_OPTYPE_REG) { PhxGp d = operand_to_gp(out); phx_x86_mov_rr(pb, d, operand_to_gp(in)); phx_x86_neg_r(pb, d); }
    else { PhxGp d = operand_to_gp(out); phx_x86_mov_rm(pb, d, x86_operand_to_mem(in)); phx_x86_neg_r(pb, d); }
}

static void x86_translateInvert(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    const LirOperand *out = &instr->output_, *in = instr->inputs_[0];
    if (lir_operand_type(in) == JIT_LIR_OPTYPE_IMM) { phx_x86_mov_ri(pb, operand_to_gp(out), ~(int64_t)lir_operand_get_constant(in)); }
    else if (lir_operand_type(in) == JIT_LIR_OPTYPE_REG) { PhxGp d = operand_to_gp(out); phx_x86_mov_rr(pb, d, operand_to_gp(in)); phx_x86_not_r(pb, d); }
    else { PhxGp d = operand_to_gp(out); phx_x86_mov_rm(pb, d, x86_operand_to_mem(in)); phx_x86_not_r(pb, d); }
}

/* ---- x86_64 Inc / Dec ---- */
static void x86_translateInc(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    const LirOperand *op = instr->inputs_[0];
    if (lir_operand_type(op) == JIT_LIR_OPTYPE_STACK) phx_x86_inc_m(pb, x86_operand_to_mem(op));
    else phx_x86_inc_r(pb, operand_to_gp(op));
}

static void x86_translateDec(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    const LirOperand *op = instr->inputs_[0];
    if (lir_operand_type(op) == JIT_LIR_OPTYPE_STACK) phx_x86_dec_m(pb, x86_operand_to_mem(op));
    else phx_x86_dec_r(pb, operand_to_gp(op));
}

/* ---- x86_64 Push / Pop ---- */
static void x86_translatePush(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    const LirOperand *op = instr->inputs_[0];
    if (lir_operand_type(op) == JIT_LIR_OPTYPE_IMM) phx_x86_push_i(pb, (int32_t)lir_operand_get_constant(op));
    else if (lir_operand_type(op) == JIT_LIR_OPTYPE_STACK) phx_x86_push_m(pb, x86_operand_to_mem(op));
    else phx_x86_push_r(pb, operand_to_gp(op));
}

static void x86_translatePop(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    const LirOperand *out = &instr->output_;
    if (lir_operand_type(out) == JIT_LIR_OPTYPE_STACK) phx_x86_pop_m(pb, x86_operand_to_mem(out));
    else phx_x86_pop_r(pb, operand_to_gp(out));
}

/* ---- x86_64 Exchange ---- */
static void x86_translateExchange(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    const LirOperand *out = &instr->output_, *inp = instr->inputs_[0];
    if (x86_is_vecd(out)) {
        PhxGp a = operand_to_fp(out), b = operand_to_fp(inp);
        phx_x86_pxor_rr(pb, a, b); phx_x86_pxor_rr(pb, b, a); phx_x86_pxor_rr(pb, a, b);
    } else {
        phx_x86_xchg_rr(pb, operand_to_gp(out), operand_to_gp(inp));
    }
}

/* ---- x86_64 Cdq / Cqo ---- */
static void x86_translateCdq(void *env, const LirInstruction *i) { (void)i; phx_x86_cdq(get_builder(env)); }
static void x86_translateCqo(void *env, const LirInstruction *i) { (void)i; phx_x86_cqo(get_builder(env)); }

/* ---- x86_64 MovZX / MovSX / MovSXD ---- */
static void x86_translateMovZX(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    const LirOperand *src = instr->inputs_[0];
    PhxGp dst = operand_to_gp(&instr->output_);
    if (lir_operand_type(src) == JIT_LIR_OPTYPE_REG) phx_x86_movzx_rr(pb, dst, operand_to_gp(src));
    else phx_x86_movzx_rm(pb, dst, x86_operand_to_mem(src));
}

static void x86_translateMovSX(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    const LirOperand *src = instr->inputs_[0];
    PhxGp dst = operand_to_gp(&instr->output_);
    if (lir_operand_type(src) == JIT_LIR_OPTYPE_REG) phx_x86_movsx_rr(pb, dst, operand_to_gp(src));
    else phx_x86_movsx_rm(pb, dst, x86_operand_to_mem(src));
}

static void x86_translateMovSXD(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    const LirOperand *src = instr->inputs_[0];
    PhxGp dst = operand_to_gp(&instr->output_);
    if (lir_operand_type(src) == JIT_LIR_OPTYPE_REG) phx_x86_movsxd_rr(pb, dst, operand_to_gp(src));
    else phx_x86_movsxd_rm(pb, dst, x86_operand_to_mem(src));
}

/* ---- x86_64 Cmp / Test / BitTest ---- */
static void x86_translateCmp(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    const LirOperand *a = instr->inputs_[0], *b = instr->inputs_[1];
    if (lir_operand_type(a) == JIT_LIR_OPTYPE_REG && lir_operand_type(b) == JIT_LIR_OPTYPE_REG) phx_x86_cmp_rr(pb, operand_to_gp(a), operand_to_gp(b));
    else if (lir_operand_type(a) == JIT_LIR_OPTYPE_REG && lir_operand_type(b) == JIT_LIR_OPTYPE_IMM) phx_x86_cmp_ri(pb, operand_to_gp(a), (int32_t)lir_operand_get_constant(b));
    else if (lir_operand_type(a) == JIT_LIR_OPTYPE_REG) phx_x86_cmp_rm(pb, operand_to_gp(a), x86_operand_to_mem(b));
    else phx_x86_cmp_mr(pb, x86_operand_to_mem(a), operand_to_gp(b));
}

static void x86_translateTest(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    const LirOperand *a = instr->inputs_[0], *b = instr->inputs_[1];
    if (lir_operand_type(b) == JIT_LIR_OPTYPE_IMM) phx_x86_test_ri(pb, operand_to_gp(a), (int32_t)lir_operand_get_constant(b));
    else phx_x86_test_rr(pb, operand_to_gp(a), operand_to_gp(b));
}

static void x86_translateBitTest(void *env, const LirInstruction *instr) {
    phx_x86_bt_ri(get_builder(env), operand_to_gp(instr->inputs_[0]),
        (uint8_t)lir_operand_get_constant(instr->inputs_[1]));
}

/* ---- x86_64 Branch ---- */
static void x86_translateBranch(void *env, const LirInstruction *instr) {
    void *block = lir_operand_get_basic_block(instr->inputs_[0]);
    phx_x86_jmp_label(get_builder(env),
        jit_environ_get_block_label(env, (const LirBasicBlock *)block));
}

static void x86_translateCondBranch(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    void *block = lir_operand_get_basic_block(instr->inputs_[0]);
    PhxLabel label = jit_environ_get_block_label(env, (const LirBasicBlock *)block);
    int op = instr->opcode_;
    if      (op == JIT_LIR_OP_BRANCHZ)  phx_x86_jz(pb, label);
    else if (op == JIT_LIR_OP_BRANCHNZ) phx_x86_jnz(pb, label);
    else if (op == JIT_LIR_OP_BRANCHA)  phx_x86_ja(pb, label);
    else if (op == JIT_LIR_OP_BRANCHB)  phx_x86_jb(pb, label);
    else if (op == JIT_LIR_OP_BRANCHAE) phx_x86_jae(pb, label);
    else if (op == JIT_LIR_OP_BRANCHBE) phx_x86_jbe(pb, label);
    else if (op == JIT_LIR_OP_BRANCHG)  phx_x86_jg(pb, label);
    else if (op == JIT_LIR_OP_BRANCHL)  phx_x86_jl(pb, label);
    else if (op == JIT_LIR_OP_BRANCHGE) phx_x86_jge(pb, label);
    else if (op == JIT_LIR_OP_BRANCHLE) phx_x86_jle(pb, label);
    else if (op == JIT_LIR_OP_BRANCHC)  phx_x86_jb(pb, label);
    else if (op == JIT_LIR_OP_BRANCHNC) phx_x86_jae(pb, label);
    else if (op == JIT_LIR_OP_BRANCHO)  phx_x86_jo(pb, label);
    else if (op == JIT_LIR_OP_BRANCHNO) phx_x86_jno(pb, label);
    else if (op == JIT_LIR_OP_BRANCHS)  phx_x86_js(pb, label);
    else if (op == JIT_LIR_OP_BRANCHNS) phx_x86_jns(pb, label);
    else if (op == JIT_LIR_OP_BRANCHE)  phx_x86_je(pb, label);
    else if (op == JIT_LIR_OP_BRANCHNE) phx_x86_jne(pb, label);
}

/* ---- x86_64 Unreachable ---- */
static void x86_translateUnreachable(void *env, const LirInstruction *i) {
    (void)i; phx_x86_ud2(get_builder(env));
}

/* ---- x86_64 Call ---- */
static void x86_translateCall(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    const LirOperand *target;
    if (instr->output_.type_ != JIT_LIR_OPTYPE_NONE)
        target = instr->inputs_[0];
    else
        target = instr->inputs_[0];

    if (lir_operand_type(target) == JIT_LIR_OPTYPE_REG)
        phx_x86_call_r(pb, operand_to_gp(target));
    else if (lir_operand_type(target) == JIT_LIR_OPTYPE_STACK || lir_operand_type(target) == JIT_LIR_OPTYPE_MEM || lir_operand_type(target) == JIT_LIR_OPTYPE_IND)
        phx_x86_call_m(pb, x86_operand_to_mem(target));
    else {
        /* Immediate → load into scratch, call */
        PhxGp r11 = {11, 8};
        phx_x86_mov_ri(pb, r11, (int64_t)lir_operand_get_constant(target));
        phx_x86_call_r(pb, r11);
    }
    /* Debug location — always create/bind label (matches C++ AddDebugEntryAction) */
    {
        PhxLabel label = phx_builder_new_label(pb);
        phx_builder_bind(pb, label);
        if (instr->origin_) {
            jit_environ_add_pending_debug_loc(env, label, instr->origin_);
        }
    }
}

/* ---- x86_64 Select (cmov) ---- */
static void x86_translateSelect(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    PhxGp dst = operand_to_gp(&instr->output_);
    phx_x86_mov_ri(pb, dst, (int64_t)lir_operand_get_constant(instr->inputs_[2]));
    phx_x86_test_rr(pb, operand_to_gp(instr->inputs_[0]), operand_to_gp(instr->inputs_[0]));
    phx_x86_cmovnz_rr(pb, dst, operand_to_gp(instr->inputs_[1]));
}

/* ---- x86_64 IntToBool ---- */
static void
x86_translateIntToBool(void *env, const LirInstruction *instr) {
    PhxBuilder *pb = get_builder(env);
    const LirOperand *input = instr->inputs_[0];
    PhxGp output = operand_to_gp(&instr->output_);
    assert(instr->output_.data_type_ == JIT_LIR_DT_8BIT);
    if (lir_operand_type(input) == JIT_LIR_OPTYPE_IMM)
        phx_x86_mov_ri(pb, output, lir_operand_get_constant(input) ? 1 : 0);
    else {
        phx_x86_test_rr(pb, operand_to_gp(input), operand_to_gp(input));
        phx_x86_setne(pb, output);
    }
}

/* ================================================================
 * x86_64 C dispatch table — replaces C++ trie for all non-Yield opcodes
 * ================================================================ */

int
autogen_c_dispatch(void *env, const LirInstruction *instr) {
    switch (instr->opcode_) {
    case JIT_LIR_OP_BIND: return 1;
    case JIT_LIR_OP_GUARD: autogen_c_TranslateGuard(env, instr); return 1;
    case JIT_LIR_OP_DEOPTPATCHPOINT: autogen_c_TranslateDeoptPatchpoint(env, instr); return 1;
    case JIT_LIR_OP_EQUAL: case JIT_LIR_OP_NOTEQUAL:
    case JIT_LIR_OP_GREATERTHANSIGNED: case JIT_LIR_OP_LESSTHANSIGNED:
    case JIT_LIR_OP_GREATERTHANEQUALSIGNED: case JIT_LIR_OP_LESSTHANEQUALSIGNED:
    case JIT_LIR_OP_GREATERTHANUNSIGNED: case JIT_LIR_OP_LESSTHANUNSIGNED:
    case JIT_LIR_OP_GREATERTHANEQUALUNSIGNED: case JIT_LIR_OP_LESSTHANEQUALUNSIGNED:
        autogen_c_TranslateCompare(env, instr); return 1;
    case JIT_LIR_OP_INTTOBOOL: x86_translateIntToBool(env, instr); return 1;
    case JIT_LIR_OP_MOVE: case JIT_LIR_OP_MOVERELAXED: x86_translateMove(env, instr); return 1;
    case JIT_LIR_OP_ADD: x86_translateAdd(env, instr); return 1;
    case JIT_LIR_OP_SUB: x86_translateSub(env, instr); return 1;
    case JIT_LIR_OP_AND: x86_translateAnd(env, instr); return 1;
    case JIT_LIR_OP_OR:  x86_translateOr(env, instr); return 1;
    case JIT_LIR_OP_XOR: x86_translateXor(env, instr); return 1;
    case JIT_LIR_OP_MUL: x86_translateMul(env, instr); return 1;
    case JIT_LIR_OP_DIV: x86_translateDiv(env, instr); return 1;
    case JIT_LIR_OP_DIVUN: x86_translateDivUn(env, instr); return 1;
    case JIT_LIR_OP_FADD: x86_translateFadd(env, instr); return 1;
    case JIT_LIR_OP_FSUB: x86_translateFsub(env, instr); return 1;
    case JIT_LIR_OP_FMUL: x86_translateFmul(env, instr); return 1;
    case JIT_LIR_OP_FDIV: x86_translateFdiv(env, instr); return 1;
    case JIT_LIR_OP_LEA: x86_translateLea(env, instr); return 1;
    case JIT_LIR_OP_NEGATE: x86_translateNegate(env, instr); return 1;
    case JIT_LIR_OP_INVERT: x86_translateInvert(env, instr); return 1;
    case JIT_LIR_OP_INC: x86_translateInc(env, instr); return 1;
    case JIT_LIR_OP_DEC: x86_translateDec(env, instr); return 1;
    case JIT_LIR_OP_PUSH: x86_translatePush(env, instr); return 1;
    case JIT_LIR_OP_POP: x86_translatePop(env, instr); return 1;
    case JIT_LIR_OP_EXCHANGE: x86_translateExchange(env, instr); return 1;
    case JIT_LIR_OP_CDQ: x86_translateCdq(env, instr); return 1;
    case JIT_LIR_OP_CQO: x86_translateCqo(env, instr); return 1;
    case JIT_LIR_OP_MOVZX: x86_translateMovZX(env, instr); return 1;
    case JIT_LIR_OP_MOVSX: x86_translateMovSX(env, instr); return 1;
    case JIT_LIR_OP_MOVSXD: x86_translateMovSXD(env, instr); return 1;
    case JIT_LIR_OP_CMP: x86_translateCmp(env, instr); return 1;
    case JIT_LIR_OP_TEST: case JIT_LIR_OP_TEST32: x86_translateTest(env, instr); return 1;
    case JIT_LIR_OP_BITTEST: x86_translateBitTest(env, instr); return 1;
    case JIT_LIR_OP_UNREACHABLE: x86_translateUnreachable(env, instr); return 1;
    case JIT_LIR_OP_BRANCH: x86_translateBranch(env, instr); return 1;
    case JIT_LIR_OP_BRANCHNZ: case JIT_LIR_OP_BRANCHZ:
    case JIT_LIR_OP_BRANCHA: case JIT_LIR_OP_BRANCHB:
    case JIT_LIR_OP_BRANCHAE: case JIT_LIR_OP_BRANCHBE:
    case JIT_LIR_OP_BRANCHG: case JIT_LIR_OP_BRANCHL:
    case JIT_LIR_OP_BRANCHGE: case JIT_LIR_OP_BRANCHLE:
    case JIT_LIR_OP_BRANCHC: case JIT_LIR_OP_BRANCHNC:
    case JIT_LIR_OP_BRANCHO: case JIT_LIR_OP_BRANCHNO:
    case JIT_LIR_OP_BRANCHS: case JIT_LIR_OP_BRANCHNS:
    case JIT_LIR_OP_BRANCHE: case JIT_LIR_OP_BRANCHNE:
        x86_translateCondBranch(env, instr); return 1;
    case JIT_LIR_OP_CALL: x86_translateCall(env, instr); return 1;
    case JIT_LIR_OP_SELECT: x86_translateSelect(env, instr); return 1;
    default:
        return 0; /* Yield*, CWD — fall through to C++ trie */
    }
}

#endif /* CINDER_X86_64 */
