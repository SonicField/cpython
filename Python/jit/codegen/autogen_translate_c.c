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

#include <assert.h>
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

/* ---- Helper: convert LIR operand to PhxXmm register (x86_64) ---- */

static inline PhxXmm
operand_to_xmm(const LirOperand *op) {
    int reg = lir_operand_get_phy_register(op).loc - PHYLOC_VECD_REG_BASE;
    return phx_xmm(reg);
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

#endif /* CINDER_AARCH64 */
