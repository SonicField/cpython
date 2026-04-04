/*
 * postgen_c.c -- C implementation of post-generation LIR rewrites
 *
 * Phase 3D: Converts postgen.cpp callback functions to pure C.
 * Uses the C Rewrite framework (rewrite_c.h) and C LIR APIs.
 *
 * Callback functions take (LirInstruction*, void* env) and return
 * LIR_REWRITE_UNCHANGED/CHANGED/REMOVED.
 *
 * NOTE: Instructions created here via lir_block_alloc_instr_before use
 * PyMem_RawCalloc. On Linux, this is compatible with C++ delete (both
 * use the system allocator). The layout-compatible structs (verified by
 * static_assert + offsetof) allow C functions to operate on C++-created
 * objects and vice versa.
 */

#include "cinderx/Jit/lir/rewrite_c.h"
#include "cinderx/Jit/lir/lir_c_api.h"
#include "cinderx/Jit/lir/lir_impl_internal.h"

#include <assert.h>
#include <stdint.h>

/*
 * rewriteBinaryOpConstantPosition:
 * If a binary operation has a constant as the first input,
 * swap operands (commutative) or materialize it to a register.
 * For div/divun, the divisor (input[2]) can't be immediate — materialize.
 */
static int
rewrite_binary_op_constant_position(LirInstruction *instr, void *env) {
    int op = instr->opcode_;
    LirBasicBlock *block = instr->basic_block_;

    /* Handle div/divun: divisor can't be immediate */
    if (op == JIT_LIR_OP_DIV || op == JIT_LIR_OP_DIVUN) {
        LirOperand *divisor = instr->inputs_[2];
        if (divisor->type_ != JIT_LIR_OPTYPE_IMM) {
            return LIR_REWRITE_UNCHANGED;
        }

        uint64_t constant = divisor->value_.constant;
        uint8_t constant_size = divisor->data_type_;

        LirInstruction *move = lir_block_alloc_instr_before(
            block, instr, JIT_LIR_OP_MOVE);
        lir_operand_set_virtual_register(&move->output_);
        lir_operand_set_data_type(&move->output_, constant_size);
        lir_operand_set_data_type(
            lir_instruction_alloc_imm_input(move, constant, constant_size),
            constant_size);

        LirOperand *linked = lir_operand_new_linked(instr, move);
        lir_operand_set_linked_instr(linked, move);
        lir_instruction_set_input(instr, 2, linked);
        return LIR_REWRITE_CHANGED;
    }

    /* Check for binary ops */
    if (op != JIT_LIR_OP_ADD && op != JIT_LIR_OP_SUB &&
        op != JIT_LIR_OP_XOR && op != JIT_LIR_OP_AND &&
        op != JIT_LIR_OP_OR && op != JIT_LIR_OP_MUL &&
        !lir_instruction_is_compare(op)) {
        return LIR_REWRITE_UNCHANGED;
    }

    int is_commutative_or_compare = (op != JIT_LIR_OP_SUB);
    LirOperand *input0 = instr->inputs_[0];
    LirOperand *input1 = instr->inputs_[1];

    if (input0->type_ != JIT_LIR_OPTYPE_IMM) {
        return LIR_REWRITE_UNCHANGED;
    }

    /* Commutative: swap operands */
    if (is_commutative_or_compare && input1->type_ != JIT_LIR_OPTYPE_IMM) {
        if (lir_instruction_is_compare(op)) {
            instr->opcode_ = lir_instruction_flip_comparison_direction(op);
        }
        LirOperand *imm = lir_instruction_remove_input(instr, 0);
        lir_instruction_append_input(instr, imm);
        return LIR_REWRITE_CHANGED;
    }

    /* Non-commutative or both immediate: materialize first input */
    uint64_t constant = lir_operand_get_constant(input0);
    uint8_t constant_size = input0->data_type_;

    LirInstruction *move = lir_block_alloc_instr_before(
        block, instr, JIT_LIR_OP_MOVE);
    lir_operand_set_virtual_register(&move->output_);
    lir_operand_set_data_type(&move->output_, constant_size);
    lir_operand_set_data_type(
        lir_instruction_alloc_imm_input(move, constant, constant_size),
        constant_size);

    LirOperand *linked = lir_operand_new_linked(instr, move);
    lir_operand_set_linked_instr(linked, move);
    lir_instruction_set_input(instr, 0, linked);

    return LIR_REWRITE_CHANGED;
}

/*
 * removePhiInstructions:
 * Remove all Phi instructions (they're resolved during register allocation).
 */
static int
remove_phi_instructions(LirInstruction *instr, void *env) {
    if (instr->opcode_ != JIT_LIR_OP_PHI) {
        return LIR_REWRITE_UNCHANGED;
    }
    LirBasicBlock *block = instr->basic_block_;
    lir_block_remove_instr(block, instr);
    lir_instruction_free(instr);
    return LIR_REWRITE_REMOVED;
}

/* ---- Public init function ---- */

/*
 * Initialize a PostGenerationRewrite-equivalent LirRewrite.
 *
 * NOTE: This is a PARTIAL conversion — only callbacks that are fully
 * converted to C are registered here. The remaining callbacks
 * (rewriteInlineHelper, rewriteBinaryOpLargeConstant, rewriteGuardLargeConstant,
 * rewriteLoadArg, rewriteLoadSecondCallResult, arch-specific rewrites)
 * still need C++ dependencies (getConfig, fitsSignedInt<32>, asmjit Utils,
 * LIRInliner, RETURN_REGS, UnorderedMap).
 *
 * Full conversion will happen incrementally as those dependencies are
 * made available in C.
 */
void
lir_postgen_rewrite_init_partial(LirRewrite *rw, LirFunction *func, void *env) {
    lir_rewrite_init(rw, func, env);
    /* Stage 1 callbacks (converted to C) */
    lir_rewrite_add_instr(rw, 1, rewrite_binary_op_constant_position);
}
