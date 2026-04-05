/*
 * postalloc_c.c -- C implementation of post-register-allocation LIR rewrites
 *
 * Phase 3D: Converts postalloc.cpp callback functions to pure C.
 * Uses the C Rewrite framework (rewrite_c.h) and C LIR APIs.
 *
 * Callback functions take (LirInstruction*, void* env) and return
 * LIR_REWRITE_UNCHANGED/CHANGED/REMOVED.
 */

#include "cinderx/Jit/lir/rewrite_c.h"
#include "cinderx/Jit/lir/lir_c_api.h"
#include "cinderx/Jit/lir/lir_impl_internal.h"

#include "Python.h"

#include <assert.h>
#include <stdint.h>

/* ================================================================
 * Callback: removePhiInstructions (stage 0, instruction-level)
 *
 * After register allocation, Phi instructions are no longer needed.
 * Remove them from the LIR.
 * ================================================================ */

static int
postalloc_remove_phi(LirInstruction *instr, void *env) {
    if (instr->opcode_ != JIT_LIR_OP_PHI) {
        return LIR_REWRITE_UNCHANGED;
    }
    LirBasicBlock *block = instr->basic_block_;
    lir_block_remove_instr(block, instr);
    lir_instruction_free(instr);
    return LIR_REWRITE_REMOVED;
}

/* ================================================================
 * Callback: rewriteSubWordRegMoves (ARM64 only, stage 0)
 *
 * AARCH64 only has 32-bit (W) and 64-bit (X) register operands.
 * Rewrite 8-bit and 16-bit register-to-register moves to use
 * 32-bit registers instead.
 * ================================================================ */

#if defined(CINDER_AARCH64)
static int
postalloc_rewrite_subword_reg_moves(LirInstruction *instr, void *env) {
    if (instr->opcode_ != JIT_LIR_OP_MOVE) {
        return LIR_REWRITE_UNCHANGED;
    }

    LirOperand *out = &instr->output_;
    if (out->type_ != JIT_LIR_OPTYPE_REG) {
        return LIR_REWRITE_UNCHANGED;
    }

    uint8_t size = out->data_type_;
    if (size != JIT_LIR_DT_8BIT && size != JIT_LIR_DT_16BIT) {
        return LIR_REWRITE_UNCHANGED;
    }

    LirOperand *in = instr->inputs_[0];
    if (in->type_ != JIT_LIR_OPTYPE_REG && in->type_ != JIT_LIR_OPTYPE_IMM) {
        return LIR_REWRITE_UNCHANGED;
    }

    lir_operand_set_data_type(out, JIT_LIR_DT_32BIT);
    if (in->data_type_ == JIT_LIR_DT_8BIT ||
        in->data_type_ == JIT_LIR_DT_16BIT) {
        lir_operand_set_data_type(in, JIT_LIR_DT_32BIT);
    }
    return LIR_REWRITE_CHANGED;
}
#endif

/* ================================================================
 * Callback: optimizeMoveInstrs (stage 1, instruction-level)
 *
 * Optimize move instructions:
 *   1. Remove moves where source == destination (same reg or stack)
 *   2. Rewrite "mov reg, 0" to "xor reg, reg" (shorter encoding)
 * ================================================================ */

static int
postalloc_optimize_move(LirInstruction *instr, void *env) {
    if (instr->opcode_ != JIT_LIR_OP_MOVE) {
        return LIR_REWRITE_UNCHANGED;
    }

    LirOperand *out = &instr->output_;
    LirOperand *in = instr->inputs_[0];

    /* Case 1: source == destination → remove */
    if ((out->type_ == JIT_LIR_OPTYPE_REG || out->type_ == JIT_LIR_OPTYPE_STACK) &&
        in->type_ == out->type_ &&
        lir_operand_get_phy_reg_or_stack(in).loc ==
            lir_operand_get_phy_reg_or_stack(out).loc) {
        lir_block_remove_instr(instr->basic_block_, instr);
        return LIR_REWRITE_REMOVED;
    }

    /* Case 2: mov reg, 0 → xor reg, reg */
    if (in->type_ == JIT_LIR_OPTYPE_IMM && !lir_operand_is_fp(in) &&
        lir_operand_get_constant(in) == 0 && out->type_ == JIT_LIR_OPTYPE_REG) {
        assert(!in->is_linked_ &&
               "Register allocation should have replaced linked operand");
        LirPhyLocation reg = lir_operand_get_phy_register(out);
        uint8_t data_type = out->data_type_;
        instr->opcode_ = JIT_LIR_OP_XOR;
        lir_operand_set_none(out);
        lir_instruction_set_num_inputs(instr, 0);
        lir_operand_set_data_type(
            lir_instruction_alloc_phyreg_input(instr, reg), data_type);
        lir_operand_set_data_type(
            lir_instruction_alloc_phyreg_input(instr, reg), data_type);
        return LIR_REWRITE_CHANGED;
    }

    return LIR_REWRITE_UNCHANGED;
}

/* ================================================================
 * Callback: rewriteBitExtensionInstrs (stage 0, instruction-level)
 *
 * Replace ZEXT and SEXT with appropriate MOVE instructions.
 * ================================================================ */

static int
postalloc_rewrite_bit_extension(LirInstruction *instr, void *env) {
    int is_sext = (instr->opcode_ == JIT_LIR_OP_SEXT);
    int is_zext = (instr->opcode_ == JIT_LIR_OP_ZEXT);

    if (!is_sext && !is_zext) {
        return LIR_REWRITE_UNCHANGED;
    }

    LirOperand *in = instr->inputs_[0];
    LirOperand *out = &instr->output_;
    uint8_t out_size = out->data_type_;

    if (in->type_ == JIT_LIR_OPTYPE_IMM) {
        uint64_t mask;
        if (out_size == JIT_LIR_DT_32BIT) {
            mask = 0xffffffffUL;
        } else if (out_size == JIT_LIR_DT_16BIT) {
            mask = 0xffffUL;
        } else if (out_size == JIT_LIR_DT_8BIT) {
            mask = 0xffUL;
        } else {
            mask = 0xffffffffffffffffUL;
        }
        lir_operand_set_constant(in, lir_operand_get_constant(in) & mask, out_size);
        instr->opcode_ = JIT_LIR_OP_MOVE;
        return LIR_REWRITE_CHANGED;
    }

    uint8_t in_size = in->data_type_;
    if (in_size >= out_size) {
        instr->opcode_ = JIT_LIR_OP_MOVE;
        return LIR_REWRITE_CHANGED;
    }

    switch (in_size) {
        case JIT_LIR_DT_8BIT:
        case JIT_LIR_DT_16BIT:
            instr->opcode_ = is_sext ? JIT_LIR_OP_MOVSX : JIT_LIR_OP_MOVZX;
            break;
        case JIT_LIR_DT_32BIT:
            if (is_sext) {
                instr->opcode_ = JIT_LIR_OP_MOVSXD;
            } else {
                /* unsigned extension 32→64: 32-bit move does the work */
                instr->opcode_ = JIT_LIR_OP_MOVE;
                lir_operand_set_data_type(out, JIT_LIR_DT_32BIT);
            }
            break;
        default:
            /* k64bit, kObject, kDouble: should not reach here */
            fprintf(stderr, "postalloc: bad input size %d for extension\n", in_size);
            abort();
    }

    return LIR_REWRITE_CHANGED;
}

/* ================================================================
 * Callback: rewriteBinaryOpInstrs (stage 0, instruction-level)
 *
 * For binary ops where OutReg == one of the input registers,
 * rewrite to two-operand form (clear output, swap inputs if needed).
 * ================================================================ */

static int
postalloc_rewrite_binary_op(LirInstruction *instr, void *env) {
    int op = instr->opcode_;

    /* Match: ADD, XOR, AND, OR, MUL, FADD, FMUL */
    if (op != JIT_LIR_OP_ADD && op != JIT_LIR_OP_XOR &&
        op != JIT_LIR_OP_AND && op != JIT_LIR_OP_OR &&
        op != JIT_LIR_OP_MUL && op != JIT_LIR_OP_FADD &&
        op != JIT_LIR_OP_FMUL) {
        return LIR_REWRITE_UNCHANGED;
    }

    LirOperand *out = &instr->output_;
    if (out->type_ != JIT_LIR_OPTYPE_REG ||
        instr->inputs_[0]->type_ != JIT_LIR_OPTYPE_REG) {
        return LIR_REWRITE_UNCHANGED;
    }

    LirPhyLocation out_reg = lir_operand_get_phy_register(out);
    LirPhyLocation in0_reg = lir_operand_get_phy_register(instr->inputs_[0]);

    if (out_reg.loc == in0_reg.loc) {
        lir_operand_set_none(out);
        return LIR_REWRITE_CHANGED;
    }

    LirOperand *in1 = instr->inputs_[1];
    int in1_loc = (in1->type_ == JIT_LIR_OPTYPE_REG)
        ? lir_operand_get_phy_register(in1).loc : LIR_REG_INVALID;
    if (out_reg.loc == in1_loc) {
        lir_operand_set_none(out);
        LirOperand *opnd0 = lir_instruction_remove_input(instr, 0);
        lir_instruction_append_input(instr, opnd0);
        return LIR_REWRITE_CHANGED;
    }

    return LIR_REWRITE_UNCHANGED;
}

/* ================================================================
 * Callback: rewriteLoadInstrs (stage 0, instruction-level)
 *
 * When loading from an absolute memory address into a register,
 * materialize the address into the register first, then load via
 * indirect. Needed when the address doesn't fit in 32 bits (x86)
 * or always on ARM64 (no absolute addressing).
 * ================================================================ */

static int
postalloc_rewrite_load(LirInstruction *instr, void *env) {
    int op = instr->opcode_;
    if (op != JIT_LIR_OP_MOVE && op != JIT_LIR_OP_MOVERELAXED) {
        return LIR_REWRITE_UNCHANGED;
    }
    if (instr->num_inputs_ != 1 || instr->inputs_[0]->type_ != JIT_LIR_OPTYPE_MEM) {
        return LIR_REWRITE_UNCHANGED;
    }

    LirOperand *out = &instr->output_;
    assert(out->type_ == JIT_LIR_OPTYPE_REG);

#if defined(CINDER_X86_64)
    /* RAX can load 64-bit addresses directly */
    if (lir_operand_get_phy_register(out).loc == 0 /* RAX */) {
        return LIR_REWRITE_UNCHANGED;
    }
#endif

    LirOperand *in = instr->inputs_[0];
    intptr_t mem_addr = (intptr_t)in->value_.address;

#if defined(CINDER_X86_64)
    if (lir_fits_signed_int32((int64_t)mem_addr)) {
        return LIR_REWRITE_UNCHANGED;
    }
#endif

    LirBasicBlock *block = instr->basic_block_;
    LirPhyLocation out_reg = lir_operand_get_phy_register(out);
    LirInstruction *m = lir_block_alloc_instr_before(
        block, instr, JIT_LIR_OP_MOVE);
    lir_operand_set_phy_register(&m->output_, out_reg);
    lir_operand_set_data_type(
        lir_instruction_alloc_imm_input(m, (uint64_t)mem_addr, in->data_type_),
        in->data_type_);

    /* Convert MEM operand to indirect through the register */
    lir_operand_set_memory_indirect_phy(in, out_reg, 0);

    return LIR_REWRITE_CHANGED;
}

/* ================================================================
 * Callback: rewriteBranchInstrs (function-level, stage 0)
 *
 * Add unconditional branch instructions at the end of basic blocks
 * when the single successor is not the next block (fallthrough).
 * ================================================================ */

static int
postalloc_rewrite_branch(LirFunction *func, void *env) {
    int changed = 0;
    size_t num_blocks = lir_func_num_blocks(func);

    for (size_t bi = 0; bi < num_blocks; bi++) {
        LirBasicBlock *block = lir_func_get_block(func, bi);
        LirBasicBlock *next_block =
            (bi + 1 < num_blocks) ? lir_func_get_block(func, bi + 1) : NULL;

        if (block->num_succs_ != 1) {
            continue;
        }

        LirInstruction *last_instr = block->instr_tail_;
        int last_opcode = last_instr ? last_instr->opcode_ : JIT_LIR_OP_NONE;

        if (last_opcode == JIT_LIR_OP_RETURN) {
            continue;
        }

        LirBasicBlock *successor = block->successors_[0];
        if (successor == next_block && next_block &&
            next_block->section_ == block->section_) {
            continue;
        }

        if (last_opcode == JIT_LIR_OP_BRANCH) {
            continue;
        }

        const void *origin = last_instr ? last_instr->origin_ : NULL;
        LirInstruction *branch = lir_block_alloc_instr(
            block, JIT_LIR_OP_BRANCH, origin);
        lir_instruction_alloc_label_input(branch, successor);

        changed = 1;
    }

    return changed ? LIR_REWRITE_CHANGED : LIR_REWRITE_UNCHANGED;
}

/* ---- Helpers for rewriteCondBranch ---- */

static void
do_rewrite_cond_branch(LirInstruction *instr, LirBasicBlock *next_block) {
    LirOperand *input = instr->inputs_[0];
    LirBasicBlock *block = instr->basic_block_;

    /* insert test Reg, Reg instruction */
    uint8_t size = input->data_type_;
    LirPhyLocation reg = lir_operand_get_phy_register(input);
    LirInstruction *test = lir_block_alloc_instr_before(
        block, instr, JIT_LIR_OP_TEST);
    lir_operand_set_data_type(
        lir_instruction_alloc_phyreg_input(test, reg), size);
    lir_operand_set_data_type(
        lir_instruction_alloc_phyreg_input(test, reg), size);

    /* convert CondBranch → BranchCC */
    LirBasicBlock *true_block = block->successors_[0];
    LirBasicBlock *false_block = block->successors_[1];
    LirBasicBlock *target_block;
    LirBasicBlock *fallthrough_block;

    int opcode = JIT_LIR_OP_BRANCHNZ;
    if (true_block == next_block) {
        opcode = lir_instruction_negate_branch_cc(opcode);
        target_block = false_block;
        fallthrough_block = true_block;
    } else {
        target_block = true_block;
        fallthrough_block = false_block;
    }

    instr->opcode_ = opcode;
    lir_instruction_set_num_inputs(instr, 0);
    lir_instruction_alloc_label_input(instr, target_block);

    if (fallthrough_block != next_block ||
        (next_block && block->section_ != next_block->section_)) {
        LirInstruction *fb = lir_block_alloc_instr(
            block, JIT_LIR_OP_BRANCH, instr->origin_);
        lir_instruction_alloc_label_input(fb, fallthrough_block);
    }
}

static void
do_rewrite_branch_cc(LirInstruction *instr, LirBasicBlock *next_block) {
    LirBasicBlock *block = instr->basic_block_;
    LirBasicBlock *true_bb = block->successors_[0];
    LirBasicBlock *false_bb = block->successors_[1];
    LirBasicBlock *fallthrough_bb;

    if (true_bb == next_block) {
        instr->opcode_ = lir_instruction_negate_branch_cc(instr->opcode_);
        lir_instruction_alloc_label_input(instr, false_bb);
        fallthrough_bb = true_bb;
    } else {
        lir_instruction_alloc_label_input(instr, true_bb);
        fallthrough_bb = false_bb;
    }

    if (fallthrough_bb != next_block ||
        (next_block && block->section_ != next_block->section_)) {
        LirInstruction *fb = lir_block_alloc_instr(
            block, JIT_LIR_OP_BRANCH, instr->origin_);
        lir_instruction_alloc_label_input(fb, fallthrough_bb);
    }
}

/* ================================================================
 * Callback: rewriteCondBranch (function-level, stage 0)
 *
 * Convert CondBranch to Test + BranchCC. Also handle bare BranchCC
 * that needs label + fallthrough branch.
 * ================================================================ */

static int
postalloc_rewrite_cond_branch(LirFunction *func, void *env) {
    int changed = 0;
    size_t num_blocks = lir_func_num_blocks(func);

    for (size_t bi = 0; bi < num_blocks; bi++) {
        LirBasicBlock *block = lir_func_get_block(func, bi);
        LirInstruction *last = block->instr_tail_;
        if (!last) {
            continue;
        }

        LirBasicBlock *next_block =
            (bi + 1 < num_blocks) ? lir_func_get_block(func, bi + 1) : NULL;

        int op = last->opcode_;
        if (op == JIT_LIR_OP_CONDBRANCH) {
            do_rewrite_cond_branch(last, next_block);
            changed = 1;
        } else if (lir_instruction_is_branch_cc(op) && last->num_inputs_ == 0) {
            do_rewrite_branch_cc(last, next_block);
            changed = 1;
        }
    }

    return changed ? LIR_REWRITE_CHANGED : LIR_REWRITE_UNCHANGED;
}

/* ================================================================
 * Callback: rewriteByteMultiply (x86_64 only, stage 0)
 *
 * Rewrite 8-bit multiply to use single-operand imul. Moves input
 * to AL if needed, sets 16-bit data type (asmjit requirement).
 * ================================================================ */

#if defined(CINDER_X86_64)
#include "cinderx/Jit/codegen/phylocation.h"

static int
postalloc_rewrite_byte_multiply(LirInstruction *instr, void *env) {
    if (instr->opcode_ != JIT_LIR_OP_MUL || instr->num_inputs_ < 2) {
        return LIR_REWRITE_UNCHANGED;
    }

    LirOperand *input0 = instr->inputs_[0];
    if (input0->data_type_ > JIT_LIR_DT_8BIT) {
        return LIR_REWRITE_UNCHANGED;
    }

    LirOperand *output = &instr->output_;
    LirPhyLocation in_reg = lir_operand_get_phy_register(input0);
    LirPhyLocation out_reg = in_reg;

    if (output->type_ == JIT_LIR_OPTYPE_REG) {
        out_reg = lir_operand_get_phy_register(output);
    }

    LirBasicBlock *block = instr->basic_block_;
    LirPhyLocation al_loc = {PHYLOC_RAX, 8}; /* AL = RAX with 8-bit size */
    LirPhyLocation rax_loc = {PHYLOC_RAX, 64};

    if (in_reg.loc != PHYLOC_RAX) {
        LirInstruction *m = lir_block_alloc_instr_before(
            block, instr, JIT_LIR_OP_MOVE);
        lir_operand_set_phy_register(&m->output_, al_loc);
        lir_operand_set_data_type(&m->output_, JIT_LIR_DT_8BIT);
        lir_operand_set_data_type(
            lir_instruction_alloc_phyreg_input(m, in_reg), JIT_LIR_DT_8BIT);
        lir_operand_set_phy_register(input0, al_loc);
    }

    /* asmjit recognizes 8-bit imul only if RAX is 16-bit */
    lir_operand_set_data_type(input0, JIT_LIR_DT_16BIT);
    lir_operand_set_none(output);

    if (out_reg.loc != PHYLOC_RAX) {
        LirInstruction *m = lir_block_alloc_instr_before(
            block, instr->next_, JIT_LIR_OP_MOVE);
        lir_operand_set_phy_register(&m->output_, out_reg);
        lir_operand_set_data_type(&m->output_, JIT_LIR_DT_8BIT);
        lir_operand_set_data_type(
            lir_instruction_alloc_phyreg_input(m, al_loc), JIT_LIR_DT_8BIT);
    }

    return LIR_REWRITE_CHANGED;
}
#endif

/* ---- Helper: move operand to a specific physical register ---- */

static int
insert_move_to_register(LirBasicBlock *block, LirInstruction *before,
                        LirOperand *op, LirPhyLocation location) {
    if (op->type_ == JIT_LIR_OPTYPE_REG &&
        lir_operand_get_phy_register(op).loc == location.loc) {
        return 0; /* already in place */
    }

    uint8_t dt = op->data_type_;
    LirInstruction *move = lir_block_alloc_instr_before(
        block, before, JIT_LIR_OP_MOVE);
    lir_operand_set_phy_register(&move->output_, location);
    lir_operand_set_data_type(&move->output_, dt);

    if (op->type_ == JIT_LIR_OPTYPE_REG) {
        lir_operand_set_data_type(
            lir_instruction_alloc_phyreg_input(move, lir_operand_get_phy_register(op)), dt);
    } else if (op->type_ == JIT_LIR_OPTYPE_IMM) {
        lir_operand_set_data_type(
            lir_instruction_alloc_imm_input(move, lir_operand_get_constant(op), dt), dt);
    } else if (op->type_ == JIT_LIR_OPTYPE_STACK) {
        lir_operand_set_data_type(
            lir_instruction_alloc_stack_input(move, lir_operand_get_stack_slot(op)), dt);
    } else {
        fprintf(stderr, "postalloc: unexpected operand type %d in insertMoveToRegister\n",
                op->type_);
        abort();
    }

    lir_operand_set_phy_register(op, location);
    lir_operand_set_data_type(op, dt);
    return 1;
}

/* ================================================================
 * Callback: rewriteDivide (x86_64 only, stage 0)
 *
 * Rewrite division instructions to use correct x86_64 registers
 * (RAX for dividend lower, RDX for upper/extension).
 * ================================================================ */

static int
postalloc_rewrite_divide(LirInstruction *instr, void *env) {
    int op = instr->opcode_;
    if (op != JIT_LIR_OP_DIV && op != JIT_LIR_OP_DIVUN) {
        return LIR_REWRITE_UNCHANGED;
    }

    int changed = 0;
    LirOperand *output = &instr->output_;
    LirBasicBlock *block = instr->basic_block_;

    LirOperand *dividend_upper = NULL;
    LirOperand *dividend_lower;
    if (instr->num_inputs_ == 3) {
        dividend_upper = instr->inputs_[0];
        dividend_lower = instr->inputs_[1];
    } else {
        dividend_lower = instr->inputs_[0];
    }

    LirPhyLocation rax_loc = {PHYLOC_RAX, 64};
    LirPhyLocation rdx_loc = {PHYLOC_RDX, 64};
    LirPhyLocation ax_loc = {PHYLOC_RAX, 16};

    LirPhyLocation out_reg = rax_loc;
    if (output->type_ != JIT_LIR_OPTYPE_NONE) {
        out_reg = lir_operand_get_phy_register(output);
    } else {
        assert(dividend_lower->type_ == JIT_LIR_OPTYPE_REG);
        out_reg = lir_operand_get_phy_register(dividend_lower);
    }

    if (dividend_lower->data_type_ == JIT_LIR_DT_8BIT) {
        /* 8-bit division: sign/zero extend to AX (16-bit) */
        assert(instr->num_inputs_ == 3);
        int ext_op;
        if (dividend_lower->type_ == JIT_LIR_OPTYPE_IMM) {
            ext_op = JIT_LIR_OP_MOVE;
        } else if (op == JIT_LIR_OP_DIV) {
            ext_op = JIT_LIR_OP_MOVSX;
        } else {
            ext_op = JIT_LIR_OP_MOVZX;
        }

        LirInstruction *move = lir_block_alloc_instr_before(block, instr, ext_op);
        lir_operand_set_phy_register(&move->output_, ax_loc);
        lir_operand_set_data_type(&move->output_, JIT_LIR_DT_16BIT);

        if (dividend_lower->type_ == JIT_LIR_OPTYPE_IMM) {
            lir_operand_set_data_type(dividend_lower, JIT_LIR_DT_16BIT);
        }

        LirOperand *divisor_removed = lir_instruction_remove_input(instr, 2);
        LirOperand *div_lower_removed = lir_instruction_remove_input(instr, 1);
        lir_instruction_append_input(move, div_lower_removed);

        LirOperand *upper_removed = lir_instruction_remove_input(instr, 0);
        (void)upper_removed; /* no longer used */

        lir_operand_set_data_type(
            lir_instruction_alloc_phyreg_input(instr, ax_loc), JIT_LIR_DT_16BIT);
        lir_instruction_append_input(instr, divisor_removed);
        changed = 1;
    } else {
        /* Move dividend_lower to RAX */
        changed |= insert_move_to_register(block, instr, dividend_lower, rax_loc);

        if (dividend_upper != NULL &&
            !(dividend_upper->type_ == JIT_LIR_OPTYPE_REG &&
              lir_operand_get_phy_register(dividend_upper).loc == PHYLOC_RDX)) {
            assert(dividend_upper->type_ == JIT_LIR_OPTYPE_IMM &&
                   lir_operand_get_constant(dividend_upper) == 0);

            if (op == JIT_LIR_OP_DIV) {
                /* sign-extend RAX into RDX */
                int extend_op;
                size_t bits = lir_operand_size_in_bits(dividend_lower);
                if (bits == 16) extend_op = JIT_LIR_OP_CWD;
                else if (bits == 32) extend_op = JIT_LIR_OP_CDQ;
                else extend_op = JIT_LIR_OP_CQO;

                LirInstruction *ext = lir_block_alloc_instr_before(
                    block, instr, extend_op);
                lir_operand_set_phy_register(&ext->output_, rdx_loc);
                lir_instruction_alloc_phyreg_input(ext, rax_loc);
            } else {
                /* zero RDX via xor */
                LirInstruction *xor = lir_block_alloc_instr_before(
                    block, instr, JIT_LIR_OP_XOR);
                lir_instruction_alloc_phyreg_input(xor, rdx_loc);
                lir_instruction_alloc_phyreg_input(xor, rdx_loc);
            }

            lir_operand_set_phy_register(dividend_upper, rdx_loc);
            lir_operand_set_data_type(dividend_upper, dividend_lower->data_type_);
            changed = 1;
        }
    }

    if (out_reg.loc != PHYLOC_RAX) {
        LirInstruction *m = lir_block_alloc_instr_before(
            block, instr->next_, JIT_LIR_OP_MOVE);
        lir_operand_set_phy_register(&m->output_, out_reg);
        lir_operand_set_data_type(&m->output_, dividend_lower->data_type_);
        lir_operand_set_data_type(
            lir_instruction_alloc_phyreg_input(m, rax_loc),
            dividend_lower->data_type_);
        changed = 1;
    }
    lir_operand_set_none(output);

    return changed ? LIR_REWRITE_CHANGED : LIR_REWRITE_UNCHANGED;
}

/* ================================================================
 * Public init — registers the converted callbacks
 *
 * 10 callbacks converted (8 cross-platform + 2 x86_64).
 * ================================================================ */

void
lir_postalloc_rewrite_init(LirRewrite *rw, LirFunction *func, void *env) {
    lir_rewrite_init(rw, func, env);

    /* Stage 0: instruction-level */
    lir_rewrite_add_instr(rw, 0, postalloc_remove_phi);
    lir_rewrite_add_instr(rw, 0, postalloc_rewrite_bit_extension);
    lir_rewrite_add_instr(rw, 0, postalloc_rewrite_load);
    lir_rewrite_add_instr(rw, 0, postalloc_rewrite_binary_op);

    /* Stage 0: function-level */
    lir_rewrite_add_func(rw, 0, postalloc_rewrite_branch);
    lir_rewrite_add_func(rw, 0, postalloc_rewrite_cond_branch);

#if defined(CINDER_X86_64)
    lir_rewrite_add_instr(rw, 0, postalloc_rewrite_byte_multiply);
    lir_rewrite_add_instr(rw, 0, postalloc_rewrite_divide);
#elif defined(CINDER_AARCH64)
    lir_rewrite_add_instr(rw, 0, postalloc_rewrite_subword_reg_moves);
#endif

    /* Stage 1: instruction-level optimizations */
    lir_rewrite_add_instr(rw, 1, postalloc_optimize_move);
}
