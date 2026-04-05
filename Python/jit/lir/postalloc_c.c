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
    size_t num_blocks = func->num_blocks_;

    for (size_t bi = 0; bi < num_blocks; bi++) {
        LirBasicBlock *block = func->blocks_[bi];
        LirBasicBlock *next_block =
            (bi + 1 < num_blocks) ? func->blocks_[bi + 1] : NULL;

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
    size_t num_blocks = func->num_blocks_;

    for (size_t bi = 0; bi < num_blocks; bi++) {
        LirBasicBlock *block = func->blocks_[bi];
        LirInstruction *last = block->instr_tail_;
        if (!last) {
            continue;
        }

        LirBasicBlock *next_block =
            (bi + 1 < num_blocks) ? func->blocks_[bi + 1] : NULL;

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
 * Callback: rewriteCallInstrs (stage 0, instruction-level)
 *
 * Rewrites call instructions:
 *   - Move function arguments to the correct registers.
 *   - Handle VectorCall, VarArgCall, and regular Call variants.
 *   - Track max stack argument buffer size in Environ.
 * ================================================================ */

/* Helper: allocate a memory-indirect input operand */
static LirOperand *
alloc_memind_input(LirInstruction *inst, LirPhyLocation base, int32_t offset) {
    LirOperand *op = lir_operand_new(inst);
    lir_operand_set_memory_indirect_phy(op, base, offset);
    lir_instruction_append_input(inst, op);
    return op;
}

/*
 * Insert a move from an operand to a memory location [base + index].
 * Handles > 32-bit immediates, FP values, stack operands.
 */
static void
insert_move_to_memory_location(
    LirBasicBlock *block,
    LirInstruction *before,
    LirPhyLocation base,
    int32_t index,
    const LirOperand *operand,
    LirPhyLocation temp)
{
    uint8_t data_type = operand->data_type_;

    if (operand->type_ == JIT_LIR_OPTYPE_IMM) {
        uint64_t constant = lir_operand_get_constant(operand);
        if (
#if defined(CINDER_X86_64)
            !lir_fits_signed_int32((int64_t)constant) ||
#endif
            lir_operand_is_fp(operand)) {
            /* Load constant to temp register first */
            LirInstruction *m1 = lir_block_alloc_instr_before(
                block, before, JIT_LIR_OP_MOVE);
            lir_operand_set_phy_register(&m1->output_, temp);
            lir_operand_set_data_type(&m1->output_, data_type);
            lir_operand_set_data_type(
                lir_instruction_alloc_imm_input(m1, constant, data_type),
                data_type);

            LirInstruction *m2 = lir_block_alloc_instr_before(
                block, before, JIT_LIR_OP_MOVE);
            lir_operand_set_memory_indirect_phy(&m2->output_, base, index);
            lir_operand_set_data_type(
                lir_instruction_alloc_phyreg_input(m2, temp), data_type);
        } else {
            LirInstruction *m = lir_block_alloc_instr_before(
                block, before, JIT_LIR_OP_MOVE);
            lir_operand_set_memory_indirect_phy(&m->output_, base, index);
            lir_operand_set_data_type(
                lir_instruction_alloc_imm_input(m, constant, data_type),
                data_type);
        }
        return;
    }

    if (operand->type_ == JIT_LIR_OPTYPE_REG) {
        LirPhyLocation loc = lir_operand_get_phy_register(operand);
        LirInstruction *m = lir_block_alloc_instr_before(
            block, before, JIT_LIR_OP_MOVE);
        lir_operand_set_memory_indirect_phy(&m->output_, base, index);
        lir_instruction_alloc_phyreg_input(m, loc);
        return;
    }

    /* Stack operand: load to temp, then store to memory */
    LirPhyLocation loc = lir_operand_get_stack_slot(operand);
    LirInstruction *m1 = lir_block_alloc_instr_before(
        block, before, JIT_LIR_OP_MOVE);
    lir_operand_set_phy_register(&m1->output_, temp);
    lir_operand_set_data_type(&m1->output_, data_type);
    lir_instruction_alloc_stack_input(m1, loc);

    LirInstruction *m2 = lir_block_alloc_instr_before(
        block, before, JIT_LIR_OP_MOVE);
    lir_operand_set_memory_indirect_phy(&m2->output_, base, index);
    lir_operand_set_data_type(
        lir_instruction_alloc_phyreg_input(m2, temp), data_type);
}

/*
 * Rewrite regular function call: move arguments to argument registers,
 * spill excess to stack.
 */
static int
rewrite_regular_function(LirInstruction *instr) {
    LirBasicBlock *block = instr->basic_block_;
    size_t num_inputs = instr->num_inputs_;
    size_t arg_reg = 0;
    size_t fp_arg_reg = 0;
    int stack_arg_size = 0;

    size_t num_arg_regs = jit_arch_num_arg_regs();
    size_t num_fp_arg_regs = jit_arch_num_fp_arg_regs();
    LirPhyLocation scratch_0 = jit_arch_scratch_0_loc();
    LirPhyLocation sp = jit_arch_stack_pointer_loc();

    for (size_t i = 1; i < num_inputs; i++) {
        LirOperand *operand = instr->inputs_[i];
        int operand_imm = (operand->type_ == JIT_LIR_OPTYPE_IMM);

        if (lir_operand_is_fp(operand)) {
            if (fp_arg_reg < num_fp_arg_regs) {
                if (operand_imm) {
                    LirInstruction *mi = lir_block_alloc_instr_before(
                        block, instr, JIT_LIR_OP_MOVE);
                    lir_operand_set_phy_register(&mi->output_, scratch_0);
                    lir_instruction_alloc_imm_input(mi,
                        lir_operand_get_constant(operand),
                        operand->data_type_);
                }
                LirPhyLocation fp_reg = jit_arch_fp_arg_reg(fp_arg_reg++);
                LirInstruction *move = lir_block_alloc_instr_before(
                    block, instr, JIT_LIR_OP_MOVE);
                lir_operand_set_phy_register(&move->output_, fp_reg);
                lir_operand_set_data_type(&move->output_, JIT_LIR_DT_DOUBLE);

                if (operand_imm) {
                    lir_instruction_alloc_phyreg_input(move, scratch_0);
                } else {
                    lir_instruction_append_input(move,
                        lir_instruction_release_input(instr, i));
                }
            } else {
                insert_move_to_memory_location(
                    block, instr, sp, stack_arg_size, operand, scratch_0);
                stack_arg_size += (int)sizeof(void*);
            }
            continue;
        }

        if (arg_reg < num_arg_regs) {
            LirPhyLocation areg = jit_arch_arg_reg(arg_reg++);
            LirInstruction *move = lir_block_alloc_instr_before(
                block, instr, JIT_LIR_OP_MOVE);
            lir_operand_set_phy_register(&move->output_, areg);
            lir_operand_set_data_type(&move->output_, operand->data_type_);
            lir_instruction_append_input(move,
                lir_instruction_release_input(instr, i));
        } else {
            insert_move_to_memory_location(
                block, instr, sp, stack_arg_size, operand, scratch_0);
            stack_arg_size += (int)sizeof(void*);
        }
    }

    return stack_arg_size;
}

/*
 * Build an argument array on the stack for vectorcall/vararg.
 */
static int
prepare_args_array(
    LirInstruction *instr,
    size_t num_args,
    size_t flags,
    size_t first_arg,
    LirPhyLocation dest,
    LirPhyLocation size_dest)
{
    LirBasicBlock *block = instr->basic_block_;
    const size_t PTR_SIZE = sizeof(void*);

    /* offset on the stack where arg reservation starts */
    const int kVectorcallArgsOffset = 1;
    size_t num_allocs = num_args + kVectorcallArgsOffset;
    int rsp_sub = (int)(((num_allocs % 2) ? num_allocs + 1 : num_allocs)
                        * PTR_SIZE);

    LirPhyLocation sp = jit_arch_stack_pointer_loc();
    LirPhyLocation scratch_0 = jit_arch_scratch_0_loc();

    /* lea dest, [sp + kVectorcallArgsOffset * PTR_SIZE] */
    LirInstruction *lea = lir_block_alloc_instr_before(
        block, instr, JIT_LIR_OP_LEA);
    lir_operand_set_phy_register(&lea->output_, dest);
    alloc_memind_input(lea, sp, (int32_t)(kVectorcallArgsOffset * PTR_SIZE));

    /* mov size_dest, num_args | flags */
    LirInstruction *mov = lir_block_alloc_instr_before(
        block, instr, JIT_LIR_OP_MOVE);
    lir_operand_set_phy_register(&mov->output_, size_dest);
    lir_operand_set_data_type(&mov->output_, JIT_LIR_DT_64BIT);
    lir_operand_set_data_type(
        lir_instruction_alloc_imm_input(mov, num_args | flags, JIT_LIR_DT_64BIT),
        JIT_LIR_DT_64BIT);

    for (size_t i = first_arg; i < first_arg + num_args; i++) {
        LirOperand *arg = instr->inputs_[i];
        int32_t arg_offset = (int32_t)((i - first_arg) * PTR_SIZE);
        insert_move_to_memory_location(
            block, instr, dest, arg_offset, arg, scratch_0);
    }
    return rsp_sub;
}

/*
 * Rewrite vectorcall: set up callable, args array, nargsf, kwnames.
 */
static int
rewrite_vectorcall_functions(LirInstruction *instr) {
    /* For vector calls there are 4 fixed arguments:
     * #0   - runtime helper function
     * #1   - flags to be added to nargsf
     * #2   - callable
     * #n-1 - kwnames */
    const int kFirstArg = 3;

    size_t flag = lir_operand_get_constant(instr->inputs_[1]);
    size_t num_args = instr->num_inputs_ - kFirstArg - 1;

    LirBasicBlock *block = instr->basic_block_;
    LirPhyLocation arg0 = jit_arch_arg_reg(0);
    LirPhyLocation arg1 = jit_arch_arg_reg(1);
    LirPhyLocation arg2 = jit_arch_arg_reg(2);
    LirPhyLocation arg3 = jit_arch_arg_reg(3);

    /* first argument: callable */
    LirInstruction *move = lir_block_alloc_instr_before(
        block, instr, JIT_LIR_OP_MOVE);
    lir_operand_set_phy_register(&move->output_, arg0);
    lir_operand_set_data_type(&move->output_, instr->inputs_[2]->data_type_);
    lir_instruction_append_input(move,
        lir_instruction_release_input(instr, 2));

    int rsp_sub = prepare_args_array(
        instr, num_args,
        flag | PY_VECTORCALL_ARGUMENTS_OFFSET,
        kFirstArg, arg1, arg2);

    /* kwnames: last input */
    LirOperand *last_input = lir_instruction_release_input(
        instr, instr->num_inputs_ - 1);
    if (last_input->type_ == JIT_LIR_OPTYPE_IMM) {
        assert(lir_operand_get_constant(last_input) == 0);
        /* xor arg3, arg3 */
        LirInstruction *xor_instr = lir_block_alloc_instr_before(
            block, instr, JIT_LIR_OP_XOR);
        lir_instruction_alloc_phyreg_input(xor_instr, arg3);
        lir_instruction_alloc_phyreg_input(xor_instr, arg3);
    } else {
        LirInstruction *move_2 = lir_block_alloc_instr_before(
            block, instr, JIT_LIR_OP_MOVE);
        lir_operand_set_phy_register(&move_2->output_, arg3);
        lir_instruction_append_input(move_2, last_input);

        /* Subtract kwnames tuple length from nargsf */
        size_t ob_size_offs = offsetof(PyVarObject, ob_size);
        LirPhyLocation tmp_reg = jit_arch_scratch_0_loc();
        LirInstruction *load = lir_block_alloc_instr_before(
            block, instr, JIT_LIR_OP_MOVE);
        lir_operand_set_phy_register(&load->output_, tmp_reg);
        alloc_memind_input(load, arg3, (int32_t)ob_size_offs);

        LirInstruction *sub = lir_block_alloc_instr_before(
            block, instr, JIT_LIR_OP_SUB);
        lir_instruction_alloc_phyreg_input(sub, arg2);
        lir_instruction_alloc_phyreg_input(sub, tmp_reg);
    }

    return rsp_sub;
}

/*
 * Rewrite vararg call: build args array, set opcode to Call.
 */
static int
rewrite_vararg_call(LirInstruction *instr) {
    LirPhyLocation arg0 = jit_arch_arg_reg(0);
    LirPhyLocation arg1 = jit_arch_arg_reg(1);

    instr->opcode_ = JIT_LIR_OP_CALL;
    int res = prepare_args_array(
        instr,
        instr->num_inputs_ - 1, /* func is 1st argument */
        0,
        1,
        arg0,
        arg1);
    lir_instruction_set_num_inputs(instr, 1);
    return res;
}

static int
postalloc_rewrite_call(LirInstruction *instr, void *env) {
    if (instr->opcode_ == JIT_LIR_OP_VARARGCALL) {
        int rsp_sub = rewrite_vararg_call(instr);
        jit_environ_update_max_arg_buffer(env, rsp_sub);
        return LIR_REWRITE_CHANGED;
    }

    if (instr->opcode_ != JIT_LIR_OP_CALL &&
        instr->opcode_ != JIT_LIR_OP_VECTORCALL) {
        return LIR_REWRITE_UNCHANGED;
    }

    LirOperand *output = &instr->output_;
    if (instr->opcode_ == JIT_LIR_OP_CALL &&
        instr->num_inputs_ == 1 &&
        output->type_ == JIT_LIR_OPTYPE_NONE) {
        return LIR_REWRITE_UNCHANGED;
    }

    int rsp_sub = 0;
    LirBasicBlock *block = instr->basic_block_;

    if (instr->opcode_ == JIT_LIR_OP_VECTORCALL) {
        rsp_sub = rewrite_vectorcall_functions(instr);
    } else {
        rsp_sub = rewrite_regular_function(instr);
    }

    lir_instruction_set_num_inputs(instr, 1); /* leave function operand only */
    instr->opcode_ = JIT_LIR_OP_CALL;

    LirInstruction *next_iter = instr->next_;

    jit_environ_update_max_arg_buffer(env, rsp_sub);

    if (output->type_ == JIT_LIR_OPTYPE_NONE) {
        return LIR_REWRITE_CHANGED;
    }

    LirPhyLocation return_reg = lir_operand_is_fp(output)
        ? jit_arch_double_return_loc()
        : jit_arch_general_return_loc();

    if (!(output->type_ == JIT_LIR_OPTYPE_REG &&
          lir_operand_get_phy_register(output).loc == return_reg.loc)) {
        LirInstruction *m = lir_block_alloc_instr_before(
            block, next_iter, JIT_LIR_OP_MOVE);
        if (output->type_ == JIT_LIR_OPTYPE_REG) {
            lir_operand_set_phy_register(&m->output_,
                lir_operand_get_phy_register(output));
        } else {
            lir_operand_set_stack_slot(&m->output_,
                lir_operand_get_stack_slot(output));
        }
        lir_operand_set_data_type(&m->output_, output->data_type_);
        lir_operand_set_data_type(
            lir_instruction_alloc_phyreg_input(m, return_reg),
            output->data_type_);
    }
    lir_operand_set_none(output);

    return LIR_REWRITE_CHANGED;
}

/* ================================================================
 * Callback: optimizeMoveSequence (function-level, stage 1)
 *
 * Track register-to-memory moves within a basic block.
 * Replace stack inputs with the register they came from.
 * Delete spills that become dead after replacement.
 * ================================================================ */

#define REG_MEM_TRACK_MAX 64

typedef struct {
    int32_t reg_loc;
    int32_t mem_loc;
} RegMemEntry;

typedef struct {
    int32_t mem_loc;
    int32_t reg_loc;
    LirInstruction *instr;
} MemRegEntry;

typedef struct {
    RegMemEntry r2m[REG_MEM_TRACK_MAX];
    int r2m_count;
    MemRegEntry m2r[REG_MEM_TRACK_MAX];
    int m2r_count;
} RegMemTracker;

static void
tracker_clear(RegMemTracker *t) {
    t->r2m_count = 0;
    t->m2r_count = 0;
}

/* Remove entry for a register from both maps */
static void
tracker_invalidate_register(RegMemTracker *t, int32_t reg) {
    for (int i = 0; i < t->r2m_count; i++) {
        if (t->r2m[i].reg_loc == reg) {
            int32_t mem = t->r2m[i].mem_loc;
            /* Remove from r2m */
            t->r2m[i] = t->r2m[--t->r2m_count];
            /* Remove corresponding m2r */
            for (int j = 0; j < t->m2r_count; j++) {
                if (t->m2r[j].mem_loc == mem) {
                    t->m2r[j] = t->m2r[--t->m2r_count];
                    break;
                }
            }
            return;
        }
    }
}

/* Remove entry for a memory location from both maps */
static void
tracker_invalidate_memory(RegMemTracker *t, int32_t mem) {
    for (int i = 0; i < t->m2r_count; i++) {
        if (t->m2r[i].mem_loc == mem) {
            int32_t reg = t->m2r[i].reg_loc;
            /* Remove from m2r */
            t->m2r[i] = t->m2r[--t->m2r_count];
            /* Remove corresponding r2m */
            for (int j = 0; j < t->r2m_count; j++) {
                if (t->r2m[j].reg_loc == reg) {
                    t->r2m[j] = t->r2m[--t->r2m_count];
                    break;
                }
            }
            return;
        }
    }
}

static void
tracker_invalidate(RegMemTracker *t, int32_t loc) {
    if (loc >= 0)
        tracker_invalidate_register(t, loc);
    else
        tracker_invalidate_memory(t, loc);
}

static void
tracker_add(RegMemTracker *t, int32_t reg, int32_t mem,
            LirInstruction *instr)
{
    tracker_invalidate_memory(t, mem);
    tracker_invalidate_register(t, reg);

    if (t->r2m_count < REG_MEM_TRACK_MAX) {
        t->r2m[t->r2m_count].reg_loc = reg;
        t->r2m[t->r2m_count].mem_loc = mem;
        t->r2m_count++;
    }
    if (t->m2r_count < REG_MEM_TRACK_MAX) {
        t->m2r[t->m2r_count].mem_loc = mem;
        t->m2r[t->m2r_count].reg_loc = reg;
        t->m2r[t->m2r_count].instr = instr;
        t->m2r_count++;
    }
}

static int32_t
tracker_get_reg_from_mem(const RegMemTracker *t, int32_t mem) {
    for (int i = 0; i < t->m2r_count; i++) {
        if (t->m2r[i].mem_loc == mem)
            return t->m2r[i].reg_loc;
    }
    return LIR_REG_INVALID;
}

static LirInstruction *
tracker_get_instr_from_mem(const RegMemTracker *t, int32_t mem) {
    for (int i = 0; i < t->m2r_count; i++) {
        if (t->m2r[i].mem_loc == mem)
            return t->m2r[i].instr;
    }
    return NULL;
}

static int
optimize_move_sequence_block(LirBasicBlock *block) {
    int changed = 0;
    RegMemTracker tracker;
    tracker_clear(&tracker);

    for (LirInstruction *instr = block->instr_head_;
         instr != NULL;
         instr = instr->next_) {

        /* Skip yields — they need special handling */
        if (lir_instruction_is_any_yield(instr->opcode_)) {
            /* still process output/invalidation below */
        } else {
            int32_t out_reg = (instr->output_.type_ == JIT_LIR_OPTYPE_REG)
                ? lir_operand_get_phy_register(&instr->output_).loc
                : LIR_REG_INVALID;
            /* for moves only we can generate A = Move A, which gets optimized out */
            if (instr->opcode_ == JIT_LIR_OP_MOVE) {
                out_reg = LIR_REG_INVALID;
            }

            for (size_t i = 0; i < instr->num_inputs_; i++) {
                LirOperand *operand = instr->inputs_[i];
                if (operand->type_ != JIT_LIR_OPTYPE_STACK) {
                    continue;
                }

                int32_t stack_slot = lir_operand_get_stack_slot(operand).loc;
                int32_t reg = tracker_get_reg_from_mem(&tracker, stack_slot);
                if (reg == LIR_REG_INVALID || reg == out_reg) {
                    continue;
                }

                uint8_t data_type = operand->data_type_;
                LirPhyLocation reg_loc = {reg, 64};
                lir_operand_set_phy_register(operand, reg_loc);
                assert(jit_lir_bit_size(data_type) ==
                       jit_lir_bit_size(operand->data_type_));
                changed = 1;

                /* If last use, delete the spill instruction */
                if (operand->last_use_) {
                    LirInstruction *spill =
                        tracker_get_instr_from_mem(&tracker, stack_slot);
                    assert(spill != NULL);
                    lir_instruction_free(
                        lir_block_remove_instr(block, spill));
                }
            }
        }

        /* Update tracking */
        int is_move = (instr->opcode_ == JIT_LIR_OP_MOVE);
        int is_push = (instr->opcode_ == JIT_LIR_OP_PUSH);
        int is_pop = (instr->opcode_ == JIT_LIR_OP_POP);

        if (is_move || is_push || is_pop) {
            if (is_move) {
                LirOperand *out = &instr->output_;
                LirOperand *in = instr->inputs_[0];
                if (out->type_ == JIT_LIR_OPTYPE_STACK &&
                    in->type_ == JIT_LIR_OPTYPE_REG) {
                    tracker_add(&tracker,
                        lir_operand_get_phy_register(in).loc,
                        lir_operand_get_stack_slot(out).loc,
                        instr);
                } else {
                    if (out->type_ == JIT_LIR_OPTYPE_STACK ||
                        out->type_ == JIT_LIR_OPTYPE_REG) {
                        tracker_invalidate(&tracker,
                            lir_operand_get_phy_reg_or_stack(out).loc);
                    }
                }
            } else if (is_pop) {
                LirOperand *opnd = &instr->output_;
                if (opnd->type_ == JIT_LIR_OPTYPE_STACK ||
                    opnd->type_ == JIT_LIR_OPTYPE_REG) {
                    tracker_invalidate(&tracker,
                        lir_operand_get_phy_reg_or_stack(opnd).loc);
                }
            }
        } else {
            /* Non-move/push/pop: clear all tracking */
            tracker_clear(&tracker);
        }
    }
    return changed;
}

static int
postalloc_optimize_move_sequence(LirFunction *func, void *env) {
    int changed = 0;
    for (size_t bi = 0; bi < func->num_blocks_; bi++) {
        if (optimize_move_sequence_block(func->blocks_[bi]))
            changed = 1;
    }
    return changed ? LIR_REWRITE_CHANGED : LIR_REWRITE_UNCHANGED;
}

/* ================================================================
 * Public init — registers ALL 12 callbacks
 * ================================================================ */

void
lir_postalloc_rewrite_init(LirRewrite *rw, LirFunction *func, void *env) {
    lir_rewrite_init(rw, func, env);

    /* Stage 0: instruction-level */
    lir_rewrite_add_instr(rw, 0, postalloc_rewrite_call);
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

    /* Stage 1: function-level optimizations */
    lir_rewrite_add_func(rw, 1, postalloc_optimize_move_sequence);

    /* Stage 1: instruction-level optimizations */
    lir_rewrite_add_instr(rw, 1, postalloc_optimize_move);
}
