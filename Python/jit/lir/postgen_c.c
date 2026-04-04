/*
 * postgen_c.c -- C implementation of post-generation LIR rewrites
 *
 * Phase 3D: Converts postgen.cpp callback functions to pure C.
 * Uses the C Rewrite framework (rewrite_c.h) and C LIR APIs.
 *
 * Callback functions take (LirInstruction*, void* env) and return
 * LIR_REWRITE_UNCHANGED/CHANGED/REMOVED.
 */

#include "cinderx/Jit/lir/rewrite_c.h"
#include "cinderx/Jit/lir/lir_c_api.h"
#include "cinderx/Jit/lir/lir_impl_internal.h"
#include "cinderx/Jit/jit_config_c.h"

#include "pycore_hashtable.h"

#include <assert.h>
#include <stdint.h>

/* ---- Helper: create linked operand for instruction ---- */

static LirOperand *
make_linked(LirInstruction *parent, LirInstruction *def) {
    LirOperand *op = lir_operand_new_linked(parent, def);
    lir_operand_set_linked_instr(op, def);
    return op;
}

/* ---- Helper: materialize large immediate to register ---- */

static LirInstruction *
materialize_imm_to_reg(LirBasicBlock *block, LirInstruction *before,
                       uint64_t constant, uint8_t data_type) {
    LirInstruction *move = lir_block_alloc_instr_before(
        block, before, JIT_LIR_OP_MOVE);
    lir_operand_set_virtual_register(&move->output_);
    lir_operand_set_data_type(&move->output_, JIT_LIR_DT_OBJECT);
    lir_operand_set_data_type(
        lir_instruction_alloc_imm_input(move, constant, data_type),
        data_type);
    return move;
}

/* ================================================================
 * Callback: rewriteInlineHelper (function-level, stage 0)
 * ================================================================ */

static int
rewrite_inline_helper(LirFunction *func, void *env) {
    const JitConfig *cfg = jit_get_config();
    if (!cfg->lir_opts.inliner) {
        return LIR_REWRITE_UNCHANGED;
    }
    return lir_inliner_inline_calls(func) ? LIR_REWRITE_CHANGED
                                          : LIR_REWRITE_UNCHANGED;
}

/* ================================================================
 * Callback: rewriteBinaryOpConstantPosition (stage 1)
 * ================================================================ */

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
        uint8_t dt = divisor->data_type_;

        LirInstruction *move = materialize_imm_to_reg(block, instr, constant, dt);
        lir_operand_set_data_type(&move->output_, dt);
        lir_instruction_set_input(instr, 2, make_linked(instr, move));
        return LIR_REWRITE_CHANGED;
    }

    /* Check for binary ops */
    if (op != JIT_LIR_OP_ADD && op != JIT_LIR_OP_SUB &&
        op != JIT_LIR_OP_XOR && op != JIT_LIR_OP_AND &&
        op != JIT_LIR_OP_OR && op != JIT_LIR_OP_MUL &&
        !lir_instruction_is_compare(op)) {
        return LIR_REWRITE_UNCHANGED;
    }

    LirOperand *input0 = instr->inputs_[0];
    LirOperand *input1 = instr->inputs_[1];

    if (input0->type_ != JIT_LIR_OPTYPE_IMM) {
        return LIR_REWRITE_UNCHANGED;
    }

    /* Commutative: swap operands */
    int is_commutative = (op != JIT_LIR_OP_SUB);
    if (is_commutative && input1->type_ != JIT_LIR_OPTYPE_IMM) {
        if (lir_instruction_is_compare(op)) {
            instr->opcode_ = lir_instruction_flip_comparison_direction(op);
        }
        LirOperand *imm = lir_instruction_remove_input(instr, 0);
        lir_instruction_append_input(instr, imm);
        return LIR_REWRITE_CHANGED;
    }

    /* Non-commutative or both immediate: materialize first input */
    uint64_t constant = lir_operand_get_constant(input0);
    uint8_t dt = input0->data_type_;
    LirInstruction *move = materialize_imm_to_reg(block, instr, constant, dt);
    lir_operand_set_data_type(&move->output_, dt);
    lir_instruction_set_input(instr, 0, make_linked(instr, move));
    return LIR_REWRITE_CHANGED;
}

/* ================================================================
 * Callback: rewriteBinaryOpLargeConstant (stage 1)
 * ================================================================ */

static int
rewrite_binary_op_large_constant(LirInstruction *instr, void *env) {
    int op = instr->opcode_;
    if (op != JIT_LIR_OP_ADD && op != JIT_LIR_OP_SUB &&
        op != JIT_LIR_OP_XOR && op != JIT_LIR_OP_AND &&
        op != JIT_LIR_OP_OR && op != JIT_LIR_OP_MUL &&
        !lir_instruction_is_compare(op)) {
        return LIR_REWRITE_UNCHANGED;
    }

    if (instr->inputs_[0]->type_ == JIT_LIR_OPTYPE_IMM) {
        return LIR_REWRITE_UNCHANGED; /* another rewrite will fix this */
    }

    LirOperand *in1 = instr->inputs_[1];
    if (in1->type_ != JIT_LIR_OPTYPE_IMM) {
        return LIR_REWRITE_UNCHANGED;
    }

    uint64_t constant = lir_operand_get_constant_or_address(in1);

#if defined(CINDER_X86_64)
    if ((lir_operand_size_in_bits(in1) < 64) ||
        lir_fits_signed_int32((int64_t)constant)) {
        return LIR_REWRITE_UNCHANGED;
    }
#elif defined(CINDER_AARCH64)
    /* ARM64: defer to C++ postgen.cpp for now (asmjit Utils needed) */
    return LIR_REWRITE_UNCHANGED;
#endif

    LirBasicBlock *block = instr->basic_block_;
    LirInstruction *move = materialize_imm_to_reg(block, instr, constant, in1->data_type_);

    /* If first operand is smaller, sign-extend it */
    if (lir_operand_size_in_bits(instr->inputs_[0]) < lir_operand_size_in_bits(in1)) {
        LirInstruction *movsx = lir_block_alloc_instr_before(
            block, instr, JIT_LIR_OP_MOVSX);
        lir_operand_set_virtual_register(&movsx->output_);
        lir_operand_set_data_type(&movsx->output_, in1->data_type_);
        lir_instruction_append_input(movsx, lir_instruction_release_input(instr, 0));
        lir_instruction_set_input(instr, 0, make_linked(instr, movsx));
    }

    lir_instruction_set_input(instr, 1, make_linked(instr, move));
    return LIR_REWRITE_CHANGED;
}

/* ================================================================
 * Callback: rewriteGuardLargeConstant (stage 1)
 * ================================================================ */

static int
rewrite_guard_large_constant(LirInstruction *instr, void *env) {
    if (instr->opcode_ != JIT_LIR_OP_GUARD) {
        return LIR_REWRITE_UNCHANGED;
    }

    const size_t kTargetIndex = 3;
    LirOperand *target_opnd = instr->inputs_[kTargetIndex];
    if (target_opnd->type_ != JIT_LIR_OPTYPE_IMM &&
        target_opnd->type_ != JIT_LIR_OPTYPE_MEM) {
        return LIR_REWRITE_UNCHANGED;
    }

    uint64_t target_imm = lir_operand_get_constant_or_address(target_opnd);

#if defined(CINDER_X86_64)
    if (lir_fits_signed_int32((int64_t)target_imm)) {
        return LIR_REWRITE_UNCHANGED;
    }
#elif defined(CINDER_AARCH64)
    /* ARM64: defer to C++ postgen.cpp for now */
    return LIR_REWRITE_UNCHANGED;
#endif

    LirBasicBlock *block = instr->basic_block_;
    LirInstruction *move = materialize_imm_to_reg(
        block, instr, target_imm, target_opnd->data_type_);
    lir_instruction_set_input(instr, kTargetIndex, make_linked(instr, move));
    return LIR_REWRITE_CHANGED;
}

/* ================================================================
 * Callback: rewriteMoveToMemoryLargeConstant (x86_64 only, stage 1)
 * ================================================================ */

#if defined(CINDER_X86_64)
static int
rewrite_move_to_memory_large_constant(LirInstruction *instr, void *env) {
    int op = instr->opcode_;
    LirOperand *out = &instr->output_;

    if ((op != JIT_LIR_OP_MOVE && op != JIT_LIR_OP_MOVERELAXED) ||
        out->type_ != JIT_LIR_OPTYPE_IND) {
        return LIR_REWRITE_UNCHANGED;
    }

    LirOperand *input = instr->inputs_[0];
    if (input->type_ != JIT_LIR_OPTYPE_IMM && input->type_ != JIT_LIR_OPTYPE_MEM) {
        return LIR_REWRITE_UNCHANGED;
    }

    uint64_t constant = lir_operand_get_constant_or_address(input);
    if (lir_fits_signed_int32((int64_t)constant)) {
        return LIR_REWRITE_UNCHANGED;
    }

    LirBasicBlock *block = instr->basic_block_;
    LirInstruction *move = materialize_imm_to_reg(
        block, instr, constant, input->data_type_);
    lir_instruction_set_input(instr, 0, make_linked(instr, move));
    return LIR_REWRITE_CHANGED;
}
#endif

/* ================================================================
 * Callback: rewritePromoteOutputSize (ARM64 only, stage 1)
 * ================================================================ */

#if defined(CINDER_AARCH64)
static int
rewrite_promote_output_size(LirInstruction *instr, void *env) {
    switch (instr->opcode_) {
        case JIT_LIR_OP_EQUAL:
        case JIT_LIR_OP_NOTEQUAL:
        case JIT_LIR_OP_GREATERTHANSIGNED:
        case JIT_LIR_OP_GREATERTHANEQUALSIGNED:
        case JIT_LIR_OP_LESSTHANSIGNED:
        case JIT_LIR_OP_LESSTHANEQUALSIGNED:
        case JIT_LIR_OP_GREATERTHANUNSIGNED:
        case JIT_LIR_OP_GREATERTHANEQUALUNSIGNED:
        case JIT_LIR_OP_LESSTHANUNSIGNED:
        case JIT_LIR_OP_LESSTHANEQUALUNSIGNED:
            if (lir_operand_size_in_bits(&instr->output_) < 32) {
                lir_operand_set_data_type(&instr->output_, JIT_LIR_DT_32BIT);
                return LIR_REWRITE_CHANGED;
            }
            return LIR_REWRITE_UNCHANGED;
        default:
            return LIR_REWRITE_UNCHANGED;
    }
}
#endif

/* ================================================================
 * Callback: rewriteLoadArg (stage 1)
 * Rewrite LoadArg → Bind with physical register for its input.
 * ================================================================ */

static int
rewrite_load_arg(LirInstruction *instr, void *env) {
    if (instr->opcode_ != JIT_LIR_OP_LOADARG) {
        return LIR_REWRITE_UNCHANGED;
    }
    instr->opcode_ = JIT_LIR_OP_BIND;
    assert(instr->num_inputs_ == 1);
    LirOperand *input = instr->inputs_[0];
    assert(input->type_ == JIT_LIR_OPTYPE_IMM);
    uint64_t arg_idx = input->value_.constant;
    LirPhyLocation loc = jit_environ_get_arg_location(env, (size_t)arg_idx);
    lir_operand_set_phy_reg_or_stack(input, loc);
    lir_operand_set_data_type(input, instr->output_.data_type_);
    return LIR_REWRITE_CHANGED;
}

/* ================================================================
 * Callback: removePhiInstructions (from postalloc, stage 0)
 * ================================================================ */

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

/* ================================================================
 * Callback: rewriteLoadSecondCallResult (stage 1)
 *
 * Replaces LoadSecondCallResult with a Move from the second return
 * register (RDX/X1), inserted immediately after the Call instruction.
 * Handles Phi chains recursively.
 * ================================================================ */

/* Forward declaration for mutual recursion */
static void populate_load_second_call_result_phi(
    uint8_t data_type, LirInstruction *phi1, LirInstruction *phi2,
    _Py_hashtable_t *seen_srcs);

static LirInstruction *
get_second_call_result(uint8_t data_type, LirOperand *src,
                       LirInstruction *instr, _Py_hashtable_t *seen_srcs) {
    /* Check if already handled */
    LirInstruction *cached = (LirInstruction *)_Py_hashtable_get(seen_srcs, src);
    if (cached != NULL) {
        return cached;
    }

    LirInstruction *src_instr = src->parent_instr_;
    LirBasicBlock *src_block = src_instr->basic_block_;

    assert(src_instr->opcode_ == JIT_LIR_OP_CALL ||
           src_instr->opcode_ == JIT_LIR_OP_PHI);

    if (src_instr->opcode_ == JIT_LIR_OP_CALL) {
        /* Verify this Call hasn't already been handled */
        LirInstruction *next_instr = src_instr->next_;
        if (next_instr != NULL) {
            LirPhyLocation ret1 = jit_environ_get_return_reg(1);
            assert(!(next_instr->opcode_ == JIT_LIR_OP_MOVE &&
                     next_instr->num_inputs_ == 1 &&
                     next_instr->inputs_[0]->type_ == JIT_LIR_OPTYPE_REG &&
                     next_instr->inputs_[0]->value_.phy_loc.loc == ret1.loc));
        }
    }

    if (instr != NULL) {
        /* Reuse existing instruction — move it after src_instr */
        LirBasicBlock *instr_block = instr->basic_block_;
        lir_block_remove_instr(instr_block, instr);
        LirInstruction *after_src = src_instr->next_;
        if (after_src != NULL) {
            lir_block_insert_instr_before(src_block, after_src, instr);
        } else {
            lir_block_append_instr(src_block, instr);
        }
        lir_instruction_set_num_inputs(instr, 0);
    }

    int new_op = (src_instr->opcode_ == JIT_LIR_OP_CALL)
        ? JIT_LIR_OP_MOVE : JIT_LIR_OP_PHI;

    if (instr != NULL) {
        instr->opcode_ = new_op;
    } else {
        instr = lir_block_alloc_instr_before(src_block, src_instr->next_, new_op);
        lir_operand_set_virtual_register(&instr->output_);
        lir_operand_set_data_type(&instr->output_, data_type);
    }

    _Py_hashtable_set(seen_srcs, src, instr);

    if (new_op == JIT_LIR_OP_MOVE) {
        LirPhyLocation ret1 = jit_environ_get_return_reg(1);
        lir_operand_set_data_type(
            lir_instruction_alloc_phyreg_input(instr, ret1), data_type);
    } else {
        populate_load_second_call_result_phi(data_type, src_instr, instr, seen_srcs);
    }

    return instr;
}

static void
populate_load_second_call_result_phi(
    uint8_t data_type, LirInstruction *phi1, LirInstruction *phi2,
    _Py_hashtable_t *seen_srcs) {
    for (size_t i = 1; i < phi1->num_inputs_; i += 2) {
        LirOperand *src1 = lir_operand_get_define(phi1->inputs_[i]);
        LirInstruction *instr2 =
            get_second_call_result(data_type, src1, NULL, seen_srcs);
        lir_instruction_alloc_label_input(
            phi2, (LirBasicBlock *)phi1->inputs_[i - 1]->value_.block);
        lir_instruction_alloc_linked_input(phi2, instr2);
    }
}

static int
rewrite_load_second_call_result(LirInstruction *instr, void *env) {
    if (instr->opcode_ != JIT_LIR_OP_LOADSECONDCALLRESULT) {
        return LIR_REWRITE_UNCHANGED;
    }

    LirOperand *src = lir_operand_get_define(instr->inputs_[0]);
    _Py_hashtable_t *seen_srcs = _Py_hashtable_new(
        _Py_hashtable_hash_ptr, _Py_hashtable_compare_direct);

    get_second_call_result(instr->output_.data_type_, src, instr, seen_srcs);

    _Py_hashtable_destroy(seen_srcs);
    return LIR_REWRITE_REMOVED;
}

/* ================================================================
 * Public init — registers ALL converted callbacks
 *
 * ALL 8 postgen callbacks converted to C.
 *
 * Remaining ARM64 gaps (deferred, not blocking):
 * - ARM64 rewriteBinaryOpLargeConstant body: needs asmjit arm Utils
 * - ARM64 rewriteGuardLargeConstant body: needs asmjit arm Utils
 * ================================================================ */

void
lir_postgen_rewrite_init(LirRewrite *rw, LirFunction *func, void *env) {
    lir_rewrite_init(rw, func, env);

    /* Stage 0: function-level */
    lir_rewrite_add_func(rw, 0, rewrite_inline_helper);

    /* Stage 1: instruction-level */
    lir_rewrite_add_instr(rw, 1, rewrite_binary_op_constant_position);
    lir_rewrite_add_instr(rw, 1, rewrite_binary_op_large_constant);
    lir_rewrite_add_instr(rw, 1, rewrite_guard_large_constant);

#if defined(CINDER_X86_64)
    lir_rewrite_add_instr(rw, 1, rewrite_move_to_memory_large_constant);
#elif defined(CINDER_AARCH64)
    lir_rewrite_add_instr(rw, 1, rewrite_promote_output_size);
#endif

    lir_rewrite_add_instr(rw, 1, rewrite_load_arg);

    lir_rewrite_add_instr(rw, 1, rewrite_load_second_call_result);
}
