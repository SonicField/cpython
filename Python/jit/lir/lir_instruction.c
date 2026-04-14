/*
 * lir_instruction.c -- C implementation of LIR instruction lifecycle
 *
 * Phase B: Operates on the LirInstruction C struct defined in lir_types_c.h.
 * Coexists with instruction.cpp until all consumers use the C API.
 */

#include "cinderx/Jit/lir/lir_impl_internal.h"

#include "Python.h"

#include <string.h>

#define INITIAL_INPUT_CAPACITY 4

/* ---- Lifecycle ---- */

LirInstruction *
lir_instruction_create(LirBasicBlock *basic_block, int opcode, const void *origin) {
    LirInstruction *inst = (LirInstruction *)PyMem_RawCalloc(
        1, sizeof(LirInstruction));
    inst->opcode_ = opcode;
    inst->basic_block_ = basic_block;
    inst->origin_ = origin;
    inst->output_.parent_instr_ = inst;
    inst->output_.type_ = JIT_LIR_OPTYPE_NONE;
    inst->output_.data_type_ = JIT_LIR_DT_OBJECT;
    inst->inputs_ = (LirOperand **)PyMem_RawCalloc(
        INITIAL_INPUT_CAPACITY, sizeof(LirOperand *));
    inst->num_inputs_ = 0;
    inst->inputs_capacity_ = INITIAL_INPUT_CAPACITY;
    return inst;
}

void
lir_instruction_free(LirInstruction *inst) {
    if (inst == NULL) return;
    /* Free owned input operands */
    for (size_t i = 0; i < inst->num_inputs_; i++) {
        lir_operand_free(inst->inputs_[i]);
    }
    PyMem_RawFree(inst->inputs_);
    /* Free indirect in output if present */
    if (!inst->output_.is_linked_ &&
        inst->output_.type_ == JIT_LIR_OPTYPE_IND) {
        lir_memind_free(inst->output_.value_.indirect);
    }
    PyMem_RawFree(inst);
}

/* ---- Accessors ---- */

int
lir_instruction_id(const LirInstruction *inst) {
    return inst->id_;
}

int
lir_instruction_opcode(const LirInstruction *inst) {
    return inst->opcode_;
}

LirOperand *
lir_instruction_output(LirInstruction *inst) {
    return &inst->output_;
}

size_t
lir_instruction_num_inputs(const LirInstruction *inst) {
    return inst->num_inputs_;
}

LirOperand *
lir_instruction_get_input(const LirInstruction *inst, size_t index) {
    return inst->inputs_[index];
}

LirBasicBlock *
lir_instruction_basic_block(const LirInstruction *inst) {
    return inst->basic_block_;
}

const void *
lir_instruction_origin(const LirInstruction *inst) {
    return inst->origin_;
}

/* ---- Input allocation ---- */

static void
ensure_input_capacity(LirInstruction *inst) {
    if (inst->num_inputs_ >= inst->inputs_capacity_) {
        size_t new_cap = inst->inputs_capacity_ == 0 ? 4 : inst->inputs_capacity_ * 2;
        inst->inputs_ = (LirOperand **)PyMem_RawRealloc(
            inst->inputs_, new_cap * sizeof(LirOperand *));
        inst->inputs_capacity_ = new_cap;
    }
}

static LirOperand *
append_input(LirInstruction *inst, LirOperand *op) {
    ensure_input_capacity(inst);
    inst->inputs_[inst->num_inputs_++] = op;
    return op;
}

LirOperand *
lir_instruction_alloc_imm_input(LirInstruction *inst, uint64_t val, int dt) {
    LirOperand *op = lir_operand_new(inst);
    lir_operand_set_constant(op, val, (uint8_t)dt);
    return append_input(inst, op);
}

LirOperand *
lir_instruction_alloc_fp_imm_input(LirInstruction *inst, double val) {
    LirOperand *op = lir_operand_new(inst);
    lir_operand_set_fp_constant(op, val);
    return append_input(inst, op);
}

LirOperand *
lir_instruction_alloc_linked_input(LirInstruction *inst,
                                    LirInstruction *def_instr) {
    LirOperand *op = lir_operand_new_linked(inst, def_instr);
    return append_input(inst, op);
}

LirOperand *
lir_instruction_alloc_phyreg_input(LirInstruction *inst, LirPhyLocation loc) {
    LirOperand *op = lir_operand_new(inst);
    lir_operand_set_phy_register(op, loc);
    return append_input(inst, op);
}

LirOperand *
lir_instruction_alloc_stack_input(LirInstruction *inst, LirPhyLocation loc) {
    LirOperand *op = lir_operand_new(inst);
    lir_operand_set_stack_slot(op, loc);
    return append_input(inst, op);
}

LirOperand *
lir_instruction_alloc_label_input(LirInstruction *inst, LirBasicBlock *block) {
    LirOperand *op = lir_operand_new(inst);
    lir_operand_set_basic_block(op, block);
    return append_input(inst, op);
}

LirOperand *
lir_instruction_alloc_addr_input(LirInstruction *inst, void *addr) {
    LirOperand *op = lir_operand_new(inst);
    lir_operand_set_mem_address(op, addr);
    return append_input(inst, op);
}

/* ---- Input manipulation ---- */

void
lir_instruction_set_input(LirInstruction *inst, size_t i, LirOperand *input) {
    assert(i < inst->num_inputs_);
    lir_operand_free(inst->inputs_[i]);
    inst->inputs_[i] = input;
    if (input) {
        input->parent_instr_ = inst;
    }
}

LirOperand *
lir_instruction_append_input(LirInstruction *inst, LirOperand *operand) {
    ensure_input_capacity(inst);
    inst->inputs_[inst->num_inputs_] = NULL;
    inst->num_inputs_++;
    lir_instruction_set_input(inst, inst->num_inputs_ - 1, operand);
    return operand;
}

LirOperand *
lir_instruction_release_input(LirInstruction *inst, size_t index) {
    assert(index < inst->num_inputs_);
    LirOperand *op = inst->inputs_[index];
    if (op) {
        op->parent_instr_ = NULL;
    }
    inst->inputs_[index] = NULL;
    return op;
}

LirOperand *
lir_instruction_remove_input(LirInstruction *inst, size_t index) {
    LirOperand *op = lir_instruction_release_input(inst, index);
    /* Shift remaining inputs left */
    for (size_t i = index; i + 1 < inst->num_inputs_; i++) {
        inst->inputs_[i] = inst->inputs_[i + 1];
    }
    inst->num_inputs_--;
    inst->inputs_[inst->num_inputs_] = NULL;
    return op;
}

LirOperand *
lir_instruction_prepend_input(LirInstruction *inst, LirOperand *operand) {
    ensure_input_capacity(inst);
    /* Shift all existing inputs right */
    for (size_t i = inst->num_inputs_; i > 0; i--) {
        inst->inputs_[i] = inst->inputs_[i - 1];
    }
    inst->inputs_[0] = NULL;
    inst->num_inputs_++;
    lir_instruction_set_input(inst, 0, operand);
    return operand;
}

void
lir_instruction_set_num_inputs(LirInstruction *inst, size_t n) {
    if (n > inst->num_inputs_) {
        size_t new_cap = inst->inputs_capacity_;
        while (new_cap < n) new_cap = new_cap == 0 ? 4 : new_cap * 2;
        if (new_cap > inst->inputs_capacity_) {
            inst->inputs_ = (LirOperand **)PyMem_RawRealloc(
                inst->inputs_, new_cap * sizeof(LirOperand *));
            inst->inputs_capacity_ = new_cap;
        }
        for (size_t i = inst->num_inputs_; i < n; i++) {
            inst->inputs_[i] = NULL;
        }
    } else if (n < inst->num_inputs_) {
        for (size_t i = n; i < inst->num_inputs_; i++) {
            lir_operand_free(inst->inputs_[i]);
            inst->inputs_[i] = NULL;
        }
    }
    inst->num_inputs_ = n;
}

/* ---- Query methods ---- */

int
lir_instruction_get_num_outputs(const LirInstruction *inst) {
    return inst->output_.type_ == JIT_LIR_OPTYPE_NONE ? 0 : 1;
}

LirOperand *
lir_instruction_get_operand_by_predecessor(const LirInstruction *inst,
                                            const LirBasicBlock *pred) {
    /* Caller must ensure inst is a Phi instruction. Phi inputs are pairs:
     * [label, value, label, value, ...]. We search for pred in the labels. */
    for (size_t i = 0; i < inst->num_inputs_; i += 2) {
        if (inst->inputs_[i]->type_ == JIT_LIR_OPTYPE_LABEL &&
            (LirBasicBlock *)inst->inputs_[i]->value_.block == pred) {
            return inst->inputs_[i + 1];
        }
    }
    return NULL;
}

/* ---- Mutation ---- */

void
lir_instruction_set_opcode(LirInstruction *inst, int opcode) {
    inst->opcode_ = opcode;
}

void
lir_instruction_set_id(LirInstruction *inst, int id) {
    inst->id_ = id;
}

void
lir_instruction_set_basic_block(LirInstruction *inst, LirBasicBlock *bb) {
    inst->basic_block_ = bb;
}

void
lir_instruction_foreach_input(const LirInstruction *inst,
                               void (*cb)(LirOperand *, void *),
                               void *ctx) {
    for (size_t i = 0; i < inst->num_inputs_; i++) {
        cb(inst->inputs_[i], ctx);
    }
}

/* ---- Opcode query functions ---- */

int
lir_instruction_is_compare(int opcode) {
    switch (opcode) {
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
            return 1;
        default:
            return 0;
    }
}

int
lir_instruction_is_branch_cc(int opcode) {
    switch (opcode) {
        case JIT_LIR_OP_BRANCHC:
        case JIT_LIR_OP_BRANCHNC:
        case JIT_LIR_OP_BRANCHO:
        case JIT_LIR_OP_BRANCHNO:
        case JIT_LIR_OP_BRANCHS:
        case JIT_LIR_OP_BRANCHNS:
        case JIT_LIR_OP_BRANCHZ:
        case JIT_LIR_OP_BRANCHNZ:
        case JIT_LIR_OP_BRANCHA:
        case JIT_LIR_OP_BRANCHB:
        case JIT_LIR_OP_BRANCHBE:
        case JIT_LIR_OP_BRANCHAE:
        case JIT_LIR_OP_BRANCHL:
        case JIT_LIR_OP_BRANCHG:
        case JIT_LIR_OP_BRANCHLE:
        case JIT_LIR_OP_BRANCHGE:
        case JIT_LIR_OP_BRANCHE:
        case JIT_LIR_OP_BRANCHNE:
            return 1;
        default:
            return 0;
    }
}

int
lir_instruction_is_any_branch(int opcode) {
    return (opcode == JIT_LIR_OP_CONDBRANCH) || lir_instruction_is_branch_cc(opcode);
}

int
lir_instruction_is_terminator(int opcode) {
    return opcode == JIT_LIR_OP_RETURN;
}

int
lir_instruction_is_any_yield(int opcode) {
    switch (opcode) {
        case JIT_LIR_OP_YIELDFROM:
        case JIT_LIR_OP_YIELDFROMHANDLESTOPASYNCITERATION:
        case JIT_LIR_OP_YIELDFROMSKIPINITIALSEND:
        case JIT_LIR_OP_YIELDINITIAL:
        case JIT_LIR_OP_YIELDVALUE:
            return 1;
        default:
            return 0;
    }
}

/* ---- Opcode manipulation ---- */

#define CASE_FLIP(op1, op2) \
    case op1: return op2;   \
    case op2: return op1;

int
lir_instruction_negate_branch_cc(int opcode) {
    switch (opcode) {
        CASE_FLIP(JIT_LIR_OP_BRANCHC, JIT_LIR_OP_BRANCHNC)
        CASE_FLIP(JIT_LIR_OP_BRANCHO, JIT_LIR_OP_BRANCHNO)
        CASE_FLIP(JIT_LIR_OP_BRANCHS, JIT_LIR_OP_BRANCHNS)
        CASE_FLIP(JIT_LIR_OP_BRANCHZ, JIT_LIR_OP_BRANCHNZ)
        CASE_FLIP(JIT_LIR_OP_BRANCHA, JIT_LIR_OP_BRANCHBE)
        CASE_FLIP(JIT_LIR_OP_BRANCHB, JIT_LIR_OP_BRANCHAE)
        CASE_FLIP(JIT_LIR_OP_BRANCHL, JIT_LIR_OP_BRANCHGE)
        CASE_FLIP(JIT_LIR_OP_BRANCHG, JIT_LIR_OP_BRANCHLE)
        CASE_FLIP(JIT_LIR_OP_BRANCHE, JIT_LIR_OP_BRANCHNE)
        default:
            assert(0 && "Not a conditional branch opcode");
            return opcode;
    }
}

int
lir_instruction_flip_branch_cc_direction(int opcode) {
    switch (opcode) {
        CASE_FLIP(JIT_LIR_OP_BRANCHA, JIT_LIR_OP_BRANCHB)
        CASE_FLIP(JIT_LIR_OP_BRANCHAE, JIT_LIR_OP_BRANCHBE)
        CASE_FLIP(JIT_LIR_OP_BRANCHL, JIT_LIR_OP_BRANCHG)
        CASE_FLIP(JIT_LIR_OP_BRANCHLE, JIT_LIR_OP_BRANCHGE)
        default:
            assert(0 && "Unable to flip branch condition for opcode");
            return opcode;
    }
}

int
lir_instruction_flip_comparison_direction(int opcode) {
    switch (opcode) {
        CASE_FLIP(JIT_LIR_OP_GREATERTHANEQUALSIGNED, JIT_LIR_OP_LESSTHANEQUALSIGNED)
        CASE_FLIP(JIT_LIR_OP_GREATERTHANEQUALUNSIGNED, JIT_LIR_OP_LESSTHANEQUALUNSIGNED)
        CASE_FLIP(JIT_LIR_OP_GREATERTHANSIGNED, JIT_LIR_OP_LESSTHANSIGNED)
        CASE_FLIP(JIT_LIR_OP_GREATERTHANUNSIGNED, JIT_LIR_OP_LESSTHANUNSIGNED)
        case JIT_LIR_OP_EQUAL: return JIT_LIR_OP_EQUAL;
        case JIT_LIR_OP_NOTEQUAL: return JIT_LIR_OP_NOTEQUAL;
        default:
            assert(0 && "Unable to flip comparison direction for opcode");
            return opcode;
    }
}

int
lir_instruction_compare_to_branch_cc(int opcode) {
    switch (opcode) {
        case JIT_LIR_OP_EQUAL: return JIT_LIR_OP_BRANCHE;
        case JIT_LIR_OP_NOTEQUAL: return JIT_LIR_OP_BRANCHNE;
        case JIT_LIR_OP_GREATERTHANUNSIGNED: return JIT_LIR_OP_BRANCHA;
        case JIT_LIR_OP_LESSTHANUNSIGNED: return JIT_LIR_OP_BRANCHB;
        case JIT_LIR_OP_GREATERTHANEQUALUNSIGNED: return JIT_LIR_OP_BRANCHAE;
        case JIT_LIR_OP_LESSTHANEQUALUNSIGNED: return JIT_LIR_OP_BRANCHBE;
        case JIT_LIR_OP_GREATERTHANSIGNED: return JIT_LIR_OP_BRANCHG;
        case JIT_LIR_OP_LESSTHANSIGNED: return JIT_LIR_OP_BRANCHL;
        case JIT_LIR_OP_GREATERTHANEQUALSIGNED: return JIT_LIR_OP_BRANCHGE;
        case JIT_LIR_OP_LESSTHANEQUALSIGNED: return JIT_LIR_OP_BRANCHLE;
        default:
            assert(0 && "Not a compare opcode");
            return opcode;
    }
}

#undef CASE_FLIP

/* ---- Opcode name table (86 entries, matches FOREACH_INSTR_TYPE order) ---- */

static const char *const s_opcode_names[] = {
    "Bind", "Nop", "Unreachable", "Call", "VectorCall", "VarArgCall",
    "Guard", "DeoptPatchpoint", "Sext", "Zext", "Negate", "Invert",
    "Add", "Sub", "And", "Xor", "Div", "DivUn", "Mul", "Or",
    "Fadd", "Fsub", "Fmul", "Fdiv",
    "LShift", "RShift", "RShiftUn", "Test", "Test32",
    "Equal", "NotEqual",
    "GreaterThanSigned", "LessThanSigned",
    "GreaterThanEqualSigned", "LessThanEqualSigned",
    "GreaterThanUnsigned", "LessThanUnsigned",
    "GreaterThanEqualUnsigned", "LessThanEqualUnsigned",
    "Cmp", "Lea", "LoadArg", "LoadSecondCallResult", "Exchange",
    "Move", "MoveRelaxed", "Push", "Pop", "Cdq", "Cwd", "Cqo",
    "Branch", "BranchNZ", "BranchZ",
    "BranchA", "BranchB", "BranchAE", "BranchBE",
    "BranchG", "BranchL", "BranchGE", "BranchLE",
    "BranchC", "BranchNC", "BranchO", "BranchNO",
    "BranchS", "BranchNS", "BranchE", "BranchNE",
    "BitTest", "Inc", "Dec", "CondBranch", "Select", "Phi", "Return",
    "MovZX", "MovSX", "MovSXD", "IntToBool",
    "YieldInitial", "YieldFrom", "YieldFromSkipInitialSend",
    "YieldFromHandleStopAsyncIteration", "YieldValue",
};

#define NUM_LIR_OPCODES (sizeof(s_opcode_names) / sizeof(s_opcode_names[0]))

const char *
lir_instruction_opcode_name(int opcode) {
    if (opcode < 0 || (size_t)opcode >= NUM_LIR_OPCODES) {
        return "<unknown>";
    }
    return s_opcode_names[opcode];
}
