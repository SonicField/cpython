/*
 * lir_instruction.c -- C implementation of LIR instruction lifecycle
 *
 * Phase B: Operates on the LirInstruction C struct defined in lir_c_api.h.
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
    inst->opcode = opcode;
    inst->basic_block = basic_block;
    inst->origin = origin;
    inst->output.parent_instr = inst;
    inst->output.type = JIT_LIR_OPTYPE_NONE;
    inst->output.data_type = JIT_LIR_DT_OBJECT;
    inst->inputs = (LirOperand **)PyMem_RawMalloc(
        INITIAL_INPUT_CAPACITY * sizeof(LirOperand *));
    inst->num_inputs = 0;
    inst->inputs_capacity = INITIAL_INPUT_CAPACITY;
    return inst;
}

void
lir_instruction_free(LirInstruction *inst) {
    if (inst == NULL) return;
    /* Free owned input operands */
    for (size_t i = 0; i < inst->num_inputs; i++) {
        lir_operand_free(inst->inputs[i]);
    }
    PyMem_RawFree(inst->inputs);
    /* Free indirect in output if present */
    if (!inst->output.is_linked &&
        inst->output.type == JIT_LIR_OPTYPE_IND) {
        lir_memind_free(inst->output.value.indirect);
    }
    PyMem_RawFree(inst);
}

/* ---- Accessors ---- */

int
lir_instruction_id(const LirInstruction *inst) {
    return inst->id;
}

int
lir_instruction_opcode(const LirInstruction *inst) {
    return inst->opcode;
}

LirOperand *
lir_instruction_output(LirInstruction *inst) {
    return &inst->output;
}

size_t
lir_instruction_num_inputs(const LirInstruction *inst) {
    return inst->num_inputs;
}

LirOperand *
lir_instruction_get_input(const LirInstruction *inst, size_t index) {
    return inst->inputs[index];
}

LirBasicBlock *
lir_instruction_basic_block(const LirInstruction *inst) {
    return inst->basic_block;
}

const void *
lir_instruction_origin(const LirInstruction *inst) {
    return inst->origin;
}

/* ---- Input allocation ---- */

static void
ensure_input_capacity(LirInstruction *inst) {
    if (inst->num_inputs >= inst->inputs_capacity) {
        size_t new_cap = inst->inputs_capacity * 2;
        inst->inputs = (LirOperand **)PyMem_RawRealloc(
            inst->inputs, new_cap * sizeof(LirOperand *));
        inst->inputs_capacity = new_cap;
    }
}

static LirOperand *
append_input(LirInstruction *inst, LirOperand *op) {
    ensure_input_capacity(inst);
    inst->inputs[inst->num_inputs++] = op;
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

/* ---- Mutation ---- */

void
lir_instruction_set_opcode(LirInstruction *inst, int opcode) {
    inst->opcode = opcode;
}

void
lir_instruction_set_id(LirInstruction *inst, int id) {
    inst->id = id;
}

void
lir_instruction_foreach_input(const LirInstruction *inst,
                               void (*cb)(LirOperand *, void *),
                               void *ctx) {
    for (size_t i = 0; i < inst->num_inputs; i++) {
        cb(inst->inputs[i], ctx);
    }
}
