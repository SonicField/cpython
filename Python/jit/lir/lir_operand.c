/*
 * lir_operand.c -- C implementation of LIR operand lifecycle and accessors
 *
 * Phase B: Operates on the LirOperand C struct defined in lir_c_api.h.
 * These functions will eventually replace the C++ OperandBase/Operand/
 * LinkedOperand class hierarchy.
 *
 * For now, these coexist with operand.cpp. The lir_c_api accessor
 * functions still route through C++ (casting void* to OperandBase*).
 * When the swap happens, lir_c_api will route here instead.
 */

#include "cinderx/Jit/lir/lir_c_api.h"

#include "Python.h"

#include <string.h>

/* ---- Lifecycle ---- */

LirOperand *
lir_operand_create(LirInstruction *parent) {
    LirOperand *op = (LirOperand *)PyMem_RawCalloc(1, sizeof(LirOperand));
    op->parent_instr = parent;
    op->type = JIT_LIR_OPTYPE_NONE;
    op->data_type = JIT_LIR_DT_OBJECT;
    return op;
}

LirOperand *
lir_operand_create_linked(LirInstruction *parent, LirOperand *def_opnd) {
    LirOperand *op = (LirOperand *)PyMem_RawCalloc(1, sizeof(LirOperand));
    op->parent_instr = parent;
    op->is_linked = 1;
    op->value.def_opnd = def_opnd;
    return op;
}

void
lir_operand_free(LirOperand *op) {
    if (op == NULL) return;
    if (!op->is_linked && op->type == JIT_LIR_OPTYPE_IND) {
        lir_memind_free(op->value.indirect);
    }
    PyMem_RawFree(op);
}

/* ---- MemoryIndirect lifecycle ---- */

LirMemoryIndirect *
lir_memind_create(LirInstruction *parent) {
    LirMemoryIndirect *ind = (LirMemoryIndirect *)PyMem_RawCalloc(
        1, sizeof(LirMemoryIndirect));
    ind->parent = parent;
    return ind;
}

void
lir_memind_free(LirMemoryIndirect *ind) {
    if (ind == NULL) return;
    lir_operand_free(ind->base_reg);
    lir_operand_free(ind->index_reg);
    PyMem_RawFree(ind);
}

/* ---- Delegation helper ---- */

static const LirOperand *
resolve(const LirOperand *op) {
    while (op->is_linked && op->value.def_opnd != NULL) {
        op = op->value.def_opnd;
    }
    return op;
}

/* ---- Getters (delegate through linked operand) ---- */

int
lir_operand_type(const LirOperand *op) {
    return (int)resolve(op)->type;
}

int
lir_operand_data_type(const LirOperand *op) {
    return (int)resolve(op)->data_type;
}

int
lir_operand_is_linked(const LirOperand *op) {
    return op->is_linked;
}

int
lir_operand_is_last_use(const LirOperand *op) {
    return op->last_use;
}

int
lir_operand_is_fp(const LirOperand *op) {
    return resolve(op)->data_type == JIT_LIR_DT_DOUBLE;
}

uint64_t
lir_operand_get_constant(const LirOperand *op) {
    return resolve(op)->value.imm;
}

double
lir_operand_get_fp_constant(const LirOperand *op) {
    const LirOperand *r = resolve(op);
    double d;
    memcpy(&d, &r->value.imm, sizeof(double));
    return d;
}

LirPhyLocation
lir_operand_get_phy_register(const LirOperand *op) {
    return resolve(op)->value.phy_loc;
}

LirPhyLocation
lir_operand_get_stack_slot(const LirOperand *op) {
    return resolve(op)->value.phy_loc;
}

void *
lir_operand_get_mem_address(const LirOperand *op) {
    return resolve(op)->value.mem_addr;
}

void *
lir_operand_get_basic_block(const LirOperand *op) {
    return resolve(op)->value.label;
}

LirMemoryIndirect *
lir_operand_get_indirect(const LirOperand *op) {
    return resolve(op)->value.indirect;
}

LirOperand *
lir_operand_get_define(LirOperand *op) {
    if (op->is_linked) {
        return op->value.def_opnd;
    }
    return op;
}

/* ---- Setters ---- */

void
lir_operand_set_constant(LirOperand *op, uint64_t val, int dt) {
    op->type = JIT_LIR_OPTYPE_IMM;
    op->data_type = (uint8_t)dt;
    op->value.imm = val;
}

void
lir_operand_set_fp_constant(LirOperand *op, double val) {
    op->type = JIT_LIR_OPTYPE_IMM;
    op->data_type = JIT_LIR_DT_DOUBLE;
    memcpy(&op->value.imm, &val, sizeof(double));
}

void
lir_operand_set_phy_register(LirOperand *op, int loc) {
    op->type = JIT_LIR_OPTYPE_REG;
    op->value.phy_loc.loc = loc;
}

void
lir_operand_set_stack_slot(LirOperand *op, int loc) {
    op->type = JIT_LIR_OPTYPE_STACK;
    op->value.phy_loc.loc = loc;
}

void
lir_operand_set_virtual_register(LirOperand *op) {
    op->type = JIT_LIR_OPTYPE_VREG;
}

void
lir_operand_set_data_type(LirOperand *op, int dt) {
    op->data_type = (uint8_t)dt;
}

void
lir_operand_set_basic_block(LirOperand *op, void *block) {
    op->type = JIT_LIR_OPTYPE_LABEL;
    op->value.label = block;
}

void
lir_operand_set_mem_address(LirOperand *op, void *addr) {
    op->type = JIT_LIR_OPTYPE_MEM;
    op->value.mem_addr = addr;
}

void
lir_operand_set_last_use(LirOperand *op) {
    op->last_use = 1;
}

void
lir_operand_set_none(LirOperand *op) {
    op->type = JIT_LIR_OPTYPE_NONE;
}

/* ---- MemoryIndirect accessors ---- */

LirOperand *
lir_memind_base_reg(const LirMemoryIndirect *ind) {
    return ind->base_reg;
}

LirOperand *
lir_memind_index_reg(const LirMemoryIndirect *ind) {
    return ind->index_reg;
}

uint8_t
lir_memind_multiplier(const LirMemoryIndirect *ind) {
    return ind->multiplier;
}

int32_t
lir_memind_offset(const LirMemoryIndirect *ind) {
    return ind->offset;
}
