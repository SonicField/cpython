/*
 * operand_impl.c -- C implementation of LIR operand operations
 *
 * Phase B1: Pure C functions operating on LirOperand/LirMemoryIndirect
 * structs. These coexist with the C++ operand.cpp — C callers use these,
 * C++ callers continue using the class methods.
 *
 * When all consumers are converted, operand.cpp is deleted and these
 * become the sole implementation.
 */

#include "cinderx/Jit/lir/lir_impl_internal.h"

#include "Python.h"

#include <assert.h>
#include <string.h>

/* ---- Operand lifecycle ---- */

LirOperand *
lir_operand_new(LirInstruction *parent) {
    LirOperand *op = (LirOperand *)PyMem_RawCalloc(1, sizeof(LirOperand));
    op->parent_instr = parent;
    op->type = JIT_LIR_OPTYPE_NONE;
    op->data_type = JIT_LIR_DT_OBJECT;
    return op;
}

LirOperand *
lir_operand_new_linked(LirInstruction *parent, LirInstruction *def_instr) {
    LirOperand *op = (LirOperand *)PyMem_RawCalloc(1, sizeof(LirOperand));
    op->parent_instr = parent;
    op->is_linked = 1;
    /* def_opnd points to the defining instruction's output operand.
     * We need jit_lir_instr_output() to get it, but that returns void*.
     * For now, store the instruction pointer — the caller must set
     * def_opnd correctly via lir_operand_set_linked_instr. */
    op->value.def_opnd = NULL;  /* caller sets via set_linked_instr */
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
lir_memind_new(LirInstruction *parent) {
    LirMemoryIndirect *mi = (LirMemoryIndirect *)PyMem_RawCalloc(
        1, sizeof(LirMemoryIndirect));
    mi->parent = parent;
    return mi;
}

void
lir_memind_free(LirMemoryIndirect *mi) {
    if (mi == NULL) return;
    lir_operand_free(mi->base_reg);
    lir_operand_free(mi->index_reg);
    PyMem_RawFree(mi);
}

/* ---- Helper: resolve linked operand ---- */

static const LirOperand *
resolve(const LirOperand *op) {
    if (op->is_linked && op->value.def_opnd != NULL) {
        return op->value.def_opnd;
    }
    return op;
}

/* ---- Operand getters ---- */

uint8_t
lir_operand_type(const LirOperand *op) {
    return resolve(op)->type;
}

uint8_t
lir_operand_data_type(const LirOperand *op) {
    return resolve(op)->data_type;
}

int
lir_operand_is_linked(const LirOperand *op) {
    return op->is_linked;
}

int
lir_operand_is_fp(const LirOperand *op) {
    return resolve(op)->data_type == JIT_LIR_DT_DOUBLE;
}

int
lir_operand_is_last_use(const LirOperand *op) {
    return op->last_use;
}

size_t
lir_operand_size_in_bits(const LirOperand *op) {
    return jit_lir_bit_size(resolve(op)->data_type);
}

LirInstruction *
lir_operand_instr(const LirOperand *op) {
    return op->parent_instr;
}

uint64_t
lir_operand_get_constant(const LirOperand *op) {
    const LirOperand *r = resolve(op);
    assert(r->type == JIT_LIR_OPTYPE_IMM);
    return r->value.imm;
}

double
lir_operand_get_fp_constant(const LirOperand *op) {
    uint64_t bits = lir_operand_get_constant(op);
    double d;
    memcpy(&d, &bits, sizeof(d));
    return d;
}

LirPhyLocation
lir_operand_get_phy_register(const LirOperand *op) {
    const LirOperand *r = resolve(op);
    assert(r->type == JIT_LIR_OPTYPE_REG);
    return r->value.phy_loc;
}

LirPhyLocation
lir_operand_get_stack_slot(const LirOperand *op) {
    const LirOperand *r = resolve(op);
    assert(r->type == JIT_LIR_OPTYPE_STACK);
    return r->value.phy_loc;
}

void *
lir_operand_get_mem_address(const LirOperand *op) {
    const LirOperand *r = resolve(op);
    assert(r->type == JIT_LIR_OPTYPE_MEM);
    return r->value.mem_addr;
}

void *
lir_operand_get_basic_block(const LirOperand *op) {
    const LirOperand *r = resolve(op);
    assert(r->type == JIT_LIR_OPTYPE_LABEL);
    return r->value.label;
}

LirMemoryIndirect *
lir_operand_get_indirect(const LirOperand *op) {
    const LirOperand *r = resolve(op);
    assert(r->type == JIT_LIR_OPTYPE_IND);
    return r->value.indirect;
}

LirOperand *
lir_operand_get_define(LirOperand *op) {
    if (op->is_linked && op->value.def_opnd != NULL) {
        return op->value.def_opnd;
    }
    return op;
}

uint64_t
lir_operand_get_constant_or_address(const LirOperand *op) {
    const LirOperand *r = resolve(op);
    if (r->type == JIT_LIR_OPTYPE_IMM) {
        return r->value.imm;
    }
    return (uint64_t)(uintptr_t)r->value.mem_addr;
}

LirInstruction *
lir_operand_get_linked_instr(const LirOperand *op) {
    assert(op->is_linked);
    if (op->value.def_opnd != NULL) {
        return op->value.def_opnd->parent_instr;
    }
    return NULL;
}

/* ---- Operand setters ---- */

void
lir_operand_set_constant(LirOperand *op, uint64_t val, uint8_t dt) {
    assert(!op->is_linked);
    op->type = JIT_LIR_OPTYPE_IMM;
    op->data_type = dt;
    op->value.imm = val;
}

void
lir_operand_set_fp_constant(LirOperand *op, double val) {
    assert(!op->is_linked);
    op->type = JIT_LIR_OPTYPE_IMM;
    op->data_type = JIT_LIR_DT_DOUBLE;
    memcpy(&op->value.imm, &val, sizeof(val));
}

void
lir_operand_set_phy_register(LirOperand *op, LirPhyLocation reg) {
    assert(!op->is_linked);
    op->type = JIT_LIR_OPTYPE_REG;
    op->value.phy_loc = reg;
}

void
lir_operand_set_stack_slot(LirOperand *op, LirPhyLocation slot) {
    assert(!op->is_linked);
    op->type = JIT_LIR_OPTYPE_STACK;
    op->value.phy_loc = slot;
}

void
lir_operand_set_mem_address(LirOperand *op, void *addr) {
    assert(!op->is_linked);
    op->type = JIT_LIR_OPTYPE_MEM;
    op->value.mem_addr = addr;
}

void
lir_operand_set_basic_block(LirOperand *op, void *block) {
    assert(!op->is_linked);
    op->type = JIT_LIR_OPTYPE_LABEL;
    op->data_type = JIT_LIR_DT_OBJECT;
    op->value.label = block;
}

void
lir_operand_set_virtual_register(LirOperand *op) {
    assert(!op->is_linked);
    op->type = JIT_LIR_OPTYPE_VREG;
}

void
lir_operand_set_data_type(LirOperand *op, uint8_t dt) {
    assert(!op->is_linked);
    op->data_type = dt;
    if (op->type == JIT_LIR_OPTYPE_REG || op->type == JIT_LIR_OPTYPE_STACK) {
        op->value.phy_loc.bit_size = jit_lir_bit_size(dt);
    }
}

void
lir_operand_set_last_use(LirOperand *op) {
    op->last_use = 1;
}

void
lir_operand_set_none(LirOperand *op) {
    assert(!op->is_linked);
    op->type = JIT_LIR_OPTYPE_NONE;
}

void
lir_operand_set_linked_instr(LirOperand *op, LirInstruction *def) {
    assert(op->is_linked);
    /* def_opnd points to the defining instruction's output operand.
     * For now, use the C API to get it. */
    op->value.def_opnd = (LirOperand *)jit_lir_instr_output(def);
}

/* ---- MemoryIndirect setters ---- */

void
lir_memind_set(LirMemoryIndirect *mi,
               LirPhyLocation base, LirPhyLocation index,
               uint8_t multiplier, int32_t offset) {
    /* Free old operands if any */
    lir_operand_free(mi->base_reg);
    lir_operand_free(mi->index_reg);

    mi->multiplier = multiplier;
    mi->offset = offset;

    if (base.loc != LIR_REG_INVALID) {
        mi->base_reg = lir_operand_new(mi->parent);
        lir_operand_set_phy_register(mi->base_reg, base);
    } else {
        mi->base_reg = NULL;
    }

    if (index.loc != LIR_REG_INVALID) {
        mi->index_reg = lir_operand_new(mi->parent);
        lir_operand_set_phy_register(mi->index_reg, index);
    } else {
        mi->index_reg = NULL;
    }
}

void
lir_memind_set_linked(LirMemoryIndirect *mi,
                      LirInstruction *base, LirInstruction *index,
                      uint8_t multiplier, int32_t offset) {
    lir_operand_free(mi->base_reg);
    lir_operand_free(mi->index_reg);

    mi->multiplier = multiplier;
    mi->offset = offset;

    if (base != NULL) {
        mi->base_reg = lir_operand_new_linked(mi->parent, base);
        lir_operand_set_linked_instr(mi->base_reg, base);
    } else {
        mi->base_reg = NULL;
    }

    if (index != NULL) {
        mi->index_reg = lir_operand_new_linked(mi->parent, index);
        lir_operand_set_linked_instr(mi->index_reg, index);
    } else {
        mi->index_reg = NULL;
    }
}
