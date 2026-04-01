/*
 * lir_block.c -- C implementation of LIR BasicBlock
 *
 * Phase B: Operates on the LirBasicBlock C struct defined in lir_c_api.h.
 * Coexists with block.cpp until all consumers use the C API.
 */

#include "cinderx/Jit/lir/lir_c_api.h"

#include "Python.h"

#define INITIAL_EDGE_CAPACITY 4

/* ---- Lifecycle ---- */

LirBasicBlock *
lir_block_create(void *function, int id) {
    LirBasicBlock *bb = (LirBasicBlock *)PyMem_RawCalloc(
        1, sizeof(LirBasicBlock));
    bb->id = id;
    bb->function = function;
    bb->successors = (LirBasicBlock **)PyMem_RawMalloc(
        INITIAL_EDGE_CAPACITY * sizeof(LirBasicBlock *));
    bb->succs_capacity = INITIAL_EDGE_CAPACITY;
    bb->predecessors = (LirBasicBlock **)PyMem_RawMalloc(
        INITIAL_EDGE_CAPACITY * sizeof(LirBasicBlock *));
    bb->preds_capacity = INITIAL_EDGE_CAPACITY;
    return bb;
}

void
lir_block_free(LirBasicBlock *bb) {
    if (bb == NULL) return;
    /* Free all owned instructions */
    LirInstruction *inst = bb->instr_head;
    while (inst != NULL) {
        LirInstruction *next = inst->next;
        lir_instruction_free(inst);
        inst = next;
    }
    PyMem_RawFree(bb->successors);
    PyMem_RawFree(bb->predecessors);
    PyMem_RawFree(bb);
}

/* ---- Accessors ---- */

int lir_block_id(const LirBasicBlock *bb) { return bb->id; }
int lir_block_section(const LirBasicBlock *bb) { return bb->section; }
void lir_block_set_section(LirBasicBlock *bb, int s) { bb->section = s; }
size_t lir_block_num_instrs(const LirBasicBlock *bb) { return bb->num_instrs; }

/* ---- Successor/predecessor management ---- */

static void
ensure_edge_cap(LirBasicBlock ***arr, size_t *cap, size_t count) {
    if (count >= *cap) {
        size_t new_cap = (*cap) * 2;
        *arr = (LirBasicBlock **)PyMem_RawRealloc(
            *arr, new_cap * sizeof(LirBasicBlock *));
        *cap = new_cap;
    }
}

void
lir_block_add_successor(LirBasicBlock *bb, LirBasicBlock *succ) {
    ensure_edge_cap(&bb->successors, &bb->succs_capacity, bb->num_succs);
    bb->successors[bb->num_succs++] = succ;
    /* Add bb as predecessor of succ */
    ensure_edge_cap(&succ->predecessors, &succ->preds_capacity, succ->num_preds);
    succ->predecessors[succ->num_preds++] = bb;
}

size_t lir_block_num_succs(const LirBasicBlock *bb) { return bb->num_succs; }
LirBasicBlock *lir_block_succ_at(const LirBasicBlock *bb, size_t i) {
    return bb->successors[i];
}

size_t lir_block_num_preds(const LirBasicBlock *bb) { return bb->num_preds; }
LirBasicBlock *lir_block_pred_at(const LirBasicBlock *bb, size_t i) {
    return bb->predecessors[i];
}

LirBasicBlock *
lir_block_false_successor(const LirBasicBlock *bb) {
    return (bb->num_succs >= 2) ? bb->successors[1] : NULL;
}

/* ---- Instruction list operations ---- */

LirInstruction *
lir_block_first_instr(const LirBasicBlock *bb) {
    return bb->instr_head;
}

LirInstruction *
lir_block_last_instr(const LirBasicBlock *bb) {
    return bb->instr_tail;
}

void
lir_block_append_instr(LirBasicBlock *bb, LirInstruction *inst) {
    inst->basic_block = bb;
    inst->prev = bb->instr_tail;
    inst->next = NULL;
    if (bb->instr_tail) {
        bb->instr_tail->next = inst;
    } else {
        bb->instr_head = inst;
    }
    bb->instr_tail = inst;
    bb->num_instrs++;
}

void
lir_block_remove_instr(LirBasicBlock *bb, LirInstruction *inst) {
    if (inst->prev) {
        inst->prev->next = inst->next;
    } else {
        bb->instr_head = inst->next;
    }
    if (inst->next) {
        inst->next->prev = inst->prev;
    } else {
        bb->instr_tail = inst->prev;
    }
    inst->prev = NULL;
    inst->next = NULL;
    inst->basic_block = NULL;
    bb->num_instrs--;
}

/* Allocate a new instruction and append it to this block. */
LirInstruction *
lir_block_alloc_instr(LirBasicBlock *bb, int opcode, const void *origin) {
    LirInstruction *inst = lir_instruction_create(bb, opcode, origin);
    lir_block_append_instr(bb, inst);
    return inst;
}

/* Iterate instructions, calling cb for each. */
void
lir_block_foreach_instr(const LirBasicBlock *bb,
                         void (*cb)(LirInstruction *, void *),
                         void *ctx) {
    LirInstruction *inst = bb->instr_head;
    while (inst != NULL) {
        LirInstruction *next = inst->next;  /* cb may remove inst */
        cb(inst, ctx);
        inst = next;
    }
}

/* Remove instructions for which is_live returns 0. */
void
lir_block_remove_dead(LirBasicBlock *bb,
                       int (*is_live)(LirInstruction *, void *),
                       void *ctx) {
    LirInstruction *inst = bb->instr_head;
    while (inst != NULL) {
        LirInstruction *next = inst->next;
        if (!is_live(inst, ctx)) {
            lir_block_remove_instr(bb, inst);
            lir_instruction_free(inst);
        }
        inst = next;
    }
}
