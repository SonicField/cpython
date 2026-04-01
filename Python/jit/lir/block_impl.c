/*
 * block_impl.c -- C implementation of LIR basic block operations
 *
 * Phase B1: Pure C functions operating on LirBasicBlock struct
 * defined in lir_c_api.h. Coexists with block.cpp — complex mutation
 * methods (splitBefore, insertBasicBlockBetween) remain in C++ until
 * Function is converted.
 */

#include "cinderx/Jit/lir/lir_impl_internal.h"

#include "Python.h"

#include <assert.h>

/* ---- Lifecycle ---- */

LirBasicBlock *
lir_block_new(void *function, int id) {
    LirBasicBlock *bb = (LirBasicBlock *)PyMem_RawCalloc(
        1, sizeof(LirBasicBlock));
    bb->id = id;
    bb->function = function;
    bb->succs_capacity = 4;
    bb->successors = (LirBasicBlock **)PyMem_RawCalloc(
        bb->succs_capacity, sizeof(LirBasicBlock *));
    bb->preds_capacity = 4;
    bb->predecessors = (LirBasicBlock **)PyMem_RawCalloc(
        bb->preds_capacity, sizeof(LirBasicBlock *));
    return bb;
}

void
lir_block_free(LirBasicBlock *bb) {
    if (bb == NULL) return;
    /* Free all owned instructions */
    LirInstruction *cur = bb->instr_head;
    while (cur != NULL) {
        LirInstruction *next = cur->next;
        lir_instruction_free(cur);
        cur = next;
    }
    PyMem_RawFree(bb->successors);
    PyMem_RawFree(bb->predecessors);
    PyMem_RawFree(bb);
}

/* ---- Getters ---- */

int lir_block_id(const LirBasicBlock *bb) { return bb->id; }
void lir_block_set_id(LirBasicBlock *bb, int id) { bb->id = id; }

void *lir_block_function(const LirBasicBlock *bb) { return bb->function; }

int lir_block_section(const LirBasicBlock *bb) { return bb->section; }
void lir_block_set_section(LirBasicBlock *bb, int section) {
    bb->section = section;
}

size_t lir_block_num_succs(const LirBasicBlock *bb) { return bb->num_succs; }
LirBasicBlock *lir_block_get_succ(const LirBasicBlock *bb, size_t i) {
    assert(i < bb->num_succs);
    return bb->successors[i];
}

size_t lir_block_num_preds(const LirBasicBlock *bb) { return bb->num_preds; }
LirBasicBlock *lir_block_get_pred(const LirBasicBlock *bb, size_t i) {
    assert(i < bb->num_preds);
    return bb->predecessors[i];
}

LirBasicBlock *lir_block_true_successor(const LirBasicBlock *bb) {
    assert(bb->num_succs >= 1);
    return bb->successors[0];
}

LirBasicBlock *lir_block_false_successor(const LirBasicBlock *bb) {
    assert(bb->num_succs >= 2);
    return bb->successors[1];
}

size_t lir_block_num_instrs(const LirBasicBlock *bb) { return bb->num_instrs; }

int lir_block_is_empty(const LirBasicBlock *bb) {
    return bb->instr_head == NULL;
}

LirInstruction *lir_block_first_instr(const LirBasicBlock *bb) {
    return bb->instr_head;
}

LirInstruction *lir_block_last_instr(const LirBasicBlock *bb) {
    return bb->instr_tail;
}

/* ---- Successor/predecessor mutation ---- */

static void
grow_ptr_array(void ***arr, size_t *cap) {
    size_t new_cap = (*cap) * 2;
    *arr = (void **)PyMem_RawRealloc(*arr, new_cap * sizeof(void *));
    *cap = new_cap;
}

void
lir_block_add_successor(LirBasicBlock *bb, LirBasicBlock *succ) {
    if (bb->num_succs >= bb->succs_capacity) {
        grow_ptr_array((void ***)&bb->successors, &bb->succs_capacity);
    }
    bb->successors[bb->num_succs++] = succ;

    if (succ->num_preds >= succ->preds_capacity) {
        grow_ptr_array((void ***)&succ->predecessors, &succ->preds_capacity);
    }
    succ->predecessors[succ->num_preds++] = bb;
}

void
lir_block_swap_successors(LirBasicBlock *bb) {
    if (bb->num_succs < 2) return;
    LirBasicBlock *tmp = bb->successors[0];
    bb->successors[0] = bb->successors[1];
    bb->successors[1] = tmp;
}

/* ---- Instruction list operations ---- */

void
lir_block_append_instr(LirBasicBlock *bb, LirInstruction *instr) {
    instr->basic_block = bb;
    instr->prev = bb->instr_tail;
    instr->next = NULL;
    if (bb->instr_tail != NULL) {
        bb->instr_tail->next = instr;
    } else {
        bb->instr_head = instr;
    }
    bb->instr_tail = instr;
    bb->num_instrs++;
}

LirInstruction *
lir_block_remove_instr(LirBasicBlock *bb, LirInstruction *instr) {
    if (instr->prev != NULL) {
        instr->prev->next = instr->next;
    } else {
        bb->instr_head = instr->next;
    }
    if (instr->next != NULL) {
        instr->next->prev = instr->prev;
    } else {
        bb->instr_tail = instr->prev;
    }
    instr->prev = NULL;
    instr->next = NULL;
    bb->num_instrs--;
    return instr;  /* caller takes ownership */
}

void
lir_block_insert_instr_before(LirBasicBlock *bb, LirInstruction *before,
                              LirInstruction *instr) {
    instr->basic_block = bb;
    instr->next = before;
    instr->prev = before->prev;
    if (before->prev != NULL) {
        before->prev->next = instr;
    } else {
        bb->instr_head = instr;
    }
    before->prev = instr;
    bb->num_instrs++;
}
