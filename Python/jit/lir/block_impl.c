/*
 * block_impl.c -- C implementation of LIR basic block operations
 *
 * Phase B1: Pure C functions operating on LirBasicBlock struct
 * defined in lir_types_c.h. Coexists with block.cpp — complex mutation
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
    bb->id_ = id;
    bb->func_ = (LirFunction *)function;
    bb->succs_capacity_ = 4;
    bb->successors_ = (LirBasicBlock **)PyMem_RawCalloc(
        bb->succs_capacity_, sizeof(LirBasicBlock *));
    bb->preds_capacity_ = 4;
    bb->predecessors_ = (LirBasicBlock **)PyMem_RawCalloc(
        bb->preds_capacity_, sizeof(LirBasicBlock *));
    return bb;
}

void
lir_block_free(LirBasicBlock *bb) {
    if (bb == NULL) return;
    /* Free all owned instructions */
    LirInstruction *cur = bb->instr_head_;
    while (cur != NULL) {
        LirInstruction *next = cur->next_;
        lir_instruction_free(cur);
        cur = next;
    }
    PyMem_RawFree(bb->successors_);
    PyMem_RawFree(bb->predecessors_);
    PyMem_RawFree(bb);
}

/* ---- Getters ---- */

int lir_block_id(const LirBasicBlock *bb) { return bb->id_; }
void lir_block_set_id(LirBasicBlock *bb, int id) { bb->id_ = id; }

void *lir_block_function(const LirBasicBlock *bb) { return bb->func_; }

int lir_block_section(const LirBasicBlock *bb) { return bb->section_; }
void lir_block_set_section(LirBasicBlock *bb, int section) {
    bb->section_ = (LirCodeSection)section;
}

size_t lir_block_num_succs(const LirBasicBlock *bb) { return bb->num_succs_; }
LirBasicBlock *lir_block_get_succ(const LirBasicBlock *bb, size_t i) {
    assert(i < bb->num_succs_);
    return bb->successors_[i];
}

size_t lir_block_num_preds(const LirBasicBlock *bb) { return bb->num_preds_; }
LirBasicBlock *lir_block_get_pred(const LirBasicBlock *bb, size_t i) {
    assert(i < bb->num_preds_);
    return bb->predecessors_[i];
}

LirBasicBlock *lir_block_true_successor(const LirBasicBlock *bb) {
    assert(bb->num_succs_ >= 1);
    return bb->successors_[0];
}

LirBasicBlock *lir_block_false_successor(const LirBasicBlock *bb) {
    assert(bb->num_succs_ >= 2);
    return bb->successors_[1];
}

size_t lir_block_num_instrs(const LirBasicBlock *bb) { return bb->num_instrs_; }

int lir_block_is_empty(const LirBasicBlock *bb) {
    return bb->instr_head_ == NULL;
}

LirInstruction *lir_block_first_instr(const LirBasicBlock *bb) {
    return bb->instr_head_;
}

LirInstruction *lir_block_last_instr(const LirBasicBlock *bb) {
    return bb->instr_tail_;
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
    if (bb->num_succs_ >= bb->succs_capacity_) {
        grow_ptr_array((void ***)&bb->successors_, &bb->succs_capacity_);
    }
    bb->successors_[bb->num_succs_++] = succ;

    if (succ->num_preds_ >= succ->preds_capacity_) {
        grow_ptr_array((void ***)&succ->predecessors_, &succ->preds_capacity_);
    }
    succ->predecessors_[succ->num_preds_++] = bb;
}

void
lir_block_swap_successors(LirBasicBlock *bb) {
    if (bb->num_succs_ < 2) return;
    LirBasicBlock *tmp = bb->successors_[0];
    bb->successors_[0] = bb->successors_[1];
    bb->successors_[1] = tmp;
}

/* ---- Instruction list operations ---- */

void
lir_block_append_instr(LirBasicBlock *bb, LirInstruction *instr) {
    instr->basic_block_ = bb;
    instr->prev_ = bb->instr_tail_;
    instr->next_ = NULL;
    if (bb->instr_tail_ != NULL) {
        bb->instr_tail_->next_ = instr;
    } else {
        bb->instr_head_ = instr;
    }
    bb->instr_tail_ = instr;
    bb->num_instrs_++;
}

LirInstruction *
lir_block_remove_instr(LirBasicBlock *bb, LirInstruction *instr) {
    if (instr->prev_ != NULL) {
        instr->prev_->next_ = instr->next_;
    } else {
        bb->instr_head_ = instr->next_;
    }
    if (instr->next_ != NULL) {
        instr->next_->prev_ = instr->prev_;
    } else {
        bb->instr_tail_ = instr->prev_;
    }
    instr->prev_ = NULL;
    instr->next_ = NULL;
    bb->num_instrs_--;
    return instr;  /* caller takes ownership */
}

/* ---- Successor mutation ---- */

void
lir_block_set_successor(LirBasicBlock *bb, size_t index, LirBasicBlock *new_succ) {
    assert(index < bb->num_succs_);
    LirBasicBlock *old_succ = bb->successors_[index];

    /* Remove bb from old_succ's predecessors */
    for (size_t i = 0; i < old_succ->num_preds_; i++) {
        if (old_succ->predecessors_[i] == bb) {
            for (size_t j = i; j + 1 < old_succ->num_preds_; j++) {
                old_succ->predecessors_[j] = old_succ->predecessors_[j + 1];
            }
            old_succ->num_preds_--;
            break;
        }
    }

    /* Set new successor */
    bb->successors_[index] = new_succ;

    /* Add bb to new_succ's predecessors */
    if (new_succ->num_preds_ >= new_succ->preds_capacity_) {
        grow_ptr_array((void ***)&new_succ->predecessors_, &new_succ->preds_capacity_);
    }
    new_succ->predecessors_[new_succ->num_preds_++] = bb;
}

/* ---- Phi fixup ---- */

void
lir_block_fixup_phis(LirBasicBlock *bb,
                     LirBasicBlock *old_pred, LirBasicBlock *new_pred) {
    /* Iterate phi instructions at the start of the block */
    for (LirInstruction *instr = bb->instr_head_; instr != NULL; instr = instr->next_) {
        if (instr->opcode_ != JIT_LIR_OP_PHI) {
            continue;
        }
        for (size_t i = 0; i < instr->num_inputs_; i++) {
            LirOperand *op = instr->inputs_[i];
            if (op->type_ == JIT_LIR_OPTYPE_LABEL &&
                op->value_.block == old_pred) {
                op->value_.block = new_pred;
            }
        }
    }
}

/* ---- Instruction insertion ---- */

void
lir_block_insert_instr_before(LirBasicBlock *bb, LirInstruction *before,
                              LirInstruction *instr) {
    instr->basic_block_ = bb;
    instr->next_ = before;
    instr->prev_ = before->prev_;
    if (before->prev_ != NULL) {
        before->prev_->next_ = instr;
    } else {
        bb->instr_head_ = instr;
    }
    before->prev_ = instr;
    bb->num_instrs_++;
}
