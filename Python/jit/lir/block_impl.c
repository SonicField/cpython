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
lir_block_destroy(LirBasicBlock *bb) {
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
    /* Does NOT free bb itself — caller handles that */
}

void
lir_block_free(LirBasicBlock *bb) {
    if (bb == NULL) return;
    lir_block_destroy(bb);
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
    size_t new_cap = (*cap == 0) ? 2 : (*cap) * 2;
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

/* ---- Block splitting ---- */

LirBasicBlock *
lir_block_insert_between(LirBasicBlock *bb, LirBasicBlock *succ_block) {
    /* Find succ_block in bb's successors */
    size_t idx = bb->num_succs_;
    for (size_t i = 0; i < bb->num_succs_; i++) {
        if (bb->successors_[i] == succ_block) {
            idx = i;
            break;
        }
    }
    assert(idx < bb->num_succs_ && "succ_block must be a successor of bb");

    /* Allocate new block after bb */
    LirBasicBlock *new_block = lir_function_alloc_block_after(bb->func_, bb);

    /* Replace successor: bb -> new_block instead of bb -> succ_block */
    bb->successors_[idx] = new_block;

    /* Add bb as predecessor of new_block */
    if (new_block->num_preds_ >= new_block->preds_capacity_) {
        grow_ptr_array((void ***)&new_block->predecessors_, &new_block->preds_capacity_);
    }
    new_block->predecessors_[new_block->num_preds_++] = bb;

    /* Remove bb from succ_block's predecessors */
    for (size_t i = 0; i < succ_block->num_preds_; i++) {
        if (succ_block->predecessors_[i] == bb) {
            for (size_t j = i; j + 1 < succ_block->num_preds_; j++) {
                succ_block->predecessors_[j] = succ_block->predecessors_[j + 1];
            }
            succ_block->num_preds_--;
            break;
        }
    }

    /* new_block -> succ_block */
    lir_block_add_successor(new_block, succ_block);

    return new_block;
}

LirBasicBlock *
lir_block_split_before(LirBasicBlock *bb, LirInstruction *instr) {
    assert(bb->func_ != NULL && "cannot split block without function");
    assert(instr->opcode_ != JIT_LIR_OP_PHI && "cannot split at phi");

    /* Verify instruction is in this block */
    int found = 0;
    for (LirInstruction *i = bb->instr_head_; i != NULL; i = i->next_) {
        if (i == instr) { found = 1; break; }
    }
    if (!found) return NULL;

    /* Allocate second block after this one */
    LirBasicBlock *second = lir_function_alloc_block_after(bb->func_, bb);

    /* Move all instructions from instr onward to second block */
    LirInstruction *cur = instr;
    while (cur != NULL) {
        LirInstruction *next = cur->next_;
        lir_block_remove_instr(bb, cur);
        cur->basic_block_ = second;
        lir_block_append_instr(second, cur);
        cur = next;
    }

    /* Fix up successors: move bb's successors to second */
    for (size_t i = 0; i < bb->num_succs_; i++) {
        LirBasicBlock *succ = bb->successors_[i];
        /* Fix phis in successor */
        lir_block_fixup_phis(succ, bb, second);
        /* Add succ as successor of second */
        if (second->num_succs_ >= second->succs_capacity_) {
            grow_ptr_array((void ***)&second->successors_, &second->succs_capacity_);
        }
        second->successors_[second->num_succs_++] = succ;
        /* Replace bb with second in succ's predecessors */
        for (size_t j = 0; j < succ->num_preds_; j++) {
            if (succ->predecessors_[j] == bb) {
                succ->predecessors_[j] = second;
            }
        }
    }

    /* Clear bb's successors and add second as sole successor */
    bb->num_succs_ = 0;
    lir_block_add_successor(bb, second);

    return second;
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

/*
 * Allocate a new instruction and insert it before 'before' in the block.
 * If before is NULL, appends to the end. This is the C equivalent of
 * BasicBlock::allocateInstrBefore(pos, opcode) with no variadic args.
 */
LirInstruction *
lir_block_alloc_instr_before(LirBasicBlock *bb, LirInstruction *before,
                              int opcode) {
    const void *origin = NULL;
    if (before != NULL) {
        origin = before->origin_;
    } else if (bb->instr_tail_ != NULL) {
        origin = bb->instr_tail_->origin_;
    }
    LirInstruction *instr = lir_instruction_create(bb, opcode, origin);
    instr->id_ = lir_function_allocate_id(bb->func_);
    if (before != NULL) {
        lir_block_insert_instr_before(bb, before, instr);
    } else {
        lir_block_append_instr(bb, instr);
    }
    return instr;
}

/**
 * Allocate a new instruction and append it at the end of the block.
 * Uses the given origin (may be NULL).
 */
LirInstruction *
lir_block_alloc_instr(LirBasicBlock *bb, int opcode, const void *origin) {
    LirInstruction *instr = lir_instruction_create(bb, opcode, origin);
    instr->id_ = lir_function_allocate_id(bb->func_);
    lir_block_append_instr(bb, instr);
    return instr;
}
