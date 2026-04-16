/*
 * function_impl.c -- C implementation of LIR Function operations
 *
 * Phase B1: Pure C functions operating on LirFunction struct.
 * Coexists with function.cpp — copyFrom (deep copy for inlining)
 * remains in C++ for now due to UnorderedMap + LinkedOperand complexity.
 */

#include "cinderx/Jit/lir/lir_impl_internal.h"

#include "Python.h"

#include <assert.h>

#define INITIAL_BLOCK_CAPACITY 16

/* ---- Lifecycle ---- */

LirFunction *
lir_function_new(const void *hir_func) {
    LirFunction *func = (LirFunction *)PyMem_RawCalloc(
        1, sizeof(LirFunction));
    func->hir_func_ = hir_func;
    func->blocks_capacity_ = INITIAL_BLOCK_CAPACITY;
    func->blocks_ = (LirBasicBlock **)PyMem_RawCalloc(
        func->blocks_capacity_, sizeof(LirBasicBlock *));
    func->num_blocks_ = 0;
    func->next_id_ = 0;
    return func;
}

void
lir_function_destroy(LirFunction *func) {
    if (func == NULL) return;
    /* Free all owned blocks (which recursively free instructions) */
    for (size_t i = 0; i < func->num_blocks_; i++) {
        lir_block_free(func->blocks_[i]);
    }
    PyMem_RawFree(func->blocks_);
    /* Does NOT free func itself — caller handles that */
}

void
lir_function_free(LirFunction *func) {
    if (func == NULL) return;
    lir_function_destroy(func);
    PyMem_RawFree(func);
}

/* ---- ID allocation ---- */

int
lir_function_allocate_id(LirFunction *func) {
    return func->next_id_++;
}

void
lir_function_set_next_id(LirFunction *func, int id) {
    func->next_id_ = id;
}

/* ---- Block management ---- */

void
lir_function_ensure_block_capacity(LirFunction *func, size_t needed) {
    if (needed <= func->blocks_capacity_) return;
    size_t new_cap = func->blocks_capacity_ == 0 ? 8 : func->blocks_capacity_ * 2;
    while (new_cap < needed) new_cap *= 2;
    if (func->blocks_ == NULL) {
        func->blocks_ = (LirBasicBlock **)PyMem_RawMalloc(
            new_cap * sizeof(LirBasicBlock *));
    } else {
        func->blocks_ = (LirBasicBlock **)PyMem_RawRealloc(
            func->blocks_, new_cap * sizeof(LirBasicBlock *));
    }
    func->blocks_capacity_ = new_cap;
}

static void
ensure_block_capacity(LirFunction *func) {
    lir_function_ensure_block_capacity(func, func->num_blocks_ + 1);
}

LirBasicBlock *
lir_function_alloc_block(LirFunction *func) {
    int id = lir_function_allocate_id(func);
    LirBasicBlock *bb = lir_block_new(func, id);
    ensure_block_capacity(func);
    func->blocks_[func->num_blocks_++] = bb;
    return bb;
}

LirBasicBlock *
lir_function_alloc_block_after(LirFunction *func, LirBasicBlock *after) {
    int id = lir_function_allocate_id(func);
    LirBasicBlock *bb = lir_block_new(func, id);
    ensure_block_capacity(func);

    /* Find the position of 'after' and insert the new block after it */
    size_t pos = func->num_blocks_; /* default: append */
    for (size_t i = 0; i < func->num_blocks_; i++) {
        if (func->blocks_[i] == after) {
            pos = i + 1;
            break;
        }
    }

    /* Shift blocks right to make room */
    for (size_t i = func->num_blocks_; i > pos; i--) {
        func->blocks_[i] = func->blocks_[i - 1];
    }
    func->blocks_[pos] = bb;
    func->num_blocks_++;
    return bb;
}

/* ---- Accessors ---- */

size_t
lir_function_num_blocks(const LirFunction *func) {
    return func->num_blocks_;
}

LirBasicBlock *
lir_function_get_block(const LirFunction *func, size_t index) {
    assert(index < func->num_blocks_);
    return func->blocks_[index];
}

LirBasicBlock *
lir_function_entry_block(const LirFunction *func) {
    if (func->num_blocks_ == 0) return NULL;
    return func->blocks_[0];
}

const void *
lir_function_hir_func(const LirFunction *func) {
    return func->hir_func_;
}

/* ---- Block sorting ---- */

void
lir_function_sort_blocks(LirFunction *func) {
    size_t out_count = 0;
    JitLirBlock *sorted = jit_lir_sort_blocks_rpo(
        (JitLirBlock *)func->blocks_, func->num_blocks_, &out_count);
    if (sorted != NULL) {
        /* Replace the block array with the sorted result */
        if (out_count > func->blocks_capacity_) {
            func->blocks_ = (LirBasicBlock **)PyMem_RawRealloc(
                func->blocks_, out_count * sizeof(LirBasicBlock *));
            func->blocks_capacity_ = out_count;
        }
        for (size_t i = 0; i < out_count; i++) {
            func->blocks_[i] = (LirBasicBlock *)sorted[i];
        }
        func->num_blocks_ = out_count;
        PyMem_RawFree(sorted);
    }
}
