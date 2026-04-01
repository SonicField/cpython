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
    func->hir_func = hir_func;
    func->blocks_capacity = INITIAL_BLOCK_CAPACITY;
    func->blocks = (LirBasicBlock **)PyMem_RawMalloc(
        func->blocks_capacity * sizeof(LirBasicBlock *));
    func->num_blocks = 0;
    func->next_id = 0;
    return func;
}

void
lir_function_free(LirFunction *func) {
    if (func == NULL) return;
    /* Free all owned blocks (which recursively free instructions) */
    for (size_t i = 0; i < func->num_blocks; i++) {
        lir_block_free(func->blocks[i]);
    }
    PyMem_RawFree(func->blocks);
    PyMem_RawFree(func);
}

/* ---- ID allocation ---- */

int
lir_function_allocate_id(LirFunction *func) {
    return func->next_id++;
}

void
lir_function_set_next_id(LirFunction *func, int id) {
    func->next_id = id;
}

/* ---- Block management ---- */

static void
ensure_block_capacity(LirFunction *func) {
    if (func->num_blocks >= func->blocks_capacity) {
        size_t new_cap = func->blocks_capacity * 2;
        func->blocks = (LirBasicBlock **)PyMem_RawRealloc(
            func->blocks, new_cap * sizeof(LirBasicBlock *));
        func->blocks_capacity = new_cap;
    }
}

LirBasicBlock *
lir_function_alloc_block(LirFunction *func) {
    int id = lir_function_allocate_id(func);
    LirBasicBlock *bb = lir_block_new(func, id);
    ensure_block_capacity(func);
    func->blocks[func->num_blocks++] = bb;
    return bb;
}

LirBasicBlock *
lir_function_alloc_block_after(LirFunction *func, LirBasicBlock *after) {
    int id = lir_function_allocate_id(func);
    LirBasicBlock *bb = lir_block_new(func, id);
    ensure_block_capacity(func);

    /* Find the position of 'after' and insert the new block after it */
    size_t pos = func->num_blocks; /* default: append */
    for (size_t i = 0; i < func->num_blocks; i++) {
        if (func->blocks[i] == after) {
            pos = i + 1;
            break;
        }
    }

    /* Shift blocks right to make room */
    for (size_t i = func->num_blocks; i > pos; i--) {
        func->blocks[i] = func->blocks[i - 1];
    }
    func->blocks[pos] = bb;
    func->num_blocks++;
    return bb;
}

/* ---- Accessors ---- */

size_t
lir_function_num_blocks(const LirFunction *func) {
    return func->num_blocks;
}

LirBasicBlock *
lir_function_get_block(const LirFunction *func, size_t index) {
    assert(index < func->num_blocks);
    return func->blocks[index];
}

LirBasicBlock *
lir_function_entry_block(const LirFunction *func) {
    if (func->num_blocks == 0) return NULL;
    return func->blocks[0];
}

const void *
lir_function_hir_func(const LirFunction *func) {
    return func->hir_func;
}

/* ---- Block sorting ---- */

void
lir_function_sort_blocks(LirFunction *func) {
    size_t out_count = 0;
    JitLirBlock *sorted = jit_lir_sort_blocks_rpo(
        (JitLirBlock *)func->blocks, func->num_blocks, &out_count);
    if (sorted != NULL) {
        /* Replace the block array with the sorted result */
        if (out_count > func->blocks_capacity) {
            func->blocks = (LirBasicBlock **)PyMem_RawRealloc(
                func->blocks, out_count * sizeof(LirBasicBlock *));
            func->blocks_capacity = out_count;
        }
        for (size_t i = 0; i < out_count; i++) {
            func->blocks[i] = (LirBasicBlock *)sorted[i];
        }
        func->num_blocks = out_count;
        PyMem_RawFree(sorted);
    }
}
