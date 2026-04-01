/*
 * lir_function.c -- C implementation of LIR Function
 *
 * Phase B Step 4: Operates on the LirFunction C struct.
 * Coexists with function.cpp until all consumers use the C API.
 *
 * Complex methods (copyFrom, SortRPO) are deferred — they remain
 * as C++ wrappers during the transition period.
 */

#include "cinderx/Jit/lir/lir_c_api.h"

#include "Python.h"

#define INITIAL_BLOCK_CAPACITY 16

/* ---- Lifecycle ---- */

LirFunction *
lir_function_create(const void *hir_func) {
    LirFunction *func = (LirFunction *)PyMem_RawCalloc(
        1, sizeof(LirFunction));
    func->hir_func = hir_func;
    func->blocks = (LirBasicBlock **)PyMem_RawMalloc(
        INITIAL_BLOCK_CAPACITY * sizeof(LirBasicBlock *));
    func->blocks_capacity = INITIAL_BLOCK_CAPACITY;
    return func;
}

void
lir_function_free(LirFunction *func) {
    if (func == NULL) return;
    for (size_t i = 0; i < func->num_blocks; i++) {
        lir_block_free(func->blocks[i]);
    }
    PyMem_RawFree(func->blocks);
    PyMem_RawFree(func);
}

/* ---- ID allocation ---- */

int
lir_function_alloc_id(LirFunction *func) {
    return func->next_id++;
}

void
lir_function_set_next_id(LirFunction *func, int id) {
    func->next_id = id;
}

/* ---- Block management ---- */

LirBasicBlock *
lir_function_alloc_block(LirFunction *func) {
    int id = lir_function_alloc_id(func);
    LirBasicBlock *bb = lir_block_create(func, id);

    /* Append to blocks array */
    if (func->num_blocks >= func->blocks_capacity) {
        size_t new_cap = func->blocks_capacity * 2;
        func->blocks = (LirBasicBlock **)PyMem_RawRealloc(
            func->blocks, new_cap * sizeof(LirBasicBlock *));
        func->blocks_capacity = new_cap;
    }
    func->blocks[func->num_blocks++] = bb;
    return bb;
}

size_t
lir_function_num_blocks(const LirFunction *func) {
    return func->num_blocks;
}

LirBasicBlock *
lir_function_block_at(const LirFunction *func, size_t index) {
    return func->blocks[index];
}

LirBasicBlock *
lir_function_entry_block(const LirFunction *func) {
    return (func->num_blocks > 0) ? func->blocks[0] : NULL;
}
