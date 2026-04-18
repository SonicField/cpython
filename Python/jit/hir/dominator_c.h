/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C DominatorAnalysis — computes immediate dominators and domination sets.
 * Replaces the C++ DominatorAnalysis class from analysis.h.
 */
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* HirFunction;

typedef struct PhxDominatorState PhxDominatorState;

/* Compute dominator tree for the function's CFG.
 * Caller must free with phx_dom_destroy(). */
PhxDominatorState *phx_dom_create(HirFunction func);

/* Get the immediate dominator of a block (by block ID).
 * Returns the block pointer (as void*), or NULL for the entry block. */
void *phx_dom_idom(const PhxDominatorState *state, int block_id);

/* Check if block A dominates block B (both by block ID).
 * Returns 1 if A dominates B, 0 otherwise. */
int phx_dom_dominates(const PhxDominatorState *state, int a_id, int b_id);

/* Get the number of blocks dominated by a block (by block ID). */
size_t phx_dom_dominated_count(const PhxDominatorState *state, int block_id);

/* Get the i-th block dominated by block_id. Returns block pointer as void*. */
void *phx_dom_dominated_get(const PhxDominatorState *state, int block_id, size_t i);

/* Free the dominator analysis state. */
void phx_dom_destroy(PhxDominatorState *state);

/* Differential verification: compare C dominator results against C++.
 * Returns 1 if all immediate dominators match, 0 if any mismatch. */
int phx_dom_verify(HirFunction func, const PhxDominatorState *c_state);

#ifdef __cplusplus
} /* extern "C" */
#endif
