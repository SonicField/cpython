/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Pure C DominatorAnalysis — computes immediate dominators and
 * domination sets using Cooper, Harvey, Kennedy's algorithm.
 */

#include "cinderx/Jit/hir/dominator_c.h"
#include "cinderx/Jit/hir/hir_cfg_rpo_c.h"
#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/hir/hir_basic_block_c.h"
#include "cinderx/Common/jit_log_c.h"
#include "Python.h"

#include <string.h>

/* ---- Dominator state ---- */

struct PhxDominatorState {
    void **idoms;
    size_t idoms_size;

    void ***dom_sets;
    size_t *dom_set_counts;
    size_t *dom_set_caps;
    size_t dom_sets_size;
};

static void dom_set_add(PhxDominatorState *s, int block_id, void *dominated) {
    if ((size_t)block_id >= s->dom_sets_size) return;
    size_t *cnt = &s->dom_set_counts[block_id];
    size_t *cap = &s->dom_set_caps[block_id];
    if (*cnt >= *cap) {
        *cap = *cap ? *cap * 2 : 4;
        s->dom_sets[block_id] = (void **)PyMem_RawRealloc(
            s->dom_sets[block_id], *cap * sizeof(void *));
    }
    s->dom_sets[block_id][*cnt] = dominated;
    (*cnt)++;
}

PhxDominatorState *phx_dom_create(HirFunction func) {
    HirCFG *cfg = (HirCFG *)hir_func_cfg_ptr(func);

    int max_bid = 0;
    size_t n_blocks = 0;
    for (HirBasicBlock *b = hir_cfg_first_block(cfg); b; b = hir_cfg_next_block(cfg, b)) {
        n_blocks++;
        if (b->id > max_bid) max_bid = b->id;
    }

    size_t arr_size = (size_t)(max_bid + 1);

    void **rpo = (void **)PyMem_RawMalloc(n_blocks * sizeof(void *));
    size_t rpo_count = hir_cfg_get_rpo_c(cfg, rpo, n_blocks);

    int *rpo_index = (int *)PyMem_RawMalloc(arr_size * sizeof(int));
    memset(rpo_index, -1, arr_size * sizeof(int));
    for (size_t i = 0; i < rpo_count; i++) {
        rpo_index[((HirBasicBlock *)rpo[i])->id] = (int)i;
    }

    PhxDominatorState *state = (PhxDominatorState *)PyMem_RawCalloc(1, sizeof(PhxDominatorState));
    state->idoms = (void **)PyMem_RawCalloc(arr_size, sizeof(void *));
    state->idoms_size = arr_size;
    state->dom_sets = (void ***)PyMem_RawCalloc(arr_size, sizeof(void **));
    state->dom_set_counts = (size_t *)PyMem_RawCalloc(arr_size, sizeof(size_t));
    state->dom_set_caps = (size_t *)PyMem_RawCalloc(arr_size, sizeof(size_t));
    state->dom_sets_size = arr_size;

    if (rpo_count == 0) goto cleanup;

    HirBasicBlock *entry = (HirBasicBlock *)rpo[0];
    state->idoms[entry->id] = entry;

    for (int changed = 1; changed;) {
        changed = 0;
        for (size_t ri = 1; ri < rpo_count; ri++) {
            HirBasicBlock *block = (HirBasicBlock *)rpo[ri];
            size_t n_in = hir_bb_in_edges_count(block);

            HirBasicBlock *pred1 = NULL;
            size_t pi;
            for (pi = 0; pi < n_in; pi++) {
                const HirEdge *e = hir_bb_in_edge(block, pi);
                HirBasicBlock *p = (HirBasicBlock *)e->from;
                if (state->idoms[p->id]) {
                    pred1 = p;
                    pi++;
                    break;
                }
            }
            if (!pred1) continue;

            for (; pi < n_in; pi++) {
                const HirEdge *e = hir_bb_in_edge(block, pi);
                HirBasicBlock *pred2 = (HirBasicBlock *)e->from;
                if (pred2 == pred1 || !state->idoms[pred2->id]) continue;

                const HirBasicBlock *a = pred1;
                const HirBasicBlock *b = pred2;
                while (a != b) {
                    while (rpo_index[a->id] > rpo_index[b->id]) {
                        a = (const HirBasicBlock *)state->idoms[a->id];
                    }
                    while (rpo_index[b->id] > rpo_index[a->id]) {
                        b = (const HirBasicBlock *)state->idoms[b->id];
                    }
                }
                pred1 = (HirBasicBlock *)a;
            }

            if (state->idoms[block->id] != pred1) {
                state->idoms[block->id] = pred1;
                changed = 1;
            }
        }
    }

    state->idoms[entry->id] = NULL;

    for (size_t ri = rpo_count; ri > 0; ri--) {
        HirBasicBlock *block = (HirBasicBlock *)rpo[ri - 1];
        dom_set_add(state, block->id, block);

        void *block_dom = state->idoms[block->id];
        if (block_dom) {
            int dom_id = ((HirBasicBlock *)block_dom)->id;
            size_t cnt = state->dom_set_counts[block->id];
            for (size_t i = 0; i < cnt; i++) {
                dom_set_add(state, dom_id, state->dom_sets[block->id][i]);
            }
        }
    }

cleanup:
    PyMem_RawFree(rpo);
    PyMem_RawFree(rpo_index);
    return state;
}

void *phx_dom_idom(const PhxDominatorState *state, int block_id) {
    if (block_id < 0 || (size_t)block_id >= state->idoms_size) return NULL;
    return state->idoms[block_id];
}

int phx_dom_dominates(const PhxDominatorState *state, int a_id, int b_id) {
    if (a_id < 0 || (size_t)a_id >= state->dom_sets_size) return 0;
    size_t cnt = state->dom_set_counts[a_id];
    for (size_t i = 0; i < cnt; i++) {
        if (((HirBasicBlock *)state->dom_sets[a_id][i])->id == b_id) return 1;
    }
    return 0;
}

size_t phx_dom_dominated_count(const PhxDominatorState *state, int block_id) {
    if (block_id < 0 || (size_t)block_id >= state->dom_sets_size) return 0;
    return state->dom_set_counts[block_id];
}

void *phx_dom_dominated_get(const PhxDominatorState *state, int block_id, size_t i) {
    return state->dom_sets[block_id][i];
}

void phx_dom_destroy(PhxDominatorState *state) {
    PyMem_RawFree(state->idoms);
    for (size_t i = 0; i < state->dom_sets_size; i++) {
        PyMem_RawFree(state->dom_sets[i]);
    }
    PyMem_RawFree(state->dom_sets);
    PyMem_RawFree(state->dom_set_counts);
    PyMem_RawFree(state->dom_set_caps);
    PyMem_RawFree(state);
}
