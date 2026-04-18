/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Pure C reverse postorder traversal of HIR CFG.
 * Iterative DFS (no recursion) to avoid stack overflow on deep CFGs.
 */

#include "cinderx/Jit/hir/hir_cfg_rpo_c.h"
#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/hir/hir_basic_block_c.h"
#include "Python.h"

#include <string.h>

typedef struct {
    HirBasicBlock *block;
    size_t edge_idx;
} RpoDfsFrame;

size_t hir_cfg_get_rpo_c(void *cfg_ptr, void **rpo_out, size_t capacity) {
    HirCFG *cfg = (HirCFG *)cfg_ptr;
    HirBasicBlock *entry = (HirBasicBlock *)cfg->entry_block;
    if (!entry) return 0;

    size_t n_blocks = 0;
    int max_bid = 0;
    for (HirBasicBlock *b = hir_cfg_first_block(cfg); b; b = hir_cfg_next_block(cfg, b)) {
        n_blocks++;
        if (b->id > max_bid) max_bid = b->id;
    }

    char *visited = (char *)PyMem_RawCalloc((size_t)(max_bid + 1), 1);
    RpoDfsFrame *stack = (RpoDfsFrame *)PyMem_RawMalloc(n_blocks * sizeof(RpoDfsFrame));
    void **postorder = (void **)PyMem_RawMalloc(n_blocks * sizeof(void *));
    size_t po_count = 0;

    stack[0].block = entry;
    stack[0].edge_idx = 0;
    visited[entry->id] = 1;
    size_t sp = 1;

    while (sp > 0) {
        RpoDfsFrame *top = &stack[sp - 1];
        HirBasicBlock *block = top->block;
        void *term = hir_bb_get_terminator(block);
        size_t n_edges = term ? hir_c_num_edges(term) : 0;

        if (top->edge_idx < n_edges) {
            size_t ei = top->edge_idx++;
            HirBasicBlock *succ = (HirBasicBlock *)hir_c_successor(term, ei);
            if (succ && !visited[succ->id]) {
                visited[succ->id] = 1;
                stack[sp].block = succ;
                stack[sp].edge_idx = 0;
                sp++;
            }
        } else {
            if (po_count < n_blocks) {
                postorder[po_count++] = block;
            }
            sp--;
        }
    }

    size_t rpo_count = po_count < capacity ? po_count : capacity;
    for (size_t i = 0; i < rpo_count; i++) {
        rpo_out[i] = postorder[po_count - 1 - i];
    }

    PyMem_RawFree(visited);
    PyMem_RawFree(stack);
    PyMem_RawFree(postorder);
    return rpo_count;
}
