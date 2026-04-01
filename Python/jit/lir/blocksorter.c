/*
 * blocksorter.c -- Topological block sorting via Tarjan SCC + RPO (pure C)
 *
 * Phase 3D conversion: blocksorter.cpp -> blocksorter.c
 *
 * Sorts LIR basic blocks in reverse post-order, handling strongly
 * connected components (loops) by recursively sorting within each SCC.
 * Uses _Py_hashtable for pointer-keyed hash maps/sets.
 */

#include "cinderx/Jit/lir/lir_c_api.h"

#include "Python.h"
#include "pycore_hashtable.h"

#include <assert.h>
#include <limits.h>
#include <string.h>

/* ---- Dynamic pointer array ---- */

typedef struct {
    void **items;
    size_t len;
    size_t cap;
} PtrVec;

static void
ptrvec_init(PtrVec *v, size_t initial_cap) {
    v->items = (void **)PyMem_RawCalloc(initial_cap, sizeof(void *));
    v->len = 0;
    v->cap = initial_cap;
}

static void
ptrvec_free(PtrVec *v) {
    PyMem_RawFree(v->items);
    v->items = NULL;
    v->len = v->cap = 0;
}

static void
ptrvec_push(PtrVec *v, void *item) {
    if (v->len >= v->cap) {
        size_t new_cap = v->cap ? v->cap * 2 : 8;
        v->items = (void **)PyMem_RawRealloc(v->items,
            new_cap * sizeof(void *));
        v->cap = new_cap;
    }
    v->items[v->len++] = item;
}

static void *
ptrvec_pop(PtrVec *v) {
    assert(v->len > 0);
    return v->items[--v->len];
}

static void
ptrvec_reverse(PtrVec *v) {
    for (size_t i = 0, j = v->len - 1; i < j; i++, j--) {
        void *tmp = v->items[i];
        v->items[i] = v->items[j];
        v->items[j] = tmp;
    }
}

/* ---- SCC block group ---- */

typedef struct SccBlock {
    JitLirBlock entry;
    _Py_hashtable_t *blocks;       /* set of JitLirBlock */
    PtrVec successors;             /* array of SccBlock* */
} SccBlock;

static SccBlock *
sccblock_new(void) {
    SccBlock *s = (SccBlock *)PyMem_RawCalloc(1, sizeof(SccBlock));
    s->entry = NULL;
    s->blocks = _Py_hashtable_new(_Py_hashtable_hash_ptr,
                                  _Py_hashtable_compare_direct);
    ptrvec_init(&s->successors, 4);
    return s;
}

static void
sccblock_free(SccBlock *s) {
    if (s == NULL) return;
    _Py_hashtable_destroy(s->blocks);
    ptrvec_free(&s->successors);
    PyMem_RawFree(s);
}

static int
sccblock_contains(const SccBlock *s, JitLirBlock block) {
    return _Py_hashtable_get_entry((_Py_hashtable_t *)s->blocks, block) != NULL;
}

static size_t
sccblock_size(const SccBlock *s) {
    return _Py_hashtable_len(s->blocks);
}

/* ---- Sorter state ---- */

typedef struct {
    JitLirBlock entry;
    JitLirBlock exit_block;

    /* Input blocks (not owned) */
    JitLirBlock *input_blocks;
    size_t num_input;
    _Py_hashtable_t *input_set;    /* set for O(1) membership */

    /* Tarjan state */
    PtrVec scc_stack;
    _Py_hashtable_t *in_stack;     /* set */
    _Py_hashtable_t *visited;      /* block -> int* (allocated index) */
    int index;

    /* Results */
    _Py_hashtable_t *block_to_scc; /* block -> SccBlock* */
    PtrVec scc_blocks;             /* array of SccBlock* (owned) */
} Sorter;

/* Store an int in the hashtable by allocating it */
static void
sorter_set_visited(Sorter *s, JitLirBlock block, int value) {
    int *p = (int *)PyMem_RawMalloc(sizeof(int));
    *p = value;
    _Py_hashtable_set(s->visited, block, p);
}

static int
sorter_get_visited(Sorter *s, JitLirBlock block) {
    int *p = (int *)_Py_hashtable_get(s->visited, block);
    return p ? *p : -1;
}

static void
sorter_update_visited(Sorter *s, JitLirBlock block, int value) {
    int *p = (int *)_Py_hashtable_get(s->visited, block);
    if (p) *p = value;
}

static void
free_visited_value(void *value) {
    PyMem_RawFree(value);
}

/* ---- Tarjan DFS ---- */

static int
dfs_search(Sorter *s, JitLirBlock block) {
    int block_index = sorter_get_visited(s, block);
    if (block_index >= 0) {
        if (_Py_hashtable_get_entry(s->in_stack, block) != NULL) {
            return block_index;
        }
        return INT_MAX;
    }

    int cur_index = s->index++;
    sorter_set_visited(s, block, cur_index);
    int low = cur_index;

    ptrvec_push(&s->scc_stack, block);
    _Py_hashtable_set(s->in_stack, block, block);

    int nsucc = (int)jit_lir_block_num_succs(block);
    for (int si = 0; si < nsucc; si++) {
        JitLirBlock succ = jit_lir_block_get_succ(block, (size_t)si);
        if (_Py_hashtable_get_entry(s->input_set, succ) == NULL ||
            succ == s->entry) {
            continue;
        }
        int min_idx = dfs_search(s, succ);
        if (min_idx < low) low = min_idx;
    }

    /* Update our low-link */
    sorter_update_visited(s, block, low);

    if (cur_index == low) {
        SccBlock *scc = sccblock_new();
        JitLirBlock bb;
        do {
            bb = (JitLirBlock)ptrvec_pop(&s->scc_stack);
            _Py_hashtable_steal(s->in_stack, bb);
            _Py_hashtable_set(scc->blocks, bb, bb);
            _Py_hashtable_set(s->block_to_scc, bb, scc);
        } while (bb != block);
        ptrvec_push(&s->scc_blocks, scc);
    }

    return low;
}

static void
calculate_scc(Sorter *s) {
    s->scc_stack.len = 0;
    _Py_hashtable_clear(s->in_stack);
    s->index = 0;

    for (size_t i = 0; i < s->num_input; i++) {
        dfs_search(s, s->input_blocks[i]);
    }
}

static void
calc_entry_blocks(Sorter *s) {
    for (size_t i = 0; i < s->num_input; i++) {
        JitLirBlock block = s->input_blocks[i];
        SccBlock *cur_scc = (SccBlock *)_Py_hashtable_get(
            s->block_to_scc, block);

        int nsucc = (int)jit_lir_block_num_succs(block);
        for (int si = 0; si < nsucc; si++) {
            JitLirBlock succ = jit_lir_block_get_succ(block, (size_t)si);
            if (_Py_hashtable_get_entry(s->input_set, succ) == NULL ||
                succ == s->entry) {
                continue;
            }

            SccBlock *succ_scc = (SccBlock *)_Py_hashtable_get(
                s->block_to_scc, succ);
            if (cur_scc == succ_scc) continue;

            assert(succ_scc->entry == NULL || succ_scc->entry == succ);
            succ_scc->entry = succ;
            ptrvec_push(&cur_scc->successors, succ_scc);
        }
    }
}

static void
sort_rpo(Sorter *s) {
    size_t num_scc = s->scc_blocks.len;
    if (num_scc == 0) return;

    /* Move scc_blocks out for processing */
    SccBlock **old_sccs = (SccBlock **)s->scc_blocks.items;
    size_t old_count = num_scc;
    ptrvec_init(&s->scc_blocks, old_count);

    /* Build index map: SccBlock* -> index in old_sccs */
    _Py_hashtable_t *idx_map = _Py_hashtable_new(
        _Py_hashtable_hash_ptr, _Py_hashtable_compare_direct);
    for (size_t i = 0; i < old_count; i++) {
        _Py_hashtable_set(idx_map, old_sccs[i], (void *)(uintptr_t)i);
    }

    SccBlock *entry_scc = (SccBlock *)_Py_hashtable_get(
        s->block_to_scc, s->entry);

    _Py_hashtable_t *visited = _Py_hashtable_new(
        _Py_hashtable_hash_ptr, _Py_hashtable_compare_direct);

    /* DFS stack: pairs of (SccBlock*, next_succ_index) */
    typedef struct { SccBlock *scc; size_t next; } StackFrame;
    StackFrame *stack = (StackFrame *)PyMem_RawMalloc(
        old_count * sizeof(StackFrame));
    size_t stack_len = 0;

    SccBlock *exit_scc = NULL;

    _Py_hashtable_set(visited, entry_scc, entry_scc);
    size_t entry_idx = (size_t)(uintptr_t)_Py_hashtable_get(
        idx_map, entry_scc);
    stack[stack_len++] = (StackFrame){old_sccs[entry_idx], 0};
    old_sccs[entry_idx] = NULL; /* taken */

    while (stack_len > 0) {
        StackFrame *top = &stack[stack_len - 1];
        SccBlock *bb = top->scc;

        if (top->next >= bb->successors.len) {
            ptrvec_push(&s->scc_blocks, bb);
            stack_len--;
            continue;
        }

        SccBlock *next_succ = (SccBlock *)bb->successors.items[top->next++];
        if (_Py_hashtable_get_entry(visited, next_succ) != NULL) {
            continue;
        }
        _Py_hashtable_set(visited, next_succ, next_succ);

        size_t si = (size_t)(uintptr_t)_Py_hashtable_get(idx_map, next_succ);
        SccBlock *succ_bb = old_sccs[si];

        if (succ_bb != NULL && succ_bb->entry == s->exit_block) {
            assert(sccblock_size(succ_bb) == 1);
            exit_scc = succ_bb;
            old_sccs[si] = NULL;
            continue;
        }
        if (succ_bb != NULL) {
            old_sccs[si] = NULL;
            stack[stack_len++] = (StackFrame){succ_bb, 0};
        }
    }

    ptrvec_reverse(&s->scc_blocks);
    if (exit_scc != NULL) {
        ptrvec_push(&s->scc_blocks, exit_scc);
    }

    /* Free remaining unreachable SCCs */
    for (size_t i = 0; i < old_count; i++) {
        if (old_sccs[i] != NULL) {
            sccblock_free(old_sccs[i]);
        }
    }
    PyMem_RawFree(old_sccs);
    PyMem_RawFree(stack);
    _Py_hashtable_destroy(idx_map);
    _Py_hashtable_destroy(visited);
}

/* ---- Recursive sort for sub-SCCs ---- */

static void sort_blocks_internal(
    JitLirBlock *blocks, size_t count,
    JitLirBlock entry, JitLirBlock exit_block,
    PtrVec *result);

static void
expand_scc(SccBlock *scc, JitLirBlock exit_block, PtrVec *result) {
    size_t sz = sccblock_size(scc);
    if (sz == 1) {
        /* Single block — find and emit it */
        _Py_hashtable_entry_t *e = NULL;
        for (size_t i = 0; i < scc->blocks->nbuckets && e == NULL; i++) {
            e = (_Py_hashtable_entry_t *)scc->blocks->buckets[i].head;
        }
        if (e) ptrvec_push(result, (void *)e->key);
    } else {
        /* Multi-block SCC — collect blocks and recursively sort */
        PtrVec sub_blocks;
        ptrvec_init(&sub_blocks, sz);
        for (size_t i = 0; i < scc->blocks->nbuckets; i++) {
            _Py_slist_item_t *item = scc->blocks->buckets[i].head;
            while (item) {
                _Py_hashtable_entry_t *entry = (_Py_hashtable_entry_t *)item;
                ptrvec_push(&sub_blocks, (void *)entry->key);
                item = item->next;
            }
        }
        sort_blocks_internal(
            (JitLirBlock *)sub_blocks.items, sub_blocks.len,
            scc->entry, exit_block, result);
        ptrvec_free(&sub_blocks);
    }
}

static void
sort_blocks_internal(
    JitLirBlock *blocks, size_t count,
    JitLirBlock entry, JitLirBlock exit_block,
    PtrVec *result)
{
    if (count == 0) return;

    Sorter s;
    memset(&s, 0, sizeof(s));
    s.entry = entry;
    s.exit_block = exit_block;
    s.input_blocks = blocks;
    s.num_input = count;

    s.input_set = _Py_hashtable_new(
        _Py_hashtable_hash_ptr, _Py_hashtable_compare_direct);
    for (size_t i = 0; i < count; i++) {
        _Py_hashtable_set(s.input_set, blocks[i], blocks[i]);
    }

    ptrvec_init(&s.scc_stack, count);
    s.in_stack = _Py_hashtable_new(
        _Py_hashtable_hash_ptr, _Py_hashtable_compare_direct);
    s.visited = _Py_hashtable_new_full(
        _Py_hashtable_hash_ptr, _Py_hashtable_compare_direct,
        NULL, free_visited_value, NULL);
    s.index = 0;

    s.block_to_scc = _Py_hashtable_new(
        _Py_hashtable_hash_ptr, _Py_hashtable_compare_direct);
    ptrvec_init(&s.scc_blocks, count);

    calculate_scc(&s);
    calc_entry_blocks(&s);
    sort_rpo(&s);

    /* Expand SCCs into result */
    for (size_t i = 0; i < s.scc_blocks.len; i++) {
        SccBlock *scc = (SccBlock *)s.scc_blocks.items[i];
        expand_scc(scc, exit_block, result);
    }

    /* Cleanup */
    for (size_t i = 0; i < s.scc_blocks.len; i++) {
        sccblock_free((SccBlock *)s.scc_blocks.items[i]);
    }
    ptrvec_free(&s.scc_blocks);
    ptrvec_free(&s.scc_stack);
    _Py_hashtable_destroy(s.in_stack);
    _Py_hashtable_destroy(s.visited);
    _Py_hashtable_destroy(s.block_to_scc);
    _Py_hashtable_destroy(s.input_set);
}

/* ---- Public API ---- */

JitLirBlock *
jit_lir_sort_blocks_rpo(JitLirBlock *blocks, size_t count,
                        size_t *out_count) {
    if (count == 0) {
        *out_count = 0;
        return NULL;
    }

    JitLirBlock entry = blocks[0];
    JitLirBlock exit_block = blocks[count - 1];

    PtrVec result;
    ptrvec_init(&result, count);

    sort_blocks_internal(blocks, count, entry, exit_block, &result);

    *out_count = result.len;
    /* Caller takes ownership of the array */
    return (JitLirBlock *)result.items;
}
