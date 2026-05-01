/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * PhxBCOffsetSet — sorted-unique ascending int dynamic array. Phase 4.X-full
 * X1a substrate per supervisor 02:36:04Z (α) smallest-first dispatch +
 * theologian docs/tier7-phase4x-full-entry-preanalysis-2026-05-01.md §2.7.
 *
 * Replaces std::set<BCIndex> at builder.cpp:429 (createBlocks block_starts;
 * sole std::set use in builder.cpp). Discharges E-8 stay-C++ exception when
 * X1b migration lands.
 *
 * Semantics match std::set<int> for the operations createBlocks needs:
 *   - sorted ascending iteration via at(i) for i in [0, size())
 *   - insert with dedup
 *   - empty-check via size() == 0
 *
 * Implementation: dynamic array, binary-search insert, memmove on shift,
 * lazy alloc (capacity 0 ⇒ data NULL). Set sizes in createBlocks are O(N)
 * in bytecode-instruction count; binary-search insert is O(log N) compare +
 * O(N) memmove worst case. Same asymptotic class as std::set::insert
 * (red-black tree O(log N) compare + rebalance), with lower constant factor
 * for dense small N (<256 typical).
 */
#pragma once

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "cinderx/Common/jit_log_c.h"  /* JIT_CHECK_C loud-fail on realloc OOM */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PhxBCOffsetSet {
    int *data;
    size_t count;
    size_t capacity; /* 0 ⇒ data NULL (lazy init) */
} PhxBCOffsetSet;

#define PHX_BC_OFFSET_SET_INITIAL_CAP 8u

static inline void phx_bc_offset_set_init(PhxBCOffsetSet *s) {
    s->data = NULL;
    s->count = 0;
    s->capacity = 0;
}

static inline void phx_bc_offset_set_destroy(PhxBCOffsetSet *s) {
    if (s->data) {
        free(s->data);
        s->data = NULL;
    }
    s->count = 0;
    s->capacity = 0;
}

static inline void phx_bc_offset_set_clear(PhxBCOffsetSet *s) {
    s->count = 0;
}

static inline size_t phx_bc_offset_set_size(const PhxBCOffsetSet *s) {
    return s->count;
}

static inline int phx_bc_offset_set_at(const PhxBCOffsetSet *s, size_t idx) {
    return s->data[idx];
}

/* Binary search for `key`. Returns the index where key is OR would be
 * inserted to maintain sorted order. Caller checks data[lo] == key for
 * presence vs absence. */
static inline size_t phx_bc_offset_set_lower_bound(
        const PhxBCOffsetSet *s, int key) {
    size_t lo = 0;
    size_t hi = s->count;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (s->data[mid] < key) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

/* Insert `key` if not already present. Returns 1 if inserted, 0 if dup. */
static inline int phx_bc_offset_set_insert(PhxBCOffsetSet *s, int key) {
    size_t pos = phx_bc_offset_set_lower_bound(s, key);
    if (pos < s->count && s->data[pos] == key) {
        return 0; /* duplicate */
    }
    if (s->count + 1 > s->capacity) {
        size_t new_cap = s->capacity ? s->capacity * 2u
                                     : PHX_BC_OFFSET_SET_INITIAL_CAP;
        int *new_data = (int *)realloc(s->data, new_cap * sizeof(int));
        JIT_CHECK_C(new_data != NULL,
                    "phx_bc_offset_set_insert realloc failed (new_cap=%zu)",
                    new_cap);
        s->data = new_data;
        s->capacity = new_cap;
    }
    if (pos < s->count) {
        memmove(&s->data[pos + 1], &s->data[pos],
                (s->count - pos) * sizeof(int));
    }
    s->data[pos] = key;
    s->count++;
    return 1;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
