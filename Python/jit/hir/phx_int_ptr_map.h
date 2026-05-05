/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * PhxIntPtrMap — open-addressed integer-keyed hash map with void* values.
 * Phase 5.A3 substrate per supervisor 13:16:01Z (Q1 = option A) +
 * docs/5a3-function-cpp-bridge-spec-2026-05-05.md §2.1.
 *
 * Use-class: int-key → pointer-value map for the C port of
 * function.cpp's deepCopyBasicBlocks helpers (block_index_map +
 * output_index_map). Replaces UnorderedMap<int, BasicBlock*> /
 * UnorderedMap<int, Instruction*>.
 *
 * Implementation: power-of-2 capacity, linear probing, load factor 0.7
 * triggers resize (double). Hash: Knuth multiplicative on uint32_t.
 * Empty slot sentinel: explicit `char occupied` flag.
 *
 * Sentinel rationale (per spec §2.1): BasicBlock id_ and Instruction
 * id_ are dense ints starting at 0, so no specific int value is a safe
 * key sentinel. The companion PhxBlockMap (builder_state_c.h:124) uses
 * value==NULL as the sentinel, which is sound only when callers
 * guarantee no NULL value can be inserted. PhxIntPtrMap takes the more
 * conservative `char occupied` route — 1 byte overhead per slot but
 * unambiguous absent-vs-NULL-value disambiguation, and reusable for
 * future int-keyed maps where NULL is a valid value.
 *
 * Allocator: plain calloc/realloc/free (matches PhxPtrMap sibling at
 * phx_ptr_map.h). Allocation OOM is loud-failed via JIT_CHECK_C per
 * feedback_no_silent_bailout_in_helpers.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "cinderx/Common/jit_log_c.h"  /* JIT_CHECK_C loud-fail on alloc OOM */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PhxIntPtrMapEntry {
    int   key;
    char  occupied; /* 1 = slot in use, 0 = empty (covers any int key) */
    void *value;    /* may be NULL when occupied=1 */
} PhxIntPtrMapEntry;

typedef struct PhxIntPtrMap {
    PhxIntPtrMapEntry *entries;
    size_t count;
    size_t capacity; /* power-of-2; 0 ⇒ entries NULL (lazy init) */
} PhxIntPtrMap;

#define PHX_INT_PTR_MAP_INITIAL_CAP 16u
#define PHX_INT_PTR_MAP_LOAD_NUM 7u
#define PHX_INT_PTR_MAP_LOAD_DEN 10u

static inline void phx_int_ptr_map_init(PhxIntPtrMap *m) {
    m->entries = NULL;
    m->count = 0;
    m->capacity = 0;
}

static inline void phx_int_ptr_map_destroy(PhxIntPtrMap *m) {
    if (m->entries) {
        free(m->entries);
        m->entries = NULL;
    }
    m->count = 0;
    m->capacity = 0;
}

static inline void phx_int_ptr_map_clear(PhxIntPtrMap *m) {
    if (m->entries && m->capacity) {
        memset(m->entries, 0, m->capacity * sizeof(PhxIntPtrMapEntry));
    }
    m->count = 0;
}

static inline size_t phx_int_ptr_map_slot(size_t cap, int key) {
    uint32_t h = (uint32_t)key * 2654435761u;
    return (size_t)h & (cap - 1u);
}

/* Returns value associated with key, or NULL if not present. NULL value
 * is also valid stored content; callers needing absent-vs-NULL-value
 * disambiguation must use phx_int_ptr_map_contains. */
static inline void *phx_int_ptr_map_lookup(const PhxIntPtrMap *m, int key) {
    if (m->capacity == 0 || m->entries == NULL) return NULL;
    size_t i = phx_int_ptr_map_slot(m->capacity, key);
    while (m->entries[i].occupied) {
        if (m->entries[i].key == key) return m->entries[i].value;
        i = (i + 1u) & (m->capacity - 1u);
    }
    return NULL;
}

static inline int phx_int_ptr_map_contains(const PhxIntPtrMap *m, int key) {
    if (m->capacity == 0 || m->entries == NULL) return 0;
    size_t i = phx_int_ptr_map_slot(m->capacity, key);
    while (m->entries[i].occupied) {
        if (m->entries[i].key == key) return 1;
        i = (i + 1u) & (m->capacity - 1u);
    }
    return 0;
}

/* Insert raw (assumes capacity > 0 and load not exceeded). Returns 1 if
 * newly inserted, 0 if updated (overwrite). */
static inline int phx_int_ptr_map_insert_raw(
        PhxIntPtrMap *m, int key, void *value) {
    size_t i = phx_int_ptr_map_slot(m->capacity, key);
    while (m->entries[i].occupied) {
        if (m->entries[i].key == key) {
            m->entries[i].value = value; /* overwrite */
            return 0;
        }
        i = (i + 1u) & (m->capacity - 1u);
    }
    m->entries[i].key = key;
    m->entries[i].value = value;
    m->entries[i].occupied = 1;
    m->count++;
    return 1;
}

static inline void phx_int_ptr_map_resize(PhxIntPtrMap *m, size_t new_cap) {
    PhxIntPtrMapEntry *old_entries = m->entries;
    size_t old_cap = m->capacity;
    PhxIntPtrMapEntry *new_entries =
        (PhxIntPtrMapEntry *)calloc(new_cap, sizeof(PhxIntPtrMapEntry));
    JIT_CHECK_C(new_entries != NULL,
                "phx_int_ptr_map_resize calloc failed (new_cap=%zu)", new_cap);
    m->entries = new_entries;
    m->capacity = new_cap;
    m->count = 0;
    if (old_entries) {
        for (size_t i = 0; i < old_cap; i++) {
            if (old_entries[i].occupied) {
                phx_int_ptr_map_insert_raw(
                    m, old_entries[i].key, old_entries[i].value);
            }
        }
        free(old_entries);
    }
}

/* Returns 1 if newly inserted, 0 if updated (overwrite). */
static inline int phx_int_ptr_map_insert(PhxIntPtrMap *m, int key, void *value) {
    if (m->capacity == 0) {
        phx_int_ptr_map_resize(m, PHX_INT_PTR_MAP_INITIAL_CAP);
    } else if ((m->count + 1u) * PHX_INT_PTR_MAP_LOAD_DEN
               > m->capacity * PHX_INT_PTR_MAP_LOAD_NUM) {
        phx_int_ptr_map_resize(m, m->capacity * 2u);
    }
    return phx_int_ptr_map_insert_raw(m, key, value);
}

/* Loud-fail variant: panics via JIT_CHECK_C if key is absent. Mirrors
 * containers.h map_get_strict semantics used by the C++ source. */
static inline void *phx_int_ptr_map_get_strict(const PhxIntPtrMap *m, int key) {
    JIT_CHECK_C(m->capacity > 0 && m->entries != NULL,
                "phx_int_ptr_map_get_strict on empty map (key=%d)", key);
    size_t i = phx_int_ptr_map_slot(m->capacity, key);
    while (m->entries[i].occupied) {
        if (m->entries[i].key == key) return m->entries[i].value;
        i = (i + 1u) & (m->capacity - 1u);
    }
    JIT_CHECK_C(0, "phx_int_ptr_map_get_strict: key %d absent", key);
    return NULL; /* unreachable */
}

static inline size_t phx_int_ptr_map_size(const PhxIntPtrMap *m) {
    return m->count;
}

static inline size_t phx_int_ptr_map_capacity(const PhxIntPtrMap *m) {
    return m->capacity;
}

/* Raw slot accessors for open-address-hash iteration. Caller iterates
 * 0..capacity-1 and skips slots where occupied==0:
 *
 *   for (size_t i = 0; i < phx_int_ptr_map_capacity(&m); i++) {
 *       if (phx_int_ptr_map_slot_occupied(&m, i)) {
 *           int   k = phx_int_ptr_map_at_key(&m, i);
 *           void *v = phx_int_ptr_map_at_value(&m, i);
 *       }
 *   }
 */
static inline int phx_int_ptr_map_slot_occupied(
        const PhxIntPtrMap *m, size_t slot_idx) {
    if (m->entries == NULL || slot_idx >= m->capacity) return 0;
    return (int)m->entries[slot_idx].occupied;
}

static inline int phx_int_ptr_map_at_key(
        const PhxIntPtrMap *m, size_t slot_idx) {
    return m->entries[slot_idx].key;
}

static inline void *phx_int_ptr_map_at_value(
        const PhxIntPtrMap *m, size_t slot_idx) {
    return m->entries[slot_idx].value;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
