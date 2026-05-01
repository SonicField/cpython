/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * PhxPtrMap — generic void*-keyed open-addressed hash map with void* values.
 * Phase 4.X-full X3a substrate per supervisor 04:04:50Z (Q-X3-1/Q-X3-2)
 * + theologian docs/tier7-phase4x-x3-preanalysis-2026-05-01.md §2 + §6.2.
 *
 * Use-class: hash-based key→value map for pointer keys + pointer values
 * (e.g. Environment ThreadedRef-keyed sets where value carries auxiliary
 * data; BlockCanonicalizer copy-tracking maps). X3b/X3c migrations
 * consume for E-1+E-2+E-3 + E-6 discharge.
 *
 * Implementation: power-of-2 capacity, linear probing, load factor 0.7
 * triggers resize (double). Hash: Knuth multiplicative on uintptr_t cast
 * of key. Empty slot sentinel: key==NULL.
 *
 * Layout: parallel keys[] + values[] entries packed as PhxPtrMapEntry
 * (key, value) for cache-friendly access. Mirrors PhxBlockMap pattern at
 * builder_state_c.h:124-128.
 *
 * NULL-key invariant: PhxPtrMap cannot store NULL as a key — NULL is
 * reserved as the empty-slot sentinel. NULL VALUE is permitted.
 *
 * Allocator: plain calloc/realloc/free (matches PhxPtrSet sibling
 * pattern at phx_ptr_set.h). Forward stance #4 discharged via
 * JIT_CHECK_C(new_entries != NULL) loud-fail post-allocation per
 * X1a 0bac3bc31d precedent.
 *
 * Q-X3-2 (supervisor 04:04:50Z): void*-VALUE only per YAGNI bound.
 * Specialize at E-9 if typed-value variant needed.
 *
 * Q-X3-5 (supervisor 04:04:50Z): tagged as E-9 forward-prep audit
 * trail. Substrate is independently consumed by X3b/X3c (E-1/E-2/E-3/E-6
 * discharge); no speculative E-9 surface added.
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

typedef struct PhxPtrMapEntry {
    void *key;   /* NULL = empty slot sentinel */
    void *value; /* NULL value PERMITTED (key absence checked via key==NULL) */
} PhxPtrMapEntry;

typedef struct PhxPtrMap {
    PhxPtrMapEntry *entries;
    size_t count;
    size_t capacity; /* power-of-2; 0 ⇒ entries NULL (lazy init) */
} PhxPtrMap;

#define PHX_PTR_MAP_INITIAL_CAP 16u
#define PHX_PTR_MAP_LOAD_NUM 7u
#define PHX_PTR_MAP_LOAD_DEN 10u

static inline void phx_ptr_map_init(PhxPtrMap *m) {
    m->entries = NULL;
    m->count = 0;
    m->capacity = 0;
}

static inline void phx_ptr_map_destroy(PhxPtrMap *m) {
    if (m->entries) {
        free(m->entries);
        m->entries = NULL;
    }
    m->count = 0;
    m->capacity = 0;
}

static inline void phx_ptr_map_clear(PhxPtrMap *m) {
    if (m->entries && m->capacity) {
        memset(m->entries, 0, m->capacity * sizeof(PhxPtrMapEntry));
    }
    m->count = 0;
}

/* Knuth multiplicative hash on uintptr_t, mask to power-of-2 capacity. */
static inline size_t phx_ptr_map_slot(size_t cap, const void *key) {
    uint64_t k = (uint64_t)(uintptr_t)key;
    uint32_t h = (uint32_t)((k * 11400714819323198485ULL) >> 32);
    return (size_t)h & (cap - 1u);
}

/* Returns value associated with key, or NULL if not present. Note: NULL
 * value is also a valid stored value, so callers needing absence-vs-NULL
 * disambiguation must use phx_ptr_map_contains. */
static inline void *phx_ptr_map_lookup(const PhxPtrMap *m, const void *key) {
    if (m->capacity == 0 || m->entries == NULL) return NULL;
    size_t i = phx_ptr_map_slot(m->capacity, key);
    while (m->entries[i].key != NULL) {
        if (m->entries[i].key == key) return m->entries[i].value;
        i = (i + 1u) & (m->capacity - 1u);
    }
    return NULL;
}

static inline int phx_ptr_map_contains(const PhxPtrMap *m, const void *key) {
    if (m->capacity == 0 || m->entries == NULL) return 0;
    size_t i = phx_ptr_map_slot(m->capacity, key);
    while (m->entries[i].key != NULL) {
        if (m->entries[i].key == key) return 1;
        i = (i + 1u) & (m->capacity - 1u);
    }
    return 0;
}

/* Insert raw (assumes capacity > 0 and load not exceeded). Returns 1 if
 * newly inserted, 0 if updated (overwrite). */
static inline int phx_ptr_map_insert_raw(
        PhxPtrMap *m, void *key, void *value) {
    size_t i = phx_ptr_map_slot(m->capacity, key);
    while (m->entries[i].key != NULL) {
        if (m->entries[i].key == key) {
            m->entries[i].value = value; /* overwrite */
            return 0;
        }
        i = (i + 1u) & (m->capacity - 1u);
    }
    m->entries[i].key = key;
    m->entries[i].value = value;
    m->count++;
    return 1;
}

static inline void phx_ptr_map_resize(PhxPtrMap *m, size_t new_cap) {
    PhxPtrMapEntry *old_entries = m->entries;
    size_t old_cap = m->capacity;
    PhxPtrMapEntry *new_entries =
        (PhxPtrMapEntry *)calloc(new_cap, sizeof(PhxPtrMapEntry));
    JIT_CHECK_C(new_entries != NULL,
                "phx_ptr_map_resize calloc failed (new_cap=%zu)", new_cap);
    m->entries = new_entries;
    m->capacity = new_cap;
    m->count = 0;
    if (old_entries) {
        for (size_t i = 0; i < old_cap; i++) {
            if (old_entries[i].key != NULL) {
                phx_ptr_map_insert_raw(
                    m, old_entries[i].key, old_entries[i].value);
            }
        }
        free(old_entries);
    }
}

/* Returns 1 if newly inserted, 0 if updated (overwrite). */
static inline int phx_ptr_map_insert(PhxPtrMap *m, void *key, void *value) {
    if (m->capacity == 0) {
        phx_ptr_map_resize(m, PHX_PTR_MAP_INITIAL_CAP);
    } else if ((m->count + 1u) * PHX_PTR_MAP_LOAD_DEN
               > m->capacity * PHX_PTR_MAP_LOAD_NUM) {
        phx_ptr_map_resize(m, m->capacity * 2u);
    }
    return phx_ptr_map_insert_raw(m, key, value);
}

static inline size_t phx_ptr_map_size(const PhxPtrMap *m) {
    return m->count;
}

static inline size_t phx_ptr_map_capacity(const PhxPtrMap *m) {
    return m->capacity;
}

/* Raw slot accessors for open-address-hash iteration. Caller iterates
 * 0..capacity-1 and skips slots where key==NULL:
 *
 *   for (size_t i = 0; i < phx_ptr_map_capacity(&m); i++) {
 *       void *key = phx_ptr_map_at_key(&m, i);
 *       if (key != NULL) { use(key, phx_ptr_map_at_value(&m, i)); }
 *   }
 */
static inline void *phx_ptr_map_at_key(const PhxPtrMap *m, size_t slot_idx) {
    if (m->entries == NULL || slot_idx >= m->capacity) return NULL;
    return m->entries[slot_idx].key;
}

static inline void *phx_ptr_map_at_value(const PhxPtrMap *m, size_t slot_idx) {
    if (m->entries == NULL || slot_idx >= m->capacity) return NULL;
    return m->entries[slot_idx].value;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
