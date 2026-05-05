/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * PhxPtrIntMap — open-addressed pointer-keyed hash map with int values.
 * Phase 5.A3 substrate per supervisor 13:16:01Z (Q1 = option A) +
 * docs/5a3-function-cpp-bridge-spec-2026-05-05.md §2.2.
 *
 * Use-class: pointer-key → int-value map for the C port of
 * function.cpp's deepCopyBasicBlocks helpers (instr_refs map).
 * Replaces UnorderedMap<LinkedOperand*, int> at function.cpp:118.
 *
 * Implementation: power-of-2 capacity, linear probing, load factor 0.7
 * triggers resize (double). Hash: Knuth multiplicative on uintptr_t
 * cast of key. Empty slot sentinel: key==NULL (mirrors PhxPtrMap).
 *
 * Sentinel rationale (per spec §2.2): pointer keys cannot be NULL in
 * the 5.A3 use-cases (LinkedOperand* always derived from existing
 * Operand instance). Int values carry no sentinel constraint — any
 * int (including 0 and negative) is a valid stored value, so callers
 * must use phx_ptr_int_map_contains to disambiguate absent-vs-zero
 * (or any specific present value).
 *
 * API choice (Q-C deferred to draft-implementation per spec §7):
 * lookup-with-default + separate _contains call, mirroring the
 * PhxIntPtrMap surface for symmetry. The out-param-with-present-flag
 * variant from the spec was rejected in favor of API uniformity across
 * substrate siblings; consumers needing both can compose two calls or
 * use phx_ptr_int_map_get_strict (loud-fail on absent).
 *
 * Allocator: plain calloc/realloc/free. Allocation OOM is loud-failed
 * via JIT_CHECK_C per feedback_no_silent_bailout_in_helpers.
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

typedef struct PhxPtrIntMapEntry {
    void *key;   /* NULL = empty slot sentinel */
    int   value; /* any int permitted (including 0 / negative) */
} PhxPtrIntMapEntry;

typedef struct PhxPtrIntMap {
    PhxPtrIntMapEntry *entries;
    size_t count;
    size_t capacity; /* power-of-2; 0 ⇒ entries NULL (lazy init) */
} PhxPtrIntMap;

#define PHX_PTR_INT_MAP_INITIAL_CAP 16u
#define PHX_PTR_INT_MAP_LOAD_NUM 7u
#define PHX_PTR_INT_MAP_LOAD_DEN 10u

static inline void phx_ptr_int_map_init(PhxPtrIntMap *m) {
    m->entries = NULL;
    m->count = 0;
    m->capacity = 0;
}

static inline void phx_ptr_int_map_destroy(PhxPtrIntMap *m) {
    if (m->entries) {
        free(m->entries);
        m->entries = NULL;
    }
    m->count = 0;
    m->capacity = 0;
}

static inline void phx_ptr_int_map_clear(PhxPtrIntMap *m) {
    if (m->entries && m->capacity) {
        memset(m->entries, 0, m->capacity * sizeof(PhxPtrIntMapEntry));
    }
    m->count = 0;
}

/* Knuth multiplicative hash on uintptr_t, mask to power-of-2 capacity. */
static inline size_t phx_ptr_int_map_slot(size_t cap, const void *key) {
    uint64_t k = (uint64_t)(uintptr_t)key;
    uint32_t h = (uint32_t)((k * 11400714819323198485ULL) >> 32);
    return (size_t)h & (cap - 1u);
}

/* Returns value associated with key, or `default_value` if not present.
 * Any int value (including 0 / negative) is a valid stored value, so
 * callers needing absent-vs-present disambiguation must use
 * phx_ptr_int_map_contains. */
static inline int phx_ptr_int_map_lookup_or(
        const PhxPtrIntMap *m, const void *key, int default_value) {
    if (m->capacity == 0 || m->entries == NULL) return default_value;
    size_t i = phx_ptr_int_map_slot(m->capacity, key);
    while (m->entries[i].key != NULL) {
        if (m->entries[i].key == key) return m->entries[i].value;
        i = (i + 1u) & (m->capacity - 1u);
    }
    return default_value;
}

static inline int phx_ptr_int_map_contains(
        const PhxPtrIntMap *m, const void *key) {
    if (m->capacity == 0 || m->entries == NULL) return 0;
    size_t i = phx_ptr_int_map_slot(m->capacity, key);
    while (m->entries[i].key != NULL) {
        if (m->entries[i].key == key) return 1;
        i = (i + 1u) & (m->capacity - 1u);
    }
    return 0;
}

/* Insert raw (assumes capacity > 0 and load not exceeded). Returns 1 if
 * newly inserted, 0 if updated (overwrite). */
static inline int phx_ptr_int_map_insert_raw(
        PhxPtrIntMap *m, void *key, int value) {
    size_t i = phx_ptr_int_map_slot(m->capacity, key);
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

static inline void phx_ptr_int_map_resize(PhxPtrIntMap *m, size_t new_cap) {
    PhxPtrIntMapEntry *old_entries = m->entries;
    size_t old_cap = m->capacity;
    PhxPtrIntMapEntry *new_entries =
        (PhxPtrIntMapEntry *)calloc(new_cap, sizeof(PhxPtrIntMapEntry));
    JIT_CHECK_C(new_entries != NULL,
                "phx_ptr_int_map_resize calloc failed (new_cap=%zu)", new_cap);
    m->entries = new_entries;
    m->capacity = new_cap;
    m->count = 0;
    if (old_entries) {
        for (size_t i = 0; i < old_cap; i++) {
            if (old_entries[i].key != NULL) {
                phx_ptr_int_map_insert_raw(
                    m, old_entries[i].key, old_entries[i].value);
            }
        }
        free(old_entries);
    }
}

/* Returns 1 if newly inserted, 0 if updated (overwrite). */
static inline int phx_ptr_int_map_insert(
        PhxPtrIntMap *m, void *key, int value) {
    if (m->capacity == 0) {
        phx_ptr_int_map_resize(m, PHX_PTR_INT_MAP_INITIAL_CAP);
    } else if ((m->count + 1u) * PHX_PTR_INT_MAP_LOAD_DEN
               > m->capacity * PHX_PTR_INT_MAP_LOAD_NUM) {
        phx_ptr_int_map_resize(m, m->capacity * 2u);
    }
    return phx_ptr_int_map_insert_raw(m, key, value);
}

/* Loud-fail variant: panics via JIT_CHECK_C if key is absent. Mirrors
 * containers.h map_get_strict semantics used by the C++ source. */
static inline int phx_ptr_int_map_get_strict(
        const PhxPtrIntMap *m, const void *key) {
    JIT_CHECK_C(m->capacity > 0 && m->entries != NULL,
                "phx_ptr_int_map_get_strict on empty map (key=%p)", key);
    size_t i = phx_ptr_int_map_slot(m->capacity, key);
    while (m->entries[i].key != NULL) {
        if (m->entries[i].key == key) return m->entries[i].value;
        i = (i + 1u) & (m->capacity - 1u);
    }
    JIT_CHECK_C(0, "phx_ptr_int_map_get_strict: key %p absent", key);
    return 0; /* unreachable */
}

static inline size_t phx_ptr_int_map_size(const PhxPtrIntMap *m) {
    return m->count;
}

static inline size_t phx_ptr_int_map_capacity(const PhxPtrIntMap *m) {
    return m->capacity;
}

/* Raw slot accessors for open-address-hash iteration. Caller iterates
 * 0..capacity-1 and skips slots where key==NULL:
 *
 *   for (size_t i = 0; i < phx_ptr_int_map_capacity(&m); i++) {
 *       void *k = phx_ptr_int_map_at_key(&m, i);
 *       if (k != NULL) { use(k, phx_ptr_int_map_at_value(&m, i)); }
 *   }
 */
static inline void *phx_ptr_int_map_at_key(
        const PhxPtrIntMap *m, size_t slot_idx) {
    if (m->entries == NULL || slot_idx >= m->capacity) return NULL;
    return m->entries[slot_idx].key;
}

static inline int phx_ptr_int_map_at_value(
        const PhxPtrIntMap *m, size_t slot_idx) {
    return m->entries[slot_idx].value;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
