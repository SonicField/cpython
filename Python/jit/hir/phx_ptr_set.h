/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * PhxPtrSet — generic void*-keyed open-addressed hash set. Phase 4.X-full
 * X2b substrate per supervisor 03:17:18Z next-up + theologian
 * docs/tier7-phase4x-full-entry-preanalysis-2026-05-01.md §2.4.
 *
 * Use-class: hash-based set membership for pointer keys (e.g. translate()
 * processed BasicBlock*, loop_headers BasicBlock*). Replaces
 * std::unordered_set<T*> at the builder.cpp seam. X2c migration consumes
 * for E-7 translate() loop discharge.
 *
 * Implementation: power-of-2 capacity, linear probing, load factor 0.7
 * triggers resize (double). Hash: Knuth multiplicative on uintptr_t cast
 * of pointer. Empty slot sentinel: key==NULL.
 *
 * NULL-key invariant: PhxPtrSet cannot store NULL as a member — NULL is
 * reserved as the empty-slot sentinel. Callers must guard NULL-pointer
 * insertion if NULL is a possible key value.
 *
 * Allocator: plain calloc/realloc/free (matches phx_ptr_array.h sibling
 * pattern). Forward stance #4 discharged via JIT_CHECK_C(new_entries !=
 * NULL) loud-fail post-allocation per X1a 0bac3bc31d precedent.
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

typedef struct PhxPtrSet {
    void **entries;
    size_t count;
    size_t capacity; /* power-of-2; 0 ⇒ entries NULL (lazy init) */
} PhxPtrSet;

#define PHX_PTR_SET_INITIAL_CAP 16u
#define PHX_PTR_SET_LOAD_NUM 7u
#define PHX_PTR_SET_LOAD_DEN 10u

static inline void phx_ptr_set_init(PhxPtrSet *s) {
    s->entries = NULL;
    s->count = 0;
    s->capacity = 0;
}

static inline void phx_ptr_set_destroy(PhxPtrSet *s) {
    if (s->entries) {
        free(s->entries);
        s->entries = NULL;
    }
    s->count = 0;
    s->capacity = 0;
}

static inline void phx_ptr_set_clear(PhxPtrSet *s) {
    if (s->entries && s->capacity) {
        memset(s->entries, 0, s->capacity * sizeof(void *));
    }
    s->count = 0;
}

/* Knuth multiplicative hash on uintptr_t, mask to power-of-2 capacity. */
static inline size_t phx_ptr_set_slot(size_t cap, const void *key) {
    uint64_t k = (uint64_t)(uintptr_t)key;
    /* Knuth 64-bit multiplicative constant; truncate to 32 bits before mask. */
    uint32_t h = (uint32_t)((k * 11400714819323198485ULL) >> 32);
    return (size_t)h & (cap - 1u);
}

/* Returns 1 if key is present, 0 otherwise. */
static inline int phx_ptr_set_contains(const PhxPtrSet *s, const void *key) {
    if (s->capacity == 0 || s->entries == NULL) return 0;
    size_t i = phx_ptr_set_slot(s->capacity, key);
    while (s->entries[i] != NULL) {
        if (s->entries[i] == key) return 1;
        i = (i + 1u) & (s->capacity - 1u);
    }
    return 0;
}

/* Insert raw (assumes capacity > 0 and load not exceeded). */
static inline int phx_ptr_set_insert_raw(PhxPtrSet *s, void *key) {
    size_t i = phx_ptr_set_slot(s->capacity, key);
    while (s->entries[i] != NULL) {
        if (s->entries[i] == key) return 0; /* already present */
        i = (i + 1u) & (s->capacity - 1u);
    }
    s->entries[i] = key;
    s->count++;
    return 1;
}

static inline void phx_ptr_set_resize(PhxPtrSet *s, size_t new_cap) {
    void **old_entries = s->entries;
    size_t old_cap = s->capacity;
    void **new_entries = (void **)calloc(new_cap, sizeof(void *));
    JIT_CHECK_C(new_entries != NULL,
                "phx_ptr_set_resize calloc failed (new_cap=%zu)", new_cap);
    s->entries = new_entries;
    s->capacity = new_cap;
    s->count = 0;
    if (old_entries) {
        for (size_t i = 0; i < old_cap; i++) {
            if (old_entries[i] != NULL) {
                phx_ptr_set_insert_raw(s, old_entries[i]);
            }
        }
        free(old_entries);
    }
}

/* Returns 1 if newly inserted, 0 if already present. */
static inline int phx_ptr_set_insert(PhxPtrSet *s, void *key) {
    if (s->capacity == 0) {
        phx_ptr_set_resize(s, PHX_PTR_SET_INITIAL_CAP);
    } else if ((s->count + 1u) * PHX_PTR_SET_LOAD_DEN
               > s->capacity * PHX_PTR_SET_LOAD_NUM) {
        phx_ptr_set_resize(s, s->capacity * 2u);
    }
    return phx_ptr_set_insert_raw(s, key);
}

static inline size_t phx_ptr_set_size(const PhxPtrSet *s) {
    return s->count;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
