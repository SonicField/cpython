/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * PhxPtrArray — generic void* dynamic array.
 * FrameState→C: replaces std::vector<Register*> and jit::Stack<Register*>.
 * Extracted to its own header to break circular include between
 * hir_instr_c.h and frame_state.h.
 */
#pragma once

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Determinism counter per template (A) (theologian 2026-05-18T20:47:18Z +
 * sup 21:12:01Z + librarian 21:09:55Z): incremented in phx_ptr_arr_push
 * when a realloc-grow fires. Single-def in builder.cpp alongside
 * phx_framestate_parent_resize_count. Read via ctypes in gate fixture to
 * assert resize-path exercised by existing JIT test suite — amortizes
 * the certainty cost across all typed-wrapper-migration sites (M1 +
 * future M2 + ...). */
extern unsigned long phx_ptr_array_resize_count;

/* P2 per-site attribution counters per theologian 2026-05-18T22:00:54Z +
 * testkeeper 22:08:07Z empirical verification (Sites A+B uncovered by
 * Step 3, Site C delta=15). Bumped at each M1 inliner.cpp site after
 * capacity-change detection at the push call. Global counter remains for
 * substrate-wide signal; per-site counters distinguish which specific
 * M1 caller fired. M2 sites add own externs incrementally per-commit
 * (substrate-M1-only scope per testkeeper 22:10:23Z). */
extern unsigned long phx_ptr_array_resize_count_inliner_excess_args;
extern unsigned long phx_ptr_array_resize_count_inliner_to_delete;
extern unsigned long phx_ptr_array_resize_count_inliner_ends;

typedef struct PhxPtrArray {
    void **data;
    size_t count;
    size_t capacity;
} PhxPtrArray;

static inline void phx_ptr_arr_init(PhxPtrArray *a) {
    a->data = NULL; a->count = 0; a->capacity = 0;
}
static inline void phx_ptr_arr_destroy(PhxPtrArray *a) {
    free(a->data); a->data = NULL; a->count = 0; a->capacity = 0;
}
static inline void phx_ptr_arr_push(PhxPtrArray *a, void *val) {
    if (a->count == a->capacity) {
        size_t new_cap = a->capacity ? a->capacity * 2 : 8;
        a->data = (void **)realloc(a->data, new_cap * sizeof(void *));
        a->capacity = new_cap;
        phx_ptr_array_resize_count++;
    }
    a->data[a->count++] = val;
}
static inline void *phx_ptr_arr_pop(PhxPtrArray *a) {
    return a->data[--a->count];
}
static inline void phx_ptr_arr_clear(PhxPtrArray *a) {
    a->count = 0;
}
static inline void phx_ptr_arr_resize(PhxPtrArray *a, size_t n) {
    if (n > a->capacity) {
        a->data = (void **)realloc(a->data, n * sizeof(void *));
        a->capacity = n;
        phx_ptr_array_resize_count++;
    }
    a->count = n;
}
static inline void phx_ptr_arr_reserve(PhxPtrArray *a, size_t n) {
    if (n > a->capacity) {
        a->data = (void **)realloc(a->data, n * sizeof(void *));
        a->capacity = n;
        phx_ptr_array_resize_count++;
    }
}
static inline void phx_ptr_arr_copy(PhxPtrArray *dst, const PhxPtrArray *src) {
    phx_ptr_arr_init(dst);
    if (src->count > 0) {
        phx_ptr_arr_reserve(dst, src->count);
        memcpy(dst->data, src->data, src->count * sizeof(void *));
        dst->count = src->count;
    }
}

#ifdef __cplusplus
} /* extern "C" */
#endif
