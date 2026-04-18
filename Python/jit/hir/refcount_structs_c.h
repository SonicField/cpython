/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C data structures for refcount insertion pass.
 * Phase R1: BorrowSupport, RegState, StateMap.
 */
#pragma once

#include "cinderx/Jit/bitvector_c.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- RefKind enum ---- */
typedef enum {
    PHX_REF_UNCOUNTED = 0,
    PHX_REF_BORROWED = 1,
    PHX_REF_OWNED = 2,
} PhxRefKind;

/* ---- BorrowSupport ----
 * Bit vector where lowest AliasClass::kNumBits hold an AliasClass
 * and remaining bits represent Phi input/output registers. */
typedef struct {
    PhxBitVector bits;
    int initialized;
} PhxBorrowSupport;

void phx_bs_clear(PhxBorrowSupport *bs);
void phx_bs_init(PhxBorrowSupport *bs, size_t num_support_bits);
int phx_bs_empty(const PhxBorrowSupport *bs);
int phx_bs_intersects_bs(const PhxBorrowSupport *a, const PhxBorrowSupport *b);
int phx_bs_intersects_acls(const PhxBorrowSupport *bs, uint64_t acls_bits);
int phx_bs_intersects_bit(const PhxBorrowSupport *bs, size_t bit);
int phx_bs_equal(const PhxBorrowSupport *a, const PhxBorrowSupport *b);
void phx_bs_add_bs(PhxBorrowSupport *dst, const PhxBorrowSupport *src);
void phx_bs_add_acls(PhxBorrowSupport *bs, uint64_t acls_bits);
void phx_bs_add_bit(PhxBorrowSupport *bs, size_t bit);
void phx_bs_remove_acls(PhxBorrowSupport *bs, uint64_t acls_bits);
void phx_bs_remove_bit(PhxBorrowSupport *bs, size_t bit);
void phx_bs_destroy(PhxBorrowSupport *bs);

/* ---- RegState ----
 * State of a live value including copies, ref kind, and borrow support. */
typedef struct {
    void *model;            /* Register* — the original register */
    void **copies;          /* dynamic array of Register* copies */
    size_t n_copies;
    size_t cap_copies;
    PhxRefKind kind;
    PhxBorrowSupport support;
} PhxRegState;

void phx_rs_init(PhxRegState *rs, void *model);
void phx_rs_destroy(PhxRegState *rs);
void *phx_rs_current(const PhxRegState *rs);
void phx_rs_add_copy(PhxRegState *rs, void *copy);
int phx_rs_kill_copy(PhxRegState *rs, void *copy);
int phx_rs_num_copies(const PhxRegState *rs);
void *phx_rs_copy(const PhxRegState *rs, size_t i);
void phx_rs_merge(PhxRegState *dst, const PhxRegState *from);
void phx_rs_set_uncounted(PhxRegState *rs);
void phx_rs_set_borrowed(PhxRegState *rs, size_t num_support_bits);
void phx_rs_set_owned(PhxRegState *rs);
int phx_rs_equal(const PhxRegState *a, const PhxRegState *b);

/* ---- StateMap ----
 * Hash map from model Register* to PhxRegState. */
typedef struct {
    void **keys;            /* Register* keys */
    PhxRegState *values;    /* RegState values */
    size_t capacity;
    size_t count;
} PhxStateMap;

void phx_sm_init(PhxStateMap *sm);
void phx_sm_destroy(PhxStateMap *sm);
PhxRegState *phx_sm_get(const PhxStateMap *sm, void *model);
PhxRegState *phx_sm_get_or_create(PhxStateMap *sm, void *model);
int phx_sm_contains(const PhxStateMap *sm, void *model);
void phx_sm_erase(PhxStateMap *sm, void *model);
size_t phx_sm_size(const PhxStateMap *sm);
int phx_sm_empty(const PhxStateMap *sm);
int phx_sm_equal(const PhxStateMap *a, const PhxStateMap *b);

/* Iteration: call with idx=0, increment until idx >= capacity.
 * Returns 1 if entry is valid (key != NULL), 0 if empty slot. */
int phx_sm_entry_valid(const PhxStateMap *sm, size_t idx);
void *phx_sm_entry_key(const PhxStateMap *sm, size_t idx);
PhxRegState *phx_sm_entry_value(PhxStateMap *sm, size_t idx);

#ifdef __cplusplus
} /* extern "C" */
#endif
