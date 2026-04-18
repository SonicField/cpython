/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C data structures for refcount insertion pass.
 */

#include "cinderx/Jit/hir/refcount_structs_c.h"
#include "Python.h"
#include <string.h>

/* ---- BorrowSupport ---- */

void phx_bs_clear(PhxBorrowSupport *bs) {
    if (bs->initialized) phx_bv_destroy(&bs->bits);
    bs->initialized = 0;
}

void phx_bs_init(PhxBorrowSupport *bs, size_t num_support_bits) {
    if (bs->initialized) phx_bv_destroy(&bs->bits);
    phx_bv_init(&bs->bits, num_support_bits);
    bs->initialized = 1;
}

int phx_bs_empty(const PhxBorrowSupport *bs) {
    if (!bs->initialized) return 1;
    return phx_bv_is_empty(&bs->bits);
}

int phx_bs_intersects_bs(const PhxBorrowSupport *a, const PhxBorrowSupport *b) {
    if (!a->initialized || !b->initialized) return 0;
    PhxBitVector tmp = phx_bv_and(&a->bits, &b->bits);
    int result = !phx_bv_is_empty(&tmp);
    phx_bv_destroy(&tmp);
    return result;
}

int phx_bs_intersects_acls(const PhxBorrowSupport *bs, uint64_t acls_bits) {
    if (!bs->initialized) return 0;
    return (phx_bv_get_chunk(&bs->bits, 0) & acls_bits) != 0;
}

int phx_bs_intersects_bit(const PhxBorrowSupport *bs, size_t bit) {
    if (!bs->initialized) return 0;
    return phx_bv_get_bit(&bs->bits, bit);
}

int phx_bs_equal(const PhxBorrowSupport *a, const PhxBorrowSupport *b) {
    if (!a->initialized && !b->initialized) return 1;
    if (!a->initialized || !b->initialized) return 0;
    return phx_bv_equal(&a->bits, &b->bits);
}

void phx_bs_add_bs(PhxBorrowSupport *dst, const PhxBorrowSupport *src) {
    if (!src->initialized) return;
    phx_bv_or_assign(&dst->bits, &src->bits);
}

void phx_bs_add_acls(PhxBorrowSupport *bs, uint64_t acls_bits) {
    uint64_t cur = phx_bv_get_chunk(&bs->bits, 0);
    phx_bv_set_chunk(&bs->bits, 0, cur | acls_bits);
}

void phx_bs_add_bit(PhxBorrowSupport *bs, size_t bit) {
    phx_bv_set_bit(&bs->bits, bit, 1);
}

void phx_bs_remove_acls(PhxBorrowSupport *bs, uint64_t acls_bits) {
    uint64_t cur = phx_bv_get_chunk(&bs->bits, 0);
    phx_bv_set_chunk(&bs->bits, 0, cur & ~acls_bits);
}

void phx_bs_remove_bit(PhxBorrowSupport *bs, size_t bit) {
    phx_bv_set_bit(&bs->bits, bit, 0);
}

void phx_bs_destroy(PhxBorrowSupport *bs) {
    if (bs->initialized) phx_bv_destroy(&bs->bits);
    bs->initialized = 0;
}

/* ---- RegState ---- */

void phx_rs_init(PhxRegState *rs, void *model) {
    memset(rs, 0, sizeof(*rs));
    rs->model = model;
    rs->kind = PHX_REF_UNCOUNTED;
    phx_rs_add_copy(rs, model);
}

void phx_rs_destroy(PhxRegState *rs) {
    PyMem_RawFree(rs->copies);
    phx_bs_destroy(&rs->support);
}

void *phx_rs_current(const PhxRegState *rs) {
    return rs->copies[rs->n_copies - 1];
}

void phx_rs_add_copy(PhxRegState *rs, void *copy) {
    if (rs->n_copies >= rs->cap_copies) {
        rs->cap_copies = rs->cap_copies ? rs->cap_copies * 2 : 4;
        rs->copies = (void **)PyMem_RawRealloc(rs->copies, rs->cap_copies * sizeof(void *));
    }
    rs->copies[rs->n_copies++] = copy;
}

int phx_rs_kill_copy(PhxRegState *rs, void *copy) {
    for (size_t i = 0; i < rs->n_copies; i++) {
        if (rs->copies[i] == copy) {
            memmove(&rs->copies[i], &rs->copies[i + 1],
                    (rs->n_copies - i - 1) * sizeof(void *));
            rs->n_copies--;
            return rs->n_copies == 0;
        }
    }
    return 0;
}

int phx_rs_num_copies(const PhxRegState *rs) { return (int)rs->n_copies; }
void *phx_rs_copy(const PhxRegState *rs, size_t i) { return rs->copies[i]; }

void phx_rs_merge(PhxRegState *dst, const PhxRegState *from) {
    if (dst->kind == from->kind) {
        if (dst->kind == PHX_REF_BORROWED) {
            phx_bs_add_bs(&dst->support, &from->support);
        }
    } else if (dst->kind == PHX_REF_UNCOUNTED) {
        dst->kind = from->kind;
        phx_bs_destroy(&dst->support);
        dst->support = from->support;
        dst->support.initialized = from->support.initialized;
        if (from->support.initialized) {
            phx_bv_init(&dst->support.bits, from->support.bits.num_bits);
            phx_bv_reset_all(&dst->support.bits);
            phx_bv_or_assign(&dst->support.bits, &from->support.bits);
        }
    } else if (from->kind == PHX_REF_UNCOUNTED) {
        /* keep dst as-is */
    } else {
        phx_rs_set_owned(dst);
    }
}

void phx_rs_set_uncounted(PhxRegState *rs) {
    rs->kind = PHX_REF_UNCOUNTED;
    phx_bs_clear(&rs->support);
}

void phx_rs_set_borrowed(PhxRegState *rs, size_t num_support_bits) {
    rs->kind = PHX_REF_BORROWED;
    phx_bs_init(&rs->support, num_support_bits);
}

void phx_rs_set_owned(PhxRegState *rs) {
    rs->kind = PHX_REF_OWNED;
    phx_bs_clear(&rs->support);
}

int phx_rs_equal(const PhxRegState *a, const PhxRegState *b) {
    if (a->model != b->model) return 0;
    if (a->kind != b->kind) return 0;
    if (a->n_copies != b->n_copies) return 0;
    for (size_t i = 0; i < a->n_copies; i++) {
        if (a->copies[i] != b->copies[i]) return 0;
    }
    return phx_bs_equal(&a->support, &b->support);
}

/* ---- StateMap ---- */

static size_t sm_hash(const void *p) {
    uintptr_t v = (uintptr_t)p;
    return (size_t)((v >> 4) ^ (v >> 16));
}

void phx_sm_init(PhxStateMap *sm) {
    sm->capacity = 32;
    sm->count = 0;
    sm->keys = (void **)PyMem_RawCalloc(sm->capacity, sizeof(void *));
    sm->values = (PhxRegState *)PyMem_RawCalloc(sm->capacity, sizeof(PhxRegState));
}

void phx_sm_destroy(PhxStateMap *sm) {
    for (size_t i = 0; i < sm->capacity; i++) {
        if (sm->keys[i]) phx_rs_destroy(&sm->values[i]);
    }
    PyMem_RawFree(sm->keys);
    PyMem_RawFree(sm->values);
    sm->keys = NULL;
    sm->values = NULL;
    sm->capacity = 0;
    sm->count = 0;
}

void phx_sm_copy(PhxStateMap *dst, const PhxStateMap *src) {
    phx_sm_init(dst);
    for (size_t i = 0; i < src->capacity; i++) {
        if (!src->keys[i]) continue;
        void *model = src->keys[i];
        const PhxRegState *srs = &src->values[i];
        PhxRegState *drs = phx_sm_get_or_create(dst, model);
        drs->kind = srs->kind;
        /* Deep-copy copies array */
        PyMem_RawFree(drs->copies);
        drs->n_copies = srs->n_copies;
        drs->cap_copies = srs->n_copies ? srs->n_copies : 1;
        drs->copies = (void **)PyMem_RawMalloc(
            drs->cap_copies * sizeof(void *));
        memcpy(drs->copies, srs->copies,
               srs->n_copies * sizeof(void *));
        /* Deep-copy borrow support */
        phx_bs_destroy(&drs->support);
        drs->support.initialized = 0;
        if (srs->support.initialized) {
            drs->support.initialized = 1;
            phx_bv_copy(&drs->support.bits, &srs->support.bits);
        }
    }
}

static void sm_grow(PhxStateMap *sm);

PhxRegState *phx_sm_get(const PhxStateMap *sm, void *model) {
    size_t mask = sm->capacity - 1;
    size_t idx = sm_hash(model) & mask;
    for (;;) {
        if (sm->keys[idx] == model) return &((PhxStateMap *)sm)->values[idx];
        if (sm->keys[idx] == NULL) return NULL;
        idx = (idx + 1) & mask;
    }
}

PhxRegState *phx_sm_get_or_create(PhxStateMap *sm, void *model) {
    if (sm->count * 2 >= sm->capacity) sm_grow(sm);
    size_t mask = sm->capacity - 1;
    size_t idx = sm_hash(model) & mask;
    for (;;) {
        if (sm->keys[idx] == model) return &sm->values[idx];
        if (sm->keys[idx] == NULL) {
            sm->keys[idx] = model;
            phx_rs_init(&sm->values[idx], model);
            sm->count++;
            return &sm->values[idx];
        }
        idx = (idx + 1) & mask;
    }
}

int phx_sm_contains(const PhxStateMap *sm, void *model) {
    return phx_sm_get(sm, model) != NULL;
}

void phx_sm_erase(PhxStateMap *sm, void *model) {
    size_t mask = sm->capacity - 1;
    size_t idx = sm_hash(model) & mask;
    for (;;) {
        if (sm->keys[idx] == model) {
            phx_rs_destroy(&sm->values[idx]);
            sm->keys[idx] = NULL;
            memset(&sm->values[idx], 0, sizeof(PhxRegState));
            sm->count--;
            /* Rehash subsequent entries to fill the gap */
            size_t next = (idx + 1) & mask;
            while (sm->keys[next]) {
                void *k = sm->keys[next];
                PhxRegState v = sm->values[next];
                sm->keys[next] = NULL;
                memset(&sm->values[next], 0, sizeof(PhxRegState));
                sm->count--;
                /* Re-insert */
                size_t ni = sm_hash(k) & mask;
                while (sm->keys[ni]) ni = (ni + 1) & mask;
                sm->keys[ni] = k;
                sm->values[ni] = v;
                sm->count++;
                next = (next + 1) & mask;
            }
            return;
        }
        if (sm->keys[idx] == NULL) return;
        idx = (idx + 1) & mask;
    }
}

size_t phx_sm_size(const PhxStateMap *sm) { return sm->count; }
int phx_sm_empty(const PhxStateMap *sm) { return sm->count == 0; }

int phx_sm_equal(const PhxStateMap *a, const PhxStateMap *b) {
    if (a->count != b->count) return 0;
    for (size_t i = 0; i < a->capacity; i++) {
        if (!a->keys[i]) continue;
        PhxRegState *bv = phx_sm_get(b, a->keys[i]);
        if (!bv) return 0;
        if (!phx_rs_equal(&a->values[i], bv)) return 0;
    }
    return 1;
}

int phx_sm_entry_valid(const PhxStateMap *sm, size_t idx) {
    return idx < sm->capacity && sm->keys[idx] != NULL;
}
void *phx_sm_entry_key(const PhxStateMap *sm, size_t idx) { return sm->keys[idx]; }
PhxRegState *phx_sm_entry_value(PhxStateMap *sm, size_t idx) { return &sm->values[idx]; }

static void sm_grow(PhxStateMap *sm) {
    size_t old_cap = sm->capacity;
    void **old_keys = sm->keys;
    PhxRegState *old_vals = sm->values;

    sm->capacity = old_cap * 2;
    sm->keys = (void **)PyMem_RawCalloc(sm->capacity, sizeof(void *));
    sm->values = (PhxRegState *)PyMem_RawCalloc(sm->capacity, sizeof(PhxRegState));
    sm->count = 0;

    for (size_t i = 0; i < old_cap; i++) {
        if (old_keys[i]) {
            size_t mask = sm->capacity - 1;
            size_t idx = sm_hash(old_keys[i]) & mask;
            while (sm->keys[idx]) idx = (idx + 1) & mask;
            sm->keys[idx] = old_keys[i];
            sm->values[idx] = old_vals[i];
            memset(&old_vals[i], 0, sizeof(PhxRegState));
            sm->count++;
        }
    }
    PyMem_RawFree(old_keys);
    PyMem_RawFree(old_vals);
}
