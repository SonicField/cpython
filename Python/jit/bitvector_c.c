/*
 * bitvector_c.c — Pure C bit vector implementation.
 *
 * Phase 3D: Replaces bitvector.cpp.
 * Uses PyMem_RawMalloc/Free for heap allocation (GIL-free).
 */

#include "cinderx/Jit/bitvector_c.h"
#include "Python.h"

#include <string.h>

static int is_short(size_t nb) {
    return nb <= PHX_BV_PTR_WIDTH;
}

static size_t num_chunks(size_t nb) {
    return nb / PHX_BV_PTR_WIDTH + (nb % PHX_BV_PTR_WIDTH != 0);
}

/* ---- Lifecycle ---- */

void phx_bv_init(PhxBitVector *bv, size_t nb) {
    bv->num_bits = nb;
    if (is_short(nb)) {
        bv->data.bits = 0;
    } else {
        size_t nc = num_chunks(nb);
        bv->data.chunks = (uint64_t *)PyMem_RawCalloc(nc, sizeof(uint64_t));
    }
}

void phx_bv_init_val(PhxBitVector *bv, size_t nb, uintptr_t val) {
    bv->num_bits = nb;
    bv->data.bits = val;
}

void phx_bv_copy(PhxBitVector *dst, const PhxBitVector *src) {
    dst->num_bits = src->num_bits;
    if (is_short(src->num_bits)) {
        dst->data.bits = src->data.bits;
    } else {
        size_t nc = num_chunks(src->num_bits);
        dst->data.chunks = (uint64_t *)PyMem_RawMalloc(nc * sizeof(uint64_t));
        memcpy(dst->data.chunks, src->data.chunks, nc * sizeof(uint64_t));
    }
}

void phx_bv_move(PhxBitVector *dst, PhxBitVector *src) {
    dst->num_bits = src->num_bits;
    dst->data = src->data;
    src->num_bits = 0;
}

void phx_bv_destroy(PhxBitVector *bv) {
    if (!is_short(bv->num_bits)) {
        PyMem_RawFree(bv->data.chunks);
        bv->data.chunks = NULL;
    }
    bv->num_bits = 0;
}

/* ---- Bit access ---- */

int phx_bv_get_bit(const PhxBitVector *bv, size_t bit) {
    if (is_short(bv->num_bits)) {
        return (bv->data.bits >> bit) & 1;
    }
    size_t idx = bit / PHX_BV_PTR_WIDTH;
    size_t off = bit % PHX_BV_PTR_WIDTH;
    return (bv->data.chunks[idx] >> off) & 1;
}

void phx_bv_set_bit(PhxBitVector *bv, size_t bit, int v) {
    if (is_short(bv->num_bits)) {
        uintptr_t b = (uintptr_t)1 << bit;
        bv->data.bits = v ? (bv->data.bits | b) : (bv->data.bits & ~b);
    } else {
        size_t idx = bit / PHX_BV_PTR_WIDTH;
        size_t off = bit % PHX_BV_PTR_WIDTH;
        uintptr_t b = (uintptr_t)1 << off;
        uint64_t *val = &bv->data.chunks[idx];
        *val = v ? (*val | b) : (*val & ~b);
    }
}

uint64_t phx_bv_get_chunk(const PhxBitVector *bv, size_t chunk) {
    if (is_short(bv->num_bits)) {
        return bv->data.bits;
    }
    return bv->data.chunks[chunk];
}

void phx_bv_set_chunk(PhxBitVector *bv, size_t chunk, uint64_t bits) {
    if (is_short(bv->num_bits)) {
        bv->data.bits = bits;
    } else {
        bv->data.chunks[chunk] = bits;
    }
}

/* ---- Bulk operations ---- */

void phx_bv_reset_all(PhxBitVector *bv) {
    if (is_short(bv->num_bits)) {
        bv->data.bits = 0;
    } else {
        size_t nc = num_chunks(bv->num_bits);
        memset(bv->data.chunks, 0, nc * sizeof(uint64_t));
    }
}

void phx_bv_fill(PhxBitVector *bv, int v) {
    if (!v) {
        phx_bv_reset_all(bv);
        return;
    }
    if (is_short(bv->num_bits)) {
        if (bv->num_bits == PHX_BV_PTR_WIDTH) {
            bv->data.bits = (uintptr_t)-1;
        } else {
            bv->data.bits = ((uintptr_t)1 << bv->num_bits) - 1;
        }
    } else {
        size_t nc = num_chunks(bv->num_bits);
        for (size_t i = 0; i < nc - 1; ++i) {
            bv->data.chunks[i] = (uint64_t)-1;
        }
        size_t remainder = bv->num_bits % PHX_BV_PTR_WIDTH;
        if (remainder == 0) {
            bv->data.chunks[nc - 1] = (uint64_t)-1;
        } else {
            bv->data.chunks[nc - 1] = ((uint64_t)1 << remainder) - 1;
        }
    }
}

/* ---- Binary operations ---- */

static PhxBitVector binary_op_new(const PhxBitVector *a, const PhxBitVector *b,
                                   uint64_t (*op)(uint64_t, uint64_t)) {
    PhxBitVector r;
    r.num_bits = a->num_bits;
    if (is_short(a->num_bits)) {
        r.data.bits = op(a->data.bits, b->data.bits);
    } else {
        size_t nc = num_chunks(a->num_bits);
        r.data.chunks = (uint64_t *)PyMem_RawMalloc(nc * sizeof(uint64_t));
        for (size_t i = 0; i < nc; i++) {
            r.data.chunks[i] = op(a->data.chunks[i], b->data.chunks[i]);
        }
    }
    return r;
}

static void binary_op_assign(PhxBitVector *dst, const PhxBitVector *src,
                              uint64_t (*op)(uint64_t, uint64_t)) {
    if (is_short(dst->num_bits)) {
        dst->data.bits = op(dst->data.bits, src->data.bits);
    } else {
        size_t nc = num_chunks(dst->num_bits);
        for (size_t i = 0; i < nc; i++) {
            dst->data.chunks[i] = op(dst->data.chunks[i], src->data.chunks[i]);
        }
    }
}

static uint64_t op_and(uint64_t a, uint64_t b) { return a & b; }
static uint64_t op_or(uint64_t a, uint64_t b) { return a | b; }
static uint64_t op_sub(uint64_t a, uint64_t b) { return a & ~b; }

PhxBitVector phx_bv_and(const PhxBitVector *a, const PhxBitVector *b) {
    return binary_op_new(a, b, op_and);
}
PhxBitVector phx_bv_or(const PhxBitVector *a, const PhxBitVector *b) {
    return binary_op_new(a, b, op_or);
}
PhxBitVector phx_bv_sub(const PhxBitVector *a, const PhxBitVector *b) {
    return binary_op_new(a, b, op_sub);
}

void phx_bv_and_assign(PhxBitVector *dst, const PhxBitVector *src) {
    binary_op_assign(dst, src, op_and);
}
void phx_bv_or_assign(PhxBitVector *dst, const PhxBitVector *src) {
    binary_op_assign(dst, src, op_or);
}
void phx_bv_sub_assign(PhxBitVector *dst, const PhxBitVector *src) {
    binary_op_assign(dst, src, op_sub);
}

/* ---- Queries ---- */

int phx_bv_equal(const PhxBitVector *a, const PhxBitVector *b) {
    if (a->num_bits != b->num_bits) return 0;
    if (is_short(a->num_bits)) {
        return a->data.bits == b->data.bits;
    }
    size_t nc = num_chunks(a->num_bits);
    return memcmp(a->data.chunks, b->data.chunks, nc * sizeof(uint64_t)) == 0;
}

size_t phx_bv_popcount(const PhxBitVector *bv) {
    if (is_short(bv->num_bits)) {
        return (size_t)__builtin_popcountll(bv->data.bits);
    }
    size_t count = 0;
    size_t nc = num_chunks(bv->num_bits);
    for (size_t i = 0; i < nc; i++) {
        count += (size_t)__builtin_popcountll(bv->data.chunks[i]);
    }
    return count;
}

int phx_bv_is_empty(const PhxBitVector *bv) {
    if (is_short(bv->num_bits)) {
        return bv->data.bits == 0;
    }
    size_t nc = num_chunks(bv->num_bits);
    for (size_t i = 0; i < nc; i++) {
        if (bv->data.chunks[i] != 0) return 0;
    }
    return 1;
}

size_t phx_bv_num_bits(const PhxBitVector *bv) {
    return bv->num_bits;
}

/* ---- Resize ---- */

size_t phx_bv_add_bits(PhxBitVector *bv, size_t n) {
    size_t new_nb = bv->num_bits + n;
    phx_bv_set_width(bv, new_nb);
    return new_nb;
}

void phx_bv_set_width(PhxBitVector *bv, size_t new_size) {
    if (bv->num_bits == new_size) return;

    int old_short = is_short(bv->num_bits);
    bv->num_bits = new_size;
    int new_short = is_short(new_size);

    if (old_short && !new_short) {
        size_t nc = num_chunks(new_size);
        uintptr_t old_bits = bv->data.bits;
        bv->data.chunks = (uint64_t *)PyMem_RawCalloc(nc, sizeof(uint64_t));
        bv->data.chunks[0] = old_bits;
    } else if (!old_short && !new_short) {
        size_t nc = num_chunks(new_size);
        bv->data.chunks = (uint64_t *)PyMem_RawRealloc(
            bv->data.chunks, nc * sizeof(uint64_t));
    } else if (!old_short && new_short) {
        uint64_t low = bv->data.chunks[0];
        PyMem_RawFree(bv->data.chunks);
        bv->data.bits = low;
    }

    /* Clear unused upper bits. */
    size_t remainder = new_size % PHX_BV_PTR_WIDTH;
    if (remainder != 0) {
        uint64_t mask = ((uint64_t)1 << remainder) - 1;
        if (new_short) {
            bv->data.bits &= mask;
        } else {
            bv->data.chunks[num_chunks(new_size) - 1] &= mask;
        }
    }
}

/* ---- Iteration ---- */

void phx_bv_for_each_set_bit(const PhxBitVector *bv,
                              void (*callback)(size_t, void*),
                              void *ctx) {
    if (is_short(bv->num_bits)) {
        uint64_t chunk = bv->data.bits;
        while (chunk) {
            int bit = __builtin_ctzl(chunk);
            chunk ^= chunk & -(int64_t)chunk;
            callback((size_t)bit, ctx);
        }
    } else {
        size_t nc = num_chunks(bv->num_bits);
        size_t base = 0;
        for (size_t i = 0; i < nc; i++) {
            uint64_t chunk = bv->data.chunks[i];
            while (chunk) {
                int bit = __builtin_ctzl(chunk);
                chunk ^= chunk & -(int64_t)chunk;
                callback(base + (size_t)bit, ctx);
            }
            base += PHX_BV_PTR_WIDTH;
        }
    }
}
