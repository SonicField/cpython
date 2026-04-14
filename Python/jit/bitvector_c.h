/*
 * bitvector_c.h — Pure C bit vector implementation.
 *
 * Phase 3D: Replaces bitvector.cpp. Short vectors (≤64 bits) are stored
 * inline; longer vectors use a heap-allocated uint64_t array.
 *
 * The C++ BitVector class in bitvector.h wraps this struct.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PHX_BV_PTR_WIDTH (sizeof(void*) * 8)

typedef struct {
    size_t num_bits;
    union {
        uintptr_t bits;       /* short vector (≤ PTR_WIDTH bits) */
        uint64_t *chunks;     /* long vector, PyMem_RawMalloc'd */
    } data;
} PhxBitVector;

/* ---- Lifecycle ---- */

/* Initialize a bit vector with 'nb' bits, all zero. */
void phx_bv_init(PhxBitVector *bv, size_t nb);

/* Initialize with a value (short vectors only, nb ≤ PTR_WIDTH). */
void phx_bv_init_val(PhxBitVector *bv, size_t nb, uintptr_t val);

/* Deep copy src into dst (dst must be uninitialized or destroyed). */
void phx_bv_copy(PhxBitVector *dst, const PhxBitVector *src);

/* Move src into dst. src is left empty (num_bits=0). */
void phx_bv_move(PhxBitVector *dst, PhxBitVector *src);

/* Free any heap allocation. Safe to call on stack-allocated bv. */
void phx_bv_destroy(PhxBitVector *bv);

/* ---- Bit access ---- */

int phx_bv_get_bit(const PhxBitVector *bv, size_t bit);
void phx_bv_set_bit(PhxBitVector *bv, size_t bit, int v);
uint64_t phx_bv_get_chunk(const PhxBitVector *bv, size_t chunk);
void phx_bv_set_chunk(PhxBitVector *bv, size_t chunk, uint64_t bits);

/* ---- Bulk operations ---- */

void phx_bv_reset_all(PhxBitVector *bv);
void phx_bv_fill(PhxBitVector *bv, int v);

/* ---- Binary operations (return new bv — caller must destroy) ---- */

PhxBitVector phx_bv_and(const PhxBitVector *a, const PhxBitVector *b);
PhxBitVector phx_bv_or(const PhxBitVector *a, const PhxBitVector *b);
PhxBitVector phx_bv_sub(const PhxBitVector *a, const PhxBitVector *b);

/* ---- In-place binary operations ---- */

void phx_bv_and_assign(PhxBitVector *dst, const PhxBitVector *src);
void phx_bv_or_assign(PhxBitVector *dst, const PhxBitVector *src);
void phx_bv_sub_assign(PhxBitVector *dst, const PhxBitVector *src);

/* ---- Queries ---- */

int phx_bv_equal(const PhxBitVector *a, const PhxBitVector *b);
size_t phx_bv_popcount(const PhxBitVector *bv);
int phx_bv_is_empty(const PhxBitVector *bv);
size_t phx_bv_num_bits(const PhxBitVector *bv);

/* ---- Resize ---- */

size_t phx_bv_add_bits(PhxBitVector *bv, size_t n);
void phx_bv_set_width(PhxBitVector *bv, size_t new_size);

/* ---- Iteration ---- */

/* Call callback(bit_index, ctx) for each set bit. */
void phx_bv_for_each_set_bit(const PhxBitVector *bv,
                              void (*callback)(size_t, void*),
                              void *ctx);

#ifdef __cplusplus
}
#endif
