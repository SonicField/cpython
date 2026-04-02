/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C-compatible HIR Type struct — layout-compatible with C++ jit::hir::Type.
 * Type is a 16-byte value type with no virtual methods, no std::string,
 * no C++ containers — pure bitfield + union.
 */
#pragma once

#include "cinderx/python.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Type bitfield layout (must match type.h):
 * - bits_     : kNumTypeBits (44 bits)
 * - lifetime_ : 2 bits
 * - spec_kind_: 3 bits
 * - padding_  : 15 bits
 * Total: 64 bits = 8 bytes
 * Followed by 8-byte specialization union.
 * Grand total: 16 bytes. */

typedef struct {
    uint64_t bits_and_flags; /* packed bitfields */
    union {
        PyTypeObject *pytype;
        PyObject *pyobject;
        intptr_t int_val;
        double double_val;
        void *ptr;
    };
} HirType;

/* Note: The bitfield packing above is a SIMPLIFIED view.
 * The actual C++ Type uses named bitfields (bits_:44, lifetime_:2,
 * spec_kind_:3, padding_:15). For C, we treat the first 8 bytes
 * as an opaque uint64_t and provide accessor functions instead of
 * matching the bitfield layout (which is implementation-defined).
 *
 * Layout compatibility is verified via static_assert in the bridge. */

/* ---- Type query functions ---- */

/* Check if a type is a subtype of another (type <= supertype).
 * This is the C equivalent of Type::operator<=(). */
int hir_type_is_subtype(HirType type, HirType supertype);

/* Get a string representation of the type.
 * Writes to caller-provided buffer. Returns chars written. */
int hir_type_to_string(HirType type, char *buf, size_t len);

#ifdef __cplusplus
} /* extern "C" */
#endif
