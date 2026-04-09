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

/* ---- Bitfield layout constants ---- */
#define HIR_TYPE_BITS_WIDTH     44
#define HIR_TYPE_LIFETIME_WIDTH 2
#define HIR_TYPE_SPEC_WIDTH     3

#define HIR_TYPE_BITS_MASK      ((1ULL << HIR_TYPE_BITS_WIDTH) - 1)
#define HIR_TYPE_LIFETIME_SHIFT HIR_TYPE_BITS_WIDTH
#define HIR_TYPE_LIFETIME_MASK  (((1ULL << HIR_TYPE_LIFETIME_WIDTH) - 1) << HIR_TYPE_LIFETIME_SHIFT)
#define HIR_TYPE_SPEC_SHIFT     (HIR_TYPE_BITS_WIDTH + HIR_TYPE_LIFETIME_WIDTH)
#define HIR_TYPE_SPEC_MASK      (((1ULL << HIR_TYPE_SPEC_WIDTH) - 1) << HIR_TYPE_SPEC_SHIFT)

/* Lifetime constants */
#define HIR_LIFETIME_BOTTOM   0
#define HIR_LIFETIME_MORTAL   (1ULL << 0)
#define HIR_LIFETIME_IMMORTAL (1ULL << 1)
#define HIR_LIFETIME_TOP      (HIR_LIFETIME_MORTAL | HIR_LIFETIME_IMMORTAL)

/* Specialization kind constants */
#define HIR_SPEC_TOP        0
#define HIR_SPEC_TYPE       1
#define HIR_SPEC_TYPE_EXACT 2
#define HIR_SPEC_OBJECT     3
#define HIR_SPEC_INT        4
#define HIR_SPEC_DOUBLE     5
#define HIR_SPEC_BOTTOM     6

/* ---- Accessors (inline for performance) ---- */

static inline uint64_t hir_type_bits(const HirType *t) {
    return t->bits_and_flags & HIR_TYPE_BITS_MASK;
}

static inline uint64_t hir_type_lifetime(const HirType *t) {
    return (t->bits_and_flags & HIR_TYPE_LIFETIME_MASK) >> HIR_TYPE_LIFETIME_SHIFT;
}

static inline uint64_t hir_type_spec_kind(const HirType *t) {
    return (t->bits_and_flags & HIR_TYPE_SPEC_MASK) >> HIR_TYPE_SPEC_SHIFT;
}

/* ---- Query functions ---- */

/* Check if two types could overlap (non-empty intersection).
 * Equivalent to (a & b) != TBottom. */
static inline int hir_type_could_be(const HirType *a, const HirType *b) {
    return (hir_type_bits(a) & hir_type_bits(b)) != 0;
}

/* Check if type has a type specialization */
static inline int hir_type_has_type_spec(const HirType *t) {
    uint64_t sk = hir_type_spec_kind(t);
    return sk == HIR_SPEC_TYPE || sk == HIR_SPEC_TYPE_EXACT;
}

/* Check if type has an exact type specialization */
static inline int hir_type_has_type_exact_spec(const HirType *t) {
    return hir_type_spec_kind(t) == HIR_SPEC_TYPE_EXACT;
}

/* Check if type has an object specialization */
static inline int hir_type_has_object_spec(const HirType *t) {
    return hir_type_spec_kind(t) == HIR_SPEC_OBJECT;
}

/* Check if type has an int specialization */
static inline int hir_type_has_int_spec(const HirType *t) {
    return hir_type_spec_kind(t) == HIR_SPEC_INT;
}

/* Check if type has a double specialization */
static inline int hir_type_has_double_spec(const HirType *t) {
    return hir_type_spec_kind(t) == HIR_SPEC_DOUBLE;
}

/* Check if type has any non-trivial specialization */
static inline int hir_type_has_spec(const HirType *t) {
    uint64_t sk = hir_type_spec_kind(t);
    return sk != HIR_SPEC_TOP && sk != HIR_SPEC_BOTTOM;
}

/* Get the type specialization (PyTypeObject*), or NULL */
static inline PyTypeObject *hir_type_type_spec(const HirType *t) {
    if (!hir_type_has_type_spec(t) && !hir_type_has_type_exact_spec(t))
        return NULL;
    return t->pytype;
}

/* Get the object specialization (PyObject*), or NULL */
static inline PyObject *hir_type_object_spec(const HirType *t) {
    if (!hir_type_has_object_spec(t))
        return NULL;
    return t->pyobject;
}

/* Get the int specialization value (only valid when has_int_spec) */
static inline intptr_t hir_type_int_spec(const HirType *t) {
    return t->int_val;
}

/* Check if a type is a subtype of another (type <= supertype).
 * This is the C equivalent of Type::operator<=(). */
int hir_type_is_subtype(HirType type, HirType supertype);

/* Equality check */
static inline int hir_type_equal(const HirType *a, const HirType *b) {
    return a->bits_and_flags == b->bits_and_flags &&
           a->int_val == b->int_val;
}

/* Get a string representation of the type.
 * Writes to caller-provided buffer. Returns chars written. */
int hir_type_to_string(HirType type, char *buf, size_t len);

/* ---- Set operations ---- */

/* Specialization subtype check (helper for operators) */
int hir_type_spec_subtype(const HirType *self, const HirType *other);

/* Type intersection (operator&) */
HirType hir_type_intersect(HirType a, HirType b);

/* Type subtraction (operator-) */
HirType hir_type_subtract(HirType a, HirType b);

/* Type union (operator|) */
HirType hir_type_union(HirType a, HirType b);

/* Check if type has a value specialization (object, int, or double)
 * and is a subtype of the given type.
 * C equivalent of Type::hasValueSpec(). */
static inline int hir_type_has_value_spec(const HirType *t, HirType ty) {
    return (hir_type_has_object_spec(t) || hir_type_has_int_spec(t) ||
            hir_type_has_double_spec(t)) &&
           hir_type_is_subtype(*t, ty);
}

/* Check if a type has a known runtime destructor.
 * C equivalent of Type::runtimePyTypeDestructor().has_value().
 * Returns 1 if the type has a known exact PyTypeObject* (not NoneType),
 * 0 otherwise. Bridge function — calls C++ implementation. */
int hir_type_has_known_destructor(const HirType *t);

/* Get the runtime PyTypeObject* for an exact type.
 * C equivalent of Type::runtimePyType().
 * Returns the PyTypeObject* if the type is exact (hasTypeExactSpec or
 * a builtin exact type), NULL otherwise. Bridge function. */
PyTypeObject *hir_type_runtime_py_type(const HirType *t);

#ifdef __cplusplus
} /* extern "C" */
#endif
