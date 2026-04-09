/* hir_type_c.c — C implementations of HIR Type operations
 *
 * Phase B: implements hir::Type operator equivalents as C functions.
 * These match the C++ implementations in type.cpp line-for-line.
 *
 * Conversion order (per theologian spec):
 *   1. specSubtype (helper for intersect/subtract/subtype)
 *   2. intersect (operator&)
 *   3. subtract (operator-)
 *   4. union (operator|) — hardest, deferred
 */

#include "cinderx/Jit/hir/hir_type_c.h"
#include <string.h>  /* strcmp */

/* ---- Helper: pack bitfields into uint64_t ---- */

static inline uint64_t hir_type_pack(uint64_t bits, uint64_t lifetime,
                                      uint64_t spec_kind) {
    return (bits & HIR_TYPE_BITS_MASK)
         | ((lifetime & 0x3) << HIR_TYPE_LIFETIME_SHIFT)
         | ((spec_kind & 0x7) << HIR_TYPE_SPEC_SHIFT);
}

static inline HirType hir_type_make(uint64_t bits, uint64_t lifetime,
                                     uint64_t spec_kind, intptr_t spec_val) {
    HirType t;
    t.bits_and_flags = hir_type_pack(bits, lifetime, spec_kind);
    t.int_val = spec_val;
    return t;
}

static inline HirType hir_type_bottom(void) {
    return hir_type_make(0, 0, HIR_SPEC_BOTTOM, 0);
}

/* ---- specSubtype: is this type's specialization a subtype of other's? ---- */
/* Matches type.cpp:511-536 */

int hir_type_spec_subtype(const HirType *self, const HirType *other) {
    uint64_t other_sk = hir_type_spec_kind(other);
    uint64_t self_sk = hir_type_spec_kind(self);

    /* Top is supertype of everything; Bottom is subtype of everything */
    if (other_sk == HIR_SPEC_TOP || self_sk == HIR_SPEC_BOTTOM) {
        return 1;
    }

    /* Unspecialized (but not Bottom) is not a subtype of any specialized */
    if (!hir_type_has_spec(self)) {
        return 0;
    }

    /* Primitive specializations: exact equality only */
    if (hir_type_has_int_spec(self) || hir_type_has_int_spec(other) ||
        hir_type_has_double_spec(self) || hir_type_has_double_spec(other)) {
        return hir_type_equal(self, other);
    }

    /* Check other's specialization in decreasing specificity */
    if (hir_type_has_object_spec(other)) {
        return hir_type_has_object_spec(self) &&
               self->pyobject == other->pyobject;
    }
    if (hir_type_has_type_exact_spec(other)) {
        return hir_type_has_type_exact_spec(self) &&
               self->pytype == other->pytype;
    }

    /* Both have type specs — check PyType_IsSubtype */
    return PyType_IsSubtype(self->pytype, other->pytype);
}

/* ---- intersect (operator&) ---- */
/* Matches type.cpp:591-635 */

/* Type category bit masks — from type_generated.h, defined in bridge */
extern const uint64_t _hir_type_kObject;    /* 0x800ffffffff */
extern const uint64_t _hir_type_kPrimitive; /* 0xfff00000000 */

HirType hir_type_intersect(HirType a, HirType b) {
    uint64_t bits_a = hir_type_bits(&a);
    uint64_t bits_b = hir_type_bits(&b);
    uint64_t life_a = hir_type_lifetime(&a);
    uint64_t life_b = hir_type_lifetime(&b);

    uint64_t bits = bits_a & bits_b;
    uint64_t lifetime = life_a & life_b;

    /* kObject/lifetime coupling: both must be non-zero or both cleared */
    if ((bits & _hir_type_kObject) == 0) {
        lifetime = 0;  /* kLifetimeBottom */
    } else if (lifetime == 0) {
        bits &= ~_hir_type_kObject;
    }

    if (bits == 0) {
        return hir_type_bottom();
    }

    /* More-specific specialization wins */
    if (hir_type_spec_subtype(&a, &b)) {
        return hir_type_make(bits, lifetime,
                             hir_type_spec_kind(&a), a.int_val);
    }
    if (hir_type_spec_subtype(&b, &a)) {
        return hir_type_make(bits, lifetime,
                             hir_type_spec_kind(&b), b.int_val);
    }

    /* Multiple inheritance tiebreaker: alphabetical by tp_name */
    if (hir_type_spec_kind(&a) == HIR_SPEC_TYPE &&
        hir_type_spec_kind(&b) == HIR_SPEC_TYPE) {
        PyTypeObject *type_a = a.pytype;
        PyTypeObject *type_b = b.pytype;
        int cmp = strcmp(type_a->tp_name, type_b->tp_name);
        if (cmp < 0 || (cmp == 0 && type_a < type_b)) {
            return hir_type_make(bits, lifetime, HIR_SPEC_TYPE,
                                 (intptr_t)type_a);
        }
        return hir_type_make(bits, lifetime, HIR_SPEC_TYPE,
                             (intptr_t)type_b);
    }

    return hir_type_bottom();
}

/* ---- is_subtype (operator<=) ---- */
/* Matches type.cpp:506-508 */

int hir_type_is_subtype(HirType type, HirType supertype) {
    uint64_t bits_t = hir_type_bits(&type);
    uint64_t bits_s = hir_type_bits(&supertype);
    uint64_t life_t = hir_type_lifetime(&type);
    uint64_t life_s = hir_type_lifetime(&supertype);

    return (bits_t & bits_s) == bits_t &&
           (life_t & life_s) == life_t &&
           hir_type_spec_subtype(&type, &supertype);
}

/* ---- subtract (operator-) ---- */
/* Matches type.cpp:637-660 */

HirType hir_type_subtract(HirType a, HirType b) {
    /* If a <= b, result is Bottom */
    if (hir_type_is_subtype(a, b)) {
        return hir_type_bottom();
    }

    /* If specialization is not a subtype, return self unchanged */
    if (!hir_type_spec_subtype(&a, &b)) {
        return a;
    }

    uint64_t bits_a = hir_type_bits(&a);
    uint64_t bits_b = hir_type_bits(&b);
    uint64_t lifetime = hir_type_lifetime(&a);
    uint64_t life_b = hir_type_lifetime(&b);

    /* Remove primitive bits directly */
    uint64_t bits = bits_a & ~(bits_b & _hir_type_kPrimitive);

    /* bits_subset helper: (a & b) == a */
    #define BITS_SUBSET(a, b) (((a) & (b)) == (a))

    /* Remove kObject bits only if lifetime is subsumed */
    if (BITS_SUBSET(hir_type_lifetime(&a), life_b)) {
        bits &= ~(bits_b & _hir_type_kObject);
    }

    /* Clear lifetime only if kObject bits are subsumed */
    if (BITS_SUBSET(bits_a & _hir_type_kObject, bits_b & _hir_type_kObject)) {
        lifetime &= ~life_b;
    }

    #undef BITS_SUBSET

    return hir_type_make(bits, lifetime,
                         hir_type_spec_kind(&a), a.int_val);
}

/* ---- union (operator|) ---- */
/* Matches type.cpp:538-589 */

/* Bridge function: checks if PyTypeObject* is a known builtin type */
extern int _hir_type_is_builtin_pytype(PyTypeObject *type);  /* defined in bridge */

HirType hir_type_union(HirType a, HirType b) {
    /* Trivial specialization-preserving cases */
    if (hir_type_is_subtype(a, b)) {
        return b;
    }
    if (hir_type_is_subtype(b, a)) {
        return a;
    }

    uint64_t bits = hir_type_bits(&a) | hir_type_bits(&b);
    uint64_t lifetime = hir_type_lifetime(&a) | hir_type_lifetime(&b);

    HirType no_spec = hir_type_make(bits, lifetime, HIR_SPEC_TOP, 0);

    /* If either lacks a PyTypeObject* spec, result is unspecialized */
    if (!hir_type_has_type_spec(&a) || !hir_type_has_type_spec(&b)) {
        return no_spec;
    }

    /* Identical object specializations */
    if (hir_type_has_object_spec(&a) && hir_type_has_object_spec(&b) &&
        a.pyobject == b.pyobject) {
        return a;
    }

    PyTypeObject *type_a = a.pytype;
    PyTypeObject *type_b = b.pytype;
    PyTypeObject *supertype;

    if (PyType_IsSubtype(type_a, type_b)) {
        supertype = type_b;
    } else if (PyType_IsSubtype(type_b, type_a)) {
        supertype = type_a;
    } else {
        return no_spec;
    }

    /* If supertype is a builtin, bits already describe it */
    if (_hir_type_is_builtin_pytype(supertype)) {
        return no_spec;
    }

    /* Exact only if both exact AND same type */
    int is_exact = hir_type_has_type_exact_spec(&a) &&
                   hir_type_has_type_exact_spec(&b) &&
                   type_a == type_b;

    return hir_type_make(bits, lifetime,
                         is_exact ? HIR_SPEC_TYPE_EXACT : HIR_SPEC_TYPE,
                         (intptr_t)supertype);
}
