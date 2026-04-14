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
#include "cinderx/Jit/threaded_compile_c.h"
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
               hir_type_type_spec(self) == hir_type_type_spec(other);
    }

    /* Both have type specs — check PyType_IsSubtype */
    return PyType_IsSubtype(hir_type_type_spec(self), hir_type_type_spec(other));
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

    PyTypeObject *type_a = hir_type_type_spec(&a);
    PyTypeObject *type_b = hir_type_type_spec(&b);
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

/* ---- Type registration map (PyTypeObject* → HirType) ---- */

typedef struct {
    PyTypeObject **type_ptr;  /* pointer to the type object global */
    uint64_t bits;            /* HirType bits (without lifetime) */
} HirTypeMapEntry;

/* The registration table maps builtin Python types to their HIR Type bits.
 * Entries use pointers-to-pointers because some type objects (like Py_None's
 * type) are only available at runtime. */
static const struct {
    PyTypeObject *type;
    uint64_t bits;
} s_pytype_to_type[] = {
    { &PyBaseObject_Type,   0x000ffffffffULL },  /* TObject */
    { &PyBool_Type,         0x00000000002ULL },  /* TBool */
    { &PyBytes_Type,        0x00001002000ULL },  /* TBytes */
    { &PyCell_Type,         0x00000000004ULL },  /* TCell */
    { &PyCode_Type,         0x00000000008ULL },  /* TCode */
    { &PyDict_Type,         0x00002004000ULL },  /* TDict */
    /* BaseException: PyExc_BaseException resolved at runtime in lookup function */
    { &PyFloat_Type,        0x00004008000ULL },  /* TFloat */
    { &PyFrame_Type,        0x00000000010ULL },  /* TFrame */
    { &PyFunction_Type,     0x00000000020ULL },  /* TFunc */
    { &PyGen_Type,          0x00000000040ULL },  /* TGen */
    { &PyList_Type,         0x00008010000ULL },  /* TList */
    { &PyLong_Type,         0x00000200402ULL },  /* TLong */
    { &PySet_Type,          0x00010020000ULL },  /* TSet */
    { &PySlice_Type,        0x00000000100ULL },  /* TSlice */
    { &PyTuple_Type,        0x00020040000ULL },  /* TTuple */
    { &PyType_Type,         0x00040080000ULL },  /* TType */
    { &PyUnicode_Type,      0x00080100000ULL },  /* TUnicode */
    /* NoneType: Py_TYPE(Py_None) — resolved at runtime in lookup function */
};

#define PYTYPE_MAP_SIZE (sizeof(s_pytype_to_type) / sizeof(s_pytype_to_type[0]))

HirType hir_type_from_pytype(PyTypeObject *type, int is_exact) {
    /* Runtime-resolved types (not in the static table). */
    if (type == Py_TYPE(Py_None)) {
        HirType r = HIR_TYPE_NONETYPE;
        return r;
    }
    if (type == (PyTypeObject *)PyExc_BaseException) {
        HirType r = HIR_TYPE_BASEEXCEPTION;
        return r;
    }

    /* Linear scan — 18 entries, called infrequently. */
    for (size_t i = 0; i < PYTYPE_MAP_SIZE; i++) {
        if (s_pytype_to_type[i].type == type) {
            uint64_t bits = s_pytype_to_type[i].bits;
            HirType r;
            /* kLifetimeTop = 3 for object types */
            r.bits_and_flags = bits | (HIR_TYPE_LIFETIME_TOP << HIR_TYPE_LIFETIME_SHIFT);
            r.int_val = 0;
            return r;
        }
    }

    /* Not found in direct lookup — walk MRO to find base type. */
#define HIR_TYPE_KUSER 0x000ffe00000ULL  /* user-defined type bits mask */

    /* Ensure type is ready (has MRO). */
    jit_compile_lock();
    if (type->tp_mro == NULL && !(type->tp_flags & Py_TPFLAGS_READY)) {
        PyType_Ready(type);
    }
    jit_compile_unlock();

    if (type->tp_mro == NULL) {
        HirType r = HIR_TYPE_BOTTOM;
        return r;
    }

    /* Walk MRO: find first base type in our registration table. */
    PyObject *mro = type->tp_mro;
    Py_ssize_t mro_len = PyTuple_GET_SIZE(mro);
    for (Py_ssize_t i = 0; i < mro_len; i++) {
        PyTypeObject *base = (PyTypeObject *)PyTuple_GET_ITEM(mro, i);

        /* Check static table. */
        for (size_t j = 0; j < PYTYPE_MAP_SIZE; j++) {
            if (s_pytype_to_type[j].type == base) {
                uint64_t base_bits = s_pytype_to_type[j].bits;
                uint64_t user_bits = base_bits & HIR_TYPE_KUSER;
                HirType r;
                r.bits_and_flags = user_bits
                    | (HIR_TYPE_LIFETIME_TOP << HIR_TYPE_LIFETIME_SHIFT)
                    | (((uint64_t)(is_exact ? HIR_SPEC_TYPE_EXACT : HIR_SPEC_TYPE))
                       << HIR_TYPE_SPEC_SHIFT);
                r.pytype = type;
                return r;
            }
        }
        /* Check runtime-resolved types. */
        if (base == Py_TYPE(Py_None) || base == (PyTypeObject *)PyExc_BaseException) {
            uint64_t base_bits = (base == Py_TYPE(Py_None))
                ? 0x00000000080ULL : 0x00000801000ULL;
            uint64_t user_bits = base_bits & HIR_TYPE_KUSER;
            HirType r;
            r.bits_and_flags = user_bits
                | (HIR_TYPE_LIFETIME_TOP << HIR_TYPE_LIFETIME_SHIFT)
                | (((uint64_t)(is_exact ? HIR_SPEC_TYPE_EXACT : HIR_SPEC_TYPE))
                   << HIR_TYPE_SPEC_SHIFT);
            r.pytype = type;
            return r;
        }
    }

    /* Fallback — should not happen (object is always in MRO). */
    HirType r = HIR_TYPE_BOTTOM;
    return r;
}

/* ---- prim_type_to_type ---- */

/* TYPED_* constants from Static Python (classloader.h) */
#ifndef TYPED_BOOL
#define TYPED_BOOL    7
#define TYPED_CHAR   11
#define TYPED_INT8    1
#define TYPED_INT16   2
#define TYPED_INT32   3
#define TYPED_INT64   4
#define TYPED_UINT8   5
#define TYPED_UINT16  6
#define TYPED_UINT32   8
#define TYPED_UINT64   9
#define TYPED_OBJECT  0
#define TYPED_DOUBLE 10
#define TYPED_ERROR  12
#endif

HirType hir_prim_type_to_type(int prim_type) {
    switch (prim_type) {
        case TYPED_BOOL:   { HirType r = HIR_TYPE_CBOOL;   return r; }
        case TYPED_CHAR:
        case TYPED_INT8:   { HirType r = HIR_TYPE_CINT8;   return r; }
        case TYPED_INT16:  { HirType r = HIR_TYPE_CINT16;  return r; }
        case TYPED_INT32:  { HirType r = HIR_TYPE_CINT32;  return r; }
        case TYPED_INT64:  { HirType r = HIR_TYPE_CINT64;  return r; }
        case TYPED_UINT8:  { HirType r = HIR_TYPE_CUINT8;  return r; }
        case TYPED_UINT16: { HirType r = HIR_TYPE_CUINT16; return r; }
        case TYPED_UINT32: { HirType r = HIR_TYPE_CUINT32; return r; }
        case TYPED_UINT64: { HirType r = HIR_TYPE_CUINT64; return r; }
        case TYPED_OBJECT: { HirType r = HIR_TYPE_OPTOBJECT; return r; }
        case TYPED_DOUBLE: { HirType r = HIR_TYPE_CDOUBLE; return r; }
        case TYPED_ERROR:  { HirType r = HIR_TYPE_CINT32;  return r; }
        default:           { HirType r = HIR_TYPE_BOTTOM;  return r; }
    }
}

/* ---- Type query functions ---- */

PyTypeObject *hir_type_unique_pytype(const HirType *t) {
    if (hir_type_has_object_spec(t)) return NULL;
    if (hir_type_has_type_spec(t)) return t->pytype;
    if (hir_type_has_type_exact_spec(t)) return t->pytype;

    /* Reverse lookup: find matching bits in the type table. */
    uint64_t bits = hir_type_bits(t);

    /* Check NoneType */
    if (bits == 0x00000000080ULL) return Py_TYPE(Py_None);

    /* Check BaseException */
    if (bits == 0x00000801000ULL) return (PyTypeObject *)PyExc_BaseException;

    /* Scan static table */
    for (size_t i = 0; i < PYTYPE_MAP_SIZE; i++) {
        if (s_pytype_to_type[i].bits == bits) {
            return s_pytype_to_type[i].type;
        }
    }

    return NULL;
}

PyTypeObject *hir_type_runtime_pytype(const HirType *t) {
    if (!hir_type_has_type_exact_spec(t) &&
        !hir_type_is_exact(t)) {
        return NULL;
    }
    if (hir_type_has_type_spec(t) || hir_type_has_type_exact_spec(t)) {
        return t->pytype;
    }
    return hir_type_unique_pytype(t);
}

PyObject *hir_type_as_object(const HirType *t) {
    uint64_t bits = hir_type_bits(t);
    /* TNoneType bits = 0x80 — check if type is subtype of NoneType */
    if ((bits & 0x00000000080ULL) == bits && bits != 0) {
        return Py_None;
    }
    if (hir_type_has_object_spec(t)) {
        return t->pyobject;
    }
    return NULL;
}

int hir_type_is_single_value(const HirType *t) {
    uint64_t bits = hir_type_bits(t);
    /* TNoneType = 0x80, TNullptr = 0x80000000000 */
    if ((bits & 0x00000000080ULL) == bits && bits != 0) return 1;
    if ((bits & 0x80000000000ULL) == bits && bits != 0) return 1;
    if (hir_type_has_object_spec(t)) return 1;
    if (hir_type_has_int_spec(t)) return 1;
    if (hir_type_has_double_spec(t)) return 1;
    return 0;
}

/* ---- asBoxed / sizeInBytes ---- */

/* Helper: check if type a is subtype of type b (bits only, no spec). */
static int bits_subtype(uint64_t a_bits, uint64_t b_bits) {
    return (a_bits & b_bits) == a_bits && a_bits != 0;
}

HirType hir_type_as_boxed(const HirType *t) {
    uint64_t bits = hir_type_bits(t);
    if (bits_subtype(bits, 0x00100000000ULL)) { /* TCBool */
        HirType r = HIR_TYPE_BOOL; return r;
    }
    if (bits_subtype(bits, 0x1fe00000000ULL)) { /* TCInt */
        HirType r = HIR_TYPE_LONG; return r;
    }
    if (bits_subtype(bits, 0x40000000000ULL)) { /* TCDouble */
        HirType r = HIR_TYPE_FLOAT; return r;
    }
    /* No boxed equivalent — return Bottom. */
    HirType r = HIR_TYPE_BOTTOM;
    return r;
}

unsigned int hir_type_size_in_bytes(const HirType *t) {
    uint64_t bits = hir_type_bits(t);
    /* CBool|CInt8|CUInt8 = 0x100|0x200|0x2000 = ... check each */
    if (bits_subtype(bits, 0x00100000000ULL | 0x00200000000ULL | 0x02000000000ULL))
        return 1;
    if (bits_subtype(bits, 0x00400000000ULL | 0x04000000000ULL))
        return 2;
    if (bits_subtype(bits, 0x00800000000ULL | 0x08000000000ULL))
        return 4;
    /* 8-byte types: CInt64|CUInt64|CPtr|CDouble|Object|Nullptr */
    return 8;
}

/* ---- Factory: fromObject ---- */

HirType hir_type_from_object(PyObject *obj) {
    if (obj == Py_None) {
        /* NoneType — always immortal in 3.12+ */
        HirType r;
        r.bits_and_flags = 0x00000000080ULL |
            (HIR_TYPE_LIFETIME_TOP << HIR_TYPE_LIFETIME_SHIFT);
        r.int_val = 0;
        return r;
    }

    /* Determine lifetime. */
    uint64_t lifetime;
    jit_compile_lock();
    lifetime = _Py_IsImmortal(obj) ? 2ULL : 1ULL;  /* kImmortal=2, kMortal=1 */
    jit_compile_unlock();

    /* Get exact type. */
    HirType base = hir_type_from_pytype(Py_TYPE(obj), 1 /* exact */);
    uint64_t bits = hir_type_bits(&base);

    /* Create specialized object type. */
    HirType r;
    r.bits_and_flags = bits | (lifetime << HIR_TYPE_LIFETIME_SHIFT)
                     | (((uint64_t)HIR_SPEC_OBJECT) << HIR_TYPE_SPEC_SHIFT);
    r.pyobject = obj;
    return r;
}
