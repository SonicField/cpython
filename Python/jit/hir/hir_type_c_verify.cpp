// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Phase 5.B c11: cross-validation static_asserts locking equivalence
// between HIR_TYPE_* C-side constants (hir_type_c.h:184-241) and the
// C++ Type::k<Name> | (Type::kLifetime<X> << SHIFT) bit-pattern
// encoding (type.h:33 Type::k*, type.h:188 toHirType encoding pattern).
//
// Foundation for c12+ generator.cpp/regalloc.cpp ports of HirType-
// touching predicates (e.g. isTypeWithReasonablePointerEq,
// bytes_from_cint_type) — without these static_asserts, the C-side
// constants are assumed to match C++ but unverified at compile-time.
// (Bridge work, per supervisor 22:38:02Z scope: ~13 static_asserts,
// no functional change.)

#include "cinderx/Jit/hir/hir_type_c.h"
#include "cinderx/Jit/hir/type.h"

namespace jit::hir {

// Helper: pack bits + lifetime into the HirType.bits_and_flags encoding
// per type.h:188-190 toHirType() formula. spec_kind defaults to 0 (no
// spec) since all our verified constants are spec-less Top-level types.
constexpr uint64_t hir_type_pack(uint64_t bits, uint64_t lifetime) {
    return bits
        | (lifetime << HIR_TYPE_LIFETIME_SHIFT)
        | (0ULL << HIR_TYPE_SPEC_SHIFT);
}

// ---- Top-level types (LifetimeTop, no spec) ----
static constexpr HirType k_c_bool = HIR_TYPE_BOOL;
static_assert(k_c_bool.bits_and_flags ==
    hir_type_pack(Type::kBool, Type::kLifetimeTop),
    "HIR_TYPE_BOOL must match C++ TBool encoding");

static constexpr HirType k_c_func = HIR_TYPE_FUNC;
static_assert(k_c_func.bits_and_flags ==
    hir_type_pack(Type::kFunc, Type::kLifetimeTop),
    "HIR_TYPE_FUNC must match C++ TFunc encoding");

static constexpr HirType k_c_gen = HIR_TYPE_GEN;
static_assert(k_c_gen.bits_and_flags ==
    hir_type_pack(Type::kGen, Type::kLifetimeTop),
    "HIR_TYPE_GEN must match C++ TGen encoding");

static constexpr HirType k_c_nonetype = HIR_TYPE_NONETYPE;
static_assert(k_c_nonetype.bits_and_flags ==
    hir_type_pack(Type::kNoneType, Type::kLifetimeTop),
    "HIR_TYPE_NONETYPE must match C++ TNoneType encoding");

static constexpr HirType k_c_slice = HIR_TYPE_SLICE;
static_assert(k_c_slice.bits_and_flags ==
    hir_type_pack(Type::kSlice, Type::kLifetimeTop),
    "HIR_TYPE_SLICE must match C++ TSlice encoding");

static constexpr HirType k_c_array = HIR_TYPE_ARRAY;
static_assert(k_c_array.bits_and_flags ==
    hir_type_pack(Type::kArray, Type::kLifetimeTop),
    "HIR_TYPE_ARRAY must match C++ TArray encoding");

// ---- Exact-type variants (LifetimeTop, no spec) ----
// Note: these are the *_EXACT bit-pattern types, distinct from spec-bearing
// TXxxExact constants (which carry SpecKind=kSpecTypeExact). Verifying the
// bit-pattern half of the C++ TXxxExact constants here.

static constexpr HirType k_c_bytesexact = HIR_TYPE_BYTESEXACT;
static_assert(k_c_bytesexact.bits_and_flags ==
    hir_type_pack(Type::kBytesExact, Type::kLifetimeTop),
    "HIR_TYPE_BYTESEXACT must match C++ TBytesExact bit-pattern");

static constexpr HirType k_c_dictexact = HIR_TYPE_DICTEXACT;
static_assert(k_c_dictexact.bits_and_flags ==
    hir_type_pack(Type::kDictExact, Type::kLifetimeTop),
    "HIR_TYPE_DICTEXACT must match C++ TDictExact bit-pattern");

static constexpr HirType k_c_listexact = HIR_TYPE_LISTEXACT;
static_assert(k_c_listexact.bits_and_flags ==
    hir_type_pack(Type::kListExact, Type::kLifetimeTop),
    "HIR_TYPE_LISTEXACT must match C++ TListExact bit-pattern");

static constexpr HirType k_c_setexact = HIR_TYPE_SETEXACT;
static_assert(k_c_setexact.bits_and_flags ==
    hir_type_pack(Type::kSetExact, Type::kLifetimeTop),
    "HIR_TYPE_SETEXACT must match C++ TSetExact bit-pattern");

static constexpr HirType k_c_tupleexact = HIR_TYPE_TUPLEEXACT;
static_assert(k_c_tupleexact.bits_and_flags ==
    hir_type_pack(Type::kTupleExact, Type::kLifetimeTop),
    "HIR_TYPE_TUPLEEXACT must match C++ TTupleExact bit-pattern");

static constexpr HirType k_c_longexact = HIR_TYPE_LONGEXACT;
static_assert(k_c_longexact.bits_and_flags ==
    hir_type_pack(Type::kLongExact, Type::kLifetimeTop),
    "HIR_TYPE_LONGEXACT must match C++ TLongExact bit-pattern");

// Phase 5.B c13: HIR_TYPE_TYPEEXACT added for c13
// phx_is_type_with_reasonable_pointer_eq port.
static constexpr HirType k_c_typeexact = HIR_TYPE_TYPEEXACT;
static_assert(k_c_typeexact.bits_and_flags ==
    hir_type_pack(Type::kTypeExact, Type::kLifetimeTop),
    "HIR_TYPE_TYPEEXACT must match C++ TTypeExact bit-pattern");

// ---- Primitive C-int types (LifetimeBottom, no spec) ----
// Used by bytes_from_cint_type (generator.cpp:124) — c12 port target.

static constexpr HirType k_c_cint8 = HIR_TYPE_CINT8;
static_assert(k_c_cint8.bits_and_flags ==
    hir_type_pack(Type::kCInt8, Type::kLifetimeBottom),
    "HIR_TYPE_CINT8 must match C++ TCInt8 encoding");

static constexpr HirType k_c_cint16 = HIR_TYPE_CINT16;
static_assert(k_c_cint16.bits_and_flags ==
    hir_type_pack(Type::kCInt16, Type::kLifetimeBottom),
    "HIR_TYPE_CINT16 must match C++ TCInt16 encoding");

static constexpr HirType k_c_cint32 = HIR_TYPE_CINT32;
static_assert(k_c_cint32.bits_and_flags ==
    hir_type_pack(Type::kCInt32, Type::kLifetimeBottom),
    "HIR_TYPE_CINT32 must match C++ TCInt32 encoding");

static constexpr HirType k_c_cint64 = HIR_TYPE_CINT64;
static_assert(k_c_cint64.bits_and_flags ==
    hir_type_pack(Type::kCInt64, Type::kLifetimeBottom),
    "HIR_TYPE_CINT64 must match C++ TCInt64 encoding");

static constexpr HirType k_c_cuint8 = HIR_TYPE_CUINT8;
static_assert(k_c_cuint8.bits_and_flags ==
    hir_type_pack(Type::kCUInt8, Type::kLifetimeBottom),
    "HIR_TYPE_CUINT8 must match C++ TCUInt8 encoding");

static constexpr HirType k_c_cuint16 = HIR_TYPE_CUINT16;
static_assert(k_c_cuint16.bits_and_flags ==
    hir_type_pack(Type::kCUInt16, Type::kLifetimeBottom),
    "HIR_TYPE_CUINT16 must match C++ TCUInt16 encoding");

static constexpr HirType k_c_cuint32 = HIR_TYPE_CUINT32;
static_assert(k_c_cuint32.bits_and_flags ==
    hir_type_pack(Type::kCUInt32, Type::kLifetimeBottom),
    "HIR_TYPE_CUINT32 must match C++ TCUInt32 encoding");

static constexpr HirType k_c_cuint64 = HIR_TYPE_CUINT64;
static_assert(k_c_cuint64.bits_and_flags ==
    hir_type_pack(Type::kCUInt64, Type::kLifetimeBottom),
    "HIR_TYPE_CUINT64 must match C++ TCUInt64 encoding");

} // namespace jit::hir
