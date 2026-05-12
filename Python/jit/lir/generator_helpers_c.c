/*
 * generator_helpers_c.c -- pure-C ports of small predicate/lookup
 * helpers from lir::generator.cpp anonymous namespace.
 *
 * Phase 5.B c12: phx_bytes_from_cint_type — port of
 * generator.cpp:124-138 bytes_from_cint_type(Type type).
 *
 * Uses HIR_TYPE_CINT8/16/32/64 + HIR_TYPE_CUINT8/16/32/64 from
 * hir_type_c.h:185-192 (cross-validated against C++ Type::kCInt8 etc.
 * by hir_type_c_verify.cpp Phase 5.B c11) and the existing C-side
 * subtype query hir_type_is_subtype (hir_type_c.h:152). No new
 * bridges introduced.
 */

#include "cinderx/Jit/lir/generator_helpers_c.h"
#include "cinderx/Common/jit_log_c.h"  /* JIT_ABORT_C */

int phx_bytes_from_cint_type(HirType type) {
    HirType cint8   = HIR_TYPE_CINT8;
    HirType cuint8  = HIR_TYPE_CUINT8;
    HirType cint16  = HIR_TYPE_CINT16;
    HirType cuint16 = HIR_TYPE_CUINT16;
    HirType cint32  = HIR_TYPE_CINT32;
    HirType cuint32 = HIR_TYPE_CUINT32;
    HirType cint64  = HIR_TYPE_CINT64;
    HirType cuint64 = HIR_TYPE_CUINT64;

    if (hir_type_is_subtype(type, cint8) || hir_type_is_subtype(type, cuint8)) {
        return 1;
    } else if (hir_type_is_subtype(type, cint16) || hir_type_is_subtype(type, cuint16)) {
        return 2;
    } else if (hir_type_is_subtype(type, cint32) || hir_type_is_subtype(type, cuint32)) {
        return 3;
    } else if (hir_type_is_subtype(type, cint64) || hir_type_is_subtype(type, cuint64)) {
        return 4;
    }
    JIT_ABORT_C("Bad primitive int type: bits_and_flags=%llu",
                (unsigned long long)type.bits_and_flags);
    return 0; /* NOTREACHED */
}

/* Phase 5.B c13: port of generator.cpp:118-123
 * isTypeWithReasonablePointerEq(Type t). Tests t against 13 type
 * constants — returns 1 if t has reasonable Python == semantics
 * (object identity implies equality), 0 otherwise. Floats excluded
 * because NaN != NaN; containers of floats included because
 * PyObject_RichCompareBool short-circuits on identity.
 *
 * Uses HIR_TYPE_TYPEEXACT (Phase 5.B c13 new constant) + 12 other
 * constants verified by hir_type_c_verify.cpp. */
int phx_is_type_with_reasonable_pointer_eq(HirType t) {
    HirType array       = HIR_TYPE_ARRAY;
    HirType bytesexact  = HIR_TYPE_BYTESEXACT;
    HirType dictexact   = HIR_TYPE_DICTEXACT;
    HirType listexact   = HIR_TYPE_LISTEXACT;
    HirType setexact    = HIR_TYPE_SETEXACT;
    HirType tupleexact  = HIR_TYPE_TUPLEEXACT;
    HirType typeexact   = HIR_TYPE_TYPEEXACT;
    HirType longexact   = HIR_TYPE_LONGEXACT;
    HirType bool_       = HIR_TYPE_BOOL;
    HirType func        = HIR_TYPE_FUNC;
    HirType gen         = HIR_TYPE_GEN;
    HirType nonetype    = HIR_TYPE_NONETYPE;
    HirType slice       = HIR_TYPE_SLICE;

    return hir_type_is_subtype(t, array)
        || hir_type_is_subtype(t, bytesexact)
        || hir_type_is_subtype(t, dictexact)
        || hir_type_is_subtype(t, listexact)
        || hir_type_is_subtype(t, setexact)
        || hir_type_is_subtype(t, tupleexact)
        || hir_type_is_subtype(t, typeexact)
        || hir_type_is_subtype(t, longexact)
        || hir_type_is_subtype(t, bool_)
        || hir_type_is_subtype(t, func)
        || hir_type_is_subtype(t, gen)
        || hir_type_is_subtype(t, nonetype)
        || hir_type_is_subtype(t, slice);
}
