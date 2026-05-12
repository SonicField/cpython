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
