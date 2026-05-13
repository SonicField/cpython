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
#include "cinderx/Jit/frame_header.h"   /* jit_frame_header_size (Phase 5.B c14 amend) */
#include "cinderx/Jit/lir/lir_c_api.h"  /* jit_is_frame_mode_lightweight */

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

/* Phase 5.B c14: port of generator.cpp:161-179 frameOffsetBefore +
 * frameOffsetOf. Both compute byte offsets into the inlined-function
 * frame header chain via the FrameState linked-list (caller_state_ptr).
 *
 * NOTE: Python 3.11 branch (PY_VERSION_HEX < 0x030C0000) NOT ported.
 * Original C++ used kJITShadowFrameSize + inlineDepth() — kJITShadowFrameSize
 * has no C-side equivalent (defined in cinder/exports.h, only included on
 * 3.11). Phoenix targets Python 3.12.13 (CLAUDE.md); 3.11 branch is dead
 * code at compile time. If 3.11 is ever revived, c14 must be revisited
 * to expose kJITShadowFrameSize as C-side constant.
 *
 * AMEND (2026-05-13 00:31Z): replaced frame_asm_c_frame_header_size with
 * direct jit_frame_header_size call to match C++ frameHeaderSize semantics
 * exactly. frame_asm_c_frame_header_size (frame_asm_c.c:351) adds
 * +sizeof(void*) when ENABLE_SHADOW_FRAMES undef — used during code
 * emission, NOT during LIR-generation. Original C++ frameHeaderSize
 * (frame_header.h:67) does NOT add the offset. Initial port called the
 * asm-helper variant by mistake → +8-byte offsets in LIR → SIGSEGV in
 * test_phoenix_benchmark_correctness::test_fibonacci (recursive _fib
 * triggers BeginInlinedFunction codegen path).
 *
 * Direct call pattern matches C++ frameHeaderSize:
 *   jit_frame_header_size(code, lightweight, sizeof(FrameHeader),
 *                         sizeof(PyObject*))
 * On 3.12+, FrameHeader = union { PyFunctionObject*; uintptr_t rtfs; }
 * = sizeof(void*) (per frame_header.h note). */

/* Helper: matches C++ frameHeaderSize exactly. */
static inline int phx_frame_header_size(PyCodeObject *code) {
    return jit_frame_header_size(code,
                                 jit_is_frame_mode_lightweight(),
                                 (int)sizeof(void *) /* FrameHeader 3.12+ */,
                                 (int)sizeof(PyObject *));
}

Py_ssize_t phx_frame_offset_before(const HirBeginInlinedFunction *instr) {
    Py_ssize_t depth = 0;
    for (HirFrameStateLayout *frame =
             (HirFrameStateLayout *)instr->caller_state_ptr;
         frame != NULL;
         frame = frame->parent) {
        depth -= phx_frame_header_size((PyCodeObject *)frame->code);
    }
    return depth;
}

Py_ssize_t phx_frame_offset_of(const HirBeginInlinedFunction *instr) {
    /* HirBeginInlinedFunction.func corresponds to C++ instr->func_, a
     * PyFunctionObject*. C++ instr->code() returns
     * (PyCodeObject*)func_->func_code (hir.h:1663). */
    PyFunctionObject *func = (PyFunctionObject *)instr->func;
    PyCodeObject *code = (PyCodeObject *)func->func_code;
    return phx_frame_offset_before(instr) - phx_frame_header_size(code);
}
