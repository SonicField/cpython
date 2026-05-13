/*
 * generator_helpers_c.h -- C header for generator_helpers_c.c
 *
 * Phase 5.B c12: pure-C ports of small predicate/lookup helpers from
 * generator.cpp anonymous namespace, parallel to inliner_helpers_c.c
 * precedent (D-1776977744).
 */

#ifndef JIT_LIR_GENERATOR_HELPERS_C_H
#define JIT_LIR_GENERATOR_HELPERS_C_H

#include "Python.h"  /* Py_ssize_t */
#include "cinderx/Jit/hir/hir_type_c.h"
#include "cinderx/Jit/hir/hir_instr_c.h"  /* HirBeginInlinedFunction (Phase 5.B c14) */

#ifdef __cplusplus
extern "C" {
#endif

/* Phase 5.B c12: returns the byte-size encoding (1/2/3/4) for a primitive
 * C-int HirType (CInt8/CInt16/CInt32/CInt64 + CUInt counterparts).
 * Aborts via JIT_ABORT_C on a non-CInt type. Original C++ at
 * generator.cpp:124-138 (DELETED in c12). */
int phx_bytes_from_cint_type(HirType type);

/* Phase 5.B c13: predicate for "type with reasonable Python == semantics"
 * (object identity implies equality). True for most types but NOT for
 * floats (NaN != NaN). True for container types containing those floats
 * because PyObject_RichCompareBool short-circuits on identity. Original
 * C++ at generator.cpp:118-123 (DELETED in c13). Returns 1/0. */
int phx_is_type_with_reasonable_pointer_eq(HirType t);

/* Phase 5.B c14: frame-offset helpers for inlined-function lowering.
 * Original C++ at generator.cpp:161-179 (DELETED in c14). Caller bridges
 * via reinterpret_cast<const HirBeginInlinedFunction*> (sizeof-pinned at
 * hir_instr_c_verify.cpp:281). 3.11 / 3.12 PY_VERSION_HEX branches
 * preserved for source-level parity. */
Py_ssize_t phx_frame_offset_before(const HirBeginInlinedFunction *instr);
Py_ssize_t phx_frame_offset_of(const HirBeginInlinedFunction *instr);

#ifdef __cplusplus
}
#endif

#endif /* JIT_LIR_GENERATOR_HELPERS_C_H */
