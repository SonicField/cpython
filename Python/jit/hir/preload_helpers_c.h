/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Pure-C helpers for preload.cpp — Tier 7 Batch 2 Cat-A extraction.
 *
 * Per theologian 20:43:11Z + 20:44:46Z: discipline-of-shipping for the
 * 'easy wins exhausted' territory. ZERO new bridges + clean Cat-A is
 * rare among remaining preload.cpp helpers; most need C++ Ref / vector
 * / unique_ptr support. resolve_field_descr is the single clean target.
 */
#pragma once

#include "cinderx/Jit/hir/hir_type_c.h"
#include "cinderx/python.h"

#ifdef __cplusplus
extern "C" {
#endif

/* resolve_field_descr ported from preload.cpp:30-40.
 *
 * Resolves the typed-Python field descriptor `descr` (a tuple ending in
 * the field-name unicode) via _PyClassLoader_ResolveFieldOffset; on
 * success writes the byte offset, primitive type (as HirType), and the
 * field name into the out-pointers. Aborts via JIT_CHECK if the field
 * cannot be resolved (matches original .cpp invariant).
 *
 * The C++ caller (preload.cpp resolve_field_descr) wraps this and
 * converts HirType → C++ Type via Type::fromHirType to preserve the
 * existing FieldInfo return type. */
void phx_resolve_field_descr(
    PyTupleObject* descr,
    Py_ssize_t* out_offset,
    HirType* out_type,
    PyUnicodeObject** out_name);

#ifdef __cplusplus
}
#endif
