/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Pure-C helpers for builtin_load_method_elimination — Cat-A extraction
 * per W42-class scoping (theologian 20:03Z + supervisor 20:04Z, W27e
 * PARTIAL precedent). Reduces builtin_load_method_elimination.cpp size
 * by ~114 lines while leaving the heavier algorithmic methods
 * (tryEliminateLoadMethod + Run) as Cat-B accepted-residual.
 */
#pragma once

#include "cinderx/Jit/hir/hir_type_c.h"
#include "cinderx/python.h"

#ifdef __cplusplus
extern "C" {
#endif

/* immutableMultithreadedTypeLookup ported from builtin_load_method_
 * elimination.cpp:34-66 (3.12+ only). Walks the MRO looking for a
 * static-builtin type to consult phx_builtin_members_lookup, otherwise
 * walks immutable-exact types' tp_dict directly. Returns borrowed ref
 * or NULL.
 *
 * The 3.10 (#if PY_VERSION_HEX < 0x030C0000) branch from the original
 * .cpp is dropped — Phoenix is a 3.12-only project (consistent with
 * emitYieldValue precedent + memory key project state). */
PyObject* phx_immutable_multithreaded_type_lookup(PyTypeObject* type, PyObject* name);

/* getMethodObjectFromType ported from builtin_load_method_elimination.cpp:
 * 72-154. Returns a directly invokable method object from a HIR
 * receiver type, or NULL if the type can't be invoked directly.
 *
 * Mirrors the .cpp's #if PY_VERSION_HEX >= 0x030C0000 branch (3.12+);
 * the < 0x030C0000 branch is dropped per 3.12-only project scope. */
PyObject* phx_get_method_object_from_type(HirType receiver_type, PyObject* name);

#ifdef __cplusplus
}
#endif
