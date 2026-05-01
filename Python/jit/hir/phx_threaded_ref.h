/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * phx_threaded_incref / phx_threaded_decref — extern "C" bridges over
 * jit::ThreadedRef<>::incref/decref (threaded_compile.h:307/337). Phase
 * 4.X-full X3a per supervisor 04:04:50Z (Q-X3-3) + theologian §6.3.
 *
 * Use-class: C-side substrate consumers (PhxPtrSet/PhxPtrMap holding
 * Python object pointers) need GIL-correct incref/decref operations.
 * The C++ ThreadedRef class handles GIL-disabled atomic-counter path
 * + standard Py_INCREF/DECREF; bridges expose this to C without
 * forcing C consumers to depend on Python.h or threaded_compile.h.
 *
 * Both functions are NULL-safe and immortal-safe (mirror ThreadedRef::
 * incref/decref short-circuits at threaded_compile.h:308/338).
 *
 * Implementation: phx_threaded_ref.cpp (extern "C" wrappers around
 * static C++ template methods). Adds C/C++ seam similar to existing
 * extern "C" hir_builder_emit_*_c bridges in builder.cpp.
 *
 * Q-X3-3 (supervisor 04:04:50Z): thin extern "C" approach;
 * ThreadedCompileContext stays C++ (Phoenix concurrency-infra class,
 * out of stay-C++ exception inventory).
 */
#pragma once

#include "Python.h"  /* PyObject opaque type for C ABI */

#ifdef __cplusplus
extern "C" {
#endif

/* GIL-correct refcount ops mirroring ThreadedRef::incref/decref. NULL +
 * immortal-object short-circuit per ThreadedRef contract. */
void phx_threaded_incref(PyObject *obj);
void phx_threaded_decref(PyObject *obj);

#ifdef __cplusplus
} /* extern "C" */
#endif
