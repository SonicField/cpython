/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * typed_argument_c.h — Pure-C surface for TypedArgument leaf-class methods.
 *
 * Phase 4.A W1 (theologian docs/tier7-phase4a-preanalysis-2026-04-30.md §3).
 * Body lives in typed_argument_c.c which includes Python.h. This header
 * stays C-clean (per feedback_no_pythonh_headers); PyTypeObject* is
 * forward-declared as opaque struct.
 */

#ifndef CINDERX_JIT_HIR_TYPED_ARGUMENT_C_H
#define CINDERX_JIT_HIR_TYPED_ARGUMENT_C_H

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque forward decl: caller passes the PyTypeObject* it already holds. */
struct _typeobject;

/* Mask of tp_flags that are safe to read across concurrent compilation
 * threads. Mirrors C++ kThreadSafeFlagsMask in hir.h (= Py_TPFLAGS_BASETYPE).
 * Kept as macro here rather than including Python.h. Static-asserted equal
 * in typed_argument_c.c (which sees the canonical Py_TPFLAGS_BASETYPE). */
#define PHX_TYPED_ARGUMENT_THREAD_SAFE_FLAGS_MASK (1UL << 10)

/* Compute the cached thread_safe_flags value from a PyTypeObject*.
 * Pure read — no refcount, no GIL serialize. The C++ stub passes the
 * cached value alongside (caller-side debug invariant per hir.cpp:756-761). */
unsigned long phx_typed_argument_thread_safe_tp_flags(
    const struct _typeobject *pytype);

/* Refcount-aware pytype slot replacement (Phase 4.A W7d Batch 76).
 * Decref the current value at *slot, store new_value, incref the new.
 * Caller MUST hold the GIL + the ThreadedCompileSerialize guard
 * (TypedArgument::operator= and copy-ctor wrap this). Both XDECREF
 * and XINCREF are NULL-safe (Py_XDECREF / Py_XINCREF semantics). */
void phx_typed_argument_pytype_swap(
    struct _typeobject **slot, struct _typeobject *new_value);

#ifdef __cplusplus
}
#endif

#endif /* CINDERX_JIT_HIR_TYPED_ARGUMENT_C_H */
