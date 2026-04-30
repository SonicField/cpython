/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * typed_argument_c.c — Pure-C body for TypedArgument leaf-class methods.
 *
 * Phase 4.A W1 (theologian docs/tier7-phase4a-preanalysis-2026-04-30.md §3).
 * Header is C-clean (forward-decls only); this .c file holds the Python.h
 * include + body per feedback_no_pythonh_headers.
 */

#include "cinderx/Jit/hir/typed_argument_c.h"
#include "Python.h"

/* Pin: PHX_TYPED_ARGUMENT_THREAD_SAFE_FLAGS_MASK in the header must equal
 * the canonical Py_TPFLAGS_BASETYPE that hir.h's kThreadSafeFlagsMask uses.
 * If CPython renumbers Py_TPFLAGS_BASETYPE, this fires at compile-time. */
_Static_assert(PHX_TYPED_ARGUMENT_THREAD_SAFE_FLAGS_MASK ==
               (unsigned long)Py_TPFLAGS_BASETYPE,
               "phx_typed_argument: header mask diverged from "
               "Py_TPFLAGS_BASETYPE");

unsigned long phx_typed_argument_thread_safe_tp_flags(
    const struct _typeobject *pytype) {
    /* Mirrors hir.h kThreadSafeFlagsMask (= Py_TPFLAGS_BASETYPE). The
     * mask is the public ABI subset of tp_flags safe to read off-thread. */
    return ((const PyTypeObject *)pytype)->tp_flags &
           (unsigned long)Py_TPFLAGS_BASETYPE;
}

void phx_typed_argument_pytype_swap(
    struct _typeobject **slot, struct _typeobject *new_value) {
    /* Phase 4.A W7d Batch 76: the only refcount-touching primitive
     * extracted to C from TypedArgument::operator= + copy-ctor. The
     * GIL + ThreadedCompileSerialize guard remain caller-side (C++
     * RAII; structurally cannot port to C without exposing the
     * serialize machinery as a C bridge — out of scope per Q-W7-3
     * stay-C++ exception for genuinely-can't-port surface).
     *
     * NULL-safe both directions: Py_XDECREF / Py_XINCREF accept NULL. */
    Py_XDECREF(*(PyTypeObject **)slot);
    *slot = new_value;
    Py_XINCREF(*(PyTypeObject **)slot);
}
