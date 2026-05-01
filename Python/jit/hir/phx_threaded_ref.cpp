/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * phx_threaded_ref bridge implementation. Phase 4.X-full X3a per
 * supervisor 04:04:50Z (Q-X3-3) + theologian §6.3.
 *
 * Thin extern "C" wrappers over jit::ThreadedRef<>::incref/decref
 * (threaded_compile.h:307/337). NULL + immortal-object short-circuits
 * inherited from the C++ implementation.
 */
#include "cinderx/Jit/hir/phx_threaded_ref.h"
#include "cinderx/Jit/threaded_compile.h"

extern "C" void phx_threaded_incref(PyObject *obj) {
    jit::ThreadedRef<>::incref(obj);
}

extern "C" void phx_threaded_decref(PyObject *obj) {
    jit::ThreadedRef<>::decref(obj);
}
