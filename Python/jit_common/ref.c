// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cinderx/python.h"

#if PY_VERSION_HEX >= 0x030E0000
#include "internal/pycore_interp.h"
#endif

#if PY_VERSION_HEX >= 0x030C0000
#include "internal/pycore_pystate.h"
#endif

#if defined(Py_REF_DEBUG) && defined(Py_GIL_DISABLED)
#include "internal/pycore_tstate.h"
#endif

#ifdef Py_GIL_DISABLED

void incref_total(PyThreadState* tstate) {
#ifdef Py_REF_DEBUG
  _PyThreadStateImpl* tstate_impl = (_PyThreadStateImpl*)tstate;
  __atomic_fetch_add(&tstate_impl->reftotal, 1, __ATOMIC_RELAXED);
#else
  (void)tstate;
#endif
}

void decref_total(PyThreadState* tstate) {
#ifdef Py_REF_DEBUG
  _PyThreadStateImpl* tstate_impl = (_PyThreadStateImpl*)tstate;
  __atomic_fetch_sub(&tstate_impl->reftotal, 1, __ATOMIC_RELAXED);
#else
  (void)tstate;
#endif
}

#else

void incref_total(PyInterpreterState* interp) {
#ifdef Py_REF_DEBUG
#if PY_VERSION_HEX >= 0x030C0000
  interp->object_state.reftotal++;
#else
  _Py_RefTotal++;
#endif
#else
  (void)interp;
#endif
}

void decref_total(PyInterpreterState* interp) {
#ifdef Py_REF_DEBUG
#if PY_VERSION_HEX >= 0x030C0000
  interp->object_state.reftotal--;
#else
  _Py_RefTotal--;
#endif
#else
  (void)interp;
#endif
}

#endif
