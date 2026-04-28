// Copyright (c) Meta Platforms, Inc. and affiliates.

// C APIs to ModuleState.

#pragma once

#include "cinderx/python.h"

#ifdef __cplusplus
extern "C" {
#endif

// Copy of CPython's vectorcall implementation for PyFunctionObject.
#if PY_VERSION_HEX >= 0x030F0000

#include "internal/pycore_function.h"

#define Ci_PyFunction_Vectorcall _PyFunction_Vectorcall
#else
extern vectorcallfunc Ci_PyFunction_Vectorcall;
#endif

// WatcherState.

int Ci_Watchers_WatchDict(PyObject* dict);
int Ci_Watchers_UnwatchDict(PyObject* dict);

int Ci_Watchers_WatchType(PyTypeObject* type);
int Ci_Watchers_UnwatchType(PyTypeObject* type);

// GlobalCacheManager.

PyObject**
Ci_GetGlobalCache(PyObject* builtins, PyObject* globals, PyObject* key);

PyObject** Ci_GetDictCache(PyObject* dict, PyObject* key);

void Ci_free_jit_list_gen(PyGenObject* obj);

// JIT generator/coroutine type pointers — W-PHASE2-CODEGEN-SLOW Phase 2.
//
// Backing storage is set by setGenType/setCoroType/setModuleState (and
// nullified by removeModuleState).  Inline accessors below avoid the
// cross-TU function-call overhead at hot-path call sites
// (gen_data_footer.c per-yield, generators_core.c per-yield).
extern PyTypeObject* phx_jit_gen_type;
extern PyTypeObject* phx_jit_coro_type;
extern PyObject* phx_jit_module_obj;

static inline PyTypeObject* Ci_JitGenType(void) {
  return phx_jit_gen_type;
}

static inline PyTypeObject* Ci_JitCoroType(void) {
  return phx_jit_coro_type;
}

static inline PyObject* Ci_JitModule(void) {
  return phx_jit_module_obj;
}

#ifdef __cplusplus
} // extern "C"
#endif
