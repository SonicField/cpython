// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cinderx/module_c_state.h"

#include "cinderx/Common/log.h"
#include "cinderx/module_state.h"

extern "C" {

#if PY_VERSION_HEX < 0x030F0000
vectorcallfunc Ci_PyFunction_Vectorcall;
#endif

// W-PHASE2-CODEGEN-SLOW Phase 2: cached pointers backing the inline
// Ci_JitGenType/Ci_JitCoroType/Ci_JitModule accessors in module_c_state.h.
// Updated at setter sites (setGenType/setCoroType/setModuleState) so the
// hot-path consumers (gen_data_footer.c, generators_core.c) read a single
// cached pointer instead of going through the cross-TU function call into
// cinderx::getModuleState()->genType() etc.
PyTypeObject* phx_jit_gen_type = nullptr;
PyTypeObject* phx_jit_coro_type = nullptr;
PyObject* phx_jit_module_obj = nullptr;

int Ci_Watchers_WatchDict(PyObject* dict) {
  return ci_watcher_state_watch_dict(&cinderx::getModuleState()->watcherState(), dict);
}

int Ci_Watchers_UnwatchDict(PyObject* dict) {
  return ci_watcher_state_unwatch_dict(&cinderx::getModuleState()->watcherState(), dict);
}

int Ci_Watchers_WatchType(PyTypeObject* type) {
  return ci_watcher_state_watch_type(&cinderx::getModuleState()->watcherState(), type);
}

int Ci_Watchers_UnwatchType(PyTypeObject* type) {
  return ci_watcher_state_unwatch_type(&cinderx::getModuleState()->watcherState(), type);
}

PyObject**
Ci_GetGlobalCache(PyObject* builtins, PyObject* globals, PyObject* key) {
  JIT_CHECK(
      PyDict_CheckExact(builtins),
      "Builtins should be a dict, but is actually a {}",
      Py_TYPE(builtins)->tp_name);
  JIT_CHECK(
      PyDict_CheckExact(globals),
      "Globals should be a dict, but is actually a {}",
      Py_TYPE(globals)->tp_name);
  JIT_CHECK(
      PyUnicode_CheckExact(key),
      "Dictionary key should be a string, but is actually a {}",
      Py_TYPE(key)->tp_name);

  return cinderx::getModuleState()->cacheManager()->getGlobalCache(
      (PyDictObject*)builtins, (PyDictObject*)globals, (PyUnicodeObject*)key);
}

PyObject** Ci_GetDictCache(PyObject* dict, PyObject* key) {
  return Ci_GetGlobalCache(dict, dict, key);
}

void Ci_free_jit_list_gen(PyGenObject* obj) {
  cinderx::getModuleState()->jitGenFreeList()->free(
      reinterpret_cast<PyObject*>(obj));
}

// Ci_JitGenType / Ci_JitCoroType / Ci_JitModule moved to static inline in
// module_c_state.h (W-PHASE2-CODEGEN-SLOW Phase 2; backed by phx_jit_*
// extern globals defined above).

} // extern "C"
