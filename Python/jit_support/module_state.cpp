// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cinderx/module_state.h"

#include "cinderx/Common/log.h"
#include "cinderx/module_c_state.h"

namespace cinderx {

// W-PHASE2-CODEGEN-SLOW Phase 1: definition of the global ModuleState
// pointer declared extern in module_state.h.  getModuleState() is inlined
// in the header (one load per call) instead of called across the TU
// boundary.  setModuleState/removeModuleState below mutate this storage.
namespace detail {
ModuleState* s_cinderx_state;
} // namespace detail

int ModuleState::traverse(visitproc visit, void* arg) {
  Py_VISIT(coro_type_);
  Py_VISIT(gen_type_);
  Py_VISIT(anext_awaitable_type_);
  Py_VISIT(sys_clear_caches_);
  Py_VISIT(builtin_next_);
  Py_VISIT(orig_sys_monitoring_register_callback_);
  Py_VISIT(orig_sys_setprofile_);
  Py_VISIT(orig_sys_settrace_);
#if PY_VERSION_HEX < 0x030E0000
  Py_VISIT(frame_reifier_);
#endif
  for (auto& [type, members] : builtin_members_) {
    Py_VISIT(members);
  }
  return 0;
}

int ModuleState::clear() {
  coro_type_.reset();
  gen_type_.reset();
  anext_awaitable_type_.reset();
  sys_clear_caches_.reset();
  builtin_next_.reset();
  orig_sys_monitoring_register_callback_.reset();
  orig_sys_setprofile_.reset();
  orig_sys_settrace_.reset();
#if PY_VERSION_HEX < 0x030E0000
  frame_reifier_.reset();
#endif
  builtin_members_.clear();
  return 0;
}

void setModuleState(PyObject* mod) {
  auto state = reinterpret_cast<cinderx::ModuleState*>(PyModule_GetState(mod));
  detail::s_cinderx_state = state;
  state->setModule(mod);
  // W-PHASE2-CODEGEN-SLOW Phase 2: keep the C-side module-object cache in
  // sync with the singleton.  The gen/coro type caches are populated by
  // setGenType/setCoroType below.
  phx_jit_module_obj = mod;
}

ModuleState* getModuleState(PyObject* mod) {
  return reinterpret_cast<ModuleState*>(PyModule_GetState(mod));
}

void removeModuleState() {
  detail::s_cinderx_state = nullptr;
  // W-PHASE2-CODEGEN-SLOW Phase 2: nullify the inline-accessor caches so
  // post-shutdown reads return NULL (matching the pre-Phase-2 behavior of
  // getModuleState()->genType() once the singleton was cleared).
  phx_jit_gen_type = nullptr;
  phx_jit_coro_type = nullptr;
  phx_jit_module_obj = nullptr;
}

// W-PHASE2-CODEGEN-SLOW Phase 2 setter hooks.  Update both the C++
// member (consumed by ModuleState::genType()/coroType() getters) and
// the C-side cached pointer (consumed by Ci_JitGenType/Ci_JitCoroType
// inline accessors in module_c_state.h).
void ModuleState::setGenType(PyTypeObject* gen_type) {
  gen_type_ = Ref<PyTypeObject>::create(gen_type);
  phx_jit_gen_type = gen_type;
}

void ModuleState::setCoroType(PyTypeObject* coro_type) {
  coro_type_ = Ref<PyTypeObject>::create(coro_type);
  phx_jit_coro_type = coro_type;
}

bool ModuleState::initBuiltinMembers() {
#if PY_VERSION_HEX >= 0x030C0000
  PyTypeObject* types[] = {
      &PyBool_Type,
      &PyBytes_Type,
      &PyByteArray_Type,
      &PyComplex_Type,
      &PyCode_Type,
      &PyDict_Type,
      &PyFloat_Type,
      &PyFrozenSet_Type,
      &PyList_Type,
      &PyLong_Type,
      Py_TYPE(Py_None),
      &PyProperty_Type,
      &PySet_Type,
      &PyTuple_Type,
      &PyUnicode_Type,
  };

  for (auto type : types) {
    PyObject* mro = type->tp_mro;
    if (mro == nullptr) {
      continue;
    }

    Ref<> type_members = Ref<>::steal(PyDict_New());
    if (type_members == nullptr) {
      return false;
    }
    for (Py_ssize_t i = 0; i < Py_SIZE(mro); i++) {
      PyTypeObject* base =
          reinterpret_cast<PyTypeObject*>(PyTuple_GetItem(mro, i));
      Py_ssize_t cur_mem = 0;
      PyObject *key, *value;
      Ref<> tp_dict = Ref<>::steal(PyType_GetDict(base));
      while (PyDict_Next(tp_dict, &cur_mem, &key, &value)) {
        if (PyDict_Contains(type_members, key)) {
          continue;
        }
        if (PyDict_SetItem(type_members, key, value) < 0) {
          return false;
        }
      }
    }

    builtin_members_.emplace(type, std::move(type_members));
  }
#endif
  return true;
}

CiWatcherState& ModuleState::watcherState() {
  return watcher_state_;
}

jit::UnorderedSet<PyObject*>& ModuleState::registeredCompilationUnits() {
  return registered_compilation_units;
}

} // namespace cinderx

// C bridge for builtin_members_ lookup — see module_state.h declaration.
// W42-class Cat-A extraction (theologian 20:03Z + supervisor 20:04Z) to
// allow blme_helpers_c.c to access the cache from pure C without
// depending on the C++ unordered_map / Ref<> types.
extern "C" PyObject* phx_builtin_members_lookup(PyTypeObject* type, PyObject* name) {
    auto& builtins = cinderx::getModuleState()->builtinMembers();
    auto it = builtins.find(type);
    if (it == builtins.end()) {
        return nullptr;
    }
    return PyDict_GetItemWithError(it->second, name);
}
