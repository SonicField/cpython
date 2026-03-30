// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cinderx/module_state.h"

#include "cinderx/Common/log.h"

namespace cinderx {

namespace {

ModuleState* s_cinderx_state;

} // namespace

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

void setModuleState(BorrowedRef<> mod) {
  auto state = reinterpret_cast<cinderx::ModuleState*>(PyModule_GetState(mod));
  s_cinderx_state = state;
  state->setModule(mod);
}

ModuleState* getModuleState() {
  return s_cinderx_state;
}

ModuleState* getModuleState(BorrowedRef<> mod) {
  return reinterpret_cast<ModuleState*>(PyModule_GetState(mod));
}

void removeModuleState() {
  s_cinderx_state = nullptr;
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

WatcherState& ModuleState::watcherState() {
  return watcher_state_;
}

jit::UnorderedSet<BorrowedRef<>>& ModuleState::registeredCompilationUnits() {
  return registered_compilation_units;
}

} // namespace cinderx
