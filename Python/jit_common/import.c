/* Copyright (c) Meta Platforms, Inc. and affiliates. */

#include "cinderx/Common/import.h"

PyObject* _Ci_CreateBuiltinModule(PyModuleDef* def, const char* name) {
  PyObject *machinery = NULL, *spec_type = NULL, *module_name = NULL;
  PyObject *module_spec = NULL, *mod = NULL;
  PyObject *result = NULL;

  machinery = PyImport_ImportModule("importlib.machinery");
  if (machinery == NULL) {
    goto cleanup;
  }
  spec_type = PyObject_GetAttrString(machinery, "ModuleSpec");
  if (spec_type == NULL) {
    goto cleanup;
  }
  module_name = PyUnicode_FromString(name);
  if (module_name == NULL) {
    goto cleanup;
  }

  PyObject* args[] = {module_name, Py_None};
  module_spec = PyObject_Vectorcall(spec_type, args, 2, NULL);
  if (module_spec == NULL) {
    goto cleanup;
  }

  mod = PyModule_FromDefAndSpec(def, module_spec);
  if (mod == NULL) {
    goto cleanup;
  }

  if (PyModule_ExecDef(mod, def) < 0) {
    goto cleanup;
  }

  PyObject* modules = PyImport_GetModuleDict();
  if (PyDict_SetItem(modules, module_name, mod) < 0) {
    goto cleanup;
  }

  result = mod;
  mod = NULL; /* transferring ownership */

cleanup:
  Py_XDECREF(machinery);
  Py_XDECREF(spec_type);
  Py_XDECREF(module_name);
  Py_XDECREF(module_spec);
  Py_XDECREF(mod);
  return result;
}
