// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "cinderx/Jit/iterator_types.h"

namespace jit {

PyTypeObject* g_range_iterator_type = nullptr;
PyTypeObject* g_list_iterator_type = nullptr;
PyTypeObject* g_tuple_iterator_type = nullptr;

void init_iterator_types() {
  // Get range_iterator type by creating a temporary range iterator.
  // This avoids referencing PyRangeIter_Type which is not exported
  // from the Python executable (not accessible to dynamically loaded
  // extension modules like _cinderx.so).
  PyObject* range_obj = PyObject_CallFunction(
      reinterpret_cast<PyObject*>(&PyRange_Type), "iii", 0, 1, 1);
  if (range_obj != nullptr) {
    PyObject* iter_obj = PyObject_GetIter(range_obj);
    Py_DECREF(range_obj);
    if (iter_obj != nullptr) {
      g_range_iterator_type = Py_TYPE(iter_obj);
      Py_DECREF(iter_obj);
    } else {
      PyErr_Clear();
    }
  } else {
    PyErr_Clear();
  }

  // Get list_iterator type by creating a temporary list iterator.
  PyObject* list_obj = PyList_New(0);
  if (list_obj != nullptr) {
    PyObject* list_iter_obj = PyObject_GetIter(list_obj);
    Py_DECREF(list_obj);
    if (list_iter_obj != nullptr) {
      g_list_iterator_type = Py_TYPE(list_iter_obj);
      Py_DECREF(list_iter_obj);
    } else {
      PyErr_Clear();
    }
  } else {
    PyErr_Clear();
  }

  // Get tuple_iterator type by creating a temporary tuple iterator.
  PyObject* tuple_obj = PyTuple_New(0);
  if (tuple_obj != nullptr) {
    PyObject* tuple_iter_obj = PyObject_GetIter(tuple_obj);
    Py_DECREF(tuple_obj);
    if (tuple_iter_obj != nullptr) {
      g_tuple_iterator_type = Py_TYPE(tuple_iter_obj);
      Py_DECREF(tuple_iter_obj);
    } else {
      PyErr_Clear();
    }
  } else {
    PyErr_Clear();
  }
}

} // namespace jit
