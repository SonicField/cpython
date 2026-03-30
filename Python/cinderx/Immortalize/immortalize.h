// Phoenix stub for cinderx/Immortalize/immortalize.h
#pragma once

#include "cinderx/python.h"
#include <stdbool.h>

#if PY_VERSION_HEX >= 0x030C0000
#define IMMORTALIZE(OBJ) Py_SET_REFCNT((OBJ), _Py_IMMORTAL_REFCNT)
#endif

inline bool can_immortalize(PyObject*) { return false; }
inline bool immortalize(PyObject*) { return false; }
inline PyObject* immortalize_heap(PyObject*) { Py_RETURN_NONE; }
