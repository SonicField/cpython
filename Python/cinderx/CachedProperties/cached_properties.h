// Phoenix stub for cinderx/CachedProperties/cached_properties.h
// CachedProperties is a CinderX feature not needed for Phoenix JIT
#ifndef Py_CACHED_PROPERTIES_H
#define Py_CACHED_PROPERTIES_H

#include "cinderx/python.h"

typedef struct {
  PyObject_HEAD
  PyObject* func;
  PyObject* name_or_descr;
} PyCachedPropertyDescrObject;

typedef struct {
  PyObject_HEAD
  PyObject* func;
  PyObject* name_or_descr;
} PyAsyncCachedPropertyDescrObject;

typedef struct {
  PyObject_HEAD
  PyObject* func;
  PyObject* name;
  PyObject* value;
} PyAsyncCachedClassPropertyDescrObject;

// Type spec and type objects declared but not defined — linking will reveal if any are actually used
extern PyType_Spec _PyCachedClassProperty_TypeSpec;
extern PyTypeObject PyAsyncCachedPropertyWithDescr_Type;
extern PyTypeObject PyCachedProperty_Type;
extern PyTypeObject PyCachedPropertyWithDescr_Type;
extern PyTypeObject PyAsyncCachedProperty_Type;
extern PyTypeObject PyAsyncCachedClassProperty_Type;

#endif /* !Py_CACHED_PROPERTIES_H */
