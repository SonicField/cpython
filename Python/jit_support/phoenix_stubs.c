/*
 * Phoenix stubs for CinderX symbols not needed by the JIT.
 *
 * These are StaticPython, StrictModule, and CheckedDict/List APIs that the
 * JIT references but does not require for core JIT functionality.  When the
 * JIT encounters a StrictModule or CheckedDict it falls back to the
 * interpreter, which is correct behaviour for stock CPython.
 */

#include "Python.h"
#include <stdlib.h>

/* ---- StrictModule stubs ---- */

/* Ci_StrictModule_Type: a type with a valid tp_name but different address
   from PyModule_Type so Ci_StrictModule_Check always returns false. */
PyTypeObject Ci_StrictModule_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "phoenix.StrictModule",
    .tp_basicsize = sizeof(PyObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
};

PyObject* Ci_StrictModule_GetDict(PyObject* mod) {
    (void)mod;
    return NULL;
}

void* Ci_StrictModule_GetDictSetter(PyObject* mod) {
    (void)mod;
    return NULL;
}

int Ci_do_strictmodule_patch(PyObject* mod, PyObject* name, PyObject* value) {
    (void)mod; (void)name; (void)value;
    return -1;
}

PyObject* _Ci_CreateStaticModule(PyModuleDef* def) {
    return PyModule_Create(def);
}

/* ---- CheckedDict / CheckedList stubs (StaticPython) ---- */

int Ci_CheckedDict_TypeCheck(PyObject* type) {
    (void)type;
    return 0;  /* Never a CheckedDict in Phoenix */
}

int Ci_CheckedList_TypeCheck(PyObject* type) {
    (void)type;
    return 0;  /* Never a CheckedList in Phoenix */
}

PyObject* Ci_CheckedDict_New(PyObject* type) {
    (void)type;
    return PyDict_New();
}

PyObject* Ci_CheckedDict_NewPresized(PyObject* type, Py_ssize_t minused) {
    (void)type;
    return _PyDict_NewPresized(minused);
}

PyObject* Ci_CheckedList_New(PyObject* type, Py_ssize_t size) {
    (void)type;
    return PyList_New(size);
}

int Ci_DictOrChecked_SetItem(PyObject* op, PyObject* key, PyObject* value) {
    return PyDict_SetItem(op, key, value);
}

int Ci_ListOrCheckedList_Append(PyObject* op, PyObject* newitem) {
    return PyList_Append(op, newitem);
}

/* ---- Typed signature stubs ---- */

PyObject* Ci_meth_get__typed_signature__(PyObject* self, void* closure) {
    (void)self; (void)closure;
    Py_RETURN_NONE;
}

PyObject* Ci_method_get_typed_signature(PyObject* self, void* closure) {
    (void)self; (void)closure;
    Py_RETURN_NONE;
}

/* ---- Misc stubs ---- */

PyObject* Ci_static_rand(PyObject* self, PyObject* args) {
    (void)self; (void)args;
    long r = rand();
    return PyLong_FromLong(r);
}

/* Object key type — not meaningful without StaticPython */
PyTypeObject _Ci_ObjectKeyType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "phoenix.ObjectKeyType",
    .tp_basicsize = sizeof(PyObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
};

/* StaticPython exception type — alias to TypeError */
PyObject* CiExc_StaticTypeError = NULL;  /* Set to PyExc_TypeError at init */

/* ---- CachedProperties type stubs ---- */

PyTypeObject PyCachedProperty_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "phoenix.CachedProperty",
    .tp_basicsize = sizeof(PyObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
};
PyTypeObject PyCachedPropertyWithDescr_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "phoenix.CachedPropertyWithDescr",
    .tp_basicsize = sizeof(PyObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
};
PyTypeObject PyAsyncCachedProperty_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "phoenix.AsyncCachedProperty",
    .tp_basicsize = sizeof(PyObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
};
PyTypeObject PyAsyncCachedPropertyWithDescr_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "phoenix.AsyncCachedPropertyWithDescr",
    .tp_basicsize = sizeof(PyObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
};
PyTypeObject PyAsyncCachedClassProperty_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "phoenix.AsyncCachedClassProperty",
    .tp_basicsize = sizeof(PyObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
};
static PyType_Slot _PyCachedClassProperty_slots[] = {
    {0, NULL},
};
PyType_Spec _PyCachedClassProperty_TypeSpec = {
    .name = "phoenix._CachedClassProperty",
    .basicsize = sizeof(PyObject),
    .flags = Py_TPFLAGS_DEFAULT,
    .slots = _PyCachedClassProperty_slots,
};

/* ---- CheckedDict/List cache stubs ---- */

void _PyCheckedDict_ClearCaches(void) {}
void _PyCheckedList_ClearCaches(void) {}

/* ---- StaticPython ClassLoader stubs ---- */
/* These return "not a static type" / NULL / 0 so the JIT uses
   dynamic dispatch instead of static type specialization. */

PyObject* _PyClassLoader_Box(int type_code, long value) {
    (void)type_code;
    return PyLong_FromLong(value);
}

void _PyClassLoader_ClearCache(void) {}
void _PyClassLoader_ClearGenericTypes(void) {}
void _PyClassLoader_ClearValueCache(void) {}
void _PyClassLoader_ClearVtables(void) {}

PyObject* _PyClassLoader_GetCodeArgumentTypeDescrs(PyCodeObject* code) {
    (void)code;
    Py_RETURN_NONE;
}

PyObject* _PyClassLoader_GetCodeReturnTypeDescr(PyCodeObject* code) {
    (void)code;
    Py_RETURN_NONE;
}

PyObject* _PyClassLoader_GetReturnTypeDescr(PyFunctionObject* func) {
    (void)func;
    Py_RETURN_NONE;
}

int _PyClassLoader_GetTypeCode(PyTypeObject* type) {
    (void)type;
    return -1;  /* TYPED_OBJECT — not a primitive type */
}

PyObject* _PyClassLoader_GetTypedArgsInfo(PyCodeObject* code, int only_primitives) {
    (void)code; (void)only_primitives;
    return PyTuple_New(0);
}

PyObject* _PyClassLoader_GetTypedArgsInfoFromThunk(PyObject* thunk, PyObject* container, int only_primitives) {
    (void)thunk; (void)container; (void)only_primitives;
    return PyTuple_New(0);
}

int _PyClassLoader_HasPrimitiveArgs(PyCodeObject* code) {
    (void)code;
    return 0;
}

PyObject* _PyClassLoader_InvokeMethod(PyObject* obj, PyObject** args, Py_ssize_t nargs) {
    (void)obj; (void)args; (void)nargs;
    PyErr_SetString(PyExc_TypeError, "Static method invocation not supported in Phoenix");
    return NULL;
}

int _PyClassLoader_IsImmutable(PyObject* container) {
    (void)container;
    return 0;
}

int _PyClassLoader_IsPatchedThunk(PyObject* obj) {
    (void)obj;
    return 0;
}

PyObject* _PyClassloader_LookupSymbol(PyObject* qualname) {
    (void)qualname;
    Py_RETURN_NONE;
}

void _PyClassLoader_NotifyDictChange(PyDictObject* dict, PyObject* key) {
    (void)dict; (void)key;
}

int _PyClassLoader_ResolveFieldOffset(PyObject* path, int* field_type) {
    (void)path; (void)field_type;
    return -1;
}

PyObject* _PyClassLoader_ResolveFunction(PyObject* path, PyObject** container) {
    (void)path;
    if (container) *container = NULL;
    Py_RETURN_NONE;
}

PyObject* _PyClassLoader_ResolveIndirectPtr(PyObject* path) {
    (void)path;
    Py_RETURN_NONE;
}

PyObject* _PyClassLoader_ResolveMethod(PyObject* path) {
    (void)path;
    Py_RETURN_NONE;
}

int _PyClassLoader_ResolvePrimitiveType(PyObject* descr) {
    (void)descr;
    return -1;  /* TYPED_OBJECT */
}

PyObject* _PyClassLoader_ResolveReturnType(PyObject* func, int* optional, int* exact, int* func_flags) {
    (void)func;
    if (optional) *optional = 0;
    if (exact) *exact = 0;
    if (func_flags) *func_flags = 0;
    return (PyObject*)&PyBaseObject_Type;
}

PyObject* _PyClassLoader_ResolveType(PyObject* path) {
    (void)path;
    Py_RETURN_NONE;
}

/* ---- AsyncGen stubs ---- */
/* _PyAsyncGenValueWrapperNew is provided by CPython (Objects/genobject.c) */

/* ---- StaticArray type stub ---- */

PyTypeObject PyStaticArray_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "phoenix.StaticArray",
    .tp_basicsize = sizeof(PyObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
};
