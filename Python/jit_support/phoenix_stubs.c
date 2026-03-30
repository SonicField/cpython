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

/* Use PyModule_Type as a stand-in so Ci_StrictModule_Check always returns 0 */
PyTypeObject Ci_StrictModule_Type = {0};

static int _phoenix_stub_init_strict_module_type = 0;

static void _ensure_strict_module_type(void) {
    if (!_phoenix_stub_init_strict_module_type) {
        /* Copy PyModule_Type but give it a different address so
           type checks against Ci_StrictModule_Type always fail */
        _phoenix_stub_init_strict_module_type = 1;
    }
}

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
int _Ci_ObjectKeyType = 0;

/* StaticPython exception type — alias to TypeError */
PyObject* CiExc_StaticTypeError = NULL;  /* Set to PyExc_TypeError at init */
