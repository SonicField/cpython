/* Copyright (c) Meta Platforms, Inc. and affiliates. */

#include "cinderx/Common/type_c.h"

#include "cinderx/python.h"

#if PY_VERSION_HEX >= 0x030C0000
#include "internal/pycore_typeobject.h"
#endif

#include "cinderx/Common/dict.h"
#include "cinderx/Common/jit_log_c.h"
#include "cinderx/Common/py-portability.h"
#include "cinderx/Jit/threaded_compile_c.h"
#include "cinderx/UpstreamBorrow/borrowed.h"

#include <stdio.h>
#include <string.h>

int jit_type_fullname(PyTypeObject *type, char *buf, size_t len) {
    PyObject *dict = _PyType_GetDict(type);
    PyObject *module_str =
        dict ? PyDict_GetItemString(dict, "__module__") : NULL;
    if (module_str != NULL && PyUnicode_Check(module_str)) {
        const char *mod = PyUnicode_AsUTF8(module_str);
        if (mod != NULL) {
            return snprintf(buf, len, "%s:%s", mod, type->tp_name);
        }
    }
    return snprintf(buf, len, "%s", type->tp_name);
}

#if PY_VERSION_HEX >= 0x030C0000
PyObject* jit_get_borrowed_type_dict_safe(PyTypeObject *self) {
    if (jit_compile_running() &&
        self->tp_flags & _Py_TPFLAGS_STATIC_BUILTIN) {
        PyInterpreterState *interp = jit_compile_interpreter();
        managed_static_type_state *state =
            Cix_PyStaticType_GetState(interp, self);
        return state->tp_dict;
    }
    return getBorrowedTypeDict(self);
}
#else
PyObject* jit_get_borrowed_type_dict_safe(PyTypeObject *self) {
    return getBorrowedTypeDict(self);
}
#endif

PyObject* jit_type_lookup_safe(PyTypeObject *type, PyObject *name) {
    JIT_CHECK_C(PyUnicode_CheckExact(name), "name must be a str");
    /* Silence false positive from TSAN when checking Py_TPFLAGS_READY. */
    JIT_COMPILE_GUARD();

    PyTupleObject *mro = (PyTupleObject *)type->tp_mro;
    Py_ssize_t n = PyTuple_GET_SIZE(mro);
    for (Py_ssize_t i = 0; i < n; i++) {
        PyTypeObject *base_ty = (PyTypeObject *)PyTuple_GET_ITEM(mro, i);
        PyObject *dict = jit_get_borrowed_type_dict_safe(base_ty);
        if (!PyType_HasFeature(base_ty, Py_TPFLAGS_READY) ||
            !hasOnlyUnicodeKeys(dict)) {
            return NULL;
        }
        PyObject *value = PyDict_GetItemWithError(dict, name);
        if (value != NULL) {
            return value;
        }
#if PY_VERSION_HEX < 0x030C0000
        JIT_CHECK_C(!PyErr_Occurred(),
                     "Thread-unsafe exception during type lookup");
#endif
    }
    return NULL;
}

int jit_ensure_version_tag(PyTypeObject *type) {
    JIT_CHECK_C(jit_compile_can_access_shared_data(),
                "Accessing type object needs lock");
    if (Ci_Type_HasValidVersionTag(type)) {
        return 1;
    }
    return PyUnstable_Type_AssignVersionTag(type) ? 1 : 0;
}
