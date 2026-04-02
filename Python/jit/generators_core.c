/*
 * generators_core.c -- JIT generator/coroutine type checks (pure C)
 *
 * Phase 3D conversion: generators_core.cpp -> generators_core.c
 * Provides JitGen_CheckExact, JitCoro_CheckExact, JitCoro_GetAwaitableIter.
 */

#include "cinderx/Jit/generators_core.h"
#include "cinderx/module_c_state.h"

#if PY_VERSION_HEX >= 0x030C0000

#include "internal/pycore_frame.h"
#include "jit_common/py-portability.h"

static int
jitgen_is_coroutine(PyObject *o)
{
    if (Py_TYPE(o) != Ci_JitGenType() && !PyGen_CheckExact(o)) {
        return 0;
    }
    PyGenObject *gen = (PyGenObject *)o;
#if PY_VERSION_HEX >= 0x030E0000
    _PyInterpreterFrame *gen_frame = &gen->gi_iframe;
#else
    _PyInterpreterFrame *gen_frame = (_PyInterpreterFrame *)(gen->gi_iframe);
#endif
    PyCodeObject *code = _PyFrame_GetCode(gen_frame);
    return (code->co_flags & CO_ITERABLE_COROUTINE) != 0;
}

int JitGen_CheckExact(PyObject *o) {
    return Py_TYPE(o) == Ci_JitGenType();
}

int JitCoro_CheckExact(PyObject *o) {
    return Py_TYPE(o) == Ci_JitCoroType();
}

PyObject *JitCoro_GetAwaitableIter(PyObject *o) {
    unaryfunc getter = NULL;
    PyTypeObject *ot;

    if (JitCoro_CheckExact(o) || PyCoro_CheckExact(o) ||
        jitgen_is_coroutine(o)) {
        return Py_NewRef(o);
    }

    ot = Py_TYPE(o);
    if (ot->tp_as_async != NULL) {
        getter = ot->tp_as_async->am_await;
    }
    if (getter != NULL) {
        PyObject *res = (*getter)(o);
        if (res != NULL) {
            if (JitCoro_CheckExact(res) || PyCoro_CheckExact(res) ||
                jitgen_is_coroutine(res)) {
#if PY_VERSION_HEX >= 0x030F0000
                PyErr_Format(
                    PyExc_TypeError,
                    "%T.__await__() must return an iterator, "
                    "not coroutine",
                    o);
#else
                PyErr_SetString(
                    PyExc_TypeError,
                    "__await__() returned a coroutine");
#endif
                Py_CLEAR(res);
            } else if (!PyIter_Check(res)) {
#if PY_VERSION_HEX >= 0x030F0000
                PyErr_Format(
                    PyExc_TypeError,
                    "%T.__await__() must return an iterator, "
                    "not %T",
                    o,
                    res);
#else
                PyErr_Format(
                    PyExc_TypeError,
                    "__await__() returned non-iterator "
                    "of type '%.100s'",
                    Py_TYPE(res)->tp_name);
#endif
                Py_CLEAR(res);
            }
        }
        return res;
    }

    PyErr_Format(
        PyExc_TypeError,
#if PY_VERSION_HEX >= 0x030E0000
        "'%.100s' object can't be awaited",
#else
        "object %.100s can't be used in 'await' expression",
#endif
        ot->tp_name);
    return NULL;
}

#endif /* PY_VERSION_HEX >= 0x030C0000 */
