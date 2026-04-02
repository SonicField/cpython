/*
 * iterator_types.c -- Cached iterator type pointers (pure C)
 *
 * Phase 3D conversion: iterator_types.cpp -> iterator_types.c
 * Caches PyTypeObject pointers for range, list, and tuple iterators
 * at JIT startup.  These are used by the HIR builder and simplifier
 * to specialise FOR_ITER.
 */

#include <Python.h>
#include <stddef.h>

PyTypeObject *jit_g_range_iterator_type = NULL;
PyTypeObject *jit_g_list_iterator_type = NULL;
PyTypeObject *jit_g_tuple_iterator_type = NULL;

void jit_init_iterator_types(void) {
    /* Get range_iterator type by creating a temporary range iterator.
     * This avoids referencing PyRangeIter_Type which is not exported
     * from the Python executable. */
    PyObject *range_obj = PyObject_CallFunction(
        (PyObject *)&PyRange_Type, "iii", 0, 1, 1);
    if (range_obj != NULL) {
        PyObject *iter_obj = PyObject_GetIter(range_obj);
        Py_DECREF(range_obj);
        if (iter_obj != NULL) {
            jit_g_range_iterator_type = Py_TYPE(iter_obj);
            Py_DECREF(iter_obj);
        } else {
            PyErr_Clear();
        }
    } else {
        PyErr_Clear();
    }

    /* Get list_iterator type by creating a temporary list iterator. */
    PyObject *list_obj = PyList_New(0);
    if (list_obj != NULL) {
        PyObject *list_iter_obj = PyObject_GetIter(list_obj);
        Py_DECREF(list_obj);
        if (list_iter_obj != NULL) {
            jit_g_list_iterator_type = Py_TYPE(list_iter_obj);
            Py_DECREF(list_iter_obj);
        } else {
            PyErr_Clear();
        }
    } else {
        PyErr_Clear();
    }

    /* Get tuple_iterator type by creating a temporary tuple iterator. */
    PyObject *tuple_obj = PyTuple_New(0);
    if (tuple_obj != NULL) {
        PyObject *tuple_iter_obj = PyObject_GetIter(tuple_obj);
        Py_DECREF(tuple_obj);
        if (tuple_iter_obj != NULL) {
            jit_g_tuple_iterator_type = Py_TYPE(tuple_iter_obj);
            Py_DECREF(tuple_iter_obj);
        } else {
            PyErr_Clear();
        }
    } else {
        PyErr_Clear();
    }
}
