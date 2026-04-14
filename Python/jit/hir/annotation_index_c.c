/*
 * annotation_index_c.c — Pure C implementation of annotation index.
 *
 * Phase 3D: Replaces annotation_index.cpp.
 * Looks up type annotations by name for emitting type guards.
 */

#include "cinderx/Jit/hir/annotation_index_c.h"
#include "cinderx/Jit/jit_config_c.h"

#include <stdlib.h>

struct HirAnnotationIndex {
    PyObject *annotations;  /* borrowed ref to tuple (pre-3.14) */
    PyObject *dict;         /* owned ref to dict, or NULL */
    Py_ssize_t size;        /* tuple size */
};

HirAnnotationIndex *hir_annotation_index_from_function(
    PyFunctionObject *func) {
    if (!jit_get_config()->emit_type_annotation_guards) {
        return NULL;
    }

#if PY_VERSION_HEX >= 0x030E0000
    PyObject *annotations = PyFunction_GetAnnotations((PyObject *)func);
    if (!PyDict_Check(annotations)) {
        return NULL;
    }
    HirAnnotationIndex *index = (HirAnnotationIndex *)malloc(
        sizeof(HirAnnotationIndex));
    if (!index) return NULL;
    index->annotations = NULL;
    index->size = 0;
    index->dict = annotations;
    Py_INCREF(index->dict);
    return index;
#else
    PyObject *ann = func->func_annotations;
    if (ann == NULL || !PyTuple_Check(ann)) {
        return NULL;
    }

    HirAnnotationIndex *index = (HirAnnotationIndex *)malloc(
        sizeof(HirAnnotationIndex));
    if (!index) return NULL;

    index->annotations = ann;  /* borrowed */
    index->size = PyTuple_GET_SIZE(ann);
    index->dict = NULL;

    /* For large annotation tuples, build a dict for O(1) lookup. */
    if (index->size >= 16) {
        index->dict = PyDict_New();
        if (index->dict) {
            for (Py_ssize_t i = 0; i < index->size; i += 2) {
                PyObject *key = PyTuple_GET_ITEM(ann, i);
                PyObject *value = PyTuple_GET_ITEM(ann, i + 1);
                PyDict_SetItem(index->dict, key, value);
            }
        }
    }

    return index;
#endif
}

PyObject *hir_annotation_index_find(HirAnnotationIndex *index,
                                    PyObject *name) {
    if (!index) return NULL;

    if (index->dict) {
        return PyDict_GetItem(index->dict, name);
    }

    for (Py_ssize_t i = 0; i < index->size; i += 2) {
        if (name == PyTuple_GET_ITEM(index->annotations, i)) {
            return PyTuple_GET_ITEM(index->annotations, i + 1);
        }
    }

    return NULL;
}

void hir_annotation_index_destroy(HirAnnotationIndex *index) {
    if (!index) return;
    Py_XDECREF(index->dict);
    free(index);
}
