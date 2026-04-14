/*
 * annotation_index_c.h — Pure C replacement for annotation_index.h
 *
 * Phase 3D: Replaces the C++ AnnotationIndex class with a C struct
 * and function API. Used by Preloader to look up type annotations
 * on function arguments.
 */
#pragma once

#include "Python.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to an annotation index. */
typedef struct HirAnnotationIndex HirAnnotationIndex;

/* Create an annotation index from a function's annotations.
 * Returns NULL if annotations are not available or the config
 * disables type annotation guards. Caller owns the result. */
HirAnnotationIndex *hir_annotation_index_from_function(
    PyFunctionObject *func);

/* Look up the annotation for a given name.
 * Returns a borrowed reference or NULL if not found. */
PyObject *hir_annotation_index_find(HirAnnotationIndex *index,
                                    PyObject *name);

/* Free an annotation index. Safe to call with NULL. */
void hir_annotation_index_destroy(HirAnnotationIndex *index);

#ifdef __cplusplus
}
#endif
