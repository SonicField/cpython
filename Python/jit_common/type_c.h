/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C-compatible type utility functions.
 */
#pragma once

#include "cinderx/python.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Write the fully qualified type name (module:name) to buf.
 * Returns number of chars written (excluding NUL). */
int jit_type_fullname(PyTypeObject *type, char *buf, size_t len);

/* Get the type's __dict__ safely during threaded compilation.
 * For static builtin types during compile, uses the interpreter-specific
 * state to avoid races. */
PyObject* jit_get_borrowed_type_dict_safe(PyTypeObject *self);

/* Simulate _PyType_Lookup() without heap mutations.
 * May return NULL even if the name exists (conservative).
 * A non-NULL return should match _PyType_Lookup(). */
PyObject* jit_type_lookup_safe(PyTypeObject *type, PyObject *name);

/* Ensure the type has a valid version tag. Returns 1 on success, 0 on failure. */
int jit_ensure_version_tag(PyTypeObject *type);

#ifdef __cplusplus
} /* extern "C" */
#endif
