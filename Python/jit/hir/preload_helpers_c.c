/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Pure-C port of resolve_field_descr from preload.cpp:30-40
 * (Tier 7 Batch 2 Cat-A extraction per theologian 20:43:11Z +
 * 20:44:46Z, ZERO new bridges needed — hir_prim_type_to_type already
 * exists in Python/jit/hir/hir_type_c.h:265).
 */

#include "cinderx/Jit/hir/preload_helpers_c.h"
#include "cinderx/Jit/hir/hir_type_c.h"
#include "cinderx/StaticPython/classloader.h"

#include <stdio.h>
#include <stdlib.h>

void phx_resolve_field_descr(
    PyTupleObject* descr,
    Py_ssize_t* out_offset,
    HirType* out_type,
    PyUnicodeObject** out_name)
{
    int field_type;
    Py_ssize_t offset = _PyClassLoader_ResolveFieldOffset(
        (PyObject*)descr, &field_type);
    if (offset == -1) {
        /* Mirror the original C++ JIT_CHECK abort semantics: log and
         * abort on failure. The caller's contract requires success. */
        fprintf(stderr,
            "phx_resolve_field_descr: failed to resolve field descriptor\n");
        abort();
    }
    *out_offset = offset;
    *out_type = hir_prim_type_to_type(field_type);
    *out_name = (PyUnicodeObject*)PyTuple_GET_ITEM(
        (PyObject*)descr, PyTuple_GET_SIZE(descr) - 1);
}
