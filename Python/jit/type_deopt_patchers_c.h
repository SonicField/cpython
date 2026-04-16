/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C implementation of type deopt patcher logic.
 * Phase 3D: replaces C++ method bodies in type_deopt_patchers.cpp.
 */
#pragma once

#include "cinderx/python.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Callback type for shouldPatchForAttr — replaces C++ template+lambda.
 * Returns non-zero if the attr value should trigger a patch. */
typedef int (*jit_attr_check_fn)(PyObject *attr, void *ctx);

/* Core logic: check if attr change on type should trigger deopt patch.
 * Returns 1 if should patch, 0 otherwise. */
int jit_should_patch_for_attr(
    PyTypeObject *old_ty,
    PyTypeObject *new_ty,
    PyObject *attr_name,
    jit_attr_check_fn check,
    void *check_ctx);

/* TypeDeoptPatcher destructor logic — unwatch type from context. */
void jit_type_deopt_patcher_destroy(
    PyTypeObject *type,
    int is_linked,
    void *patcher);

/* TypeAttrDeoptPatcher::maybePatch — check attr identity. */
int jit_type_attr_patcher_maybe_patch(
    PyTypeObject *type,
    PyTypeObject *new_ty,
    PyObject *attr_name,
    PyObject *target_object);

/* SplitDictDeoptPatcher::maybePatch — check split dict keys. */
int jit_split_dict_patcher_maybe_patch(
    PyTypeObject *type,
    PyTypeObject *new_ty,
    PyObject *attr_name,
    PyDictKeysObject *keys);

#ifdef __cplusplus
} /* extern "C" */
#endif
