/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C implementation of type deopt patcher logic.
 * Phase 3D: replaces C++ method bodies in type_deopt_patchers.cpp.
 */

#include "cinderx/python.h"
#include "cinderx/Common/type_c.h"
#include "cinderx/Jit/type_deopt_patchers_c.h"

int jit_should_patch_for_attr(
    PyTypeObject *old_ty,
    PyTypeObject *new_ty,
    PyObject *attr_name,
    jit_attr_check_fn check,
    void *check_ctx) {
    if (new_ty != old_ty) {
        return 1;
    }
    PyObject *attr = jit_type_lookup_safe(new_ty, attr_name);
    return check(attr, check_ctx) || !PyUnstable_Type_AssignVersionTag(new_ty);
}

/* --- Extern C wrapper for Context::unwatchType --- */

/* Forward declaration — implemented in context.cpp */
extern void jit_context_unwatch_type(void *ctx, PyTypeObject *type, void *patcher);

void jit_type_deopt_patcher_destroy(
    PyTypeObject *type,
    int is_linked,
    void *patcher) {
    if (is_linked && type != NULL) {
        /* getContext() equivalent — implemented in context.cpp */
        extern void *jit_get_context_ptr(void);
        void *ctx = jit_get_context_ptr();
        if (ctx != NULL) {
            jit_context_unwatch_type(ctx, type, patcher);
        }
    }
}

/* --- TypeAttrDeoptPatcher --- */

static int type_attr_check(PyObject *attr, void *ctx) {
    PyObject *target = (PyObject *)ctx;
    return attr != target;
}

int jit_type_attr_patcher_maybe_patch(
    PyTypeObject *type,
    PyTypeObject *new_ty,
    PyObject *attr_name,
    PyObject *target_object) {
    return jit_should_patch_for_attr(
        type, new_ty, attr_name, type_attr_check, (void *)target_object);
}

/* --- SplitDictDeoptPatcher --- */

/* The split dict check needs both keys AND new_ty, so we use a struct. */
typedef struct {
    PyDictKeysObject *keys;
    PyTypeObject *new_ty;
} SplitDictCtx;

static int split_dict_check_full(PyObject *attr, void *ctx) {
    SplitDictCtx *sd = (SplitDictCtx *)ctx;
    if (attr != NULL) {
        return 1;
    }
    if (!PyType_HasFeature(sd->new_ty, Py_TPFLAGS_HEAPTYPE)) {
        return 1;
    }
    PyHeapTypeObject *ht = (PyHeapTypeObject *)sd->new_ty;
    return ht->ht_cached_keys != sd->keys;
}

int jit_split_dict_patcher_maybe_patch(
    PyTypeObject *type,
    PyTypeObject *new_ty,
    PyObject *attr_name,
    PyDictKeysObject *keys) {
    SplitDictCtx ctx = {keys, new_ty};
    return jit_should_patch_for_attr(
        type, new_ty, attr_name, split_dict_check_full, &ctx);
}
