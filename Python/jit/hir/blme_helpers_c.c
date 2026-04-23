/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Pure-C port of methods 1+2 from
 * builtin_load_method_elimination.cpp (W42-class Cat-A extraction per
 * theologian 20:03:03Z + supervisor 20:04:42Z + W27e PARTIAL precedent).
 *
 * Methods 3+4 (tryEliminateLoadMethod + Run) remain in the .cpp as
 * Cat-B accepted-residual due to heavier C++ bridge surface
 * (LoadMethodBase / GetSecondOutput / CallMethod / VectorCall accessors
 * + ExpandInto / ReplaceWith / Destroy + env::AllocateRegister /
 * addReference; ~17 new bridges would be needed, exceeds W25b
 * minimal-bridge budget).
 *
 * Bridge-to-C++: phx_builtin_members_lookup (declared in
 * Python/cinderx/module_state.h) provides borrowed-PyObject* lookup
 * into ModuleState::builtinMembers without exposing the C++
 * unordered_map / Ref<> types to C.
 */

#include "cinderx/Jit/hir/blme_helpers_c.h"
#include "cinderx/Jit/hir/hir_type_c.h"

/* _PyType_GetDict is internal API (Include/internal/pycore_typeobject.h)
 * which can't be safely included in .c files lacking Py_BUILD_CORE
 * without dragging in the rest of pycore. Forward-declare the prototype
 * locally — semantics are stable: returns a borrowed reference to the
 * type's tp_dict (does not need GIL release/reacquire). */
extern PyObject* _PyType_GetDict(PyTypeObject* type);

/* Forward declaration of the C bridge (defined in module_state.cpp). */
extern PyObject* phx_builtin_members_lookup(PyTypeObject* type, PyObject* name);

PyObject* phx_immutable_multithreaded_type_lookup(PyTypeObject* type, PyObject* name) {
    PyObject* mro = type->tp_mro;
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(mro); i++) {
        PyTypeObject* mro_type = (PyTypeObject*)PyTuple_GET_ITEM(mro, i);
        if (PyType_HasFeature(mro_type, _Py_TPFLAGS_STATIC_BUILTIN)) {
            /* The builtin_members cache is authoritative for static
             * built-in types — return the looked-up member (or NULL
             * if absent / not cached). */
            return phx_builtin_members_lookup(mro_type, name);
        } else if (
            !PyType_HasFeature(mro_type, Py_TPFLAGS_IMMUTABLETYPE) ||
            !PyType_CheckExact(mro_type)) {
            /* Untrusted base type — bail. */
            return NULL;
        }

        PyObject* method_obj = PyDict_GetItemWithError(_PyType_GetDict(mro_type), name);
        if (method_obj != NULL) {
            return method_obj;
        }
    }
    return NULL;
}

PyObject* phx_get_method_object_from_type(HirType receiver_type, PyObject* name) {
    /* 3.12+ path: require an exact-type spec. */
    if (!hir_type_has_type_exact_spec(&receiver_type)) {
        return NULL;
    }
    PyTypeObject* type = hir_type_runtime_py_type(&receiver_type);
    if (type == NULL) {
        /* Caller (tryEliminateLoadMethod) DCHECKs receiver_type ==
         * TBottom in this case; here we just return NULL — the C++
         * caller's DCHECK still runs against the original Type. */
        return NULL;
    }

    PyObject* method_obj = NULL;
    /* In 3.12 we can't do PyType_Lookup for built-in types in
     * multi-threaded compile (no current runtime). Use the
     * builtin_members cache via the C bridge. */
    if (PyType_HasFeature(type, _Py_TPFLAGS_STATIC_BUILTIN)) {
        method_obj = phx_builtin_members_lookup(type, name);
    } else if (
        PyType_HasFeature(type, Py_TPFLAGS_IMMUTABLETYPE) &&
        PyType_CheckExact(type) && type->tp_dictoffset == 0) {
        method_obj = phx_immutable_multithreaded_type_lookup(type, name);
        /* Restrict to the descriptor / function classes the caller
         * (tryEliminateLoadMethod) understands. */
        if (method_obj != NULL &&
            Py_TYPE(method_obj) != &PyClassMethodDescr_Type &&
            Py_TYPE(method_obj) != &PyMethodDescr_Type &&
            Py_TYPE(method_obj) != &PyWrapperDescr_Type &&
            Py_TYPE(method_obj) != &PyFunction_Type) {
            method_obj = NULL;
        }
    }
    return method_obj;
}
