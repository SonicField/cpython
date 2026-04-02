// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <Python.h>

/* ---- C API (implemented in iterator_types.c) ---- */
#ifdef __cplusplus
extern "C" {
#endif

extern PyTypeObject *jit_g_range_iterator_type;
extern PyTypeObject *jit_g_list_iterator_type;
extern PyTypeObject *jit_g_tuple_iterator_type;

void jit_init_iterator_types(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

/* ---- C++ convenience (namespace jit) ---- */
#ifdef __cplusplus
namespace jit {

inline PyTypeObject*& g_range_iterator_type = jit_g_range_iterator_type;
inline PyTypeObject*& g_list_iterator_type  = jit_g_list_iterator_type;
inline PyTypeObject*& g_tuple_iterator_type = jit_g_tuple_iterator_type;

inline void init_iterator_types() {
    jit_init_iterator_types();
}

} // namespace jit
#endif
