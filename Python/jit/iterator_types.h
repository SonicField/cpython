// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <Python.h>

namespace jit {

// Cached type pointers — nullptr until init_iterator_types() is called.
extern PyTypeObject* g_range_iterator_type;
extern PyTypeObject* g_list_iterator_type;
extern PyTypeObject* g_tuple_iterator_type;

// Initialise iterator type pointers at CinderX startup.
// Must be called after Python is fully initialised.
void init_iterator_types();

} // namespace jit
