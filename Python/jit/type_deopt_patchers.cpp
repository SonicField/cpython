// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Phase 3D: All method bodies delegate to C functions in
// type_deopt_patchers_c.c. Class hierarchy stays in the header.

#include "cinderx/Jit/type_deopt_patchers.h"
#include "cinderx/Jit/type_deopt_patchers_c.h"
#include "cinderx/Jit/threaded_compile.h"

namespace jit {

TypeDeoptPatcher::~TypeDeoptPatcher() {
  jit_type_deopt_patcher_destroy(type_, isLinked(), this);
}

TypeAttrDeoptPatcher::TypeAttrDeoptPatcher(
    PyTypeObject* type,
    PyUnicodeObject* attr_name,
    PyObject* target_object)
    : TypeDeoptPatcher{type} {
  ThreadedCompileSerialize guard;
  attr_name_.reset(attr_name);
  target_object_.reset(target_object);
}

bool TypeAttrDeoptPatcher::maybePatch(PyTypeObject* new_ty) {
  bool should_patch = jit_type_attr_patcher_maybe_patch(
      type_, new_ty, (PyObject*)attr_name_.get(), target_object_.get());
  if (should_patch) {
    patch();
  }
  return should_patch;
}

void TypeAttrDeoptPatcher::onPatch() {
  attr_name_.reset();
  target_object_.reset();
}

SplitDictDeoptPatcher::SplitDictDeoptPatcher(
    PyTypeObject* type,
    PyUnicodeObject* attr_name,
    PyDictKeysObject* keys)
    : TypeDeoptPatcher{type}, keys_{keys} {
  ThreadedCompileSerialize guard;
  attr_name_.reset(attr_name);
}

bool SplitDictDeoptPatcher::maybePatch(PyTypeObject* new_ty) {
  bool should_patch = jit_split_dict_patcher_maybe_patch(
      type_, new_ty, (PyObject*)attr_name_.get(), keys_);
  if (should_patch) {
    patch();
  }
  return should_patch;
}

void SplitDictDeoptPatcher::onPatch() {
  attr_name_.reset();
}

} // namespace jit
