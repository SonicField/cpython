// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Phase 3D: Reduced — TypeDeoptPatcher constructor, maybePatch, type(),
// onUnpatch inlined to header. Remaining: destructor (needs Context),
// TypeAttrDeoptPatcher, SplitDictDeoptPatcher (templates, lambdas, Context).

#include "cinderx/Jit/type_deopt_patchers.h"
#include "cinderx/Jit/context.h"

#include "cinderx/Common/type.h"
#include "cinderx/Common/util.h"

namespace jit {

template <typename Body>
bool shouldPatchForAttr(
    BorrowedRef<PyTypeObject> old_ty,
    BorrowedRef<PyTypeObject> new_ty,
    BorrowedRef<PyUnicodeObject> attr_name,
    Body body) {
  if (new_ty != old_ty) {
    return true;
  }
  BorrowedRef<> attr{typeLookupSafe(new_ty, attr_name)};
  return body(attr) || !PyUnstable_Type_AssignVersionTag(new_ty);
}

TypeDeoptPatcher::~TypeDeoptPatcher() {
  if (isLinked() && type_ != nullptr) {
    Context* ctx = getContext();
    if (ctx != nullptr) {
      ctx->unwatchType(type_, this);
    }
  }
}

TypeAttrDeoptPatcher::TypeAttrDeoptPatcher(
    BorrowedRef<PyTypeObject> type,
    BorrowedRef<PyUnicodeObject> attr_name,
    BorrowedRef<> target_object)
    : TypeDeoptPatcher{type} {
  ThreadedCompileSerialize guard;
  attr_name_.reset(attr_name);
  target_object_.reset(target_object);
}

bool TypeAttrDeoptPatcher::maybePatch(BorrowedRef<PyTypeObject> new_ty) {
  bool should_patch =
      shouldPatchForAttr(type_, new_ty, attr_name_, [&](BorrowedRef<> attr) {
        return attr != target_object_;
      });
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
    BorrowedRef<PyTypeObject> type,
    BorrowedRef<PyUnicodeObject> attr_name,
    PyDictKeysObject* keys)
    : TypeDeoptPatcher{type}, keys_{keys} {
  ThreadedCompileSerialize guard;
  attr_name_.reset(attr_name);
}

bool SplitDictDeoptPatcher::maybePatch(BorrowedRef<PyTypeObject> new_ty) {
  bool should_patch =
      shouldPatchForAttr(type_, new_ty, attr_name_, [&](BorrowedRef<> attr) {
        if (attr != nullptr) {
          return true;
        }
        if (!PyType_HasFeature(new_ty, Py_TPFLAGS_HEAPTYPE)) {
          return true;
        }
        BorrowedRef<PyHeapTypeObject> ht(new_ty);
        return ht->ht_cached_keys != keys_;
      });
  if (should_patch) {
    patch();
  }
  return should_patch;
}

void SplitDictDeoptPatcher::onPatch() {
  attr_name_.reset();
}

} // namespace jit
