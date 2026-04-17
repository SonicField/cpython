// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include "cinderx/Common/log.h"
#include "cinderx/Common/ref.h"
#include "cinderx/Jit/code_patcher.h"
#include "cinderx/Jit/threaded_compile.h"

namespace jit {

// Patch a DeoptPatchpoint when the given PyTypeObject changes at all. This
// should only be used (instead of a more specific subclass) in cases where it
// is impossible to check the property we care about in maybePatch() (e.g., if
// the change to the type happens after PyType_Modified() is called).
class TypeDeoptPatcher : public JumpPatcher {
 public:
  explicit TypeDeoptPatcher(PyTypeObject* type) : type_{type} {}
  virtual ~TypeDeoptPatcher();

  virtual bool maybePatch(PyTypeObject*) {
    patch();
    return true;
  }

  PyTypeObject* type() const { return type_; }

 protected:
  void onUnpatch() override {
    JIT_ABORT(
        "TypeDeoptPatcher for type {} being unpatched but that's not supported!",
        type_->tp_name);
  }

  // The type being watched.  It outlives this object because this object will
  // be cleaned up by a type watcher notification.
  PyTypeObject* type_;
};

// Patch a DeoptPatchpoint when the given PyTypeObject no longer has the given
// PyObject* at the specified name.
class TypeAttrDeoptPatcher : public TypeDeoptPatcher {
 public:
  TypeAttrDeoptPatcher(
      PyTypeObject* type,
      PyUnicodeObject* attr_name,
      PyObject* target_object);

  bool maybePatch(PyTypeObject* new_ty) override;

 private:
  void onPatch() override;

  ThreadedRef<PyUnicodeObject> attr_name_;
  ThreadedRef<> target_object_;
};

class SplitDictDeoptPatcher : public TypeDeoptPatcher {
 public:
  SplitDictDeoptPatcher(
      PyTypeObject* type,
      PyUnicodeObject* attr_name,
      PyDictKeysObject* keys);

  bool maybePatch(PyTypeObject* new_ty) override;

 private:
  void onPatch() override;

  ThreadedRef<PyUnicodeObject> attr_name_;

  // We don't need to hold a strong reference to keys_ like we do for
  // attr_name_ because calls to PyTypeModified() happen before the old keys
  // object is decrefed.
  PyDictKeysObject* keys_;
};

} // namespace jit
