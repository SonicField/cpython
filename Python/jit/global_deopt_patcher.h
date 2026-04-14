// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include "cinderx/Common/ref.h"
#include "cinderx/Jit/code_patcher.h"
#include "cinderx/Jit/threaded_compile.h"

namespace jit {

// Patch a DeoptPatchpoint when a module global (dict[key]) changes from the
// expected value.  Used by simplifyVectorCallGlobal to eliminate function-
// identity GuardIs checks on inlined callees loaded via LoadGlobalCached.
//
// The patcher is triggered by GlobalCacheManager::notifyDictUpdate, which
// fires synchronously within PyDict_SetItem (before the mutation completes).
class GlobalDeoptPatcher : public JumpPatcher {
 public:
  GlobalDeoptPatcher(
      BorrowedRef<PyDictObject> globals,
      BorrowedRef<PyUnicodeObject> key_name,
      BorrowedRef<> expected_value)
      : globals_{globals} {
    ThreadedCompileSerialize guard;
    key_name_.reset(key_name);
    expected_value_.reset(expected_value);
  }

  ~GlobalDeoptPatcher();

  bool maybePatch(BorrowedRef<> new_value) {
    if (new_value == expected_value_.get()) {
      return false;
    }
    patch();
    return true;
  }

  BorrowedRef<PyDictObject> globals() const { return globals_; }
  BorrowedRef<PyUnicodeObject> keyName() const { return key_name_; }

 private:
  void onPatch() override {
    key_name_.reset();
    expected_value_.reset();
  }

  BorrowedRef<PyDictObject> globals_;
  ThreadedRef<PyUnicodeObject> key_name_;
  ThreadedRef<> expected_value_;
};

} // namespace jit
