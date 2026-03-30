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
      BorrowedRef<> expected_value);

  // Called when a watched global changes.  If the new value differs from
  // the expected value, patch the compiled code to deopt.  Returns true if
  // the patcher fired (and should be removed from the watch list).
  ~GlobalDeoptPatcher();
  bool maybePatch(BorrowedRef<> new_value);

  BorrowedRef<PyDictObject> globals() const { return globals_; }
  BorrowedRef<PyUnicodeObject> keyName() const { return key_name_; }

 private:
  void onPatch() override;

  BorrowedRef<PyDictObject> globals_;
  ThreadedRef<PyUnicodeObject> key_name_;
  ThreadedRef<> expected_value_;
};

} // namespace jit
