// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cinderx/Jit/global_deopt_patcher.h"
#include "cinderx/Jit/context.h"

namespace jit {

GlobalDeoptPatcher::GlobalDeoptPatcher(
    BorrowedRef<PyDictObject> globals,
    BorrowedRef<PyUnicodeObject> key_name,
    BorrowedRef<> expected_value)
    : globals_{globals} {
  ThreadedCompileSerialize guard;
  key_name_.reset(key_name);
  expected_value_.reset(expected_value);
}

GlobalDeoptPatcher::~GlobalDeoptPatcher() {
  // If the patcher is still linked and has not yet been patched, it is still
  // registered in global_deopt_patchers_ and must be unregistered to prevent
  // dangling pointers (e.g., when a function is recompiled during warmup).
  if (isLinked() && key_name_.get() != nullptr) {
    Context* ctx = getContext();
    if (ctx != nullptr) {
      ctx->unwatchGlobal(globals_, key_name_, this);
    }
  }
}

bool GlobalDeoptPatcher::maybePatch(BorrowedRef<> new_value) {
  if (new_value == expected_value_.get()) {
    return false;
  }
  patch();
  return true;
}

void GlobalDeoptPatcher::onPatch() {
  key_name_.reset();
  expected_value_.reset();
}

} // namespace jit
