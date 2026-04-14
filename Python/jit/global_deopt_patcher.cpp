// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Phase 3D: Reduced — constructor, maybePatch, onPatch inlined to header.
// Remaining: destructor (needs Context for unwatchGlobal).

#include "cinderx/Jit/global_deopt_patcher.h"
#include "cinderx/Jit/context.h"

namespace jit {

GlobalDeoptPatcher::~GlobalDeoptPatcher() {
  if (isLinked() && key_name_.get() != nullptr) {
    Context* ctx = getContext();
    if (ctx != nullptr) {
      ctx->unwatchGlobal(globals_, key_name_, this);
    }
  }
}

} // namespace jit
