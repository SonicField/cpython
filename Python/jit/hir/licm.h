// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/Jit/hir/pass.h"

namespace jit::hir {

// Loop-Invariant Code Motion (LICM) pass.
//
// Hoists loop-invariant guard instructions (GuardType, GuardIs) out of
// loop bodies into the loop preheader. This eliminates redundant per-
// iteration type checks for SSA values defined outside the loop.
//
// Safety: Only hoists guards on SSA values that are NOT phi nodes and
// whose definitions are outside the loop body. Keeps the original
// FrameState for deopt correctness.
class LICM : public Pass {
 public:
  LICM() : Pass("LICM") {}

  void Run(Function& irfunc) override;

  static std::unique_ptr<LICM> Factory() {
    return std::make_unique<LICM>();
  }
};

} // namespace jit::hir
