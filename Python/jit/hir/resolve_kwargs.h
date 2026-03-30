// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/Jit/hir/pass.h"

namespace jit::hir {

// Pre-inlining pass that resolves keyword argument calls into positional
// argument calls when the callee is a known function without **kwargs.
//
// This enables the inliner to inline calls that use keyword arguments at the
// call site but have a callee with regular positional parameters.
//
// Transforms:
//   VectorCall(func, arg_a, arg_b, kwnames=("y", "x"), KwArgs)
// Into:
//   VectorCall(func, reordered_args..., flags=0)
// where arguments are reordered to match the callee parameter positions.
class ResolveKwargs : public Pass {
 public:
  ResolveKwargs() : Pass("ResolveKwargs") {}

  void Run(Function& irfunc) override;

  static std::unique_ptr<ResolveKwargs> Factory() {
    return std::make_unique<ResolveKwargs>();
  }
};

} // namespace jit::hir
