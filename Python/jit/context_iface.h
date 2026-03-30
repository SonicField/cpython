// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/python.h"

#include "cinderx/Common/ref.h"

namespace jit {

class alignas(16) CodeRuntime;

class IJitContext {
 public:
  IJitContext() {}
  virtual ~IJitContext() = default;

  virtual CodeRuntime* lookupCodeRuntime(
      BorrowedRef<PyFunctionObject> func) = 0;

  virtual BorrowedRef<> zero() = 0;

  // Return whether this context has compiled the given function.
  virtual bool didCompile(BorrowedRef<PyFunctionObject> /*func*/) { return false; }

  // Return whether a compiled function is currently deoptimized.
  virtual bool isDeoptimized(BorrowedRef<PyFunctionObject> /*func*/) { return false; }
};

} // namespace jit
