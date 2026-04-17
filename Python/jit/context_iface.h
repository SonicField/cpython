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
      PyFunctionObject* func) = 0;

  virtual PyObject* zero() = 0;

  // Return whether this context has compiled the given function.
  virtual bool didCompile(PyFunctionObject* /*func*/) { return false; }

  // Return whether a compiled function is currently deoptimized.
  virtual bool isDeoptimized(PyFunctionObject* /*func*/) { return false; }
};

} // namespace jit
