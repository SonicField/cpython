// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/Common/ref.h"

#include <string_view>

namespace jit {

class IJITList {
 public:
  IJITList() = default;
  virtual ~IJITList() = default;

  IJITList(const IJITList&) = delete;
  IJITList& operator=(const IJITList&) = delete;

  virtual bool parseLine(std::string_view line) = 0;
  virtual void parseFile(const char* filename) = 0;

  virtual int lookupFunc(PyFunctionObject* function) const = 0;
  virtual int lookupCode(PyCodeObject* code) const = 0;
  virtual int lookupName(PyObject* module_name, PyObject* qualname)
      const = 0;

  virtual Ref<> getList() const = 0;
};

} // namespace jit
