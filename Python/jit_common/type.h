// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/python.h"

#include "cinderx/Common/ref.h"
#include "cinderx/Common/type_c.h"

#include <string>

namespace jit {

// When possible, return the fully qualified name of the given type (including
// its module). Falls back to the type's bare name.
inline std::string typeFullname(PyTypeObject* type) {
  char buf[512];
  jit_type_fullname(type, buf, sizeof(buf));
  return std::string(buf);
}

// Simulate _PyType_Lookup(), but in a way that should avoid any heap mutations
// (caches, refcount operations, arbitrary code execution).
inline BorrowedRef<> typeLookupSafe(
    BorrowedRef<PyTypeObject> type,
    BorrowedRef<> name) {
  return jit_type_lookup_safe(type, name);
}

// Attempt to ensure that the given type has a valid version tag, returning
// true if successful.
inline bool ensureVersionTag(BorrowedRef<PyTypeObject> type) {
  return jit_ensure_version_tag(type) != 0;
}

} // namespace jit
