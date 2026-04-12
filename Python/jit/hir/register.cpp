// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Minimal C++ stubs — name generation uses snprintf instead of fmt::format.

#include "cinderx/Jit/hir/register.h"

#include "cinderx/Common/log.h"

#include <cstdio>
#include <ostream>

namespace jit::hir {

const std::string& Register::name() const {
  if (name_.empty()) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "v%d", id_);
    name_ = buf;
  }
  return name_;
}

std::ostream& operator<<(std::ostream& os, const Register& reg) {
  return os << reg.name();
}

std::ostream& operator<<(std::ostream& os, RefKind kind) {
  switch (kind) {
    case RefKind::kUncounted:
      return os << "Uncounted";
    case RefKind::kBorrowed:
      return os << "Borrowed";
    case RefKind::kOwned:
      return os << "Owned";
  }
  JIT_ABORT("Bad RefKind {}", static_cast<int>(kind));
}

std::ostream& operator<<(std::ostream& os, ValueKind kind) {
  switch (kind) {
    case ValueKind::kObject:
      return os << "Object";
    case ValueKind::kSigned:
      return os << "Signed";
    case ValueKind::kUnsigned:
      return os << "Unsigned";
    case ValueKind::kBool:
      return os << "Bool";
    case ValueKind::kDouble:
      return os << "Double";
  }
  JIT_ABORT("Bad ValueKind {}", static_cast<int>(kind));
}

} // namespace jit::hir
