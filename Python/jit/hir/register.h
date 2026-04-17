// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/Jit/hir/type.h"

#include <cstdio>
#include <cstdlib>
#include <ostream>

struct HirRegisterLayoutVerifier;

namespace jit::hir {

class Instr;

// HIR operates on an infinite number of virtual registers, which are
// represented by the Register class. After SSAify has run on a Function, its
// Registers represent SSA values, and their Types should be kept up-to-date and
// trusted.
class Register {
 public:
  explicit Register(int i) : id_(i) {}

  // An integer identifier for this register. This is unique per `Function`.
  int id() const {
    return id_;
  }

  // The type of this value. Only meaningful for SSA-form HIR.
  Type type() const {
    return type_;
  }
  void set_type(Type type) {
    type_ = type;
  }

  // Shorthand for checking the type of this Register.
  bool isA(Type type) const {
    return type_ <= type;
  }

  // The instruction that defined this value. Always set, but only meaningful
  // for SSA-form HIR.
  Instr* instr() const {
    return instr_;
  }
  void set_instr(Instr* instr) {
    instr_ = instr;
  }

  const char* name() const {
    if (!name_) {
      name_ = static_cast<char*>(malloc(32));
      std::snprintf(name_, 32, "v%d", id_);
    }
    return name_;
  }

  ~Register() { free(name_); }

 private:
  DISALLOW_COPY_AND_ASSIGN(Register);
  friend struct ::HirRegisterLayoutVerifier;

  Type type_{TTop};
  Instr* instr_{nullptr};
  int id_{-1};
  mutable char* name_{nullptr};
};

// The refcount semantics of a value held in a Register.
enum class RefKind : char {
  // A PyObject* that is either null or points to an immortal object, and
  // doesn't need to be reference counted, or a primitive.
  kUncounted,
  // A PyObject* with a borrowed reference.
  kBorrowed,
  // A PyObject* that owns a reference.
  kOwned,
};

// The kind of value held in a Register.
enum class ValueKind : char {
  // A PyObject*.
  kObject,
  // A signed 64-bit integer.
  kSigned,
  // An unsigned 64-bit integer.
  kUnsigned,
  // A C bool.
  kBool,
  // A C Double
  kDouble,
};

struct RegState {
  RegState() = default;
  RegState(Register* reg, RefKind ref_kind, ValueKind value_kind)
      : reg{reg}, ref_kind{ref_kind}, value_kind{value_kind} {}

  bool operator==(const RegState& other) const {
    return (reg == other.reg) && (ref_kind == other.ref_kind) &&
        (value_kind == other.value_kind);
  }

  Register* reg{nullptr};
  RefKind ref_kind{RefKind::kUncounted};
  ValueKind value_kind{ValueKind::kObject};
};

inline std::ostream& operator<<(std::ostream& os, const Register& reg) {
  return os << reg.name();
}

inline std::ostream& operator<<(std::ostream& os, RefKind kind) {
  switch (kind) {
    case RefKind::kUncounted: return os << "Uncounted";
    case RefKind::kBorrowed: return os << "Borrowed";
    case RefKind::kOwned: return os << "Owned";
  }
  return os;
}

inline std::ostream& operator<<(std::ostream& os, ValueKind kind) {
  switch (kind) {
    case ValueKind::kObject: return os << "Object";
    case ValueKind::kSigned: return os << "Signed";
    case ValueKind::kUnsigned: return os << "Unsigned";
    case ValueKind::kBool: return os << "Bool";
    case ValueKind::kDouble: return os << "Double";
  }
  return os;
}

inline auto format_as(const jit::hir::RefKind& kind) {
  return fmt::underlying(kind);
}

inline auto format_as(const jit::hir::ValueKind& kind) {
  return fmt::underlying(kind);
}

} // namespace jit::hir

template <>
struct fmt::formatter<jit::hir::Register> : fmt::ostream_formatter {};
