// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cinderx/Jit/hir/type.h"
#include "cinderx/Jit/hir/hir_type_c.h"

#include "cinderx/StaticPython/static_array.h"
#include "cinderx/StaticPython/type_code.h"

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace jit::hir {

static_assert(sizeof(Type) == 16, "Type should fit in two registers");
static_assert(sizeof(intptr_t) == sizeof(int64_t), "Expected 64-bit pointers");

namespace {
// Phase 3D: static maps + toString helpers deleted (dead code).
} // namespace

std::string Type::specString() const {
  // Phase 3D: delegate to C (diagnostic formatting).
  HirType h = toHirType(*this);
  char buf[256];
  size_t n = hir_type_to_string_c(&h, buf, sizeof(buf), 0);
  return std::string(buf, n < sizeof(buf) ? n : sizeof(buf) - 1);
}

// Phase 3D: typeToName, makeSortedBits, joinParts deleted (dead code —
// toString now delegates to hir_type_to_string_c).

std::string Type::toString() const {
  // Phase 3D: delegate to C implementation.
  HirType h = toHirType(*this);
  char buf[256];
  size_t n = hir_type_to_string_c(&h, buf, sizeof(buf), 0);
  return std::string(buf, n < sizeof(buf) ? n : sizeof(buf) - 1);
}

// Phase 3D: delegate to C implementation.
std::string Type::toStringSafe() const {
  HirType h = toHirType(*this);
  char buf[256];
  size_t n = hir_type_to_string_c(&h, buf, sizeof(buf), 1);
  return std::string(buf, n < sizeof(buf) ? n : sizeof(buf) - 1);
}

// Phase 3D: fromTypeImpl body deleted — fromType/fromTypeExact delegate to C.
// Declaration kept in type.h for ABI compatibility but body is unused.
Type Type::fromTypeImpl(PyTypeObject* type, bool exact) {
  return fromHirType(hir_type_from_pytype(type, exact ? 1 : 0));
}

// Phase 3D: delegate to C implementation.
Type Type::fromType(PyTypeObject* type) {
  return fromHirType(hir_type_from_pytype(type, 0));
}

// Phase 3D: delegate to C implementation (assertion-verified zero mismatches).
Type Type::fromTypeExact(PyTypeObject* type) {
  return fromHirType(hir_type_from_pytype(type, 1));
}

// Phase 3D: delegate to C (assertion-verified zero mismatches).
Type Type::fromObject(PyObject* obj) {
  return fromHirType(hir_type_from_object(obj));
}

// Phase 3D: delegate to C (assertion-verified zero mismatches).
PyTypeObject* Type::uniquePyType() const {
  HirType h = toHirType(*this);
  return hir_type_unique_pytype(&h);
}

PyTypeObject* Type::runtimePyType() const {
  HirType h = toHirType(*this);
  return hir_type_runtime_pytype(&h);
}

std::optional<destructor> Type::runtimePyTypeDestructor() const {
  // If we do not have a runtime type that we can determine from this type, then
  // we cannot reliably determine the destructor.
  auto type = runtimePyType();
  if (type == nullptr) {
    return std::nullopt;
  }

  // If the type is the none type (which we can statically determine), then we
  // should not return the destructor. It's technically harmless to call it in
  // 3.11+, but in 3.10 it will crash.
  if (type == Py_TYPE(Py_None)) {
    return std::nullopt;
  }

  // Since we now have a destructor function that we can return, we can make it
  // into an optional and return it.
  return std::make_optional(type->tp_dealloc);
}

PyObject* Type::asObject() const {
  HirType h = toHirType(*this);
  return hir_type_as_object(&h);
}

bool Type::isSingleValue() const {
  HirType h = toHirType(*this);
  return hir_type_is_single_value(&h);
}

// Phase 3D: operators delegate to C API in hir_type_c.c.
bool Type::operator<=(Type other) const {
  return hir_type_is_subtype(toHirType(*this), toHirType(other));
}

bool Type::specSubtype(Type other) const {
  HirType a = toHirType(*this), b = toHirType(other);
  return hir_type_spec_subtype(&a, &b);
}

Type Type::operator|(Type other) const {
  return fromHirType(hir_type_union(toHirType(*this), toHirType(other)));
}

Type Type::operator&(Type other) const {
  return fromHirType(hir_type_intersect(toHirType(*this), toHirType(other)));
}

Type Type::operator-(Type rhs) const {
  return fromHirType(hir_type_subtract(toHirType(*this), toHirType(rhs)));
}

Type Type::asBoxed() const {
  HirType h = toHirType(*this);
  return fromHirType(hir_type_as_boxed(&h));
}

unsigned int Type::sizeInBytes() const {
  HirType h = toHirType(*this);
  return hir_type_size_in_bytes(&h);
}

Type OwnedType::toHir() const {
  int prim_type = _PyClassLoader_GetTypeCode(type);
  if (prim_type != TYPED_OBJECT) {
    JIT_CHECK(!optional, "primitive types cannot be optional");
    return prim_type_to_type(prim_type);
  }

  Type hir_type = exact ? Type::fromTypeExact(type) : Type::fromType(type);
  if (optional) {
    hir_type |= TNoneType;
  }
  return hir_type;
}

Type prim_type_to_type(int prim_type) {
  return Type::fromHirType(hir_prim_type_to_type(prim_type));
}

} // namespace jit::hir
