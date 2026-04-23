// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef incl_JIT_HIR_TYPE_INL_H
#error "hir_type_inl.h should only be included by hir_type.h"
#endif

#include "cinderx/Common/log.h"
#include "cinderx/StaticPython/type_code.h"

#include <cstring>
#include <optional>
#include <string>

static_assert(sizeof(jit::hir::Type) == 16, "Type should fit in two registers");
static_assert(sizeof(intptr_t) == sizeof(int64_t), "Expected 64-bit pointers");

namespace jit::hir {

inline std::size_t Type::hash() const {
  static_assert(sizeof(std::size_t) == sizeof(int_), "Unexpected size_t size");
  std::size_t i;
  std::memcpy(&i, this, sizeof(i));
  return combineHash(i, int_);
}

inline Type Type::fromCBool(bool b) {
  return Type{kCBool, kLifetimeBottom, kSpecInt, b};
}

inline Type Type::fromCDouble(double_t d) {
  return Type{kCDouble, d};
}

inline bool Type::CIntFitsType(int64_t i, Type t) {
  return t == TCInt64 || (t == TCInt32 && i >= INT32_MIN && i <= INT32_MAX) ||
      (t == TCInt16 && i >= INT64_MIN && i <= INT16_MAX) ||
      (i >= INT8_MIN && i <= INT8_MAX);
}

inline Type Type::fromCInt(int64_t i, Type t) {
  JIT_DCHECK(
      t == TCInt64 || t == TCInt32 || t == TCInt16 || t == TCInt8,
      "expected signed value");
  JIT_DCHECK(CIntFitsType(i, t), "int value out of range");
  return Type{t.bits_, kLifetimeBottom, kSpecInt, i};
}

inline Type Type::fromCPtr(void* p) {
  return Type{
      TCPtr.bits_, kLifetimeBottom, kSpecInt, reinterpret_cast<intptr_t>(p)};
}

inline bool Type::CUIntFitsType(uint64_t i, Type t) {
  return t == TCUInt64 || (t == TCUInt32 && i <= UINT32_MAX) ||
      (t == TCUInt16 && i <= UINT16_MAX) || i <= UINT8_MAX;
}

inline Type Type::fromCUInt(uint64_t i, Type t) {
  JIT_DCHECK(
      t == TCUInt64 || t == TCUInt32 || t == TCUInt16 || t == TCUInt8,
      "expected unsigned value");
  JIT_DCHECK(Type::CUIntFitsType(i, t), "int value out of range");
  return Type{t.bits_, kLifetimeBottom, kSpecInt, (intptr_t)i};
}

inline bool Type::hasTypeSpec() const {
  auto sk = specKind();
  return sk == kSpecType || sk == kSpecTypeExact || sk == kSpecObject;
}

inline bool Type::hasTypeExactSpec() const {
  return specKind() == kSpecTypeExact || specKind() == kSpecObject;
}

inline bool Type::hasObjectSpec() const {
  return specKind() == kSpecObject;
}

inline bool Type::hasIntSpec() const {
  return specKind() == kSpecInt;
}

inline bool Type::hasDoubleSpec() const {
  return specKind() == kSpecDouble;
}

inline bool Type::hasValueSpec(Type ty) const {
  return (hasObjectSpec() || hasIntSpec() || hasDoubleSpec()) && *this <= ty;
}

inline PyTypeObject* Type::typeSpec() const {
  JIT_DCHECK(hasTypeSpec(), "Type has no type specialization");
  return specKind() == kSpecObject ? Py_TYPE(pyobject_) : pytype_;
}

inline PyObject* Type::objectSpec() const {
  JIT_DCHECK(hasObjectSpec(), "Type has invalid value specialization");
  return pyobject_;
}

inline intptr_t Type::intSpec() const {
  JIT_DCHECK(hasIntSpec(), "Type has invalid value specialization");
  return int_;
}

inline double_t Type::doubleSpec() const {
  JIT_DCHECK(hasDoubleSpec(), "Type has invalid value specialization");
  return double_;
}

inline Type Type::unspecialized() const {
  return Type{bits_, lifetime_};
}

inline Type Type::dropMortality() const {
  if (lifetime_ == kLifetimeBottom) {
    return *this;
  }
  return Type{bits_, kLifetimeTop, specKind(), int_};
}

inline bool Type::hasSpec() const {
  return spec_kind_ != kSpecTop && spec_kind_ != kSpecBottom;
}

inline Type::SpecKind Type::specKind() const {
  return static_cast<SpecKind>(spec_kind_);
}

inline bool Type::isExact() const {
  return hasTypeExactSpec() || *this <= TBuiltinExact;
}

inline bool Type::couldBe(Type other) const {
  return (*this & other) != TBottom;
}

inline bool Type::operator==(Type other) const {
  return memcmp(this, &other, sizeof(*this)) == 0;
}

inline bool Type::operator!=(Type other) const {
  return !operator==(other);
}

inline bool Type::operator<(Type other) const {
  return *this != other && *this <= other;
}

inline Type& Type::operator|=(Type other) {
  return *this = *this | other;
}

inline Type& Type::operator&=(Type other) {
  return *this = *this & other;
}

inline Type& Type::operator-=(Type other) {
  return *this = *this - other;
}

// ---- Batch 2-E: bodies relocated from type.cpp (header-inline). ----
// Most are trivial delegations to hir_type_c.h C functions; the two
// non-trivial ones (runtimePyTypeDestructor + OwnedType::toHir) keep
// their original C++ shape.

inline std::string Type::specString() const {
  HirType h = toHirType(*this);
  char buf[256];
  size_t n = hir_type_to_string_c(&h, buf, sizeof(buf), 0);
  return std::string(buf, n < sizeof(buf) ? n : sizeof(buf) - 1);
}

inline std::string Type::toString() const {
  HirType h = toHirType(*this);
  char buf[256];
  size_t n = hir_type_to_string_c(&h, buf, sizeof(buf), 0);
  return std::string(buf, n < sizeof(buf) ? n : sizeof(buf) - 1);
}

inline std::string Type::toStringSafe() const {
  HirType h = toHirType(*this);
  char buf[256];
  size_t n = hir_type_to_string_c(&h, buf, sizeof(buf), 1);
  return std::string(buf, n < sizeof(buf) ? n : sizeof(buf) - 1);
}

inline Type Type::fromTypeImpl(PyTypeObject* type, bool exact) {
  return fromHirType(hir_type_from_pytype(type, exact ? 1 : 0));
}

inline Type Type::fromType(PyTypeObject* type) {
  return fromHirType(hir_type_from_pytype(type, 0));
}

inline Type Type::fromTypeExact(PyTypeObject* type) {
  return fromHirType(hir_type_from_pytype(type, 1));
}

inline Type Type::fromObject(PyObject* obj) {
  return fromHirType(hir_type_from_object(obj));
}

inline PyTypeObject* Type::uniquePyType() const {
  HirType h = toHirType(*this);
  return hir_type_unique_pytype(&h);
}

inline PyTypeObject* Type::runtimePyType() const {
  HirType h = toHirType(*this);
  return hir_type_runtime_pytype(&h);
}

inline std::optional<destructor> Type::runtimePyTypeDestructor() const {
  // If we cannot determine a runtime type, the destructor is unknown.
  auto type = runtimePyType();
  if (type == nullptr) {
    return std::nullopt;
  }
  // Calling Py_None's destructor is harmless on 3.11+ but crashes on 3.10.
  if (type == Py_TYPE(Py_None)) {
    return std::nullopt;
  }
  return std::make_optional(type->tp_dealloc);
}

inline PyObject* Type::asObject() const {
  HirType h = toHirType(*this);
  return hir_type_as_object(&h);
}

inline bool Type::isSingleValue() const {
  HirType h = toHirType(*this);
  return hir_type_is_single_value(&h);
}

inline bool Type::operator<=(Type other) const {
  return hir_type_is_subtype(toHirType(*this), toHirType(other));
}

inline bool Type::specSubtype(Type other) const {
  HirType a = toHirType(*this), b = toHirType(other);
  return hir_type_spec_subtype(&a, &b);
}

inline Type Type::operator|(Type other) const {
  return fromHirType(hir_type_union(toHirType(*this), toHirType(other)));
}

inline Type Type::operator&(Type other) const {
  return fromHirType(hir_type_intersect(toHirType(*this), toHirType(other)));
}

inline Type Type::operator-(Type rhs) const {
  return fromHirType(hir_type_subtract(toHirType(*this), toHirType(rhs)));
}

inline Type Type::asBoxed() const {
  HirType h = toHirType(*this);
  return fromHirType(hir_type_as_boxed(&h));
}

inline unsigned int Type::sizeInBytes() const {
  HirType h = toHirType(*this);
  return hir_type_size_in_bytes(&h);
}

inline Type OwnedType::toHir() const {
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

inline Type prim_type_to_type(int prim_type) {
  return Type::fromHirType(hir_prim_type_to_type(prim_type));
}

} // namespace jit::hir
