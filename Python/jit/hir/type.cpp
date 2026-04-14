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

// For Types where it makes sense, map them to their corresponding
// PyTypeObject*.
const std::unordered_map<Type, PyTypeObject*>& typeToPyType() {
  static auto const map = [] {
    const std::unordered_map<Type, PyTypeObject*> result_map{
        {TObject, &PyBaseObject_Type},
        {TBool, &PyBool_Type},
        {TBytes, &PyBytes_Type},
        {TCell, &PyCell_Type},
        {TCode, &PyCode_Type},
        {TDict, &PyDict_Type},
        {TBaseException, reinterpret_cast<PyTypeObject*>(PyExc_BaseException)},
        {TFloat, &PyFloat_Type},
        {TFrame, &PyFrame_Type},
        {TFunc, &PyFunction_Type},
        {TGen, &PyGen_Type},
        {TList, &PyList_Type},
        {TLong, &PyLong_Type},
        {TSet, &PySet_Type},
        {TSlice, &PySlice_Type},
        {TTuple, &PyTuple_Type},
        {TType, &PyType_Type},
        {TUnicode, &PyUnicode_Type},
#if PY_VERSION_HEX < 0x030C0000
        {TWaitHandle, &Ci_PyWaitHandle_Type},
#endif
        {TNoneType, Py_TYPE(Py_None)},
    };

    // After construction, verify that all appropriate types have an entry in
    // this table. Except for TWaitHandle, which hasn't been ported to 3.12 yet
    // and TArray which is a heap type so can't be included in this static
    // table.
#define CHECK_TY(name, bits, lifetime, flags)        \
  JIT_CHECK(                                         \
      T##name <= TArray || T##name <= TWaitHandle || \
          ((flags) & kTypeHasUniquePyType) == 0 ||   \
          result_map.contains(T##name),              \
      "Type {} missing entry in typeToPyType()",     \
      T##name);
    HIR_TYPES(CHECK_TY)
#undef CHECK_TY

    return result_map;
  }();

  return map;
}

// Like typeToPyType(), but including Exact types in the key set (e.g., mapping
// TListExact -> PyList_Type).
const std::unordered_map<Type, PyTypeObject*>& typeToPyTypeWithExact() {
  static auto const map = [] {
    auto result_map = typeToPyType();
    for (auto& pair : typeToPyType()) {
      if (pair.first == TObject) {
        result_map.emplace(TObjectExact, &PyBaseObject_Type);
      } else if (pair.first == TLong) {
        result_map.emplace(TLongExact, &PyLong_Type);
      } else {
        result_map.emplace(pair.first & TBuiltinExact, pair.second);
      }
    }
    return result_map;
  }();

  return map;
}

// The inverse of typeToPyType().
const std::unordered_map<PyTypeObject*, Type>& pyTypeToType() {
  static auto const map = [] {
    std::unordered_map<PyTypeObject*, Type> result_map;
    for (auto& pair : typeToPyType()) {
      bool inserted = result_map.emplace(pair.second, pair.first).second;
      JIT_CHECK(inserted, "Duplicate key type: {}", pair.second->tp_name);
    }
    return result_map;
  }();

  return map;
}

// Like pyTypeToType(), but for Type::fromTypeExact(). It wants only the
// components of a type that can represent an exact type: the builtin exact
// type, or user-defined subtypes for exact specialization. These can be
// selected for most types by intersecting with TBuiltinExact or TUser,
// respectively.
//
// The only exceptions that we have to adjust for in this map are predefined
// Types that have other predefined Types as subtypes: TObject (where we leave
// out all other types) and TLong (where we leave out TBool).
const std::unordered_map<PyTypeObject*, Type>& pyTypeToTypeForExact() {
  static auto const map = [] {
    auto result_map = pyTypeToType();
    result_map.at(&PyBaseObject_Type) = TObjectExact | TObjectUser;
    result_map.at(&PyLong_Type) = TLongExact | TLongUser;
    return result_map;
  }();

  return map;
}

static std::string
truncatedStr(const char* data, std::size_t size, char delim) {
  const Py_ssize_t kMaxStrChars = 20;
  if (size <= kMaxStrChars) {
    return fmt::format("{}{}{}", delim, fmt::string_view{data, size}, delim);
  }
  return fmt::format(
      "{}{}{}...", delim, fmt::string_view{data, kMaxStrChars}, delim);
}

} // namespace

std::string Type::specString() const {
  if (hasIntSpec()) {
    if (*this <= TCBool) {
      return int_ ? "true" : "false";
    }
    if (*this <= TCPtr) {
      return fmt::format("{}", getStablePointer(ptr_));
    }
    JIT_DCHECK(
        *this <= TCInt8 || *this <= TCInt16 || *this <= TCInt32 ||
            *this <= TCInt64 || *this <= TCUInt8 || *this <= TCUInt16 ||
            *this <= TCUInt32 || *this <= TCUInt64,
        "Invalid specialization");
    return fmt::format("{}", int_);
  }

  if (hasDoubleSpec()) {
    return fmt::format("{}", double_);
  }

  if (!hasObjectSpec()) {
    if (hasTypeExactSpec()) {
      return fmt::format("{}:Exact", typeSpec()->tp_name);
    }
    return typeSpec()->tp_name;
  }

  if (*this <= TUnicode) {
    Py_ssize_t size;
    auto utf8 = PyUnicode_AsUTF8AndSize(objectSpec(), &size);
    if (utf8 == nullptr) {
      PyErr_Clear();
      return "encoding error";
    }
    return truncatedStr(utf8, size, '"');
  }

  if (typeSpec() == &PyCFunction_Type) {
    PyCFunctionObject* func =
        reinterpret_cast<PyCFunctionObject*>(objectSpec());
    const char* func_name = func->m_ml->ml_name;
    return fmt::format(
        "{}:{}:{}",
        typeSpec()->tp_name,
        func_name,
        getStablePointer(objectSpec()));
  }

  if (*this <= TType) {
    return fmt::format(
        "{}:obj", reinterpret_cast<PyTypeObject*>(objectSpec())->tp_name);
  }

  if (*this <= TBytes) {
    char* buffer;
    Py_ssize_t size;
    if (PyBytes_AsStringAndSize(objectSpec(), &buffer, &size) < 0) {
      PyErr_Clear();
      return "unknown error";
    }
    return truncatedStr(buffer, size, '\'');
  }

  if (*this <= TBool) {
    return objectSpec() == Py_True ? "True" : "False";
  }

  if (*this <= TLong) {
    int overflow = 0;
    auto value = PyLong_AsLongLongAndOverflow(objectSpec(), &overflow);
    if (value == -1) {
      if (overflow == -1) {
        return "underflow";
      }
      if (overflow == 1) {
        return "overflow";
      }
      if (PyErr_Occurred()) {
        PyErr_Clear();
        return "error";
      }
    }
    return fmt::format("{}", value);
  }

  if (*this <= TFloat) {
    auto value = PyFloat_AsDouble(objectSpec());
    if (value == -1.0 && PyErr_Occurred()) {
      return "error";
    }
    return fmt::format("{}", value);
  }

  if (*this <= TCode) {
    auto name = reinterpret_cast<PyCodeObject*>(objectSpec())->co_name;
    if (name != nullptr && PyUnicode_Check(name)) {
      return fmt::format("\"{}\"", PyUnicode_AsUTF8(name));
    }
  }

  // We want to avoid invoking arbitrary Python during compilation, so don't
  // call PyObject_Repr() or anything similar.
  return fmt::format(
      "{}:{}", typeSpec()->tp_name, getStablePointer(objectSpec()));
}

static auto typeToName() {
  std::unordered_map<Type, std::string> map{
#define TY(name, ...) {T##name, #name},
      HIR_TYPES(TY)
#undef TY
  };
  return map;
}

// Return a list of pairs of predefined type bit patterns and their name, used
// to create string representations of nontrivial union types.
static auto makeSortedBits() {
  std::vector<std::pair<Type::bits_t, std::string>> vec;

  // Exclude predefined types with nontrivial mortality, since their 'bits'
  // component is the same as the version with kLifetime{Top,Bottom}.
  //
  // Also exclude any strict supertype of Nullptr, to give strings like
  // {List|Dict|Nullptr} rather than {OptList|Dict}.
  auto include_bits = [](Type::bits_t bits, size_t flags, const char* name) {
    if ((flags & kTypeHasTrivialMortality) == 0 ||
        (((Type::kNullptr & bits) == Type::kNullptr) &&
         bits != Type::kNullptr)) {
      return false;
    }

    JIT_CHECK(
        (bits & Type::kObject) == bits || (bits & Type::kPrimitive) == bits,
        "Bits for {} should be subset of kObject or kPrimitive",
        name);
    return true;
  };
#define TY(name, bits, lifetime, flags)            \
  if (include_bits(Type::k##name, flags, #name)) { \
    vec.emplace_back(Type::k##name, #name);        \
  }
  HIR_TYPES(TY)
#undef TY

  // Sort the vector so types with the most bits set show up first.
  auto pred = [](auto& a, auto& b) {
    return popcount(a.first) > popcount(b.first);
  };
  std::sort(vec.begin(), vec.end(), pred);
  JIT_CHECK(
      vec.back().first == Type::kBottom, "Bottom should be at end of vec");
  vec.pop_back();
  return vec;
}

static std::string joinParts(std::vector<std::string>& parts) {
  if (parts.size() == 1) {
    return parts.front();
  }

  // Always show the parts in alphabetical order, regardless of which has the
  // most bits.
  std::sort(parts.begin(), parts.end());
  return fmt::format("{{{}}}", fmt::join(parts, "|"));
}

std::string Type::toString() const {
  std::string base;

  static auto const type_names = typeToName();
  auto it = type_names.find(unspecialized());
  if (it != type_names.end()) {
    base = it->second;
  } else {
    // Search the list of predefined type names, starting with the ones
    // containing the most bits.
    static auto const sorted_bits = makeSortedBits();
    bits_t bits_left = bits_;
    std::vector<std::string> parts, obj_parts;
    for (auto& pair : sorted_bits) {
      auto bits = pair.first;
      if ((bits_left & bits) == bits) {
        if (bits & kObject) {
          obj_parts.emplace_back(pair.second);
        } else {
          parts.emplace_back(pair.second);
        }
        bits_left ^= bits;
        if (bits_left == 0) {
          break;
        }
      }
    }
    JIT_CHECK(bits_left == 0, "Type contains invalid bits");

    // If we have a nontrivial lifetime component, turn obj_parts into one part
    // with that prepended, then combine that with parts.
    if (lifetime_ != kLifetimeTop && lifetime_ != kLifetimeBottom) {
      const char* mortal = lifetime_ == kLifetimeMortal ? "Mortal" : "Immortal";
      parts.emplace_back(fmt::format("{}{}", mortal, joinParts(obj_parts)));
    } else {
      parts.insert(parts.end(), obj_parts.begin(), obj_parts.end());
    }
    base = joinParts(parts);
  }

  return hasSpec() ? fmt::format("{}[{}]", base, specString()) : base;
}

std::string Type::toStringSafe() const {
  switch (spec_kind_) {
    case kSpecTop:
      return "Top";
    case kSpecType:
      return std::string("Type(") + (pytype_ ? pytype_->tp_name : "nullptr") +
          ")";
    case kSpecTypeExact:
      return std::string("TypeExact(") +
          (pytype_ ? pytype_->tp_name : "nullptr") + ")";
    case kSpecObject:
      return std::string("Object(") +
          (pyobject_ ? (pyobject_->ob_type ? pyobject_->ob_type->tp_name
                                           : "unknown_type")
                     : "nullptr") +
          ")";
    case kSpecInt:
      return "Int";
    case kSpecDouble:
      return "Double";
    case kSpecBottom:
      return "Bottom";
    default:
      return "Unknown";
  }
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
