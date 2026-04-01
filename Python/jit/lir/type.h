// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <fmt/ostream.h>

#include <cstdint>
#include <iosfwd>

/* ---- C API (implemented in type.c) ---- */
#ifdef __cplusplus
extern "C" {
#endif

size_t jit_lir_bit_size(int dt);
const char* jit_lir_data_type_name(int dt);
const char* jit_lir_operand_type_name(int ty);

#ifdef __cplusplus
} /* extern "C" */
#endif

namespace jit::lir {

/*
 * Operand types:
 *   - None:   the operand is not used.
 *   - Vreg:   the operand is in a virtual register (not yet
 *             allocated to a physical location);
 *   - Reg:    the operand is allocated to a physical register;
 *   - Stack:  the operand is allocated to a memory stack slot;
 *   - Mem:    the operand is allocated to a memory address;
 *   - Ind:    the operand is a memory indirect reference
 *   - Imm:    the operand is an immediate value;
 *   - Lbl:    the operand refers to a basic block.
 */
#define FOREACH_OPERAND_TYPE(X) \
  X(None)                       \
  X(Vreg)                       \
  X(Reg)                        \
  X(Stack)                      \
  X(Mem)                        \
  X(Ind)                        \
  X(Imm)                        \
  X(Label)

enum class OperandType : uint8_t {
#define OPERAND_DECL_TYPE(v, ...) k##v,
  FOREACH_OPERAND_TYPE(OPERAND_DECL_TYPE)
#undef OPERAND_DECL_TYPE
};

/*
 * Operand data types.  Includes sized integers, 64-bit doubles, and PyObject*
 * values.
 */
#define FOREACH_OPERAND_DATA_TYPE(X) \
  X(8bit)                            \
  X(16bit)                           \
  X(32bit)                           \
  X(64bit)                           \
  X(Double)                          \
  X(Object)

enum class DataType : uint8_t {
#define DECL_DATA_TYPE_ENUM(s, ...) k##s,
  FOREACH_OPERAND_DATA_TYPE(DECL_DATA_TYPE_ENUM)
#undef DECL_DATA_TYPE_ENUM
};

/* Inline C++ wrappers that forward to the C functions */
inline size_t bitSize(DataType dt) {
  return jit_lir_bit_size(static_cast<int>(dt));
}

inline std::ostream& operator<<(std::ostream& os, OperandType ty) {
  return os << jit_lir_operand_type_name(static_cast<int>(ty));
}

inline std::ostream& operator<<(std::ostream& os, DataType dt) {
  return os << jit_lir_data_type_name(static_cast<int>(dt));
}

} // namespace jit::lir

template <>
struct fmt::formatter<jit::lir::OperandType> : fmt::ostream_formatter {};

template <>
struct fmt::formatter<jit::lir::DataType> : fmt::ostream_formatter {};
