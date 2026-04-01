// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>
#include <string>

/* ---- C API (implemented in symbol_mapping.c) ---- */
#ifdef __cplusplus
extern "C" {
#endif

const uint64_t* jit_lir_py_function_from_name(const char* name);

#ifdef __cplusplus
} /* extern "C" */
#endif

namespace jit::lir {

// Inline C++ wrapper that forwards to the C function.
inline const uint64_t* pyFunctionFromName(std::string_view name) {
  // string_view may not be null-terminated; use std::string for safety.
  std::string name_str(name);
  return jit_lir_py_function_from_name(name_str.c_str());
}

} // namespace jit::lir
