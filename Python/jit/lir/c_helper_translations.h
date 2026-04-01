// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>
#include <string>

/* ---- C API (implemented in c_helper_translations.c) ---- */
#ifdef __cplusplus
extern "C" {
#endif

const char* jit_lir_map_c_helper_to_lir(uint64_t addr);
void jit_lir_set_cast_addr(uint64_t addr);

#ifdef __cplusplus
} /* extern "C" */
#endif

#include "cinderx/Jit/jit_rt.h"

namespace jit::lir {

// Ensure JITRT_Cast address is registered on first use.
inline void ensureCHelperInit() {
  static bool initialized = false;
  if (!initialized) {
    jit_lir_set_cast_addr(reinterpret_cast<uint64_t>(JITRT_Cast));
    initialized = true;
  }
}

// Inline C++ wrapper — returns pointer to static std::string for API compat.
// The C function returns const char*; we wrap it in a thread-local string.
inline const std::string* mapCHelperToLIR(uint64_t addr) {
  ensureCHelperInit();
  const char* result = jit_lir_map_c_helper_to_lir(addr);
  if (result == nullptr) {
    return nullptr;
  }
  static thread_local std::string cached;
  cached = result;
  return &cached;
}

} // namespace jit::lir
