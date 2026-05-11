// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>

/* ---- C API (implemented in c_helper_translations.c) ---- */
#ifdef __cplusplus
extern "C" {
#endif

void jit_lir_set_cast_addr(uint64_t addr);

/* Phase 5.B PIVOT: programmatic LirFunction* path for c-helper LIRs.
 * Returns the helper's LirFunction* (process-singleton, owned by
 * c_helper_translations) or NULL if addr does not match a registered
 * helper. The returned pointer must NOT be freed by the caller. */
struct LirFunction;
struct LirFunction* jit_lir_map_c_helper_to_lir_func(uint64_t addr);

#ifdef __cplusplus
} /* extern "C" */
#endif
