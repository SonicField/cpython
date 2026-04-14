/*
 * hir_stats_c.h — Pure C interface for HIR statistics collection.
 *
 * Phase 3D: Replaces hir_stats.h (C++ HIRStats pass class).
 * The stats pass is diagnostic-only (gated by dump_hir_stats config).
 */
#pragma once

#include "cinderx/Jit/hir/hir_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Run HIR stats collection on a function and dump results as JSON to
 * the JIT log output. func_name is used for the JSON "function" field. */
void hir_stats_run(HirFunction func, const char *func_name);

#ifdef __cplusplus
}
#endif
