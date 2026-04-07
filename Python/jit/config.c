/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Canonical JIT configuration storage — pure C.
 * C++ callers access this via jit::getConfig() / jit::getMutableConfig()
 * which delegate through the bridge (jit_config_c_bridge.cpp).
 */

#include "cinderx/Jit/jit_config_c.h"

JitConfig g_jit_config_c = {
    /* state */
    JIT_STATE_NOT_INITIALIZED,
    /* force_init (-1 = unset) */
    -1,
    /* frame_mode */
#ifdef ENABLE_LIGHTWEIGHT_FRAMES
    JIT_FRAME_LIGHTWEIGHT,
#else
    JIT_FRAME_NORMAL,
#endif
    /* allow_jit_list_wildcards */       0,
    /* compile_all_static_functions */   0,
    /* multiple_code_sections */         0,
    /* multithreaded_compile_test */     0,
    /* use_huge_pages */                 1,
    /* stable_frame */                   1,
    /* attr_caches */
#ifdef Py_GIL_DISABLED
    0,
#else
    1,
#endif
    /* collect_attr_cache_stats */       0,
    /* emit_type_annotation_guards */    0,
    /* specialized_opcodes */            1,
    /* support_instrumentation */        0,
    /* refine_static_python */           1,
    /* compile_perf_trampoline_prefork */ 0,
    /* dump_hir_stats */                 0,
    /* hir_opts */
    { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
    /* lir_opts */
    { 1 },
    /* simplifier */
    { 100, 1000 },
    /* inliner_cost_limit */            2000,
    /* batch_compile_workers */          0,
    /* preload_dependent_limit */        99,
    /* cold_code_section_size */         0,
    /* hot_code_section_size */          0,
    /* max_code_size */                  0,
    /* attr_cache_size */                4,
    /* compile_after_n_calls (UINT32_MAX = never) */
    0xFFFFFFFF,
    /* gdb */
    { 0, 0 },
    /* jit_list */
    { "", 0, 0 },
    /* log (output_file set to stderr lazily — not a constant initializer in C) */
    { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, NULL /* stderr, set by bridge */ },
    /* asm_syntax */
    JIT_ASM_ATT
};
