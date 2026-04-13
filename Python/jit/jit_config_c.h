/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C-compatible JIT configuration struct — Phase B replacement for
 * the C++ jit::Config struct.
 *
 * Initially a read-only view: jit_get_config() returns a pointer to
 * a C struct populated from the C++ Config. Future: C struct becomes
 * canonical storage, C++ Config wraps it.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Enums ---- */

typedef enum {
    JIT_STATE_NOT_INITIALIZED = 0,
    JIT_STATE_RUNNING,
    JIT_STATE_PAUSED,
    JIT_STATE_FINALIZING
} JitState;

typedef enum {
    JIT_FRAME_NORMAL = 0,
    JIT_FRAME_SHADOW,
    JIT_FRAME_LIGHTWEIGHT
} JitFrameMode;

typedef enum {
    JIT_ASM_ATT = 0,
    JIT_ASM_INTEL
} JitAsmSyntax;

/* ---- Sub-structs ---- */

typedef struct {
    int begin_inlined_function_elim;
    int builtin_load_method_elim;
    int clean_cfg;
    int dead_code_elim;
    int dynamic_comparison_elim;
    int guard_type_removal;
    int inliner;
    int insert_update_prev_instr;
    int phi_elim;
    int simplify;
} JitHIROpts;

typedef struct {
    int inliner;
} JitLIROpts;

typedef struct {
    size_t iteration_limit;
    size_t new_block_limit;
} JitSimplifierCfg;

typedef struct {
    int supported;
    int write_elf_objects;
} JitGdbOpts;

typedef struct {
    char filename[256];
    int error_on_parse;
    int match_line_numbers;
} JitListOpts;

typedef struct {
    int debug;
    int debug_inliner;
    int debug_refcount;
    int debug_regalloc;
    int dump_hir_initial;
    int dump_hir_passes;
    int dump_hir_final;
    int dump_lir;
    int lir_origin;
    int dump_asm;
    int symbolize_funcs;
    int dump_stats;
    FILE *output_file;
} JitLogOpts;

/* ---- Main config struct ---- */

typedef struct {
    JitState state;
    int force_init;          /* -1 = unset (std::nullopt), 0 = false, 1 = true */
    JitFrameMode frame_mode;
    int allow_jit_list_wildcards;
    int compile_all_static_functions;
    int multiple_code_sections;
    int multithreaded_compile_test;
    int use_huge_pages;
    int stable_frame;
    int attr_caches;
    int collect_attr_cache_stats;
    int emit_type_annotation_guards;
    int specialized_opcodes;
    int support_instrumentation;
    int refine_static_python;
    int compile_perf_trampoline_prefork;
    int dump_hir_stats;
    JitHIROpts hir_opts;
    JitLIROpts lir_opts;
    JitSimplifierCfg simplifier;
    size_t inliner_cost_limit;
    size_t batch_compile_workers;
    size_t preload_dependent_limit;
    size_t cold_code_section_size;
    size_t hot_code_section_size;
    size_t max_code_size;
    uint32_t attr_cache_size;
    uint32_t compile_after_n_calls; /* UINT32_MAX = unset */
    JitGdbOpts gdb;
    JitListOpts jit_list;
    JitLogOpts log;
    JitAsmSyntax asm_syntax;
} JitConfig;

/* ---- Canonical global ---- */

/* Canonical JIT config storage — defined in config.c.
 * C callers access directly; C++ callers go through jit::getConfig(). */
extern JitConfig g_jit_config_c;

/* ---- Accessors ---- */

/* Get the current JIT config (read-only).
 * Static inline — no function call overhead on hot paths.
 * The config is synced from C++ at JIT init via jit_config_sync(). */
static inline const JitConfig* jit_get_config(void) {
    return &g_jit_config_c;
}

/* Get a mutable pointer to the JIT config.
 * Call jit_config_sync() after modifying C++ config directly. */
JitConfig* jit_get_mutable_config(void);

/* Sync C++ Config → C JitConfig. Called once at JIT init and
 * after any C++ config modification (flag processing, etc.). */
void jit_config_sync(void);

/* State query helpers */
int jit_is_initialized(void);
int jit_is_usable(void);
int jit_is_paused(void);

#ifdef __cplusplus
} /* extern "C" */
#endif
