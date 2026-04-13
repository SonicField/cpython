/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Bridge between C JitConfig (canonical, defined in config.c) and
 * C++ jit::Config (legacy wrapper for unconverted callers).
 * Syncs fields on each access (safe, minimal overhead since config
 * is read infrequently relative to compilation).
 */

#include "cinderx/Jit/jit_config_c.h"
#include "cinderx/Jit/config.h"

#include <cstring>

/* C++ global — legacy wrapper.  Defined here (moved from config.cpp).
 * 180+ C++ callers use jit::getConfig() which returns this.
 * As callers migrate to jit_get_config(), this will be deleted. */
namespace jit {
Config g_jit_config;
} // namespace jit

static void sync_cpp_to_c(const jit::Config& src, JitConfig* dst) {
    dst->state = static_cast<JitState>(src.state);
    dst->force_init = src.force_init.has_value()
        ? (src.force_init.value() ? 1 : 0) : -1;
    dst->frame_mode = static_cast<JitFrameMode>(src.frame_mode);
    dst->allow_jit_list_wildcards = src.allow_jit_list_wildcards;
    dst->compile_all_static_functions = src.compile_all_static_functions;
    dst->multiple_code_sections = src.multiple_code_sections;
    dst->multithreaded_compile_test = src.multithreaded_compile_test;
    dst->use_huge_pages = src.use_huge_pages;
    dst->stable_frame = src.stable_frame;
    dst->attr_caches = src.attr_caches;
    dst->collect_attr_cache_stats = src.collect_attr_cache_stats;
    dst->emit_type_annotation_guards = src.emit_type_annotation_guards;
    dst->specialized_opcodes = src.specialized_opcodes;
    dst->support_instrumentation = src.support_instrumentation;
    dst->refine_static_python = src.refine_static_python;
    dst->compile_perf_trampoline_prefork = src.compile_perf_trampoline_prefork;
    dst->dump_hir_stats = src.dump_hir_stats;

    /* HIR optimizations */
    dst->hir_opts.begin_inlined_function_elim = src.hir_opts.begin_inlined_function_elim;
    dst->hir_opts.builtin_load_method_elim = src.hir_opts.builtin_load_method_elim;
    dst->hir_opts.clean_cfg = src.hir_opts.clean_cfg;
    dst->hir_opts.dead_code_elim = src.hir_opts.dead_code_elim;
    dst->hir_opts.dynamic_comparison_elim = src.hir_opts.dynamic_comparison_elim;
    dst->hir_opts.guard_type_removal = src.hir_opts.guard_type_removal;
    dst->hir_opts.inliner = src.hir_opts.inliner;
    dst->hir_opts.insert_update_prev_instr = src.hir_opts.insert_update_prev_instr;
    dst->hir_opts.phi_elim = src.hir_opts.phi_elim;
    dst->hir_opts.simplify = src.hir_opts.simplify;

    /* LIR optimizations */
    dst->lir_opts.inliner = src.lir_opts.inliner;

    /* Simplifier */
    dst->simplifier.iteration_limit = src.simplifier.iteration_limit;
    dst->simplifier.new_block_limit = src.simplifier.new_block_limit;

    /* Size/count fields */
    dst->inliner_cost_limit = src.inliner_cost_limit;
    dst->batch_compile_workers = src.batch_compile_workers;
    dst->preload_dependent_limit = src.preload_dependent_limit;
    dst->cold_code_section_size = src.cold_code_section_size;
    dst->hot_code_section_size = src.hot_code_section_size;
    dst->max_code_size = src.max_code_size;
    dst->attr_cache_size = src.attr_cache_size;
    dst->compile_after_n_calls = src.compile_after_n_calls.value_or(UINT32_MAX);

    /* GDB options */
    dst->gdb.supported = src.gdb.supported;
    dst->gdb.write_elf_objects = src.gdb.write_elf_objects;

    /* JIT list options */
    std::strncpy(dst->jit_list.filename, src.jit_list.filename.c_str(),
                 sizeof(dst->jit_list.filename) - 1);
    dst->jit_list.filename[sizeof(dst->jit_list.filename) - 1] = '\0';
    dst->jit_list.error_on_parse = src.jit_list.error_on_parse;
    dst->jit_list.match_line_numbers = src.jit_list.match_line_numbers;

    /* Log options */
    dst->log.debug = src.log.debug;
    dst->log.debug_inliner = src.log.debug_inliner;
    dst->log.debug_refcount = src.log.debug_refcount;
    dst->log.debug_regalloc = src.log.debug_regalloc;
    dst->log.dump_hir_initial = src.log.dump_hir_initial;
    dst->log.dump_hir_passes = src.log.dump_hir_passes;
    dst->log.dump_hir_final = src.log.dump_hir_final;
    dst->log.dump_lir = src.log.dump_lir;
    dst->log.lir_origin = src.log.lir_origin;
    dst->log.dump_asm = src.log.dump_asm;
    dst->log.symbolize_funcs = src.log.symbolize_funcs;
    dst->log.dump_stats = src.log.dump_stats;
    dst->log.output_file = src.log.output_file;

    /* ASM syntax */
    dst->asm_syntax = static_cast<JitAsmSyntax>(src.asm_syntax);
}

extern "C" {

/* jit_get_config() is now static inline in jit_config_c.h —
 * returns &g_jit_config_c directly, zero overhead. */

void jit_config_sync(void) {
    sync_cpp_to_c(jit::getConfig(), &g_jit_config_c);
}

JitConfig* jit_get_mutable_config(void) {
    sync_cpp_to_c(jit::getMutableConfig(), &g_jit_config_c);
    return &g_jit_config_c;
}

int jit_is_initialized(void) {
    return jit::isJitInitialized() ? 1 : 0;
}

int jit_is_usable(void) {
    return jit::isJitUsable() ? 1 : 0;
}

int jit_is_paused(void) {
    return jit::isJitPaused() ? 1 : 0;
}

} /* extern "C" */
