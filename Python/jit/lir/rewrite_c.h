/*
 * rewrite_c.h -- C replacement for the Rewrite framework
 *
 * Phase 3D: Replaces rewrite.h (C++ class with std::function, templates,
 * std::unordered_map) with a simple C struct + function pointer arrays.
 *
 * Key simplification: only instruction-level rewrites are used in practice.
 * The C++ framework supported function/block/instruction levels, but neither
 * PostGenerationRewrite nor PostRegAllocRewrite registers any function or
 * block rewrites. All callbacks are instruction-level.
 */

#ifndef JIT_LIR_REWRITE_C_H
#define JIT_LIR_REWRITE_C_H

#include "cinderx/Jit/lir/lir_types_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Rewrite result — matches lir::RewriteResult enum */
#define LIR_REWRITE_UNCHANGED 0
#define LIR_REWRITE_CHANGED   1
#define LIR_REWRITE_REMOVED   2

/* Maximum rewrite functions per stage. Actual usage:
 * PostGenerationRewrite: stage0=1, stage1=6-7
 * PostRegAllocRewrite: stage0=8-9, stage1=2-3
 * 16 is generous headroom. */
#define LIR_MAX_REWRITES_PER_STAGE 16

/* Maximum number of stages. Only 0 and 1 are used. */
#define LIR_MAX_STAGES 4

/*
 * Function-level rewrite callback.
 * Called once per iteration with the whole function.
 * Returns LIR_REWRITE_UNCHANGED or LIR_REWRITE_CHANGED.
 */
typedef int (*LirFuncRewriteFn)(LirFunction *func, void *env);

/*
 * Instruction rewrite callback.
 * Takes the instruction pointer and an opaque environment pointer.
 * Returns LIR_REWRITE_UNCHANGED, LIR_REWRITE_CHANGED, or LIR_REWRITE_REMOVED.
 */
typedef int (*LirInstrRewriteFn)(LirInstruction *instr, void *env);

/*
 * A single stage's rewrite function list.
 */
typedef struct {
    LirFuncRewriteFn func_rewrites[LIR_MAX_REWRITES_PER_STAGE];
    int func_count;
    LirInstrRewriteFn instr_rewrites[LIR_MAX_REWRITES_PER_STAGE];
    int instr_count;
} LirRewriteStage;

/*
 * LirRewrite — C replacement for the Rewrite class.
 *
 * Usage:
 *   LirRewrite rw;
 *   lir_rewrite_init(&rw, function, env);
 *   lir_rewrite_add(&rw, 0, my_rewrite_fn);
 *   lir_rewrite_add(&rw, 1, another_rewrite_fn);
 *   lir_rewrite_run(&rw);
 */
typedef struct {
    LirFunction *function;
    void *env;  /* codegen::Environ* — opaque in C */
    LirRewriteStage stages[LIR_MAX_STAGES];
    int num_stages;  /* highest stage index + 1 */
} LirRewrite;

/* Initialize a rewrite context. */
void lir_rewrite_init(LirRewrite *rw, LirFunction *func, void *env);

/* Register a function-level rewrite for a given stage. */
void lir_rewrite_add_func(LirRewrite *rw, int stage, LirFuncRewriteFn fn);

/* Register an instruction-level rewrite for a given stage. */
void lir_rewrite_add_instr(LirRewrite *rw, int stage, LirInstrRewriteFn fn);

/*
 * Run all registered rewrites.
 * For each stage (in order), repeatedly applies all instruction-level
 * rewrites until no changes occur (fixed-point iteration).
 */
void lir_rewrite_run(LirRewrite *rw);

/* Initialize a PostGenerationRewrite (postgen_c.c). */
void lir_postgen_rewrite_init(LirRewrite *rw, LirFunction *func, void *env);

/* Initialize a PostRegAllocRewrite (postalloc_c.c). */
void lir_postalloc_rewrite_init(LirRewrite *rw, LirFunction *func, void *env);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* JIT_LIR_REWRITE_C_H */
