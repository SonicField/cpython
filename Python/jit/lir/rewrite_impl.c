/*
 * rewrite_impl.c -- C implementation of the LIR Rewrite framework
 *
 * Phase 3D: Replaces rewrite.cpp (C++ class with std::function, templates,
 * concepts, std::unordered_map) with ~80 lines of pure C.
 *
 * The framework iterates function-level and instruction-level rewrite
 * functions in staged order, repeating each stage until a fixed point
 * (no more changes).
 */

#include "cinderx/Jit/lir/rewrite_c.h"

#include <assert.h>
#include <string.h>

void
lir_rewrite_init(LirRewrite *rw, LirFunction *func, void *env) {
    memset(rw, 0, sizeof(*rw));
    rw->function = func;
    rw->env = env;
}

void
lir_rewrite_add_func(LirRewrite *rw, int stage, LirFuncRewriteFn fn) {
    assert(stage >= 0 && stage < LIR_MAX_STAGES);
    LirRewriteStage *s = &rw->stages[stage];
    assert(s->func_count < LIR_MAX_REWRITES_PER_STAGE);
    s->func_rewrites[s->func_count++] = fn;
    if (stage + 1 > rw->num_stages) {
        rw->num_stages = stage + 1;
    }
}

void
lir_rewrite_add_instr(LirRewrite *rw, int stage, LirInstrRewriteFn fn) {
    assert(stage >= 0 && stage < LIR_MAX_STAGES);
    LirRewriteStage *s = &rw->stages[stage];
    assert(s->instr_count < LIR_MAX_REWRITES_PER_STAGE);
    s->instr_rewrites[s->instr_count++] = fn;
    if (stage + 1 > rw->num_stages) {
        rw->num_stages = stage + 1;
    }
}

/*
 * Run one stage: first apply function-level rewrites, then iterate all
 * blocks and instructions applying instruction-level rewrites.
 * Repeat until no changes (fixed-point).
 *
 * Matches C++ Rewrite::runOneStage semantics.
 */
static void
run_one_stage(LirRewrite *rw, int stage) {
    LirRewriteStage *s = &rw->stages[stage];
    int has_func = s->func_count > 0;
    int has_instr = s->instr_count > 0;

    if (!has_func && !has_instr) return;

    int changed;
    do {
        changed = 0;

        /* Function-level rewrites */
        if (has_func) {
            for (int fi = 0; fi < s->func_count; fi++) {
                int result = s->func_rewrites[fi](rw->function, rw->env);
                if (result != LIR_REWRITE_UNCHANGED) {
                    changed = 1;
                }
            }
        }

        /* Instruction-level rewrites — nested fixed-point per instruction.
         * Matches C++ runOneTypeRewrites: repeat ALL callbacks on the SAME
         * instruction until none changes, then move to next instruction. */
        if (has_instr) {
            for (size_t bi = 0; bi < rw->function->num_blocks_; bi++) {
                LirBasicBlock *bb = rw->function->blocks_[bi];
                LirInstruction *instr = bb->instr_head_;
                while (instr != NULL) {
                    LirInstruction *next = instr->next_;
                    int instr_changed;
                    do {
                        instr_changed = 0;
                        for (int fi = 0; fi < s->instr_count; fi++) {
                            int result = s->instr_rewrites[fi](instr, rw->env);
                            if (result != LIR_REWRITE_UNCHANGED) {
                                instr_changed = 1;
                                changed = 1;
                            }
                            if (result == LIR_REWRITE_REMOVED) {
                                goto next_instr;
                            }
                        }
                    } while (instr_changed);
                    next_instr:
                    instr = next;
                }
            }
        }
    } while (changed);
}

void
lir_rewrite_run(LirRewrite *rw) {
    for (int stage = 0; stage < rw->num_stages; stage++) {
        run_one_stage(rw, stage);
    }
}
