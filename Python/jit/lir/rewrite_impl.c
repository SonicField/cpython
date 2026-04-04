/*
 * rewrite_impl.c -- C implementation of the LIR Rewrite framework
 *
 * Phase 3D: Replaces rewrite.cpp (C++ class with std::function, templates,
 * concepts, std::unordered_map) with ~60 lines of pure C.
 *
 * The framework iterates instruction-level rewrite functions in staged
 * order, repeating each stage until a fixed point (no more changes).
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
lir_rewrite_add(LirRewrite *rw, int stage, LirInstrRewriteFn fn) {
    assert(stage >= 0 && stage < LIR_MAX_STAGES);
    LirRewriteStage *s = &rw->stages[stage];
    assert(s->count < LIR_MAX_REWRITES_PER_STAGE);
    s->funcs[s->count++] = fn;
    if (stage + 1 > rw->num_stages) {
        rw->num_stages = stage + 1;
    }
}

/*
 * Run one stage: iterate all blocks and instructions, applying each
 * registered rewrite. Repeat until no changes (fixed-point).
 */
static void
run_one_stage(LirRewrite *rw, int stage) {
    LirRewriteStage *s = &rw->stages[stage];
    if (s->count == 0) return;

    int changed;
    do {
        changed = 0;
        for (size_t bi = 0; bi < rw->function->num_blocks_; bi++) {
            LirBasicBlock *bb = rw->function->blocks_[bi];
            LirInstruction *instr = bb->instr_head_;
            while (instr != NULL) {
                LirInstruction *next = instr->next_;
                for (int fi = 0; fi < s->count; fi++) {
                    int result = s->funcs[fi](instr, rw->env);
                    if (result != LIR_REWRITE_UNCHANGED) {
                        changed = 1;
                    }
                    if (result == LIR_REWRITE_REMOVED) {
                        /* Instruction was removed — stop processing it */
                        goto next_instr;
                    }
                }
                next_instr:
                instr = next;
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
