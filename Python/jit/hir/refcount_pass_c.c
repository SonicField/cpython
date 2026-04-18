/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C refcount insertion pass — helper functions.
 * Phase R3a: insertIncref/Decref, borrow support, killRegister.
 */

#include "cinderx/Jit/hir/refcount_pass_c.h"
#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/hir/hir_basic_block_c.h"
#include "cinderx/Jit/hir/hir_cfg_rpo_c.h"
#include "cinderx/Jit/hir/instr_effects_c.h"
#include "cinderx/Common/jit_log_c.h"
#include "Python.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_c_rc_log = -1;
static int c_rc_log_enabled(void) {
    if (g_c_rc_log < 0) g_c_rc_log = getenv("RC_DIFF") != NULL;
    return g_c_rc_log;
}

/* AliasClass bits for managed heap (AManagedHeapAny) */
#define ALIAS_MANAGED_HEAP_ANY 0x3FC

/* Bridge: check if register type is subtype of TObject (needs C++ constant) */
extern int phx_rc_reg_is_object(void *reg);

void phx_rc_insert_incref(PhxRefcountEnv *env, void *reg, void *cursor) {
    if (env->mutate && c_rc_log_enabled()) {
        const HirBasicBlock *blk = (const HirBasicBlock *)((HirInstrLayout *)cursor)->block;
        fprintf(stderr, "C +ref v%d bb%d\n", hir_reg_id(reg), hir_bb_id(blk));
    }
    void *incref = phx_rc_reg_is_object(reg)
        ? hir_c_create_incref(reg)
        : hir_c_create_xincref(reg);
    hir_c_copy_bytecode_offset(incref, cursor);
    hir_c_insert_before_pure(incref, cursor,
        (const HirBasicBlock *)((HirInstrLayout *)cursor)->block);
}

void phx_rc_insert_decref(PhxRefcountEnv *env, void *reg, void *cursor) {
    if (env->mutate && c_rc_log_enabled()) {
        const HirBasicBlock *blk = (const HirBasicBlock *)((HirInstrLayout *)cursor)->block;
        fprintf(stderr, "C -ref v%d bb%d\n", hir_reg_id(reg), hir_bb_id(blk));
    }
    void *decref = phx_rc_reg_is_object(reg)
        ? hir_c_create_decref(reg)
        : hir_c_create_xdecref(reg);
    hir_c_copy_bytecode_offset(decref, cursor);
    hir_c_insert_before_pure(decref, cursor,
        (const HirBasicBlock *)((HirInstrLayout *)cursor)->block);
}

void phx_rc_register_borrow_support(PhxRefcountEnv *env, PhxRegState *rstate) {
    if (rstate->kind != PHX_REF_BORROWED || phx_bs_empty(&rstate->support))
        return;
    phx_bs_add_bs(&env->borrow_support, &rstate->support);
    /* Add to borrowed_regs sorted array */
    if (env->n_borrowed >= env->cap_borrowed) {
        env->cap_borrowed = env->cap_borrowed ? env->cap_borrowed * 2 : 8;
        env->borrowed_regs = (PhxRegState **)PyMem_RawRealloc(
            env->borrowed_regs, env->cap_borrowed * sizeof(PhxRegState *));
    }
    env->borrowed_regs[env->n_borrowed++] = rstate;
}

static void invalidate_bs_impl(PhxRefcountEnv *env, void *cursor,
                                int by_acls, uint64_t acls_bits, size_t bit) {
    int any_match = by_acls
        ? phx_bs_intersects_acls(&env->borrow_support, acls_bits)
        : phx_bs_intersects_bit(&env->borrow_support, bit);
    if (!any_match) return;

    size_t dst = 0;
    for (size_t i = 0; i < env->n_borrowed; i++) {
        PhxRegState *rstate = env->borrowed_regs[i];
        int intersects = by_acls
            ? phx_bs_intersects_acls(&rstate->support, acls_bits)
            : phx_bs_intersects_bit(&rstate->support, bit);
        if (intersects) {
            phx_rs_set_owned(rstate);
            if (env->mutate) {
                phx_rc_insert_incref(env, phx_rs_current(rstate), cursor);
            }
        } else {
            env->borrowed_regs[dst++] = rstate;
        }
    }
    env->n_borrowed = dst;

    if (by_acls) {
        phx_bs_remove_acls(&env->borrow_support, acls_bits);
    } else {
        phx_bs_remove_bit(&env->borrow_support, bit);
    }
}

void phx_rc_invalidate_bs_acls(PhxRefcountEnv *env, void *cursor, uint64_t acls_bits) {
    invalidate_bs_impl(env, cursor, 1, acls_bits, 0);
}

void phx_rc_invalidate_bs_bit(PhxRefcountEnv *env, void *cursor, size_t bit) {
    invalidate_bs_impl(env, cursor, 0, 0, bit);
}

void phx_rc_kill_register(PhxRefcountEnv *env, PhxRegState *rstate,
                          void *copy, void *cursor) {
    if (!phx_rs_kill_copy(rstate, copy))
        return;

    void *model = rstate->model;
    if (rstate->kind == PHX_REF_OWNED) {
        int bit = phx_rc_env_reg_bit(env, model);
        if (bit >= 0) {
            phx_rc_invalidate_bs_bit(env, cursor, (size_t)bit);
        }
        phx_rc_invalidate_bs_acls(env, cursor, ALIAS_MANAGED_HEAP_ANY);
        if (env->mutate) {
            phx_rc_insert_decref(env, copy, cursor);
        }
    }

    /* Remove from borrowed_regs */
    for (size_t i = 0; i < env->n_borrowed; i++) {
        if (env->borrowed_regs[i] == rstate) {
            memmove(&env->borrowed_regs[i], &env->borrowed_regs[i + 1],
                    (env->n_borrowed - i - 1) * sizeof(PhxRegState *));
            env->n_borrowed--;
            break;
        }
    }

    phx_sm_erase(&env->live_regs, model);
}

/* ---- R3b-1: State initialization ---- */

/* Forward declarations */
extern void *hir_chase_assign(void *reg);
extern void *phx_rc_model_reg(void *reg);
extern int hir_liveness_is_last_use(const void *state, void *instr, void *reg);
extern int hir_liveness_is_live_in(const void *state, const void *block, void *reg);
extern void hir_liveness_foreach_live_in(
    const void *state, const void *block,
    void (*func)(void *reg, void *ctx), void *ctx);

static void *model_reg_rc(void *reg) {
    return phx_rc_model_reg(reg);
}

/* Deep-copy state into env->live_regs and re-initialize borrow support. */
void phx_rc_use_in_state(PhxRefcountEnv *env, const PhxStateMap *state) {
    phx_sm_destroy(&env->live_regs);
    phx_sm_copy(&env->live_regs, state);

    phx_bs_init(&env->borrow_support, env->num_support_bits);
    env->n_borrowed = 0;

    for (size_t i = 0; i < env->live_regs.capacity; i++) {
        if (env->live_regs.keys[i]) {
            phx_rc_register_borrow_support(env, &env->live_regs.values[i]);
        }
    }
}

/* Kill registers sorted: borrowed first (to avoid unnecessary promotions),
 * then by model register ID within each group. */
void phx_rc_kill_registers(PhxRefcountEnv *env, void **regs, size_t n_regs,
                           void *cursor) {
    if (n_regs == 0) return;

    typedef struct { void *copy; PhxRegState *rstate; } RegCopy;
    RegCopy *rcs = (RegCopy *)PyMem_RawMalloc(n_regs * sizeof(RegCopy));

    size_t n_rcs = 0;
    for (size_t i = 0; i < n_regs; i++) {
        void *model = model_reg_rc(regs[i]);
        PhxRegState *rs = phx_sm_get(&env->live_regs, model);
        if (rs) {
            rcs[n_rcs].copy = regs[i];
            rcs[n_rcs].rstate = rs;
            n_rcs++;
        }
    }

    /* Sort: borrowed before owned, then by model register ID */
    for (size_t i = 1; i < n_rcs; i++) {
        RegCopy tmp = rcs[i];
        size_t j = i;
        while (j > 0) {
            int a_bor = (rcs[j-1].rstate->kind == PHX_REF_BORROWED) ? 1 : 0;
            int b_bor = (tmp.rstate->kind == PHX_REF_BORROWED) ? 1 : 0;
            int swap = 0;
            if (b_bor && !a_bor) swap = 1;
            else if (a_bor == b_bor) {
                int a_id = hir_reg_id(rcs[j-1].rstate->model);
                int b_id = hir_reg_id(tmp.rstate->model);
                if (b_id < a_id) swap = 1;
            }
            if (!swap) break;
            rcs[j] = rcs[j-1];
            j--;
        }
        rcs[j] = tmp;
    }

    for (size_t i = 0; i < n_rcs; i++) {
        phx_rc_kill_register(env, rcs[i].rstate, rcs[i].copy, cursor);
    }

    PyMem_RawFree(rcs);
}

/* ---- R3b-2: Phi handling + block entry ---- */

extern int phx_rc_condbranch_check_type_is_wait_handle(void *instr);
extern int phx_rc_merge_verify(const PhxRegState *c_dst, const PhxRegState *c_from,
                               const PhxRegState *c_result);

/* collectPredStates: collect predecessor out-states, sorted by block ID. */
PhxPredState *phx_rc_collect_pred_states(
    PhxRefcountEnv *env, void *block, size_t *n_preds)
{
    const HirBasicBlock *bb = (const HirBasicBlock *)block;
    size_t n_in = hir_bb_in_edges_count(bb);
    PhxPredState *preds = (PhxPredState *)PyMem_RawMalloc(
        n_in * sizeof(PhxPredState));
    size_t count = 0;

    for (size_t i = 0; i < n_in; i++) {
        const HirEdge *edge = hir_bb_in_edge(bb, i);
        void *pred = edge->from;
        PhxBlockState *bs = NULL;
        for (size_t j = 0; j < env->n_block_states; j++) {
            if (env->block_keys[j] == pred) {
                bs = &env->block_states[j];
                break;
            }
        }
        if (!bs) continue;
        preds[count].block = pred;
        preds[count].state = &bs->out;
        count++;
    }

    /* Sort by block ID */
    for (size_t i = 1; i < count; i++) {
        PhxPredState tmp = preds[i];
        size_t j = i;
        while (j > 0 &&
               hir_bb_id((const HirBasicBlock *)preds[j-1].block) >
               hir_bb_id((const HirBasicBlock *)tmp.block)) {
            preds[j] = preds[j-1];
            j--;
        }
        preds[j] = tmp;
    }

    *n_preds = count;
    return preds;
}

/* isLiveIn: check if reg is live-in to block.
 * Phi outputs are NOT live-in to the block they're defined in. */
int phx_rc_is_live_in(void *block, void *reg, const PhxStateMap *in_state) {
    void *instr = hir_reg_instr_ptr(reg);
    if (instr && hir_c_is_phi(instr) &&
        hir_c_block(instr) == block) {
        return 0;
    }
    void *model = model_reg_rc(reg);
    return phx_sm_contains(in_state, model);
}

/* collectPhiInputs: collect predecessor inputs for a phi instruction.
 * Relies on preds being sorted by block ID (same order as phi->basic_blocks). */
PhxPhiInput *phx_rc_collect_phi_inputs(
    const PhxPredState *preds, size_t n_preds,
    const void *phi, size_t *n_inputs)
{
    PhxPhiInput *inputs = (PhxPhiInput *)PyMem_RawMalloc(
        n_preds * sizeof(PhxPhiInput));
    size_t count = 0;
    size_t pred_idx = 0;

    size_t n_phi_blocks = hir_phi_num_blocks(phi);
    for (size_t phi_idx = 0; pred_idx < n_preds && phi_idx < n_phi_blocks;
         phi_idx++) {
        void *phi_block = hir_phi_block_at(phi, phi_idx);
        if (phi_block != preds[pred_idx].block) {
            continue;
        }
        void *input_reg = hir_c_get_operand((void *)phi, phi_idx);
        void *input_model = model_reg_rc(input_reg);
        const PhxRegState *rs = phx_sm_get(preds[pred_idx].state, input_model);
        JIT_DCHECK_C(rs != NULL, "Phi input not found in pred state");
        inputs[count].block = preds[pred_idx].block;
        inputs[count].rstate = rs;
        count++;
        pred_idx++;
    }

    JIT_DCHECK_C(count > 0, "Processing block with no visited predecessors");
    *n_inputs = count;
    return inputs;
}

/* processPhis: inspect phi inputs and decide merged state for phi outputs.
 * Returns PhxPhiSupport with dead input info and forwarding map. */
PhxPhiSupport phx_rc_process_phis(
    PhxRefcountEnv *env, void *block,
    const PhxPredState *preds, size_t n_preds,
    PhxStateMap *in_state)
{
    PhxPhiSupport support;
    memset(&support, 0, sizeof(support));
    phx_bs_init(&support.dead, env->num_support_bits);
    support.forward_keys = NULL;
    support.forward_vals = NULL;
    support.n_forwards = 0;
    support.cap_forwards = 0;

    const HirBasicBlock *bb = (const HirBasicBlock *)block;
    void *instr = hir_bb_first_instr(bb);

    while (instr && hir_c_is_phi(instr)) {
        void *output = hir_c_output(instr);
        PhxRegState *rstate = phx_sm_get(in_state, output);
        JIT_DCHECK_C(rstate != NULL, "Phi output not in in_state");

        if (phx_rc_is_uncounted(output) || rstate->kind == PHX_REF_OWNED) {
            instr = hir_bb_next_instr(bb, instr);
            continue;
        }

        size_t n_inputs = 0;
        PhxPhiInput *inputs = phx_rc_collect_phi_inputs(
            preds, n_preds, instr, &n_inputs);

        int promote_output = 0;
        for (size_t i = 0; i < n_inputs; i++) {
            void *model = inputs[i].rstate->model;
            if (!phx_rc_is_live_in(block, model, in_state) &&
                inputs[i].rstate->kind == PHX_REF_OWNED) {
                promote_output = 1;

                int model_bit = phx_rc_env_reg_bit(env, model);
                JIT_DCHECK_C(model_bit >= 0, "Dead owned phi input has no bit");
                phx_bs_add_bit(&support.dead, (size_t)model_bit);

                /* Add/update forwarding entry */
                size_t fi;
                int found = 0;
                for (fi = 0; fi < support.n_forwards; fi++) {
                    if (support.forward_keys[fi] == (size_t)model_bit) {
                        found = 1;
                        break;
                    }
                }
                if (!found) {
                    if (support.n_forwards >= support.cap_forwards) {
                        support.cap_forwards = support.cap_forwards
                            ? support.cap_forwards * 2 : 4;
                        support.forward_keys = (size_t *)PyMem_RawRealloc(
                            support.forward_keys,
                            support.cap_forwards * sizeof(size_t));
                        support.forward_vals = (PhxBorrowSupport *)PyMem_RawRealloc(
                            support.forward_vals,
                            support.cap_forwards * sizeof(PhxBorrowSupport));
                    }
                    fi = support.n_forwards++;
                    support.forward_keys[fi] = (size_t)model_bit;
                    phx_bs_init(&support.forward_vals[fi],
                                env->num_support_bits);
                }
                int output_bit = phx_rc_env_reg_bit(env, output);
                JIT_DCHECK_C(output_bit >= 0, "Phi output has no bit");
                phx_bs_add_bit(&support.forward_vals[fi], (size_t)output_bit);
            }
        }

        if (promote_output) {
            phx_rs_set_owned(rstate);
            PyMem_RawFree(inputs);
            instr = hir_bb_next_instr(bb, instr);
            continue;
        }

        /* Otherwise: borrowed from owned inputs + support of borrowed inputs */
        phx_rs_set_borrowed(rstate, env->num_support_bits);
        for (size_t i = 0; i < n_inputs; i++) {
            if (inputs[i].rstate->kind == PHX_REF_OWNED) {
                int bit = phx_rc_env_reg_bit(env,
                    inputs[i].rstate->model);
                if (bit >= 0) {
                    phx_bs_add_bit(&rstate->support, (size_t)bit);
                }
            } else if (inputs[i].rstate->kind == PHX_REF_BORROWED) {
                phx_bs_add_bs(&rstate->support, &inputs[i].rstate->support);
            }
        }

        PyMem_RawFree(inputs);
        instr = hir_bb_next_instr(bb, instr);
    }

    return support;
}

void phx_rc_phi_support_destroy(PhxPhiSupport *ps) {
    phx_bs_destroy(&ps->dead);
    for (size_t i = 0; i < ps->n_forwards; i++) {
        phx_bs_destroy(&ps->forward_vals[i]);
    }
    PyMem_RawFree(ps->forward_keys);
    PyMem_RawFree(ps->forward_vals);
}

/* initializeInState: first-visit setup for multi-predecessor block in-state.
 * Populates in_state with live-in registers and phi outputs. */

typedef struct {
    void **regs;
    size_t count;
    size_t cap;
} RegCollector;

static void init_in_state_collect_reg(void *reg, void *ctx_raw) {
    RegCollector *c = (RegCollector *)ctx_raw;
    if (c->count >= c->cap) {
        c->cap = c->cap ? c->cap * 2 : 32;
        c->regs = (void **)PyMem_RawRealloc(c->regs,
            c->cap * sizeof(void *));
    }
    c->regs[c->count++] = reg;
}

void phx_rc_initialize_in_state(
    void *block, PhxStateMap *in_state,
    PhxRefcountEnv *env, const PhxStateMap *pred_state)
{
    const HirBasicBlock *bb = (const HirBasicBlock *)block;

    RegCollector collector = {NULL, 0, 0};
    hir_liveness_foreach_live_in(env->liveness_state, block,
                                  init_in_state_collect_reg, &collector);

    for (size_t i = 0; i < collector.count; i++) {
        void *current = collector.regs[i];
        void *model = model_reg_rc(current);
        PhxRegState *existing = phx_sm_get(in_state, model);
        if (existing) continue;

        PhxRegState *rs = phx_sm_get_or_create(in_state, model);
        /* Clear the auto-added copy (model) since we'll add copies manually */
        phx_rs_kill_copy(rs, model);

        const PhxRegState *pred_rs = phx_sm_get(pred_state, model);
        if (!pred_rs) continue;

        for (int ci = 0, cn = phx_rs_num_copies(pred_rs); ci < cn; ci++) {
            void *copy = phx_rs_copy(pred_rs, ci);
            if (hir_liveness_is_live_in(env->liveness_state, block, copy)) {
                phx_rs_add_copy(rs, copy);
            }
        }
    }

    PyMem_RawFree(collector.regs);

    /* Add phi outputs */
    void *instr = hir_bb_first_instr(bb);
    while (instr && hir_c_is_phi(instr)) {
        void *output = hir_c_output(instr);
        int already_present = phx_sm_contains(in_state, output);
        PhxRegState *rs = phx_sm_get_or_create(in_state, output);
        (void)rs;
        JIT_DCHECK_C(!already_present,
                     "Phi output register shouldn't exist in map yet");
        instr = hir_bb_next_instr(bb, instr);
    }
}

/* useSimpleInState: compute and activate in-state for ≤1 predecessor blocks. */
void phx_rc_use_simple_in_state(PhxRefcountEnv *env, void *block) {
    const HirBasicBlock *bb = (const HirBasicBlock *)block;
    size_t n_in = hir_bb_in_edges_count(bb);

    if (n_in == 0) {
        PhxStateMap empty;
        phx_sm_init(&empty);
        phx_rc_use_in_state(env, &empty);
        phx_sm_destroy(&empty);
        return;
    }

    JIT_DCHECK_C(n_in == 1, "Only blocks with <= 1 predecessors are supported");
    void *first_instr = hir_bb_first_instr(bb);
    JIT_DCHECK_C(!first_instr || !hir_c_is_phi(first_instr),
                 "Phis in a single-predecessor block are unsupported");

    const HirEdge *in_edge = hir_bb_in_edge(bb, 0);
    void *pred = in_edge->from;
    PhxBlockState *bs = phx_rc_env_block_state(env, pred);
    phx_rc_use_in_state(env, &bs->out);

    /* Adjust for CondBranch in predecessor */
    void *term = hir_bb_last_instr((const HirBasicBlock *)pred);
    int32_t op = hir_c_opcode(term);
    if (op == HIR_OP_CondBranch || op == HIR_OP_CondBranchIterNotDone) {
        void *false_bb = hir_c_condbranch_false_target(term);
        if (block == false_bb) {
            void *reg = model_reg_rc(hir_c_get_operand(term, 0));
            PhxRegState *rs = phx_sm_get(&env->live_regs, reg);
            if (rs) phx_rs_set_uncounted(rs);
        }
    } else if (op == HIR_OP_CondBranchCheckType) {
        if (phx_rc_condbranch_check_type_is_wait_handle(term)) {
            void *true_bb = hir_c_condbranch_true_target(term);
            if (block == true_bb) {
                void *reg = model_reg_rc(hir_c_get_operand(term, 0));
                PhxRegState *rs = phx_sm_get(&env->live_regs, reg);
                if (rs) phx_rs_set_uncounted(rs);
            }
        }
    }

    /* Kill registers that die across the edge */
    void **dying = NULL;
    size_t n_dying = 0, cap_dying = 0;

    for (size_t i = 0; i < env->live_regs.capacity; i++) {
        if (!env->live_regs.keys[i]) continue;
        PhxRegState *rstate = &env->live_regs.values[i];
        for (int ci = phx_rs_num_copies(rstate) - 1; ci >= 0; ci--) {
            void *reg = phx_rs_copy(rstate, ci);
            if (!hir_liveness_is_live_in(env->liveness_state, block, reg)) {
                if (n_dying >= cap_dying) {
                    cap_dying = cap_dying ? cap_dying * 2 : 16;
                    dying = (void **)PyMem_RawRealloc(dying,
                        cap_dying * sizeof(void *));
                }
                dying[n_dying++] = reg;
            }
        }
    }

    if (n_dying > 0) {
        phx_rc_kill_registers(env, dying, n_dying,
                              hir_bb_first_instr(bb));
    }
    PyMem_RawFree(dying);
}

/* updateInState: orchestrate in-state computation for multi-predecessor blocks. */
void phx_rc_update_in_state(PhxRefcountEnv *env, void *block) {
    const HirBasicBlock *bb = (const HirBasicBlock *)block;
    if (hir_bb_in_edges_count(bb) <= 1) {
        phx_rc_use_simple_in_state(env, block);
        return;
    }

    size_t n_preds = 0;
    PhxPredState *preds = phx_rc_collect_pred_states(env, block, &n_preds);

    PhxBlockState *bstate = phx_rc_env_block_state(env, block);
    PhxStateMap *in_state = &bstate->in;

    /* First visit: initialize in-state */
    int first_visit = phx_sm_empty(in_state);
    if (first_visit && n_preds > 0) {
        phx_rc_initialize_in_state(block, in_state, env, preds[0].state);
    }

    /* Process phis */
    PhxPhiSupport phi_support = phx_rc_process_phis(
        env, block, preds, n_preds, in_state);

    /* Merge predecessor states for non-phi live-in values */
    for (size_t i = 0; i < in_state->capacity; i++) {
        if (!in_state->keys[i]) continue;
        PhxRegState *rstate = &in_state->values[i];
        void *model = rstate->model;

        if (phx_rc_is_uncounted(phx_rs_current(rstate)) ||
            rstate->kind == PHX_REF_OWNED) {
            continue;
        }

        /* Skip phi outputs defined in this block */
        void *model_instr = hir_reg_instr_ptr(model);
        if (model_instr && hir_c_is_phi(model_instr) &&
            hir_c_block(model_instr) == block) {
            continue;
        }

        for (size_t pi = 0; pi < n_preds; pi++) {
            const PhxRegState *pred_rs = phx_sm_get(preds[pi].state, model);
            if (pred_rs) {
                PhxRegState pre_merge = *rstate;
                phx_rs_merge(rstate, pred_rs);
                JIT_DCHECK_C(
                    phx_rc_merge_verify(&pre_merge, pred_rs, rstate),
                    "phx_rs_merge C/C++ divergence");
                if (rstate->kind == PHX_REF_OWNED) break;
            }
        }

        /* If borrowed from now-dead phi inputs, forward to phi outputs */
        if (rstate->kind == PHX_REF_BORROWED &&
            phx_bs_intersects_bs(&rstate->support, &phi_support.dead)) {
            for (size_t fi = 0; fi < phi_support.n_forwards; fi++) {
                if (phx_bs_intersects_bit(&rstate->support,
                                           phi_support.forward_keys[fi])) {
                    phx_bs_remove_bit(&rstate->support,
                                      phi_support.forward_keys[fi]);
                    phx_bs_add_bs(&rstate->support,
                                  &phi_support.forward_vals[fi]);
                }
            }
        }
    }

    phx_rc_use_in_state(env, in_state);
    phx_rc_phi_support_destroy(&phi_support);
    PyMem_RawFree(preds);
}

/* ---- R3b-3: Per-instruction processing ---- */

extern int phx_rc_is_passthrough(void *instr);
extern int phx_rc_is_guard_is(void *instr);

/* isInFrameState: check if reg appears in localsplus or stack of any
 * FrameState in the chain. */
int phx_rc_is_in_frame_state(const void *frame_state, void *reg) {
    const HirFrameStateLayout *fs = (const HirFrameStateLayout *)frame_state;
    while (fs) {
        for (size_t i = 0; i < fs->localsplus.count; i++) {
            if (fs->localsplus.data[i] == reg) return 1;
        }
        for (size_t i = 0; i < fs->stack.count; i++) {
            if (fs->stack.data[i] == reg) return 1;
        }
        fs = fs->parent;
    }
    return 0;
}

/* stealInputs: process operands stolen by an instruction. */
void phx_rc_steal_inputs(PhxRefcountEnv *env, void *instr,
                         uint64_t stolen_mask, void *cursor)
{
    if (stolen_mask == 0) return;

    size_t n_ops = hir_c_num_operands(instr);
    for (size_t i = 0; i < n_ops; i++) {
        if (!(stolen_mask & ((uint64_t)1 << i))) continue;

        void *raw_reg = hir_c_get_operand(instr, i);
        void *reg = model_reg_rc(raw_reg);
        PhxRegState *rstate = phx_sm_get(&env->live_regs, reg);
        if (!rstate) continue;

        int is_dying = hir_liveness_is_last_use(
            env->liveness_state, instr, raw_reg);

        if (rstate->kind == PHX_REF_OWNED && is_dying) {
            int32_t op = hir_c_opcode(instr);
            if (op == HIR_OP_YieldValue ||
                op == HIR_OP_YieldFrom ||
                op == HIR_OP_YieldAndYieldFrom ||
                op == HIR_OP_YieldFromHandleStopAsyncIteration ||
                op == HIR_OP_InitialYield) {
                HirDeoptLayout *deopt = hir_c_as_deopt_mut(instr);
                if (deopt && deopt->frame_state &&
                    phx_rc_is_in_frame_state(deopt->frame_state, raw_reg)) {
                    if (env->mutate) {
                        phx_rc_insert_incref(env, raw_reg, instr);
                        continue;
                    }
                }
            }
            phx_rs_set_borrowed(rstate, env->num_support_bits);
            continue;
        }

        if (env->mutate && rstate->kind != PHX_REF_UNCOUNTED) {
            phx_rc_insert_incref(env, raw_reg, instr);
        }
    }
}

/* processOutput: track the output of an instruction. */
void phx_rc_process_output(PhxRefcountEnv *env, void *instr,
                           const void *effects_ptr)
{
    const HirMemoryEffects *effects = (const HirMemoryEffects *)effects_ptr;
    void *output = hir_c_output(instr);
    if (!output) return;

    if (phx_rc_is_passthrough(instr) && !phx_rc_is_guard_is(instr)) {
        void *model_out = model_reg_rc(output);
        PhxRegState *rstate = phx_sm_get(&env->live_regs, model_out);
        JIT_DCHECK_C(rstate != NULL, "Passthrough output not in live_regs");
        phx_rs_add_copy(rstate, output);
        if (phx_rc_is_uncounted(output)) {
            phx_rs_set_uncounted(rstate);
        }
        return;
    }

    int was_present = phx_sm_contains(&env->live_regs, output);
    PhxRegState *rstate = phx_sm_get_or_create(&env->live_regs, output);
    JIT_DCHECK_C(!was_present, "Register already defined in processOutput");
    if (phx_rc_is_uncounted(output)) {
        /* Already uncounted by default */
    } else if (effects->borrows_output) {
        phx_rs_set_borrowed(rstate, env->num_support_bits);
        phx_bs_add_acls(&rstate->support, effects->borrow_support);
        phx_rc_register_borrow_support(env, rstate);
    } else {
        phx_rs_set_owned(rstate);
    }
}

extern size_t hir_liveness_get_dying_regs(
    const void *state, void *instr, void **out_regs, size_t capacity);

/* Get dying regs for an instruction from precomputed liveness data.
 * Caller must free the returned array. */
static void **collect_dying_regs(PhxRefcountEnv *env, void *instr,
                                  size_t *n_dying_out) {
    void *stack_buf[16];
    size_t n = hir_liveness_get_dying_regs(
        env->liveness_state, instr, stack_buf, 16);

    void **dying = NULL;
    if (n > 0) {
        dying = (void **)PyMem_RawMalloc(n * sizeof(void *));
        memcpy(dying, stack_buf, n * sizeof(void *));
    }
    *n_dying_out = n;
    return dying;
}

static int is_dying_reg(PhxRefcountEnv *env, void *instr, void *reg) {
    return hir_liveness_is_last_use(env->liveness_state, instr, reg);
}

/* processInstr: process a single instruction — effects, steals, output,
 * dying regs. */
void phx_rc_process_instr(PhxRefcountEnv *env, void *instr) {
    int32_t op = hir_c_opcode(instr);
    JIT_DCHECK_C(op != HIR_OP_Incref && op != HIR_OP_Decref &&
                 op != HIR_OP_XDecref && op != HIR_OP_Snapshot,
                 "Unsupported instruction in processInstr");

    if (hir_c_num_edges(instr) > 0) {
        return;
    }

    if (hir_c_is_phi(instr)) {
        /* Phi: collect deferred deaths, kill after last phi in block.
         * Only the output can die at a phi (operands belong to predecessors). */
        void *output = hir_c_output(instr);
        int output_dying = output && is_dying_reg(env, instr, output);
        if (output_dying) {
            if (env->n_deferred >= env->cap_deferred) {
                env->cap_deferred = env->cap_deferred
                    ? env->cap_deferred * 2 : 8;
                env->deferred_deaths = (void **)PyMem_RawRealloc(
                    env->deferred_deaths,
                    env->cap_deferred * sizeof(void *));
            }
            env->deferred_deaths[env->n_deferred++] = output;
        }

        void *block = hir_c_block(instr);
        void *next = hir_bb_next_instr((const HirBasicBlock *)block, instr);
        if (next && !hir_c_is_phi(next) && env->n_deferred > 0) {
            phx_rc_kill_registers(env, env->deferred_deaths,
                                  env->n_deferred, next);
            env->n_deferred = 0;
        }
        return;
    }

    HirMemoryEffects effects = hir_memory_effects(instr);

    /* Invalidate borrow support by AliasClass */
    if (effects.may_store != 0) {
        phx_rc_invalidate_bs_acls(env, instr, effects.may_store);
    }

    /* Steal inputs */
    phx_rc_steal_inputs(env, instr, effects.stolen_mask, instr);

    /* Return check */
    if (op == HIR_OP_Return) {
        JIT_DCHECK_C(phx_sm_size(&env->live_regs) == 1,
                     "Unexpected live value(s) at Return");
        void *ret_op = model_reg_rc(hir_c_get_operand(instr, 0));
        PhxRegState *ret_rs = phx_sm_get(&env->live_regs, ret_op);
        JIT_DCHECK_C(ret_rs && phx_rs_num_copies(ret_rs) == 1,
                     "Return should have exactly one copy");
        JIT_DCHECK_C(!ret_rs || ret_rs->kind != PHX_REF_OWNED,
                     "Return operand should not be owned at exit");
        return;
    }

    /* Fill deopt live regs (mutation phase only) */
    if (env->mutate) {
        phx_rc_fill_deopt_live_regs(&env->live_regs, instr);
    }

    /* Terminator check */
    if (hir_c_is_terminator(instr)) {
        return;
    }

    /* Process output */
    phx_rc_process_output(env, instr, &effects);

    /* Kill dying registers after this instruction */
    void *block = hir_c_block(instr);
    void *next_instr = hir_bb_next_instr((const HirBasicBlock *)block, instr);

    size_t n_dying = 0;
    void **dying = collect_dying_regs(env, instr, &n_dying);
    if (n_dying > 0 && next_instr) {
        phx_rc_kill_registers(env, dying, n_dying, next_instr);
    }
    PyMem_RawFree(dying);
}

/* ---- R3b-4: exitBlock + main pass ---- */

/* exitBlock: reconcile edge state, insert Increfs for transition. */
void phx_rc_exit_block(PhxRefcountEnv *env, void *block, const void *out_edge) {
    const HirEdge *edge = (const HirEdge *)out_edge;
    void *succ = edge->to;
    const HirBasicBlock *succ_bb = (const HirBasicBlock *)succ;

    if (hir_bb_in_edges_count(succ_bb) == 1) return;

    PhxBlockState *succ_bs = NULL;
    for (size_t i = 0; i < env->n_block_states; i++) {
        if (env->block_keys[i] == succ) {
            succ_bs = &env->block_states[i];
            break;
        }
    }
    if (!succ_bs) return;
    const PhxStateMap *to_regs = &succ_bs->in;

    typedef struct { void *reg; int count; } RegIncref;
    RegIncref *increfs_arr = NULL;
    size_t n_increfs = 0, cap_increfs = 0;

    for (size_t i = 0; i < env->live_regs.capacity; i++) {
        if (!env->live_regs.keys[i]) continue;
        void *model = env->live_regs.keys[i];
        const PhxRegState *from_rstate = &env->live_regs.values[i];

        if (from_rstate->kind == PHX_REF_UNCOUNTED) continue;

        int to_owned = 0;
        if (phx_rc_is_live_in(succ, model, to_regs)) {
            const PhxRegState *to_rs = phx_sm_get(to_regs, model);
            if (to_rs && to_rs->kind == PHX_REF_OWNED) to_owned = 1;
        }

        int inc_count = to_owned - (from_rstate->kind == PHX_REF_OWNED ? 1 : 0);

        /* Add incref for each owned phi output this value feeds */
        size_t phi_count = 0;
        const PhxPhiUseEntry *phi_entries = phx_rc_env_phi_uses(
            env, model, block, &phi_count);
        for (size_t pi = 0; pi < phi_count; pi++) {
            const PhxRegState *phi_rs = phx_sm_get(to_regs,
                phi_entries[pi].phi_output);
            if (phi_rs && phi_rs->kind == PHX_REF_OWNED) {
                inc_count++;
            }
        }

        if (inc_count > 0) {
            if (n_increfs >= cap_increfs) {
                cap_increfs = cap_increfs ? cap_increfs * 2 : 8;
                increfs_arr = (RegIncref *)PyMem_RawRealloc(
                    increfs_arr, cap_increfs * sizeof(RegIncref));
            }
            increfs_arr[n_increfs].reg = phx_rs_current(from_rstate);
            increfs_arr[n_increfs].count = inc_count;
            n_increfs++;
        } else {
            JIT_DCHECK_C(inc_count == 0, "Invalid state transition in exitBlock");
        }
    }

    /* Sort by register ID */
    for (size_t i = 1; i < n_increfs; i++) {
        RegIncref tmp = increfs_arr[i];
        size_t j = i;
        while (j > 0 &&
               hir_reg_id(increfs_arr[j-1].reg) > hir_reg_id(tmp.reg)) {
            increfs_arr[j] = increfs_arr[j-1];
            j--;
        }
        increfs_arr[j] = tmp;
    }

    const HirBasicBlock *bb = (const HirBasicBlock *)block;
    void *cursor = hir_bb_last_instr(bb);
    for (size_t i = 0; i < n_increfs; i++) {
        for (int c = 0; c < increfs_arr[i].count; c++) {
            phx_rc_insert_incref(env, increfs_arr[i].reg, cursor);
        }
    }

    PyMem_RawFree(increfs_arr);
}

/* ---- Simple worklist (FIFO queue with dedup) ---- */

typedef struct {
    void **queue;
    size_t head;
    size_t tail;
    size_t capacity;
    void **set_keys;
    size_t set_count;
    size_t set_cap;
} PhxWorklist;

static void wl_init(PhxWorklist *wl, size_t cap) {
    wl->queue = (void **)PyMem_RawMalloc(cap * sizeof(void *));
    wl->head = 0;
    wl->tail = 0;
    wl->capacity = cap;
    wl->set_keys = (void **)PyMem_RawCalloc(cap, sizeof(void *));
    wl->set_count = 0;
    wl->set_cap = cap;
}

static void wl_destroy(PhxWorklist *wl) {
    PyMem_RawFree(wl->queue);
    PyMem_RawFree(wl->set_keys);
}

static int wl_empty(const PhxWorklist *wl) {
    return wl->head == wl->tail;
}

static int wl_set_contains(const PhxWorklist *wl, void *item) {
    for (size_t i = 0; i < wl->set_count; i++) {
        if (wl->set_keys[i] == item) return 1;
    }
    return 0;
}

static void wl_set_add(PhxWorklist *wl, void *item) {
    if (wl->set_count >= wl->set_cap) {
        wl->set_cap *= 2;
        wl->set_keys = (void **)PyMem_RawRealloc(
            wl->set_keys, wl->set_cap * sizeof(void *));
    }
    wl->set_keys[wl->set_count++] = item;
}

static void wl_set_remove(PhxWorklist *wl, void *item) {
    for (size_t i = 0; i < wl->set_count; i++) {
        if (wl->set_keys[i] == item) {
            wl->set_keys[i] = wl->set_keys[--wl->set_count];
            return;
        }
    }
}

static void wl_push(PhxWorklist *wl, void *item) {
    if (wl_set_contains(wl, item)) return;
    if (wl->tail >= wl->capacity) {
        wl->capacity *= 2;
        wl->queue = (void **)PyMem_RawRealloc(
            wl->queue, wl->capacity * sizeof(void *));
    }
    wl->queue[wl->tail++] = item;
    wl_set_add(wl, item);
}

static void *wl_front(const PhxWorklist *wl) {
    return wl->queue[wl->head];
}

static void wl_pop(PhxWorklist *wl) {
    void *item = wl->queue[wl->head++];
    wl_set_remove(wl, item);
}

/* ---- Main pass ---- */

void phx_rc_run(PhxRefcountEnv *env) {
    HirCFG *cfg = (HirCFG *)hir_func_cfg_ptr(env->func);

    /* Get RPO traversal */
    size_t rpo_cap = 64;
    void **rpo_blocks = (void **)PyMem_RawMalloc(rpo_cap * sizeof(void *));
    size_t n_rpo = hir_cfg_get_rpo_c(cfg, rpo_blocks, rpo_cap);
    if (n_rpo > rpo_cap) {
        rpo_cap = n_rpo;
        rpo_blocks = (void **)PyMem_RawRealloc(
            rpo_blocks, rpo_cap * sizeof(void *));
        hir_cfg_get_rpo_c(cfg, rpo_blocks, rpo_cap);
    }

    /* Analysis phase: fixed-point iteration */
    PhxWorklist worklist;
    wl_init(&worklist, n_rpo * 2);
    for (size_t i = 0; i < n_rpo; i++) {
        wl_push(&worklist, rpo_blocks[i]);
    }

    while (!wl_empty(&worklist)) {
        void *block = wl_front(&worklist);
        wl_pop(&worklist);

        phx_rc_update_in_state(env, block);

        const HirBasicBlock *bb = (const HirBasicBlock *)block;
        void *instr = hir_bb_first_instr(bb);
        while (instr) {
            void *next = hir_bb_next_instr(bb, instr);
            phx_rc_process_instr(env, instr);
            instr = next;
        }

        PhxBlockState *bstate = phx_rc_env_block_state(env, block);
        if (!phx_sm_equal(&env->live_regs, &bstate->out)) {
            phx_sm_destroy(&bstate->out);
            bstate->out = env->live_regs;
            phx_sm_init(&env->live_regs);

            size_t n_out = hir_bb_out_edges_count(bb);
            for (size_t ei = 0; ei < n_out; ei++) {
                const HirEdge *edge = hir_bb_out_edge(bb, ei);
                wl_push(&worklist, edge->to);
            }
        }
    }

    wl_destroy(&worklist);
    /* Mutation phase */
    env->mutate = 1;
    for (size_t bi = 0; bi < n_rpo; bi++) {
        void *block = rpo_blocks[bi];
        const HirBasicBlock *bb = (const HirBasicBlock *)block;

        void *first_instr = hir_bb_first_instr(bb);

        if (hir_bb_in_edges_count(bb) <= 1) {
            phx_rc_use_simple_in_state(env, block);
        } else {
            PhxBlockState *bstate = phx_rc_env_block_state(env, block);
            phx_rc_use_in_state(env, &bstate->in);
        }

        void *instr = first_instr;
        while (instr) {
            void *next = hir_bb_next_instr(bb, instr);
            phx_rc_process_instr(env, instr);
            instr = next;
        }
        if (hir_bb_out_edges_count(bb) == 1) {
            const HirEdge *out_edge = hir_bb_out_edge(bb, 0);
            phx_rc_exit_block(env, block, out_edge);
        }
    }
    PyMem_RawFree(rpo_blocks);
}
