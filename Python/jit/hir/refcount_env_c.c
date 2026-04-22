/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C Env struct initialization for refcount insertion pass.
 */

#include "cinderx/Jit/hir/refcount_env_c.h"
#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/hir/hir_basic_block_c.h"
#include "cinderx/Common/jit_log_c.h"
#include "Python.h"

#include <string.h>

/* HirInstr / HirRegister / hir_chase_assign come from hir_c_api.h post-W25b. */
extern void *hir_liveness_create(void *func);
extern int hir_liveness_is_last_use(const void *state, void *instr, void *reg);
extern void hir_liveness_destroy(void *state);

#define ALIAS_CLASS_NUM_BITS 10

/* ---- reg_to_bit helpers ---- */

static void env_add_bit(PhxRefcountEnv *env, void *model) {
    for (size_t i = 0; i < env->bit_reg_count; i++) {
        if (env->bit_reg_keys[i] == model) return;
    }
    if (env->bit_reg_count >= env->bit_reg_cap) {
        env->bit_reg_cap = env->bit_reg_cap ? env->bit_reg_cap * 2 : 16;
        env->bit_reg_keys = (void **)PyMem_RawRealloc(env->bit_reg_keys,
            env->bit_reg_cap * sizeof(void *));
        env->bit_reg_vals = (int *)PyMem_RawRealloc(env->bit_reg_vals,
            env->bit_reg_cap * sizeof(int));
    }
    env->bit_reg_keys[env->bit_reg_count] = model;
    env->bit_reg_vals[env->bit_reg_count] = (int)env->num_support_bits;
    env->bit_reg_count++;
    env->num_support_bits++;
}

/* ---- phi_uses helpers ---- */

static void env_add_phi_use(PhxRefcountEnv *env, void *model_reg,
                            void *pred_block, void *phi_output) {
    if (env->n_phi_uses >= env->cap_phi_uses) {
        env->cap_phi_uses = env->cap_phi_uses ? env->cap_phi_uses * 2 : 32;
        env->phi_uses = (PhxPhiUseEntry *)PyMem_RawRealloc(env->phi_uses,
            env->cap_phi_uses * sizeof(PhxPhiUseEntry));
    }
    PhxPhiUseEntry *e = &env->phi_uses[env->n_phi_uses++];
    e->model_reg = model_reg;
    e->pred_block = pred_block;
    e->phi_output = phi_output;
}

static int phi_use_cmp(const void *a, const void *b) {
    const PhxPhiUseEntry *ea = (const PhxPhiUseEntry *)a;
    const PhxPhiUseEntry *eb = (const PhxPhiUseEntry *)b;
    if (ea->model_reg < eb->model_reg) return -1;
    if (ea->model_reg > eb->model_reg) return 1;
    if (ea->pred_block < eb->pred_block) return -1;
    if (ea->pred_block > eb->pred_block) return 1;
    return 0;
}

/* ---- modelReg: chase through passthroughs (matches phx_rc_model_reg) ---- */

extern void *phx_rc_model_reg(void *reg);

static void *model_reg(void *reg) {
    return phx_rc_model_reg(reg);
}

/* ---- Public API ---- */

PhxRefcountEnv *phx_rc_env_create(HirFunction func) {
    PhxRefcountEnv *env = (PhxRefcountEnv *)PyMem_RawCalloc(1, sizeof(PhxRefcountEnv));
    env->func = func;
    env->num_support_bits = ALIAS_CLASS_NUM_BITS;

    env->liveness_state = hir_liveness_create(func);

    HirCFG *cfg = (HirCFG *)hir_func_cfg_ptr(func);

    for (HirBasicBlock *bb = hir_cfg_first_block(cfg); bb;
         bb = hir_cfg_next_block(cfg, bb)) {
        void *instr = hir_bb_first_instr(bb);
        while (instr && hir_c_is_phi(instr)) {
            void *output = hir_c_output(instr);
            env_add_bit(env, output);

            HirPhi *phi = (HirPhi *)instr;
            size_t n_ops = hir_c_num_operands(instr);
            for (size_t i = 0; i < n_ops && i < phi->bb_count; i++) {
                void *operand = hir_c_get_operand(instr, i);
                void *m = model_reg(operand);
                env_add_bit(env, m);
                env_add_phi_use(env, m, phi->bb_data[i], output);
            }

            instr = hir_bb_next_instr(bb, instr);
        }
    }

    if (env->n_phi_uses > 1) {
        qsort(env->phi_uses, env->n_phi_uses, sizeof(PhxPhiUseEntry), phi_use_cmp);
    }

    phx_sm_init(&env->live_regs);

    return env;
}

void phx_rc_env_destroy(PhxRefcountEnv *env) {
    hir_liveness_destroy(env->liveness_state);
    PyMem_RawFree(env->bit_reg_keys);
    PyMem_RawFree(env->bit_reg_vals);
    PyMem_RawFree(env->phi_uses);
    for (size_t i = 0; i < env->n_block_states; i++) {
        phx_sm_destroy(&env->block_states[i].in);
        phx_sm_destroy(&env->block_states[i].out);
    }
    PyMem_RawFree(env->block_keys);
    PyMem_RawFree(env->block_states);
    phx_sm_destroy(&env->live_regs);
    phx_bs_destroy(&env->borrow_support);
    PyMem_RawFree(env->deferred_deaths);
    PyMem_RawFree(env->borrowed_regs);
    PyMem_RawFree(env);
}

int phx_rc_env_reg_bit(const PhxRefcountEnv *env, void *model_reg) {
    for (size_t i = 0; i < env->bit_reg_count; i++) {
        if (env->bit_reg_keys[i] == model_reg) return env->bit_reg_vals[i];
    }
    return -1;
}

const PhxPhiUseEntry *phx_rc_env_phi_uses(
    const PhxRefcountEnv *env, void *model_reg, void *pred_block, size_t *count) {
    *count = 0;
    PhxPhiUseEntry key;
    key.model_reg = model_reg;
    key.pred_block = pred_block;
    key.phi_output = NULL;

    size_t lo = 0, hi = env->n_phi_uses;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        int c = phi_use_cmp(&env->phi_uses[mid], &key);
        if (c < 0) lo = mid + 1;
        else hi = mid;
    }

    const PhxPhiUseEntry *start = &env->phi_uses[lo];
    while (lo < env->n_phi_uses &&
           env->phi_uses[lo].model_reg == model_reg &&
           env->phi_uses[lo].pred_block == pred_block) {
        (*count)++;
        lo++;
    }
    return *count > 0 ? start : NULL;
}

PhxBlockState *phx_rc_env_block_state(PhxRefcountEnv *env, void *block) {
    for (size_t i = 0; i < env->n_block_states; i++) {
        if (env->block_keys[i] == block) return &env->block_states[i];
    }
    if (env->n_block_states >= env->cap_block_states) {
        env->cap_block_states = env->cap_block_states ? env->cap_block_states * 2 : 16;
        env->block_keys = (void **)PyMem_RawRealloc(env->block_keys,
            env->cap_block_states * sizeof(void *));
        env->block_states = (PhxBlockState *)PyMem_RawRealloc(env->block_states,
            env->cap_block_states * sizeof(PhxBlockState));
    }
    size_t idx = env->n_block_states++;
    env->block_keys[idx] = block;
    memset(&env->block_states[idx], 0, sizeof(PhxBlockState));
    phx_sm_init(&env->block_states[idx].in);
    phx_sm_init(&env->block_states[idx].out);
    return &env->block_states[idx];
}

/* phx_rc_is_uncounted is implemented in refcount_env_bridge.cpp
 * because it needs C++ Type constants (TMortalObject). */
