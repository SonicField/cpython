/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C refcount insertion pass — helper functions.
 * Phase R3a: insertIncref/Decref, borrow support, killRegister.
 */

#include "cinderx/Jit/hir/refcount_pass_c.h"
#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/hir/hir_basic_block_c.h"
#include "cinderx/Common/jit_log_c.h"
#include "Python.h"

#include <string.h>

/* AliasClass bits for managed heap (AManagedHeapAny) */
#define ALIAS_MANAGED_HEAP_ANY 0x3FC

/* Bridge: check if register type is subtype of TObject (needs C++ constant) */
extern int phx_rc_reg_is_object(void *reg);

void phx_rc_insert_incref(PhxRefcountEnv *env, void *reg, void *cursor) {
    void *incref = phx_rc_reg_is_object(reg)
        ? hir_c_create_incref(reg)
        : hir_c_create_xincref(reg);
    hir_c_copy_bytecode_offset(incref, cursor);
    hir_c_insert_before_pure(incref, cursor,
        (const HirBasicBlock *)((HirInstrLayout *)cursor)->block);
}

void phx_rc_insert_decref(PhxRefcountEnv *env, void *reg, void *cursor) {
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
extern int hir_liveness_is_last_use(const void *state, void *instr, void *reg);

static void *model_reg_rc(void *reg) {
    return hir_chase_assign(reg);
}

/* Copy state into env and re-initialize borrow support tracking. */
void phx_rc_use_in_state(PhxRefcountEnv *env, const PhxStateMap *state) {
    phx_sm_destroy(&env->live_regs);
    env->live_regs = *state;
    /* Re-init the destination (source is shallow-moved) */
    phx_sm_init((PhxStateMap *)state);

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
