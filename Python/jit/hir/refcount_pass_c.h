/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C refcount insertion pass — helper functions and main pass.
 * Phase R3: pass logic.
 */
#pragma once

#include "cinderx/Jit/hir/refcount_env_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- R3a: helper functions ---- */

/* Insert an Incref of reg before cursor. Only call when env->mutate is set. */
void phx_rc_insert_incref(PhxRefcountEnv *env, void *reg, void *cursor);

/* Insert a Decref of reg before cursor. Only call when env->mutate is set. */
void phx_rc_insert_decref(PhxRefcountEnv *env, void *reg, void *cursor);

/* Track borrow support for a RegState if it's borrowed with non-empty support. */
void phx_rc_register_borrow_support(PhxRefcountEnv *env, PhxRegState *rstate);

/* Invalidate borrow support by AliasClass bits, promoting borrowed→owned. */
void phx_rc_invalidate_bs_acls(PhxRefcountEnv *env, void *cursor, uint64_t acls_bits);

/* Invalidate borrow support by register bit index. */
void phx_rc_invalidate_bs_bit(PhxRefcountEnv *env, void *cursor, size_t bit);

/* Kill a register copy. If last copy, untrack and insert Decref if owned. */
void phx_rc_kill_register(PhxRefcountEnv *env, PhxRegState *rstate,
                          void *copy, void *cursor);

/* Copy state into env and re-initialize borrow support tracking. */
void phx_rc_use_in_state(PhxRefcountEnv *env, const PhxStateMap *state);

/* Kill registers sorted: borrowed first, then by model ID. */
void phx_rc_kill_registers(PhxRefcountEnv *env, void **regs, size_t n_regs,
                           void *cursor);

/* ---- R3b-2: phi handling + block entry ---- */

/* Predecessor state for multi-predecessor blocks. */
typedef struct {
    void *block;             /* BasicBlock* */
    const PhxStateMap *state; /* pointer to predecessor's out-state */
} PhxPredState;

/* Phi input: predecessor block + RegState for the value it provides. */
typedef struct {
    void *block;             /* BasicBlock* */
    const PhxRegState *rstate;
} PhxPhiInput;

/* Phi support info: dead phi inputs and their forwarding targets. */
typedef struct {
    PhxBorrowSupport dead;
    /* Parallel arrays: forward_keys[i] → forward_values[i] */
    size_t *forward_keys;          /* bit index of dead model */
    PhxBorrowSupport *forward_vals; /* forwarded-to bits */
    size_t n_forwards;
    size_t cap_forwards;
} PhxPhiSupport;

/* Collect predecessor out-states for a multi-predecessor block, sorted by
 * block ID. Returns array (caller must free), sets *n_preds. */
PhxPredState *phx_rc_collect_pred_states(
    PhxRefcountEnv *env, void *block, size_t *n_preds);

/* Check if reg is live-in to block (not a phi output defined in that block). */
int phx_rc_is_live_in(void *block, void *reg, const PhxStateMap *in_state);

/* Collect phi inputs from predecessor states.
 * Returns array (caller must free), sets *n_inputs. */
PhxPhiInput *phx_rc_collect_phi_inputs(
    const PhxPredState *preds, size_t n_preds,
    const void *phi, size_t *n_inputs);

/* Process all phis in block, deciding merged ownership/borrow for outputs.
 * Returns PhxPhiSupport (caller must destroy with phx_rc_phi_support_destroy). */
PhxPhiSupport phx_rc_process_phis(
    PhxRefcountEnv *env, void *block,
    const PhxPredState *preds, size_t n_preds,
    PhxStateMap *in_state);

void phx_rc_phi_support_destroy(PhxPhiSupport *ps);

/* Initialize in-state for a first-visit multi-predecessor block. */
void phx_rc_initialize_in_state(
    void *block, PhxStateMap *in_state,
    PhxRefcountEnv *env, const PhxStateMap *pred_state);

/* Compute and activate in-state for a block with ≤1 predecessors. */
void phx_rc_use_simple_in_state(PhxRefcountEnv *env, void *block);

/* Update in-state for a multi-predecessor block: initialize, process phis,
 * merge predecessor states, update borrow support forwarding. */
void phx_rc_update_in_state(PhxRefcountEnv *env, void *block);

#ifdef __cplusplus
} /* extern "C" */
#endif
