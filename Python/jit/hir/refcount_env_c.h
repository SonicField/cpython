/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C Env struct for refcount insertion pass.
 * Phase R2: initialization and query functions.
 */
#pragma once

#include "cinderx/Jit/hir/refcount_structs_c.h"
#include "cinderx/Jit/hir/hir_c_api.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- PhiUse entry: (model_reg, pred_block) → phi_output ---- */
typedef struct {
    void *model_reg;    /* Register* */
    void *pred_block;   /* BasicBlock* */
    void *phi_output;   /* Register* (phi output that this feeds) */
} PhxPhiUseEntry;

/* ---- BlockState: in/out StateMaps per block ---- */
typedef struct {
    PhxStateMap in;
    PhxStateMap out;
} PhxBlockState;

/* ---- Refcount Env ---- */
typedef struct {
    void *func;                     /* HirFunction */

    /* Liveness (opaque, from liveness_c) */
    void *liveness_state;           /* HirLivenessState* */

    /* Support bit assignments for phi registers */
    size_t num_support_bits;
    void **bit_reg_keys;            /* Register* keys */
    int *bit_reg_vals;              /* support bit index values */
    size_t bit_reg_count;
    size_t bit_reg_cap;

    /* Phi use entries (flat, sorted by model_reg then pred_block) */
    PhxPhiUseEntry *phi_uses;
    size_t n_phi_uses;
    size_t cap_phi_uses;

    /* Per-block in/out states */
    void **block_keys;              /* BasicBlock* keys */
    PhxBlockState *block_states;    /* parallel array */
    size_t n_block_states;
    size_t cap_block_states;

    /* Mutable pass state */
    int mutate;
    PhxStateMap live_regs;
    PhxBorrowSupport borrow_support;

    /* Deferred phi output deaths */
    void **deferred_deaths;
    size_t n_deferred;
    size_t cap_deferred;

    /* Borrowed model register keys (void* keys into live_regs, not value ptrs).
     * Value pointers go stale on sm_grow — store keys and look up on demand. */
    void **borrowed_regs;
    size_t n_borrowed;
    size_t cap_borrowed;
} PhxRefcountEnv;

/* Initialize the env: run liveness, collect phi metadata, assign bits. */
PhxRefcountEnv *phx_rc_env_create(HirFunction func);

/* Destroy and free the env. */
void phx_rc_env_destroy(PhxRefcountEnv *env);

/* Look up the support bit for a register. Returns -1 if not found. */
int phx_rc_env_reg_bit(const PhxRefcountEnv *env, void *model_reg);

/* Get phi outputs for a (model_reg, pred_block) pair.
 * Returns pointer to first match and sets *count. */
const PhxPhiUseEntry *phx_rc_env_phi_uses(
    const PhxRefcountEnv *env, void *model_reg, void *pred_block, size_t *count);

/* Get or create BlockState for a block. */
PhxBlockState *phx_rc_env_block_state(PhxRefcountEnv *env, void *block);

/* Check if a register is definitely not reference-counted. */
int phx_rc_is_uncounted(void *reg);

/* Refcount-pass support helpers (Batch 2-B: promoted from
 * refcount_env_bridge.cpp, replaces the per-caller extern decls in
 * refcount_pass_c.c with a canonical header). */
int phx_rc_reg_is_object(void *reg);
int phx_rc_condbranch_check_type_is_wait_handle(void *instr);
int phx_rc_is_passthrough(void *instr);
int phx_rc_is_guard_is(void *instr);
void phx_rc_fill_deopt_live_regs(const PhxStateMap *live_regs, void *instr_ptr);
void *phx_rc_model_reg(void *reg);
size_t phx_rc_get_rpo(void *func_ptr, void **out, size_t capacity);

#ifdef __cplusplus
} /* extern "C" */
#endif
