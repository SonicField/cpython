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

#ifdef __cplusplus
} /* extern "C" */
#endif
