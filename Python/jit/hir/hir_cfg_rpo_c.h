/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C GetRPOTraversal — pure C reverse postorder traversal of HIR CFG.
 * Replaces cfg.cpp GetRPOTraversal/GetPostOrderTraversal for C callers.
 */
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Compute reverse postorder traversal of the CFG starting from entry_block.
 * Writes block pointers (as void*) into rpo_out, up to capacity entries.
 * Returns number of blocks written.
 * Caller must provide rpo_out with at least capacity entries. */
size_t hir_cfg_get_rpo_c(void *cfg, void **rpo_out, size_t capacity);

#ifdef __cplusplus
} /* extern "C" */
#endif
