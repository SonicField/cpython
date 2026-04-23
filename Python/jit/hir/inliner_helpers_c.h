/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Pure-C helpers for inliner.cpp — Tier 7 Batch 3 Cat-A extraction
 * (theologian 20:52:14Z + supervisor 20:52:14Z; codeCost is the only
 * pure-C-tractable helper among inliner's 8 functions).
 */
#pragma once

#include "cinderx/python.h"

#ifdef __cplusplus
extern "C" {
#endif

/* codeCost ported from inliner.cpp:100-108.
 *
 * Counts real opcodes (skipping inline-cache CACHE entries) in a
 * PyCodeObject. Matches the existing iteration semantics of
 * BytecodeInstructionBlock — the JitBytecodeBlock C iteration
 * already handles EXTENDED_ARG / EXTENDED_OPCODE / specialization /
 * cache-skipping. */
size_t phx_code_cost(PyCodeObject* code);

#ifdef __cplusplus
}
#endif
