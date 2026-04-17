/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C replacement for instr_effects.h — memory effects and arbitrary execution
 * queries for HIR instructions. Pure opcode dispatch, no state.
 */
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int borrows_output;
    uint64_t borrow_support;
    uint64_t stolen_mask;
    uint64_t may_store;
} HirMemoryEffects;

HirMemoryEffects hir_memory_effects(const void *instr);
int hir_has_arbitrary_execution(const void *instr);

#ifdef __cplusplus
}
#endif
