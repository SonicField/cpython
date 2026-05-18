/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * PhxFrameStatePtrMap — typed alias over PhxPtrMap specialized for
 * FrameState*-keyed FrameState*-valued maps. E-9 substrate 2.8 per
 * scribe D-1777602396; discharges UnorderedMap<FrameState*, FrameState*>
 * at builder.cpp:800 (framestate_parent inline-bookkeeping during the
 * disconnect-then-relink sequence around SSAify on inlined functions).
 *
 * Per theologian 19:21:55Z + alex_no_methodology team-decision: simpler
 * alias + inline typed accessors (vs standalone struct primitive). Zero
 * indirection cost; type-safety + caller-readability over void*-cast
 * boilerplate.
 *
 * NULL-key constraint inherited from PhxPtrMap (phx_ptr_map.h:20-21):
 * FrameState* NULL is forbidden as key. The builder.cpp:800 caller
 * already guards `if (fs == nullptr) continue` before insert; constraint
 * preserved at the wire site.
 *
 * Insert semantics: phx_frame_state_ptr_map_insert returns 1 if newly
 * inserted, 0 if key existed (overwrites silently). Compatible with the
 * UnorderedMap.emplace(...).second contract at builder.cpp:800 where
 * the caller asserts JIT_CHECK(inserted, "no duplicate FrameState
 * pointers") — assertion fires before any overwrite would be visible.
 *
 * Iteration: slot-index walk identical to PhxPtrMap (phx_ptr_map.h:173-189
 * pattern); unordered. Matches UnorderedMap unordered-iteration semantic.
 */
#pragma once

#include "cinderx/Jit/hir/phx_ptr_map.h"

#ifdef __cplusplus

namespace jit::hir {
struct FrameState;
}

typedef PhxPtrMap PhxFrameStatePtrMap;

static inline void phx_frame_state_ptr_map_init(PhxFrameStatePtrMap *m) {
    phx_ptr_map_init(m);
}

static inline void phx_frame_state_ptr_map_destroy(PhxFrameStatePtrMap *m) {
    phx_ptr_map_destroy(m);
}

static inline int phx_frame_state_ptr_map_insert(
        PhxFrameStatePtrMap *m,
        jit::hir::FrameState *key,
        jit::hir::FrameState *value) {
    return phx_ptr_map_insert(m, (void *)key, (void *)value);
}

static inline size_t phx_frame_state_ptr_map_size(const PhxFrameStatePtrMap *m) {
    return phx_ptr_map_size(m);
}

static inline size_t phx_frame_state_ptr_map_capacity(
        const PhxFrameStatePtrMap *m) {
    return phx_ptr_map_capacity(m);
}

static inline jit::hir::FrameState *phx_frame_state_ptr_map_at_key(
        const PhxFrameStatePtrMap *m, size_t slot_idx) {
    return (jit::hir::FrameState *)phx_ptr_map_at_key(m, slot_idx);
}

static inline jit::hir::FrameState *phx_frame_state_ptr_map_at_value(
        const PhxFrameStatePtrMap *m, size_t slot_idx) {
    return (jit::hir::FrameState *)phx_ptr_map_at_value(m, slot_idx);
}

#endif /* __cplusplus */
