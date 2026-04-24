/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * PhxHirBuilderState — Phase 3 Batch 1 foundation per
 * docs/tier7-phase3-hirbuilder-state-extraction-spec.md §3.1.
 *
 * Class A members (5 immutable + nullable opaque pointers) extracted
 * from HIRBuilder. Class B members (block_map_, exception_table_, etc.)
 * remain C++-side and access via _cpp suffix bridges (push/clear/iterate)
 * until per-batch migration per spec §5 #1. (pending_b2_blocks_ was
 * dead-state-deleted Phase 3 Batch 3 — zero writers post-W26 refactor.)
 *
 * Authorization: theologian 23:05:15Z + supervisor 23:02:34Z (Y) atomic.
 */

#ifndef PHX_BUILDER_STATE_C_H
#define PHX_BUILDER_STATE_C_H

#include "Python.h"

#ifdef __cplusplus
extern "C" {
#endif

/* PhxHirBuilderState — opaque holder for HIRBuilder Class A state.
 * code + preloader are immutable post-ctor; current_func/func/kwnames
 * mutate during translate()/emit. Subsequent batches add Class B
 * opaque pointers (block_map, exception_table, temps,
 * static_method_stack). */
typedef struct PhxHirBuilderState {
    void *code;            /* PyCodeObject* (ctor, immutable) */
    const void *preloader; /* const Preloader& (ctor, immutable) */
    void *current_func;    /* Function* (mutable, nullable) */
    void *func;            /* Register* (mutable, nullable) */
    void *kwnames;         /* Register* (mutable, nullable) */
} PhxHirBuilderState;

/* Initialize state_ Class A fields from HIRBuilder ctor args. Mutable
 * fields (current_func, func, kwnames) are NULL-initialized matching
 * the existing C++ default-member-initialization. */
void hir_builder_state_init(
    PhxHirBuilderState *state,
    void *code,
    const void *preloader);

/* Pure-C port of HIRBuilder::parseExceptionTable. Reads
 * state->code's co_exceptiontable, decodes 3.12+ varint format, and
 * pushes ExceptionTableEntry records via the _cpp bridge below. */
void hir_builder_state_parse_exception_table_c(
    PhxHirBuilderState *state,
    void *builder);

/* C++-side bridge: pushes a single entry onto HIRBuilder.exception_table_
 * (a std::vector<ExceptionTableEntry>). 5-arg flat signature avoids
 * struct-on-stack ABI complexity (theologian 23:05:15Z (d)). 'end' is the
 * absolute byte offset after the try range (start + size * sizeof(_Py_CODEUNIT))
 * — pre-computed by the C body. lasti is 0 or 1 (decoded as bit). */
void hir_builder_state_exception_table_push_cpp(
    void *builder,
    int start,
    int end,
    int target,
    int depth,
    int lasti);

/* Phase 3 Batch 2: Class B read-side bridges for exception_table_.
 * Closes Class B-kept disposition (read+write both via bridge). */

/* Returns current vector size (number of entries). */
int hir_builder_state_exception_table_size_cpp(void *builder);

/* Reads entry fields by-value via flat ptr-args (matches push_cpp pattern).
 * idx must be < size returned by _size_cpp. */
void hir_builder_state_exception_table_entry_cpp(
    void *builder,
    int idx,
    int *out_start,
    int *out_end,
    int *out_target,
    int *out_depth,
    int *out_lasti);

/* Pure-C port of HIRBuilder::findExceptionHandler. Linear scan via
 * size+entry bridges; on match, writes index of first matching entry
 * to *out_idx and returns 1; else 0. C++ shim converts index → pointer
 * via &exception_table_[idx] preserving caller-contract. */
int hir_builder_state_find_exception_handler_c(
    PhxHirBuilderState *state,
    void *builder,
    int off,
    int *out_idx);

/* Phase 3 Batch 4 (X) Class B-kept disposition closure for block_map_:
 * lookup the BasicBlock* registered for a given BCOffset in
 * HIRBuilder.block_map_.blocks (a std::unordered_map<BCOffset,BasicBlock*>).
 * JIT_DCHECK panics on not-found, mirroring the existing C++
 * HIRBuilder::getBlockAtOff semantics. Returns BasicBlock* as void*. */
void *hir_builder_state_block_map_blocks_lookup_cpp(
    void *builder,
    int off);

/* Phase 3 Batch 5 (P-strict) Class B-kept disposition closure for
 * static_method_stack_ (jit::Stack<Register*>): pop the top entry and
 * return it. Renamed from hir_builder_static_method_stack_pop_c to
 * align with state-bridge _cpp suffix convention (Batch 2 + 4 precedent).
 * The push side stays C++-direct from C++ method context (1 site at
 * builder.cpp:3449); push_cpp bridge deferred per as-needed discipline
 * (theologian 00:28:34Z + supervisor 00:28:51Z). */
void *hir_builder_state_static_method_stack_pop_cpp(void *builder);

/* Phase 3 Batch 6 (R-single ATOMIC) Class B-kept disposition closure
 * for temps_ (TempAllocator): allocate a stack-temp Register from
 * HIRBuilder.temps_.AllocateStack(). Renamed from
 * hir_builder_temps_alloc_stack to align with state-bridge _cpp suffix
 * convention (Batch 2/4/5 precedent). 71 C-side callers in
 * builder_emit_c.c sed-renamed in lockstep. The other TempAllocator
 * methods (AllocateNonStack, GetOrAllocateStack) stay C++-direct from
 * C++ method context per as-needed discipline (zero C-side callers
 * verified pre-Step-A by generalist 00:51:54Z + theologian 00:53:06Z).
 *
 * Closes Phase 3 §5 forcing-decision validation: all 5 Class B members
 * disposed (4 closed via _cpp bridges, 1 dead-deleted). */
void *hir_builder_state_temps_alloc_stack_cpp(void *builder);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PHX_BUILDER_STATE_C_H */
