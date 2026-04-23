/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * PhxHirBuilderState — Phase 3 Batch 1 foundation per
 * docs/tier7-phase3-hirbuilder-state-extraction-spec.md §3.1.
 *
 * Class A members (5 immutable + nullable opaque pointers) extracted
 * from HIRBuilder. Class B members (block_map_, exception_table_, etc.)
 * remain C++-side and access via _cpp suffix bridges (push/clear/iterate)
 * until per-batch migration per spec §5 #1.
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
 * opaque pointers (block_map, exception_table, pending_b2_blocks,
 * temps, static_method_stack). */
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

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PHX_BUILDER_STATE_C_H */
