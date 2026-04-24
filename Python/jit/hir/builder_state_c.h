/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * PhxHirBuilderState — Phase 3 Batch 1 foundation per
 * docs/tier7-phase3-hirbuilder-state-extraction-spec.md §3.1
 * + Tier 8 pilot Phase A per
 * docs/tier8-class-b-cport-migrate-arm-spec.md.
 *
 * Class A members (5 immutable + nullable opaque pointers) extracted
 * from HIRBuilder. Class B members migrated per-batch:
 *   exception_table_  → PhxExceptionTable (Tier 8 pilot Phase A)
 *   block_map_, temps_, static_method_stack_  remain C++-side via _cpp
 *   bridges (Phase 3 Batch 4/5/6 closure).
 *   pending_b2_blocks_  dead-state-deleted Phase 3 Batch 3.
 *
 * Authorization: theologian 23:05:15Z + supervisor 23:02:34Z (Y) atomic
 * + theologian 01:17:48Z + supervisor 01:18:35Z + supervisor 03:44:19Z
 * (Tier 8 Phase A patch-apply post Test #6 BREAKTHROUGH).
 */

#ifndef PHX_BUILDER_STATE_C_H
#define PHX_BUILDER_STATE_C_H

#include "Python.h"
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ExceptionTableEntry — POD mirror of the (now-deleted) C++
 * HIRBuilder::ExceptionTableEntry struct. Tier 8 pilot Phase A
 * moves the struct to C-side ownership; field types flatten BCOffset →
 * int and bool → unsigned char. C++ readers wrap as `BCOffset{entry.start}`
 * etc. when constructing typed values. */
typedef struct ExceptionTableEntry {
    int start;             /* byte offset, inclusive */
    int end;               /* byte offset, exclusive */
    int target;            /* handler entry byte offset */
    int depth;             /* stack depth at handler entry */
    unsigned char lasti;   /* whether to push lasti */
} ExceptionTableEntry;

/* PhxExceptionTable — typed-inline dynamic array replacing the
 * (now-deleted) std::vector<ExceptionTableEntry> exception_table_
 * field. Tier 8 pilot Phase A purpose-built minimal C container per
 * theologian 01:12:09Z + supervisor 01:12:26Z (B) typed-inline.
 *
 * Lazy-init: data starts NULL + count=0 + capacity=0; first push
 * mallocs (capacity_initial = 4, doubling). Cleanup via
 * phx_exception_table_destroy at HIRBuilder dtor. */
typedef struct PhxExceptionTable {
    ExceptionTableEntry *data;
    size_t count;
    size_t capacity;
} PhxExceptionTable;

static inline void phx_exception_table_init(PhxExceptionTable *t) {
    t->data = NULL;
    t->count = 0;
    t->capacity = 0;
}

static inline void phx_exception_table_destroy(PhxExceptionTable *t) {
    if (t->data) {
        free(t->data);
        t->data = NULL;
    }
    t->count = 0;
    t->capacity = 0;
}

static inline void phx_exception_table_push(
        PhxExceptionTable *t, const ExceptionTableEntry *e) {
    if (t->count == t->capacity) {
        size_t new_cap = t->capacity ? t->capacity * 2 : 4;
        t->data = (ExceptionTableEntry*)realloc(
            t->data, new_cap * sizeof(ExceptionTableEntry));
        t->capacity = new_cap;
    }
    t->data[t->count++] = *e;
}

static inline size_t phx_exception_table_size(const PhxExceptionTable *t) {
    return t->count;
}

static inline const ExceptionTableEntry *phx_exception_table_at(
        const PhxExceptionTable *t, size_t idx) {
    return &t->data[idx];
}

static inline void phx_exception_table_clear(PhxExceptionTable *t) {
    t->count = 0;
    /* Retain capacity for cheap re-fill in inlined-callee suppression. */
}

/* PhxHirBuilderState — opaque holder for HIRBuilder Class A state +
 * Tier 8-migrated Class B containers (currently exception_table_phx
 * post Phase A pilot; remaining 3 Class B containers stay C++-side).
 * code + preloader are immutable post-ctor; current_func/func/kwnames
 * mutate during translate()/emit. */
typedef struct PhxHirBuilderState {
    void *code;            /* PyCodeObject* (ctor, immutable) */
    const void *preloader; /* const Preloader& (ctor, immutable) */
    void *current_func;    /* Function* (mutable, nullable) */
    void *func;            /* Register* (mutable, nullable) */
    void *kwnames;         /* Register* (mutable, nullable) */
    PhxExceptionTable exception_table_phx; /* Tier 8 pilot Phase A */
} PhxHirBuilderState;

/* Initialize state_ Class A fields from HIRBuilder ctor args. Mutable
 * fields (current_func, func, kwnames) are NULL-initialized matching
 * the existing C++ default-member-initialization. Tier 8 pilot Phase A
 * also initializes exception_table_phx (lazy data alloc). */
void hir_builder_state_init(
    PhxHirBuilderState *state,
    void *code,
    const void *preloader);

/* HIRBuilder dtor cleanup: free PhxExceptionTable.data malloc.
 * Called from C++ HIRBuilder destructor (Tier 8 pilot Phase A). */
void hir_builder_state_destroy(PhxHirBuilderState *state);

/* Pure-C port of HIRBuilder::parseExceptionTable. Tier 8 pilot Phase A:
 * reads state->code's co_exceptiontable, decodes 3.12+ varint format,
 * and pushes ExceptionTableEntry records into
 * state->exception_table_phx via phx_exception_table_push (replaces
 * deleted Phase 3 Batch 1 push_cpp bridge). */
void hir_builder_state_parse_exception_table_c(
    PhxHirBuilderState *state,
    void *builder);

/* Pure-C port of HIRBuilder::findExceptionHandler. Tier 8 pilot Phase A:
 * linear scan via PhxExceptionTable directly (replaces Phase 3 Batch 2
 * size_cpp/entry_cpp bridges). On match, writes matching entry index
 * to *out_idx and returns 1; else 0. C++ shim (transient compatibility
 * layer per Phase A) converts index → pointer via phx_exception_table_at
 * preserving caller-contract; Phase B will delete the shim and rewire
 * callers to use this body directly. */
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
