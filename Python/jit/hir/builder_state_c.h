/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * PhxHirBuilderState — Phase 3 Batch 1 foundation per
 * docs/tier7-phase3-hirbuilder-state-extraction-spec.md §3.1
 * + Tier 8 pilot Phase A per
 * docs/tier8-class-b-cport-migrate-arm-spec.md.
 *
 * Class A members (5 immutable + nullable opaque pointers) extracted
 * from HIRBuilder. Class B members migrated per-batch:
 *   exception_table_         → PhxExceptionTable (Tier 8 pilot 1 Phase A/B)
 *   block_map_.blocks        → PhxBlockMap        (Tier 8 pilot 2 Phase A)
 *   block_map_.bc_blocks     → PhxBcBlockArray    (Tier 8 pilot 2 Phase B)
 *   temps_, static_method_stack_  remain C++-side via _cpp bridges
 *                                 (Phase 3 Batch 5/6 closure).
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
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "cinderx/Common/jit_log_c.h"  /* JIT_DCHECK_C (W-I3 (III) sentinel) */
#include "cinderx/Jit/hir/hir_instr_c.h"  /* HirTempAllocator (Pilot 3 step 2) */

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

/* PhxBlockMap — open-addressed integer-keyed hash table mapping BCOffset
 * (int) → BasicBlock* (void*). Tier 8 SECOND-PILOT Phase A replacement
 * for the (now-deleted) std::unordered_map<BCOffset,BasicBlock*>
 * block_map_.blocks field. Custom hash chosen per spec §2.1 strict path
 * (theologian 10:25:08Z + supervisor 10:25:19Z): D2 measurement
 * (generalist 10:24:28Z) showed re._parser:Tokenizer.__next at 294
 * blocks; PhxArray linear-scan REJECTED, custom hash REQUIRED.
 *
 * Layout: power-of-2 capacity, linear probing, load factor 0.7 triggers
 * resize (double). Hash: Knuth multiplicative on uint32_t. Empty slot
 * sentinel: value==NULL. NULL-value invariant verified pre-commit
 * (generalist 10:31:03Z): the sole insert site (builder.cpp:1226) takes
 * value from CFG::AllocateBlock() = `new BasicBlock(id)` (cfg.h:139),
 * which never returns NULL. */
typedef struct PhxBlockMapEntry {
    int key;     /* BCOffset.value() */
    void *value; /* BasicBlock*; NULL = empty slot */
} PhxBlockMapEntry;

typedef struct PhxBlockMap {
    PhxBlockMapEntry *entries;
    size_t count;
    size_t capacity; /* power-of-2; 0 ⇒ data NULL (lazy init) */
} PhxBlockMap;

#define PHX_BLOCK_MAP_INITIAL_CAP 16u
#define PHX_BLOCK_MAP_LOAD_NUM 7u
#define PHX_BLOCK_MAP_LOAD_DEN 10u

static inline void phx_block_map_init(PhxBlockMap *m) {
    m->entries = NULL;
    m->count = 0;
    m->capacity = 0;
}

static inline void phx_block_map_destroy(PhxBlockMap *m) {
    if (m->entries) {
        free(m->entries);
        m->entries = NULL;
    }
    m->count = 0;
    m->capacity = 0;
}

static inline void phx_block_map_clear(PhxBlockMap *m) {
    if (m->entries && m->capacity) {
        for (size_t i = 0; i < m->capacity; i++) {
            m->entries[i].value = NULL;
        }
    }
    m->count = 0;
}

/* Knuth multiplicative hash, mask to power-of-2 capacity. */
static inline size_t phx_block_map_slot(size_t cap, int key) {
    uint32_t h = (uint32_t)key * 2654435761u;
    return (size_t)h & (cap - 1u);
}

/* Insert raw (assumes capacity > 0 and load not exceeded). */
static inline void phx_block_map_insert_raw(
        PhxBlockMap *m, int key, void *value) {
    size_t i = phx_block_map_slot(m->capacity, key);
    while (m->entries[i].value != NULL) {
        if (m->entries[i].key == key) {
            m->entries[i].value = value; /* overwrite, mirrors map[k]=v */
            return;
        }
        i = (i + 1u) & (m->capacity - 1u);
    }
    m->entries[i].key = key;
    m->entries[i].value = value;
    m->count++;
}

static inline void phx_block_map_resize(PhxBlockMap *m, size_t new_cap) {
    PhxBlockMapEntry *old_entries = m->entries;
    size_t old_cap = m->capacity;
    PhxBlockMapEntry *new_entries =
        (PhxBlockMapEntry*)calloc(new_cap, sizeof(PhxBlockMapEntry));
    m->entries = new_entries;
    m->capacity = new_cap;
    m->count = 0;
    if (old_entries) {
        for (size_t i = 0; i < old_cap; i++) {
            if (old_entries[i].value != NULL) {
                phx_block_map_insert_raw(
                    m, old_entries[i].key, old_entries[i].value);
            }
        }
        free(old_entries);
    }
}

static inline void phx_block_map_insert(
        PhxBlockMap *m, int key, void *value) {
    if (m->capacity == 0) {
        phx_block_map_resize(m, PHX_BLOCK_MAP_INITIAL_CAP);
    } else if ((m->count + 1u) * PHX_BLOCK_MAP_LOAD_DEN
               > m->capacity * PHX_BLOCK_MAP_LOAD_NUM) {
        phx_block_map_resize(m, m->capacity * 2u);
    }
    phx_block_map_insert_raw(m, key, value);
}

/* Lookup; panics via JIT_CHECK_C on miss. Mirrors the JIT_DCHECK
 * semantics of the deleted C++ HIRBuilder::getBlockAtOff. */
void *phx_block_map_lookup_or_panic(const PhxBlockMap *m, int key);

/* Lookup; returns NULL if not found (caller checks). */
static inline void *phx_block_map_lookup(const PhxBlockMap *m, int key) {
    if (m->capacity == 0) {
        return NULL;
    }
    size_t i = phx_block_map_slot(m->capacity, key);
    while (m->entries[i].value != NULL) {
        if (m->entries[i].key == key) {
            return m->entries[i].value;
        }
        i = (i + 1u) & (m->capacity - 1u);
    }
    return NULL;
}

/* PhxBcBlockEntry — POD mirror of BytecodeInstructionBlock {start, end}
 * fields, sufficient to reconstruct one (the third constructor arg
 * `code` is constant per-compile and lives on PhxHirBuilderState.code).
 * Tier 8 SECOND-PILOT Phase B (theologian 11:13:21Z + supervisor
 * 11:06:31Z): replaces the (now-deleted) std::unordered_map<BasicBlock*,
 * BytecodeInstructionBlock> bc_blocks field via a dense array indexed
 * by BasicBlock::id, exploiting the production-validated invariant that
 * BasicBlock ids are allocation-monotonic from 0 (cfg.h:139-144). */
typedef struct PhxBcBlockEntry {
    int start; /* BCIndex.value() */
    int end;   /* BCIndex.value() */
#ifdef Py_DEBUG
    /* W-I3-RUNTIME-ASSERT (III) per docs/w-i3-runtime-assert-spec.md §2:
     * pydebug-only sentinel storing the block_id this entry was inserted
     * for. Verified at lookup; mismatch => I3 invariant violation
     * (BasicBlock::id mutated post-allocation, OR inserter/reader
     * disagree on indexing). Zero release-build cost. */
    int sentinel_id;
#endif
} PhxBcBlockEntry;

typedef struct PhxBcBlockArray {
    PhxBcBlockEntry *data;
    size_t count;    /* high-water-mark id+1 ever inserted (== max id +1) */
    size_t capacity; /* allocated slots */
} PhxBcBlockArray;

#define PHX_BC_BLOCK_ARRAY_INITIAL_CAP 16u

static inline void phx_bc_block_array_init(PhxBcBlockArray *a) {
    a->data = NULL;
    a->count = 0;
    a->capacity = 0;
}

static inline void phx_bc_block_array_destroy(PhxBcBlockArray *a) {
    if (a->data) {
        free(a->data);
        a->data = NULL;
    }
    a->count = 0;
    a->capacity = 0;
}

static inline void phx_bc_block_array_clear(PhxBcBlockArray *a) {
    a->count = 0;
    /* Retain capacity for cheap re-fill on inlined-callee re-createBlocks. */
}

/* Insert at array[block_id] = {start, end}. Lazily grows capacity to
 * cover block_id in a SINGLE realloc (theologian 11:19:42Z verification:
 * use max(old*2, needed) instead of a doubling loop, so block_id=1000
 * with old_cap=16 is one realloc not seven). Intermediate slots between
 * old count and block_id are zero-filled (start=0, end=0 sentinel —
 * never read because callers only look up block ids actually allocated
 * by createBlocks; I2 invariant). */
static inline void phx_bc_block_array_insert(
        PhxBcBlockArray *a, int block_id, int start, int end) {
    size_t needed = (size_t)block_id + 1u;
    if (needed > a->capacity) {
        size_t doubled = a->capacity ? a->capacity * 2u : PHX_BC_BLOCK_ARRAY_INITIAL_CAP;
        size_t new_cap = doubled > needed ? doubled : needed;
        size_t old_cap = a->capacity;
        PhxBcBlockEntry *new_data = (PhxBcBlockEntry*)realloc(
            a->data, new_cap * sizeof(PhxBcBlockEntry));
        memset(new_data + old_cap, 0,
               (new_cap - old_cap) * sizeof(PhxBcBlockEntry));
        a->data = new_data;
        a->capacity = new_cap;
    }
    a->data[block_id].start = start;
    a->data[block_id].end = end;
#ifdef Py_DEBUG
    a->data[block_id].sentinel_id = block_id;  /* W-I3 (III) */
#endif
    if (needed > a->count) {
        a->count = needed;
    }
}

/* Read array[block_id]. Caller must guarantee block_id < a->count
 * (invariant I2: never look up an id beyond createBlocks high-water).
 * Returns by value; entry is just 8 bytes (release) / 12 bytes (pydebug
 * with W-I3 (III) sentinel field). */
static inline PhxBcBlockEntry phx_bc_block_array_at(
        const PhxBcBlockArray *a, int block_id) {
#ifdef Py_DEBUG
    JIT_DCHECK_C(
        a->data[block_id].sentinel_id == block_id,
        "W-I3 invariant violation: bc_block_array[%d].sentinel_id=%d "
        "(BasicBlock::id mutated post-allocation, OR inserter/reader "
        "disagree on indexing)",
        block_id, a->data[block_id].sentinel_id);
#endif
    return a->data[block_id];
}

/* PhxHirBuilderState — opaque holder for HIRBuilder Class A state +
 * Tier 8-migrated Class B containers (currently exception_table_phx
 * post pilot 1, block_map_phx post pilot 2; remaining 2 Class B
 * containers stay C++-side). code + preloader are immutable post-ctor;
 * current_func/func/kwnames mutate during translate()/emit. */
typedef struct PhxHirBuilderState {
    void *code;            /* PyCodeObject* (ctor, immutable) */
    const void *preloader; /* const Preloader& (ctor, immutable) */
    void *current_func;    /* Function* (mutable, nullable) */
    void *func;            /* Register* (mutable, nullable) */
    void *kwnames;         /* Register* (mutable, nullable) */
    PhxExceptionTable exception_table_phx; /* Tier 8 pilot 1 (Phase A) */
    PhxBlockMap block_map_phx;             /* Tier 8 pilot 2 (Phase A) */
    PhxBcBlockArray bc_block_array_phx;    /* Tier 8 pilot 2 (Phase B) */
    HirTempAllocator temps_phx;            /* Phase 4.C Pilot 3 step 2 */
    HirOperandStack op_stack_phx;          /* Phase 4.C Pilot 4 step 1 */
} PhxHirBuilderState;

/* Phase 4.D pilot step 1 (Batch 53): PhxTranslationContext C struct.
 * POD layout matches the C++ TranslationContext (builder.cpp:344+).
 * Field order pinned by static_asserts in builder.cpp:1011-1014.
 * Used by hir_c_tc_* primitives so emit-class methods become 1-line
 * shims around C-side dispatch. */
typedef struct PhxTranslationContext {
    void *block;                  /* BasicBlock* — first field, offset 0 */
    HirFrameStateLayout frame;    /* sizeof(FrameState) opaque body */
} PhxTranslationContext;

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

/* Tier 8 SECOND-PILOT (block_map_) Phase A canonical state accessor.
 * Returns the PhxHirBuilderState owned by the C++ HIRBuilder identified
 * by the opaque builder handle. C-side callers obtain &state directly
 * without indirecting through type-erased per-field bridges; replaces
 * the deleted hir_builder_state_block_map_blocks_lookup_cpp + the
 * deleted hir_builder_get_block_at_off (net bridge delta = -1 per Tier
 * 8 spec §5 #11). Pattern reused by future Phase B/C/D pilots when
 * additional fields migrate to PhxHirBuilderState; when HIRBuilder
 * itself becomes pure-C the accessor body becomes a struct field
 * read. */
PhxHirBuilderState *phx_hir_builder_state(void *builder);

/* Phase 4.C Pilot 4 step 3 (Batch 50) CLOSURE: static_method_stack_
 * bridges (push_cpp + pop_cpp) fully retired. All callers in
 * builder_emit_c.c (B49) + bridge bodies in builder.cpp + decls here
 * + friend decls in builder.h + script falsifier entry retired.
 * Storage = PhxHirBuilderState.op_stack_phx (single source of truth
 * since B48 mirror-collapse). */

/* Phase 4.C Pilot 3 step 6 (Batch 47) CLOSURE: temps_ bridge
 * fully retired. All 71 C-side callers in builder_emit_c.c migrated
 * to direct hir_c_temps_alloc_stack(&phx_hir_builder_state(builder)
 * ->temps_phx) over B45-B47 (24+24+23). Bridge function deleted from
 * builder.cpp + decl deleted here + friend decl deleted in builder.h
 * + scripts/w45_bridge_drift_falsifier.sh:101 entry removed.
 *
 * Class B-kept disposition closure: all 5 Class B members now resolved
 * (4 closed via _cpp bridges + this 1 elevated to direct C-state-struct
 * access). Pilot 3 of Phase 4.C complete. */

/* Phase 4.D D5a: NEW C entries for top-level HIR build orchestration.
 *
 * phx_preloader_make_function — bridges Preloader::makeFunction()
 * (returns Function* released from std::unique_ptr; transfers ownership
 * to caller — caller must wrap in unique_ptr<Function> or delete).
 *
 * phx_hir_build — top-level entry mirroring hir::buildHIR() free
 * function. Constructs HIRBuilder + calls instance buildHIR(); returns
 * Function* released from std::unique_ptr (transfers ownership).
 *
 * BOTH entries co-exist with existing C++ entries this batch — D5a is
 * NO-OP behaviorally. compiler.cpp:236 still calls hir::buildHIR(); D5c
 * will rewire to phx_hir_build.
 *
 * preloader_handle = const Preloader* cast to const void* for C boundary.
 */
void *phx_preloader_make_function(const void *preloader_handle);
void *phx_hir_build(const void *preloader_handle);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PHX_BUILDER_STATE_C_H */
