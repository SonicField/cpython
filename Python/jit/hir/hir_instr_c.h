/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C-compatible struct definitions for HIR instruction types.
 * T2-B: Layout-compatible with C++ Instr, DeoptBase, etc.
 *
 * Uses field-flattening macros instead of struct embedding to avoid
 * C/C++ tail padding reuse mismatch. C struct embedding pads inner
 * structs to alignment boundary; C++ inheritance reuses tail padding.
 * Macros inline the fields directly, matching C++ layout exactly.
 */
#pragma once

#include "cinderx/Jit/hir/hir_instr_info_c.h"
#include "cinderx/Jit/hir/hir_type_c.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
/* PhxPtrArray must be available before frame_state.h (circular include). */
#include "cinderx/Jit/hir/phx_ptr_array.h"

#ifdef __cplusplus
/* H2-E1+E2: no more C++ containers in DeoptBase.
 * Only frame_state.h needed for visitUses C++ bridge. */
#include "cinderx/Jit/hir/frame_state.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ---- IntrusiveListNode ---- */
typedef struct HirListNode {
    struct HirListNode *prev;
    struct HirListNode *next;
} HirListNode;

/* ---- Edge ---- */
typedef struct HirEdge {
    void *from;
    void *to;
} HirEdge;

/* ---- Edge pointer array (replaces std::unordered_set<const Edge*>) ----
 * Used by BasicBlock for in_edges_ and out_edges_.
 * Small counts (1-4 typical), linear scan is faster than hash set. */
typedef struct PhxEdgePtrArray {
    const HirEdge **data;
    size_t count;
    size_t capacity;
} PhxEdgePtrArray;

static inline void phx_edge_arr_init(PhxEdgePtrArray *a) {
    a->data = NULL;
    a->count = 0;
    a->capacity = 0;
}

static inline void phx_edge_arr_destroy(PhxEdgePtrArray *a) {
    free(a->data);
    a->data = NULL;
    a->count = 0;
    a->capacity = 0;
}

static inline void phx_edge_arr_insert(PhxEdgePtrArray *a, const HirEdge *e) {
    if (a->count == a->capacity) {
        size_t new_cap = a->capacity ? a->capacity * 2 : 4;
        a->data = (const HirEdge **)realloc(a->data, new_cap * sizeof(const HirEdge *));
        a->capacity = new_cap;
    }
    a->data[a->count++] = e;
}

static inline void phx_edge_arr_erase(PhxEdgePtrArray *a, const HirEdge *e) {
    for (size_t i = 0; i < a->count; i++) {
        if (a->data[i] == e) {
            /* Swap with last element (order doesn't matter). */
            a->data[i] = a->data[a->count - 1];
            a->count--;
            return;
        }
    }
}

static inline int phx_edge_arr_empty(const PhxEdgePtrArray *a) {
    return a->count == 0;
}

/* PhxPtrArray type + functions now in phx_ptr_array.h (included above). */

/* ---- OperandType (C equivalent of hir::OperandType) ---- */
typedef enum {
    HIR_CONSTRAINT_kType = 0,
    HIR_CONSTRAINT_kMatchAllAsCInt,
    HIR_CONSTRAINT_kMatchAllAsPrimitive,
    HIR_CONSTRAINT_kTupleExactOrCPtr,
    HIR_CONSTRAINT_kListOrChkList,
    HIR_CONSTRAINT_kDictOrChkDict,
    HIR_CONSTRAINT_kOptObjectOrCInt,
    HIR_CONSTRAINT_kOptObjectOrCIntOrCBool,
} HirConstraint;

typedef struct {
    HirConstraint kind;
    HirType type;
} HirOperandType;

/* ---- Field-flattening macros ----
 * These inline base class fields into derived structs, avoiding
 * struct embedding and its tail padding problems. */

#define HIR_INSTR_FIELDS                                    \
    HirListNode block_node;     /* intrusive list node */   \
    int32_t opcode;             /* enum Opcode */           \
    int32_t bytecode_offset;    /* BCOffset */              \
    void *output;               /* Register* */             \
    void *block                 /* BasicBlock* */

#define HIR_DEOPT_FIELDS                                    \
    HIR_INSTR_FIELDS;                                       \
    void *live_regs_data; size_t live_regs_count; size_t live_regs_cap; /* PhxRegStateArray */ \
    void *frame_state;          /* unique_ptr<FrameState> */\
    void *guilty_reg;           /* Register* */             \
    int32_t nonce;                                          \
    int32_t _deopt_pad0;                                    \
    char *descr;                /* H2-E1: was std::string */\
    uint8_t suppress_exception_deopt

/* DeoptBaseWithNameIdx: DeoptBase + int name_idx_ */
#define HIR_DEOPT_NAMEIDX_FIELDS  \
    HIR_DEOPT_FIELDS;             \
    int32_t name_idx

/* CheckBaseWithName: CheckBase (= DeoptBase) + BorrowedRef<> name_ */
#define HIR_CHECK_WITH_NAME_FIELDS  \
    HIR_DEOPT_FIELDS;               \
    void *name

/* LoadSuperBase: DeoptBaseWithNameIdx + bool no_args_in_super_call_ */
#define HIR_LOAD_SUPER_FIELDS       \
    HIR_DEOPT_NAMEIDX_FIELDS;       \
    uint8_t no_args_in_super_call

/* ---- Register C struct ---- */
typedef struct HirRegisterLayout {
    HirType type;      /* 16 bytes — matches Type binary layout */
    void *instr;       /* 8 bytes — defining Instr* */
    int id;            /* 4 bytes */
    int _pad0;         /* 4 bytes — alignment */
    char *name;        /* 8 bytes — malloc'd lazy name */
} HirRegisterLayout;

/* ---- Environment C struct (must precede accessors) ---- */
typedef struct HirEnvironment {
    void **reg_data;               /* Register** flat array */
    size_t reg_count;              /* number of slots (may have NULL gaps) */
    size_t reg_capacity;           /* allocated capacity */
    char references_opaque[56];    /* opaque: std::unordered_set<ThreadedRef<>> */
    int next_register_id;
    int next_load_type_attr_cache;
    int next_load_type_method_cache;
    int _env_pad0;
} HirEnvironment;

/* ---- Environment C accessors ---- */

static inline void *hir_env_get_register(const void *env, int id) {
    const HirEnvironment *e = (const HirEnvironment *)env;
    if (id < 0 || (size_t)id >= e->reg_count) return NULL;
    return e->reg_data[id];
}

static inline size_t hir_env_reg_count(const void *env) {
    return ((const HirEnvironment *)env)->reg_count;
}

static inline void **hir_env_reg_data(const void *env) {
    return (void **)((const HirEnvironment *)env)->reg_data;
}

static inline int hir_env_next_register_id(const void *env) {
    return ((const HirEnvironment *)env)->next_register_id;
}

static inline int hir_env_num_load_type_attr_caches(const void *env) {
    return ((const HirEnvironment *)env)->next_load_type_attr_cache;
}

static inline int hir_env_num_load_type_method_caches(const void *env) {
    return ((const HirEnvironment *)env)->next_load_type_method_cache;
}

/* ---- FrameState C struct ---- */
typedef struct HirFrameStateLayout {
    ssize_t cur_instr_offs;           /* 8 bytes — BCOffset */
    PhxPtrArray localsplus;           /* 24 bytes */
    int nlocals;                      /* 4 bytes */
    int _fs_pad0;                     /* 4 bytes */
    PhxPtrArray stack;                /* 24 bytes */
    void *block_stack_data;           /* 24 bytes — PhxExecBlockArray */
    size_t block_stack_count;
    size_t block_stack_cap;
    void *code;                       /* 8 bytes — PyCodeObject* */
    void *globals;                    /* 8 bytes — PyDictObject* */
    void *builtins;                   /* 8 bytes — PyDictObject* */
    struct HirFrameStateLayout *parent; /* 8 bytes */
} HirFrameStateLayout;

/* HirEnvironment struct defined above (before accessors) */

/* ---- FrameState C accessors ---- */

static inline ssize_t hir_fs_cur_instr_offs(const void *fs) {
    return ((const HirFrameStateLayout *)fs)->cur_instr_offs;
}

static inline int hir_fs_nlocals(const void *fs) {
    return ((const HirFrameStateLayout *)fs)->nlocals;
}

static inline void *hir_fs_code(const void *fs) {
    return ((const HirFrameStateLayout *)fs)->code;
}

static inline void *hir_fs_globals(const void *fs) {
    return ((const HirFrameStateLayout *)fs)->globals;
}

static inline void *hir_fs_builtins(const void *fs) {
    return ((const HirFrameStateLayout *)fs)->builtins;
}

static inline void *hir_fs_parent(const void *fs) {
    return ((const HirFrameStateLayout *)fs)->parent;
}

static inline size_t hir_fs_stack_size(const void *fs) {
    return ((const HirFrameStateLayout *)fs)->stack.count;
}

static inline void *hir_fs_stack_at(const void *fs, size_t i) {
    return ((const HirFrameStateLayout *)fs)->stack.data[i];
}

static inline size_t hir_fs_localsplus_count(const void *fs) {
    return ((const HirFrameStateLayout *)fs)->localsplus.count;
}

static inline void *hir_fs_localsplus_at(const void *fs, size_t i) {
    return ((const HirFrameStateLayout *)fs)->localsplus.data[i];
}

static inline size_t hir_fs_inline_depth(const void *fs) {
    int depth = -1;
    const HirFrameStateLayout *f = (const HirFrameStateLayout *)fs;
    while (f) { depth++; f = f->parent; }
    return (size_t)depth;
}

/* ---- Register C accessors ---- */

static inline HirType hir_reg_type(const void *reg) {
    return ((const HirRegisterLayout *)reg)->type;
}

static inline void hir_reg_set_type(void *reg, HirType type) {
    ((HirRegisterLayout *)reg)->type = type;
}

static inline int hir_reg_id(const void *reg) {
    return ((const HirRegisterLayout *)reg)->id;
}

static inline void *hir_reg_instr_ptr(const void *reg) {
    return ((const HirRegisterLayout *)reg)->instr;
}

static inline void hir_reg_set_instr(void *reg, void *instr) {
    ((HirRegisterLayout *)reg)->instr = instr;
}

static inline const char *hir_reg_name(const void *reg) {
    HirRegisterLayout *r = (HirRegisterLayout *)reg;
    if (!r->name) {
        r->name = (char *)malloc(32);
        snprintf(r->name, 32, "v%d", r->id);
    }
    return r->name;
}

/* ---- Base structs (standalone use) ----
 * Named HirInstrLayout / HirDeoptLayout to avoid collision with
 * the void* HirInstr typedef in hir_c_api.h.  C passes include
 * both headers; the void* handle is cast to these layout structs
 * inside the static-inline accessors below. */

typedef struct HirInstrLayout {
    HIR_INSTR_FIELDS;
} HirInstrLayout;

typedef struct HirDeoptLayout {
    HIR_DEOPT_FIELDS;
} HirDeoptLayout;

typedef struct HirCondBranchInstr {
    HIR_INSTR_FIELDS;
    HirEdge true_edge;
    HirEdge false_edge;
} HirCondBranchInstr;

/* ==== T2-B Batch 2: Simple custom-field instruction structs ====
 * All add a single int32_t op_ field after their base fields. */

/* ---- Operation-kind types (DeoptBase + enum) ---- */
typedef struct { HIR_DEOPT_FIELDS; int32_t op; } HirBinaryOp;
typedef struct { HIR_DEOPT_FIELDS; int32_t op; } HirUnaryOp;
typedef struct { HIR_DEOPT_FIELDS; int32_t op; } HirInPlaceOp;
typedef struct { HIR_DEOPT_FIELDS; int32_t op; } HirLongBinaryOp;
typedef struct { HIR_DEOPT_FIELDS; int32_t op; } HirLongInPlaceOp;
typedef struct { HIR_DEOPT_FIELDS; int32_t op; } HirFloatBinaryOp;

/* ---- Operation-kind types (Instr + enum) ---- */
typedef struct { HIR_INSTR_FIELDS; int32_t op; } HirIntBinaryOp;
typedef struct { HIR_INSTR_FIELDS; int32_t op; } HirDoubleBinaryOp;
typedef struct { HIR_INSTR_FIELDS; int32_t op; } HirPrimitiveUnaryOp;

/* ---- Comparison types (DeoptBase + enum) ---- */
typedef struct { HIR_DEOPT_FIELDS; int32_t op; } HirCompare;
typedef struct { HIR_DEOPT_FIELDS; int32_t op; } HirCompareBool;

/* ---- Comparison types (Instr + enum) ---- */
typedef struct { HIR_INSTR_FIELDS; int32_t op; } HirFloatCompare;
typedef struct { HIR_INSTR_FIELDS; int32_t op; } HirLongCompare;
typedef struct { HIR_INSTR_FIELDS; int32_t op; } HirUnicodeCompare;
typedef struct { HIR_INSTR_FIELDS; int32_t op; } HirPrimitiveCompare;

/* ==== T2-B Batch 3: Type-field instruction structs ====
 * All add one HirType (16 bytes) field after base fields. */

/* ---- Type-field types (Instr + HirType) ---- */
typedef struct { HIR_INSTR_FIELDS; HirType type; } HirLoadConst;
typedef struct { HIR_INSTR_FIELDS; HirType type; } HirRefineType;
typedef struct { HIR_INSTR_FIELDS; HirType type; } HirBitCast;
typedef struct { HIR_INSTR_FIELDS; HirType type; } HirReturn;
typedef struct { HIR_INSTR_FIELDS; HirType type; } HirUseType;
typedef struct { HIR_INSTR_FIELDS; HirType type; } HirIntConvert;
typedef struct { HIR_INSTR_FIELDS; HirType type; } HirPrimitiveUnbox;
typedef struct { HIR_INSTR_FIELDS; HirType type; } HirGetSecondOutput;
typedef struct { HIR_INSTR_FIELDS; HirType type; } HirStoreArrayItem;
typedef struct { HIR_INSTR_FIELDS; intptr_t offset; HirType type; } HirLoadArrayItem;

/* ---- Type-field types (DeoptBase + HirType) ---- */
typedef struct { HIR_DEOPT_FIELDS; HirType type; } HirPrimitiveBox;

/* ==== T2-B Batch 4a: Scalar-field instruction structs ==== */

/* ---- Single scalar field (Instr base) ---- */
typedef struct { HIR_INSTR_FIELDS; HirEdge edge; } HirBranch;
typedef struct { HIR_INSTR_FIELDS; int32_t field; } HirSetFunctionAttr;
typedef struct { HIR_INSTR_FIELDS; size_t index; } HirCallIntrinsic;
typedef struct { HIR_INSTR_FIELDS; void *addr; } HirCallStaticRetVoid;
typedef struct { HIR_INSTR_FIELDS; void *exc; } HirIndexUnbox;
typedef struct { HIR_INSTR_FIELDS; size_t idx; } HirLoadTupleItem;
typedef struct { HIR_INSTR_FIELDS; intptr_t item_idx; } HirLoadSplitDictItem;
typedef struct { HIR_INSTR_FIELDS; int32_t cells; } HirInitFrameCellVars;
typedef struct { HIR_INSTR_FIELDS; int32_t cache_id; } HirLoadTypeAttrCacheEntryType;
typedef struct { HIR_INSTR_FIELDS; int32_t cache_id; } HirLoadTypeAttrCacheEntryValue;
typedef struct { HIR_INSTR_FIELDS; int32_t cache_id; } HirLoadTypeMethodCacheEntryType;
typedef struct { HIR_INSTR_FIELDS; int32_t cache_id; } HirLoadTypeMethodCacheEntryValue;

/* ---- Multi-field (Instr base) ---- */
typedef struct { HIR_INSTR_FIELDS; void *addr; HirType ret_type; } HirCallStatic;
/* inline_depth defaults to -1 in C++ (hir.h:1731).
 * Use hir_c_init_end_inlined() for any C factory — enforces the invariant. */
typedef struct { HIR_INSTR_FIELDS; void *begin; int32_t inline_depth; } HirEndInlinedFunction;
typedef struct { HIR_INSTR_FIELDS; int32_t line_no; int32_t _pad; void *parent; } HirUpdatePrevInstr;
typedef struct { HIR_INSTR_FIELDS; uint32_t arg_idx; int32_t _pad; HirType type; } HirLoadArg;

/* ---- Single scalar field (DeoptBase base) ---- */
typedef struct { HIR_DEOPT_FIELDS; uint32_t flags; } HirVectorCall;
typedef struct { HIR_DEOPT_FIELDS; uint32_t flags; } HirCallEx;
typedef struct { HIR_DEOPT_FIELDS; void *pytype; } HirTpAlloc;
typedef struct { HIR_DEOPT_FIELDS; void *target; } HirGuardIs;
typedef struct { HIR_DEOPT_FIELDS; HirType target; } HirGuardType;
typedef struct { HIR_DEOPT_FIELDS; size_t capacity; } HirMakeDict;
typedef struct { HIR_DEOPT_FIELDS; void *patcher; } HirDeoptPatchpoint;
typedef struct { HIR_DEOPT_FIELDS; uint8_t is_aenter; } HirRaiseAwaitableError;
typedef struct { HIR_DEOPT_FIELDS; int32_t conversion; } HirBuildInterpolation;
typedef struct { HIR_DEOPT_FIELDS; int32_t converter_idx; } HirConvertValue;
typedef struct { HIR_DEOPT_FIELDS; int32_t special_idx; } HirLoadSpecial;

/* ---- Multi-field (DeoptBase base) ---- */
typedef struct { HIR_DEOPT_FIELDS; void *pytype; uint8_t optional; uint8_t exact; } HirCast;
typedef struct { HIR_DEOPT_FIELDS; void *funcptr; void *func_descr; } HirLoadFunctionIndirect;
typedef struct { HIR_DEOPT_FIELDS; const char *fmt; void *exc_type; } HirRaiseStatic;
typedef struct { HIR_DEOPT_FIELDS; const char *name; HirType ret_type; } HirCallInd;
typedef struct { HIR_DEOPT_FIELDS; size_t capacity; HirType type; } HirMakeCheckedDict;
typedef struct { HIR_DEOPT_FIELDS; HirType type; } HirMakeCheckedList;

/* ---- CondBranch derived ---- */
typedef struct { HIR_INSTR_FIELDS; HirEdge true_edge; HirEdge false_edge; HirType type; } HirCondBranchCheckType;

/* ==== H2-A: Missing DEFINE_SIMPLE_INSTR C structs ====
 * No custom fields — just base class layout. */

/* ---- Instr-base (no output, no DeoptBase) ---- */
typedef struct { HIR_INSTR_FIELDS; } HirAssign;
typedef struct { HIR_INSTR_FIELDS; } HirUnreachable;

/* ---- DeoptBase (no custom fields) ---- */
typedef struct { HIR_INSTR_FIELDS; } HirDecref;
typedef struct { HIR_INSTR_FIELDS; } HirXDecref;
typedef struct { HIR_INSTR_FIELDS; } HirIncref;
typedef struct { HIR_INSTR_FIELDS; } HirXIncref;
typedef struct { HIR_DEOPT_FIELDS; } HirDeopt;
typedef struct { HIR_INSTR_FIELDS; } HirAtQuiescentState;
typedef struct { HIR_DEOPT_FIELDS; } HirRunPeriodicTasks;
typedef struct { HIR_DEOPT_FIELDS; } HirIsTruthy;
typedef struct { HIR_INSTR_FIELDS; } HirLoadCellItem;
typedef struct { HIR_INSTR_FIELDS; } HirLoadCurrentFunc;
typedef struct { HIR_INSTR_FIELDS; } HirLoadEvalBreaker;
typedef struct { HIR_INSTR_FIELDS; } HirLoadFrame;
typedef struct { HIR_INSTR_FIELDS; } HirLoadVarObjectSize;
typedef struct { HIR_DEOPT_FIELDS; } HirCheckErrOccurred;
typedef struct { HIR_DEOPT_NAMEIDX_FIELDS; } HirDeleteAttr;
typedef struct { HIR_DEOPT_FIELDS; } HirDeleteSubscr;
typedef struct { HIR_DEOPT_FIELDS; } HirRaise;
typedef struct { HIR_DEOPT_FIELDS; } HirMakeSet;
typedef struct { HIR_INSTR_FIELDS; } HirStealCellItem;
typedef struct { HIR_INSTR_FIELDS; } HirSwapCellItem;
typedef struct { HIR_INSTR_FIELDS; } HirSetCellItem;
typedef struct { HIR_INSTR_FIELDS; } HirSetCurrentAwaiter;
typedef struct { HIR_INSTR_FIELDS; } HirWaitHandleRelease;
typedef struct { HIR_INSTR_FIELDS; } HirWaitHandleLoadWaiter;
typedef struct { HIR_INSTR_FIELDS; } HirWaitHandleLoadCoroOrResult;
typedef struct { HIR_DEOPT_FIELDS; } HirGetAIter;
typedef struct { HIR_DEOPT_FIELDS; } HirGetANext;
typedef struct { HIR_DEOPT_FIELDS; } HirGetIter;
typedef struct { HIR_DEOPT_FIELDS; } HirGetTuple;
typedef struct { HIR_DEOPT_FIELDS; } HirGetLength;
typedef struct { HIR_DEOPT_FIELDS; } HirSetUpdate;
typedef struct { HIR_DEOPT_FIELDS; } HirDictUpdate;
typedef struct { HIR_DEOPT_FIELDS; } HirListExtend;
typedef struct { HIR_DEOPT_FIELDS; } HirListAppend;
typedef struct { HIR_DEOPT_FIELDS; } HirCopyDictWithoutKeys;
typedef struct { HIR_DEOPT_FIELDS; } HirMakeTupleFromList;
typedef struct { HIR_DEOPT_FIELDS; } HirMatchKeys;
typedef struct { HIR_DEOPT_FIELDS; } HirDictMerge;
typedef struct { HIR_DEOPT_FIELDS; } HirDictSubscr;
typedef struct { HIR_DEOPT_FIELDS; } HirInvokeIterNext;
typedef struct { HIR_DEOPT_FIELDS; } HirStoreSubscr;
typedef struct { HIR_DEOPT_FIELDS; } HirSetSetItem;
typedef struct { HIR_DEOPT_FIELDS; } HirUnicodeConcat;
typedef struct { HIR_DEOPT_FIELDS; } HirUnicodeRepeat;
typedef struct { HIR_DEOPT_FIELDS; } HirUnicodeSubscr;
typedef struct { HIR_DEOPT_FIELDS; } HirCheckSequenceBounds;
typedef struct { HIR_INSTR_FIELDS; } HirPrimitiveBoxBool;
typedef struct { HIR_DEOPT_NAMEIDX_FIELDS; } HirImportFrom;
typedef struct { HIR_INSTR_FIELDS; } HirMatchClass;
typedef struct { HIR_DEOPT_FIELDS; } HirYieldValue;
typedef struct { HIR_DEOPT_FIELDS; } HirYieldAndYieldFrom;
typedef struct { HIR_DEOPT_FIELDS; } HirYieldFromHandleStopAsyncIteration;
typedef struct { HIR_DEOPT_FIELDS; } HirInitialYield;
typedef struct { HIR_DEOPT_FIELDS; } HirSend;
typedef struct { HIR_DEOPT_FIELDS; } HirMakeCell;
typedef struct { HIR_DEOPT_FIELDS; } HirMakeFunction;
typedef struct { HIR_INSTR_FIELDS; } HirBatchDecref;

/* ==== T2-B Batch 4b: Container-field instruction structs ====
 * Types with std::string, std::vector, std::unique_ptr, BorrowedRef.
 * C++ containers stored as opaque byte arrays. */

/* ==== H2-A2: DEFINE_SIMPLE_INSTR types with intermediate base classes ====
 * These were missing from H2-A because they inherit through intermediate
 * base classes (CheckBase, CheckBaseWithName, DeoptBaseWithNameIdx,
 * LoadMethodBase, LoadSuperBase, CondBranchBase). */

/* ---- CheckBase types (no extra fields beyond DeoptBase) ---- */
typedef struct { HIR_DEOPT_FIELDS; } HirCheckExc;
typedef struct { HIR_DEOPT_FIELDS; } HirCheckNeg;
typedef struct { HIR_DEOPT_FIELDS; } HirIsNegativeAndErrOccurred;

/* ---- CheckBaseWithName types (DeoptBase + void* name_) ---- */
typedef struct { HIR_CHECK_WITH_NAME_FIELDS; } HirCheckVar;
typedef struct { HIR_CHECK_WITH_NAME_FIELDS; } HirCheckFreevar;
typedef struct { HIR_CHECK_WITH_NAME_FIELDS; } HirCheckField;

/* ---- DeoptBaseWithNameIdx types (no extra fields) ---- */
typedef struct { HIR_DEOPT_NAMEIDX_FIELDS; } HirLoadAttrCached;
typedef struct { HIR_DEOPT_NAMEIDX_FIELDS; } HirStoreAttr;
typedef struct { HIR_DEOPT_NAMEIDX_FIELDS; } HirStoreAttrCached;
typedef struct { HIR_DEOPT_NAMEIDX_FIELDS; } HirLoadGlobal;
typedef struct { HIR_DEOPT_NAMEIDX_FIELDS; } HirLoadModuleAttrCached;

/* ---- LoadMethodBase types (= DeoptBaseWithNameIdx, no extra fields) ---- */
typedef struct { HIR_DEOPT_NAMEIDX_FIELDS; } HirLoadMethod;
typedef struct { HIR_DEOPT_NAMEIDX_FIELDS; } HirLoadMethodCached;
typedef struct { HIR_DEOPT_NAMEIDX_FIELDS; } HirLoadModuleMethodCached;

/* ---- LoadSuperBase types (DeoptBaseWithNameIdx + bool) ---- */
typedef struct { HIR_LOAD_SUPER_FIELDS; } HirLoadMethodSuper;
typedef struct { HIR_LOAD_SUPER_FIELDS; } HirLoadAttrSuper;

/* ---- LoadAttr (DeoptBaseWithNameIdx + bool already_optimized_) ---- */
typedef struct { HIR_DEOPT_NAMEIDX_FIELDS; uint8_t already_optimized; } HirLoadAttr;

/* ---- CondBranchBase aliases (same layout as HirCondBranchInstr) ---- */
typedef HirCondBranchInstr HirCondBranch;
typedef HirCondBranchInstr HirCondBranchIterNotDone;

/* ---- Simple container types ---- */
typedef struct { HIR_DEOPT_FIELDS; uint32_t flags; } HirCallMethod;
typedef struct { HIR_INSTR_FIELDS; int32_t func; } HirCallCFunc;
typedef struct { HIR_INSTR_FIELDS; void *frame_state_ptr; } HirSnapshot;

/* ---- BorrowedRef types (void* layout-compatible) ---- */
typedef struct { HIR_INSTR_FIELDS; void *code; void *builtins; void *globals; int32_t name_idx; } HirLoadGlobalCached;

/* ---- DeoptBaseWithNameIdx types ---- */
typedef struct { HIR_DEOPT_NAMEIDX_FIELDS; int32_t cache_id; } HirFillTypeAttrCache;
typedef struct { HIR_DEOPT_NAMEIDX_FIELDS; int32_t cache_id; } HirFillTypeMethodCache;

/* ---- std::string types (opaque blob) ---- */
typedef struct { HIR_INSTR_FIELDS; char *name; size_t offset; HirType type; uint8_t borrowed; } HirLoadField;
typedef struct { HIR_INSTR_FIELDS; char *name; size_t offset; HirType type; } HirStoreField;

/* ---- std::vector type ---- */
typedef struct { HIR_INSTR_FIELDS; void **bb_data; size_t bb_count; size_t bb_cap; } HirPhi;

/* ---- Complex multi-inheritance + container types ---- */
typedef struct { HIR_INSTR_FIELDS; void *func; void *reifier; void *caller_state_ptr; char *fullname; } HirBeginInlinedFunction;

/* ---- LoadAttrSpecial (DeoptBase + id + failure_fmt_str) ---- */
typedef struct { HIR_DEOPT_FIELDS; void *id; const char *failure_fmt; } HirLoadAttrSpecial;

/* ---- UnpackExToTuple (DeoptBase + before + after) ---- */
typedef struct { HIR_DEOPT_FIELDS; int32_t before; int32_t after; } HirUnpackExToTuple;

/* ---- InvokeStaticFunction (DeoptBase + func + ret_type) ---- */
typedef struct { HIR_DEOPT_FIELDS; void *func; HirType ret_type; } HirInvokeStaticFunction;

/* ---- Opaque nested container (ProfiledTypes) ---- */
typedef struct { HIR_INSTR_FIELDS; char types_storage[24]; } HirHintType;

/* ==== Phase H1b: Unified per-opcode data union ====
 * Single tagged union containing all per-opcode data fields.
 * Used by the flattened HirInstrUnified struct.
 * Simple opcodes (no custom fields) need no union entry. */

typedef union HirOpcodeData {
    /* ---- Operation-kind (single enum) ---- */
    struct { int32_t op; } binary_op;       /* BinaryOp, UnaryOp, InPlaceOp, etc. */
    struct { int32_t op; } compare;         /* Compare, CompareBool, FloatCompare, etc. */

    /* ---- Type field (single HirType) ---- */
    struct { HirType type; } typed;         /* LoadConst, RefineType, BitCast, Return, etc. */

    /* ---- Branch/Edge ---- */
    struct { HirEdge edge; } branch;        /* Branch */
    struct { HirEdge true_edge; HirEdge false_edge; } cond_branch; /* CondBranch */
    struct { HirEdge true_edge; HirEdge false_edge; HirType type; } cond_branch_check_type;

    /* ---- Single scalar ---- */
    struct { int32_t field; } set_func_attr;        /* SetFunctionAttr */
    struct { size_t index; } call_intrinsic;        /* CallIntrinsic */
    struct { void *addr; } call_static_ret_void;    /* CallStaticRetVoid */
    struct { void *exc; } index_unbox;              /* IndexUnbox */
    struct { size_t idx; } load_tuple_item;         /* LoadTupleItem */
    struct { intptr_t item_idx; } load_split_dict_item; /* LoadSplitDictItem */
    struct { int32_t cells; } init_frame_cell_vars; /* InitFrameCellVars */
    struct { int32_t cache_id; } cache_entry;       /* LoadType*CacheEntry* */
    struct { uint32_t flags; } call_flags;          /* VectorCall, CallEx, CallMethod */
    struct { void *pytype; } tp_alloc;              /* TpAlloc */
    struct { void *target; } guard_is;              /* GuardIs */
    struct { HirType target; } guard_type;          /* GuardType */
    struct { size_t capacity; } make_dict;          /* MakeDict */
    struct { void *patcher; } deopt_patchpoint;     /* DeoptPatchpoint */
    struct { uint8_t is_aenter; } raise_awaitable;  /* RaiseAwaitableError */
    struct { int32_t conversion; } format_value;    /* FormatValue, BuildInterpolation */
    struct { int32_t converter_idx; } convert_value; /* ConvertValue */
    struct { int32_t special_idx; } load_special;   /* LoadSpecial */
    struct { uint8_t already_optimized; } load_attr; /* LoadAttr */

    /* ---- Multi-field scalar ---- */
    struct { void *addr; HirType ret_type; } call_static;       /* CallStatic */
    struct { void *begin; int32_t inline_depth; } end_inlined;  /* EndInlinedFunction */
    struct { int32_t line_no; int32_t _pad; void *parent; } update_prev_instr; /* UpdatePrevInstr */
    struct { uint32_t arg_idx; int32_t _pad; HirType type; } load_arg; /* LoadArg */
    struct { intptr_t offset; HirType type; } load_array_item;  /* LoadArrayItem */
    struct { HirType type; } store_array_item;                  /* StoreArrayItem */
    struct { void *pytype; uint8_t optional; uint8_t exact; } cast; /* Cast */
    struct { void *funcptr; void *func_descr; } load_func_indirect;  /* LoadFunctionIndirect */
    struct { const char *fmt; void *exc_type; } raise_static;   /* RaiseStatic */
    struct { const char *name; HirType ret_type; } call_ind;    /* CallInd */
    struct { size_t capacity; HirType type; } make_checked_dict; /* MakeCheckedDict */
    struct { HirType type; } make_checked_list;                 /* MakeCheckedList */
    struct { int32_t before; int32_t after; } unpack_ex;        /* UnpackExToTuple */
    struct { void *func; HirType ret_type; } invoke_static;     /* InvokeStaticFunction */

    /* ---- BorrowedRef types ---- */
    struct { void *code; void *builtins; void *globals; int32_t name_idx; } load_global_cached;

    /* ---- Container types (opaque storage) ---- */
    struct { char *name; size_t offset; HirType type; uint8_t borrowed; } load_field;
    struct { char *name; size_t offset; HirType type; } store_field;
    struct { void **bb_data; size_t bb_count; size_t bb_cap; } phi; /* Phi: PhxBlockArray */
    struct { char types_storage[24]; } hint_type;               /* HintType: ProfiledTypes */
    struct { void *func; void *reifier; void *caller_state_ptr; char *fullname; } begin_inlined;
    struct { void *frame_state_ptr; } snapshot;                 /* Snapshot */
    struct { int32_t func; } call_cfunc;                        /* CallCFunc */

    /* ---- Cache types ---- */
    struct { int32_t name_idx; int32_t cache_id; } fill_cache;  /* FillTypeAttrCache, FillTypeMethodCache */
    struct { void *id; const char *failure_fmt; } load_attr_special; /* LoadAttrSpecial */
} HirOpcodeData;

/* ==== Unified instruction struct ====
 * Base fields + per-opcode data union.
 * sizeof(HirOpcodeData) is dominated by load_field/store_field (~57 bytes)
 * and begin_inlined (~56 bytes). */

/* Note: Two variants — one for Instr base, one for DeoptBase.
 * The DeoptBase variant includes deopt fields between base and union.
 * For a truly unified struct, we use the largest (DeoptBase) as the
 * common layout, with the understanding that non-deopt instructions
 * leave the deopt fields unused. */

/* ---- Field accessors ----
 * All take void* to match hir_c_api.h's opaque HirInstr handle.
 * Internally cast to HirInstrLayout for base field access,
 * or to per-opcode structs for opcode-specific fields. */

static inline int32_t hir_c_opcode(const void *instr) {
    return ((const HirInstrLayout *)instr)->opcode;
}

static inline int32_t hir_c_bytecode_offset(const void *instr) {
    return ((const HirInstrLayout *)instr)->bytecode_offset;
}

static inline void *hir_c_output(const void *instr) {
    return ((const HirInstrLayout *)instr)->output;
}

static inline int hir_c_has_output(const void *instr) {
    return ((const HirInstrLayout *)instr)->output != NULL;
}

static inline void *hir_c_block(const void *instr) {
    return ((const HirInstrLayout *)instr)->block;
}

/* Safe downcast: returns NULL if not a DeoptBase subclass */
static inline const HirDeoptLayout *hir_c_as_deopt(const void *instr) {
    if (!hir_instr_info_is_deopt_base(hir_c_opcode(instr))) {
        return NULL;
    }
    return (const HirDeoptLayout *)instr;
}

/* Mutable variant */
static inline HirDeoptLayout *hir_c_as_deopt_mut(void *instr) {
    if (!hir_instr_info_is_deopt_base(hir_c_opcode(instr))) {
        return NULL;
    }
    return (HirDeoptLayout *)instr;
}

static inline const HirCondBranchInstr *hir_c_as_condbranch(const void *instr) {
    return (const HirCondBranchInstr *)instr;
}

/* ==== Per-opcode data accessors via C struct casts ====
 * Pattern: cast void* to the per-opcode struct and read the field.
 * Step 6 validated this approach: C struct layout matches C++ at runtime. */

/* Branch edge target (the block a Branch jumps to). */
static inline void *hir_c_branch_target(const void *instr) {
    return ((const HirBranch *)instr)->edge.to;
}

/* CondBranch edges */
static inline void *hir_c_condbranch_true_target(const void *instr) {
    return ((const HirCondBranchInstr *)instr)->true_edge.to;
}
static inline void *hir_c_condbranch_false_target(const void *instr) {
    return ((const HirCondBranchInstr *)instr)->false_edge.to;
}

static inline int32_t hir_c_binary_op_kind(const void *instr) {
    return ((const HirBinaryOp *)instr)->op;
}

static inline int32_t hir_c_unary_op_kind(const void *instr) {
    return ((const HirUnaryOp *)instr)->op;
}

static inline int32_t hir_c_compare_op(const void *instr) {
    return ((const HirCompare *)instr)->op;
}

static inline int32_t hir_c_inplace_op_kind(const void *instr) {
    return ((const HirInPlaceOp *)instr)->op;
}

static inline HirType hir_c_load_const_type(const void *instr) {
    return ((const HirLoadConst *)instr)->type;
}

static inline HirType hir_c_guard_type_target(const void *instr) {
    return ((const HirGuardType *)instr)->target;
}

static inline void *hir_c_guard_is_target(const void *instr) {
    return ((const HirGuardIs *)instr)->target;
}

static inline int32_t hir_c_load_arg_idx(const void *instr) {
    return ((const HirLoadArg *)instr)->arg_idx;
}

static inline HirType hir_c_load_arg_type(const void *instr) {
    return ((const HirLoadArg *)instr)->type;
}

static inline int32_t hir_c_cache_entry_id(const void *instr) {
    return ((const HirLoadTypeAttrCacheEntryType *)instr)->cache_id;
}

static inline uint32_t hir_c_call_flags(const void *instr) {
    return ((const HirVectorCall *)instr)->flags;
}

static inline void *hir_c_deopt_patchpoint_patcher(const void *instr) {
    return ((const HirDeoptPatchpoint *)instr)->patcher;
}

static inline HirType hir_c_return_type(const void *instr) {
    return ((const HirReturn *)instr)->type;
}

/* Bytecode offset setter */
static inline void hir_c_set_bytecode_offset(void *instr, int32_t off) {
    ((HirInstrLayout *)instr)->bytecode_offset = off;
}

/* Bytecode offset copy (C-level, no bridge needed) */
static inline void hir_c_copy_bytecode_offset(void *dst, const void *src) {
    ((HirInstrLayout *)dst)->bytecode_offset =
        ((const HirInstrLayout *)src)->bytecode_offset;
}

/* ==== Edge/successor access ====
 * numEdges: Branch=1, CondBranch*=2, else=0.
 * successor(i): read edge[i].to from per-opcode struct layout. */

static inline size_t hir_c_num_edges(const void *instr) {
    int op = hir_c_opcode(instr);
    if (op == HIR_OP_Branch) return 1;
    if (op == HIR_OP_CondBranch ||
        op == HIR_OP_CondBranchIterNotDone ||
        op == HIR_OP_CondBranchCheckType) return 2;
    return 0;
}

static inline HirEdge *hir_c_edge_at(void *instr, size_t i) {
    int op = hir_c_opcode(instr);
    if (op == HIR_OP_Branch) {
        return &((HirBranch *)instr)->edge;
    }
    if (op == HIR_OP_CondBranch ||
        op == HIR_OP_CondBranchIterNotDone ||
        op == HIR_OP_CondBranchCheckType) {
        HirCondBranchInstr *cb = (HirCondBranchInstr *)instr;
        return i == 0 ? &cb->true_edge : &cb->false_edge;
    }
    return NULL;
}

static inline void hir_c_set_block(void *instr, void *block) {
    ((HirInstrLayout *)instr)->block = block;
    if (hir_instr_info_is_terminator(hir_c_opcode(instr))) {
        size_t n = hir_c_num_edges(instr);
        for (size_t i = 0; i < n; i++) {
            HirEdge *e = hir_c_edge_at(instr, i);
            hir_edge_set_from(e, (HirBasicBlock *)block);
        }
    }
}

static inline void *hir_c_successor(const void *instr, size_t i) {
    int op = hir_c_opcode(instr);
    if (op == HIR_OP_Branch) {
        return ((const HirBranch *)instr)->edge.to;
    }
    /* CondBranch variants: i=0 → true_edge, i=1 → false_edge */
    const HirCondBranchInstr *cb = (const HirCondBranchInstr *)instr;
    if (i == 0) return cb->true_edge.to;
    return cb->false_edge.to;
}

/* ==== Register layout ====
 * Minimal C view of hir::Register for field access.
 * Fields: HirType type_ (16 bytes), void* instr_ (8 bytes), int id_ (4 bytes).
 * Followed by std::string name_ (opaque, not accessed from C). */

typedef struct HirRegLayout {
    HirType type;       /* Register::type_ */
    void *instr;        /* Register::instr_ — points to defining Instr */
    int32_t id;         /* Register::id_ */
} HirRegLayout;

/* Get the defining instruction for a register */
static inline void *hir_c_reg_instr(const void *reg) {
    return ((const HirRegLayout *)reg)->instr;
}

/* Set output register on an instruction (mirrors Instr::setOutput) */
static inline void hir_c_set_output(void *instr, void *dst) {
    HirInstrLayout *i = (HirInstrLayout *)instr;
    /* Clear old output's back-pointer */
    if (i->output != NULL) {
        ((HirRegLayout *)i->output)->instr = NULL;
    }
    /* Set new output's back-pointer */
    if (dst != NULL) {
        ((HirRegLayout *)dst)->instr = instr;
    }
    i->output = dst;
}

/* ==== Operand access ====
 * Operands are stored BEFORE the instruction in memory:
 *   [Register*[N] operands] [size_t num_operands] [HirInstrLayout fields...]
 * NumOperands is at ((size_t*)instr)[-1].
 * Operands array starts at ((void**)instr) - 1 - N. */

static inline size_t hir_c_num_operands(const void *instr) {
    return ((const size_t *)instr)[-1];
}

static inline void *hir_c_get_operand(const void *instr, size_t i) {
    const size_t n = hir_c_num_operands(instr);
    void *const *ops = (void *const *)((const size_t *)instr - 1) - n;
    return ops[i];
}

static inline void hir_c_set_operand(void *instr, size_t i, void *reg) {
    const size_t n = hir_c_num_operands(instr);
    void **ops = (void **)((size_t *)instr - 1) - n;
    ops[i] = reg;
}

/* ==== Instruction allocation ====
 * Mirrors Instr::allocate: calloc preamble + struct, return struct ptr.
 * Layout: [Register*[num_ops]] [size_t num_ops] [HirInstrLayout...] */

static inline void *hir_c_alloc_instr(size_t struct_size, size_t num_operands) {
    size_t preamble = num_operands * sizeof(void *) + sizeof(size_t);
    char *ptr = (char *)calloc(preamble + struct_size, 1);
    if (!ptr) return NULL;
    ptr += num_operands * sizeof(void *);
    *(size_t *)ptr = num_operands;
    ptr += sizeof(size_t);
    /* Initialize IntrusiveListNode (block_node_) at offset 0 in Instr.
     * The node's unlinked state requires prev_=this, next_=this (self-pointers).
     * calloc zeros these to NULL, which makes isLinked() return true and
     * triggers an assertion on first insert.  Fix: set both pointers to self. */
    void **node = (void **)ptr;
    node[0] = ptr;  /* prev_ = self */
    node[1] = ptr;  /* next_ = self */
    /* bytecode_offset defaults to -1 in C++ (BCOffset{-1}), but calloc
     * gives 0. Set here so the invariant is self-enforcing — even if
     * hir_c_init_instr is accidentally skipped. */
    ((HirInstrLayout *)ptr)->bytecode_offset = -1;
    return ptr;
}

/* Initialize base Instr fields after allocation.
 * Sets opcode and bytecode_offset. IntrusiveListNode already initialized
 * by hir_c_alloc_instr. bytecode_offset defaults to -1 (matching C++
 * Instr::Instr default: BCOffset bytecode_offset_{-1}). */
static inline void hir_c_init_instr(void *instr, int32_t opcode) {
    HirInstrLayout *i = (HirInstrLayout *)instr;
    i->opcode = opcode;
    i->bytecode_offset = -1;
}

/* Initialize DeoptBase fields after hir_c_init_instr.
 * H2-E1+E2: All DeoptBase containers are now calloc-safe:
 * - live_regs_storage: PhxRegStateArray (data=NULL, count=0, capacity=0)
 * - descr: char* (NULL = no description)
 * - frame_state: raw pointer (NULL)
 * Only nonce needs explicit initialization (C++ defaults to -1, calloc to 0). */
static inline void hir_c_init_deopt(void *instr, int32_t opcode) {
    hir_c_init_instr(instr, opcode);
    /* nonce defaults to 0 from calloc; C++ constructor uses -1. */
    ((HirDeoptLayout *)instr)->nonce = -1;
    /* All other POD fields (frame_state, guilty_reg, descr, live_regs,
     * suppress_exception_deopt) are correctly zero/NULL from calloc. */
}

/* Initialize EndInlinedFunction fields after hir_c_init_instr.
 * inline_depth defaults to 0 from calloc; C++ constructor uses -1.
 * Same enforcement pattern as bytecode_offset and nonce. */
static inline void hir_c_init_end_inlined(void *instr, int32_t opcode) {
    hir_c_init_instr(instr, opcode);
    ((HirEndInlinedFunction *)instr)->inline_depth = -1;
}

/* ==== Slab base pointer ====
 * The slab layout is: [Register*[N]] [size_t N] [Instr struct...]
 * base() returns the start of the slab allocation. */
static inline void *hir_c_instr_base(void *instr) {
    size_t n = hir_c_num_operands(instr);
    return (char *)instr - n * sizeof(void *) - sizeof(size_t);
}

/* ==== C++ destruction helpers ====
 * These thin wrappers are implemented in hir_c_api.cpp. They handle
 * C++ members that can't be cleaned up from pure C. */

/* FrameState: has std::vector and jit::Stack members. */
void hir_c_destroy_frame_state(void *frame_state);

/* Edge: ~Edge() calls set_from/set_to(nullptr) which modifies std::set
 * on BasicBlock (in_edges_/out_edges_). Must be called for Branch (1 edge)
 * and CondBranchBase subtypes (2 edges). */
void hir_c_destroy_edge(void *edge_ptr);

/* HintType: ProfiledTypes = std::vector<std::vector<Type>>. */
void hir_c_destroy_profiled_types(void *types_ptr);

/* ==== Instruction destruction (pure C dispatch) ====
 * Replaces the C++ Instr::Destroy() FOREACH_OPCODE dispatch.
 * Handles per-type cleanup explicitly, then frees the slab. */
static inline void hir_c_destroy_instr_impl(void *instr) {
    void *allocation_base = hir_c_instr_base(instr);
    int32_t op = hir_c_opcode(instr);

    if (hir_instr_info_is_deopt_base(op)) {
        /* DeoptBase: free descr, live_regs data, delete frame_state */
        HirDeoptLayout *d = (HirDeoptLayout *)instr;
        free(d->descr);
        free(d->live_regs_data);
        if (d->frame_state) {
            hir_c_destroy_frame_state(d->frame_state);
        }
    } else if (op == HIR_OP_Snapshot) {
        HirSnapshot *s = (HirSnapshot *)instr;
        if (s->frame_state_ptr) {
            hir_c_destroy_frame_state(s->frame_state_ptr);
        }
    } else if (op == HIR_OP_BeginInlinedFunction) {
        HirBeginInlinedFunction *b = (HirBeginInlinedFunction *)instr;
        if (b->caller_state_ptr) {
            hir_c_destroy_frame_state(b->caller_state_ptr);
        }
        free(b->fullname);
    } else if (op == HIR_OP_Phi) {
        HirPhi *p = (HirPhi *)instr;
        free(p->bb_data);
    } else if (op == HIR_OP_LoadField) {
        HirLoadField *lf = (HirLoadField *)instr;
        free(lf->name);
    } else if (op == HIR_OP_StoreField) {
        HirStoreField *sf = (HirStoreField *)instr;
        free(sf->name);
    } else if (op == HIR_OP_Branch) {
        /* Branch has 1 Edge — ~Edge() unlinks from BasicBlock edge sets. */
        HirBranch *br = (HirBranch *)instr;
        hir_c_destroy_edge(&br->edge);
    } else if (op == HIR_OP_CondBranch ||
               op == HIR_OP_CondBranchIterNotDone ||
               op == HIR_OP_CondBranchCheckType) {
        /* CondBranchBase subtypes have 2 Edges. */
        HirCondBranchInstr *cb = (HirCondBranchInstr *)instr;
        hir_c_destroy_edge(&cb->true_edge);
        hir_c_destroy_edge(&cb->false_edge);
    } else if (op == HIR_OP_HintType) {
        /* HintType has ProfiledTypes (std::vector<std::vector<Type>>). */
        HirHintType *ht = (HirHintType *)instr;
        hir_c_destroy_profiled_types(ht->types_storage);
    }
    /* All other types: no owned resources beyond the slab. */

    free(allocation_base);
}

/* ==== Instruction factory: LoadConst ====
 * Pure C factory — Instr base + HirType field, 0 operands.
 * No DeoptBase, no C++ containers. */

static inline void *hir_c_create_load_const(void *dst_reg, HirType type) {
    HirLoadConst *lc = (HirLoadConst *)hir_c_alloc_instr(sizeof(HirLoadConst), 0);
    if (!lc) return NULL;
    hir_c_init_instr(lc, HIR_OP_LoadConst);
    lc->type = type;
    hir_c_set_output(lc, dst_reg);
    return lc;
}

/* ==== Instruction factory: Branch ====
 * WARNING: DO NOT USE for production — this pure C factory writes edge.to
 * directly, bypassing C++ Edge::set_to() which manages BasicBlock::in_edges_.
 * Use hir_c_create_branch_cpp() from hir_c_api.h instead.
 * Kept only for read-path testing (struct layout verification). */

static inline void *hir_c_create_branch(void *target_block) {
    HirBranch *br = (HirBranch *)hir_c_alloc_instr(sizeof(HirBranch), 0);
    if (!br) return NULL;
    hir_c_init_instr(br, HIR_OP_Branch);
    br->edge.to = target_block;
    return br;
}

/* ==== Instruction factory: CondBranch ====
 * WARNING: DO NOT USE for production — same Edge::set_to() bypass issue
 * as hir_c_create_branch above. Use hir_c_create_cond_branch_cpp() instead. */

static inline void *hir_c_create_cond_branch(void *cond_reg,
                                              void *true_block,
                                              void *false_block) {
    HirCondBranchInstr *cb = (HirCondBranchInstr *)hir_c_alloc_instr(
        sizeof(HirCondBranchInstr), 1);
    if (!cb) return NULL;
    hir_c_init_instr(cb, HIR_OP_CondBranch);
    cb->true_edge.to = true_block;
    cb->false_edge.to = false_block;
    hir_c_set_operand(cb, 0, cond_reg);
    return cb;
}

/* ==== Instruction factory: UseType ====
 * Pure C factory — 1 operand (val reg) + HirType field. No output. */

static inline void *hir_c_create_use_type(void *val_reg, HirType type) {
    HirUseType *ut = (HirUseType *)hir_c_alloc_instr(sizeof(HirUseType), 1);
    if (!ut) return NULL;
    hir_c_init_instr(ut, HIR_OP_UseType);
    ut->type = type;
    hir_c_set_operand(ut, 0, val_reg);
    return ut;
}

/* ==== Instruction factory: PrimitiveCompare ====
 * Pure C factory — 2 operands + int32_t op. Has output. */

static inline void *hir_c_create_primitive_compare(void *dst_reg, int32_t op,
                                                    void *left, void *right) {
    HirPrimitiveCompare *pc = (HirPrimitiveCompare *)hir_c_alloc_instr(
        sizeof(HirPrimitiveCompare), 2);
    if (!pc) return NULL;
    hir_c_init_instr(pc, HIR_OP_PrimitiveCompare);
    pc->op = op;
    hir_c_set_output(pc, dst_reg);
    hir_c_set_operand(pc, 0, left);
    hir_c_set_operand(pc, 1, right);
    return pc;
}

/* ==== Instruction factory: IntBinaryOp ====
 * Pure C factory — 2 operands + int32_t op. Has output. */

static inline void *hir_c_create_int_binary_op(void *dst_reg, int32_t op,
                                                void *left, void *right) {
    HirIntBinaryOp *ib = (HirIntBinaryOp *)hir_c_alloc_instr(
        sizeof(HirIntBinaryOp), 2);
    if (!ib) return NULL;
    hir_c_init_instr(ib, HIR_OP_IntBinaryOp);
    ib->op = op;
    hir_c_set_output(ib, dst_reg);
    hir_c_set_operand(ib, 0, left);
    hir_c_set_operand(ib, 1, right);
    return ib;
}

/* ==== Instruction factory: DoubleBinaryOp ====
 * Pure C factory — 2 operands + int32_t op. Has output. */

static inline void *hir_c_create_double_binary_op(void *dst_reg, int32_t op,
                                                   void *left, void *right) {
    HirDoubleBinaryOp *db = (HirDoubleBinaryOp *)hir_c_alloc_instr(
        sizeof(HirDoubleBinaryOp), 2);
    if (!db) return NULL;
    hir_c_init_instr(db, HIR_OP_DoubleBinaryOp);
    db->op = op;
    hir_c_set_output(db, dst_reg);
    hir_c_set_operand(db, 0, left);
    hir_c_set_operand(db, 1, right);
    return db;
}

/* ==== Instruction factory: PrimitiveUnaryOp ====
 * Pure C factory — 1 operand + int32_t op. Has output. */

static inline void *hir_c_create_primitive_unary_op(void *dst_reg, int32_t op,
                                                     void *operand) {
    HirPrimitiveUnaryOp *pu = (HirPrimitiveUnaryOp *)hir_c_alloc_instr(
        sizeof(HirPrimitiveUnaryOp), 1);
    if (!pu) return NULL;
    hir_c_init_instr(pu, HIR_OP_PrimitiveUnaryOp);
    pu->op = op;
    hir_c_set_output(pu, dst_reg);
    hir_c_set_operand(pu, 0, operand);
    return pu;
}

/* ==== Instruction factory: FloatCompare ====
 * Pure C factory — 2 operands + int32_t op. Has output. */

static inline void *hir_c_create_float_compare(void *dst_reg, int32_t op,
                                                void *left, void *right) {
    HirFloatCompare *fc = (HirFloatCompare *)hir_c_alloc_instr(
        sizeof(HirFloatCompare), 2);
    if (!fc) return NULL;
    hir_c_init_instr(fc, HIR_OP_FloatCompare);
    fc->op = op;
    hir_c_set_output(fc, dst_reg);
    hir_c_set_operand(fc, 0, left);
    hir_c_set_operand(fc, 1, right);
    return fc;
}

/* ==== Instruction factory: LongCompare ====
 * Pure C factory — 2 operands + int32_t op. Has output. */

static inline void *hir_c_create_long_compare(void *dst_reg, int32_t op,
                                               void *left, void *right) {
    HirLongCompare *lc = (HirLongCompare *)hir_c_alloc_instr(
        sizeof(HirLongCompare), 2);
    if (!lc) return NULL;
    hir_c_init_instr(lc, HIR_OP_LongCompare);
    lc->op = op;
    hir_c_set_output(lc, dst_reg);
    hir_c_set_operand(lc, 0, left);
    hir_c_set_operand(lc, 1, right);
    return lc;
}

/* ==== Instruction factory: UnicodeCompare ====
 * Pure C factory — 2 operands + int32_t op. Has output. */

static inline void *hir_c_create_unicode_compare(void *dst_reg, int32_t op,
                                                  void *left, void *right) {
    HirUnicodeCompare *uc = (HirUnicodeCompare *)hir_c_alloc_instr(
        sizeof(HirUnicodeCompare), 2);
    if (!uc) return NULL;
    hir_c_init_instr(uc, HIR_OP_UnicodeCompare);
    uc->op = op;
    hir_c_set_output(uc, dst_reg);
    hir_c_set_operand(uc, 0, left);
    hir_c_set_operand(uc, 1, right);
    return uc;
}

/* ==== Instruction factory: PrimitiveBoxBool ====
 * Pure C factory — 1 operand, no fields. Has output. */

static inline void *hir_c_create_primitive_box_bool(void *dst_reg, void *src) {
    HirInstrLayout *pb = (HirInstrLayout *)hir_c_alloc_instr(
        sizeof(HirInstrLayout), 1);
    if (!pb) return NULL;
    hir_c_init_instr(pb, HIR_OP_PrimitiveBoxBool);
    hir_c_set_output(pb, dst_reg);
    hir_c_set_operand(pb, 0, src);
    return pb;
}

/* ==== Instruction factory: CIntToCBool ====
 * Pure C factory — 1 operand, no fields. Has output. */

static inline void *hir_c_create_cint_to_cbool(void *dst_reg, void *src) {
    HirInstrLayout *ct = (HirInstrLayout *)hir_c_alloc_instr(
        sizeof(HirInstrLayout), 1);
    if (!ct) return NULL;
    hir_c_init_instr(ct, HIR_OP_CIntToCBool);
    hir_c_set_output(ct, dst_reg);
    hir_c_set_operand(ct, 0, src);
    return ct;
}

/* ==== Instruction factory: BitCast ====
 * Pure C factory — 1 operand + HirType field. Has output. */

static inline void *hir_c_create_bit_cast(void *dst_reg, void *src, HirType type) {
    HirBitCast *bc = (HirBitCast *)hir_c_alloc_instr(sizeof(HirBitCast), 1);
    if (!bc) return NULL;
    hir_c_init_instr(bc, HIR_OP_BitCast);
    bc->type = type;
    hir_c_set_output(bc, dst_reg);
    hir_c_set_operand(bc, 0, src);
    return bc;
}

/* ==== Instruction factory: RefineType ====
 * Pure C factory — 1 operand + HirType field. Has output. */

static inline void *hir_c_create_refine_type(void *dst_reg, HirType type, void *src) {
    HirRefineType *rt = (HirRefineType *)hir_c_alloc_instr(sizeof(HirRefineType), 1);
    if (!rt) return NULL;
    hir_c_init_instr(rt, HIR_OP_RefineType);
    rt->type = type;
    hir_c_set_output(rt, dst_reg);
    hir_c_set_operand(rt, 0, src);
    return rt;
}

/* ==== Instruction factory: PrimitiveUnbox ====
 * Pure C factory — 1 operand + HirType field. Has output. */

static inline void *hir_c_create_primitive_unbox(void *dst_reg, void *src, HirType type) {
    HirPrimitiveUnbox *pu = (HirPrimitiveUnbox *)hir_c_alloc_instr(
        sizeof(HirPrimitiveUnbox), 1);
    if (!pu) return NULL;
    hir_c_init_instr(pu, HIR_OP_PrimitiveUnbox);
    pu->type = type;
    hir_c_set_output(pu, dst_reg);
    hir_c_set_operand(pu, 0, src);
    return pu;
}

/* ==== Instruction factory: IndexUnbox ====
 * Pure C factory — 1 operand + void* exc field. Has output. */

static inline void *hir_c_create_index_unbox(void *dst_reg, void *src, void *exc) {
    HirIndexUnbox *iu = (HirIndexUnbox *)hir_c_alloc_instr(
        sizeof(HirIndexUnbox), 1);
    if (!iu) return NULL;
    hir_c_init_instr(iu, HIR_OP_IndexUnbox);
    iu->exc = exc;
    hir_c_set_output(iu, dst_reg);
    hir_c_set_operand(iu, 0, src);
    return iu;
}

/* ==== Cache entry factories (pure C, scalar cache_id field) ==== */

static inline void *hir_c_create_load_type_attr_cache_entry_type(
        void *dst_reg, int32_t cache_id) {
    HirLoadTypeAttrCacheEntryType *lc = (HirLoadTypeAttrCacheEntryType *)
        hir_c_alloc_instr(sizeof(HirLoadTypeAttrCacheEntryType), 0);
    if (!lc) return NULL;
    hir_c_init_instr(lc, HIR_OP_LoadTypeAttrCacheEntryType);
    lc->cache_id = cache_id;
    hir_c_set_output(lc, dst_reg);
    return lc;
}

static inline void *hir_c_create_load_type_attr_cache_entry_value(
        void *dst_reg, int32_t cache_id) {
    HirLoadTypeAttrCacheEntryValue *lc = (HirLoadTypeAttrCacheEntryValue *)
        hir_c_alloc_instr(sizeof(HirLoadTypeAttrCacheEntryValue), 0);
    if (!lc) return NULL;
    hir_c_init_instr(lc, HIR_OP_LoadTypeAttrCacheEntryValue);
    lc->cache_id = cache_id;
    hir_c_set_output(lc, dst_reg);
    return lc;
}

static inline void *hir_c_create_load_type_method_cache_entry_type(
        void *dst_reg, int32_t cache_id) {
    HirLoadTypeMethodCacheEntryType *lc = (HirLoadTypeMethodCacheEntryType *)
        hir_c_alloc_instr(sizeof(HirLoadTypeMethodCacheEntryType), 0);
    if (!lc) return NULL;
    hir_c_init_instr(lc, HIR_OP_LoadTypeMethodCacheEntryType);
    lc->cache_id = cache_id;
    hir_c_set_output(lc, dst_reg);
    return lc;
}

static inline void *hir_c_create_load_type_method_cache_entry_value(
        void *dst_reg, int32_t cache_id, void *receiver) {
    HirLoadTypeMethodCacheEntryValue *lc = (HirLoadTypeMethodCacheEntryValue *)
        hir_c_alloc_instr(sizeof(HirLoadTypeMethodCacheEntryValue), 1);
    if (!lc) return NULL;
    hir_c_init_instr(lc, HIR_OP_LoadTypeMethodCacheEntryValue);
    lc->cache_id = cache_id;
    hir_c_set_output(lc, dst_reg);
    hir_c_set_operand(lc, 0, receiver);
    return lc;
}

static inline void *hir_c_create_load_split_dict_item(
        void *dst_reg, void *src, intptr_t item_idx) {
    HirLoadSplitDictItem *ls = (HirLoadSplitDictItem *)
        hir_c_alloc_instr(sizeof(HirLoadSplitDictItem), 1);
    if (!ls) return NULL;
    hir_c_init_instr(ls, HIR_OP_LoadSplitDictItem);
    ls->item_idx = item_idx;
    hir_c_set_output(ls, dst_reg);
    hir_c_set_operand(ls, 0, src);
    return ls;
}

/* ==== Opcode predicates ====
 * Direct opcode field read — no C++ bridge. These replace the
 * hir_instr_is_* functions in hir_c_api.h for C consumers that
 * include this header. */

static inline int hir_c_is_branch(const void *instr) {
    return hir_c_opcode(instr) == HIR_OP_Branch;
}
static inline int hir_c_is_phi(const void *instr) {
    return hir_c_opcode(instr) == HIR_OP_Phi;
}
static inline int hir_c_is_snapshot(const void *instr) {
    return hir_c_opcode(instr) == HIR_OP_Snapshot;
}
static inline int hir_c_is_assign(const void *instr) {
    return hir_c_opcode(instr) == HIR_OP_Assign;
}
static inline int hir_c_is_condbranch(const void *instr) {
    int op = hir_c_opcode(instr);
    return op == HIR_OP_CondBranch ||
           op == HIR_OP_CondBranchIterNotDone ||
           op == HIR_OP_CondBranchCheckType;
}
static inline int hir_c_is_istruthy(const void *instr) {
    return hir_c_opcode(instr) == HIR_OP_IsTruthy;
}
static inline int hir_c_is_compare(const void *instr) {
    return hir_c_opcode(instr) == HIR_OP_Compare;
}
static inline int hir_c_is_comparebool(const void *instr) {
    return hir_c_opcode(instr) == HIR_OP_CompareBool;
}
static inline int hir_c_is_vectorcall(const void *instr) {
    return hir_c_opcode(instr) == HIR_OP_VectorCall;
}
static inline int hir_c_is_guard_type(const void *instr) {
    return hir_c_opcode(instr) == HIR_OP_GuardType;
}
static inline int hir_c_is_primitive_box(const void *instr) {
    return hir_c_opcode(instr) == HIR_OP_PrimitiveBox;
}
static inline int hir_c_is_terminator(const void *instr) {
    return hir_instr_info_is_terminator(hir_c_opcode(instr));
}
static inline int hir_c_is_deopt_base(const void *instr) {
    return hir_instr_info_is_deopt_base(hir_c_opcode(instr));
}
static inline int hir_c_is_replayable(const void *instr) {
    return hir_instr_info_is_replayable(hir_c_opcode(instr));
}

/* ==== Phi query functions (no BasicBlock dependency) ==== */

static inline size_t hir_phi_num_blocks(const void *phi) {
    return ((const HirPhi *)phi)->bb_count;
}

static inline void *hir_phi_block_at(const void *phi, size_t i) {
    return ((const HirPhi *)phi)->bb_data[i];
}

static inline void *hir_phi_is_trivial_impl(const void *phi) {
    void *out = hir_c_output(phi);
    void *val = NULL;
    size_t n = hir_c_num_operands(phi);
    for (size_t i = 0; i < n; i++) {
        void *reg = hir_c_get_operand(phi, i);
        if (reg != out && reg != val) {
            if (val != NULL) return NULL;
            val = reg;
        }
    }
    return val;
}

/* ==== visitUses — C implementation with C++ delegation ====
 * Iterates operand array in pure C. Delegates Snapshot/DeoptBase
 * extension uses (frame_state, live_regs, guilty_reg) to C++ bridge.
 *
 * Callback signature: int (*)(void **reg_slot, void *ctx)
 *   - reg_slot: pointer to the Register* slot (mutable for replacement)
 *   - return 1 to continue, 0 to stop early
 *
 * Forward-declare the bridge function (defined in hir_c_api.cpp). */
int hir_c_visit_deopt_extension(void *instr,
                                int (*callback)(void **reg_slot, void *ctx),
                                void *ctx);

static inline int hir_c_visit_uses(void *instr,
                                   int (*callback)(void **reg_slot, void *ctx),
                                   void *ctx) {
    int op = hir_c_opcode(instr);

    /* Snapshot: no operands, only frame_state uses */
    if (op == HIR_OP_Snapshot) {
        return hir_c_visit_deopt_extension(instr, callback, ctx);
    }

    /* Base: iterate operand array in pure C */
    size_t n = hir_c_num_operands(instr);
    void **ops = (void **)((size_t *)instr - 1) - n;
    for (size_t i = 0; i < n; i++) {
        if (!callback(&ops[i], ctx)) return 0;
    }

    /* DeoptBase extension: frame_state + live_regs + guilty_reg */
    if (hir_instr_info_is_deopt_base(op)) {
        return hir_c_visit_deopt_extension(instr, callback, ctx);
    }

    return 1;
}


/* ==== CRTP Phase 2: Simple instruction factories ====
 * Used to replace InstrT::create() callers in optimization passes. */

/* Assign: 1 operand (src), has output (dst). No DeoptBase. */
static inline void *hir_c_create_assign(void *dst_reg, void *src_reg) {
    HirInstrLayout *a = (HirInstrLayout *)hir_c_alloc_instr(sizeof(HirInstrLayout), 1);
    if (!a) return NULL;
    hir_c_init_instr(a, HIR_OP_Assign);
    hir_c_set_output(a, dst_reg);
    hir_c_set_operand(a, 0, src_reg);
    return a;
}

/* Incref: 1 operand (reg). No output, no DeoptBase. */
static inline void *hir_c_create_incref(void *reg) {
    HirInstrLayout *i = (HirInstrLayout *)hir_c_alloc_instr(sizeof(HirInstrLayout), 1);
    if (!i) return NULL;
    hir_c_init_instr(i, HIR_OP_Incref);
    hir_c_set_operand(i, 0, reg);
    return i;
}

/* XIncref: 1 operand (reg). No output, no DeoptBase. */
static inline void *hir_c_create_xincref(void *reg) {
    HirInstrLayout *i = (HirInstrLayout *)hir_c_alloc_instr(sizeof(HirInstrLayout), 1);
    if (!i) return NULL;
    hir_c_init_instr(i, HIR_OP_XIncref);
    hir_c_set_operand(i, 0, reg);
    return i;
}

/* Decref: 1 operand (reg). No output, no DeoptBase. */
static inline void *hir_c_create_decref(void *reg) {
    HirInstrLayout *i = (HirInstrLayout *)hir_c_alloc_instr(sizeof(HirInstrLayout), 1);
    if (!i) return NULL;
    hir_c_init_instr(i, HIR_OP_Decref);
    hir_c_set_operand(i, 0, reg);
    return i;
}

/* XDecref: 1 operand (reg). No output, no DeoptBase. */
static inline void *hir_c_create_xdecref(void *reg) {
    HirInstrLayout *i = (HirInstrLayout *)hir_c_alloc_instr(sizeof(HirInstrLayout), 1);
    if (!i) return NULL;
    hir_c_init_instr(i, HIR_OP_XDecref);
    hir_c_set_operand(i, 0, reg);
    return i;
}

/* BatchDecref: N operands. No output, no DeoptBase. */
static inline void *hir_c_create_batch_decref(size_t num) {
    HirInstrLayout *b = (HirInstrLayout *)hir_c_alloc_instr(sizeof(HirInstrLayout), num);
    if (!b) return NULL;
    hir_c_init_instr(b, HIR_OP_BatchDecref);
    return b;
}

/* UpdatePrevInstr: 0 operands + line_no + parent (FrameState*). */
static inline void *hir_c_create_update_prev_instr(int32_t line_no, void *parent) {
    HirUpdatePrevInstr *u = (HirUpdatePrevInstr *)hir_c_alloc_instr(sizeof(HirUpdatePrevInstr), 0);
    if (!u) return NULL;
    hir_c_init_instr(u, HIR_OP_UpdatePrevInstr);
    u->line_no = line_no;
    u->parent = parent;
    return u;
}

/* EndInlinedFunction: 0 operands + begin (BeginInlinedFunction*). */
static inline void *hir_c_create_end_inlined_function(void *begin) {
    HirEndInlinedFunction *e = (HirEndInlinedFunction *)hir_c_alloc_instr(sizeof(HirEndInlinedFunction), 0);
    if (!e) return NULL;
    hir_c_init_end_inlined(e, HIR_OP_EndInlinedFunction);
    e->begin = begin;
    return e;
}

/* Unreachable: 0 operands. No output, no DeoptBase. */
static inline void *hir_c_create_unreachable(void) {
    HirInstrLayout *u = (HirInstrLayout *)hir_c_alloc_instr(sizeof(HirInstrLayout), 0);
    if (!u) return NULL;
    hir_c_init_instr(u, HIR_OP_Unreachable);
    return u;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
