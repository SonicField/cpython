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
    char live_regs_storage[24]; /* std::vector<RegState> */ \
    void *frame_state;          /* unique_ptr<FrameState> */\
    void *guilty_reg;           /* Register* */             \
    int32_t nonce;                                          \
    int32_t _deopt_pad0;                                    \
    char descr_storage[32];     /* std::string */           \
    uint8_t suppress_exception_deopt

/* ---- Base structs (standalone use) ---- */

typedef struct HirInstr {
    HIR_INSTR_FIELDS;
} HirInstr;

typedef struct HirDeoptInstr {
    HIR_DEOPT_FIELDS;
} HirDeoptInstr;

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
typedef struct { HIR_DEOPT_FIELDS; void *funcptr; void *descr; } HirLoadFunctionIndirect;
typedef struct { HIR_DEOPT_FIELDS; const char *fmt; void *exc_type; } HirRaiseStatic;
typedef struct { HIR_DEOPT_FIELDS; const char *name; HirType ret_type; } HirCallInd;
typedef struct { HIR_DEOPT_FIELDS; size_t capacity; HirType type; } HirMakeCheckedDict;
typedef struct { HIR_DEOPT_FIELDS; HirType type; } HirMakeCheckedList;

/* ---- CondBranch derived ---- */
typedef struct { HIR_INSTR_FIELDS; HirEdge true_edge; HirEdge false_edge; HirType type; } HirCondBranchCheckType;

/* ==== T2-B Batch 4b: Container-field instruction structs ====
 * Types with std::string, std::vector, std::unique_ptr, BorrowedRef.
 * C++ containers stored as opaque byte arrays. */

/* DeoptBaseWithNameIdx: DeoptBase + int name_idx_ */
#define HIR_DEOPT_NAMEIDX_FIELDS  \
    HIR_DEOPT_FIELDS;             \
    int32_t name_idx

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
typedef struct { HIR_INSTR_FIELDS; char name_storage[32]; size_t offset; HirType type; uint8_t borrowed; } HirLoadField;
typedef struct { HIR_INSTR_FIELDS; char name_storage[32]; size_t offset; HirType type; } HirStoreField;

/* ---- std::vector type ---- */
typedef struct { HIR_INSTR_FIELDS; char basic_blocks_storage[24]; } HirPhi;

/* ---- Complex multi-inheritance + container types ---- */
typedef struct { HIR_INSTR_FIELDS; void *func; void *reifier; void *caller_state_ptr; char fullname_storage[32]; } HirBeginInlinedFunction;

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
    struct { void *funcptr; void *descr; } load_func_indirect;  /* LoadFunctionIndirect */
    struct { const char *fmt; void *exc_type; } raise_static;   /* RaiseStatic */
    struct { const char *name; HirType ret_type; } call_ind;    /* CallInd */
    struct { size_t capacity; HirType type; } make_checked_dict; /* MakeCheckedDict */
    struct { HirType type; } make_checked_list;                 /* MakeCheckedList */
    struct { int32_t before; int32_t after; } unpack_ex;        /* UnpackExToTuple */
    struct { void *func; HirType ret_type; } invoke_static;     /* InvokeStaticFunction */

    /* ---- BorrowedRef types ---- */
    struct { void *code; void *builtins; void *globals; int32_t name_idx; } load_global_cached;

    /* ---- Container types (opaque storage) ---- */
    struct { char name_storage[32]; size_t offset; HirType type; uint8_t borrowed; } load_field;
    struct { char name_storage[32]; size_t offset; HirType type; } store_field;
    struct { char basic_blocks_storage[24]; } phi;              /* Phi: vector<BasicBlock*> */
    struct { char types_storage[24]; } hint_type;               /* HintType: ProfiledTypes */
    struct { void *func; void *reifier; void *caller_state_ptr; char fullname_storage[32]; } begin_inlined;
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

/* ---- Field accessors ---- */

static inline int32_t hir_instr_opcode(const HirInstr *instr) {
    return instr->opcode;
}

static inline int32_t hir_instr_bytecode_offset(const HirInstr *instr) {
    return instr->bytecode_offset;
}

static inline void *hir_instr_output(const HirInstr *instr) {
    return instr->output;
}

static inline int hir_instr_has_output(const HirInstr *instr) {
    return instr->output != NULL;
}

/* Safe downcast: returns NULL if not a DeoptBase subclass (T2-C1) */
static inline const HirDeoptInstr *hir_instr_as_deopt(const HirInstr *instr) {
    if (!hir_instr_info_is_deopt_base(instr->opcode)) {
        return NULL;
    }
    return (const HirDeoptInstr *)instr;
}

/* Mutable variant */
static inline HirDeoptInstr *hir_instr_as_deopt_mut(HirInstr *instr) {
    if (!hir_instr_info_is_deopt_base(instr->opcode)) {
        return NULL;
    }
    return (HirDeoptInstr *)instr;
}

static inline const HirCondBranchInstr *hir_instr_as_condbranch(
        const HirInstr *instr) {
    return (const HirCondBranchInstr *)instr;
}

/* ==== Step 6: Per-opcode data accessors via C struct casts ====
 * These validate that C consumers can access per-opcode fields
 * through the existing layout-compatible structs.
 * Pattern: cast HirInstr* to the per-opcode struct and read the field. */

/* Get the target block from a Branch instruction's edge.
 * This is the canonical Step 6 validation: a C consumer (clean_cfg.c)
 * calls this to access per-opcode data through the C struct layout. */
static inline void *hir_branch_edge_target(const HirInstr *instr) {
    return ((const HirBranch *)instr)->edge.to;
}

static inline int32_t hir_binary_op_kind(const HirInstr *instr) {
    return ((const HirBinaryOp *)instr)->op;
}

static inline int32_t hir_unary_op_kind(const HirInstr *instr) {
    return ((const HirUnaryOp *)instr)->op;
}

static inline int32_t hir_compare_op(const HirInstr *instr) {
    return ((const HirCompare *)instr)->op;
}

static inline int32_t hir_inplace_op_kind(const HirInstr *instr) {
    return ((const HirInPlaceOp *)instr)->op;
}

static inline HirType hir_load_const_type(const HirInstr *instr) {
    return ((const HirLoadConst *)instr)->type;
}

static inline HirType hir_guard_type_target(const HirInstr *instr) {
    return ((const HirGuardType *)instr)->target;
}

static inline void *hir_guard_is_target(const HirInstr *instr) {
    return ((const HirGuardIs *)instr)->target;
}

static inline int32_t hir_load_arg_idx(const HirInstr *instr) {
    return ((const HirLoadArg *)instr)->arg_idx;
}

static inline HirType hir_load_arg_type(const HirInstr *instr) {
    return ((const HirLoadArg *)instr)->type;
}

static inline int32_t hir_cache_entry_id(const HirInstr *instr) {
    return ((const HirLoadTypeAttrCacheEntryType *)instr)->cache_id;
}

static inline uint32_t hir_call_flags(const HirInstr *instr) {
    return ((const HirVectorCall *)instr)->flags;
}

static inline void *hir_deopt_patchpoint_patcher(const HirInstr *instr) {
    return ((const HirDeoptPatchpoint *)instr)->patcher;
}

static inline HirType hir_return_type(const HirInstr *instr) {
    return ((const HirReturn *)instr)->type;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
