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

/* Bytecode offset copy (C-level, no bridge needed) */
static inline void hir_c_copy_bytecode_offset(void *dst, const void *src) {
    ((HirInstrLayout *)dst)->bytecode_offset =
        ((const HirInstrLayout *)src)->bytecode_offset;
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

#ifdef __cplusplus
} /* extern "C" */
#endif
