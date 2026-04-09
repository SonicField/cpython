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
    void *_vtable;              /* C++ vtable pointer */    \
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
typedef struct { HIR_INSTR_FIELDS; void *_vtable_inline; void *begin; int32_t inline_depth; } HirEndInlinedFunction;
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
typedef struct { HIR_INSTR_FIELDS; void *_vtable_inline; void *func; void *reifier; void *caller_state_ptr; char fullname_storage[32]; } HirBeginInlinedFunction;

/* ---- Opaque nested container (ProfiledTypes) ---- */
typedef struct { HIR_INSTR_FIELDS; char types_storage[24]; } HirHintType;

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

/* Downcast: caller must verify opcode is a DeoptBase subclass first */
static inline const HirDeoptInstr *hir_instr_as_deopt(const HirInstr *instr) {
    return (const HirDeoptInstr *)instr;
}

static inline const HirCondBranchInstr *hir_instr_as_condbranch(
        const HirInstr *instr) {
    return (const HirCondBranchInstr *)instr;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
