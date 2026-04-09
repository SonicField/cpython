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
