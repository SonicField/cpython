/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C-compatible struct definitions for HIR instruction base types.
 * T2-B Phase 1: Layout-compatible with C++ Instr, DeoptBase,
 * CondBranchBase classes. For READING fields from C code only —
 * construction remains C++ until T2-E.
 *
 * Verified via sizeof + offsetof static_asserts in hir_instr_c_verify.cpp.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- IntrusiveListNode (matches jit::IntrusiveListNode) ---- */
typedef struct HirListNode {
    struct HirListNode *prev;
    struct HirListNode *next;
} HirListNode;

/* ---- Edge (matches jit::hir::Edge) ---- */
typedef struct HirEdge {
    void *from;   /* BasicBlock* */
    void *to;     /* BasicBlock* */
} HirEdge;

/* ---- HirInstr (matches jit::hir::Instr) ----
 * Base struct for all 168 HIR instructions.
 * Offset 0 is the C++ vtable pointer (opaque, do not touch from C). */
typedef struct HirInstr {
    void *_vtable;              /* C++ vtable pointer — do not access */
    HirListNode block_node;     /* intrusive list node (prev, next) */
    int32_t opcode;             /* enum Opcode underlying int */
    int32_t bytecode_offset;    /* BCOffset (wraps int, default -1) */
    void *output;               /* Register* */
    void *block;                /* BasicBlock* */
} HirInstr;
/* Expected size: 48 bytes */

/* ---- HirDeoptInstr (matches jit::hir::DeoptBase) ----
 * Extends HirInstr. Contains C++ container types as opaque blobs. */
typedef struct HirDeoptInstr {
    HirInstr base;
    /* std::vector<RegState> — opaque 24-byte blob */
    char live_regs_storage[24];
    /* std::unique_ptr<FrameState> — 8-byte pointer */
    void *frame_state;
    /* Register* */
    void *guilty_reg;
    /* int nonce (default -1) */
    int32_t nonce;
    /* 4 bytes padding */
    int32_t _pad0;
    /* std::string — opaque 32-byte blob (SSO, compiler-dependent) */
    char descr_storage[32];
    /* bool */
    uint8_t suppress_exception_deopt;
    /* 7 bytes padding */
    uint8_t _pad1[7];
} HirDeoptInstr;
/* Expected size: 128 bytes (verify with static_assert) */

/* ---- HirCondBranchInstr (matches jit::hir::CondBranchBase) ----
 * Extends HirInstr (NOT DeoptBase). */
typedef struct HirCondBranchInstr {
    HirInstr base;
    HirEdge true_edge;
    HirEdge false_edge;
} HirCondBranchInstr;
/* Expected size: 80 bytes */

/* ---- Field accessors for HirInstr ---- */

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

/* ---- Field accessors for HirDeoptInstr ---- */

static inline void *hir_deopt_instr_frame_state(const HirDeoptInstr *instr) {
    return instr->frame_state;
}

static inline void *hir_deopt_instr_guilty_reg(const HirDeoptInstr *instr) {
    return instr->guilty_reg;
}

static inline int32_t hir_deopt_instr_nonce(const HirDeoptInstr *instr) {
    return instr->nonce;
}

/* ---- Downcast helpers ---- */

static inline const HirDeoptInstr *hir_instr_as_deopt(const HirInstr *instr) {
    /* Caller must verify opcode is a DeoptBase subclass first */
    return (const HirDeoptInstr *)instr;
}

static inline const HirCondBranchInstr *hir_instr_as_condbranch(
        const HirInstr *instr) {
    return (const HirCondBranchInstr *)instr;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
