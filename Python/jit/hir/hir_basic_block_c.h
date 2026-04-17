/*
 * hir_basic_block_c.h -- C struct definition for HIR BasicBlock
 *
 * Phase H1a: C-compatible layout for hir::BasicBlock.
 * Matches C++ class layout field-by-field, validated by
 * sizeof + offsetof static_asserts in hir.cpp.
 *
 * This enables C code to access BasicBlock fields directly
 * via reinterpret_cast, same pattern as LIR (lir_types_c.h).
 */

#ifndef JIT_HIR_BASIC_BLOCK_C_H
#define JIT_HIR_BASIC_BLOCK_C_H

#include "cinderx/Jit/hir/hir_instr_c.h"  /* PhxEdgePtrArray, HirEdge */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- IntrusiveListNode (C equivalent) ---- */
typedef struct HirIntrusiveListNode {
    struct HirIntrusiveListNode *prev_;
    struct HirIntrusiveListNode *next_;
} HirIntrusiveListNode;

/* ---- Instr::List (IntrusiveList<Instr>) ---- */
typedef struct HirInstrList {
    HirIntrusiveListNode root_;       /* sentinel node (16 bytes) */
    size_t node_member_offset_;       /* offset of block_node_ in Instr */
} HirInstrList;

/* ---- HirBasicBlock ---- */
typedef struct HirBasicBlock {
    int id;                           /* 4 bytes */
    /* 4 bytes padding (align cfg_node to 8) */

    HirIntrusiveListNode cfg_node;    /* 16 bytes: prev/next in CFG block list */

    HirInstrList instrs_;             /* 24 bytes: instruction intrusive list */

    PhxEdgePtrArray out_edges_;       /* 24 bytes: outgoing edges (already C) */
    PhxEdgePtrArray in_edges_;        /* 24 bytes: incoming edges (already C) */
} HirBasicBlock;

/* Expected size: 4 + 4(pad) + 16 + 24 + 24 + 24 = 96 bytes */

/* ---- Edge management ---- */

void hir_edge_set_from(HirEdge *edge, HirBasicBlock *new_from);
void hir_edge_set_to(HirEdge *edge, HirBasicBlock *new_to);
void hir_edge_destroy(HirEdge *edge);

/* ---- Accessors ---- */

int hir_bb_id(const HirBasicBlock *bb);
size_t hir_bb_in_edges_count(const HirBasicBlock *bb);
size_t hir_bb_out_edges_count(const HirBasicBlock *bb);
const HirEdge *hir_bb_in_edge(const HirBasicBlock *bb, size_t i);
const HirEdge *hir_bb_out_edge(const HirBasicBlock *bb, size_t i);

/* ---- Edge-aware operations ---- */

void hir_bb_retarget_preds(HirBasicBlock *bb, HirBasicBlock *target);

/* ---- Instruction list operations ---- */

int hir_bb_empty(const HirBasicBlock *bb);
void *hir_bb_first_instr(const HirBasicBlock *bb);
void *hir_bb_last_instr(const HirBasicBlock *bb);
void *hir_bb_next_instr(const HirBasicBlock *bb, void *instr);
void *hir_bb_prev_instr(const HirBasicBlock *bb, void *instr);
void *hir_bb_append_instr(HirBasicBlock *bb, void *instr);
void hir_bb_push_front_instr(HirBasicBlock *bb, void *instr);
void *hir_bb_pop_front_instr(HirBasicBlock *bb);
void hir_bb_insert_before(HirBasicBlock *bb, void *instr, void *before);
void hir_bb_clear(HirBasicBlock *bb);
void *hir_bb_get_terminator(const HirBasicBlock *bb);
void *hir_bb_entry_snapshot(const HirBasicBlock *bb);
int hir_bb_is_trampoline(const HirBasicBlock *bb);

/* ---- CFG C struct ---- */
typedef struct HirCFG {
    void *entry_block;                /* BasicBlock* */
    HirIntrusiveListNode block_root;  /* sentinel for block list */
    size_t block_node_offset;         /* offsetof(cfg_node) in BasicBlock */
    int next_block_id;                /* next block ID to allocate */
    int _cfg_pad0;                    /* alignment padding */
} HirCFG;

/* ---- CFG block list operations ---- */

static inline HirBasicBlock *hir_cfg_first_block(const HirCFG *cfg) {
    const HirIntrusiveListNode *s = &cfg->block_root;
    if (s->next_ == s) return NULL;
    return (HirBasicBlock *)((char *)s->next_ - cfg->block_node_offset);
}

static inline HirBasicBlock *hir_cfg_next_block(const HirCFG *cfg, const HirBasicBlock *bb) {
    const HirIntrusiveListNode *node =
        (const HirIntrusiveListNode *)((const char *)bb + cfg->block_node_offset);
    if (node->next_ == &cfg->block_root) return NULL;
    return (HirBasicBlock *)((char *)node->next_ - cfg->block_node_offset);
}

static inline void hir_cfg_insert_block(HirCFG *cfg, HirBasicBlock *bb) {
    HirIntrusiveListNode *node =
        (HirIntrusiveListNode *)((char *)bb + cfg->block_node_offset);
    HirIntrusiveListNode *s = (HirIntrusiveListNode *)&cfg->block_root;
    HirIntrusiveListNode *prev = s->prev_;
    prev->next_ = node;
    node->prev_ = prev;
    node->next_ = s;
    s->prev_ = node;
}

static inline void hir_cfg_remove_block(HirCFG *cfg, HirBasicBlock *bb) {
    HirIntrusiveListNode *node =
        (HirIntrusiveListNode *)((char *)bb + cfg->block_node_offset);
    node->prev_->next_ = node->next_;
    node->next_->prev_ = node->prev_;
    node->prev_ = node;
    node->next_ = node;
    (void)cfg;
}

/* ---- Phi query (needs HirBasicBlock for block id) ---- */

static inline size_t hir_phi_block_index(const void *phi, const HirBasicBlock *block) {
    const HirPhi *p = (const HirPhi *)phi;
    const int target_id = block->id;
    size_t lo = 0, hi = p->bb_count;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        int mid_id = ((const HirBasicBlock *)p->bb_data[mid])->id;
        if (mid_id < target_id) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

/* ---- Instr set_block (needs HirBasicBlock + hir_edge_set_from) ---- */

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

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* JIT_HIR_BASIC_BLOCK_C_H */
