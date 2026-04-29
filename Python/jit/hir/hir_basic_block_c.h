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
#include "cinderx/Common/jit_log_c.h"      /* JIT_DCHECK_C (Batch 23) */

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

/* Initialize dst as a copy of src by routing both endpoints through
 * hir_edge_set_from / hir_edge_set_to so the BasicBlock in_edges_
 * tracking stays consistent. Used by Edge::Edge(const Edge&) port
 * (Phase 4.A Batch 15). */
static inline void hir_c_edge_copy_init(HirEdge *dst, const HirEdge *src) {
    hir_edge_set_from(dst, (HirBasicBlock *)src->from);
    hir_edge_set_to(dst, (HirBasicBlock *)src->to);
}

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

/* CFG destructor (C port of CFG::~CFG) */
void hir_cfg_destroy_c(HirCFG *cfg);

/* BasicBlock destructor — implemented in hir_c_api.cpp (C++ side delegates
 * to `delete static_cast<BasicBlock*>(block)`). Promoted to this header
 * per W25 Step B Class C1 → C2 conversion (hir_bb_* namespace family
 * belongs in hir_basic_block_c.h). */
void hir_bb_destroy(void *block);

/* W25 Step B-3.5 hir_bb_* family promotions (Class C1 → C2). */
void hir_bb_set_successor_null(void *block, size_t idx);
void hir_bb_remove_phi_predecessor(void *block, void *pred);

/* W25 Step B-77a hir_bb_in_edges_list promotion (Class C1 → C2):
 * impl in hir_c_api.cpp, callers in pass_output_type_c.c. */
size_t hir_bb_in_edges_list(void *block, void **out_from, size_t capacity);

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

/* Phi setArgs sort comparator (Batch 20). Operates on an array of
 * HirPhiArgPair where .key is a BasicBlock* and .value is a Register*.
 * The C++ shim builds the pair array from the unordered_map, qsorts
 * via this comparator, then hands sorted parallel arrays to
 * hir_c_phi_apply_args. Lives here (not hir_instr_c.h) because it
 * dereferences HirBasicBlock.id. */
typedef struct HirPhiArgPair {
    void *key;     /* BasicBlock* */
    void *value;   /* Register* */
} HirPhiArgPair;

static inline int hir_c_phi_pair_cmp_by_block_id(const void *a, const void *b) {
    int ia = ((const HirBasicBlock *)((const HirPhiArgPair *)a)->key)->id;
    int ib = ((const HirBasicBlock *)((const HirPhiArgPair *)b)->key)->id;
    return (ia > ib) - (ia < ib);
}

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

/* Instr::getDominatingFrameState port (Batch 21). Reverse-walks the
 * containing block from the target towards the head, returning the
 * FrameState* of the nearest dominating Snapshot. Stops early if a
 * non-replayable Instr is encountered (the Snapshot would not survive
 * deopt past it). Returns NULL when block_ is unset, when no Snapshot
 * precedes the target, or when a non-replayable interposes.
 *
 * C++ const_reverse_iterator_to(*this) followed by ++it lands on the
 * instruction *before* this in the list — same as hir_bb_prev_instr,
 * which returns the prev or NULL when prev is the sentinel head. */
static inline void *hir_c_get_dominating_frame_state(const void *instr) {
    void *block = ((const HirInstrLayout *)instr)->block;
    if (block == NULL) return NULL;
    HirBasicBlock *bb = (HirBasicBlock *)block;
    void *cur = hir_bb_prev_instr(bb, (void *)instr);
    while (cur != NULL) {
        int op = hir_c_opcode(cur);
        if (op == HIR_OP_Snapshot) {
            return hir_c_snapshot_get_frame_state(cur);
        }
        if (!hir_c_is_replayable(cur)) {
            return NULL;
        }
        cur = hir_bb_prev_instr(bb, cur);
    }
    return NULL;
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

/* ---- Instr mutation: pure C ---- */

static inline void hir_c_link(void *instr, void *block) {
    hir_c_set_block(instr, block);
}

static inline void hir_c_unlink(void *instr, const HirBasicBlock *bb) {
    HirIntrusiveListNode *node =
        (HirIntrusiveListNode *)((char *)instr + bb->instrs_.node_member_offset_);
    node->prev_->next_ = node->next_;
    node->next_->prev_ = node->prev_;
    node->prev_ = node;
    node->next_ = node;
    hir_c_set_block(instr, NULL);
}

static inline void hir_c_insert_before_pure(void *instr, void *before, const HirBasicBlock *bb) {
    HirIntrusiveListNode *node =
        (HirIntrusiveListNode *)((char *)instr + bb->instrs_.node_member_offset_);
    HirIntrusiveListNode *b =
        (HirIntrusiveListNode *)((char *)before + bb->instrs_.node_member_offset_);
    HirIntrusiveListNode *prev = b->prev_;
    prev->next_ = node;
    node->prev_ = prev;
    node->next_ = b;
    b->prev_ = node;
    hir_c_link(instr, (void *)bb);
}

static inline void hir_c_insert_after_pure(void *instr, void *after, const HirBasicBlock *bb) {
    HirIntrusiveListNode *node =
        (HirIntrusiveListNode *)((char *)instr + bb->instrs_.node_member_offset_);
    HirIntrusiveListNode *a =
        (HirIntrusiveListNode *)((char *)after + bb->instrs_.node_member_offset_);
    HirIntrusiveListNode *next = a->next_;
    a->next_ = node;
    node->prev_ = a;
    node->next_ = next;
    next->prev_ = node;
    hir_c_link(instr, (void *)bb);
}

/* C++ bridge for Phi shell allocation, defined in hir.cpp. Used by
 * the Batch 25 add/remove predecessor flows so they can build new Phis
 * with the right operand count before applying sorted args. */
void *hir_make_phi_with_count_c(void *dst, size_t count);

/* Forward decl: hir_c_instr_replace_with is defined later in this header
 * (Batch 23 section). The Batch 25 add/remove flows below need it. */
static inline void hir_c_instr_replace_with(void *self, void *replacement);

/* Build sorted (key, value) parallel arrays for the post-add Phi
 * state. Returns 1 on success (caller must free *out_keys + *out_values),
 * 0 if old_pred is not a current predecessor of old_phi (no-op signal —
 * the C++ shim's collect-first-then-process pass already filters but
 * this guard keeps the C function safe to call directly). */
static inline int hir_c_phi_collect_add_args(void *old_phi,
                                             void *old_pred,
                                             void *new_pred,
                                             void ***out_keys,
                                             void ***out_values,
                                             size_t *out_n) {
    HirPhi *p = (HirPhi *)old_phi;
    size_t old_n = p->bb_count;

    int has_old = 0;
    for (size_t i = 0; i < old_n; i++) {
        if (p->bb_data[i] == old_pred) { has_old = 1; break; }
    }
    if (!has_old) {
        *out_keys = NULL;
        *out_values = NULL;
        *out_n = 0;
        return 0;
    }

    size_t new_n = old_n + 1;
    HirPhiArgPair *pairs =
        (HirPhiArgPair *)malloc(new_n * sizeof(HirPhiArgPair));
    size_t idx = 0;
    for (size_t i = 0; i < old_n; i++) {
        void *block = p->bb_data[i];
        void *reg = hir_c_get_operand(old_phi, i);
        pairs[idx].key = block;
        pairs[idx].value = reg;
        idx++;
        if (block == old_pred) {
            pairs[idx].key = new_pred;
            pairs[idx].value = reg;
            idx++;
        }
    }
    qsort(pairs, new_n, sizeof(HirPhiArgPair), hir_c_phi_pair_cmp_by_block_id);

    *out_keys = (void **)malloc(new_n * sizeof(void *));
    *out_values = (void **)malloc(new_n * sizeof(void *));
    for (size_t i = 0; i < new_n; i++) {
        (*out_keys)[i] = pairs[i].key;
        (*out_values)[i] = pairs[i].value;
    }
    free(pairs);
    *out_n = new_n;
    return 1;
}

/* Build sorted (key, value) parallel arrays for the post-remove Phi
 * state. Drops every entry whose key equals old_pred. Returns count
 * via *out_n; *out_keys / *out_values are NULL when count == 0
 * (zero-pred Phi corner case — caller must free otherwise). */
static inline void hir_c_phi_collect_remove_args(void *old_phi,
                                                 void *old_pred,
                                                 void ***out_keys,
                                                 void ***out_values,
                                                 size_t *out_n) {
    HirPhi *p = (HirPhi *)old_phi;
    size_t old_n = p->bb_count;

    size_t new_n = 0;
    for (size_t i = 0; i < old_n; i++) {
        if (p->bb_data[i] != old_pred) new_n++;
    }
    if (new_n == 0) {
        *out_keys = NULL;
        *out_values = NULL;
        *out_n = 0;
        return;
    }

    HirPhiArgPair *pairs =
        (HirPhiArgPair *)malloc(new_n * sizeof(HirPhiArgPair));
    size_t idx = 0;
    for (size_t i = 0; i < old_n; i++) {
        void *block = p->bb_data[i];
        if (block == old_pred) continue;
        pairs[idx].key = block;
        pairs[idx].value = hir_c_get_operand(old_phi, i);
        idx++;
    }
    qsort(pairs, new_n, sizeof(HirPhiArgPair), hir_c_phi_pair_cmp_by_block_id);

    *out_keys = (void **)malloc(new_n * sizeof(void *));
    *out_values = (void **)malloc(new_n * sizeof(void *));
    for (size_t i = 0; i < new_n; i++) {
        (*out_keys)[i] = pairs[i].key;
        (*out_values)[i] = pairs[i].value;
    }
    free(pairs);
    *out_n = new_n;
}

/* BasicBlock::addPhiPredecessor per-Phi step (Batch 25). Allocates a
 * new Phi with one extra operand slot, applies the sorted args via
 * hir_c_phi_apply_args, replaces old with new in the block, then
 * destroys old. No-op when old_pred is not a current predecessor. */
static inline void hir_c_phi_add_predecessor(void *old_phi,
                                             void *old_pred,
                                             void *new_pred) {
    void **keys = NULL, **values = NULL;
    size_t n = 0;
    if (!hir_c_phi_collect_add_args(old_phi, old_pred, new_pred,
                                    &keys, &values, &n)) {
        return;
    }
    void *new_phi = hir_make_phi_with_count_c(hir_c_output(old_phi), n);
    hir_c_phi_apply_args(new_phi, keys, values, n);
    free(keys);
    free(values);

    hir_c_instr_replace_with(old_phi, new_phi);
    hir_c_destroy_instr_impl(old_phi);
}

/* BasicBlock::removePhiPredecessor per-Phi step (Batch 25). Allocates
 * a new Phi with one fewer operand slot, applies the sorted args,
 * replaces old with new, destroys old. */
static inline void hir_c_phi_remove_predecessor(void *old_phi,
                                                void *old_pred) {
    void **keys = NULL, **values = NULL;
    size_t n = 0;
    hir_c_phi_collect_remove_args(old_phi, old_pred, &keys, &values, &n);

    void *new_phi = hir_make_phi_with_count_c(hir_c_output(old_phi), n);
    hir_c_phi_apply_args(new_phi, keys, values, n);
    free(keys);
    free(values);

    hir_c_instr_replace_with(old_phi, new_phi);
    hir_c_destroy_instr_impl(old_phi);
}

/* BasicBlock::fixupPhis per-Phi remap step (Batch 24). Walk the Phi's
 * basic_blocks_, swap any occurrence of old_pred with new_pred, then
 * re-apply via the Batch 20 sort+apply path so the post-fixup state
 * stays sorted by block id. Pure C; reuses HirPhiArgPair comparator
 * + hir_c_phi_apply_args. No-op when old_pred is not present. */
static inline void hir_c_phi_fixup_predecessor(void *phi,
                                               void *old_pred,
                                               void *new_pred) {
    HirPhi *p = (HirPhi *)phi;
    size_t n = p->bb_count;
    if (n == 0) return;

    HirPhiArgPair *pairs =
        (HirPhiArgPair *)malloc(n * sizeof(HirPhiArgPair));
    for (size_t i = 0; i < n; i++) {
        void *block = p->bb_data[i];
        if (block == old_pred) block = new_pred;
        pairs[i].key = block;
        pairs[i].value = hir_c_get_operand(phi, i);
    }
    qsort(pairs, n, sizeof(HirPhiArgPair), hir_c_phi_pair_cmp_by_block_id);

    void **keys = (void **)malloc(n * sizeof(void *));
    void **values = (void **)malloc(n * sizeof(void *));
    for (size_t i = 0; i < n; i++) {
        keys[i] = pairs[i].key;
        values[i] = pairs[i].value;
    }
    free(pairs);

    hir_c_phi_apply_args(phi, keys, values, n);

    free(keys);
    free(values);
}

/* ---- Instr lifecycle (Batch 23) ---- */

/* Instr::link port. Sets self.block_ to block; asserts self was
 * unlinked. JIT_CHECK_C (release-fatal) preserves the C++ JIT_CHECK
 * semantics — double-linking would silently overwrite block_ and
 * leave the IntrusiveList wiring pointing at the prior block. */
static inline void hir_c_instr_link(void *self, void *block) {
    JIT_CHECK_C(((HirInstrLayout *)self)->block == NULL,
                "Instr is already linked");
    hir_c_set_block(self, block);
}

/* Instr::unlink port. Asserts self is currently linked, then removes
 * from its block's IntrusiveList and sets block_=NULL via the
 * existing hir_c_unlink helper. */
static inline void hir_c_instr_unlink(void *self) {
    HirBasicBlock *bb = (HirBasicBlock *)((HirInstrLayout *)self)->block;
    JIT_CHECK_C(bb != NULL, "Instr isn't linked");
    hir_c_unlink(self, bb);
}

/* Instr::ReplaceWith port. Inserts replacement before self in self's
 * block, copies self's bytecode_offset into replacement, then unlinks
 * self. Net effect: replacement takes self's slot, self is removed. */
static inline void hir_c_instr_replace_with(void *self, void *replacement) {
    const HirBasicBlock *bb =
        (const HirBasicBlock *)((const HirInstrLayout *)self)->block;
    hir_c_insert_before_pure(replacement, self, bb);
    hir_c_set_bytecode_offset(replacement,
                              ((const HirInstrLayout *)self)->bytecode_offset);
    hir_c_unlink(self, bb);
}

/* Instr::ExpandInto port. Inserts each of `expansion[0..n)` after the
 * previous instr (starting from self), copying self's bytecode_offset
 * into each, then unlinks self. n=0 is just an unlink. */
static inline void hir_c_instr_expand_into(void *self,
                                           void **expansion,
                                           size_t n) {
    const HirBasicBlock *bb =
        (const HirBasicBlock *)((const HirInstrLayout *)self)->block;
    int32_t off = ((const HirInstrLayout *)self)->bytecode_offset;
    void *last = self;
    for (size_t i = 0; i < n; i++) {
        hir_c_insert_after_pure(expansion[i], last, bb);
        hir_c_set_bytecode_offset(expansion[i], off);
        last = expansion[i];
    }
    hir_c_unlink(self, bb);
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* JIT_HIR_BASIC_BLOCK_C_H */
