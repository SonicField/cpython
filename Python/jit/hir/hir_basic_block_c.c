/*
 * hir_basic_block_c.c -- C implementation of HIR BasicBlock operations
 *
 * Phase H1b: Pure C functions operating on HirBasicBlock struct.
 * Uses container_of pattern for intrusive list traversal.
 */

#include "cinderx/Jit/hir/hir_basic_block_c.h"

#include <assert.h>
#include <stddef.h>
#include <string.h>

/* ---- container_of: intrusive list node → Instr ---- */

static inline void *
node_to_instr(const HirBasicBlock *bb, HirIntrusiveListNode *node) {
    return (char *)node - bb->instrs_.node_member_offset_;
}

static inline HirIntrusiveListNode *
instr_to_node(const HirBasicBlock *bb, void *instr) {
    return (HirIntrusiveListNode *)((char *)instr + bb->instrs_.node_member_offset_);
}

/* ---- Sentinel helpers ---- */

static inline HirIntrusiveListNode *
sentinel(HirBasicBlock *bb) {
    return &bb->instrs_.root_;
}

static inline const HirIntrusiveListNode *
sentinel_const(const HirBasicBlock *bb) {
    return &bb->instrs_.root_;
}

/* ---- Accessors ---- */

int
hir_bb_id(const HirBasicBlock *bb) {
    return bb->id;
}

size_t
hir_bb_in_edges_count(const HirBasicBlock *bb) {
    return bb->in_edges_.count;
}

size_t
hir_bb_out_edges_count(const HirBasicBlock *bb) {
    return bb->out_edges_.count;
}

const HirEdge *
hir_bb_in_edge(const HirBasicBlock *bb, size_t i) {
    assert(i < bb->in_edges_.count);
    return bb->in_edges_.data[i];
}

const HirEdge *
hir_bb_out_edge(const HirBasicBlock *bb, size_t i) {
    assert(i < bb->out_edges_.count);
    return bb->out_edges_.data[i];
}

/* ---- Instruction list queries ---- */

int
hir_bb_empty(const HirBasicBlock *bb) {
    const HirIntrusiveListNode *s = sentinel_const(bb);
    return s->next_ == s;
}

void *
hir_bb_first_instr(const HirBasicBlock *bb) {
    const HirIntrusiveListNode *s = sentinel_const(bb);
    if (s->next_ == s) return NULL;
    return node_to_instr(bb, s->next_);
}

void *
hir_bb_last_instr(const HirBasicBlock *bb) {
    const HirIntrusiveListNode *s = sentinel_const(bb);
    if (s->prev_ == s) return NULL;
    return node_to_instr(bb, s->prev_);
}

void *
hir_bb_next_instr(const HirBasicBlock *bb, void *instr) {
    HirIntrusiveListNode *node = instr_to_node(bb, instr);
    HirIntrusiveListNode *next = node->next_;
    if (next == sentinel_const(bb)) return NULL;
    return node_to_instr(bb, next);
}

void *
hir_bb_prev_instr(const HirBasicBlock *bb, void *instr) {
    HirIntrusiveListNode *node = instr_to_node(bb, instr);
    HirIntrusiveListNode *prev = node->prev_;
    if (prev == sentinel_const(bb)) return NULL;
    return node_to_instr(bb, prev);
}

/* ---- Edge management (set_from / set_to) ---- */

void
hir_edge_set_from(HirEdge *edge, HirBasicBlock *new_from) {
    if (edge->from) {
        phx_edge_arr_erase(&((HirBasicBlock *)edge->from)->out_edges_,
                           (const HirEdge *)edge);
    }
    if (new_from) {
        phx_edge_arr_insert(&new_from->out_edges_, (const HirEdge *)edge);
    }
    edge->from = new_from;
}

void
hir_edge_set_to(HirEdge *edge, HirBasicBlock *new_to) {
    if (edge->to) {
        phx_edge_arr_erase(&((HirBasicBlock *)edge->to)->in_edges_,
                           (const HirEdge *)edge);
    }
    if (new_to) {
        phx_edge_arr_insert(&new_to->in_edges_, (const HirEdge *)edge);
    }
    edge->to = new_to;
}

void
hir_edge_destroy(HirEdge *edge) {
    hir_edge_set_from(edge, NULL);
    hir_edge_set_to(edge, NULL);
}

/* ---- retargetPreds ---- */

void
hir_bb_retarget_preds(HirBasicBlock *bb, HirBasicBlock *target) {
    /* Snapshot: set_to modifies in_edges_ (swap-and-pop erase), so we
     * can't iterate the live array while mutating it. */
    size_t n = bb->in_edges_.count;
    const HirEdge **snapshot = (const HirEdge **)alloca(n * sizeof(const HirEdge *));
    memcpy(snapshot, bb->in_edges_.data, n * sizeof(const HirEdge *));
    for (size_t i = 0; i < n; i++) {
        hir_edge_set_to((HirEdge *)snapshot[i], target);
    }
}

/* ---- Insert before / clear / GetTerminator ---- */

void
hir_bb_insert_before(HirBasicBlock *bb, void *instr, void *before) {
    /* Propagate bytecodeOffset from adjacent instruction if missing. */
    int32_t off = hir_c_bytecode_offset(instr);
    if (off == -1) {
        HirIntrusiveListNode *before_node = instr_to_node(bb, before);
        const HirIntrusiveListNode *s = sentinel_const(bb);
        if (before_node->prev_ != s) {
            void *prev_instr = node_to_instr(bb, before_node->prev_);
            hir_c_set_bytecode_offset(instr, hir_c_bytecode_offset(prev_instr));
        } else {
            hir_c_set_bytecode_offset(instr, hir_c_bytecode_offset(before));
        }
    }
    /* Intrusive list insert: instr goes before 'before'. */
    HirIntrusiveListNode *node = instr_to_node(bb, instr);
    HirIntrusiveListNode *b = instr_to_node(bb, before);
    HirIntrusiveListNode *prev = b->prev_;
    prev->next_ = node;
    node->prev_ = prev;
    node->next_ = b;
    b->prev_ = node;
}

void
hir_bb_clear(HirBasicBlock *bb) {
    while (!hir_bb_empty(bb)) {
        void *instr = hir_bb_pop_front_instr(bb);
        hir_c_destroy_instr_impl(instr);
    }
}

void *
hir_bb_get_terminator(const HirBasicBlock *bb) {
    return hir_bb_last_instr(bb);
}

void *
hir_bb_entry_snapshot(const HirBasicBlock *bb) {
    void *instr = hir_bb_first_instr(bb);
    while (instr) {
        if (hir_c_is_phi(instr)) {
            instr = hir_bb_next_instr(bb, instr);
            continue;
        }
        if (hir_c_is_snapshot(instr)) {
            return instr;
        }
        return NULL;
    }
    return NULL;
}

int
hir_bb_is_trampoline(const HirBasicBlock *bb) {
    void *instr = hir_bb_first_instr(bb);
    while (instr) {
        if (hir_c_is_branch(instr)) {
            HirBasicBlock *succ =
                (HirBasicBlock *)hir_c_successor(instr, 0);
            if (succ == bb) return 0;
            if (hir_bb_empty(succ)) return 1;
            return !hir_c_is_phi(hir_bb_first_instr(succ));
        }
        if (hir_c_is_snapshot(instr)) {
            instr = hir_bb_next_instr(bb, instr);
            continue;
        }
        return 0;
    }
    return 0;
}

/* ---- Instruction list mutation ---- */

void *
hir_bb_append_instr(HirBasicBlock *bb, void *instr) {
    HirIntrusiveListNode *node = instr_to_node(bb, instr);
    HirIntrusiveListNode *s = sentinel(bb);
    /* Insert before sentinel = append to end */
    HirIntrusiveListNode *prev = s->prev_;
    prev->next_ = node;
    node->prev_ = prev;
    node->next_ = s;
    s->prev_ = node;
    return instr;
}

void
hir_bb_push_front_instr(HirBasicBlock *bb, void *instr) {
    HirIntrusiveListNode *node = instr_to_node(bb, instr);
    HirIntrusiveListNode *s = sentinel(bb);
    /* Insert after sentinel = push to front */
    HirIntrusiveListNode *next = s->next_;
    s->next_ = node;
    node->prev_ = s;
    node->next_ = next;
    next->prev_ = node;
}

void *
hir_bb_pop_front_instr(HirBasicBlock *bb) {
    HirIntrusiveListNode *s = sentinel(bb);
    if (s->next_ == s) return NULL;
    HirIntrusiveListNode *first = s->next_;
    /* Unlink first */
    s->next_ = first->next_;
    first->next_->prev_ = s;
    first->prev_ = first;
    first->next_ = first;
    return node_to_instr(bb, first);
}
