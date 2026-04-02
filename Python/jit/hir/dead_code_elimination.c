/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Dead code elimination pass — removes instructions whose outputs are
 * not used by any instruction with side effects or control flow.
 */

#include "cinderx/Jit/hir/dead_code_elimination_c.h"
#include "cinderx/Jit/hir/hir_c_api.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ---- Simple pointer hash set (open addressing) ---- */

typedef struct {
    void **buckets;
    size_t cap; /* always a power of 2 */
    size_t len;
} PtrSet;

static void ptrset_init(PtrSet *s, size_t cap) {
    s->cap = cap;
    s->len = 0;
    s->buckets = (void **)calloc(cap, sizeof(void *));
}

static int ptrset_contains(const PtrSet *s, void *p) {
    if (!s->buckets) return 0;
    size_t mask = s->cap - 1;
    size_t idx = ((uintptr_t)p >> 3) & mask;
    for (;;) {
        if (s->buckets[idx] == NULL) return 0;
        if (s->buckets[idx] == p) return 1;
        idx = (idx + 1) & mask;
    }
}

/* Returns 1 if newly inserted, 0 if already present */
static int ptrset_insert(PtrSet *s, void *p) {
    /* Rehash at 50% load */
    if (s->len * 2 >= s->cap) {
        size_t new_cap = s->cap * 2;
        void **new_buckets = (void **)calloc(new_cap, sizeof(void *));
        size_t mask = new_cap - 1;
        for (size_t i = 0; i < s->cap; i++) {
            if (s->buckets[i]) {
                size_t idx = ((uintptr_t)s->buckets[i] >> 3) & mask;
                while (new_buckets[idx]) idx = (idx + 1) & mask;
                new_buckets[idx] = s->buckets[i];
            }
        }
        free(s->buckets);
        s->buckets = new_buckets;
        s->cap = new_cap;
    }
    size_t mask = s->cap - 1;
    size_t idx = ((uintptr_t)p >> 3) & mask;
    for (;;) {
        if (s->buckets[idx] == NULL) {
            s->buckets[idx] = p;
            s->len++;
            return 1;
        }
        if (s->buckets[idx] == p) return 0;
        idx = (idx + 1) & mask;
    }
}

static void ptrset_free(PtrSet *s) {
    free(s->buckets);
}

/* ---- Simple FIFO worklist ---- */

typedef struct {
    void **items;
    size_t head;
    size_t len;
    size_t cap;
} Worklist;

static void wl_init(Worklist *w, size_t cap) {
    w->items = (void **)malloc(cap * sizeof(void *));
    w->head = 0;
    w->len = 0;
    w->cap = cap;
}

static int wl_empty(const Worklist *w) { return w->len == 0; }

static void wl_push(Worklist *w, void *p) {
    if (w->len >= w->cap) {
        size_t new_cap = w->cap * 2;
        void **new_items = (void **)malloc(new_cap * sizeof(void *));
        /* Copy from head to end, then wrap */
        for (size_t i = 0; i < w->len; i++) {
            new_items[i] = w->items[(w->head + i) % w->cap];
        }
        free(w->items);
        w->items = new_items;
        w->head = 0;
        w->cap = new_cap;
    }
    w->items[(w->head + w->len) % w->cap] = p;
    w->len++;
}

static void *wl_pop(Worklist *w) {
    void *p = w->items[w->head];
    w->head = (w->head + 1) % w->cap;
    w->len--;
    return p;
}

static void wl_free(Worklist *w) { free(w->items); }

/* ---- isUseful predicate ---- */

static int is_useful(HirInstr instr) {
    if (hir_instr_is_terminator(instr)) return 1;
    if (hir_instr_is_snapshot(instr)) return 1;
    if (hir_instr_has_deopt_base(instr) && !hir_instr_is_primitive_box(instr))
        return 1;
    if (!hir_instr_is_phi(instr) &&
        hir_memory_effects_may_store(instr) != HIR_ACLS_EMPTY)
        return 1;
    return 0;
}

/* ---- visitUses callback for DCE ---- */

typedef struct {
    PtrSet *live_set;
    Worklist *worklist;
} DceCtx;

static int dce_visit_cb(HirRegister *reg_slot, void *ctx) {
    DceCtx *dce = (DceCtx *)ctx;
    HirRegister reg = *reg_slot;
    HirInstr def = hir_reg_instr(reg);
    if (!ptrset_contains(dce->live_set, def)) {
        wl_push(dce->worklist, def);
    }
    return 1; /* continue */
}

/* ---- DCE pass ---- */

void hir_dead_code_elimination_run(HirFunction func) {
    HirCFG cfg = hir_func_cfg(func);

    /* Phase 1: seed worklist with useful instructions */
    Worklist worklist;
    wl_init(&worklist, 128);

    HirBasicBlock block = hir_cfg_blocks_first(cfg);
    while (block != NULL) {
        HirInstr instr = hir_block_first(block);
        while (instr != NULL) {
            if (is_useful(instr)) {
                wl_push(&worklist, instr);
            }
            instr = hir_block_next(block, instr);
        }
        block = hir_cfg_blocks_next(cfg, block);
    }

    /* Phase 2: transitive closure — mark reachable instructions as live */
    PtrSet live_set;
    ptrset_init(&live_set, 256);

    DceCtx dce_ctx = { &live_set, &worklist };

    while (!wl_empty(&worklist)) {
        HirInstr live_op = (HirInstr)wl_pop(&worklist);
        if (ptrset_insert(&live_set, live_op)) {
            hir_instr_visit_uses(live_op, dce_visit_cb, &dce_ctx);
        }
    }

    /* Phase 3: delete dead instructions */
    block = hir_cfg_blocks_first(cfg);
    while (block != NULL) {
        HirBasicBlock next_block = hir_cfg_blocks_next(cfg, block);
        HirInstr instr = hir_block_first(block);
        while (instr != NULL) {
            HirInstr next = hir_block_next(block, instr);
            if (!ptrset_contains(&live_set, instr)) {
                hir_instr_unlink(instr);
                hir_instr_delete(instr);
            }
            instr = next;
        }
        block = next_block;
    }

    ptrset_free(&live_set);
    wl_free(&worklist);
}
