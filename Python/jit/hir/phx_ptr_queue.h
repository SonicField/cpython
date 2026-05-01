/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * PhxPtrQueue — generic void* FIFO queue (linear array). Phase 4.X-full
 * X2a substrate per supervisor 03:06:27Z (b) PROMOTE disposition +
 * theologian docs/tier7-phase4x-full-entry-preanalysis-2026-05-01.md §2.4.
 *
 * Promoted from licm_c.c file-scoped PhxQueue (linear array, head/tail
 * grow unbounded) to shared header. Bonus: forward stance #4 discharge
 * — original PyMem_RawRealloc had silent-skip on OOM (W2-A1 class
 * latent violation); promoted version uses JIT_CHECK_C(new_data != NULL)
 * loud-fail per X1a 0bac3bc31d precedent.
 *
 * Use-class: per-call FIFO worklist (e.g. LICM loop-body BFS, translate()
 * dispatch queue). Bounded by per-call lifetime; unbounded growth is
 * acceptable since head/tail reset between calls via destroy+init.
 *
 * Allocator switched from PyMem_RawMalloc/Realloc/Free (PhxQueue original)
 * to plain malloc/realloc/free — matches phx_ptr_array.h sibling pattern;
 * shared header should not require Python.h-class allocator transitively.
 *
 * NOT a circular buffer: head and tail grow monotonically; reset on
 * init/destroy cycle. For long-lived queues with high churn, prefer
 * a circular buffer (currently no consumer needs that semantic).
 */
#pragma once

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "cinderx/Common/jit_log_c.h"  /* JIT_CHECK_C loud-fail on realloc OOM */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PhxPtrQueue {
    void **data;
    size_t head;
    size_t tail;
    size_t capacity; /* 0 ⇒ data NULL (lazy init) */
} PhxPtrQueue;

#define PHX_PTR_QUEUE_INITIAL_CAP 16u

static inline void phx_ptr_queue_init(PhxPtrQueue *q) {
    q->data = NULL;
    q->head = 0;
    q->tail = 0;
    q->capacity = 0;
}

static inline void phx_ptr_queue_destroy(PhxPtrQueue *q) {
    if (q->data) {
        free(q->data);
        q->data = NULL;
    }
    q->head = 0;
    q->tail = 0;
    q->capacity = 0;
}

static inline int phx_ptr_queue_empty(const PhxPtrQueue *q) {
    return q->head == q->tail;
}

static inline void phx_ptr_queue_push(PhxPtrQueue *q, void *val) {
    if (q->tail >= q->capacity) {
        size_t new_cap = q->capacity ? q->capacity * 2u
                                     : PHX_PTR_QUEUE_INITIAL_CAP;
        void **new_data = (void **)realloc(q->data, new_cap * sizeof(void *));
        JIT_CHECK_C(new_data != NULL,
                    "phx_ptr_queue_push realloc failed (new_cap=%zu)",
                    new_cap);
        q->data = new_data;
        q->capacity = new_cap;
    }
    q->data[q->tail++] = val;
}

static inline void *phx_ptr_queue_pop(PhxPtrQueue *q) {
    return q->data[q->head++];
}

#ifdef __cplusplus
} /* extern "C" */
#endif
