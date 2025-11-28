// Copyright (c) Meta Platforms, Inc. and affiliates.
// Ported to CPython by Alex Turner

#ifndef Py_INTERNAL_WS_DEQUE_H
#define Py_INTERNAL_WS_DEQUE_H

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include "pycore_pyatomic_ft_wrappers.h"  // _Py_atomic_*
#include "pyatomic.h"                      // Atomic operations
#include <stdint.h>                        // uintptr_t
#include <stdlib.h>                        // calloc, free
#include <assert.h>                        // assert

// This implements the Chase-Lev work stealing deque first described in
//
//   "Dynamic Circular Work-Stealing Deque"
//   (https://dl.acm.org/doi/10.1145/1073970.1073974)
//
// and later specified using C11 atomics in
//
//   "Correct and Efficient Work-Stealing for Weak Memory Models"
//   (https://dl.acm.org/doi/10.1145/2442516.2442524)
//
// This implementation uses CPython's atomic abstractions (pycore_atomic.h)
// instead of raw C11 atomics for portability.

// WSArray: Circular buffer backing storage for the deque
// Arrays are linked into a singly linked list as they grow.
typedef struct _PyWSArray {
    struct _PyWSArray *next;
    size_t size;
    uintptr_t buf[];  // Flexible array member - actual size determined at allocation
} _PyWSArray;

static inline _PyWSArray *
_PyWSArray_New(size_t size)
{
    // size must be a power of two > 0
    assert(size > 0 && (size & (size - 1)) == 0);

    _PyWSArray *arr = (struct _PyWSArray *)calloc(
        1, sizeof(_PyWSArray) + sizeof(uintptr_t) * size);
    if (arr == NULL) {
        return NULL;
    }
    arr->size = size;
    arr->next = NULL;
    return arr;
}

static inline void
_PyWSArray_Destroy(_PyWSArray *arr)
{
    if (arr == NULL) {
        return;
    }
    if (arr->next != NULL) {
        _PyWSArray_Destroy(arr->next);
        arr->next = NULL;
    }
    free(arr);
}

static inline void *
_PyWSArray_Get(_PyWSArray *arr, size_t idx)
{
    // Use relaxed load - synchronization handled by deque operations
    uintptr_t val = _Py_atomic_load_uintptr_relaxed(&arr->buf[idx & (arr->size - 1)]);
    return (void *)val;
}

static inline void
_PyWSArray_Put(_PyWSArray *arr, size_t idx, void *obj)
{
    // Use relaxed store - synchronization handled by deque operations
    _Py_atomic_store_uintptr_relaxed(&arr->buf[idx & (arr->size - 1)], (uintptr_t)obj);
}

static inline _PyWSArray *
_PyWSArray_Grow(_PyWSArray *arr, size_t top, size_t bot)
{
    size_t new_size = arr->size << 1;
    assert(new_size > arr->size);

    _PyWSArray *new_arr = _PyWSArray_New(new_size);
    if (new_arr == NULL) {
        return NULL;
    }
    new_arr->next = arr;

    // Copy elements from old array to new array
    for (size_t i = top; i < bot; i++) {
        PyObject *obj = (PyObject *)_PyWSArray_Get(arr, i);
        _PyWSArray_Put(new_arr, i, obj);
    }

    return new_arr;
}

// Initial size for work-stealing deque arrays
static const size_t _Py_WSDEQUE_INITIAL_ARRAY_SIZE = 1 << 12;  // 4096 elements

// Cache line size for padding to prevent false sharing
// Ideally this would be determined based on architecture, but hardcoded for now.
#define _Py_CACHELINE_SIZE 64

// _PyWSDeque: Lock-free work-stealing deque
//
// The deque has two ends: top and bottom.
// - The owner thread pushes and pops from the bottom (LIFO)
// - Worker threads steal from the top (FIFO relative to push)
//
// Cache-line padding prevents false sharing between top and bot.
typedef struct {
    // Top index - accessed by both owner and workers (steal)
    // Padded to prevent false sharing with bot
    union {
        size_t top;
        uint8_t top_padding[_Py_CACHELINE_SIZE];
    };

    // Bottom index - primarily accessed by owner
    // Padded to prevent false sharing with arr
    union {
        size_t bot;
        uint8_t bot_padding[_Py_CACHELINE_SIZE];
    };

    // Pointer to current array (can be replaced during resize)
    _PyWSArray *arr;

    // Number of times the array has been resized (for testing/debugging)
    int num_resizes;
} _PyWSDeque;

static inline void
_PyWSDeque_Init(_PyWSDeque *deque)
{
    _PyWSArray *arr = _PyWSArray_New(_Py_WSDEQUE_INITIAL_ARRAY_SIZE);
    // TODO: Handle allocation failure
    _Py_atomic_store_ptr_relaxed(&deque->arr, arr);

    // This fixes a small bug in the paper. When these are initialized to 0,
    // attempting to `take` on a newly empty deque will succeed; subtracting 1
    // from `bot` will cause it to wrap, and the check for a non-empty deque,
    // `top <= bot`, will succeed. Initializing these both to 1 ensures that
    // bot will not wrap.
    _Py_atomic_store_ssize_relaxed((Py_ssize_t *)&deque->top, 1);
    _Py_atomic_store_ssize_relaxed((Py_ssize_t *)&deque->bot, 1);
    _Py_atomic_store_int_relaxed(&deque->num_resizes, 0);
}

static inline void
_PyWSDeque_Fini(_PyWSDeque *deque)
{
    _PyWSArray *arr = (_PyWSArray *)_Py_atomic_load_ptr(&deque->arr);
    _PyWSArray_Destroy(arr);
}

// Take: Pop from bottom (owner only - LIFO)
// Returns NULL if deque is empty or if lost race with steal
static inline PyObject *
_PyWSDeque_Take(_PyWSDeque *deque)
{
    size_t bot = _Py_atomic_load_ssize_relaxed((Py_ssize_t *)&deque->bot) - 1;
    _PyWSArray *arr = (_PyWSArray *)_Py_atomic_load_ptr(&deque->arr);
    _Py_atomic_store_ssize_relaxed((Py_ssize_t *)&deque->bot, bot);

    // Ensure bot write is visible before loading top
    _Py_atomic_fence_seq_cst();

    size_t top = _Py_atomic_load_ssize_relaxed((Py_ssize_t *)&deque->top);

    PyObject *res = NULL;
    if (top <= bot) {
        // Not empty
        res = (PyObject *)_PyWSArray_Get(arr, bot);
        if (top == bot) {
            // One element in the queue - need to compete with steal
            size_t expected_top = top;
            if (!_Py_atomic_compare_exchange_ssize(
                    (Py_ssize_t *)&deque->top,
                    (Py_ssize_t *)&expected_top,
                    top + 1)) {
                // Failed race with another thread stealing from us
                res = NULL;
            }
            _Py_atomic_store_ssize_relaxed((Py_ssize_t *)&deque->bot, bot + 1);
        }
    }
    else {
        // Empty - restore bot
        _Py_atomic_store_ssize_relaxed((Py_ssize_t *)&deque->bot, bot + 1);
    }

    return res;
}

// Push: Add to bottom (owner only)
static inline void
_PyWSDeque_Push(_PyWSDeque *deque, void *obj)
{
    size_t bot = _Py_atomic_load_ssize_relaxed((Py_ssize_t *)&deque->bot);
    size_t top = _Py_atomic_load_ssize_acquire((Py_ssize_t *)&deque->top);
    _PyWSArray *arr = (_PyWSArray *)_Py_atomic_load_ptr(&deque->arr);

    assert(bot >= top);

    if (bot - top > arr->size - 1) {
        // Full, need to grow the underlying array.
        //
        // NB: This differs from the paper. The paper's implementation
        // is specified as the following pseudocode,
        //
        //     resize(q);
        //     a = load_explicit(&q->array, relaxed);
        //
        // however, no implementation is provided for `resize`. Using a relaxed
        // store here should be correct: all other threads will (eventually)
        // see the update atomically and we don't have to worry about another
        // thread growing the array concurrently as only the thread that owns
        // the deque is allowed to do so.
        _PyWSArray *new_arr = _PyWSArray_Grow(arr, top, bot);
        // TODO: Handle allocation failure
        _Py_atomic_store_ptr_relaxed(&deque->arr, new_arr);
        arr = (_PyWSArray *)_Py_atomic_load_ptr(&deque->arr);
        _Py_atomic_add_int(&deque->num_resizes, 1);
    }

    _PyWSArray_Put(arr, bot, obj);

    // Ensure the element write is visible before incrementing bot
    _Py_atomic_fence_release();

    _Py_atomic_store_ssize_relaxed((Py_ssize_t *)&deque->bot, bot + 1);
}

// Steal: Pop from top (workers - FIFO)
// Returns NULL if deque is empty or if lost race
static inline void *
_PyWSDeque_Steal(_PyWSDeque *deque)
{
    while (1) {
        size_t top = _Py_atomic_load_ssize_acquire((Py_ssize_t *)&deque->top);

        // Ensure top is loaded before bot
        _Py_atomic_fence_seq_cst();

        size_t bot = _Py_atomic_load_ssize_acquire((Py_ssize_t *)&deque->bot);
        void *res = NULL;

        if (top < bot) {
            // Not empty
            // Note: Using acquire instead of consume (consume not available in pyatomic.h)
            _PyWSArray *arr = (_PyWSArray *)_Py_atomic_load_ptr_acquire(&deque->arr);
            res = _PyWSArray_Get(arr, top);

            // Try to increment top
            size_t expected_top = top;
            if (!_Py_atomic_compare_exchange_ssize(
                    (Py_ssize_t *)&deque->top,
                    (Py_ssize_t *)&expected_top,
                    top + 1)) {
                // Lost race - retry
                continue;
            }
        }
        return res;
    }
}

// Get number of resizes (for testing/debugging)
static inline int
_PyWSDeque_GetNumResizes(_PyWSDeque *deque)
{
    return _Py_atomic_load_int_relaxed(&deque->num_resizes);
}

// Get approximate size (may be stale due to concurrent operations)
static inline size_t
_PyWSDeque_Size(_PyWSDeque *deque)
{
    size_t bot = _Py_atomic_load_ssize_relaxed((Py_ssize_t *)&deque->bot);
    size_t top = _Py_atomic_load_ssize_acquire((Py_ssize_t *)&deque->top);
    return bot < top ? 0 : bot - top;
}

#ifdef __cplusplus
}
#endif

#endif // Py_INTERNAL_WS_DEQUE_H
