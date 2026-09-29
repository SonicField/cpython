// Chase-Lev work-stealing deque for parallel GC.

#ifndef Py_INTERNAL_WS_DEQUE_H
#define Py_INTERNAL_WS_DEQUE_H

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include "pycore_pyatomic_ft_wrappers.h"  // _Py_atomic_*
#include "pyatomic.h"                     // Atomic operations
#include "pycore_pymem.h"                 // PyMem_RawCalloc, PyMem_RawFree
#include <assert.h>
#include <stdint.h>                        // uintptr_t

// Prefetch primitives.  These are local to avoid a circular include.
#if defined(__GNUC__) || defined(__clang__)
#  define _PyWS_PREFETCH(ptr) __builtin_prefetch((ptr), 0, 0)
#elif defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
#  include <intrin.h>
#  define _PyWS_PREFETCH(ptr) _mm_prefetch((const char *)(ptr), 0)
#else
#  define _PyWS_PREFETCH(ptr) ((void)(ptr))
#endif

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

// Arrays form a singly linked list as the deque grows.
typedef struct _PyWSArray {
    struct _PyWSArray *next;
    size_t size;
    uintptr_t buf[];
} _PyWSArray;

static inline _PyWSArray *
_PyWSArray_New(size_t size)
{
    assert(size > 0 && (size & (size - 1)) == 0);

    _PyWSArray *arr = PyMem_RawCalloc(
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
    PyMem_RawFree(arr);
}

static inline void *
_PyWSArray_Get(_PyWSArray *arr, size_t idx)
{
    uintptr_t val = _Py_atomic_load_uintptr_relaxed(
        &arr->buf[idx & (arr->size - 1)]);
    return (void *)val;
}

static inline void
_PyWSArray_Put(_PyWSArray *arr, size_t idx, void *obj)
{
    _Py_atomic_store_uintptr_relaxed(
        &arr->buf[idx & (arr->size - 1)], (uintptr_t)obj);
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

    for (size_t i = top; i < bot; i++) {
        PyObject *obj = (PyObject *)_PyWSArray_Get(arr, i);
        _PyWSArray_Put(new_arr, i, obj);
    }

    return new_arr;
}

static const size_t _Py_WSDEQUE_INITIAL_ARRAY_SIZE = 1 << 12;

#define _Py_WSDEQUE_CACHELINE_SIZE 64

// The owner thread pushes and pops from the bottom (LIFO).  Other threads
// steal from the top (FIFO relative to pushes).
typedef struct {
    // Cache-line padding separates the thief-shared top from the owner-local
    // bottom.
    union {
        size_t top;
        uint8_t top_padding[_Py_WSDEQUE_CACHELINE_SIZE];
    };

    union {
        size_t bot;
        uint8_t bot_padding[_Py_WSDEQUE_CACHELINE_SIZE];
    };

    _PyWSArray *arr;
    int num_resizes;
} _PyWSDeque;

static inline int
_PyWSDeque_Init(_PyWSDeque *deque)
{
    _PyWSArray *arr = _PyWSArray_New(_Py_WSDEQUE_INITIAL_ARRAY_SIZE);
    if (arr == NULL) {
        return -1;
    }
    _Py_atomic_store_ptr_relaxed(&deque->arr, arr);

    // This fixes a small bug in the paper. When these are initialized to 0,
    // attempting to `take` on a newly empty deque will succeed; subtracting 1
    // from `bot` will cause it to wrap, and the check for a non-empty deque,
    // `top <= bot`, will succeed. Initializing these both to 1 ensures that
    // bot will not wrap.
    _Py_atomic_store_ssize_relaxed((Py_ssize_t *)&deque->top, 1);
    _Py_atomic_store_ssize_relaxed((Py_ssize_t *)&deque->bot, 1);
    _Py_atomic_store_int_relaxed(&deque->num_resizes, 0);
    assert(deque->top == 1 && deque->bot == 1);
    return 0;
}

static inline void
_PyWSDeque_Fini(_PyWSDeque *deque)
{
    _PyWSArray *arr = (_PyWSArray *)_Py_atomic_load_ptr(&deque->arr);
    _PyWSArray_Destroy(arr);
}

// Pop from the bottom.  This may only be called by the owner.  Return NULL if
// the deque is empty or a thief wins the race for the final item.
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
        res = (PyObject *)_PyWSArray_Get(arr, bot);
        if (top == bot) {
            // Compete with thieves for the final item.
            size_t expected_top = top;
            if (!_Py_atomic_compare_exchange_ssize(
                    (Py_ssize_t *)&deque->top,
                    (Py_ssize_t *)&expected_top,
                    top + 1)) {
                res = NULL;
            }
            _Py_atomic_store_ssize_relaxed((Py_ssize_t *)&deque->bot, bot + 1);
        }
    }
    else {
        _Py_atomic_store_ssize_relaxed((Py_ssize_t *)&deque->bot, bot + 1);
    }

    return res;
}

// Push to the bottom.  This may only be called by the owner.
static inline int
_PyWSDeque_Push(_PyWSDeque *deque, void *obj)
{
    size_t bot = _Py_atomic_load_ssize_relaxed((Py_ssize_t *)&deque->bot);
    size_t top = _Py_atomic_load_ssize_acquire((Py_ssize_t *)&deque->top);
    _PyWSArray *arr = (_PyWSArray *)_Py_atomic_load_ptr(&deque->arr);

    assert(bot >= top);

    if (bot - top > arr->size - 1) {
        // Unlike the paper, keep the newly allocated array directly.
        // The array pointer must use release semantics so that all writes to
        // the new array (copying elements in _PyWSArray_Grow) are visible
        // before any thief can see the new pointer via acquire load in Steal.
        _PyWSArray *new_arr = _PyWSArray_Grow(arr, top, bot);
        if (new_arr == NULL) {
            return -1;
        }
        _Py_atomic_store_ptr_release(&deque->arr, new_arr);
        arr = (_PyWSArray *)_Py_atomic_load_ptr(&deque->arr);
        _Py_atomic_add_int(&deque->num_resizes, 1);
    }

    _PyWSArray_Put(arr, bot, obj);

    // Ensure the element write is visible before incrementing bot
    _Py_atomic_fence_release();

    _Py_atomic_store_ssize_relaxed((Py_ssize_t *)&deque->bot, bot + 1);
    return 0;
}

// Discard queued entries after all owners and thieves have stopped.
static inline void
_PyWSDeque_Reset(_PyWSDeque *deque)
{
    _Py_atomic_store_ssize_relaxed((Py_ssize_t *)&deque->top, 1);
    _Py_atomic_store_ssize_relaxed((Py_ssize_t *)&deque->bot, 1);
}

// Pop from the top.  This may be called by any thief.  Return NULL if the
// deque is empty.
static inline void *
_PyWSDeque_Steal(_PyWSDeque *deque)
{
    _PyWS_PREFETCH(&deque->arr);

    while (1) {
        size_t top = _Py_atomic_load_ssize_acquire((Py_ssize_t *)&deque->top);

        // Ensure top is loaded before bot
        _Py_atomic_fence_seq_cst();

        size_t bot = _Py_atomic_load_ssize_acquire((Py_ssize_t *)&deque->bot);
        void *res = NULL;

        if (top < bot) {
            // pyatomic.h does not provide consume loads.
            _PyWSArray *arr = (_PyWSArray *)_Py_atomic_load_ptr_acquire(
                &deque->arr);

            _PyWS_PREFETCH(&arr->buf[top & (arr->size - 1)]);

            res = _PyWSArray_Get(arr, top);

            size_t expected_top = top;
            if (!_Py_atomic_compare_exchange_ssize(
                    (Py_ssize_t *)&deque->top,
                    (Py_ssize_t *)&expected_top,
                    top + 1)) {
                continue;
            }
        }
        return res;
    }
}

static inline int
_PyWSDeque_GetNumResizes(_PyWSDeque *deque)
{
    return _Py_atomic_load_int_relaxed(&deque->num_resizes);
}

// This is only an estimate when other threads are stealing.
static inline size_t
_PyWSDeque_Size(_PyWSDeque *deque)
{
    size_t bot = _Py_atomic_load_ssize_relaxed((Py_ssize_t *)&deque->bot);
    size_t top = _Py_atomic_load_ssize_acquire((Py_ssize_t *)&deque->top);
    return bot < top ? 0 : bot - top;
}

// Thread-local work buffer.  It amortizes deque synchronization and uses LIFO
// order for cache locality.

#define _PyGC_LOCAL_BUFFER_SIZE 1024

typedef struct {
    PyObject *items[_PyGC_LOCAL_BUFFER_SIZE];
    size_t count;
} _PyGCLocalBuffer;

static inline int
_PyGCLocalBuffer_IsEmpty(_PyGCLocalBuffer *buf)
{
    return buf->count == 0;
}

static inline int
_PyGCLocalBuffer_IsFull(_PyGCLocalBuffer *buf)
{
    return buf->count >= _PyGC_LOCAL_BUFFER_SIZE;
}

// The caller must ensure that the buffer is not full.
static inline void
_PyGCLocalBuffer_Push(_PyGCLocalBuffer *buf, PyObject *obj)
{
    assert(buf->count < _PyGC_LOCAL_BUFFER_SIZE);
    buf->items[buf->count++] = obj;
}

// The caller must ensure that the buffer is not empty.
static inline PyObject *
_PyGCLocalBuffer_Pop(_PyGCLocalBuffer *buf)
{
    assert(buf->count > 0);
    return buf->items[--buf->count];
}

static inline void
_PyGCLocalBuffer_Reset(_PyGCLocalBuffer *buf)
{
    buf->count = 0;
}

#define _PyGCLocalBuffer_Init _PyGCLocalBuffer_Reset

// Pull up to half a local buffer from the owner's deque.
static inline size_t
_PyGC_RefillLocalFromDeque(_PyGCLocalBuffer *local, _PyWSDeque *deque)
{
    const size_t max_pull = _PyGC_LOCAL_BUFFER_SIZE / 2;
    size_t pulled = 0;

    while (pulled < max_pull && !_PyGCLocalBuffer_IsFull(local)) {
        PyObject *obj = _PyWSDeque_Take(deque);
        if (obj == NULL) {
            break;
        }
        _PyGCLocalBuffer_Push(local, obj);
        pulled++;
    }
    return pulled;
}

// Expose half of a full local buffer to thieves.
static inline int
_PyGC_OverflowFlush(_PyGCLocalBuffer *local, _PyWSDeque *deque)
{
    assert(local->count >= _PyGC_LOCAL_BUFFER_SIZE / 2);
    size_t flush_count = _PyGC_LOCAL_BUFFER_SIZE / 2;
    for (size_t i = 0; i < flush_count; i++) {
        PyObject *obj = _PyGCLocalBuffer_Pop(local);
        if (_PyWSDeque_Push(deque, obj) < 0) {
            return -1;
        }
    }
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif  // Py_INTERNAL_WS_DEQUE_H
