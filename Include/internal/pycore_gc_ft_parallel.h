// Parallel garbage collector for the free-threaded build.

#ifndef Py_INTERNAL_GC_FT_PARALLEL_H
#define Py_INTERNAL_GC_FT_PARALLEL_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

#if defined(Py_GIL_DISABLED) && defined(Py_PARALLEL_GC)

#include "pycore_gc.h"           // _PyGC_BITS_*
#include "pycore_gc_barrier.h"   // checked mutex and condition operations
#include "pycore_ws_deque.h"     // _PyWSDeque
#include "pycore_pythread.h"

// Return true if this call atomically changed the bit from one to zero.
static inline int
_PyGC_TryClearBit(PyObject *op, uint8_t bit)
{
    uint8_t old_bits = _Py_atomic_and_uint8(&op->ob_gc_bits, ~bit);
    return (old_bits & bit) != 0;
}

static inline int
_PyGC_IsUnreachable(PyObject *op)
{
    uint8_t bits = _Py_atomic_load_uint8_relaxed(&op->ob_gc_bits);
    return (bits & _PyGC_BITS_UNREACHABLE) != 0;
}

// Return true if this call changed the object from unreachable to reachable.
static inline int
_PyGC_TryMarkReachable(PyObject *op)
{
    assert(op != NULL);
    if (!(_Py_atomic_load_uint8_relaxed(&op->ob_gc_bits) &
          _PyGC_BITS_UNREACHABLE))
    {
        return 0;
    }
    return _PyGC_TryClearBit(op, _PyGC_BITS_UNREACHABLE);
}

struct _PyGCThreadPool;
struct _PyGCPageBucket;
typedef struct _PyGCPoolWorkerArgs _PyGCPoolWorkerArgs;
struct mi_page_s;  // mi_page_t from mimalloc
typedef struct mi_page_s mi_page_t;

typedef struct _PyGCWorkerState {
    _PyWSDeque deque;

    // Per-worker condition variable for targeted wakeup.
    PyMUTEX_T wake_mutex;
    PyCOND_T wake_cond;
    int wake_flag;  // 0=sleeping, 1=wake up
} _PyGCWorkerState;

typedef struct {
    struct _PyGCPageBucket *buckets;  // Page buckets (not owned)
    int skip_deferred;

    int error_flag;    // Set if any worker encounters an error (use atomics)
    Py_ssize_t outstanding;
    int active_workers;

} _PyGCWorkDescriptor;

// Worker pool for parallel GC. It is created at startup or by
// gc.enable_parallel(), retained across collections and fork in the parent,
// and destroyed by reconfiguration, gc.disable_parallel(), or finalization.
//
// Dispatch uses per-worker condition variables. Worker completions are
// signalled through done_mutex, done_cond, and workers_done_count.
typedef struct _PyGCThreadPool {
    // Includes the collecting thread as worker zero.
    int num_workers;
    PyThread_handle_t *threads; // Thread handles for workers 1..N-1

    // Persistent worker states (allocated once, reused across collections)
    _PyGCWorkerState *workers;  // Per-worker state including deques

    _PyGCPoolWorkerArgs *worker_args;

    // Current work descriptor - set by main thread before signalling workers
    _PyGCWorkDescriptor *current_work;

    // Per-collection done signaling. Workers signal done_cond when they
    // finish; main waits on done_cond until workers_done_count reaches
    // the count of woken workers. Mirrors the GIL parallel GC.
    PyMUTEX_T done_mutex;
    PyCOND_T done_cond;
    int workers_done_count;  // Protected by done_mutex

    // Worker control
    int shutdown;           // 1 = pool is shutting down (use atomics)

    size_t threads_created;
} _PyGCThreadPool;

PyAPI_FUNC(int) _PyGC_ThreadPoolInit(
    PyInterpreterState *interp, int num_workers);
PyAPI_FUNC(void) _PyGC_ThreadPoolFini(PyInterpreterState *interp);
PyAPI_FUNC(void) _PyGC_ThreadPoolBeforeFork(PyInterpreterState *interp);
PyAPI_FUNC(void) _PyGC_ThreadPoolAfterFork(PyInterpreterState *interp);
PyAPI_FUNC(void) _PyGC_ThreadPoolAfterForkChild(PyInterpreterState *interp);

typedef struct _PyGCPageBucket {
    mi_page_t **pages;       // Array of page pointers
    size_t num_pages;        // Number of pages assigned
    size_t capacity;         // Allocated capacity
} _PyGCPageBucket;

typedef struct {
    int num_workers;
    _PyGCPageBucket *buckets;     // One bucket per worker
} _PyGCFTParState;

// Assign pages to worker buckets using sequential filling.
// Returns 0 on success, -1 on error.
// Must be called with world stopped.
PyAPI_FUNC(int) _PyGC_AssignPagesToBuckets(
    PyInterpreterState *interp,
    _PyGCFTParState *state);

PyAPI_FUNC(void) _PyGC_FreeBuckets(_PyGCFTParState *state);

PyAPI_FUNC(int) _PyGC_ParallelMarkHeapWithPool(
    PyInterpreterState *interp,
    _PyGCFTParState *state,
    int skip_deferred_objects);

// gc_refs is stored in ob_tid while the world is stopped.
static inline Py_ssize_t
gc_get_refs_atomic(PyObject *op)
{
    return (Py_ssize_t)_Py_atomic_load_uintptr_relaxed(&op->ob_tid);
}

// Return zero when parallel GC is disabled, otherwise the configured number
// of participants.
static inline int
_PyGC_GetParallelWorkers(PyInterpreterState *interp)
{
    struct _gc_runtime_state *gc = &interp->gc;
    if (!gc->parallel_gc_enabled) {
        return 0;
    }
    _PyGCThreadPool *pool = gc->thread_pool;
    if (pool != NULL) {
        return pool->num_workers;
    }
    return gc->parallel_gc_num_workers;
}

// Returns worker count if parallel should be used, 0 otherwise.
static inline int
_PyGC_ShouldUseParallel(PyInterpreterState *interp)
{
    int workers = _PyGC_GetParallelWorkers(interp);
    if (workers <= 1) {
        return 0;
    }
    return workers;
}

#endif  // Py_GIL_DISABLED && Py_PARALLEL_GC

#ifdef __cplusplus
}
#endif

#endif  // Py_INTERNAL_GC_FT_PARALLEL_H
