// Free-threading parallel garbage collector.
// This extends gc_free_threading.c with parallel marking support.

#ifndef Py_INTERNAL_GC_FT_PARALLEL_H
#define Py_INTERNAL_GC_FT_PARALLEL_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

#ifdef Py_GIL_DISABLED

#include "pycore_gc.h"       // _PyGC_BITS_*
#include "pycore_ws_deque.h" // Chase-Lev work-stealing deque

// For barrier implementation we need pthread primitives
#ifdef _POSIX_THREADS
#include <pthread.h>
#include <unistd.h>  // sysconf for CPU count
#include <sched.h>   // sched_yield for spin-wait
#endif

//-----------------------------------------------------------------------------
// Barrier Synchronization for FTP Parallel GC
//-----------------------------------------------------------------------------

// A barrier for synchronizing N worker threads.
// All N threads must reach the barrier before any can proceed.
typedef struct {
    unsigned int num_left;   // Threads remaining before barrier lifts
    unsigned int capacity;   // Total number of threads
    unsigned int epoch;      // Disambiguates spurious wakeups
    pthread_mutex_t lock;
    pthread_cond_t cond;
} _PyGCFTBarrier;

// Initialize barrier for capacity threads
static inline void
_PyGCFTBarrier_Init(_PyGCFTBarrier *barrier, int capacity)
{
    barrier->capacity = capacity;
    barrier->num_left = capacity;
    barrier->epoch = 0;
    pthread_mutex_init(&barrier->lock, NULL);
    pthread_cond_init(&barrier->cond, NULL);
}

// Finalize barrier resources
static inline void
_PyGCFTBarrier_Fini(_PyGCFTBarrier *barrier)
{
    pthread_cond_destroy(&barrier->cond);
    pthread_mutex_destroy(&barrier->lock);
}

// Wait at barrier - blocks until all threads arrive
static inline void
_PyGCFTBarrier_Wait(_PyGCFTBarrier *barrier)
{
    pthread_mutex_lock(&barrier->lock);

    unsigned int current_epoch = barrier->epoch;
    barrier->num_left--;

    if (barrier->num_left == 0) {
        // Last thread to arrive - lift the barrier
        barrier->epoch++;
        barrier->num_left = barrier->capacity;
        pthread_cond_broadcast(&barrier->cond);
    } else {
        // Wait until the barrier is lifted
        while (barrier->epoch == current_epoch) {
            pthread_cond_wait(&barrier->cond, &barrier->lock);
        }
    }

    pthread_mutex_unlock(&barrier->lock);
}

//-----------------------------------------------------------------------------
// Atomic GC bit operations for parallel marking
//-----------------------------------------------------------------------------

// Try to atomically set a GC bit on an object.
// Returns 1 if this call set the bit (was previously 0).
// Returns 0 if the bit was already set.
//
// Uses atomic fetch-or instead of CAS loop because:
// 1. GC bits only transition 0->1 during marking (monotonic)
// 2. fetch-or is a single atomic instruction (no retry loop)
// 3. Returns old value so we can detect if we were first
static inline int
_PyGC_TrySetBit(PyObject *op, uint8_t bit)
{
    uint8_t old_bits = _Py_atomic_or_uint8(&op->ob_gc_bits, bit);
    return !(old_bits & bit);  // Return 1 if bit was not set before
}

// Try to atomically clear a GC bit on an object.
// Returns 1 if this call cleared the bit (was previously 1).
// Returns 0 if the bit was already clear.
//
// Uses atomic fetch-and instead of CAS loop (same rationale as TrySetBit).
static inline int
_PyGC_TryClearBit(PyObject *op, uint8_t bit)
{
    uint8_t old_bits = _Py_atomic_and_uint8(&op->ob_gc_bits, ~bit);
    return (old_bits & bit) != 0;  // Return 1 if bit was set before
}

// Atomically set a GC bit (fire-and-forget, for when we don't need return value).
static inline void
_PyGC_AtomicSetBit(PyObject *op, uint8_t bit)
{
    (void)_Py_atomic_or_uint8(&op->ob_gc_bits, bit);
}

// Atomically clear a GC bit (fire-and-forget, for when we don't need return value).
static inline void
_PyGC_AtomicClearBit(PyObject *op, uint8_t bit)
{
    (void)_Py_atomic_and_uint8(&op->ob_gc_bits, ~bit);
}

// Convenience functions for common operations

// Try to mark object as ALIVE. Returns 1 if we were first to mark it.
// Uses check-first optimization: fast relaxed load before atomic RMW.
// This is a significant win for objects visited multiple times (type objects,
// builtins, shared containers) - we skip the expensive atomic RMW entirely.
static inline int
_PyGC_TryMarkAlive(PyObject *op)
{
    // Fast path: check if already marked (relaxed load - very cheap)
    if (_Py_atomic_load_uint8_relaxed(&op->ob_gc_bits) & _PyGC_BITS_ALIVE) {
        return 0;  // Already marked, skip atomic RMW
    }
    // Slow path: not marked yet (or race), do atomic set
    return _PyGC_TrySetBit(op, _PyGC_BITS_ALIVE);
}

// Check if object is marked ALIVE (atomic read).
static inline int
_PyGC_IsAlive(PyObject *op)
{
    return (_Py_atomic_load_uint8_relaxed(&op->ob_gc_bits) & _PyGC_BITS_ALIVE) != 0;
}

// Check if object is marked UNREACHABLE (atomic read).
static inline int
_PyGC_IsUnreachable(PyObject *op)
{
    return (_Py_atomic_load_uint8_relaxed(&op->ob_gc_bits) & _PyGC_BITS_UNREACHABLE) != 0;
}

// Try to mark object as reachable by clearing UNREACHABLE bit.
// Returns 1 if we were the first to mark it (and should traverse).
// Uses check-first optimization: fast relaxed load before atomic RMW.
static inline int
_PyGC_TryMarkReachable(PyObject *op)
{
    // Fast path: check if already reachable (relaxed load - very cheap)
    if (!(_Py_atomic_load_uint8_relaxed(&op->ob_gc_bits) & _PyGC_BITS_UNREACHABLE)) {
        return 0;  // Already reachable, skip atomic RMW
    }
    // Slow path: still unreachable (or race), do atomic clear
    return _PyGC_TryClearBit(op, _PyGC_BITS_UNREACHABLE);
}

//-----------------------------------------------------------------------------
// Persistent Thread Pool for Parallel GC
//-----------------------------------------------------------------------------
// Note: _PyGCLocalBuffer is defined in pycore_ws_deque.h (shared with GIL build)

// Work types for the thread pool
typedef enum {
    _PyGC_WORK_NONE = 0,        // No work pending
    _PyGC_WORK_PROPAGATE,       // Propagate alive from roots
    _PyGC_WORK_MARK_PAGES,      // Mark alive from page buckets
    _PyGC_WORK_SHUTDOWN         // Shutdown workers
} _PyGCWorkType;

// Forward declarations
struct _PyGCThreadPool;
struct _PyGCPageBucket;

// Per-worker state for parallel GC (define early so _PyGCWorkDescriptor can use it)
typedef struct _PyGCWorkerState {
    _PyWSDeque deque;        // Work-stealing deque for this worker
    _PyGCLocalBuffer local;  // Fast local buffer (no fences needed)
    int worker_id;           // Worker identifier (0..num_workers-1)

    // Thread-local memory pool for deque backing storage
    // Pre-allocated once at gc.enable_parallel() time to avoid malloc during hot path
    void *local_pool;        // Pre-allocated buffer for deque (2MB)
    size_t local_pool_size;  // Size of local_pool in entries

    // Statistics (updated atomically)
    size_t objects_marked;   // Objects marked by this worker
    size_t objects_stolen;   // Objects stolen from other workers
    size_t steals_attempted; // Number of steal attempts
} _PyGCWorkerState;

// Work descriptor - describes work for a single collection
typedef struct {
    _PyGCWorkType type;

    // For PROPAGATE work
    PyObject **roots;           // Array of root objects (not owned)
    size_t num_roots;           // Number of roots

    // For MARK_PAGES work
    struct _PyGCPageBucket *buckets;  // Page buckets (not owned)

    // Shared worker state (deques, etc.)
    _PyGCWorkerState *workers;        // Now uses the full typedef

    // Result
    volatile int error_flag;    // Set if any worker encounters an error

    // Statistics
    size_t *per_worker_marked;  // Per-worker objects marked (caller allocates)
} _PyGCWorkDescriptor;

// Persistent thread pool for parallel GC
// Created once on gc.enable_parallel(), destroyed on gc.disable_parallel()
//
// Uses barrier synchronization like the GIL-based parallel GC:
// - mark_barrier: Workers wait here until main signals work ready
// - done_barrier: All workers (including main as worker 0) wait here when done
//
// This guarantees correct termination: barriers only release when ALL
// participants arrive, ensuring no work is in flight.
typedef struct _PyGCThreadPool {
    int num_workers;            // Number of workers (including main thread as worker 0)
    pthread_t *threads;         // Thread handles for workers 1..N-1

    // Persistent worker states (allocated once, reused across collections)
    _PyGCWorkerState *workers;  // Per-worker state including deques

    // Barrier synchronization (like GIL-based parallel GC)
    _PyGCFTBarrier mark_barrier;     // Workers wait here for work
    _PyGCFTBarrier done_barrier;     // All workers wait here when done

    // Worker control
    volatile int shutdown;           // 1 = pool is shutting down

    // Debug/testing counters (for assertions)
    size_t threads_created;          // Total threads ever created (should equal num_workers-1)
    size_t collections_completed;    // Number of GC collections processed
} _PyGCThreadPool;

// Thread pool management functions
PyAPI_FUNC(int) _PyGC_ThreadPoolInit(PyInterpreterState *interp, int num_workers);
PyAPI_FUNC(void) _PyGC_ThreadPoolFini(PyInterpreterState *interp);
PyAPI_FUNC(int) _PyGC_ThreadPoolIsActive(PyInterpreterState *interp);

// Get thread pool statistics for testing
PyAPI_FUNC(size_t) _PyGC_ThreadPoolGetThreadsCreated(PyInterpreterState *interp);
PyAPI_FUNC(size_t) _PyGC_ThreadPoolGetCollectionsCompleted(PyInterpreterState *interp);

//-----------------------------------------------------------------------------
// Page-based work distribution
//-----------------------------------------------------------------------------

// A page assignment for a single worker
typedef struct _PyGCPageBucket {
    mi_page_t **pages;       // Array of page pointers
    size_t num_pages;        // Number of pages assigned
    size_t capacity;         // Allocated capacity
} _PyGCPageBucket;

// State for page-based parallel GC
typedef struct {
    int num_workers;
    _PyGCPageBucket *buckets;     // One bucket per worker
    _PyGCWorkerState *workers;    // Per-worker state (including deques)
    pthread_t *threads;           // Thread handles (workers 1..N-1)

    // Barriers for phase synchronization
    _PyGCFTBarrier phase_barrier;  // Sync between GC phases
    _PyGCFTBarrier done_barrier;   // Sync at end of parallel work

    // Flags
    volatile int error_flag;       // Set if any worker encounters an error
    volatile int workers_done;     // Count of workers that finished

    // Statistics
    size_t total_pages;
    size_t total_objects;
    size_t *per_worker_marked;    // Per-worker objects marked (saved before free)
} _PyGCFTParState;

//-----------------------------------------------------------------------------
// Page counting and assignment
//-----------------------------------------------------------------------------

// Count total GC pages across all heaps in the interpreter.
// Must be called with world stopped.
PyAPI_FUNC(size_t) _PyGC_CountPages(PyInterpreterState *interp);

// Assign pages to worker buckets using sequential filling.
// Returns 0 on success, -1 on error.
// Must be called with world stopped.
PyAPI_FUNC(int) _PyGC_AssignPagesToBuckets(
    PyInterpreterState *interp,
    _PyGCFTParState *state);

// Free resources used by page buckets.
PyAPI_FUNC(void) _PyGC_FreeBuckets(_PyGCFTParState *state);

//-----------------------------------------------------------------------------
// Parallel marking (Phase 1: mark alive)
//-----------------------------------------------------------------------------

// Run parallel mark-alive phase.
// Each worker processes its assigned pages and uses work-stealing
// for transitively discovered objects.
PyAPI_FUNC(int) _PyGC_ParallelMarkAlive(
    PyInterpreterState *interp,
    _PyGCFTParState *state);

// Parallel propagation from roots.
// Takes initial roots (already marked alive) and transitively marks
// all reachable objects as alive using parallel workers.
// This is the integration point for gc_mark_alive_from_roots().
PyAPI_FUNC(int) _PyGC_ParallelPropagateAlive(
    PyInterpreterState *interp,
    PyObject **initial_roots,
    size_t num_roots,
    int num_workers);

// Parallel propagation using persistent thread pool.
// Same as _PyGC_ParallelPropagateAlive but uses the thread pool
// instead of spawning new threads per collection.
// Requires thread pool to be initialized first.
PyAPI_FUNC(int) _PyGC_ParallelPropagateAliveWithPool(
    PyInterpreterState *interp,
    PyObject **initial_roots,
    size_t num_roots,
    int num_workers);

//-----------------------------------------------------------------------------
// Atomic gc_refs operations for parallel deduce_unreachable
//-----------------------------------------------------------------------------
// These are used when multiple workers may be modifying gc_refs concurrently.
// gc_refs is stored in ob_tid during GC (when world is stopped).

// Atomically decrement gc_refs (used by visit_decref in parallel mode)
static inline void
gc_decref_atomic(PyObject *op)
{
    // ob_tid is uintptr_t but we treat it as signed for gc_refs
    // Decrement by 1 using atomic add of (uintptr_t)-1
    _Py_atomic_add_uintptr(&op->ob_tid, (uintptr_t)-1);
}

// Atomically add to gc_refs (used by update_refs in parallel mode)
static inline void
gc_add_refs_atomic(PyObject *op, Py_ssize_t refs)
{
    _Py_atomic_add_uintptr(&op->ob_tid, (uintptr_t)refs);
}

// Atomically initialize gc_refs if not already unreachable
// Returns 1 if we initialized (set unreachable bit), 0 if already set
static inline int
gc_maybe_init_refs_atomic(PyObject *op)
{
    // Use atomic try-set for unreachable bit
    if (_PyGC_TrySetBit(op, _PyGC_BITS_UNREACHABLE)) {
        // We set the bit, so we're responsible for zeroing ob_tid
        // Use release semantics so other threads see the zero
        _Py_atomic_store_uintptr_release(&op->ob_tid, 0);
        return 1;
    }
    return 0;  // Already unreachable
}

// Wait until ob_tid has been initialized for gc_refs tracking.
// Thread IDs are typically large values (> 1M on 64-bit systems).
// gc_refs are small values (refcount - internal refs, typically < 100k).
// We spin-wait until ob_tid looks like a valid gc_refs value.
static inline void
gc_wait_for_refs_init(PyObject *op)
{
    // Threshold to distinguish thread ID from gc_refs
    // Thread IDs are usually large (process_id << 32 | thread_number)
    // gc_refs are small (object refcount - internal refs)
    const uintptr_t THREAD_ID_THRESHOLD = 0x100000;  // 1M

    int spins = 0;
    while (1) {
        uintptr_t val = _Py_atomic_load_uintptr_acquire(&op->ob_tid);
        if (val < THREAD_ID_THRESHOLD) {
            // Looks like a valid gc_refs value
            break;
        }
        // Still looks like a thread ID, spin-wait
        if (++spins > 1000) {
            // Yield after many spins to avoid burning CPU
            sched_yield();
            spins = 0;
        }
    }
}

// Atomically get gc_refs (relaxed load for checking)
static inline Py_ssize_t
gc_get_refs_atomic(PyObject *op)
{
    return (Py_ssize_t)_Py_atomic_load_uintptr_relaxed(&op->ob_tid);
}

//-----------------------------------------------------------------------------
// Parallel update_refs for deduce_unreachable_heap
//-----------------------------------------------------------------------------

// Run parallel update_refs phase.
// This is the first phase of deduce_unreachable_heap.
// Returns the total candidate count across all workers, or -1 on error.
PyAPI_FUNC(Py_ssize_t) _PyGC_ParallelUpdateRefs(
    PyInterpreterState *interp,
    _PyGCFTParState *state);

//-----------------------------------------------------------------------------
// Parallel mark_heap_visitor for deduce_unreachable_heap
//-----------------------------------------------------------------------------

// Run parallel mark_heap_visitor phase.
// Finds objects with gc_refs > 0 (external references) and transitively
// clears the UNREACHABLE bit on all reachable objects.
// This is the second phase of deduce_unreachable_heap (after update_refs).
// Returns 0 on success, -1 on error.
PyAPI_FUNC(int) _PyGC_ParallelMarkHeap(
    PyInterpreterState *interp,
    _PyGCFTParState *state,
    int skip_deferred_objects);

//-----------------------------------------------------------------------------
// Parallel scan_heap_visitor for deduce_unreachable_heap
//-----------------------------------------------------------------------------

// Thread-local worklist for parallel scan_heap
// Uses the same ob_tid linking as the serial worklist
struct _PyGCScanWorklist {
    uintptr_t head;  // Head of linked list (via ob_tid)
    size_t count;    // Number of items in list
};

// Per-worker scan state
struct _PyGCScanWorkerState {
    struct _PyGCScanWorklist unreachable;       // Thread-local unreachable list
    struct _PyGCScanWorklist legacy_finalizers; // Thread-local legacy finalizers
    size_t long_lived_total;                    // Count of reachable objects
    int reason;                                 // GC reason (for shutdown check)
};

// Parallel scan_heap output - passed to _PyGC_ParallelScanHeap
struct _PyGCScanHeapResult {
    uintptr_t *unreachable_head;     // Pointer to worklist head
    uintptr_t *legacy_finalizers_head;
    Py_ssize_t *long_lived_total;    // Pointer to counter
    int gc_reason;                   // GC reason for shutdown check
};

// Run parallel scan_heap_visitor phase.
// Scans all objects, pushing unreachable ones to worklists and restoring
// ob_tid for reachable ones.
// This is the third phase of deduce_unreachable_heap (after mark_heap).
// Returns 0 on success, -1 on error.
PyAPI_FUNC(int) _PyGC_ParallelScanHeap(
    PyInterpreterState *interp,
    _PyGCFTParState *state,
    struct _PyGCScanHeapResult *result);

// Default threshold for parallel GC (minimum roots before using parallel)
#define _PyGC_PARALLEL_THRESHOLD_DEFAULT 10000

// Get number of parallel GC workers based on configuration.
// Returns 0 if parallel GC is disabled, otherwise the worker count.
static inline int
_PyGC_GetParallelWorkers(PyInterpreterState *interp)
{
    struct _gc_runtime_state *gc = &interp->gc;
    if (!gc->parallel_gc_enabled) {
        return 0;
    }
    if (gc->parallel_gc_num_workers > 0) {
        return gc->parallel_gc_num_workers;
    }
    // Auto: use half of available CPUs, minimum 2, maximum 8
    // Note: os.cpu_count() uses sysconf(_SC_NPROCESSORS_ONLN) on Linux
    long ncpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpus < 1) ncpus = 1;
    int workers = (int)(ncpus / 2);
    if (workers < 2) workers = 2;
    if (workers > 8) workers = 8;
    return workers;
}

// Check if parallel GC should be used for given number of roots.
// Returns worker count if parallel should be used, 0 otherwise.
static inline int
_PyGC_ShouldUseParallel(PyInterpreterState *interp, size_t num_roots)
{
    int workers = _PyGC_GetParallelWorkers(interp);
    if (workers <= 1) {
        return 0;
    }
    // Check threshold
    struct _gc_runtime_state *gc = &interp->gc;
    size_t threshold = gc->parallel_gc_threshold;
    if (threshold == 0) {
        threshold = _PyGC_PARALLEL_THRESHOLD_DEFAULT;
    }
    if (num_roots < threshold) {
        return 0;  // Too few roots, overhead not worth it
    }
    return workers;
}

//-----------------------------------------------------------------------------
// Testing / debugging APIs (exposed for unit tests)
//-----------------------------------------------------------------------------

#ifdef Py_DEBUG

// Get page count for testing
PyAPI_FUNC(size_t) _PyGC_TestCountPages(void);

// Test page assignment algorithm
// Returns array of page counts per worker (caller must free)
PyAPI_FUNC(size_t *) _PyGC_TestPageAssignment(
    size_t total_pages,
    int num_workers);

// Test atomic bit operations
// Creates test object and runs concurrent bit operations
// Returns 0 on success, -1 on failure
PyAPI_FUNC(int) _PyGC_TestAtomicBitOps(int num_threads);

// Test REAL page enumeration (stops the world, runs actual code path)
// Returns array of [total_pages, bucket0, bucket1, ...] (num_workers+1 elements)
// Caller must free with PyMem_RawFree. Returns NULL on error.
PyAPI_FUNC(size_t *) _PyGC_TestRealPageEnumeration(int num_workers);

// Test REAL parallel marking (stops the world, runs actual marking)
// Returns array of [total_objects, worker0_marked, worker1_marked, ...]
// (num_workers+1 elements). Caller must free with PyMem_RawFree.
// Returns NULL on error.
PyAPI_FUNC(size_t *) _PyGC_TestParallelMark(int num_workers);

#endif  // Py_DEBUG

#endif  // Py_GIL_DISABLED

#ifdef __cplusplus
}
#endif

#endif  // Py_INTERNAL_GC_FT_PARALLEL_H
