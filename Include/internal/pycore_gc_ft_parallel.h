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

#include "pycore_gc.h"           // _PyGC_BITS_*
#include "pycore_ws_deque.h"     // Chase-Lev work-stealing deque, _PyGCLocalBuffer
#include "pycore_gc_barrier.h"   // _PyGCBarrier (shared with GIL build)

// For portable thread management (PyThread_handle_t, PyThread_start_joinable_thread, etc.)
#include "pycore_pythread.h"

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

// Try to mark object as ALIVE and clear UNREACHABLE. Returns 1 if we should traverse.
//
// OPTIMIZATION: Uses relaxed read + relaxed write instead of atomic RMW.
// During STW, all threads are GC workers cooperating. We only need memory
// visibility at barrier sync points, not per-operation atomicity.
//
// If two workers race to mark the same object:
// - Both do relaxed read, both see not-ALIVE
// - Both write ALIVE (idempotent - same result)
// - Both return 1 and traverse the object's referents
// - Duplicate traversal is acceptable: discovered refs hit the relaxed read
//   check on the next level and stop propagating
//
// This eliminates 2 expensive atomic RMW operations per object:
// - _PyGC_TrySetBit (atomic fetch-or): ~10-20 cycles
// - _PyGC_AtomicClearBit (atomic fetch-and): ~10-20 cycles
// Replaced by 1 relaxed read + 1 relaxed store: ~2-3 cycles total.
static inline int
_PyGC_TryMarkAlive(PyObject *op)
{
    // Relaxed read - filters most already-marked objects
    if (_Py_atomic_load_uint8_relaxed(&op->ob_gc_bits) & _PyGC_BITS_ALIVE) {
        return 0;  // Already marked
    }
    // Relaxed write - no atomic RMW needed during STW
    // Compute new bits: set ALIVE, clear UNREACHABLE
    // Use relaxed store for portability (byte-level atomicity on all architectures)
    uint8_t new_bits = (op->ob_gc_bits | _PyGC_BITS_ALIVE) & ~_PyGC_BITS_UNREACHABLE;
    _Py_atomic_store_uint8_relaxed(&op->ob_gc_bits, new_bits);
    return 1;
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
    _PyGC_WORK_PROPAGATE,       // Propagate alive from roots (mark_alive)
    _PyGC_WORK_UPDATE_REFS,     // Initialize gc_refs on heap (deduce_unreachable phase 1)
    _PyGC_WORK_MARK_HEAP,       // Find roots and mark reachable (deduce_unreachable phase 2)
    _PyGC_WORK_SCAN_HEAP,       // Collect unreachable objects (deduce_unreachable phase 3)
    _PyGC_WORK_ASYNC_CLEANUP,   // Single-threaded concurrent cleanup (runs in background)
    _PyGC_WORK_SHUTDOWN         // Shutdown workers
} _PyGCWorkType;

// Forward declarations
struct _PyGCThreadPool;
struct _PyGCPageBucket;
struct mi_page_s;  // mi_page_t from mimalloc
typedef struct mi_page_s mi_page_t;

// Per-worker state for parallel GC (define early so _PyGCWorkDescriptor can use it)
typedef struct _PyGCWorkerState {
    _PyWSDeque deque;        // Work-stealing deque for this worker
    _PyGCLocalBuffer local;  // Fast local buffer (no fences needed)
    int worker_id;           // Worker identifier (0..num_workers-1)

    // Thread-local memory pool for deque backing storage
    // Pre-allocated once at gc.enable_parallel() time to avoid malloc during hot path
    void *local_pool;        // Pre-allocated buffer for deque (2MB)
    size_t local_pool_size;  // Size of local_pool in entries

    // Python thread state for this worker (for Py_REF_DEBUG in debug builds)
    // Created at pool init, destroyed at pool fini
    PyThreadState *tstate;

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

    // For UPDATE_REFS / MARK_HEAP work
    struct _PyGCPageBucket *buckets;  // Page buckets (not owned)
    int skip_deferred;                // For MARK_HEAP: skip deferred objects

    // For SCAN_HEAP work - uses dynamic page distribution
    struct _PyGCScanHeapResult *scan_result;    // Output pointers (not owned)
    struct _PyGCScanWorkerState *scan_workers;  // Per-worker scan state (not owned)
    mi_page_t **page_array;                     // Array of pages to scan
    size_t total_pages;                         // Number of pages
    _Atomic(int) page_counter;                  // Atomic counter for work distribution

    // Shared worker state (deques, etc.)
    _PyGCWorkerState *workers;        // Now uses the full typedef

    // For ASYNC_CLEANUP work (single-threaded concurrent background cleanup)
    PyObject **async_cleanup_objects;   // Array of objects to clean up (owned)
    Py_ssize_t async_cleanup_count;     // Number of objects
    struct _gc_runtime_state *gcstate;  // GC state for clearing collecting flag

    // Result
    volatile int error_flag;    // Set if any worker encounters an error

    // Statistics
    size_t *per_worker_marked;  // Per-worker objects marked (caller allocates)
    Py_ssize_t *per_worker_refs; // Per-worker candidate count (for UPDATE_REFS)
} _PyGCWorkDescriptor;

// Persistent thread pool for parallel GC
// Created once on gc.enable_parallel(), destroyed on gc.disable_parallel()
//
// Uses barrier synchronization like the GIL-based parallel GC:
// - mark_barrier: Workers wait here until main signals work ready
// - done_barrier: All workers (including main as worker 0) wait here when done
// - phase_barrier: For multi-phase operations (e.g., UPDATE_REFS init/compute)
//
// This guarantees correct termination: barriers only release when ALL
// participants arrive, ensuring no work is in flight.
typedef struct _PyGCThreadPool {
    // Interpreter state - needed to create Python thread states for workers
    PyInterpreterState *interp;

    int num_workers;            // Number of workers (including main thread as worker 0)
    PyThread_handle_t *threads; // Thread handles for workers 1..N-1

    // Persistent worker states (allocated once, reused across collections)
    _PyGCWorkerState *workers;  // Per-worker state including deques

    // Current work descriptor - set by main thread before signalling workers
    _PyGCWorkDescriptor *current_work;

    // Barrier synchronization (like GIL-based parallel GC)
    _PyGCBarrier mark_barrier;     // Workers wait here for work
    _PyGCBarrier done_barrier;     // All workers wait here when done
    _PyGCBarrier phase_barrier;    // For multi-phase operations
    _PyGCBarrier async_done_barrier; // For concurrent work (N-1 workers, excludes main thread)

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
    PyThread_handle_t *threads;   // Thread handles (workers 1..N-1)

    // Barriers for phase synchronization
    _PyGCBarrier phase_barrier;  // Sync between GC phases
    _PyGCBarrier done_barrier;   // Sync at end of parallel work

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

// Parallel propagation using persistent thread pool.
// This is the only supported parallel propagation function.
// Requires thread pool to be initialized first.
PyAPI_FUNC(int) _PyGC_ParallelPropagateAliveWithPool(
    PyInterpreterState *interp,
    PyObject **initial_roots,
    size_t num_roots,
    int num_workers);

//-----------------------------------------------------------------------------
// Pool-based parallel GC phases (require thread pool to be initialized)
//-----------------------------------------------------------------------------

// Pool-based parallel update_refs
PyAPI_FUNC(Py_ssize_t) _PyGC_ParallelUpdateRefsWithPool(
    PyInterpreterState *interp,
    _PyGCFTParState *state);

// Pool-based parallel mark_heap
PyAPI_FUNC(int) _PyGC_ParallelMarkHeapWithPool(
    PyInterpreterState *interp,
    _PyGCFTParState *state,
    int skip_deferred_objects);

// Pool-based parallel scan_heap
PyAPI_FUNC(int) _PyGC_ParallelScanHeapWithPool(
    PyInterpreterState *interp,
    _PyGCFTParState *state,
    struct _PyGCScanHeapResult *result);

// Start concurrent cleanup in background worker (single-threaded).
// Objects array is copied by the function (caller frees original).
// The background worker will set gcstate->collecting = 0 when done.
// Pool must exist (guaranteed if parallel_gc_enabled is true).
PyAPI_FUNC(void) _PyGC_StartAsyncCleanup(
    PyInterpreterState *interp,
    PyObject **objects,
    Py_ssize_t count);

//-----------------------------------------------------------------------------
// Ad-hoc thread versions (spawn threads per-collection instead of using pool)
// These duplicate the pool-based functions but create threads on each call.
// Retained for debugging and comparison with pool-based versions.
//-----------------------------------------------------------------------------

// Ad-hoc parallel update_refs (spawns its own threads)
PyAPI_FUNC(Py_ssize_t) _PyGC_ParallelUpdateRefs(
    PyInterpreterState *interp,
    _PyGCFTParState *state);

// Ad-hoc parallel mark_heap (spawns its own threads)
PyAPI_FUNC(int) _PyGC_ParallelMarkHeap(
    PyInterpreterState *interp,
    _PyGCFTParState *state,
    int skip_deferred_objects);

// Ad-hoc parallel scan_heap (spawns its own threads)
PyAPI_FUNC(int) _PyGC_ParallelScanHeap(
    PyInterpreterState *interp,
    _PyGCFTParState *state,
    struct _PyGCScanHeapResult *result);

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

// Atomically get gc_refs (relaxed load for checking)
static inline Py_ssize_t
gc_get_refs_atomic(PyObject *op)
{
    return (Py_ssize_t)_Py_atomic_load_uintptr_relaxed(&op->ob_tid);
}

//-----------------------------------------------------------------------------
// Parallel scan_heap_visitor structs for deduce_unreachable_heap
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

    // Unique IDs collected during scan for batch release (no locks during STW)
    Py_ssize_t *unique_ids;                     // Array of unique IDs to release
    size_t unique_id_count;                     // Number of IDs collected
    size_t unique_id_capacity;                  // Capacity of array
};

// Parallel scan_heap output - passed to _PyGC_ParallelScanHeapWithPool
struct _PyGCScanHeapResult {
    uintptr_t *unreachable_head;     // Pointer to worklist head
    uintptr_t *legacy_finalizers_head;
    Py_ssize_t *long_lived_total;    // Pointer to counter
    int gc_reason;                   // GC reason for shutdown check
};

// Get number of parallel GC workers based on configuration.
// Returns 0 if parallel GC is disabled, otherwise the worker count.
static inline int
_PyGC_GetParallelWorkers(PyInterpreterState *interp)
{
    struct _gc_runtime_state *gc = &interp->gc;
    if (!gc->parallel_gc_enabled) {
        return 0;
    }
    // num_workers is required and validated by gc.enable_parallel()
    return gc->parallel_gc_num_workers;
}

// Check if parallel GC should be used.
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

//-----------------------------------------------------------------------------
// Statistics API
//-----------------------------------------------------------------------------

// Get parallel GC statistics as a Python dictionary.
// Returns a new reference to a dict with 'enabled', 'num_workers', 'phase_timing'.
// Returns NULL on error.
PyAPI_FUNC(PyObject *) _PyGC_FTParallelGetStats(PyInterpreterState *interp);

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
