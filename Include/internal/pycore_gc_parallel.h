// Copyright (c) Meta Platforms, Inc. and affiliates.
// Ported to CPython by Alex Turner

#ifndef Py_INTERNAL_GC_PARALLEL_H
#define Py_INTERNAL_GC_PARALLEL_H

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include "Python.h"
#include "pycore_pystate.h"        // _PyInterpreterState
#include "pycore_ws_deque.h"       // _PyWSDeque, _PyGCLocalBuffer
#include "pycore_condvar.h"        // PyMUTEX_T, PyCOND_T
#include "pycore_gc.h"             // PyGC_Head
#include "pycore_gc_barrier.h"     // _PyGCBarrier (shared with FTP)

// Parallel GC Configuration
// Only enabled when built with --with-parallel-gc
// Mutual exclusion with --disable-gil (free-threading uses different GC)

#ifdef Py_PARALLEL_GC

// =============================================================================
// Platform Requirements
// =============================================================================
//
// Parallel GC requires 64-bit pointers because multi-gigabyte heaps that
// benefit from parallel GC are only possible on 64-bit systems.
//
#if SIZEOF_VOID_P < 8
#  error "Parallel GC requires 64-bit platform (SIZEOF_VOID_P >= 8)"
#endif

// Maximum number of worker threads for parallel GC
#define _PyGC_MAX_WORKERS 1024

// =============================================================================
// Split Vector for Parallel Work Distribution
// =============================================================================
//
// During the serial update_refs phase (which already walks the entire list),
// we record pointers into a growable vector at fixed intervals. This allows
// parallel subtract_refs to quickly find evenly-spaced start/end positions.
//
// Benefits over bidirectional scan:
// - No atomics for partitioning
// - No extra list traversal (piggybacks on update_refs)
// - No thread creation (uses existing worker pool)
// - Fine-grained load balancing (8K resolution)
// - Good cache locality (sequential access during recording)
//

// Objects between split points (tunable)
#define _PyGC_SPLIT_INTERVAL 8192

// Initial capacity for split vector (entries, not bytes)
#define _PyGC_SPLIT_VECTOR_INITIAL_CAPACITY 128

// Split vector: growable array of pointers into the GC list
typedef struct {
    PyGC_Head **entries;    // Array of pointers into the GC list
    size_t count;           // Number of entries recorded
    size_t capacity;        // Current capacity
    size_t interval;        // Objects between split points
} _PyGCSplitVector;

// =============================================================================
// Worker Thread Phases
// =============================================================================
// Workers wait on mark_barrier, then dispatch based on current phase.
// The main GC thread sets the phase before signalling the barrier.

typedef enum {
    _PyGC_PHASE_IDLE,           // Waiting for work
    _PyGC_PHASE_SUBTRACT_REFS,  // Decrement gc_refs for internal references
    _PyGC_PHASE_MARK,           // Work-stealing parallel marking
} _PyGCPhase;

// =============================================================================
// Worker Thread State
// =============================================================================
// Note: _PyGCLocalBuffer is defined in pycore_ws_deque.h (shared with FTP)

// Forward declaration
typedef struct _PyParallelGCState _PyParallelGCState;

// Per-worker state for parallel GC
typedef struct {
    // Work-stealing deque for marking queue
    _PyWSDeque deque;

    // Fast local buffer - avoids expensive deque operations
    // Push/pop with zero memory fences, only touches deque when full/empty
    _PyGCLocalBuffer local_buffer;

    // Static slice assignment (for temporal locality)
    // Each worker gets a contiguous portion of the GC list
    // This preserves allocation order locality - objects allocated together
    // tend to reference each other and stay on the same worker
    PyGC_Head *slice_start;  // First object in this worker's slice (inclusive)
    PyGC_Head *slice_end;    // End of slice (exclusive, or list head)

    // Thread-local memory pool for deque arrays
    // Pre-allocated to avoid calloc during collections
    // Size: 256K entries = 2MB per worker (handles up to 256K objects per worker)
    void *local_pool;           // Pre-allocated buffer
    size_t local_pool_size;     // Size in entries (not bytes)
    int local_pool_in_use;      // 1 if deque is using local_pool, 0 if using malloc'd array

    // Statistics (for debugging/profiling)
    unsigned long objects_marked;
    unsigned long steal_attempts;
    unsigned long steal_successes;
    unsigned long objects_discovered;     // Children found via tp_traverse
    unsigned long traversals_performed;   // Number of tp_traverse calls
    unsigned long roots_in_slice;         // Roots found in this worker's slice
    unsigned long pool_overflows;         // Times we exceeded local pool and fell back to malloc

    // Per-worker timing for profiling (nanoseconds)
    int64_t work_start_ns;                // When worker started processing
    int64_t work_end_ns;                  // When worker finished processing
    unsigned long objects_in_segment;     // Objects in assigned segment (for subtract_refs)

    // Random seed for steal victim selection
    unsigned int steal_seed;

    // Back-pointer to global parallel GC state
    _PyParallelGCState *par_gc;

    // Thread ID (for debugging)
    unsigned long thread_id;

    // Thread handle (platform-specific)
#ifdef _POSIX_THREADS
    pthread_t thread;
#elif defined(NT_THREADS)
    HANDLE thread;
#endif

    // Worker should exit when this is set (accessed atomically for thread safety)
    int should_exit;

    // Python thread state for this worker (created at startup, needed for Py_REF_DEBUG)
    PyThreadState *tstate;

    // Current phase for this worker (set by main thread before barrier)
    _PyGCPhase phase;

} _PyParallelGCWorker;

// =============================================================================
// Global Parallel GC State
// =============================================================================

// Global state for parallel garbage collection
struct _PyParallelGCState {
    // Interpreter state - needed to create Python thread states for workers
    PyInterpreterState *interp;

    // Number of worker threads
    size_t num_workers;

    // Split vector for work distribution
    // Populated during serial update_refs, used by parallel subtract_refs
    _PyGCSplitVector split_vector;

    // Synchronizes all workers before marking reachable objects
    _PyGCBarrier mark_barrier;

    // Synchronizes all worker threads and the main thread at the end of
    // parallel collection
    _PyGCBarrier done_barrier;

    // Synchronizes worker startup - ensures all workers are ready before
    // ParallelStart returns (prevents race condition in Stop)
    _PyGCBarrier startup_barrier;

    // Tracks the number of workers actively running. When this reaches zero
    // it is safe to destroy shared state.
    int num_workers_active;

    // Lock for num_workers_active
    PyMUTEX_T active_lock;

    // Condition variable for waiting on workers to finish
    PyCOND_T workers_done_cond;

    // Flag indicating parallel GC is enabled
    int enabled;

    // Statistics for TDD/debugging
    size_t roots_found;                      // Total roots identified in last collection
    size_t roots_distributed;                // Roots distributed to workers in last collection
    size_t parallel_collections_attempted;   // Times parallel marking was attempted
    size_t parallel_collections_succeeded;   // Times parallel marking was used (vs serial fallback)

    // Phase timing (nanoseconds, for benchmarking)
    // Set during parallel collection, exposed via gc.get_parallel_stats()
    int timing_valid;                 // 1 if timing is from a complete parallel collection
    int64_t gc_start_ns;              // Start of GC (before update_refs)
    int64_t update_refs_end_ns;       // End of update_refs phase
    int64_t mark_alive_end_ns;        // End of mark_alive_from_roots phase
    int64_t phase_start_ns;           // Start of parallel GC (= update_refs_end_ns, kept for compatibility)
    int64_t subtract_refs_end_ns;     // End of subtract_refs phase
    int64_t mark_end_ns;              // End of mark phase
    int64_t cleanup_end_ns;           // End of cleanup phase (finalization, deallocation)

    // Worker threads (flexible array member - allocated based on num_workers)
    _PyParallelGCWorker workers[];
};

// =============================================================================
// API Functions
// =============================================================================

// Initialize parallel GC with num_workers worker threads
// Returns 0 on success, -1 on error (with exception set)
PyAPI_FUNC(int) _PyGC_ParallelInit(PyInterpreterState *interp, size_t num_workers);

// Shutdown parallel GC and clean up all worker threads
PyAPI_FUNC(void) _PyGC_ParallelFini(PyInterpreterState *interp);

// Start worker threads (called after initialization)
PyAPI_FUNC(int) _PyGC_ParallelStart(PyInterpreterState *interp);

// Stop worker threads (but don't destroy state - can restart later)
PyAPI_FUNC(void) _PyGC_ParallelStop(PyInterpreterState *interp);

// Check if parallel GC is enabled
PyAPI_FUNC(int) _PyGC_ParallelIsEnabled(PyInterpreterState *interp);

// Enable or disable parallel GC at runtime (workers must be started/stopped separately)
PyAPI_FUNC(void) _PyGC_ParallelSetEnabled(PyInterpreterState *interp, int enabled);

// Get current configuration
PyAPI_FUNC(PyObject *) _PyGC_ParallelGetConfig(PyInterpreterState *interp);

// Get current statistics (for testing/debugging)
PyAPI_FUNC(PyObject *) _PyGC_ParallelGetStats(PyInterpreterState *interp);

// Parallel marking entry point (called from gc.c)
// Returns 1 if parallel marking was used, 0 if should fall back to serial
PyAPI_FUNC(int) _PyGC_ParallelMoveUnreachable(
    PyInterpreterState *interp,
    PyGC_Head *young,
    PyGC_Head *unreachable
);

// Parallel subtract_refs: decrement gc_refs for internal references
// Uses split vector from par_gc state (populated by update_refs_with_splits)
// Uses atomic decrement since references can cross segment boundaries
// Returns 1 on success, 0 if should fall back to serial
PyAPI_FUNC(int) _PyGC_ParallelSubtractRefs(
    PyInterpreterState *interp,
    PyGC_Head *base
);

// =============================================================================
// Split Vector Operations
// =============================================================================

// Initialise split vector with default capacity and interval
// Returns 0 on success, -1 on allocation failure
PyAPI_FUNC(int) _PyGCSplitVector_Init(_PyGCSplitVector *vec);

// Free split vector resources
PyAPI_FUNC(void) _PyGCSplitVector_Fini(_PyGCSplitVector *vec);

// Clear split vector (reset count to 0, keep capacity)
PyAPI_FUNC(void) _PyGCSplitVector_Clear(_PyGCSplitVector *vec);

// Push a split point onto the vector (grows if needed)
// Returns 0 on success, -1 on allocation failure
PyAPI_FUNC(int) _PyGCSplitVector_Push(_PyGCSplitVector *vec, PyGC_Head *gc);

#endif // Py_PARALLEL_GC

#ifdef __cplusplus
}
#endif

#endif // Py_INTERNAL_GC_PARALLEL_H
