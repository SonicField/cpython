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
#include "pycore_ws_deque.h"       // _PyWSDeque
#include "pycore_condvar.h"        // PyMUTEX_T, PyCOND_T
#include "pycore_gc.h"             // PyGC_Head

// Parallel GC Configuration
// Only enabled when built with --with-parallel-gc
// Mutual exclusion with --disable-gil (free-threading uses different GC)

#ifdef Py_PARALLEL_GC

// =============================================================================
// Barrier Synchronization
// =============================================================================

// A barrier for synchronizing N threads.
//
// All N threads must reach the barrier before it is lifted, unblocking all
// threads.
typedef struct {
    // Number of threads left to reach the barrier before it can be lifted
    unsigned int num_left;

    // Total number of threads managed by the barrier
    unsigned int capacity;

    // The epoch advances once all threads reach the barrier; it
    // disambiguates spurious wakeups from true wakeups that happen once all
    // threads have reached the barrier.
    unsigned int epoch;

    PyMUTEX_T lock;
    PyCOND_T cond;
} _PyGCBarrier;

// Barrier functions (implemented in Python/gc_parallel.c)
void _PyGCBarrier_Init(_PyGCBarrier *barrier, int capacity);
void _PyGCBarrier_Fini(_PyGCBarrier *barrier);
void _PyGCBarrier_Wait(_PyGCBarrier *barrier);

// =============================================================================
// Worker Thread State
// =============================================================================

// Forward declaration
typedef struct _PyParallelGCState _PyParallelGCState;

// Per-worker state for parallel GC
typedef struct {
    // Work-stealing deque for marking queue
    _PyWSDeque deque;

    // Statistics (for debugging/profiling)
    unsigned long objects_marked;
    unsigned long steal_attempts;
    unsigned long steal_successes;

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

    // Worker should exit when this is set
    int should_exit;

} _PyParallelGCWorker;

// =============================================================================
// Global Parallel GC State
// =============================================================================

// Global state for parallel garbage collection
struct _PyParallelGCState {
    // Number of worker threads
    size_t num_workers;

    // Minimum generation to use parallel GC for
    // (generations < min_gen use serial GC)
    int min_gen;

    // Synchronizes all workers before marking reachable objects
    _PyGCBarrier mark_barrier;

    // Synchronizes all worker threads and the main thread at the end of
    // parallel collection
    _PyGCBarrier done_barrier;

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

#endif // Py_PARALLEL_GC

#ifdef __cplusplus
}
#endif

#endif // Py_INTERNAL_GC_PARALLEL_H
