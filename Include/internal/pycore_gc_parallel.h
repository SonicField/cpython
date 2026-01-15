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
#include "pycore_pythread.h"       // PyThread_handle_t, PyThread_start_joinable_thread
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
// CPU Primitives for Spin-Wait Loops
// =============================================================================
//
// _Py_cpu_relax(): Hint to CPU during spin-wait loops.
// Reduces power consumption and avoids pipeline stalls.
// Much lighter than OS thread yield (sched_yield/SwitchToThread).
//

#if defined(__x86_64__) || defined(__i386__)
    #define _Py_cpu_relax() __asm__ volatile("pause")
#elif defined(__aarch64__)
    #define _Py_cpu_relax() __asm__ volatile("yield")
#elif defined(_M_X64) || defined(_M_IX86)
    // MSVC x86/x64
    #include <intrin.h>
    #define _Py_cpu_relax() _mm_pause()
#elif defined(_M_ARM64)
    // MSVC ARM64
    #include <intrin.h>
    #define _Py_cpu_relax() __yield()
#else
    // Fallback: no-op
    #define _Py_cpu_relax() ((void)0)
#endif

// =============================================================================
// Prefetch Primitives
// =============================================================================
//
// Cache prefetch hints for improved memory access patterns.
// Locality levels: 0=NTA (non-temporal), 1=T2, 2=T1, 3=T0 (highest)
//

#if defined(__GNUC__) || defined(__clang__)
    #define _PyGC_PREFETCH(ptr, locality) __builtin_prefetch((ptr), 0, (locality))
#elif defined(_MSC_VER)
    #include <intrin.h>
    // MSVC: _MM_HINT_T0=3, T1=2, T2=1, NTA=0 (matches GCC locality values)
    #define _PyGC_PREFETCH(ptr, locality) \
        _mm_prefetch((const char*)(ptr), (locality))
#else
    #define _PyGC_PREFETCH(ptr, locality) ((void)(ptr))
#endif

// Convenience macros matching FTP GC naming
#define _PyGC_PREFETCH_T0(ptr)  _PyGC_PREFETCH((ptr), 3)  // All cache levels
#define _PyGC_PREFETCH_T1(ptr)  _PyGC_PREFETCH((ptr), 2)  // L2 and above
#define _PyGC_PREFETCH_T2(ptr)  _PyGC_PREFETCH((ptr), 1)  // L3 and above
#define _PyGC_PREFETCH_NTA(ptr) _PyGC_PREFETCH((ptr), 0)  // Non-temporal

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
// Work Queue for Pipelined Producer-Consumer mark_alive
// =============================================================================
//
// The pipelined producer-consumer design replaces work-stealing for mark_alive.
// Main thread expands interpreter roots by one level, pushing level-1 children
// to a shared queue. Workers claim batches from the queue and traverse subtrees.
//
// Why this design?
// - Work-stealing fails for mark_alive because ~100 interpreter roots form a hub
// - First worker marks most of the heap before others can steal (99% imbalance)
// - Level-1 expansion gives thousands of distributed starting points
// - Pipelined: workers start consuming as producer pushes (no barrier needed)
//
// Block-based storage:
// - Contiguous memory blocks for cache-friendly sequential access
// - Pre-allocated initial blocks to avoid allocation during GC
// - Growable when needed for larger heaps
//

// Objects per block (32KB / 8 bytes = 4096 pointers)
#define _PyGC_QUEUE_BLOCK_SIZE 4096

// Objects per worker batch claim (balances granularity vs CAS overhead)
#define _PyGC_QUEUE_BATCH_SIZE 64

// Initial pre-allocated blocks (4096 * 8 = 32K objects without allocation)
#define _PyGC_QUEUE_INITIAL_BLOCKS 8

// Single block of object pointers
typedef struct {
    PyObject *items[_PyGC_QUEUE_BLOCK_SIZE];
} _PyGCQueueBlock;

// Work queue for producer-consumer parallel mark_alive
// Producer: main thread pushes level-1 objects
// Consumers: worker threads claim batches and traverse subtrees
typedef struct {
    // Block storage - either points to initial_blocks or heap-allocated array
    _PyGCQueueBlock *blocks;
    Py_ssize_t num_blocks;      // Number of blocks in use
    Py_ssize_t capacity;        // Total block capacity

    // Write index: next position to write (producer only, no contention)
    // Cache-line padded to avoid false sharing with read_index
    Py_ssize_t write_index;
    char _pad1[64 - sizeof(Py_ssize_t)];

    // Read index: next position to read (consumers compete via CAS)
    // Cache-line padded to avoid false sharing with write_index
    Py_ssize_t read_index;
    char _pad2[64 - sizeof(Py_ssize_t)];

    // Producer completion flag (consumers spin until set when queue empty)
    int producer_done;

    // Pre-allocated initial blocks (avoids heap allocation for typical heaps)
    _PyGCQueueBlock initial_blocks[_PyGC_QUEUE_INITIAL_BLOCKS];
} _PyGCWorkQueue;

// =============================================================================
// Counting Semaphore for Coordinator-Based Termination
// =============================================================================
//
// A counting semaphore implemented using CPython threading primitives.
// Used to wake idle workers when work becomes available, and to coordinate
// global termination detection in the work-stealing mark phase.
//
// Why a semaphore instead of condition variable?
// - Semaphore tokens persist: Post() before Wait() still wakes waiter
// - Counting: Can wake exactly N workers with Post(sema, N)
// - Simpler correctness: No lost wakeup bugs with proper ordering
//

typedef struct {
    Py_ssize_t tokens;      // Available tokens (can be negative if waiters > posts)
    PyMUTEX_T lock;         // Protects tokens and condition variable
    PyCOND_T cond;          // Workers wait here when no tokens available
} _PyGCSemaphore;

// =============================================================================
// Worker Thread Phases
// =============================================================================
// Workers wait on mark_barrier, then dispatch based on current phase.
// The main GC thread sets the phase before signalling the barrier.

typedef enum {
    _PyGC_PHASE_IDLE,               // Waiting for work
    _PyGC_PHASE_MARK_ALIVE,         // Pre-mark from interpreter roots (work-stealing)
    _PyGC_PHASE_MARK_ALIVE_QUEUE,   // Pre-mark from queue (producer-consumer)
    _PyGC_PHASE_SUBTRACT_REFS,      // Decrement gc_refs for internal references
    _PyGC_PHASE_MARK,               // Work-stealing parallel marking
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

    // Thread handle (portable via PyThread API)
    PyThread_handle_t thread;

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

    // Work queue for pipelined producer-consumer mark_alive
    // Populated during level-1 expansion, consumed by workers
    _PyGCWorkQueue work_queue;

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

    // =========================================================================
    // Coordinator-Based Termination for Work-Stealing Mark Phase
    // =========================================================================
    //
    // The mark phase uses work-stealing with coordinator-based termination:
    // - Workers process their deques and steal from others
    // - When a worker's steal fails, it tries to become the coordinator
    // - The coordinator polls all deques and wakes idle workers when work exists
    // - Termination: coordinator detects all workers idle AND no work remains
    //
    // This avoids:
    // - False termination (exiting while work remains in other deques)
    // - Wasted spins (idle workers sleep instead of spinning)
    // - Fixed-attempt heuristics (accurate termination detection)
    //

    // Count of workers actively in the marking phase
    // Termination when this reaches 0 (after coordinator decrements)
    // Cache-line padded to avoid false sharing
    int num_workers_marking;
    char _pad_marking[64 - sizeof(int)];

    // Mutex protecting coordinator election
    PyMUTEX_T steal_coord_lock;

    // Pointer to current coordinator worker (NULL if none)
    // Protected by steal_coord_lock
    _PyParallelGCWorker *steal_coordinator;

    // Semaphore for waking idle workers
    // Coordinator posts tokens when work is found or termination detected
    _PyGCSemaphore steal_sema;

    // =========================================================================

    // Flag indicating parallel GC is enabled
    int enabled;

    // Statistics for TDD/debugging
    size_t roots_found;                      // Interpreter roots found by mark_alive
    size_t roots_distributed;                // Level-1 children distributed to queue by mark_alive
    size_t gc_roots_found;                   // GC roots found by segment scanning (gc_refs > 0)
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

// Parallel mark_alive: pre-mark reachable objects from interpreter roots
// Uses work-stealing to traverse from interpreter roots (sysdict, builtins, etc.)
// Returns 1 if parallel marking was used, 0 if should fall back to serial
PyAPI_FUNC(int) _PyGC_ParallelMarkAliveFromRoots(
    PyInterpreterState *interp,
    PyGC_Head *containers
);

// Parallel mark_alive with pipelined producer-consumer design
// Main thread expands interpreter roots by one level, workers consume from queue
// Better work distribution than work-stealing for hub-structured graphs
// Returns 1 if parallel marking was used, 0 if should fall back to serial
PyAPI_FUNC(int) _PyGC_ParallelMarkAliveFromQueue(
    PyInterpreterState *interp,
    PyGC_Head *containers
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

// =============================================================================
// Work Queue Operations
// =============================================================================

// Initialise work queue with pre-allocated blocks
// Returns 0 on success, -1 on allocation failure
PyAPI_FUNC(int) _PyGCWorkQueue_Init(_PyGCWorkQueue *queue);

// Free work queue resources
PyAPI_FUNC(void) _PyGCWorkQueue_Fini(_PyGCWorkQueue *queue);

// Reset work queue for new collection (keeps capacity)
PyAPI_FUNC(void) _PyGCWorkQueue_Reset(_PyGCWorkQueue *queue);

// Push an object to the queue (producer only - not thread-safe for multiple producers)
// Returns 0 on success, -1 on allocation failure
PyAPI_FUNC(int) _PyGCWorkQueue_Push(_PyGCWorkQueue *queue, PyObject *obj);

// Signal that producer is done (no more items will be pushed)
PyAPI_FUNC(void) _PyGCWorkQueue_ProducerDone(_PyGCWorkQueue *queue);

// Claim a batch of objects from the queue (thread-safe for multiple consumers)
// Returns the number of objects claimed (0 if queue empty and producer done)
// out: array to store claimed objects (must have space for max_batch items)
// max_batch: maximum number of objects to claim
PyAPI_FUNC(Py_ssize_t) _PyGCWorkQueue_ClaimBatch(
    _PyGCWorkQueue *queue,
    PyObject **out,
    Py_ssize_t max_batch
);

// =============================================================================
// Semaphore Operations
// =============================================================================

// Initialise semaphore with 0 tokens
// Returns 0 on success, -1 on error
PyAPI_FUNC(int) _PyGCSemaphore_Init(_PyGCSemaphore *sema);

// Destroy semaphore
PyAPI_FUNC(void) _PyGCSemaphore_Fini(_PyGCSemaphore *sema);

// Post n tokens to semaphore, waking up to n waiters
PyAPI_FUNC(void) _PyGCSemaphore_Post(_PyGCSemaphore *sema, Py_ssize_t n);

// Wait for a token (blocks if none available)
PyAPI_FUNC(void) _PyGCSemaphore_Wait(_PyGCSemaphore *sema);

#endif // Py_PARALLEL_GC

#ifdef __cplusplus
}
#endif

#endif // Py_INTERNAL_GC_PARALLEL_H
