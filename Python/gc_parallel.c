// Parallel Garbage Collection for CPython
// Ported from CinderX ParallelGC by Alex Turner

#include "Python.h"

#ifdef Py_PARALLEL_GC

#include "pycore_gc_parallel.h"
#include "pycore_pystate.h"
#include "pycore_interp.h"
#include "pycore_gc.h"  // For GC internals
#include "pycore_time.h"  // For PyTime_PerfCounterRaw
#include "pycore_typeobject.h"  // For types_state and _Py_MAX_MANAGED_STATIC_* constants
#include "pycore_frame.h"  // For _PyInterpreterFrame
#include "pycore_stackref.h"  // For PyStackRef_*
#include "condvar.h"  // PyMUTEX_INIT, PyCOND_INIT, etc.

#ifdef _POSIX_THREADS
#include <pthread.h>
#include <unistd.h>
#elif defined(NT_THREADS)
#include <windows.h>
#endif

// =============================================================================
// GC Helper Functions
// =============================================================================

// Get gc_refs count from PyGC_Head
// (Mirrors gc_get_refs() from Python/gc.c)
static inline Py_ssize_t
gc_get_refs(PyGC_Head *g)
{
    return (Py_ssize_t)(g->_gc_prev >> _PyGC_PREV_SHIFT);
}

// Check if object has COLLECTING flag set (non-atomic version)
static inline int
gc_is_collecting(PyGC_Head *g)
{
    return (g->_gc_prev & _PyGC_PREV_MASK_COLLECTING) != 0;
}

// =============================================================================
// Atomic Marking Helpers for Parallel GC
// =============================================================================
//
// During parallel marking, we use the COLLECTING flag in _gc_prev to track
// which objects have been visited. The algorithm is:
//
// 1. Before parallel marking: All objects in collection set have COLLECTING=1
// 2. During marking: Workers atomically clear COLLECTING to mark as reachable
// 3. After marking: Objects still with COLLECTING=1 are unreachable
//
// We use Fetch-And (atomic AND) to atomically clear the COLLECTING bit.
// This is faster than CAS (compare-and-swap) because:
// - Fetch-And always succeeds in one operation (no retry loop needed)
// - The old value returned tells us if we were the one who cleared the bit
// - Combined with a check-first relaxed load, this minimizes contention

// Check if object has COLLECTING flag set (is in collection set and not yet marked)
static inline int
gc_is_collecting_atomic(PyGC_Head *gc)
{
    uintptr_t prev = _Py_atomic_load_uintptr_relaxed(&gc->_gc_prev);
    return (prev & _PyGC_PREV_MASK_COLLECTING) != 0;
}

// Atomically try to mark object as reachable by clearing COLLECTING flag.
// Returns 1 if we successfully marked it (we should process it).
// Returns 0 if already marked by another worker (skip it).
//
// OPTIMIZATION: Uses Fetch-And instead of CAS for better performance.
// - CAS can fail under contention and needs retry loops
// - Fetch-And always succeeds in one atomic operation
// - The old value tells us if we were the one who cleared the bit
//
// Additional optimization: Check-first with relaxed load.
// - Relaxed load is ~10x cheaper than atomic RMW
// - Significant win for shared objects (types, builtins, modules)
// - Common case: object already marked by another worker
static inline int
gc_try_mark_reachable_atomic(PyGC_Head *gc)
{
    // Fast path: check if already marked reachable (relaxed load - very cheap)
    uintptr_t prev = _Py_atomic_load_uintptr_relaxed(&gc->_gc_prev);
    if (!(prev & _PyGC_PREV_MASK_COLLECTING)) {
        // Already marked reachable, skip expensive atomic RMW
        return 0;
    }

    // Slow path: still has COLLECTING (or race), do atomic clear
    // Fetch-And atomically clears the bit and returns the OLD value
    uintptr_t old_prev = _Py_atomic_and_uintptr(
        &gc->_gc_prev,
        ~_PyGC_PREV_MASK_COLLECTING
    );

    // If old value had COLLECTING set, we successfully claimed this object
    // If old value didn't have COLLECTING, another worker beat us (race case)
    return (old_prev & _PyGC_PREV_MASK_COLLECTING) != 0;
}

// =============================================================================
// Barrier Synchronization
// =============================================================================
// Note: _PyGCBarrier_Init, _PyGCBarrier_Fini, _PyGCBarrier_Wait are now
// inline functions defined in pycore_gc_barrier.h (shared with FTP build)

// =============================================================================
// Worker Thread Function
// =============================================================================

// Visit function called by tp_traverse() to discover child objects
// This is called for each object reference during traversal
//
// IMPORTANT: We use Fetch-And to atomically mark objects as reachable:
// 1. Check if object is in collection set (has COLLECTING flag)
// 2. Atomically clear COLLECTING flag (marks as reachable)
// 3. If we cleared the flag (old value had it set), enqueue for traversal
// 4. If flag was already clear, another worker already claimed it - skip
//
// OPTIMIZATION: Uses local buffer to avoid expensive deque operations.
// Push to local buffer first (zero fences), only flush to deque when full.
static int
_parallel_gc_visit_and_enqueue(PyObject *op, void *arg)
{
    _PyParallelGCWorker *worker = (_PyParallelGCWorker *)arg;

    // Skip non-GC objects (e.g., ints, strings, etc.)
    if (!PyObject_IS_GC(op)) {
        return 0;
    }

    PyGC_Head *gc = _Py_AS_GC(op);

    // Check if object is in collection set and try to mark as reachable
    // Objects not in collection set (COLLECTING flag not set) are either:
    // - In a different generation (skip)
    // - Already marked reachable by another worker (skip)
    if (!gc_try_mark_reachable_atomic(gc)) {
        // Object not in collection set, or already marked by another worker
        return 0;
    }

    // We successfully marked this object as reachable
    worker->objects_discovered++;

    // Local buffer with chunked striping
    // Push to local buffer first (zero fences, just array indexing)
    if (_PyGCLocalBuffer_IsFull(&worker->local_buffer)) {
        // Local buffer full - use compile-time selected flush strategy
        _PyGC_OverflowFlush(&worker->local_buffer, &worker->deque);
    }
    _PyGCLocalBuffer_Push(&worker->local_buffer, op);

    return 0;  // Continue traversal
}

// =============================================================================
// Batch Operations (use shared implementations from pycore_ws_deque.h)
// =============================================================================

// Wrapper: refill local buffer from own deque
static inline void
refill_local_buffer_from_deque(_PyParallelGCWorker *worker)
{
    _PyGC_RefillLocalFromDeque(&worker->local_buffer, &worker->deque);
}

// Wrapper: batch steal from another worker's deque
static inline size_t
steal_batch_from_worker(_PyParallelGCWorker *thief, _PyParallelGCWorker *victim)
{
    return _PyGC_BatchSteal(&thief->local_buffer, &victim->deque);
}

// =============================================================================
// Forward Declarations
// =============================================================================

// Worker function for parallel subtract_refs (defined later)
static void
_parallel_subtract_refs_worker(_PyParallelGCWorker *worker,
                               PyGC_Head *start,
                               PyGC_Head *end);

// Coordinator-based termination functions (defined after semaphore implementation)
static int _PyGCWorker_TakeCoordinator(_PyParallelGCWorker *worker);
static void _PyGCWorker_DropCoordinator(_PyParallelGCWorker *worker);
static void _PyGCWorker_CoordinateStealing(_PyParallelGCWorker *worker);

// Worker thread entry point
static void *
_parallel_gc_worker_thread(void *arg)
{
    _PyParallelGCWorker *worker = (_PyParallelGCWorker *)arg;
    _PyParallelGCState *par_gc = worker->par_gc;

    // Create a Python thread state for this worker thread.
    // This is required for Py_REF_DEBUG (debug builds) where Py_INCREF/Py_DECREF
    // call _Py_INCREF_IncRefTotal() which needs _PyThreadState_GET() to return
    // a valid thread state. Without this, tp_traverse callbacks that call
    // Py_INCREF (like ctypes) would crash.
    PyThreadState *tstate = _PyThreadState_New(par_gc->interp, _PyThreadState_WHENCE_UNKNOWN);
    if (tstate != NULL) {
        _PyThreadState_Bind(tstate);
        // Set thread-local storage so _PyThreadState_GET() returns our tstate
        _Py_tss_tstate = tstate;
        _Py_tss_interp = par_gc->interp;
        worker->tstate = tstate;
    }
    // If tstate creation failed, continue anyway - will crash on Py_INCREF
    // in debug builds, but that's better than failing silently

    // Signal that we're ready - this synchronizes with ParallelStart()
    // to ensure all workers are initialized before Start() returns
    _PyGCBarrier_Wait(&par_gc->startup_barrier);

    while (!_Py_atomic_load_int(&worker->should_exit)) {
        // =======================================================================
        // Wait for start signal from main thread
        // =======================================================================
        // Workers wait on mark_barrier until main thread signals work is ready
        _PyGCBarrier_Wait(&par_gc->mark_barrier);

        // Check if we should exit (signaled during shutdown)
        if (_Py_atomic_load_int(&worker->should_exit)) {
            break;
        }

        // =======================================================================
        // Dispatch based on current phase
        // =======================================================================
        switch (worker->phase) {
        case _PyGC_PHASE_SUBTRACT_REFS:
            // Process assigned segment: call tp_traverse with atomic decref
            _parallel_subtract_refs_worker(worker, worker->slice_start, worker->slice_end);
            break;

        case _PyGC_PHASE_MARK_ALIVE_QUEUE:
            // =======================================================================
            // Pipelined producer-consumer mark_alive
            // =======================================================================
            // Workers claim batches from the shared work queue (filled by main thread)
            // and traverse subtrees locally. No work-stealing between workers.
            {
                PyObject *batch[_PyGC_QUEUE_BATCH_SIZE];
                _PyGCWorkQueue *queue = &par_gc->work_queue;

                while (1) {
                    // Claim a batch from the shared queue
                    Py_ssize_t count = _PyGCWorkQueue_ClaimBatch(
                        queue, batch, _PyGC_QUEUE_BATCH_SIZE);

                    if (count == 0) {
                        break;  // Queue empty and producer done
                    }

                    // Process each object in the batch
                    for (Py_ssize_t i = 0; i < count; i++) {
                        PyObject *obj = batch[i];
                        worker->objects_marked++;

                        // Call tp_traverse to discover children
                        // Children are enqueued to our local buffer/deque
                        traverseproc traverse = Py_TYPE(obj)->tp_traverse;
                        if (traverse != NULL) {
                            traverse(obj, (visitproc)_parallel_gc_visit_and_enqueue, worker);
                            worker->traversals_performed++;
                        }
                    }

                    // Process any locally discovered children (depth-first)
                    // This drains our local buffer/deque before claiming more from queue
                    while (!_PyGCLocalBuffer_IsEmpty(&worker->local_buffer)) {
                        PyObject *obj = _PyGCLocalBuffer_Pop(&worker->local_buffer);
                        worker->objects_marked++;

                        traverseproc traverse = Py_TYPE(obj)->tp_traverse;
                        if (traverse != NULL) {
                            traverse(obj, (visitproc)_parallel_gc_visit_and_enqueue, worker);
                            worker->traversals_performed++;
                        }
                    }

                    // Also drain own deque if local buffer was emptied
                    refill_local_buffer_from_deque(worker);
                    while (!_PyGCLocalBuffer_IsEmpty(&worker->local_buffer)) {
                        PyObject *obj = _PyGCLocalBuffer_Pop(&worker->local_buffer);
                        worker->objects_marked++;

                        traverseproc traverse = Py_TYPE(obj)->tp_traverse;
                        if (traverse != NULL) {
                            traverse(obj, (visitproc)_parallel_gc_visit_and_enqueue, worker);
                            worker->traversals_performed++;
                        }
                    }
                }
            }
            break;

        case _PyGC_PHASE_MARK_ALIVE:
            // Falls through to PHASE_MARK - same work-stealing traversal logic
            // The only difference is how roots are distributed (interpreter roots
            // vs objects with gc_refs > 0), but traversal is identical
        case _PyGC_PHASE_MARK:
        default:
            // =======================================================================
            // Main marking loop with coordinator-based termination
            // =======================================================================
            // Phase 1: Process local buffer (fast path - no fences)
            // Phase 2: Refill from own deque (batch pull)
            // Phase 3: Batch steal from other workers
            // Termination: Coordinator detects all workers idle AND no work remains
            //
            // This replaces the old fixed-attempt termination (exit after N failed
            // steals) with accurate termination detection that avoids both:
            // - False termination (exiting while work remains in other deques)
            // - Wasted spins (idle workers sleep instead of spinning)
            {
                int consecutive_failed_steals = 0;

                do {
                    // ===========================================================
                    // Inner work loop: process local buffer, deque, and steal
                    // ===========================================================
                    while (1) {
                        // ================================================================
                        // PHASE 1: Process local buffer (FAST PATH - zero fences!)
                        // ================================================================
                        while (!_PyGCLocalBuffer_IsEmpty(&worker->local_buffer)) {
                            consecutive_failed_steals = 0;  // Got work, reset counter
                            PyObject *obj = _PyGCLocalBuffer_Pop(&worker->local_buffer);

                            // Prefetch next object's type to hide memory latency
                            if (!_PyGCLocalBuffer_IsEmpty(&worker->local_buffer)) {
                                PyObject *next_obj = worker->local_buffer.items[
                                    worker->local_buffer.count - 1];
                                _PyGC_PREFETCH_T2(Py_TYPE(next_obj));
                            }

                            worker->objects_marked++;

                            // Call tp_traverse to discover children
                            traverseproc traverse = Py_TYPE(obj)->tp_traverse;
                            if (traverse != NULL) {
                                traverse(obj, (visitproc)_parallel_gc_visit_and_enqueue, worker);
                                worker->traversals_performed++;
                            }
                        }

                        // ================================================================
                        // PHASE 2: Refill from own deque (batch pull ~512 items)
                        // ================================================================
                        refill_local_buffer_from_deque(worker);
                        if (!_PyGCLocalBuffer_IsEmpty(&worker->local_buffer)) {
                            continue;  // Got work from deque, continue Phase 1
                        }

                        // ================================================================
                        // PHASE 3: Batch steal from other workers (~512 items)
                        // ================================================================
                        // Try to steal from a random victim
                        worker->steal_seed = worker->steal_seed * 1103515245 + 12345;
                        size_t victim_id = (worker->steal_seed / 65536) % par_gc->num_workers;
                        if (victim_id == worker->thread_id) {
                            victim_id = (victim_id + 1) % par_gc->num_workers;
                        }

                        _PyParallelGCWorker *victim = &par_gc->workers[victim_id];
                        worker->steal_attempts++;

                        size_t batch_stolen = steal_batch_from_worker(worker, victim);
                        if (batch_stolen > 0) {
                            worker->steal_successes += batch_stolen;
                            consecutive_failed_steals = 0;  // Got work!
                            // Continue to Phase 1 with stolen work
                        }
                        else {
                            consecutive_failed_steals++;
                            // After trying each worker once, go to coordinator election
                            if (consecutive_failed_steals >= (int)par_gc->num_workers) {
                                break;  // Exit inner loop, try coordinator election
                            }
                            // Brief backoff before trying next victim
                            for (int i = 0; i < 4; i++) {
                                _Py_cpu_relax();
                            }
                        }
                    }

                    // ===========================================================
                    // Coordinator election: try to become coordinator or sleep
                    // ===========================================================
                    if (_PyGCWorker_TakeCoordinator(worker)) {
                        // We are the coordinator: poll deques, wake workers or terminate
                        _PyGCWorker_CoordinateStealing(worker);
                        _PyGCWorker_DropCoordinator(worker);
                        consecutive_failed_steals = 0;  // May have woken workers, retry
                    }
                    else {
                        // Another worker is coordinator, sleep until woken
                        // CRITICAL: decrement BEFORE waiting (coordinator sees accurate count)
                        _Py_atomic_add_int(&par_gc->num_workers_marking, -1);
                        _PyGCSemaphore_Wait(&par_gc->steal_sema);
                        consecutive_failed_steals = 0;  // Woken up, retry stealing
                    }

                } while (_Py_atomic_load_int_acquire(&par_gc->num_workers_marking) > 0);
            }
            break;

        case _PyGC_PHASE_IDLE:
            // Worker has no work this phase, just wait at barrier
            break;
        }

        // =======================================================================
        // Signal completion to main thread
        // =======================================================================
        // Workers wait on done_barrier until all workers finish
        _PyGCBarrier_Wait(&par_gc->done_barrier);
    }

    // Clean up Python thread state
    // The order here is critical:
    // 1. PyThreadState_Clear() requires current_fast_get()->interp == tstate->interp
    //    so we MUST still be the current thread (TLS set)
    // 2. PyThreadState_Delete() calls tstate_verify_not_active() which requires
    //    the tstate is NOT the current thread (TLS cleared)
    if (worker->tstate != NULL) {
        // First clear while we're still the current thread
        PyThreadState_Clear(worker->tstate);
        // Now clear TLS so we're no longer "current"
        _Py_tss_tstate = NULL;
        _Py_tss_interp = NULL;
        // Now delete (requires that we're NOT the current thread)
        PyThreadState_Delete(worker->tstate);
        worker->tstate = NULL;
    }

    return NULL;
}

// =============================================================================
// Thread Pool Lifecycle
// =============================================================================

int
_PyGC_ParallelInit(PyInterpreterState *interp, size_t num_workers)
{
    if (num_workers == 0 || num_workers > _PyGC_MAX_WORKERS) {
        PyErr_SetString(PyExc_ValueError,
                       "num_workers must be between 1 and _PyGC_MAX_WORKERS");
        return -1;
    }

    // Allocate parallel GC state
    size_t state_size = sizeof(_PyParallelGCState) +
                       num_workers * sizeof(_PyParallelGCWorker);
    _PyParallelGCState *par_gc = (_PyParallelGCState *)PyMem_Calloc(1, state_size);
    if (par_gc == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    par_gc->interp = interp;
    par_gc->num_workers = num_workers;
    par_gc->enabled = 1;
    par_gc->num_workers_active = 0;

    // Initialize split vector for work distribution
    if (_PyGCSplitVector_Init(&par_gc->split_vector) < 0) {
        PyMem_Free(par_gc);
        PyErr_NoMemory();
        return -1;
    }

    // Initialize work queue for pipelined producer-consumer mark_alive
    if (_PyGCWorkQueue_Init(&par_gc->work_queue) < 0) {
        _PyGCSplitVector_Fini(&par_gc->split_vector);
        PyMem_Free(par_gc);
        PyErr_NoMemory();
        return -1;
    }

    // Initialize barriers
    // mark_barrier: all workers + main thread (to signal start)
    // done_barrier: all workers + main thread (to signal completion)
    // startup_barrier: all workers + main thread (to ensure workers are ready)
    _PyGCBarrier_Init(&par_gc->mark_barrier, (int)num_workers + 1);
    _PyGCBarrier_Init(&par_gc->done_barrier, (int)num_workers + 1);
    _PyGCBarrier_Init(&par_gc->startup_barrier, (int)num_workers + 1);

    // Initialize locks
    PyMUTEX_INIT(&par_gc->active_lock);
    PyCOND_INIT(&par_gc->workers_done_cond);

    // Initialize coordinator-based termination state
    par_gc->num_workers_marking = 0;
    PyMUTEX_INIT(&par_gc->steal_coord_lock);
    par_gc->steal_coordinator = NULL;
    if (_PyGCSemaphore_Init(&par_gc->steal_sema) < 0) {
        PyCOND_FINI(&par_gc->workers_done_cond);
        PyMUTEX_FINI(&par_gc->active_lock);
        _PyGCBarrier_Fini(&par_gc->startup_barrier);
        _PyGCBarrier_Fini(&par_gc->done_barrier);
        _PyGCBarrier_Fini(&par_gc->mark_barrier);
        _PyGCWorkQueue_Fini(&par_gc->work_queue);
        _PyGCSplitVector_Fini(&par_gc->split_vector);
        PyMem_Free(par_gc);
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize steal semaphore");
        return -1;
    }

    // Initialize workers with thread-local memory pools
    // Each worker gets a 2MB pre-allocated buffer for their deque
    // This eliminates malloc/calloc calls during the hot path of collections
    size_t pool_entries = _Py_WSDEQUE_PARALLEL_GC_SIZE;  // 256K entries = 2MB
    size_t pool_bytes = sizeof(_PyWSArray) + sizeof(uintptr_t) * pool_entries;

    for (size_t i = 0; i < num_workers; i++) {
        _PyParallelGCWorker *worker = &par_gc->workers[i];

        // Allocate thread-local pool (done once at enable time, not per-collection)
        worker->local_pool = PyMem_RawCalloc(1, pool_bytes);
        worker->local_pool_size = pool_entries;
        worker->pool_overflows = 0;

        if (worker->local_pool != NULL) {
            // Initialize deque with pre-allocated buffer
            worker->local_pool_in_use = _PyWSDeque_InitWithBuffer(
                &worker->deque,
                worker->local_pool,
                pool_bytes,
                pool_entries
            );
        } else {
            // Fallback to regular init if pool allocation failed
            worker->local_pool_in_use = 0;
            _PyWSDeque_Init(&worker->deque);
        }

        worker->objects_marked = 0;
        worker->steal_attempts = 0;
        worker->steal_successes = 0;
        worker->objects_discovered = 0;
        worker->traversals_performed = 0;
        worker->roots_in_slice = 0;
        worker->slice_start = NULL;
        worker->slice_end = NULL;
        worker->work_start_ns = 0;           // Per-worker profiling
        worker->work_end_ns = 0;             // Per-worker profiling
        worker->objects_in_segment = 0;      // Per-worker profiling
        _PyGCLocalBuffer_Reset(&worker->local_buffer);  // Initialize local buffer
        worker->steal_seed = (unsigned int)(i + 1);
        worker->par_gc = par_gc;
        worker->thread_id = i;
        worker->tstate = NULL;  // Will be created when worker thread starts
        worker->phase = _PyGC_PHASE_IDLE;  // Start idle, main thread sets phase
        _Py_atomic_store_int(&worker->should_exit, 0);
    }

    // Store in interpreter state
    interp->gc.parallel_gc = par_gc;

    return 0;
}

void
_PyGC_ParallelFini(PyInterpreterState *interp)
{
    // Get from interpreter state
    _PyParallelGCState *par_gc = interp->gc.parallel_gc;

    if (par_gc == NULL) {
        return;
    }

    // Make sure workers are stopped
    _PyGC_ParallelStop(interp);

    // Clean up workers
    for (size_t i = 0; i < par_gc->num_workers; i++) {
        _PyParallelGCWorker *worker = &par_gc->workers[i];

        // Clean up deque - use external version if we're using local pool
        if (worker->local_pool_in_use) {
            _PyWSDeque_FiniExternal(&worker->deque, worker->local_pool);
        } else {
            _PyWSDeque_Fini(&worker->deque);
        }

        // Free the local pool
        if (worker->local_pool != NULL) {
            PyMem_RawFree(worker->local_pool);
            worker->local_pool = NULL;
        }
    }

    // Clean up barriers
    _PyGCBarrier_Fini(&par_gc->mark_barrier);
    _PyGCBarrier_Fini(&par_gc->done_barrier);
    _PyGCBarrier_Fini(&par_gc->startup_barrier);

    // Clean up split vector
    _PyGCSplitVector_Fini(&par_gc->split_vector);

    // Clean up work queue
    _PyGCWorkQueue_Fini(&par_gc->work_queue);

    // Clean up coordinator-based termination state
    _PyGCSemaphore_Fini(&par_gc->steal_sema);
    PyMUTEX_FINI(&par_gc->steal_coord_lock);

    // Clean up locks
    PyMUTEX_FINI(&par_gc->active_lock);
    PyCOND_FINI(&par_gc->workers_done_cond);

    // Free state
    PyMem_Free(par_gc);

    // Clear from interpreter state
    interp->gc.parallel_gc = NULL;
}

int
_PyGC_ParallelStart(PyInterpreterState *interp)
{
    // Get from interpreter state
    _PyParallelGCState *par_gc = interp->gc.parallel_gc;

    if (par_gc == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                       "Parallel GC not initialized");
        return -1;
    }

    // Start worker threads
    for (size_t i = 0; i < par_gc->num_workers; i++) {
        _PyParallelGCWorker *worker = &par_gc->workers[i];

#ifdef _POSIX_THREADS
        int rc = pthread_create(&worker->thread, NULL,
                               _parallel_gc_worker_thread, worker);
        if (rc != 0) {
            PyErr_Format(PyExc_RuntimeError,
                        "Failed to create worker thread %zu: error %d",
                        i, rc);
            // Note: Already-created threads are not cleaned up here.
            // This is acceptable because thread creation failure during init
            // is a fatal error - the interpreter cannot function properly.
            return -1;
        }
#elif defined(NT_THREADS)
        worker->thread = CreateThread(NULL, 0,
                                     (LPTHREAD_START_ROUTINE)_parallel_gc_worker_thread,
                                     worker, 0, NULL);
        if (worker->thread == NULL) {
            PyErr_Format(PyExc_RuntimeError,
                        "Failed to create worker thread %zu", i);
            return -1;
        }
#else
#error "Unsupported threading platform"
#endif

        par_gc->num_workers_active++;
    }

    // Wait for all workers to be ready before returning
    // This ensures ParallelStop won't race with worker initialization
    _PyGCBarrier_Wait(&par_gc->startup_barrier);

    return 0;
}

void
_PyGC_ParallelStop(PyInterpreterState *interp)
{
    // Get from interpreter state
    _PyParallelGCState *par_gc = interp->gc.parallel_gc;

    if (par_gc == NULL || par_gc->num_workers_active == 0) {
        return;
    }

    // Signal workers to exit
    for (size_t i = 0; i < par_gc->num_workers; i++) {
        _Py_atomic_store_int(&par_gc->workers[i].should_exit, 1);
    }

    // Wake up workers from mark_barrier so they can check should_exit and exit
    // Workers check should_exit right after mark_barrier and break if set
    // They will NOT reach done_barrier when exiting, so we only signal mark_barrier
    _PyGCBarrier_Wait(&par_gc->mark_barrier);

    // Wait for workers to finish (they exit after seeing should_exit)
    for (size_t i = 0; i < par_gc->num_workers; i++) {
        _PyParallelGCWorker *worker = &par_gc->workers[i];

#ifdef _POSIX_THREADS
        pthread_join(worker->thread, NULL);
#elif defined(NT_THREADS)
        WaitForSingleObject(worker->thread, INFINITE);
        CloseHandle(worker->thread);
#endif

        // Reset should_exit for potential restart
        _Py_atomic_store_int(&worker->should_exit, 0);
    }

    par_gc->num_workers_active = 0;
}

int
_PyGC_ParallelIsEnabled(PyInterpreterState *interp)
{
    // Get from interpreter state
    _PyParallelGCState *par_gc = interp->gc.parallel_gc;

    return (par_gc != NULL && par_gc->enabled);
}

void
_PyGC_ParallelSetEnabled(PyInterpreterState *interp, int enabled)
{
    _PyParallelGCState *par_gc = interp->gc.parallel_gc;
    if (par_gc != NULL) {
        par_gc->enabled = enabled;
    }
}

PyObject *
_PyGC_ParallelGetConfig(PyInterpreterState *interp)
{
    // Get from interpreter state
    _PyParallelGCState *par_gc = interp->gc.parallel_gc;

    PyObject *result = PyDict_New();
    if (result == NULL) {
        return NULL;
    }

    // Always available since we're built with Py_PARALLEL_GC
    if (PyDict_SetItemString(result, "available", Py_True) < 0) {
        Py_DECREF(result);
        return NULL;
    }

    int enabled = (par_gc != NULL && par_gc->enabled);
    int num_workers = (par_gc != NULL) ? (int)par_gc->num_workers : 0;

    if (PyDict_SetItemString(result, "enabled",
                            enabled ? Py_True : Py_False) < 0) {
        Py_DECREF(result);
        return NULL;
    }

    PyObject *workers_obj = PyLong_FromLong(num_workers);
    if (workers_obj == NULL) {
        Py_DECREF(result);
        return NULL;
    }

    if (PyDict_SetItemString(result, "num_workers", workers_obj) < 0) {
        Py_DECREF(workers_obj);
        Py_DECREF(result);
        return NULL;
    }
    Py_DECREF(workers_obj);

    return result;
}

PyObject *
_PyGC_ParallelGetStats(PyInterpreterState *interp)
{
    // Get from interpreter state
    _PyParallelGCState *par_gc = interp->gc.parallel_gc;

    PyObject *result = PyDict_New();
    if (result == NULL) {
        return NULL;
    }

    // If parallel GC not enabled, return empty stats
    if (par_gc == NULL || !par_gc->enabled) {
        if (PyDict_SetItemString(result, "enabled", Py_False) < 0) {
            Py_DECREF(result);
            return NULL;
        }
        return result;
    }

    // Add enabled flag
    if (PyDict_SetItemString(result, "enabled", Py_True) < 0) {
        Py_DECREF(result);
        return NULL;
    }

    // Add num_workers
    PyObject *num_workers_obj = PyLong_FromSize_t(par_gc->num_workers);
    if (num_workers_obj == NULL) {
        Py_DECREF(result);
        return NULL;
    }
    if (PyDict_SetItemString(result, "num_workers", num_workers_obj) < 0) {
        Py_DECREF(num_workers_obj);
        Py_DECREF(result);
        return NULL;
    }
    Py_DECREF(num_workers_obj);

    // Add global statistics
    PyObject *roots_found_obj = PyLong_FromSize_t(par_gc->roots_found);
    if (roots_found_obj == NULL) {
        Py_DECREF(result);
        return NULL;
    }
    if (PyDict_SetItemString(result, "roots_found", roots_found_obj) < 0) {
        Py_DECREF(roots_found_obj);
        Py_DECREF(result);
        return NULL;
    }
    Py_DECREF(roots_found_obj);

    PyObject *roots_distributed_obj = PyLong_FromSize_t(par_gc->roots_distributed);
    if (roots_distributed_obj == NULL) {
        Py_DECREF(result);
        return NULL;
    }
    if (PyDict_SetItemString(result, "roots_distributed", roots_distributed_obj) < 0) {
        Py_DECREF(roots_distributed_obj);
        Py_DECREF(result);
        return NULL;
    }
    Py_DECREF(roots_distributed_obj);

    PyObject *collections_attempted_obj = PyLong_FromSize_t(par_gc->parallel_collections_attempted);
    if (collections_attempted_obj == NULL) {
        Py_DECREF(result);
        return NULL;
    }
    if (PyDict_SetItemString(result, "collections_attempted", collections_attempted_obj) < 0) {
        Py_DECREF(collections_attempted_obj);
        Py_DECREF(result);
        return NULL;
    }
    Py_DECREF(collections_attempted_obj);

    PyObject *collections_succeeded_obj = PyLong_FromSize_t(par_gc->parallel_collections_succeeded);
    if (collections_succeeded_obj == NULL) {
        Py_DECREF(result);
        return NULL;
    }
    if (PyDict_SetItemString(result, "collections_succeeded", collections_succeeded_obj) < 0) {
        Py_DECREF(collections_succeeded_obj);
        Py_DECREF(result);
        return NULL;
    }
    Py_DECREF(collections_succeeded_obj);

    // Calculate total objects traversed (sum of all workers' discoveries)
    unsigned long total_objects_traversed = 0;
    for (size_t i = 0; i < par_gc->num_workers; i++) {
        total_objects_traversed += par_gc->workers[i].objects_discovered;
    }
    PyObject *objects_traversed_obj = PyLong_FromUnsignedLong(total_objects_traversed);
    if (objects_traversed_obj == NULL) {
        Py_DECREF(result);
        return NULL;
    }
    if (PyDict_SetItemString(result, "objects_traversed", objects_traversed_obj) < 0) {
        Py_DECREF(objects_traversed_obj);
        Py_DECREF(result);
        return NULL;
    }
    Py_DECREF(objects_traversed_obj);

    // Add per-worker statistics
    PyObject *workers_list = PyList_New(par_gc->num_workers);
    if (workers_list == NULL) {
        Py_DECREF(result);
        return NULL;
    }

    for (size_t i = 0; i < par_gc->num_workers; i++) {
        _PyParallelGCWorker *worker = &par_gc->workers[i];

        PyObject *worker_dict = PyDict_New();
        if (worker_dict == NULL) {
            Py_DECREF(workers_list);
            Py_DECREF(result);
            return NULL;
        }

        PyObject *objects_marked_obj = PyLong_FromUnsignedLong(worker->objects_marked);
        if (objects_marked_obj == NULL) {
            Py_DECREF(worker_dict);
            Py_DECREF(workers_list);
            Py_DECREF(result);
            return NULL;
        }
        if (PyDict_SetItemString(worker_dict, "objects_marked", objects_marked_obj) < 0) {
            Py_DECREF(objects_marked_obj);
            Py_DECREF(worker_dict);
            Py_DECREF(workers_list);
            Py_DECREF(result);
            return NULL;
        }
        Py_DECREF(objects_marked_obj);

        PyObject *steal_attempts_obj = PyLong_FromUnsignedLong(worker->steal_attempts);
        if (steal_attempts_obj == NULL) {
            Py_DECREF(worker_dict);
            Py_DECREF(workers_list);
            Py_DECREF(result);
            return NULL;
        }
        if (PyDict_SetItemString(worker_dict, "steal_attempts", steal_attempts_obj) < 0) {
            Py_DECREF(steal_attempts_obj);
            Py_DECREF(worker_dict);
            Py_DECREF(workers_list);
            Py_DECREF(result);
            return NULL;
        }
        Py_DECREF(steal_attempts_obj);

        PyObject *steal_successes_obj = PyLong_FromUnsignedLong(worker->steal_successes);
        if (steal_successes_obj == NULL) {
            Py_DECREF(worker_dict);
            Py_DECREF(workers_list);
            Py_DECREF(result);
            return NULL;
        }
        if (PyDict_SetItemString(worker_dict, "steal_successes", steal_successes_obj) < 0) {
            Py_DECREF(steal_successes_obj);
            Py_DECREF(worker_dict);
            Py_DECREF(workers_list);
            Py_DECREF(result);
            return NULL;
        }
        Py_DECREF(steal_successes_obj);

        // Traversal statistics
        PyObject *objects_discovered_obj = PyLong_FromUnsignedLong(worker->objects_discovered);
        if (objects_discovered_obj == NULL) {
            Py_DECREF(worker_dict);
            Py_DECREF(workers_list);
            Py_DECREF(result);
            return NULL;
        }
        if (PyDict_SetItemString(worker_dict, "objects_discovered", objects_discovered_obj) < 0) {
            Py_DECREF(objects_discovered_obj);
            Py_DECREF(worker_dict);
            Py_DECREF(workers_list);
            Py_DECREF(result);
            return NULL;
        }
        Py_DECREF(objects_discovered_obj);

        PyObject *traversals_performed_obj = PyLong_FromUnsignedLong(worker->traversals_performed);
        if (traversals_performed_obj == NULL) {
            Py_DECREF(worker_dict);
            Py_DECREF(workers_list);
            Py_DECREF(result);
            return NULL;
        }
        if (PyDict_SetItemString(worker_dict, "traversals_performed", traversals_performed_obj) < 0) {
            Py_DECREF(traversals_performed_obj);
            Py_DECREF(worker_dict);
            Py_DECREF(workers_list);
            Py_DECREF(result);
            return NULL;
        }
        Py_DECREF(traversals_performed_obj);

        // Static slicing statistics
        PyObject *roots_in_slice_obj = PyLong_FromUnsignedLong(worker->roots_in_slice);
        if (roots_in_slice_obj == NULL) {
            Py_DECREF(worker_dict);
            Py_DECREF(workers_list);
            Py_DECREF(result);
            return NULL;
        }
        if (PyDict_SetItemString(worker_dict, "roots_in_slice", roots_in_slice_obj) < 0) {
            Py_DECREF(roots_in_slice_obj);
            Py_DECREF(worker_dict);
            Py_DECREF(workers_list);
            Py_DECREF(result);
            return NULL;
        }
        Py_DECREF(roots_in_slice_obj);

        // Per-worker profiling: timing and segment info
        PyObject *work_time_ns_obj = PyLong_FromLongLong(worker->work_end_ns - worker->work_start_ns);
        if (work_time_ns_obj == NULL) {
            Py_DECREF(worker_dict);
            Py_DECREF(workers_list);
            Py_DECREF(result);
            return NULL;
        }
        if (PyDict_SetItemString(worker_dict, "work_time_ns", work_time_ns_obj) < 0) {
            Py_DECREF(work_time_ns_obj);
            Py_DECREF(worker_dict);
            Py_DECREF(workers_list);
            Py_DECREF(result);
            return NULL;
        }
        Py_DECREF(work_time_ns_obj);

        PyObject *objects_in_segment_obj = PyLong_FromUnsignedLong(worker->objects_in_segment);
        if (objects_in_segment_obj == NULL) {
            Py_DECREF(worker_dict);
            Py_DECREF(workers_list);
            Py_DECREF(result);
            return NULL;
        }
        if (PyDict_SetItemString(worker_dict, "objects_in_segment", objects_in_segment_obj) < 0) {
            Py_DECREF(objects_in_segment_obj);
            Py_DECREF(worker_dict);
            Py_DECREF(workers_list);
            Py_DECREF(result);
            return NULL;
        }
        Py_DECREF(objects_in_segment_obj);

        PyList_SET_ITEM(workers_list, i, worker_dict);
    }

    if (PyDict_SetItemString(result, "workers", workers_list) < 0) {
        Py_DECREF(workers_list);
        Py_DECREF(result);
        return NULL;
    }
    Py_DECREF(workers_list);

    // Add phase timing (nanoseconds)
    PyObject *phase_timing = PyDict_New();
    if (phase_timing == NULL) {
        Py_DECREF(result);
        return NULL;
    }

    // Calculate phase durations from recorded timestamps
    // Only populate if timing_valid is set (complete parallel collection ran)
    int64_t update_refs_ns = 0;
    int64_t mark_alive_ns = 0;
    int64_t subtract_refs_ns = 0;
    int64_t mark_ns = 0;
    int64_t cleanup_ns = 0;
    int64_t total_ns = 0;

    if (par_gc->timing_valid) {
        update_refs_ns = par_gc->update_refs_end_ns - par_gc->gc_start_ns;
        mark_alive_ns = par_gc->mark_alive_end_ns - par_gc->update_refs_end_ns;
        subtract_refs_ns = par_gc->subtract_refs_end_ns - par_gc->mark_alive_end_ns;
        mark_ns = par_gc->mark_end_ns - par_gc->subtract_refs_end_ns;
        cleanup_ns = par_gc->cleanup_end_ns - par_gc->mark_end_ns;
        total_ns = par_gc->cleanup_end_ns - par_gc->gc_start_ns;
    }

    // Helper macro to add int64 to dict
    #define ADD_TIMING(name, value) do { \
        PyObject *obj = PyLong_FromLongLong(value); \
        if (obj == NULL || PyDict_SetItemString(phase_timing, name, obj) < 0) { \
            Py_XDECREF(obj); \
            Py_DECREF(phase_timing); \
            Py_DECREF(result); \
            return NULL; \
        } \
        Py_DECREF(obj); \
    } while (0)

    ADD_TIMING("update_refs_ns", update_refs_ns);
    ADD_TIMING("mark_alive_ns", mark_alive_ns);
    ADD_TIMING("subtract_refs_ns", subtract_refs_ns);
    ADD_TIMING("mark_ns", mark_ns);
    ADD_TIMING("cleanup_ns", cleanup_ns);
    ADD_TIMING("total_ns", total_ns);

    #undef ADD_TIMING

    if (PyDict_SetItemString(result, "phase_timing", phase_timing) < 0) {
        Py_DECREF(phase_timing);
        Py_DECREF(result);
        return NULL;
    }
    Py_DECREF(phase_timing);

    return result;
}


// =============================================================================
// Split Vector Operations
// =============================================================================
//
// The split vector records pointers into the GC list at regular intervals
// during the serial update_refs phase. This allows parallel subtract_refs
// to quickly find evenly-spaced start/end positions for each worker.

int
_PyGCSplitVector_Init(_PyGCSplitVector *vec)
{
    vec->entries = (PyGC_Head **)PyMem_RawCalloc(
        _PyGC_SPLIT_VECTOR_INITIAL_CAPACITY, sizeof(PyGC_Head *));
    if (vec->entries == NULL) {
        return -1;
    }
    vec->count = 0;
    vec->capacity = _PyGC_SPLIT_VECTOR_INITIAL_CAPACITY;
    vec->interval = _PyGC_SPLIT_INTERVAL;
    return 0;
}

void
_PyGCSplitVector_Fini(_PyGCSplitVector *vec)
{
    if (vec->entries != NULL) {
        PyMem_RawFree(vec->entries);
        vec->entries = NULL;
    }
    vec->count = 0;
    vec->capacity = 0;
}

void
_PyGCSplitVector_Clear(_PyGCSplitVector *vec)
{
    vec->count = 0;
}

int
_PyGCSplitVector_Push(_PyGCSplitVector *vec, PyGC_Head *gc)
{
    // Grow if needed
    if (vec->count >= vec->capacity) {
        size_t new_capacity = vec->capacity * 2;
        PyGC_Head **new_entries = (PyGC_Head **)PyMem_RawRealloc(
            vec->entries, new_capacity * sizeof(PyGC_Head *));
        if (new_entries == NULL) {
            return -1;
        }
        vec->entries = new_entries;
        vec->capacity = new_capacity;
    }
    vec->entries[vec->count++] = gc;
    return 0;
}


// =============================================================================
// Work Queue Operations
// =============================================================================
//
// Block-based work queue for pipelined producer-consumer mark_alive.
// See pycore_gc_parallel.h for design rationale.
//
// Concurrency model:
// - Single producer (main thread) pushes level-1 objects
// - Multiple consumers (workers) claim batches with atomic CAS
// - Producer signals completion with atomic store
// - Consumers spin on queue until producer done and queue empty
//
// Memory model:
// - write_index: release store after writing item
// - read_index: acquire load before reading, CAS to claim batch
// - producer_done: release store by producer, acquire load by consumers

int
_PyGCWorkQueue_Init(_PyGCWorkQueue *queue)
{
    // Point to pre-allocated initial blocks
    queue->blocks = queue->initial_blocks;
    queue->num_blocks = 0;
    queue->capacity = _PyGC_QUEUE_INITIAL_BLOCKS;

    // Reset indices and completion flag
    queue->write_index = 0;
    queue->read_index = 0;
    queue->producer_done = 0;

    return 0;
}

void
_PyGCWorkQueue_Fini(_PyGCWorkQueue *queue)
{
    // If using heap-allocated blocks, free them
    if (queue->blocks != queue->initial_blocks && queue->blocks != NULL) {
        PyMem_RawFree(queue->blocks);
    }
    queue->blocks = NULL;
    queue->num_blocks = 0;
    queue->capacity = 0;
}

void
_PyGCWorkQueue_Reset(_PyGCWorkQueue *queue)
{
    // Reset for new collection, keeping any allocated capacity
    queue->num_blocks = 0;
    queue->write_index = 0;
    queue->read_index = 0;
    queue->producer_done = 0;
}

// Grow the block array when capacity is exceeded
static int
_PyGCWorkQueue_Grow(_PyGCWorkQueue *queue)
{
    Py_ssize_t new_capacity = queue->capacity * 2;

    // Allocate new block array
    _PyGCQueueBlock *new_blocks = (_PyGCQueueBlock *)PyMem_RawCalloc(
        new_capacity, sizeof(_PyGCQueueBlock));
    if (new_blocks == NULL) {
        return -1;
    }

    // Copy existing blocks
    if (queue->num_blocks > 0) {
        memcpy(new_blocks, queue->blocks,
               (size_t)queue->num_blocks * sizeof(_PyGCQueueBlock));
    }

    // Free old heap-allocated blocks (if any)
    if (queue->blocks != queue->initial_blocks) {
        PyMem_RawFree(queue->blocks);
    }

    queue->blocks = new_blocks;
    queue->capacity = new_capacity;

    return 0;
}

int
_PyGCWorkQueue_Push(_PyGCWorkQueue *queue, PyObject *obj)
{
    // Calculate block and offset for current write position
    Py_ssize_t idx = queue->write_index;
    Py_ssize_t block = idx / _PyGC_QUEUE_BLOCK_SIZE;
    Py_ssize_t offset = idx % _PyGC_QUEUE_BLOCK_SIZE;

    // Ensure we have enough blocks
    while (block >= queue->capacity) {
        if (_PyGCWorkQueue_Grow(queue) < 0) {
            return -1;
        }
    }

    // Track actual blocks in use
    if (block >= queue->num_blocks) {
        queue->num_blocks = block + 1;
    }

    // Write the item
    queue->blocks[block].items[offset] = obj;

    // Release store: make item visible to consumers before updating index
    _Py_atomic_store_ssize_release(&queue->write_index, idx + 1);

    return 0;
}

void
_PyGCWorkQueue_ProducerDone(_PyGCWorkQueue *queue)
{
    // Release store: ensure all pushed items are visible before signalling done
    _Py_atomic_store_int_release(&queue->producer_done, 1);
}

Py_ssize_t
_PyGCWorkQueue_ClaimBatch(_PyGCWorkQueue *queue, PyObject **out, Py_ssize_t max_batch)
{
    int spin_count = 0;
    const int MAX_SPINS = 1024;

    while (1) {
        // Acquire load: see items written before write_index update
        Py_ssize_t read = _Py_atomic_load_ssize_acquire(&queue->read_index);
        Py_ssize_t write = _Py_atomic_load_ssize_acquire(&queue->write_index);

        if (read >= write) {
            // Queue appears empty - check if producer is done
            if (_Py_atomic_load_int_acquire(&queue->producer_done)) {
                // Double-check write_index in case producer pushed more items
                // between our read of write_index and producer_done
                write = _Py_atomic_load_ssize_acquire(&queue->write_index);
                if (read >= write) {
                    return 0;  // Queue is truly empty and producer is done
                }
                // Producer pushed more items, continue to claim them
            }
            else {
                // Spin-wait with exponential backoff
                if (spin_count < MAX_SPINS) {
                    // Backoff: 1, 2, 4, 8, ... up to 32 iterations of cpu_relax
                    int backoff_exp = spin_count / 128;
                    if (backoff_exp > 5) backoff_exp = 5;
                    int backoff_iters = 1 << backoff_exp;
                    for (volatile int i = 0; i < backoff_iters; i++) {
                        _Py_cpu_relax();
                    }
                    spin_count++;
                    continue;
                }
                // Give up after MAX_SPINS, will retry at higher level
                return 0;
            }
        }

        // Reset spin count - we have work available
        spin_count = 0;

        // Calculate batch size
        Py_ssize_t available = write - read;
        Py_ssize_t batch = available < max_batch ? available : max_batch;

        // Try to claim the batch with CAS
        if (_Py_atomic_compare_exchange_ssize(&queue->read_index, &read, read + batch)) {
            // Successfully claimed [read, read + batch)
            // Copy items to output buffer
            for (Py_ssize_t i = 0; i < batch; i++) {
                Py_ssize_t item_idx = read + i;
                Py_ssize_t blk = item_idx / _PyGC_QUEUE_BLOCK_SIZE;
                Py_ssize_t off = item_idx % _PyGC_QUEUE_BLOCK_SIZE;
                out[i] = queue->blocks[blk].items[off];
            }
            return batch;
        }
        // CAS failed - another worker claimed, retry
    }
}


// =============================================================================
// Counting Semaphore for Coordinator-Based Termination
// =============================================================================

int
_PyGCSemaphore_Init(_PyGCSemaphore *sema)
{
    sema->tokens = 0;
    if (PyMUTEX_INIT(&sema->lock) != 0) {
        return -1;
    }
    if (PyCOND_INIT(&sema->cond) != 0) {
        PyMUTEX_FINI(&sema->lock);
        return -1;
    }
    return 0;
}

void
_PyGCSemaphore_Fini(_PyGCSemaphore *sema)
{
    PyCOND_FINI(&sema->cond);
    PyMUTEX_FINI(&sema->lock);
}

void
_PyGCSemaphore_Post(_PyGCSemaphore *sema, Py_ssize_t n)
{
    PyMUTEX_LOCK(&sema->lock);
    sema->tokens += n;
    // Wake up to n waiters
    for (Py_ssize_t i = 0; i < n; i++) {
        PyCOND_SIGNAL(&sema->cond);
    }
    PyMUTEX_UNLOCK(&sema->lock);
}

void
_PyGCSemaphore_Wait(_PyGCSemaphore *sema)
{
    PyMUTEX_LOCK(&sema->lock);
    while (sema->tokens <= 0) {
        PyCOND_WAIT(&sema->cond, &sema->lock);
    }
    sema->tokens--;
    PyMUTEX_UNLOCK(&sema->lock);
}


// =============================================================================
// Coordinator-Based Termination for Work-Stealing Mark Phase
// =============================================================================

// Try to become the coordinator. Returns 1 if successful, 0 if another worker
// is already coordinator.
static int
_PyGCWorker_TakeCoordinator(_PyParallelGCWorker *worker)
{
    _PyParallelGCState *par_gc = worker->par_gc;
    int success = 0;

    PyMUTEX_LOCK(&par_gc->steal_coord_lock);
    if (par_gc->steal_coordinator == NULL) {
        par_gc->steal_coordinator = worker;
        success = 1;
    }
    PyMUTEX_UNLOCK(&par_gc->steal_coord_lock);

    return success;
}

// Give up coordinator role.
static void
_PyGCWorker_DropCoordinator(_PyParallelGCWorker *worker)
{
    _PyParallelGCState *par_gc = worker->par_gc;

    PyMUTEX_LOCK(&par_gc->steal_coord_lock);
    if (par_gc->steal_coordinator == worker) {
        par_gc->steal_coordinator = NULL;
    }
    PyMUTEX_UNLOCK(&par_gc->steal_coord_lock);
}

// Coordinator logic: poll all deques, wake workers if work exists, or terminate.
// Called when this worker has become the coordinator.
static void
_PyGCWorker_CoordinateStealing(_PyParallelGCWorker *worker)
{
    _PyParallelGCState *par_gc = worker->par_gc;
    int backoff = 1;

    while (1) {
        // Load current count of active workers
        int num_marking = _Py_atomic_load_int_acquire(&par_gc->num_workers_marking);

        // TERMINATION: coordinator is the only active worker
        // All others have: emptied their deques, failed steals, decremented counter
        if (num_marking == 1) {
            // Decrement our own count (now 0)
            _Py_atomic_add_int(&par_gc->num_workers_marking, -1);
            // Wake all waiters so they can exit the loop
            Py_ssize_t to_wake = (Py_ssize_t)par_gc->num_workers - 1;
            if (to_wake > 0) {
                _PyGCSemaphore_Post(&par_gc->steal_sema, to_wake);
            }
            return;
        }

        // Scan all deques for work
        Py_ssize_t work_available = 0;
        for (size_t i = 0; i < par_gc->num_workers; i++) {
            work_available += _PyWSDeque_Size(&par_gc->workers[i].deque);
        }

        if (work_available > 0) {
            // Calculate how many workers to wake (at most 1 per 64 items of work)
            Py_ssize_t workers_to_wake = work_available / 64;
            if (workers_to_wake < 1) {
                workers_to_wake = 1;
            }
            // Don't wake more than are sleeping
            Py_ssize_t sleeping = (Py_ssize_t)par_gc->num_workers - num_marking;
            if (workers_to_wake > sleeping) {
                workers_to_wake = sleeping;
            }

            if (workers_to_wake > 0) {
                // CRITICAL: increment counter BEFORE posting to semaphore
                // This prevents race where worker wakes but coordinator doesn't see it
                _Py_atomic_add_int(&par_gc->num_workers_marking, (int)workers_to_wake);
                _PyGCSemaphore_Post(&par_gc->steal_sema, workers_to_wake);
            }
            // Give up coordinator role and re-enter work loop
            return;
        }

        // No work found, backoff and retry
        for (int i = 0; i < backoff; i++) {
            _Py_cpu_relax();
        }
        backoff = backoff * 2;
        if (backoff > 64) {
            backoff = 64;
        }
    }
}


// =============================================================================
// Parallel Marking
// =============================================================================

int
_PyGC_ParallelMoveUnreachable(
    PyInterpreterState *interp,
    PyGC_Head *young,
    PyGC_Head *unreachable)
{
    // Get parallel GC state
    _PyParallelGCState *par_gc = interp->gc.parallel_gc;

    // If parallel GC not enabled, fall back to serial
    if (par_gc == NULL || !par_gc->enabled || par_gc->num_workers_active == 0) {
        return 0;  // Use serial marking
    }

    // Increment attempt counter
    par_gc->parallel_collections_attempted++;

    // ==========================================================================
    // STEP 1: Count total objects and assign static slices to workers
    // ==========================================================================
    //
    // Static slicing (like CinderX's Ci_assign_worker_slices) gives each worker
    // a contiguous portion of the GC list. This preserves temporal locality:
    // - CPython's GC list is in allocation order (newest at head)
    // - Objects allocated together tend to reference each other
    // - Keeping them on the same worker reduces cross-worker cache invalidation
    // - Work-stealing still handles load imbalance for uneven slices
    //
    // This replaces the old round-robin root distribution which scattered
    // related objects across workers, maximizing cache thrashing.

    PyGC_Head *gc = _PyGCHead_NEXT(young);
    size_t total_objects = 0;
    size_t total_roots = 0;

    // First pass: count total objects and roots
    // Objects already marked (COLLECTING cleared) by mark_alive_from_roots are treated as roots
    // (they're already known reachable and don't need parallel processing)
    while (gc != young) {
        total_objects++;
        // Objects with COLLECTING cleared are already marked as reachable
        if (!gc_is_collecting(gc)) {
            total_roots++;
        }
        else {
            Py_ssize_t refs = gc_get_refs(gc);
            if (refs > 0) {
                total_roots++;
            }
        }
        gc = _PyGCHead_NEXT(gc);
    }

    // If no roots or too few objects, fall back to serial.
    //
    // Parallel marking has fixed overhead from:
    // - Atomic CAS operations in work-stealing deque
    // - Barrier synchronization between workers
    // - Cache coherency traffic between cores
    //
    // This overhead is only amortized for larger heaps. Require:
    // - Minimum 10,000 total objects (base overhead threshold)
    // - At least 1,000 objects per worker (work distribution threshold)
    //
    // Below these thresholds, serial marking is faster.
    const size_t MIN_TOTAL_OBJECTS = 10000;
    const size_t MIN_OBJECTS_PER_WORKER = 1000;
    size_t min_threshold = par_gc->num_workers * MIN_OBJECTS_PER_WORKER;
    if (min_threshold < MIN_TOTAL_OBJECTS) {
        min_threshold = MIN_TOTAL_OBJECTS;
    }

    if (total_roots == 0 || total_objects < min_threshold) {
        return 0;  // Not worth parallelizing
    }

    // Update statistics
    par_gc->roots_found = total_roots;
    par_gc->roots_distributed = 0;  // Will be updated as we find roots in slices

    // ==========================================================================
    // STEP 2: Assign contiguous slices to workers and find roots
    // ==========================================================================
    //
    // Each worker gets a contiguous slice: objects [i*N/W, (i+1)*N/W)
    // Where N = total_objects, W = num_workers, i = worker index
    //
    // Benefits over round-robin:
    // CHUNKED STRIPING: Balance locality with distribution
    // - Objects in same 1024-chunk go to same worker (cache locality)
    // - Chunks distributed across workers (breaks anti-patterns)
    // - For layered heaps: each layer spans all workers
    (void)total_objects;  // Unused with chunked striping
    size_t seen = 0;
    size_t distributed = 0;

    // Reset worker slice tracking and local buffers
    for (size_t i = 0; i < par_gc->num_workers; i++) {
        _PyParallelGCWorker *worker = &par_gc->workers[i];
        worker->slice_start = NULL;
        worker->slice_end = NULL;
        worker->roots_in_slice = 0;
        // Reset local buffer for new collection (no fences, just zero count)
        _PyGCLocalBuffer_Reset(&worker->local_buffer);
    }

    gc = _PyGCHead_NEXT(young);
    while (gc != young) {
        // CHUNKED STRIPING: Assign objects in chunks of 1024 to workers
        // This balances locality (1024 objects together) with distribution
        // (each layer spans all workers, breaking anti-patterns in layered heaps)
        #define CHUNK_SIZE 1024
        size_t chunk_id = seen / CHUNK_SIZE;
        size_t worker_idx = chunk_id % par_gc->num_workers;

        _PyParallelGCWorker *worker = &par_gc->workers[worker_idx];

        // Track slice boundaries for debugging/stats
        if (worker->slice_start == NULL) {
            worker->slice_start = gc;
        }
        worker->slice_end = _PyGCHead_NEXT(gc);

        // Check if this object is already marked (COLLECTING cleared by mark_alive_from_roots)
        if (!gc_is_collecting(gc)) {
            // Already marked as reachable by mark_alive_from_roots.
            // All referents are also marked, so no need to push to deque.
            // COLLECTING is already cleared.
            distributed++;
        }
        // Check if this object is a root (has external references)
        else {
            Py_ssize_t refs = gc_get_refs(gc);
            if (refs > 0) {
                // This is a root - mark as reachable and push to this worker's deque
                PyObject *op = _Py_FROM_GC(gc);

                // Mark root as reachable by clearing COLLECTING flag
                // (This must be done BEFORE workers start, so no CAS needed here)
                gc->_gc_prev &= ~_PyGC_PREV_MASK_COLLECTING;

                _PyWSDeque_Push(&worker->deque, op);
                worker->roots_in_slice++;
                distributed++;
            }
        }

        gc = _PyGCHead_NEXT(gc);
        seen++;
    }

    // Update statistics
    par_gc->roots_distributed = distributed;

    // ==========================================================================
    // STEP 3: Signal workers to start and wait for completion
    // ==========================================================================
    //
    // Use barriers to coordinate with worker threads:
    // 1. mark_barrier: Release workers to start marking
    // 2. done_barrier: Wait for all workers to finish

    // Initialize coordinator-based termination state for this collection
    // All workers start as "marking", coordinator will detect when all are idle
    _Py_atomic_store_int_release(&par_gc->num_workers_marking, (int)par_gc->num_workers);
    par_gc->steal_coordinator = NULL;
    // Reset semaphore tokens (in case any leaked from previous collection)
    PyMUTEX_LOCK(&par_gc->steal_sema.lock);
    par_gc->steal_sema.tokens = 0;
    PyMUTEX_UNLOCK(&par_gc->steal_sema.lock);

    // Set phase for all workers before signaling
    for (size_t i = 0; i < par_gc->num_workers; i++) {
        par_gc->workers[i].phase = _PyGC_PHASE_MARK;
    }

    // Signal workers to start (they're waiting on mark_barrier)
    _PyGCBarrier_Wait(&par_gc->mark_barrier);

    // Wait for workers to finish (they'll signal done_barrier when done)
    _PyGCBarrier_Wait(&par_gc->done_barrier);

    // Record end time for mark phase (only if timing not already captured)
    if (!par_gc->timing_valid) {
        PyTime_t mark_end;
        (void)PyTime_PerfCounterRaw(&mark_end);
        par_gc->mark_end_ns = mark_end;

        // Mark timing as valid - full parallel collection completed
        par_gc->timing_valid = 1;
    }

    // ==========================================================================
    // STEP 4: Sweep - move unmarked objects to unreachable list
    // ==========================================================================
    //
    // After parallel marking, objects are in two states:
    // - COLLECTING flag cleared: reachable (visited by workers)
    // - COLLECTING flag still set: unreachable (never visited)
    //
    // We sweep through the young list single-threaded:
    // 1. Reachable objects: restore _gc_prev as doubly-linked list pointer
    // 2. Unreachable objects: move to unreachable list

    // Previous element in young list (for restoring _gc_prev pointers)
    PyGC_Head *prev = young;
    gc = _PyGCHead_NEXT(young);

    // Flags for unreachable list (preserve old space bit)
    uintptr_t flags = 2 | (gc->_gc_next & _PyGC_NEXT_MASK_OLD_SPACE_1);  // NEXT_MASK_UNREACHABLE = 2

    while (gc != young) {
        PyGC_Head *next = _PyGCHead_NEXT(gc);

        if (gc_is_collecting_atomic(gc)) {
            // Object still has COLLECTING flag - it's unreachable
            // Move to unreachable list

            // Unlink from young list (update prev's next pointer)
            prev->_gc_next = gc->_gc_next;

            // Add to unreachable list with NEXT_MASK_UNREACHABLE flag
            PyGC_Head *last = (PyGC_Head *)(unreachable->_gc_prev);
            last->_gc_next = flags | (uintptr_t)gc;
            _PyGCHead_SET_PREV(gc, last);
            gc->_gc_next = flags | (uintptr_t)unreachable;
            unreachable->_gc_prev = (uintptr_t)gc;

            // Don't advance prev - we removed current element
        }
        else {
            // Object is reachable - restore _gc_prev as list pointer
            _PyGCHead_SET_PREV(gc, prev);
            prev = gc;
        }

        gc = next;
    }

    // Fix up young list tail
    young->_gc_prev = (uintptr_t)prev;
    // Clean up any pollution of list head next pointer flags
    young->_gc_next &= _PyGC_PREV_MASK;
    unreachable->_gc_next &= _PyGC_PREV_MASK;

    // Update success counter
    par_gc->parallel_collections_succeeded++;

    // Return 1 to indicate parallel marking was used successfully
    return 1;
}

// =============================================================================
// Atomic gc_refs Decrement for Parallel subtract_refs
// =============================================================================
//
// In subtract_refs, we call tp_traverse on each object and decrement gc_refs
// for each referenced object. When parallelised, multiple workers may
// decrement the SAME object's gc_refs simultaneously (if it's referenced from
// objects in different workers' segments).
//
// Solution: Use atomic subtract for gc_refs.
//
// gc_refs is stored in the upper bits of _gc_prev (shifted by _PyGC_PREV_SHIFT).
// To atomically decrement gc_refs by 1, we atomically subtract (1 << SHIFT).
//
// This is safe because:
// - Multiple decrements are commutative (order doesn't matter)
// - Final value is correct regardless of order
// - Atomic add is well-supported and fast on all platforms

// Atomically decrement gc_refs by 1
static inline void
gc_decref_atomic(PyGC_Head *gc)
{
    _Py_atomic_add_uintptr(&gc->_gc_prev, -((uintptr_t)1 << _PyGC_PREV_SHIFT));
}

// Thread-safe visit_decref callback for parallel subtract_refs
// Called by tp_traverse for each object reference
// Decrements gc_refs of referenced object if it's in the collection set
static int
visit_decref_atomic(PyObject *op, void *parent)
{
    (void)parent;  // Unused, but required by visitor signature

    if (PyObject_IS_GC(op)) {
        PyGC_Head *gc = _Py_AS_GC(op);

        // Only decrement if object is in collection set (has COLLECTING flag)
        // Use relaxed load for check (atomic decrement will be correct anyway)
        uintptr_t prev = _Py_atomic_load_uintptr_relaxed(&gc->_gc_prev);
        if (prev & _PyGC_PREV_MASK_COLLECTING) {
            gc_decref_atomic(gc);
        }
    }
    return 0;  // Continue traversal
}

// =============================================================================
// Parallel subtract_refs
// =============================================================================
//
// Parallel version of subtract_refs(). Each worker processes a segment of the
// GC list, calling tp_traverse on each object. The visitor callback
// atomically decrements gc_refs for each referenced object.
//
// Work distribution strategy:
// 1. During update_refs, we record "waypoints" (object pointers) at ~8K intervals
//    into the split_vector. For 1M objects, this gives ~122 waypoints.
// 2. We divide the waypoints among N workers, giving each worker ~(122/N) waypoints.
// 3. Each worker processes from their first waypoint to their last waypoint.
// 4. Result: each worker processes roughly (total_objects / N) objects.
//
// The 8K interval is just a granularity choice for the intermediate waypoint
// vector - it determines how finely we can divide work, not the segment size.

// Worker function: process segment [start, end)
static void
_parallel_subtract_refs_worker(_PyParallelGCWorker *worker,
                               PyGC_Head *start,
                               PyGC_Head *end)
{
    // Record start time for profiling
    PyTime_t work_start;
    (void)PyTime_PerfCounterRaw(&work_start);
    worker->work_start_ns = work_start;

    PyGC_Head *gc = start;
    unsigned long count = 0;
    unsigned long segment_size = 0;

    while (gc != end) {
        // Prefetch next node
        PyGC_Head *next = _PyGCHead_NEXT(gc);
        _PyGC_PREFETCH_T0(next);  // Read, high temporal locality

        segment_size++;

        // Skip objects already marked reachable (COLLECTING cleared by mark_alive_from_roots).
        // These are known reachable and don't need subtract_refs processing.
        // All their referents are also marked (mark_alive uses transitive closure).
        if (!gc_is_collecting(gc)) {
            gc = next;
            continue;
        }

        PyObject *op = _Py_FROM_GC(gc);

        // Call tp_traverse with atomic visitor
        traverseproc traverse = Py_TYPE(op)->tp_traverse;
        if (traverse != NULL) {
            traverse(op, (visitproc)visit_decref_atomic, op);
            count++;
        }

        gc = next;
    }

    worker->traversals_performed += count;
    worker->objects_in_segment = segment_size;

    // Record end time for profiling
    PyTime_t work_end;
    (void)PyTime_PerfCounterRaw(&work_end);
    worker->work_end_ns = work_end;
}

// Entry point: use split vector to distribute work to workers
int
_PyGC_ParallelSubtractRefs(PyInterpreterState *interp, PyGC_Head *base)
{
    _PyParallelGCState *par_gc = interp->gc.parallel_gc;

    if (par_gc == NULL || !par_gc->enabled || par_gc->num_workers_active == 0) {
        return 0;  // Fall back to serial
    }

    _PyGCSplitVector *splits = &par_gc->split_vector;

    // Need at least 2 split points (start and end)
    if (splits->count < 2) {
        return 0;  // Not enough split points, fall back to serial
    }

    // Record start time for phase timing
    PyTime_t start_time;
    (void)PyTime_PerfCounterRaw(&start_time);
    par_gc->phase_start_ns = start_time;

    // Divide split vector entries among workers
    // Each worker processes a range of split vector entries
    size_t entries_per_worker = splits->count / par_gc->num_workers;
    if (entries_per_worker < 1) {
        entries_per_worker = 1;
    }

    size_t workers_to_use = 0;
    for (size_t i = 0; i < par_gc->num_workers; i++) {
        size_t start_idx = i * entries_per_worker;
        size_t end_idx;

        if (i == par_gc->num_workers - 1) {
            // Last worker gets everything remaining
            end_idx = splits->count - 1;
        } else {
            end_idx = (i + 1) * entries_per_worker;
            if (end_idx >= splits->count) {
                end_idx = splits->count - 1;
            }
        }

        if (start_idx >= splits->count - 1) {
            // No work for this worker
            par_gc->workers[i].phase = _PyGC_PHASE_IDLE;
        } else {
            par_gc->workers[i].slice_start = splits->entries[start_idx];
            par_gc->workers[i].slice_end = splits->entries[end_idx];
            par_gc->workers[i].phase = _PyGC_PHASE_SUBTRACT_REFS;
            workers_to_use++;
        }
    }

    if (workers_to_use == 0) {
        return 0;  // No workers assigned, fall back to serial
    }

    // Signal workers to start
    _PyGCBarrier_Wait(&par_gc->mark_barrier);

    // Wait for completion
    _PyGCBarrier_Wait(&par_gc->done_barrier);

    // Record end time for subtract_refs phase (only if timing not already captured)
    if (!par_gc->timing_valid) {
        PyTime_t end_time;
        (void)PyTime_PerfCounterRaw(&end_time);
        par_gc->subtract_refs_end_ns = end_time;
    }

    return 1;
}

// =============================================================================
// Parallel Mark Alive From Roots
// =============================================================================
// Pre-marks reachable objects from interpreter roots using parallel traversal.
// Uses the same work-stealing infrastructure as PHASE_MARK.

// Helper: Collect interpreter roots and distribute to worker deques
// Returns the number of roots found
static size_t
gc_collect_interpreter_roots(_PyParallelGCState *par_gc, PyInterpreterState *interp)
{
    size_t roots_found = 0;
    size_t worker_idx = 0;

    // Macro to enqueue a root object to worker deques (round-robin)
    #define ENQUEUE_ROOT(op) \
        do { \
            if ((op) != NULL && _PyObject_IS_GC((PyObject *)(op)) && \
                _PyObject_GC_IS_TRACKED((PyObject *)(op))) { \
                PyGC_Head *gc = _Py_AS_GC((PyObject *)(op)); \
                if (gc_try_mark_reachable_atomic(gc)) { \
                    _PyWSDeque_Push(&par_gc->workers[worker_idx].deque, (PyObject *)(op)); \
                    worker_idx = (worker_idx + 1) % par_gc->num_workers; \
                    roots_found++; \
                } \
            } \
        } while (0)

    // Interpreter roots
    ENQUEUE_ROOT(interp->sysdict);
    ENQUEUE_ROOT(interp->builtins);
    ENQUEUE_ROOT(interp->dict);

    // Type dicts and subclasses for builtin types
    struct types_state *types = &interp->types;
    for (int i = 0; i < _Py_MAX_MANAGED_STATIC_BUILTIN_TYPES; i++) {
        ENQUEUE_ROOT(types->builtins.initialized[i].tp_dict);
        ENQUEUE_ROOT(types->builtins.initialized[i].tp_subclasses);
    }
    for (int i = 0; i < _Py_MAX_MANAGED_STATIC_EXT_TYPES; i++) {
        ENQUEUE_ROOT(types->for_extensions.initialized[i].tp_dict);
        ENQUEUE_ROOT(types->for_extensions.initialized[i].tp_subclasses);
    }

    // Thread stacks - traverse all threads' frames
    _Py_FOR_EACH_TSTATE_BEGIN(interp, tstate) {
        // Current frame and all frames on call stack
        _PyInterpreterFrame *frame = tstate->current_frame;
        while (frame != NULL) {
            // Skip interpreter-owned frames
            if (frame->owner >= FRAME_OWNED_BY_INTERPRETER) {
                frame = frame->previous;
                continue;
            }

            // Skip frames with NULL stackpointer
            if (frame->stackpointer == NULL) {
                frame = frame->previous;
                continue;
            }

            // Visit the executable (code object)
            if (!PyStackRef_IsNullOrInt(frame->f_executable)) {
                PyObject *exec = PyStackRef_AsPyObjectBorrow(frame->f_executable);
                ENQUEUE_ROOT(exec);
            }

            // Visit f_globals
            ENQUEUE_ROOT(frame->f_globals);

            // Visit f_builtins
            ENQUEUE_ROOT(frame->f_builtins);

            // Visit f_locals if set
            ENQUEUE_ROOT(frame->f_locals);

            // Visit all stackrefs from stackpointer down to localsplus
            _PyStackRef *top = frame->stackpointer;
            while (top != frame->localsplus) {
                --top;
                if (!PyStackRef_IsNullOrInt(*top)) {
                    PyObject *stackval = PyStackRef_AsPyObjectBorrow(*top);
                    ENQUEUE_ROOT(stackval);
                }
            }

            frame = frame->previous;
        }
    }
    _Py_FOR_EACH_TSTATE_END(interp);

    #undef ENQUEUE_ROOT

    return roots_found;
}

// =============================================================================
// Level-1 Expansion for Pipelined Producer-Consumer mark_alive
// =============================================================================
//
// The pipelined producer-consumer design addresses the work distribution problem
// in parallel mark_alive:
//
// Problem: Work-stealing mark_alive fails because ~100 interpreter roots form a hub.
//          The first worker marks most of the heap before others can steal.
//
// Solution: Main thread expands roots by one level (tp_traverse), pushing level-1
//           children to a shared work queue. This gives thousands of starting points
//           instead of ~100 concentrated roots.
//
// Pipelining: Workers start consuming from the queue while the producer is still
//             pushing items. No barrier needed between producer and consumers.

// Context passed to level-1 visitor callback
typedef struct {
    _PyGCWorkQueue *queue;
    size_t items_pushed;
    size_t roots_expanded;  // Count of roots successfully expanded
} _PyGCLevel1VisitorContext;

// Visitor callback for level-1 expansion
// Called by tp_traverse for each child of an interpreter root
static int
push_level1_child(PyObject *obj, void *arg)
{
    _PyGCLevel1VisitorContext *ctx = (_PyGCLevel1VisitorContext *)arg;

    // Skip NULL and non-GC objects
    if (obj == NULL || !_PyObject_IS_GC(obj)) {
        return 0;
    }

    // Skip untracked objects
    if (!_PyObject_GC_IS_TRACKED(obj)) {
        return 0;
    }

    PyGC_Head *gc = _Py_AS_GC(obj);

    // Only process objects in the collection set (have COLLECTING flag)
    // Try to atomically mark as reachable - if successful, push to queue
    if (gc_try_mark_reachable_atomic(gc)) {
        // We claimed this object - push to work queue
        if (_PyGCWorkQueue_Push(ctx->queue, obj) == 0) {
            ctx->items_pushed++;
        }
        // If push fails (OOM), we still marked the object as reachable
        // so it won't be collected - safe but suboptimal
    }

    return 0;  // Continue traversal
}

// Expand interpreter roots by one level into the work queue
// Returns the number of level-1 objects pushed to the queue
// Also stores the number of roots expanded in *roots_expanded_out if not NULL
static size_t
gc_expand_roots_to_queue(_PyParallelGCState *par_gc, PyInterpreterState *interp,
                         size_t *roots_expanded_out)
{
    _PyGCWorkQueue *queue = &par_gc->work_queue;
    _PyGCLevel1VisitorContext ctx = { .queue = queue, .items_pushed = 0, .roots_expanded = 0 };

    // Reset queue for this collection
    _PyGCWorkQueue_Reset(queue);

    // Macro to expand a root object (traverse one level)
    #define EXPAND_ROOT(op) \
        do { \
            if ((op) != NULL && _PyObject_IS_GC((PyObject *)(op)) && \
                _PyObject_GC_IS_TRACKED((PyObject *)(op))) { \
                PyGC_Head *gc = _Py_AS_GC((PyObject *)(op)); \
                /* Mark the root itself as reachable */ \
                if (gc_try_mark_reachable_atomic(gc)) { \
                    ctx.roots_expanded++; \
                    /* Traverse to get level-1 children */ \
                    traverseproc traverse = Py_TYPE((PyObject *)(op))->tp_traverse; \
                    if (traverse != NULL) { \
                        traverse((PyObject *)(op), push_level1_child, &ctx); \
                    } \
                } \
            } \
        } while (0)

    // Expand interpreter roots
    EXPAND_ROOT(interp->sysdict);
    EXPAND_ROOT(interp->builtins);
    EXPAND_ROOT(interp->dict);

    // Expand type dicts and subclasses for builtin types
    struct types_state *types = &interp->types;
    for (int i = 0; i < _Py_MAX_MANAGED_STATIC_BUILTIN_TYPES; i++) {
        EXPAND_ROOT(types->builtins.initialized[i].tp_dict);
        EXPAND_ROOT(types->builtins.initialized[i].tp_subclasses);
    }
    for (int i = 0; i < _Py_MAX_MANAGED_STATIC_EXT_TYPES; i++) {
        EXPAND_ROOT(types->for_extensions.initialized[i].tp_dict);
        EXPAND_ROOT(types->for_extensions.initialized[i].tp_subclasses);
    }

    // Expand thread stacks - traverse all threads' frames
    _Py_FOR_EACH_TSTATE_BEGIN(interp, tstate) {
        _PyInterpreterFrame *frame = tstate->current_frame;
        while (frame != NULL) {
            // Skip interpreter-owned frames
            if (frame->owner >= FRAME_OWNED_BY_INTERPRETER) {
                frame = frame->previous;
                continue;
            }

            // Skip frames with NULL stackpointer
            if (frame->stackpointer == NULL) {
                frame = frame->previous;
                continue;
            }

            // Expand the executable (code object)
            if (!PyStackRef_IsNullOrInt(frame->f_executable)) {
                PyObject *exec = PyStackRef_AsPyObjectBorrow(frame->f_executable);
                EXPAND_ROOT(exec);
            }

            // Expand f_globals, f_builtins, f_locals
            EXPAND_ROOT(frame->f_globals);
            EXPAND_ROOT(frame->f_builtins);
            EXPAND_ROOT(frame->f_locals);

            // Expand all stackrefs from stackpointer down to localsplus
            _PyStackRef *top = frame->stackpointer;
            while (top != frame->localsplus) {
                --top;
                if (!PyStackRef_IsNullOrInt(*top)) {
                    PyObject *stackval = PyStackRef_AsPyObjectBorrow(*top);
                    EXPAND_ROOT(stackval);
                }
            }

            frame = frame->previous;
        }
    }
    _Py_FOR_EACH_TSTATE_END(interp);

    #undef EXPAND_ROOT

    // Signal that producer is done
    _PyGCWorkQueue_ProducerDone(queue);

    // Return roots expanded count if requested
    if (roots_expanded_out != NULL) {
        *roots_expanded_out = ctx.roots_expanded;
    }

    return ctx.items_pushed;
}

int
_PyGC_ParallelMarkAliveFromRoots(PyInterpreterState *interp, PyGC_Head *containers)
{
    _PyParallelGCState *par_gc = interp->gc.parallel_gc;

    if (par_gc == NULL || !par_gc->enabled || par_gc->num_workers_active == 0) {
        return 0;  // Fall back to serial
    }

    // Collect interpreter roots and distribute to worker deques
    size_t roots_found = gc_collect_interpreter_roots(par_gc, interp);

    // Only update stats if we found roots (don't overwrite previous values)
    if (roots_found > 0) {
        par_gc->roots_found = roots_found;
        par_gc->roots_distributed = roots_found;
    }

    if (roots_found == 0) {
        // No roots to process - still "successful" parallel execution
        // Record timing for empty mark_alive phase
        if (!par_gc->timing_valid) {
            PyTime_t end_time;
            (void)PyTime_PerfCounterRaw(&end_time);
            par_gc->mark_alive_end_ns = end_time;
        }
        return 1;
    }

    // Initialize coordinator-based termination state for this collection
    _Py_atomic_store_int_release(&par_gc->num_workers_marking, (int)par_gc->num_workers);
    par_gc->steal_coordinator = NULL;
    PyMUTEX_LOCK(&par_gc->steal_sema.lock);
    par_gc->steal_sema.tokens = 0;
    PyMUTEX_UNLOCK(&par_gc->steal_sema.lock);

    // Set phase for all workers
    for (size_t i = 0; i < par_gc->num_workers; i++) {
        par_gc->workers[i].phase = _PyGC_PHASE_MARK_ALIVE;
    }

    // Signal workers to start
    _PyGCBarrier_Wait(&par_gc->mark_barrier);

    // Wait for completion
    _PyGCBarrier_Wait(&par_gc->done_barrier);

    // Record end time for mark_alive phase
    if (!par_gc->timing_valid) {
        PyTime_t end_time;
        (void)PyTime_PerfCounterRaw(&end_time);
        par_gc->mark_alive_end_ns = end_time;
    }

    return 1;
}

// =============================================================================
// Pipelined Producer-Consumer mark_alive
// =============================================================================
//
// Alternative to work-stealing mark_alive that uses a shared queue fed by
// level-1 expansion from interpreter roots. This provides better work
// distribution for hub-structured object graphs.

int
_PyGC_ParallelMarkAliveFromQueue(PyInterpreterState *interp, PyGC_Head *containers)
{
    _PyParallelGCState *par_gc = interp->gc.parallel_gc;

    if (par_gc == NULL || !par_gc->enabled || par_gc->num_workers_active == 0) {
        return 0;  // Fall back to serial
    }

    // Expand interpreter roots by one level into the work queue
    // This also marks the roots themselves as reachable
    size_t roots_expanded = 0;
    size_t level1_count = gc_expand_roots_to_queue(par_gc, interp, &roots_expanded);

    // Update stats - roots_found is the number of roots we expanded,
    // roots_distributed is the number of level-1 children pushed to queue
    if (roots_expanded > 0) {
        par_gc->roots_found = roots_expanded;
        par_gc->roots_distributed = level1_count;
    }

    if (roots_expanded == 0) {
        // No roots found to expand - still "successful" parallel execution
        // Record timing for empty mark_alive phase
        if (!par_gc->timing_valid) {
            PyTime_t end_time;
            (void)PyTime_PerfCounterRaw(&end_time);
            par_gc->mark_alive_end_ns = end_time;
        }
        return 1;
    }

    // Set phase for all workers to use queue-based processing
    for (size_t i = 0; i < par_gc->num_workers; i++) {
        par_gc->workers[i].phase = _PyGC_PHASE_MARK_ALIVE_QUEUE;
    }

    // Signal workers to start
    // Producer has already expanded all roots and filled the work queue
    _PyGCBarrier_Wait(&par_gc->mark_barrier);

    // Wait for completion
    _PyGCBarrier_Wait(&par_gc->done_barrier);

    // Record end time for mark_alive phase
    if (!par_gc->timing_valid) {
        PyTime_t end_time;
        (void)PyTime_PerfCounterRaw(&end_time);
        par_gc->mark_alive_end_ns = end_time;
    }

    return 1;
}

#endif // Py_PARALLEL_GC
