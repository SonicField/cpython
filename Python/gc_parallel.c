// Parallel Garbage Collection for CPython
// Ported from CinderX ParallelGC by Alex Turner

#include "Python.h"

#ifdef Py_PARALLEL_GC

#include "pycore_gc_parallel.h"
#include "pycore_pystate.h"
#include "pycore_interp.h"
#include "pycore_gc.h"  // For GC internals
#include "condvar.h"  // PyMUTEX_INIT, PyCOND_INIT, etc.

#include <stdio.h>

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

    // RE-ENABLED: Local buffer with chunked striping
    // Push to local buffer first (zero fences, just array indexing)
    if (_PyGCLocalBuffer_IsFull(&worker->local_buffer)) {
        // Buffer full - flush half to deque for work-stealing
        size_t flush_count = _PyGC_LOCAL_BUFFER_SIZE / 2;
        for (size_t i = 0; i < flush_count; i++) {
            PyObject *item = _PyGCLocalBuffer_Pop(&worker->local_buffer);
            _PyWSDeque_Push(&worker->deque, item);
        }
    }
    _PyGCLocalBuffer_Push(&worker->local_buffer, op);

    return 0;  // Continue traversal
}

// =============================================================================
// Batch Operations (ported from FTP gc_free_threading_parallel.c)
// =============================================================================

// Batch refill local buffer from own deque
// Amortizes deque take overhead by pulling up to half buffer at once
static void
refill_local_buffer_from_deque(_PyParallelGCWorker *worker)
{
    _PyGCLocalBuffer *local = &worker->local_buffer;
    // Pull up to half the buffer size to leave room for discovered children
    const size_t max_pull = _PyGC_LOCAL_BUFFER_SIZE / 2;
    size_t pulled = 0;

    while (pulled < max_pull && !_PyGCLocalBuffer_IsFull(local)) {
        PyObject *obj = _PyWSDeque_Take(&worker->deque);
        if (obj == NULL) {
            break;  // Deque is empty
        }
        _PyGCLocalBuffer_Push(local, obj);
        pulled++;
    }
}

// Batch steal from another worker's deque
// Amortizes stealing overhead by stealing up to half buffer at once
// Returns number of objects stolen
static size_t
steal_batch_from_worker(_PyParallelGCWorker *thief, _PyParallelGCWorker *victim)
{
    // OPTIMIZATION: Lazy size check before expensive steal attempts
    // Relaxed loads are much cheaper than the seq_cst fence in Steal()
    // During termination, most deques are empty - this avoids contention
    if (_PyWSDeque_Size(&victim->deque) == 0) {
        return 0;  // Skip empty deque
    }

    _PyGCLocalBuffer *local = &thief->local_buffer;
    // Steal up to half buffer size
    const size_t max_steal = _PyGC_LOCAL_BUFFER_SIZE / 2;
    size_t stolen = 0;

    while (stolen < max_steal && !_PyGCLocalBuffer_IsFull(local)) {
        PyObject *obj = (PyObject *)_PyWSDeque_Steal(&victim->deque);
        if (obj == NULL) {
            break;  // Victim's deque is empty
        }
        _PyGCLocalBuffer_Push(local, obj);
        stolen++;
    }
    return stolen;
}

// Worker thread entry point
static void *
_parallel_gc_worker_thread(void *arg)
{
    _PyParallelGCWorker *worker = (_PyParallelGCWorker *)arg;
    _PyParallelGCState *par_gc = worker->par_gc;

    // Signal that we're ready - this synchronizes with ParallelStart()
    // to ensure all workers are initialized before Start() returns
    _PyGCBarrier_Wait(&par_gc->startup_barrier);

    while (!worker->should_exit) {
        // =======================================================================
        // STEP 6: Wait for start signal from main thread
        // =======================================================================
        // Workers wait on mark_barrier until main thread signals work is ready
        _PyGCBarrier_Wait(&par_gc->mark_barrier);

        // Check if we should exit (signaled during shutdown)
        if (worker->should_exit) {
            break;
        }

        // =======================================================================
        // STEP 3: Main marking loop (FTP-style three-phase approach)
        // =======================================================================
        // Phase 1: Process local buffer (fast path - no fences)
        // Phase 2: Refill from own deque (batch pull)
        // Phase 3: Batch steal from other workers
        int steal_attempts_since_work = 0;
        const int MAX_STEAL_ATTEMPTS = (int)par_gc->num_workers * 2;

        while (1) {
            // ================================================================
            // PHASE 1: Process local buffer (FAST PATH - zero fences!)
            // ================================================================
            while (!_PyGCLocalBuffer_IsEmpty(&worker->local_buffer)) {
                steal_attempts_since_work = 0;  // Got work, reset counter
                PyObject *obj = _PyGCLocalBuffer_Pop(&worker->local_buffer);

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
            if (steal_attempts_since_work >= MAX_STEAL_ATTEMPTS) {
                break;  // No work left, exit
            }

            // Pick a random victim
            worker->steal_seed = worker->steal_seed * 1103515245 + 12345;
            size_t victim_id = (worker->steal_seed / 65536) % par_gc->num_workers;
            if (victim_id == worker->thread_id) {
                victim_id = (victim_id + 1) % par_gc->num_workers;
            }

            _PyParallelGCWorker *victim = &par_gc->workers[victim_id];
            worker->steal_attempts++;
            steal_attempts_since_work++;

            size_t batch_stolen = steal_batch_from_worker(worker, victim);
            if (batch_stolen > 0) {
                worker->steal_successes += batch_stolen;
                steal_attempts_since_work = 0;  // Got work!
                // Continue to Phase 1 with stolen work
            }
            // If steal failed, loop will try another victim
        }

        // =======================================================================
        // STEP 6: Signal completion to main thread
        // =======================================================================
        // Workers wait on done_barrier until all workers finish
        _PyGCBarrier_Wait(&par_gc->done_barrier);
    }

    return NULL;
}

// =============================================================================
// Thread Pool Lifecycle
// =============================================================================

int
_PyGC_ParallelInit(PyInterpreterState *interp, size_t num_workers)
{
    if (num_workers == 0 || num_workers > 1024) {
        PyErr_SetString(PyExc_ValueError,
                       "num_workers must be between 1 and 1024");
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

    par_gc->num_workers = num_workers;
    par_gc->enabled = 1;
    par_gc->num_workers_active = 0;

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

    // Initialize workers with thread-local memory pools
    // Each worker gets a 2MB pre-allocated buffer for their deque
    // This eliminates malloc/calloc calls during the hot path of collections
    size_t pool_entries = _Py_WSDEQUE_LARGE_ARRAY_SIZE;  // 262144 entries
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
        worker->objects_discovered = 0;      // NEW: Step 3b
        worker->traversals_performed = 0;    // NEW: Step 3b
        worker->roots_in_slice = 0;          // NEW: Static slicing
        worker->slice_start = NULL;          // NEW: Static slicing
        worker->slice_end = NULL;            // NEW: Static slicing
        _PyGCLocalBuffer_Reset(&worker->local_buffer);  // Initialize local buffer
        worker->steal_seed = (unsigned int)(i + 1);
        worker->par_gc = par_gc;
        worker->thread_id = i;
        worker->should_exit = 0;
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
            // TODO: Clean up already-created threads
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
        par_gc->workers[i].should_exit = 1;
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
        worker->should_exit = 0;
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

    // NEW: Step 3b - Calculate total objects traversed (sum of all workers' discoveries)
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

        // NEW: Step 3b traversal statistics
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

        // NEW: Static slicing statistics
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

        PyList_SET_ITEM(workers_list, i, worker_dict);
    }

    if (PyDict_SetItemString(result, "workers", workers_list) < 0) {
        Py_DECREF(workers_list);
        Py_DECREF(result);
        return NULL;
    }
    Py_DECREF(workers_list);

    return result;
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
    while (gc != young) {
        total_objects++;
        Py_ssize_t refs = gc_get_refs(gc);
        if (refs > 0) {
            total_roots++;
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

        // Check if this object is a root (has external references)
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

        gc = _PyGCHead_NEXT(gc);
        seen++;
    }

    // Update statistics
    par_gc->roots_distributed = distributed;

    // ==========================================================================
    // STEP 6: Signal workers to start and wait for completion
    // ==========================================================================
    //
    // Use barriers to coordinate with worker threads:
    // 1. mark_barrier: Release workers to start marking
    // 2. done_barrier: Wait for all workers to finish

    // Signal workers to start (they're waiting on mark_barrier)
    _PyGCBarrier_Wait(&par_gc->mark_barrier);

    // Wait for workers to finish (they'll signal done_barrier when done)
    _PyGCBarrier_Wait(&par_gc->done_barrier);

    // ==========================================================================
    // STEP 7: Sweep - move unmarked objects to unreachable list
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

#endif // Py_PARALLEL_GC
