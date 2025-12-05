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
// We use compare-and-swap (CAS) to atomically mark objects, ensuring each
// object is only processed once even with concurrent workers.

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
static inline int
gc_try_mark_reachable_atomic(PyGC_Head *gc)
{
    // Read current value
    uintptr_t prev = _Py_atomic_load_uintptr_relaxed(&gc->_gc_prev);

    // Check if COLLECTING flag is set
    if (!(prev & _PyGC_PREV_MASK_COLLECTING)) {
        // Already marked reachable by another worker
        return 0;
    }

    // Try to clear COLLECTING flag atomically
    uintptr_t new_prev = prev & ~_PyGC_PREV_MASK_COLLECTING;

    // CAS: Only succeed if value hasn't changed
    int success = _Py_atomic_compare_exchange_uintptr(
        &gc->_gc_prev,
        &prev,
        new_prev
    );

    return success;
}

// =============================================================================
// Barrier Synchronization
// =============================================================================

void
_PyGCBarrier_Init(_PyGCBarrier *barrier, int capacity)
{
    barrier->capacity = capacity;
    barrier->num_left = capacity;
    barrier->epoch = 0;
    PyMUTEX_INIT(&barrier->lock);
    PyCOND_INIT(&barrier->cond);
}

void
_PyGCBarrier_Fini(_PyGCBarrier *barrier)
{
    PyMUTEX_FINI(&barrier->lock);
    PyCOND_FINI(&barrier->cond);
}

void
_PyGCBarrier_Wait(_PyGCBarrier *barrier)
{
    PyMUTEX_LOCK(&barrier->lock);
    barrier->num_left--;
    if (barrier->num_left == 0) {
        // We were the last one to get to the barrier; reset it and unblock
        // everyone else.
        barrier->num_left = barrier->capacity;
        barrier->epoch++;
        PyCOND_BROADCAST(&barrier->cond);
    }
    else {
        unsigned int epoch = barrier->epoch;
        while (epoch == barrier->epoch) {
            PyCOND_WAIT(&barrier->cond, &barrier->lock);
        }
    }
    PyMUTEX_UNLOCK(&barrier->lock);
}

// =============================================================================
// Worker Thread Function
// =============================================================================

// Visit function called by tp_traverse() to discover child objects
// This is called for each object reference during traversal
//
// IMPORTANT: We use atomic CAS to mark objects as reachable. The algorithm:
// 1. Check if object is in collection set (has COLLECTING flag)
// 2. Atomically try to clear COLLECTING flag (marks as reachable)
// 3. If CAS succeeds, we claimed this object - enqueue for traversal
// 4. If CAS fails, another worker already claimed it - skip
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
    // Add to work queue for traversal of its children
    worker->objects_discovered++;
    _PyWSDeque_Push(&worker->deque, op);

    return 0;  // Continue traversal
}

// Worker thread entry point
static void *
_parallel_gc_worker_thread(void *arg)
{
    _PyParallelGCWorker *worker = (_PyParallelGCWorker *)arg;
    _PyParallelGCState *par_gc = worker->par_gc;

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
        // STEP 3: Main marking loop
        // =======================================================================
        // Process objects from local deque until empty
        // NOTE: For Step 3, we just count objects without actual traversal
        // Actual object traversal requires GIL handling (Step 3b)
        while (1) {
            // Try to pop work from local deque (LIFO - good for cache locality)
            PyObject *obj = _PyWSDeque_Take(&worker->deque);

            if (obj == NULL) {
                // Local deque empty
                // =============================================================
                // STEP 4: Try work-stealing from other workers
                // =============================================================
                int steal_attempts_made = 0;
                const int MAX_STEAL_ATTEMPTS = (int)par_gc->num_workers * 2;

                while (steal_attempts_made < MAX_STEAL_ATTEMPTS) {
                    // Pick a random victim worker to steal from
                    // Use simple linear congruential generator for random selection
                    worker->steal_seed = worker->steal_seed * 1103515245 + 12345;
                    size_t victim_id = (worker->steal_seed / 65536) % par_gc->num_workers;

                    // Don't steal from ourselves
                    if (victim_id == worker->thread_id) {
                        victim_id = (victim_id + 1) % par_gc->num_workers;
                    }

                    _PyParallelGCWorker *victim = &par_gc->workers[victim_id];

                    // Try to steal from victim's deque (FIFO - from top)
                    worker->steal_attempts++;
                    steal_attempts_made++;

                    obj = (PyObject *)_PyWSDeque_Steal(&victim->deque);
                    if (obj != NULL) {
                        // Steal successful!
                        worker->steal_successes++;
                        break;  // Got work, process it
                    }

                    // Steal failed, try another victim
                }

                // If still no work after all steal attempts, we're done
                if (obj == NULL) {
                    break;
                }
            }

            // Mark this object (increment counter for statistics)
            worker->objects_marked++;

            // ================================================================
            // STEP 3b: Object traversal via tp_traverse()
            // ================================================================
            // Call tp_traverse to discover children and add them to work queue
            // This is SAFE because:
            // 1. Main thread holds GIL (in gc.collect())
            // 2. All Python threads are blocked (waiting for GIL)
            // 3. Object graph is frozen (no mutations)
            // 4. tp_traverse() only reads object fields
            traverseproc traverse = Py_TYPE(obj)->tp_traverse;
            if (traverse != NULL) {
                // Call tp_traverse with our visit callback
                // The callback will add discovered children to our work queue
                traverse(obj, (visitproc)_parallel_gc_visit_and_enqueue, worker);
                worker->traversals_performed++;
            }
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
    par_gc->min_gen = 0;  // Use parallel GC for all generations
    par_gc->enabled = 1;
    par_gc->num_workers_active = 0;

    // Initialize barriers
    // mark_barrier: all workers + main thread (to signal start)
    // done_barrier: all workers + main thread (to signal completion)
    _PyGCBarrier_Init(&par_gc->mark_barrier, (int)num_workers + 1);
    _PyGCBarrier_Init(&par_gc->done_barrier, (int)num_workers + 1);

    // Initialize locks
    PyMUTEX_INIT(&par_gc->active_lock);
    PyCOND_INIT(&par_gc->workers_done_cond);

    // Initialize workers
    for (size_t i = 0; i < num_workers; i++) {
        _PyParallelGCWorker *worker = &par_gc->workers[i];
        _PyWSDeque_Init(&worker->deque);
        worker->objects_marked = 0;
        worker->steal_attempts = 0;
        worker->steal_successes = 0;
        worker->objects_discovered = 0;      // NEW: Step 3b
        worker->traversals_performed = 0;    // NEW: Step 3b
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
        _PyWSDeque_Fini(&worker->deque);
    }

    // Clean up barriers
    _PyGCBarrier_Fini(&par_gc->mark_barrier);
    _PyGCBarrier_Fini(&par_gc->done_barrier);

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
    // STEP 1: Scan young generation list for roots
    // ==========================================================================
    //
    // Roots are objects with external references (gc_refs > 0).
    // These are the starting points for the marking phase.

    PyGC_Head *gc = _PyGCHead_NEXT(young);
    size_t total_roots = 0;

    while (gc != young) {
        // Roots are objects with external references (gc_refs > 0)
        Py_ssize_t refs = gc_get_refs(gc);
        if (refs > 0) {
            total_roots++;
        }

        gc = _PyGCHead_NEXT(gc);
    }

    // Update statistics (always update roots_found, even if we fall back)
    // Note: We keep the stats from the last SUCCESSFUL scan (roots >= threshold)
    // This way multiple gc.collect() calls don't overwrite good stats with zeros

    // If no roots or too few objects, fall back to serial
    // Parallel marking has overhead, so only use it if worthwhile
    if (total_roots == 0 || total_roots < par_gc->num_workers * 4) {
        // Don't update stats - keep previous values
        // (This prevents later empty collections from overwriting earlier good stats)
        return 0;  // Not worth parallelizing
    }

    // We have enough roots - update statistics
    par_gc->roots_found = total_roots;
    par_gc->roots_distributed = 0;  // Will be set in Step 2

    // ==========================================================================
    // STEP 2: Distribute roots to worker deques
    // ==========================================================================
    //
    // Distribute roots to worker deques in round-robin fashion for load balancing.
    // Each worker will start marking from their assigned roots.

    gc = _PyGCHead_NEXT(young);
    size_t worker_idx = 0;
    size_t distributed = 0;

    while (gc != young) {
        Py_ssize_t refs = gc_get_refs(gc);

        if (refs > 0) {
            // This is a root - mark as reachable and push to worker's deque
            PyObject *op = _Py_FROM_GC(gc);
            _PyParallelGCWorker *worker = &par_gc->workers[worker_idx];

            // Mark root as reachable by clearing COLLECTING flag
            // (This must be done BEFORE workers start, so no CAS needed here)
            gc->_gc_prev &= ~_PyGC_PREV_MASK_COLLECTING;

            _PyWSDeque_Push(&worker->deque, op);

            distributed++;

            // Round-robin to next worker
            worker_idx = (worker_idx + 1) % par_gc->num_workers;
        }

        gc = _PyGCHead_NEXT(gc);
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
