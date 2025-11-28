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

// Worker thread entry point
static void *
_parallel_gc_worker_thread(void *arg)
{
    _PyParallelGCWorker *worker = (_PyParallelGCWorker *)arg;
    _PyParallelGCState *par_gc = worker->par_gc;

    // TODO: Actual GC work will be added in Phase 3
    // For now, workers just wait on barriers to test thread pool

    while (!worker->should_exit) {
        // Wait for work (will be signaled by main thread)
        // For Phase 2, just yield to avoid busy-waiting
#ifdef _POSIX_THREADS
        usleep(10000);  // 10ms
#elif defined(NT_THREADS)
        Sleep(10);
#endif
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
    // mark_barrier: all workers wait here before marking
    // done_barrier: all workers + main thread wait here when done
    _PyGCBarrier_Init(&par_gc->mark_barrier, (int)num_workers);
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

    // Wait for workers to finish
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

    // TODO: Implement actual parallel marking algorithm
    // For Phase 4 initial implementation, we'll just return 0 to use serial
    // This proves the hook works and can be called from gc.c

    // Future phases will implement:
    // 1. Scan young list for roots (gc_refs > 0)
    // 2. Distribute roots to worker deques
    // 3. Signal workers to start marking
    // 4. Workers traverse object graphs with work-stealing
    // 5. Wait for workers to complete
    // 6. Return 1 to indicate parallel marking was used

    return 0;  // Fall back to serial for now
}

#endif // Py_PARALLEL_GC
