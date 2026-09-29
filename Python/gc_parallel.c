// Parallel cyclic garbage collection for GIL-enabled builds.

#include "Python.h"

#if defined(Py_PARALLEL_GC) && !defined(Py_GIL_DISABLED)

#include "pycore_gc.h"             // GC internals
#include "pycore_gc_parallel.h"
#include "pycore_interp.h"
#include "pycore_object.h"         // _PyObject_IsFreed()

// Mirrors gc_get_refs() from Python/gc.c. Parallel workers require an atomic
// load so that race detectors see the matching accesses.
static inline Py_ssize_t
gc_get_refs(PyGC_Head *g)
{
    uintptr_t prev = _Py_atomic_load_uintptr_relaxed(&g->_gc_prev);
    return (Py_ssize_t)(prev >> _PyGC_PREV_SHIFT);
}

static inline int
gc_is_collecting(PyGC_Head *g)
{
    uintptr_t prev = _Py_atomic_load_uintptr_relaxed(&g->_gc_prev);
    return (prev & _PyGC_PREV_MASK_COLLECTING) != 0;
}

// A collecting object becomes reachable when a worker atomically clears its
// COLLECTING flag. The previous value determines which worker claimed it.
static inline int
gc_try_mark_reachable_atomic(PyGC_Head *gc)
{
    assert(gc != NULL);

    uintptr_t prev = _Py_atomic_load_uintptr_relaxed(&gc->_gc_prev);
    if (!(prev & _PyGC_PREV_MASK_COLLECTING)) {
        return 0;
    }

    uintptr_t old_prev = _Py_atomic_and_uintptr(
        &gc->_gc_prev,
        ~_PyGC_PREV_MASK_COLLECTING
    );

    int marked = (old_prev & _PyGC_PREV_MASK_COLLECTING) != 0;

    // Pair with publication of the object's fields before tp_traverse reads
    // them. This is required on weakly ordered architectures.
    if (marked) {
        _Py_atomic_fence_acquire();
    }

    return marked;
}

static int
_parallel_gc_visit_and_enqueue(PyObject *op, void *arg)
{
    _PyParallelGCWorker *worker = arg;

    _PyObject_ASSERT(op, !_PyObject_IsFreed(op));
    if (!PyObject_IS_GC(op)) {
        return 0;
    }

    PyGC_Head *gc = _Py_AS_GC(op);

    if (!gc_try_mark_reachable_atomic(gc)) {
        return 0;
    }

    if (_PyGCLocalBuffer_IsFull(&worker->local_buffer)) {
        if (_PyGC_OverflowFlush(&worker->local_buffer,
                                &worker->deque) < 0)
        {
            return -1;
        }
    }
    _PyGCLocalBuffer_Push(&worker->local_buffer, op);

    return 0;
}

static inline int
drain_local_buffer(_PyParallelGCWorker *worker)
{
    while (!_PyGCLocalBuffer_IsEmpty(&worker->local_buffer)) {
        PyObject *obj = _PyGCLocalBuffer_Pop(&worker->local_buffer);
        _PyObject_ASSERT(obj, !_PyObject_IsFreed(obj));

        traverseproc traverse = Py_TYPE(obj)->tp_traverse;
        if (traverse != NULL) {
            if (traverse(obj, _parallel_gc_visit_and_enqueue, worker) != 0)
            {
                return -1;
            }
        }
    }
    return 0;
}

static int
parallel_mark_worker(_PyParallelGCWorker *worker)
{
    int result = 0;

    if (worker->slice_start != NULL && worker->slice_end != NULL) {
        PyGC_Head *gc = worker->slice_start;
        PyGC_Head *end = worker->slice_end;
        while (gc != end) {
            PyGC_Head *next = _PyGCHead_NEXT(gc);

            _PyObject_ASSERT(_Py_FROM_GC(gc),
                             !_PyObject_IsFreed(_Py_FROM_GC(gc)));

            if (!gc_is_collecting(gc)) {
                gc = next;
                continue;
            }

            Py_ssize_t refs = gc_get_refs(gc);
            if (refs != 0) {
                PyObject *op = _Py_FROM_GC(gc);
                _PyObject_ASSERT_WITH_MSG(op, refs > 0,
                                          "refcount is too small");
                if (!gc_try_mark_reachable_atomic(gc)) {
                    gc = next;
                    continue;
                }
                if (_PyGCLocalBuffer_IsFull(&worker->local_buffer) &&
                    _PyGC_OverflowFlush(&worker->local_buffer,
                                        &worker->deque) < 0)
                {
                    result = -1;
                    break;
                }
                _PyGCLocalBuffer_Push(&worker->local_buffer, op);
            }

            gc = next;
        }
        worker->slice_start = NULL;
        worker->slice_end = NULL;
    }

    while (result == 0) {
        if (drain_local_buffer(worker) < 0) {
            result = -1;
            break;
        }
        _PyGC_RefillLocalFromDeque(&worker->local_buffer, &worker->deque);
        if (_PyGCLocalBuffer_IsEmpty(&worker->local_buffer)) {
            break;
        }
    }
    return result;
}

static void
_parallel_subtract_refs_worker(PyGC_Head *start,
                               PyGC_Head *end);

static void
_parallel_gc_worker_thread(void *arg)
{
    _PyParallelGCWorker *worker = arg;
    _PyParallelGCState *par_gc = worker->par_gc;

    _PyGCBarrier_Wait(&par_gc->startup_barrier);

    while (1) {
        _PyGC_MUTEX_LOCK(&worker->wake_mutex);
        while (!worker->wake_flag) {
            _PyGC_COND_WAIT(&worker->wake_cond, &worker->wake_mutex);
        }
        worker->wake_flag = 0;
        _PyGC_MUTEX_UNLOCK(&worker->wake_mutex);

        if (_Py_atomic_load_int(&worker->should_exit)) {
            break;
        }

        switch (worker->phase) {
        case _PyGC_PHASE_SUBTRACT_REFS:
            _parallel_subtract_refs_worker(worker->slice_start,
                                           worker->slice_end);
            break;

        case _PyGC_PHASE_MARK:
            worker->error = parallel_mark_worker(worker) < 0;
            break;

        case _PyGC_PHASE_IDLE:
            break;

        default:
            Py_UNREACHABLE();
        }

        _PyGC_MUTEX_LOCK(&par_gc->done_mutex);
        par_gc->workers_done_count++;
        _PyGC_COND_SIGNAL(&par_gc->done_cond);
        _PyGC_MUTEX_UNLOCK(&par_gc->done_mutex);
    }
}

static void
dispatch_and_wait(_PyParallelGCState *par_gc, size_t active_workers)
{
    assert(active_workers > 0);
    assert(active_workers <= par_gc->num_workers);

    _PyGC_MUTEX_LOCK(&par_gc->done_mutex);
    par_gc->workers_done_count = 0;
    _PyGC_MUTEX_UNLOCK(&par_gc->done_mutex);

    for (size_t i = 0; i < active_workers; i++) {
        _PyGC_MUTEX_LOCK(&par_gc->workers[i].wake_mutex);
        par_gc->workers[i].wake_flag = 1;
        _PyGC_COND_SIGNAL(&par_gc->workers[i].wake_cond);
        _PyGC_MUTEX_UNLOCK(&par_gc->workers[i].wake_mutex);
    }

    _PyGC_MUTEX_LOCK(&par_gc->done_mutex);
    while (par_gc->workers_done_count < (int)active_workers) {
        _PyGC_COND_WAIT(&par_gc->done_cond, &par_gc->done_mutex);
    }
    _PyGC_MUTEX_UNLOCK(&par_gc->done_mutex);
}

int
_PyGC_ParallelInit(PyInterpreterState *interp, size_t num_workers)
{
    if (num_workers < _PyGC_PARALLEL_MIN_WORKERS ||
        num_workers > _PyGC_PARALLEL_MAX_WORKERS)
    {
        PyErr_Format(PyExc_ValueError,
                     "num_workers must be between 2 and %zu, got %zu",
                     (size_t)_PyGC_PARALLEL_MAX_WORKERS, num_workers);
        return -1;
    }

    size_t state_size = sizeof(_PyParallelGCState) +
                       num_workers * sizeof(_PyParallelGCWorker);
    _PyParallelGCState *par_gc = PyMem_Calloc(1, state_size);
    if (par_gc == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    par_gc->num_workers = num_workers;
    par_gc->enabled = 1;
    par_gc->num_workers_active = 0;

    if (_PyGCSplitVector_Init(&par_gc->split_vector) < 0) {
        PyMem_Free(par_gc);
        PyErr_NoMemory();
        return -1;
    }

    for (size_t i = 0; i < num_workers; i++) {
        _PyGC_MUTEX_INIT(&par_gc->workers[i].wake_mutex);
        _PyGC_COND_INIT(&par_gc->workers[i].wake_cond);
        par_gc->workers[i].wake_flag = 0;
    }
    _PyGC_MUTEX_INIT(&par_gc->done_mutex);
    _PyGC_COND_INIT(&par_gc->done_cond);
    par_gc->workers_done_count = 0;

    _PyGCBarrier_Init(&par_gc->startup_barrier, (unsigned int)num_workers + 1);

    for (size_t i = 0; i < num_workers; i++) {
        _PyParallelGCWorker *worker = &par_gc->workers[i];
        if (_PyWSDeque_Init(&worker->deque) < 0) {
            for (size_t j = 0; j < i; j++) {
                _PyWSDeque_Fini(&par_gc->workers[j].deque);
            }
            for (size_t j = 0; j < num_workers; j++) {
                _PyGC_COND_FINI(&par_gc->workers[j].wake_cond);
                _PyGC_MUTEX_FINI(&par_gc->workers[j].wake_mutex);
            }
            _PyGCBarrier_Fini(&par_gc->startup_barrier);
            _PyGC_COND_FINI(&par_gc->done_cond);
            _PyGC_MUTEX_FINI(&par_gc->done_mutex);
            _PyGCSplitVector_Fini(&par_gc->split_vector);
            PyMem_Free(par_gc);
            PyErr_NoMemory();
            return -1;
        }

        worker->slice_start = NULL;
        worker->slice_end = NULL;
        worker->error = 0;
        _PyGCLocalBuffer_Reset(&worker->local_buffer);
        worker->par_gc = par_gc;
        worker->phase = _PyGC_PHASE_IDLE;
        _Py_atomic_store_int(&worker->should_exit, 0);
    }

    interp->gc.parallel_gc = par_gc;

    assert(par_gc->enabled == 1);
    assert(par_gc->num_workers == num_workers);

    return 0;
}

void
_PyGC_ParallelFini(PyInterpreterState *interp)
{
    _PyParallelGCState *par_gc = interp->gc.parallel_gc;

    if (par_gc == NULL) {
        return;
    }

    _PyGC_ParallelStop(interp);

    for (size_t i = 0; i < par_gc->num_workers; i++) {
        _PyParallelGCWorker *worker = &par_gc->workers[i];

        _PyWSDeque_Fini(&worker->deque);
        _PyGC_COND_FINI(&worker->wake_cond);
        _PyGC_MUTEX_FINI(&worker->wake_mutex);
    }

    _PyGCBarrier_Fini(&par_gc->startup_barrier);
    _PyGC_COND_FINI(&par_gc->done_cond);
    _PyGC_MUTEX_FINI(&par_gc->done_mutex);

    _PyGCSplitVector_Fini(&par_gc->split_vector);
    PyMem_Free(par_gc);
    interp->gc.parallel_gc = NULL;
}

int
_PyGC_ParallelStart(PyInterpreterState *interp)
{
    assert(interp != NULL);
    _PyParallelGCState *par_gc = interp->gc.parallel_gc;

    if (par_gc == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                       "Parallel GC not initialized");
        return -1;
    }
    for (size_t i = 0; i < par_gc->num_workers; i++) {
        _PyParallelGCWorker *worker = &par_gc->workers[i];

        PyThread_ident_t ident;
        int rc = PyThread_start_joinable_thread(
            _parallel_gc_worker_thread, worker, &ident, &worker->thread);
        if (rc != 0) {
            PyErr_Format(PyExc_RuntimeError,
                        "Failed to create worker thread %zu: error %d",
                        i, rc);

            // Remove the workers that were never created from the startup
            // barrier, then join the workers that did start.
            unsigned int missing = (unsigned int)(par_gc->num_workers - i);
            _PyGC_MUTEX_LOCK(&par_gc->startup_barrier.lock);
            par_gc->startup_barrier.capacity -= missing;
            par_gc->startup_barrier.num_left -= missing;
            _PyGC_MUTEX_UNLOCK(&par_gc->startup_barrier.lock);
            _PyGCBarrier_Wait(&par_gc->startup_barrier);
            _PyGC_ParallelStop(interp);
            return -1;
        }

        par_gc->num_workers_active++;
    }

    // Ensure ParallelStop cannot race with worker initialization.
    _PyGCBarrier_Wait(&par_gc->startup_barrier);

    assert((size_t)par_gc->num_workers_active == par_gc->num_workers);

    return 0;
}

void
_PyGC_ParallelStop(PyInterpreterState *interp)
{
    _PyParallelGCState *par_gc = interp->gc.parallel_gc;

    if (par_gc == NULL || par_gc->num_workers_active == 0) {
        return;
    }

    size_t active_workers = (size_t)par_gc->num_workers_active;
    for (size_t i = 0; i < active_workers; i++) {
        _Py_atomic_store_int(&par_gc->workers[i].should_exit, 1);
    }

    for (size_t i = 0; i < active_workers; i++) {
        _PyGC_MUTEX_LOCK(&par_gc->workers[i].wake_mutex);
        par_gc->workers[i].wake_flag = 1;
        _PyGC_COND_SIGNAL(&par_gc->workers[i].wake_cond);
        _PyGC_MUTEX_UNLOCK(&par_gc->workers[i].wake_mutex);
    }

    for (size_t i = 0; i < active_workers; i++) {
        _PyParallelGCWorker *worker = &par_gc->workers[i];

        if (PyThread_join_thread(worker->thread) != 0) {
            Py_FatalError("failed to join parallel GC worker");
        }

        _Py_atomic_store_int(&worker->should_exit, 0);
    }

    par_gc->num_workers_active = 0;
}

void
_PyGC_ParallelBeforeFork(PyInterpreterState *interp)
{
    if (_PyGC_ParallelIsEnabled(interp)) {
        _PyGC_ParallelStop(interp);
    }
}

void
_PyGC_ParallelAfterFork(PyInterpreterState *interp)
{
    _PyParallelGCState *par_gc = interp->gc.parallel_gc;
    if (par_gc == NULL || !par_gc->enabled ||
        par_gc->num_workers_active != 0)
    {
        return;
    }
    if (_PyGC_ParallelStart(interp) < 0) {
        // Fork itself succeeded.  If helper recreation fails, leave parallel
        // collection disabled and do not leak the helper failure into the
        // caller's otherwise successful fork operation.
        par_gc->enabled = 0;
        PyErr_Clear();
    }
}

void
_PyGC_ParallelAfterForkChild(PyInterpreterState *interp)
{
    _PyParallelGCState *par_gc = interp->gc.parallel_gc;
    if (par_gc != NULL) {
        // Creating threads between fork() and exec() is unsafe.  Leave the
        // child serial; gc.enable_parallel() can create a fresh pool later.
        par_gc->enabled = 0;
    }
}

int
_PyGC_ParallelIsEnabled(PyInterpreterState *interp)
{
    _PyParallelGCState *par_gc = interp->gc.parallel_gc;

    return (par_gc != NULL && par_gc->enabled);
}

PyObject *
_PyGC_ParallelGetConfig(PyInterpreterState *interp)
{
    _PyParallelGCState *par_gc = interp->gc.parallel_gc;

    PyObject *result = PyDict_New();
    if (result == NULL) {
        return NULL;
    }

    if (PyDict_SetItemString(result, "available", Py_True) < 0) {
        Py_DECREF(result);
        return NULL;
    }

    int enabled = (par_gc != NULL && par_gc->enabled);
    int num_workers = enabled ? (int)par_gc->num_workers : 0;

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

// Split Vector Operations
//
// The split vector records pointers into the GC list at regular intervals
// during the serial update_refs phase. This allows parallel subtract_refs
// to quickly find evenly-spaced start/end positions for each worker.

int
_PyGCSplitVector_Init(_PyGCSplitVector *vec)
{
    assert(vec != NULL);
    vec->entries = PyMem_RawCalloc(
        _PyGC_SPLIT_VECTOR_INITIAL_CAPACITY, sizeof(PyGC_Head *));
    if (vec->entries == NULL) {
        return -1;
    }
    vec->count = 0;
    vec->capacity = _PyGC_SPLIT_VECTOR_INITIAL_CAPACITY;
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
    assert(vec != NULL);
    assert(gc != NULL);
    if (vec->count >= vec->capacity) {
        size_t new_capacity = vec->capacity * 2;
        PyGC_Head **new_entries = PyMem_RawRealloc(
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

static size_t
assign_slices(_PyParallelGCState *par_gc, _PyGCPhase phase)
{
    _PyGCSplitVector *splits = &par_gc->split_vector;
    assert(splits->count >= 2);

    size_t slices = splits->count - 1;
    size_t active = Py_MIN(par_gc->num_workers, slices);
    if (active < 2) {
        return 0;
    }

    size_t entries_per_worker = splits->count / active;
    assert(entries_per_worker > 0);

    for (size_t i = 0; i < active; i++) {
        size_t start_idx = i * entries_per_worker;
        size_t end_idx = i == active - 1
            ? splits->count - 1
            : (i + 1) * entries_per_worker;
        assert(start_idx < splits->count - 1);
        assert(end_idx < splits->count);

        par_gc->workers[i].slice_start = splits->entries[start_idx];
        par_gc->workers[i].slice_end = splits->entries[end_idx];
        par_gc->workers[i].phase = phase;
    }
    return active;
}


int
_PyGC_ParallelMoveUnreachable(
    PyInterpreterState *interp,
    PyGC_Head *young,
    PyGC_Head *unreachable)
{
    _PyParallelGCState *par_gc = interp->gc.parallel_gc;

    if (par_gc == NULL || !par_gc->enabled ||
        par_gc->num_workers_active == 0)
    {
        return 0;
    }

    for (size_t i = 0; i < par_gc->num_workers; i++) {
        _PyParallelGCWorker *worker = &par_gc->workers[i];
        worker->error = 0;
        _PyGCLocalBuffer_Reset(&worker->local_buffer);
    }

    size_t active = assign_slices(par_gc, _PyGC_PHASE_MARK);
    if (active == 0) {
        return 0;
    }

    // gcstate->collecting prevents another collection from entering while
    // these workers operate on the generation lists.
    dispatch_and_wait(par_gc, active);

    int failed = 0;
    for (size_t i = 0; i < active; i++) {
        if (par_gc->workers[i].error) {
            failed = 1;
        }
    }
    if (failed) {
        for (PyGC_Head *gc = _PyGCHead_NEXT(young);
             gc != young;
             gc = _PyGCHead_NEXT(gc))
        {
            _Py_atomic_or_uintptr(&gc->_gc_prev,
                                  _PyGC_PREV_MASK_COLLECTING);
        }
        for (size_t i = 0; i < active; i++) {
            _PyWSDeque_Reset(&par_gc->workers[i].deque);
            _PyGCLocalBuffer_Reset(&par_gc->workers[i].local_buffer);
        }
        return 0;
    }

    // Reachable objects have COLLECTING clear. Restore their list links while
    // moving the remaining objects to unreachable.
    PyGC_Head *prev = young;
    PyGC_Head *gc = _PyGCHead_NEXT(young);

    // Python/gc.c uses the low bit of _gc_next for NEXT_MASK_UNREACHABLE.
    const uintptr_t unreachable_flag = 1;

    while (gc != young) {
        PyGC_Head *next = _PyGCHead_NEXT(gc);

        if (gc_is_collecting(gc)) {
            prev->_gc_next = gc->_gc_next;

            PyGC_Head *last = (PyGC_Head *)(unreachable->_gc_prev);
            last->_gc_next = unreachable_flag | (uintptr_t)gc;
            _PyGCHead_SET_PREV(gc, last);
            gc->_gc_next = unreachable_flag | (uintptr_t)unreachable;
            unreachable->_gc_prev = (uintptr_t)gc;

        }
        else {
            _PyGCHead_SET_PREV(gc, prev);
            prev = gc;
        }

        gc = next;
    }

    young->_gc_prev = (uintptr_t)prev;
    // Remove the temporary flag from the unreachable list head.
    unreachable->_gc_next &= ~unreachable_flag;

    return 1;
}

// Parallel version of subtract_refs(). Each worker processes a segment of the
// GC list. The shared visitor uses atomic decrements in parallel-GC builds.
static void
_parallel_subtract_refs_worker(PyGC_Head *start,
                               PyGC_Head *end)
{
    PyGC_Head *gc = start;

    while (gc != end) {
        PyGC_Head *next = _PyGCHead_NEXT(gc);
        _PyGC_PREFETCH_T0(next);

        if (!gc_is_collecting(gc)) {
            gc = next;
            continue;
        }

        PyObject *op = _Py_FROM_GC(gc);

        traverseproc traverse = Py_TYPE(op)->tp_traverse;
        if (traverse != NULL) {
            traverse(op, _PyGC_VisitDecref, op);
        }

        gc = next;
    }
}

int
_PyGC_ParallelSubtractRefs(PyInterpreterState *interp)
{
    _PyParallelGCState *par_gc = interp->gc.parallel_gc;

    if (par_gc == NULL || !par_gc->enabled ||
        par_gc->num_workers_active == 0)
    {
        return 0;
    }

    if (par_gc->split_vector.count < 2) {
        return 0;
    }

    size_t active = assign_slices(par_gc, _PyGC_PHASE_SUBTRACT_REFS);
    if (active == 0) {
        return 0;
    }

    // gcstate->collecting prevents another collection from entering while
    // these workers operate on the generation lists.
    dispatch_and_wait(par_gc, active);

    return 1;
}

#endif  // Py_PARALLEL_GC && !Py_GIL_DISABLED
