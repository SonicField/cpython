// Tests for parallel garbage collector infrastructure.

#include "parts.h"
#include "pycore_pythread.h"       // PyThread_start_joinable_thread()
#include "pycore_ws_deque.h"       // _PyWSDeque, _PyGCLocalBuffer
#if defined(_POSIX_THREADS) || defined(NT_THREADS)
#  include "pycore_gc_barrier.h"   // _PyGCBarrier
#  define _Py_TEST_GC_BARRIER
#endif
#if defined(Py_PARALLEL_GC) && !defined(Py_GIL_DISABLED)
#  include "pycore_gc_parallel.h"
#  include "pycore_stackref.h"
#endif

static int
deque_init(_PyWSDeque *deque)
{
    if (_PyWSDeque_Init(deque) < 0) {
        PyErr_NoMemory();
        return -1;
    }
    return 0;
}

#ifdef _Py_TEST_GC_BARRIER
static void
join_thread_or_fatal(PyThread_handle_t thread)
{
    if (PyThread_join_thread(thread) != 0) {
        Py_FatalError("failed to join test worker thread");
    }
}
#endif

// Barrier tests.

#ifdef _Py_TEST_GC_BARRIER

// The unsafe_ prefix prevents test_capi from discovering this helper.  It is
// called only in a subprocess because the assertion terminates the process.
static PyObject *
unsafe_barrier_capacity_zero(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCBarrier barrier;
    _PyGCBarrier_Init(&barrier, 0);
    _PyGCBarrier_Fini(&barrier);
    PyErr_SetString(PyExc_AssertionError,
                    "Init(capacity=0) should have aborted");
    return NULL;
}

typedef struct {
    _PyGCBarrier *barrier;
    int arrived;
} barrier_worker_args;

static void
barrier_worker(void *arg)
{
    barrier_worker_args *args = (barrier_worker_args *)arg;
    args->arrived = 1;
    _PyGCBarrier_Wait(args->barrier);
}

static PyObject *
test_barrier_basic(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    enum { num_threads = 4 };
    _PyGCBarrier barrier;
    _PyGCBarrier_Init(&barrier, num_threads);

    PyThread_handle_t threads[num_threads - 1];
    PyThread_ident_t thread_ids[num_threads - 1];
    barrier_worker_args args[num_threads - 1];
    int threads_created = 0;

    for (int i = 0; i < num_threads - 1; i++) {
        args[i].barrier = &barrier;
        args[i].arrived = 0;
        int rc = PyThread_start_joinable_thread(
            barrier_worker, &args[i], &thread_ids[i], &threads[i]);
        if (rc != 0) {
            unsigned int missing = (num_threads - 1) - threads_created;
            _PyGC_MUTEX_LOCK(&barrier.lock);
            barrier.capacity -= missing;
            barrier.num_left -= missing;
            _PyGC_MUTEX_UNLOCK(&barrier.lock);
            _PyGCBarrier_Wait(&barrier);
            for (int j = 0; j < threads_created; j++) {
                join_thread_or_fatal(threads[j]);
            }
            _PyGCBarrier_Fini(&barrier);
            PyErr_Format(PyExc_RuntimeError,
                         "failed to create barrier test worker: error %d", rc);
            return NULL;
        }
        threads_created++;
    }

    _PyGCBarrier_Wait(&barrier);

    for (int i = 0; i < threads_created; i++) {
        join_thread_or_fatal(threads[i]);
        if (!args[i].arrived) {
            _PyGCBarrier_Fini(&barrier);
            PyErr_SetString(PyExc_AssertionError,
                            "Worker did not arrive at barrier");
            return NULL;
        }
    }

    _PyGCBarrier_Fini(&barrier);
    Py_RETURN_NONE;
}

static PyObject *
test_barrier_multiple_rounds(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCBarrier barrier;
    _PyGCBarrier_Init(&barrier, 1);

    unsigned int epoch_before = barrier.epoch;
    _PyGCBarrier_Wait(&barrier);
    if (barrier.epoch != epoch_before + 1) {
        _PyGCBarrier_Fini(&barrier);
        PyErr_SetString(PyExc_AssertionError,
                        "Epoch should increment by 1 per cycle");
        return NULL;
    }

    _PyGCBarrier_Wait(&barrier);
    if (barrier.epoch != epoch_before + 2) {
        _PyGCBarrier_Fini(&barrier);
        PyErr_SetString(PyExc_AssertionError,
                        "Epoch should increment by 1 per cycle (round 2)");
        return NULL;
    }

    _PyGCBarrier_Fini(&barrier);
    Py_RETURN_NONE;
}

typedef struct {
    _PyGCBarrier *barrier;
    unsigned int epoch_after_round1;
    unsigned int epoch_after_round2;
} epoch_worker_args;

static void
epoch_worker(void *arg)
{
    epoch_worker_args *args = (epoch_worker_args *)arg;
    _PyGCBarrier_Wait(args->barrier);
    args->epoch_after_round1 = args->barrier->epoch;
    _PyGCBarrier_Wait(args->barrier);
    args->epoch_after_round2 = args->barrier->epoch;
}

static PyObject *
test_barrier_epoch_distinguishes(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCBarrier barrier;
    _PyGCBarrier_Init(&barrier, 2);

    epoch_worker_args args = { .barrier = &barrier };
    PyThread_handle_t thread;
    PyThread_ident_t thread_id;
    int rc = PyThread_start_joinable_thread(
        epoch_worker, &args, &thread_id, &thread);
    if (rc != 0) {
        _PyGCBarrier_Fini(&barrier);
        PyErr_Format(PyExc_RuntimeError,
                     "failed to create barrier test worker: error %d", rc);
        return NULL;
    }

    unsigned int epoch_before = barrier.epoch;
    _PyGCBarrier_Wait(&barrier);
    _PyGCBarrier_Wait(&barrier);

    join_thread_or_fatal(thread);

    if (args.epoch_after_round1 == epoch_before) {
        _PyGCBarrier_Fini(&barrier);
        PyErr_SetString(PyExc_AssertionError,
                        "Epoch should differ after round 1");
        return NULL;
    }
    if (args.epoch_after_round2 == args.epoch_after_round1) {
        _PyGCBarrier_Fini(&barrier);
        PyErr_SetString(PyExc_AssertionError,
                        "Epoch should differ between rounds");
        return NULL;
    }

    _PyGCBarrier_Fini(&barrier);
    Py_RETURN_NONE;
}

static PyObject *
test_barrier_postcondition(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCBarrier barrier;
    _PyGCBarrier_Init(&barrier, 1);

    unsigned int epoch_before = barrier.epoch;
    _PyGCBarrier_Wait(&barrier);

    if (barrier.epoch == epoch_before) {
        _PyGCBarrier_Fini(&barrier);
        PyErr_SetString(PyExc_AssertionError,
                        "Epoch did not advance after Wait");
        return NULL;
    }
    if (barrier.num_left != barrier.capacity) {
        _PyGCBarrier_Fini(&barrier);
        PyErr_SetString(PyExc_AssertionError,
                        "num_left not reset after barrier lift");
        return NULL;
    }

    _PyGCBarrier_Fini(&barrier);
    Py_RETURN_NONE;
}

#endif  // _Py_TEST_GC_BARRIER

// Local-buffer tests.

static PyObject *
test_localbuffer_push_pop(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCLocalBuffer buf;
    _PyGCLocalBuffer_Init(&buf);

    if (!_PyGCLocalBuffer_IsEmpty(&buf)) {
        PyErr_SetString(PyExc_AssertionError, "New buffer should be empty");
        return NULL;
    }

    PyObject *obj = PyLong_FromLong(42);
    if (obj == NULL) {
        return NULL;
    }

    _PyGCLocalBuffer_Push(&buf, obj);
    if (_PyGCLocalBuffer_IsEmpty(&buf)) {
        Py_DECREF(obj);
        PyErr_SetString(PyExc_AssertionError,
                        "Buffer should not be empty after push");
        return NULL;
    }

    PyObject *result = _PyGCLocalBuffer_Pop(&buf);
    if (result != obj) {
        Py_DECREF(obj);
        PyErr_SetString(PyExc_AssertionError,
                        "Pop should return pushed object");
        return NULL;
    }
    if (!_PyGCLocalBuffer_IsEmpty(&buf)) {
        Py_DECREF(obj);
        PyErr_SetString(PyExc_AssertionError,
                        "Buffer should be empty after pop");
        return NULL;
    }

    Py_DECREF(obj);
    Py_RETURN_NONE;
}

static PyObject *
test_localbuffer_push_full(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCLocalBuffer buf;
    _PyGCLocalBuffer_Init(&buf);

    PyObject *obj = PyLong_FromLong(99);
    if (obj == NULL) {
        return NULL;
    }

    for (int i = 0; i < _PyGC_LOCAL_BUFFER_SIZE; i++) {
        Py_INCREF(obj);
        _PyGCLocalBuffer_Push(&buf, obj);
    }

    if (!_PyGCLocalBuffer_IsFull(&buf)) {
        Py_DECREF(obj);
        PyErr_SetString(PyExc_AssertionError,
                        "Buffer should be full after 1024 pushes");
        return NULL;
    }

    for (int i = 0; i < _PyGC_LOCAL_BUFFER_SIZE; i++) {
        PyObject *r = _PyGCLocalBuffer_Pop(&buf);
        Py_DECREF(r);
    }

    Py_DECREF(obj);
    Py_RETURN_NONE;
}

static PyObject *
test_overflow_flush_precondition(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCLocalBuffer buf;
    _PyGCLocalBuffer_Init(&buf);
    _PyWSDeque deque;
    if (deque_init(&deque) < 0) {
        return NULL;
    }

    PyObject *obj = PyLong_FromLong(7);
    if (obj == NULL) {
        _PyWSDeque_Fini(&deque);
        return NULL;
    }

    for (int i = 0; i < _PyGC_LOCAL_BUFFER_SIZE; i++) {
        Py_INCREF(obj);
        _PyGCLocalBuffer_Push(&buf, obj);
    }

    if (_PyGC_OverflowFlush(&buf, &deque) < 0) {
        _PyWSDeque_Fini(&deque);
        for (int i = 0; i < _PyGC_LOCAL_BUFFER_SIZE + 1; i++) {
            Py_DECREF(obj);
        }
        PyErr_NoMemory();
        return NULL;
    }

    if (buf.count != _PyGC_LOCAL_BUFFER_SIZE / 2) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_Format(PyExc_AssertionError,
                    "Expected %d items after flush, got %zu",
                    _PyGC_LOCAL_BUFFER_SIZE / 2, buf.count);
        return NULL;
    }

    size_t deque_size = _PyWSDeque_Size(&deque);
    if (deque_size != _PyGC_LOCAL_BUFFER_SIZE / 2) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_Format(PyExc_AssertionError,
                    "Expected %d items in deque after flush, got %zu",
                    _PyGC_LOCAL_BUFFER_SIZE / 2, deque_size);
        return NULL;
    }

    while (!_PyGCLocalBuffer_IsEmpty(&buf)) {
        Py_DECREF(_PyGCLocalBuffer_Pop(&buf));
    }
    PyObject *r;
    while ((r = _PyWSDeque_Take(&deque)) != NULL) {
        Py_DECREF(r);
    }

    Py_DECREF(obj);
    _PyWSDeque_Fini(&deque);
    Py_RETURN_NONE;
}

static PyObject *
test_overflow_flush_normal(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCLocalBuffer buf;
    _PyGCLocalBuffer_Init(&buf);
    _PyWSDeque deque;
    if (deque_init(&deque) < 0) {
        return NULL;
    }

    PyObject *obj = PyLong_FromLong(42);
    if (obj == NULL) {
        _PyWSDeque_Fini(&deque);
        return NULL;
    }

    int total_pushed = 0;
    for (int round = 0; round < 3; round++) {
        while (!_PyGCLocalBuffer_IsFull(&buf)) {
            Py_INCREF(obj);
            _PyGCLocalBuffer_Push(&buf, obj);
            total_pushed++;
        }
        if (_PyGC_OverflowFlush(&buf, &deque) < 0) {
            _PyWSDeque_Fini(&deque);
            for (int i = 0; i < total_pushed + 1; i++) {
                Py_DECREF(obj);
            }
            PyErr_NoMemory();
            return NULL;
        }
    }

    int counted = (int)buf.count;
    PyObject *r;
    while ((r = _PyWSDeque_Take(&deque)) != NULL) {
        Py_DECREF(r);
        counted++;
    }
    while (!_PyGCLocalBuffer_IsEmpty(&buf)) {
        Py_DECREF(_PyGCLocalBuffer_Pop(&buf));
    }

    if (counted != total_pushed) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_Format(PyExc_AssertionError,
                    "Conservation violated: pushed %d, found %d",
                    total_pushed, counted);
        return NULL;
    }

    Py_DECREF(obj);
    _PyWSDeque_Fini(&deque);
    Py_RETURN_NONE;
}

// GIL-collector infrastructure tests.

#if defined(Py_PARALLEL_GC) && !defined(Py_GIL_DISABLED)

static PyObject *
test_splitvector_init_push(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCSplitVector vec;
    if (_PyGCSplitVector_Init(&vec) < 0) {
        return PyErr_NoMemory();
    }

    if (vec.entries == NULL || vec.count != 0) {
        _PyGCSplitVector_Fini(&vec);
        PyErr_SetString(PyExc_AssertionError,
                        "Init should set entries!=NULL, count=0");
        return NULL;
    }

    size_t initial_cap = vec.capacity;
    for (size_t i = 0; i < initial_cap + 10; i++) {
        if (_PyGCSplitVector_Push(&vec, (PyGC_Head *)(uintptr_t)(i + 1)) < 0) {
            _PyGCSplitVector_Fini(&vec);
            return PyErr_NoMemory();
        }
    }

    if (vec.count != initial_cap + 10) {
        _PyGCSplitVector_Fini(&vec);
        PyErr_Format(PyExc_AssertionError,
                     "Expected count=%zu, got %zu",
                     initial_cap + 10, vec.count);
        return NULL;
    }
    if (vec.capacity <= initial_cap) {
        _PyGCSplitVector_Fini(&vec);
        PyErr_SetString(PyExc_AssertionError, "Capacity should have grown");
        return NULL;
    }

    _PyGCSplitVector_Clear(&vec);
    if (vec.count != 0) {
        _PyGCSplitVector_Fini(&vec);
        PyErr_SetString(PyExc_AssertionError, "Clear should set count=0");
        return NULL;
    }

    _PyGCSplitVector_Fini(&vec);
    Py_RETURN_NONE;
}

static PyObject *
parallel_gc_stackref_visits(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    PyObject *child = PyList_New(0);
    if (child == NULL) {
        return NULL;
    }
    PyObject_GC_UnTrack(child);

    _PyStackRef owned = PyStackRef_FromPyObjectStealMortal(child);
    _PyStackRef borrowed = PyStackRef_Borrow(owned);
    PyGC_Head *gc = _Py_AS_GC(child);
    uintptr_t saved_prev = gc->_gc_prev;
    gc->_gc_prev = _PyGC_PREV_MASK_COLLECTING |
                    ((uintptr_t)1 << _PyGC_PREV_SHIFT);

    int err = _PyGC_VisitStackRef(&borrowed, _PyGC_VisitDecref, child);
    Py_ssize_t after_borrowed =
        (Py_ssize_t)(gc->_gc_prev >> _PyGC_PREV_SHIFT);
    if (err == 0) {
        err = _PyGC_VisitStackRef(&owned, _PyGC_VisitDecref, child);
    }
    Py_ssize_t after_owned =
        (Py_ssize_t)(gc->_gc_prev >> _PyGC_PREV_SHIFT);

    gc->_gc_prev = saved_prev;
    PyStackRef_CLOSE(owned);

    if (err != 0) {
        return NULL;
    }
    if (after_borrowed != 1 || after_owned != 0) {
        PyErr_Format(PyExc_AssertionError,
                     "stack-ref visits changed gc_refs to %zd then %zd",
                     after_borrowed, after_owned);
        return NULL;
    }
    Py_RETURN_NONE;
}

#endif  // Py_PARALLEL_GC && !Py_GIL_DISABLED

#ifdef Py_PARALLEL_GC

typedef struct {
    unsigned long caller;
    Py_ssize_t caller_visits;
    Py_ssize_t helper_visits;
} traverse_probe_context;

typedef struct {
    PyObject_HEAD
    traverse_probe_context *context;
} traverse_probe_object;

static int
traverse_probe_traverse(PyObject *op, visitproc visit, void *arg)
{
    traverse_probe_object *probe = (traverse_probe_object *)op;
    if (PyThread_get_thread_ident() == probe->context->caller) {
        _Py_atomic_add_ssize(&probe->context->caller_visits, 1);
    }
    else {
        _Py_atomic_add_ssize(&probe->context->helper_visits, 1);
    }
    return 0;
}

static void
traverse_probe_dealloc(PyObject *op)
{
    PyObject_GC_UnTrack(op);
    PyObject_GC_Del(op);
}

static PyTypeObject TraverseProbe_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_testinternalcapi._TraverseProbe",
    .tp_basicsize = sizeof(traverse_probe_object),
    .tp_dealloc = traverse_probe_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = traverse_probe_traverse,
};

static PyObject *
parallel_gc_helper_visits(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    enum { probe_count = 40000 };

    if (PyType_Ready(&TraverseProbe_Type) < 0) {
        return NULL;
    }

    traverse_probe_context context = {
        .caller = PyThread_get_thread_ident(),
    };
    traverse_probe_object **probes = PyMem_New(
        traverse_probe_object *, probe_count);
    if (probes == NULL) {
        return PyErr_NoMemory();
    }

    Py_ssize_t created = 0;
    Py_ssize_t collected = 0;
    for (; created < probe_count; created++) {
        traverse_probe_object *probe = PyObject_GC_New(
            traverse_probe_object, &TraverseProbe_Type);
        if (probe == NULL) {
            break;
        }
        probe->context = &context;
        PyObject_GC_Track(probe);
        probes[created] = probe;
    }

    if (created == probe_count) {
        int was_enabled = PyGC_IsEnabled();
        PyGC_Enable();
        collected = PyGC_Collect();
        if (!was_enabled) {
            PyGC_Disable();
        }
    }

    for (Py_ssize_t i = 0; i < created; i++) {
        Py_DECREF(probes[i]);
    }
    PyMem_Free(probes);

    if (created != probe_count) {
        return NULL;
    }
    if (collected < 0) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "garbage collection failed");
        }
        return NULL;
    }
    return Py_BuildValue("nn",
                         _Py_atomic_load_ssize(&context.caller_visits),
                         _Py_atomic_load_ssize(&context.helper_visits));
}

#endif  // Py_PARALLEL_GC

#define TEST_METHOD(name) {#name, name, METH_NOARGS, NULL}

static PyMethodDef test_methods[] = {
#ifdef _Py_TEST_GC_BARRIER
    TEST_METHOD(unsafe_barrier_capacity_zero),
    TEST_METHOD(test_barrier_basic),
    TEST_METHOD(test_barrier_multiple_rounds),
    TEST_METHOD(test_barrier_epoch_distinguishes),
    TEST_METHOD(test_barrier_postcondition),
#endif
    TEST_METHOD(test_localbuffer_push_pop),
    TEST_METHOD(test_localbuffer_push_full),
    TEST_METHOD(test_overflow_flush_precondition),
    TEST_METHOD(test_overflow_flush_normal),
#if defined(Py_PARALLEL_GC) && !defined(Py_GIL_DISABLED)
    TEST_METHOD(test_splitvector_init_push),
    TEST_METHOD(parallel_gc_stackref_visits),
#endif
#ifdef Py_PARALLEL_GC
    TEST_METHOD(parallel_gc_helper_visits),
#endif
    {NULL, NULL, 0, NULL}
};

#undef TEST_METHOD

int
_PyTestInternalCapi_Init_ParallelGC(PyObject *mod)
{
    return PyModule_AddFunctions(mod, test_methods);
}
