// Tests for the parallel GC work-stealing deque.

#include "parts.h"
#include "pycore_pythread.h"       // PyThread_start_joinable_thread()
#include "pycore_ws_deque.h"       // _PyWSDeque
static int
deque_init(_PyWSDeque *deque)
{
    if (_PyWSDeque_Init(deque) < 0) {
        PyErr_NoMemory();
        return -1;
    }
    return 0;
}

static int
deque_push(_PyWSDeque *deque, void *item)
{
    if (_PyWSDeque_Push(deque, item) < 0) {
        PyErr_NoMemory();
        return -1;
    }
    return 0;
}

static void
join_thread_or_fatal(PyThread_handle_t thread)
{
    if (PyThread_join_thread(thread) != 0) {
        Py_FatalError("failed to join test worker thread");
    }
}

// Basic deque operations.

static PyObject *
test_ws_deque_init_fini(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyWSDeque deque;
    if (deque_init(&deque) < 0) {
        return NULL;
    }

    if (_PyWSDeque_Size(&deque) != 0) {
        _PyWSDeque_Fini(&deque);
        PyErr_SetString(PyExc_AssertionError, "New deque should be empty");
        return NULL;
    }

    if (_PyWSDeque_GetNumResizes(&deque) != 0) {
        _PyWSDeque_Fini(&deque);
        PyErr_SetString(PyExc_AssertionError,
                        "New deque should have 0 resizes");
        return NULL;
    }

    _PyWSDeque_Fini(&deque);
    Py_RETURN_NONE;
}

static PyObject *
test_ws_deque_push_take_single(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyWSDeque deque;
    if (deque_init(&deque) < 0) {
        return NULL;
    }

    PyObject *obj = PyLong_FromLong(42);
    if (obj == NULL) {
        _PyWSDeque_Fini(&deque);
        return NULL;
    }
    if (deque_push(&deque, obj) < 0) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        return NULL;
    }

    if (_PyWSDeque_Size(&deque) != 1) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_SetString(PyExc_AssertionError,
                        "Deque size should be 1 after push");
        return NULL;
    }

    PyObject *result = _PyWSDeque_Take(&deque);

    if (result != obj) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_SetString(PyExc_AssertionError,
                        "Take should return the pushed object");
        return NULL;
    }

    if (_PyWSDeque_Size(&deque) != 0) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_SetString(PyExc_AssertionError,
                        "Deque should be empty after take");
        return NULL;
    }

    Py_DECREF(obj);
    _PyWSDeque_Fini(&deque);
    Py_RETURN_NONE;
}

static PyObject *
test_ws_deque_push_steal_single(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyWSDeque deque;
    if (deque_init(&deque) < 0) {
        return NULL;
    }

    PyObject *obj = PyLong_FromLong(123);
    if (obj == NULL) {
        _PyWSDeque_Fini(&deque);
        return NULL;
    }
    if (deque_push(&deque, obj) < 0) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        return NULL;
    }

    PyObject *result = (PyObject *)_PyWSDeque_Steal(&deque);

    if (result != obj) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_SetString(PyExc_AssertionError,
                        "Steal should return the pushed object");
        return NULL;
    }

    if (_PyWSDeque_Size(&deque) != 0) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_SetString(PyExc_AssertionError,
                        "Deque should be empty after steal");
        return NULL;
    }

    Py_DECREF(obj);
    _PyWSDeque_Fini(&deque);
    Py_RETURN_NONE;
}

static PyObject *
test_ws_deque_lifo_order(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyWSDeque deque;
    if (deque_init(&deque) < 0) {
        return NULL;
    }

    enum { count = 10 };
    PyObject *objects[count];

    for (int i = 0; i < count; i++) {
        objects[i] = PyLong_FromLong(i);
        if (objects[i] == NULL) {
            for (int j = 0; j < i; j++) {
                Py_DECREF(objects[j]);
            }
            _PyWSDeque_Fini(&deque);
            return NULL;
        }
        if (deque_push(&deque, objects[i]) < 0) {
            for (int j = 0; j <= i; j++) {
                Py_DECREF(objects[j]);
            }
            _PyWSDeque_Fini(&deque);
            return NULL;
        }
    }

    for (int i = count - 1; i >= 0; i--) {
        PyObject *result = _PyWSDeque_Take(&deque);
        if (result != objects[i]) {
            for (int j = 0; j < count; j++) {
                Py_DECREF(objects[j]);
            }
            _PyWSDeque_Fini(&deque);
            PyErr_Format(PyExc_AssertionError,
                        "Expected object %d, got different object", i);
            return NULL;
        }
    }

    for (int i = 0; i < count; i++) {
        Py_DECREF(objects[i]);
    }

    _PyWSDeque_Fini(&deque);
    Py_RETURN_NONE;
}

static PyObject *
test_ws_deque_fifo_order(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyWSDeque deque;
    if (deque_init(&deque) < 0) {
        return NULL;
    }

    enum { count = 10 };
    PyObject *objects[count];

    for (int i = 0; i < count; i++) {
        objects[i] = PyLong_FromLong(i);
        if (objects[i] == NULL) {
            for (int j = 0; j < i; j++) {
                Py_DECREF(objects[j]);
            }
            _PyWSDeque_Fini(&deque);
            return NULL;
        }
        if (deque_push(&deque, objects[i]) < 0) {
            for (int j = 0; j <= i; j++) {
                Py_DECREF(objects[j]);
            }
            _PyWSDeque_Fini(&deque);
            return NULL;
        }
    }

    for (int i = 0; i < count; i++) {
        PyObject *result = (PyObject *)_PyWSDeque_Steal(&deque);
        if (result != objects[i]) {
            for (int j = 0; j < count; j++) {
                Py_DECREF(objects[j]);
            }
            _PyWSDeque_Fini(&deque);
            PyErr_Format(PyExc_AssertionError,
                        "Expected object %d, got different object", i);
            return NULL;
        }
    }

    for (int i = 0; i < count; i++) {
        Py_DECREF(objects[i]);
    }

    _PyWSDeque_Fini(&deque);
    Py_RETURN_NONE;
}

// Edge cases.

static PyObject *
test_ws_deque_take_empty(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyWSDeque deque;
    if (deque_init(&deque) < 0) {
        return NULL;
    }

    PyObject *result = _PyWSDeque_Take(&deque);

    if (result != NULL) {
        Py_DECREF(result);
        _PyWSDeque_Fini(&deque);
        PyErr_SetString(PyExc_AssertionError,
                       "Take from empty deque should return NULL");
        return NULL;
    }

    _PyWSDeque_Fini(&deque);
    Py_RETURN_NONE;
}

static PyObject *
test_ws_deque_steal_empty(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyWSDeque deque;
    if (deque_init(&deque) < 0) {
        return NULL;
    }

    PyObject *result = (PyObject *)_PyWSDeque_Steal(&deque);

    if (result != NULL) {
        Py_DECREF(result);
        _PyWSDeque_Fini(&deque);
        PyErr_SetString(PyExc_AssertionError,
                       "Steal from empty deque should return NULL");
        return NULL;
    }

    _PyWSDeque_Fini(&deque);
    Py_RETURN_NONE;
}

static PyObject *
test_ws_deque_resize(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyWSDeque deque;
    if (deque_init(&deque) < 0) {
        return NULL;
    }

    const int count = 5000;
    PyObject *obj = PyLong_FromLong(42);
    if (obj == NULL) {
        _PyWSDeque_Fini(&deque);
        return NULL;
    }

    for (int i = 0; i < count; i++) {
        if (deque_push(&deque, obj) < 0) {
            Py_DECREF(obj);
            _PyWSDeque_Fini(&deque);
            return NULL;
        }
    }

    int num_resizes = _PyWSDeque_GetNumResizes(&deque);
    if (num_resizes < 1) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_Format(PyExc_AssertionError,
                    "Expected at least 1 resize, got %d", num_resizes);
        return NULL;
    }

    size_t size = _PyWSDeque_Size(&deque);
    if (size != (size_t)count) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_Format(PyExc_AssertionError,
                    "Expected size %d, got %zu", count, size);
        return NULL;
    }

    for (int i = 0; i < count; i++) {
        PyObject *result = _PyWSDeque_Take(&deque);
        if (result == NULL) {
            Py_DECREF(obj);
            _PyWSDeque_Fini(&deque);
            PyErr_Format(PyExc_AssertionError,
                        "Failed to take element %d", i);
            return NULL;
        }
    }

    Py_DECREF(obj);
    _PyWSDeque_Fini(&deque);
    Py_RETURN_NONE;
}

static PyObject *
ws_deque_grow_oom(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyWSDeque deque;
    if (_PyWSDeque_Init(&deque) < 0) {
        PyErr_SetString(PyExc_AssertionError,
                        "initial deque allocation unexpectedly failed");
        return NULL;
    }

    int sentinel;
    for (size_t i = 0; i < _Py_WSDEQUE_INITIAL_ARRAY_SIZE; i++) {
        if (_PyWSDeque_Push(&deque, &sentinel) < 0) {
            _PyWSDeque_Fini(&deque);
            PyErr_SetString(PyExc_AssertionError,
                            "deque filled before the growth allocation");
            return NULL;
        }
    }

    if (_PyWSDeque_Push(&deque, &sentinel) == 0) {
        _PyWSDeque_Fini(&deque);
        PyErr_SetString(PyExc_AssertionError,
                        "deque growth unexpectedly succeeded");
        return NULL;
    }
    if (_PyWSDeque_Size(&deque) != _Py_WSDEQUE_INITIAL_ARRAY_SIZE) {
        _PyWSDeque_Fini(&deque);
        PyErr_SetString(PyExc_AssertionError,
                        "failed growth changed the deque size");
        return NULL;
    }

    _PyWSDeque_Reset(&deque);
    if (_PyWSDeque_Size(&deque) != 0) {
        _PyWSDeque_Fini(&deque);
        PyErr_SetString(PyExc_AssertionError,
                        "reset did not empty the deque");
        return NULL;
    }
    _PyWSDeque_Fini(&deque);
    Py_RETURN_NONE;
}

// Concurrent operations.

typedef struct {
    _PyWSDeque *deque;
    int *start;
    int *producer_done;
    int num_successful;
} steal_worker_args;

static void
steal_worker(void *arg)
{
    steal_worker_args *args = (steal_worker_args *)arg;
    args->num_successful = 0;

    while (!_Py_atomic_load_int(args->start)) {
    }
    while (!_Py_atomic_load_int(args->producer_done) ||
           _PyWSDeque_Size(args->deque) != 0)
    {
        void *obj = _PyWSDeque_Steal(args->deque);
        if (obj != NULL) {
            args->num_successful++;
        }
    }
}

static PyObject *
test_ws_deque_concurrent_push_steal(
    PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyWSDeque deque;
    if (deque_init(&deque) < 0) {
        return NULL;
    }

    enum {
        num_items = 1000,
        num_workers = 4,
    };

    PyObject *obj = PyLong_FromLong(42);
    if (obj == NULL) {
        _PyWSDeque_Fini(&deque);
        return NULL;
    }

    PyThread_handle_t workers[num_workers];
    PyThread_ident_t worker_ids[num_workers];
    steal_worker_args args[num_workers];
    int workers_created = 0;
    int start = 0;
    int producer_done = 0;

    for (int i = 0; i < num_workers; i++) {
        args[i].deque = &deque;
        args[i].start = &start;
        args[i].producer_done = &producer_done;
        args[i].num_successful = 0;
        int rc = PyThread_start_joinable_thread(
            steal_worker, &args[i], &worker_ids[i], &workers[i]);
        if (rc != 0) {
            _Py_atomic_store_int(&producer_done, 1);
            _Py_atomic_store_int(&start, 1);
            for (int j = 0; j < workers_created; j++) {
                join_thread_or_fatal(workers[j]);
            }
            Py_DECREF(obj);
            _PyWSDeque_Fini(&deque);
            PyErr_Format(PyExc_RuntimeError,
                         "failed to create deque test worker: error %d", rc);
            return NULL;
        }
        workers_created++;
    }

    _Py_atomic_store_int(&start, 1);
    for (int i = 0; i < num_items; i++) {
        if (deque_push(&deque, obj) < 0) {
            _Py_atomic_store_int(&producer_done, 1);
            for (int j = 0; j < workers_created; j++) {
                join_thread_or_fatal(workers[j]);
            }
            Py_DECREF(obj);
            _PyWSDeque_Fini(&deque);
            return NULL;
        }
    }
    _Py_atomic_store_int(&producer_done, 1);

    for (int i = 0; i < workers_created; i++) {
        join_thread_or_fatal(workers[i]);
    }

    int total_stolen = 0;
    for (int i = 0; i < num_workers; i++) {
        total_stolen += args[i].num_successful;
    }

    int drained = 0;
    PyObject *result;
    while ((result = _PyWSDeque_Take(&deque)) != NULL) {
        drained++;
    }

    if (total_stolen != num_items || drained != 0) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_Format(PyExc_AssertionError,
                    "Expected %d items total, got %d stolen + %d drained = %d",
                    num_items, total_stolen, drained, total_stolen + drained);
        return NULL;
    }

    Py_DECREF(obj);
    _PyWSDeque_Fini(&deque);
    Py_RETURN_NONE;
}

// Deque invariant tests.

static PyObject *
test_deque_init_values(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyWSDeque deque;
    if (deque_init(&deque) < 0) {
        return NULL;
    }

    // Starting at one prevents Take() from wrapping on an empty deque.
    size_t top = _Py_atomic_load_ssize_relaxed((Py_ssize_t *)&deque.top);
    size_t bot = _Py_atomic_load_ssize_relaxed((Py_ssize_t *)&deque.bot);

    if (top != 1 || bot != 1) {
        _PyWSDeque_Fini(&deque);
        PyErr_Format(PyExc_AssertionError,
                    "Expected top=1, bot=1, got top=%zu, bot=%zu", top, bot);
        return NULL;
    }

    _PyWSDeque_Fini(&deque);
    Py_RETURN_NONE;
}

static PyObject *
test_deque_top_leq_bot(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyWSDeque deque;
    if (deque_init(&deque) < 0) {
        return NULL;
    }

    PyObject *obj = PyLong_FromLong(42);
    if (obj == NULL) {
        _PyWSDeque_Fini(&deque);
        return NULL;
    }

    for (int i = 0; i < 100; i++) {
        if (deque_push(&deque, obj) < 0) {
            Py_DECREF(obj);
            _PyWSDeque_Fini(&deque);
            return NULL;
        }

        size_t top = _Py_atomic_load_ssize_relaxed((Py_ssize_t *)&deque.top);
        size_t bot = _Py_atomic_load_ssize_relaxed((Py_ssize_t *)&deque.bot);
        if (top > bot) {
            Py_DECREF(obj);
            _PyWSDeque_Fini(&deque);
            PyErr_Format(PyExc_AssertionError,
                         "top > bot after push %d: top=%zu, bot=%zu",
                         i, top, bot);
            return NULL;
        }
    }

    for (int i = 0; i < 100; i++) {
        PyObject *r = _PyWSDeque_Take(&deque);
        if (r == NULL) {
            Py_DECREF(obj);
            _PyWSDeque_Fini(&deque);
            PyErr_SetString(PyExc_AssertionError, "Deque unexpectedly empty");
            return NULL;
        }

        size_t top = _Py_atomic_load_ssize_relaxed((Py_ssize_t *)&deque.top);
        size_t bot = _Py_atomic_load_ssize_relaxed((Py_ssize_t *)&deque.bot);
        if (top > bot) {
            Py_DECREF(obj);
            _PyWSDeque_Fini(&deque);
            PyErr_Format(PyExc_AssertionError,
                         "top > bot after take %d: top=%zu, bot=%zu",
                         i, top, bot);
            return NULL;
        }
    }

    // Taking from an empty deque must preserve the index invariant.
    PyObject *r = _PyWSDeque_Take(&deque);
    if (r != NULL) {
        Py_DECREF(r);
    }
    size_t top = _Py_atomic_load_ssize_relaxed((Py_ssize_t *)&deque.top);
    size_t bot = _Py_atomic_load_ssize_relaxed((Py_ssize_t *)&deque.bot);
    if (top > bot) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_Format(PyExc_AssertionError,
                    "top > bot after empty take: top=%zu, bot=%zu", top, bot);
        return NULL;
    }

    Py_DECREF(obj);
    _PyWSDeque_Fini(&deque);
    Py_RETURN_NONE;
}

static PyObject *
test_deque_grow_chain_fini(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyWSDeque deque;
    if (deque_init(&deque) < 0) {
        return NULL;
    }

    PyObject *obj = PyLong_FromLong(42);
    if (obj == NULL) {
        _PyWSDeque_Fini(&deque);
        return NULL;
    }

    const int count = 10000;
    for (int i = 0; i < count; i++) {
        if (deque_push(&deque, obj) < 0) {
            Py_DECREF(obj);
            _PyWSDeque_Fini(&deque);
            return NULL;
        }
    }

    int resizes = _PyWSDeque_GetNumResizes(&deque);
    if (resizes < 1) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_SetString(PyExc_AssertionError, "Expected at least 1 resize");
        return NULL;
    }

    for (int i = 0; i < count; i++) {
        if (_PyWSDeque_Take(&deque) == NULL) {
            Py_DECREF(obj);
            _PyWSDeque_Fini(&deque);
            PyErr_SetString(PyExc_AssertionError, "Deque unexpectedly empty");
            return NULL;
        }
    }

    Py_DECREF(obj);
    _PyWSDeque_Fini(&deque);
    Py_RETURN_NONE;
}

#define TEST_METHOD(name) {#name, name, METH_NOARGS, NULL}

static PyMethodDef test_methods[] = {
    TEST_METHOD(test_ws_deque_init_fini),
    TEST_METHOD(test_ws_deque_push_take_single),
    TEST_METHOD(test_ws_deque_push_steal_single),
    TEST_METHOD(test_ws_deque_lifo_order),
    TEST_METHOD(test_ws_deque_fifo_order),
    TEST_METHOD(test_ws_deque_take_empty),
    TEST_METHOD(test_ws_deque_steal_empty),
    TEST_METHOD(test_ws_deque_resize),
    TEST_METHOD(ws_deque_grow_oom),
    TEST_METHOD(test_ws_deque_concurrent_push_steal),
    TEST_METHOD(test_deque_init_values),
    TEST_METHOD(test_deque_top_leq_bot),
    TEST_METHOD(test_deque_grow_chain_fini),
    {NULL, NULL, 0, NULL}
};

#undef TEST_METHOD

int
_PyTestInternalCapi_Init_WSDeque(PyObject *mod)
{
    return PyModule_AddFunctions(mod, test_methods);
}
