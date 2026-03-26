// Test suite for work-stealing deque (_PyWSDeque)

#include "parts.h"
#include "pycore_ws_deque.h"      // _PyWSDeque

#include <pthread.h>                // pthread_create

// ============================================================================
// Basic Operations Tests
// ============================================================================

static PyObject *
test_ws_deque_init_fini(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyWSDeque deque;
    _PyWSDeque_Init(&deque);

    // Verify initial state
    if (_PyWSDeque_Size(&deque) != 0) {
        _PyWSDeque_Fini(&deque);
        PyErr_SetString(PyExc_AssertionError, "New deque should be empty");
        return NULL;
    }

    if (_PyWSDeque_GetNumResizes(&deque) != 0) {
        _PyWSDeque_Fini(&deque);
        PyErr_SetString(PyExc_AssertionError, "New deque should have 0 resizes");
        return NULL;
    }

    _PyWSDeque_Fini(&deque);
    Py_RETURN_NONE;
}

static PyObject *
test_ws_deque_push_take_single(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyWSDeque deque;
    _PyWSDeque_Init(&deque);

    // Create a test object
    PyObject *obj = PyLong_FromLong(42);
    if (obj == NULL) {
        _PyWSDeque_Fini(&deque);
        return NULL;
    }
    Py_INCREF(obj);  // Keep a reference for comparison

    // Push the object
    _PyWSDeque_Push(&deque, obj);

    // Verify size
    if (_PyWSDeque_Size(&deque) != 1) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_SetString(PyExc_AssertionError, "Deque size should be 1 after push");
        return NULL;
    }

    // Take the object back
    PyObject *result = _PyWSDeque_Take(&deque);

    // Verify we got the same object
    if (result != obj) {
        Py_XDECREF(result);
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_SetString(PyExc_AssertionError, "Take should return the pushed object");
        return NULL;
    }

    // Verify deque is empty
    if (_PyWSDeque_Size(&deque) != 0) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_SetString(PyExc_AssertionError, "Deque should be empty after take");
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
    _PyWSDeque_Init(&deque);

    // Create a test object
    PyObject *obj = PyLong_FromLong(123);
    if (obj == NULL) {
        _PyWSDeque_Fini(&deque);
        return NULL;
    }
    Py_INCREF(obj);  // Keep a reference for comparison

    // Push the object
    _PyWSDeque_Push(&deque, obj);

    // Steal the object
    PyObject *result = (PyObject *)_PyWSDeque_Steal(&deque);

    // Verify we got the same object
    if (result != obj) {
        Py_XDECREF(result);
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_SetString(PyExc_AssertionError, "Steal should return the pushed object");
        return NULL;
    }

    // Verify deque is empty
    if (_PyWSDeque_Size(&deque) != 0) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_SetString(PyExc_AssertionError, "Deque should be empty after steal");
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
    _PyWSDeque_Init(&deque);

    // Push multiple objects
    const int count = 10;
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
        Py_INCREF(objects[i]);  // Keep a reference
        _PyWSDeque_Push(&deque, objects[i]);
    }

    // Verify LIFO order (take should return in reverse order)
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

    // Cleanup
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
    _PyWSDeque_Init(&deque);

    // Push multiple objects
    const int count = 10;
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
        Py_INCREF(objects[i]);  // Keep a reference
        _PyWSDeque_Push(&deque, objects[i]);
    }

    // Verify FIFO order (steal should return in original order)
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

    // Cleanup
    for (int i = 0; i < count; i++) {
        Py_DECREF(objects[i]);
    }

    _PyWSDeque_Fini(&deque);
    Py_RETURN_NONE;
}

// ============================================================================
// Edge Cases
// ============================================================================

static PyObject *
test_ws_deque_take_empty(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyWSDeque deque;
    _PyWSDeque_Init(&deque);

    // Try to take from empty deque
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
    _PyWSDeque_Init(&deque);

    // Try to steal from empty deque
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
    _PyWSDeque_Init(&deque);

    // Push enough elements to trigger resize
    // Initial size is _Py_WSDEQUE_INITIAL_ARRAY_SIZE (4096)
    const int count = 5000;  // More than initial size
    PyObject *obj = PyLong_FromLong(42);
    if (obj == NULL) {
        _PyWSDeque_Fini(&deque);
        return NULL;
    }

    for (int i = 0; i < count; i++) {
        Py_INCREF(obj);
        _PyWSDeque_Push(&deque, obj);
    }

    // Verify resize happened
    int num_resizes = _PyWSDeque_GetNumResizes(&deque);
    if (num_resizes < 1) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_Format(PyExc_AssertionError,
                    "Expected at least 1 resize, got %d", num_resizes);
        return NULL;
    }

    // Verify size
    size_t size = _PyWSDeque_Size(&deque);
    if (size != (size_t)count) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_Format(PyExc_AssertionError,
                    "Expected size %d, got %zu", count, size);
        return NULL;
    }

    // Drain the deque
    for (int i = 0; i < count; i++) {
        PyObject *result = _PyWSDeque_Take(&deque);
        if (result == NULL) {
            Py_DECREF(obj);
            _PyWSDeque_Fini(&deque);
            PyErr_Format(PyExc_AssertionError,
                        "Failed to take element %d", i);
            return NULL;
        }
        Py_DECREF(result);  // Decrement the reference we added during push
    }

    Py_DECREF(obj);
    _PyWSDeque_Fini(&deque);
    Py_RETURN_NONE;
}

static PyObject *
test_ws_deque_init_with_undersized_buffer(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyWSDeque deque;

    // Provide a buffer that is too small — InitWithBuffer should fall back
    // to regular Init (malloc) and return 0 to indicate fallback.
    char tiny_buffer[16];
    size_t requested_size = 64;  // Needs much more than 16 bytes
    int result = _PyWSDeque_InitWithBuffer(
        &deque, tiny_buffer, sizeof(tiny_buffer), requested_size);

    // result == 0 means buffer was too small, fell back to malloc
    if (result != 0) {
        _PyWSDeque_Fini(&deque);
        PyErr_SetString(PyExc_AssertionError,
                        "Expected fallback (return 0) for undersized buffer");
        return NULL;
    }

    // Deque should still be functional (initialized via malloc fallback)
    PyObject *obj = PyLong_FromLong(99);
    if (obj == NULL) {
        _PyWSDeque_Fini(&deque);
        return NULL;
    }

    Py_INCREF(obj);
    _PyWSDeque_Push(&deque, obj);

    PyObject *taken = _PyWSDeque_Take(&deque);
    if (taken != obj) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_SetString(PyExc_AssertionError,
                        "Deque push/take failed after buffer fallback");
        return NULL;
    }
    Py_DECREF(taken);
    Py_DECREF(obj);
    _PyWSDeque_Fini(&deque);
    Py_RETURN_NONE;
}

static PyObject *
test_ws_deque_init_with_exact_buffer(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyWSDeque deque;

    // Provide a buffer that is exactly large enough
    size_t requested_size = 64;
    size_t required = sizeof(_PyWSArray) + sizeof(uintptr_t) * requested_size;
    char *buffer = PyMem_RawCalloc(1, required);
    if (buffer == NULL) {
        return PyErr_NoMemory();
    }

    int result = _PyWSDeque_InitWithBuffer(
        &deque, buffer, required, requested_size);

    // result == 1 means buffer was used successfully
    if (result != 1) {
        _PyWSDeque_Fini(&deque);
        PyMem_RawFree(buffer);
        PyErr_SetString(PyExc_AssertionError,
                        "Expected success (return 1) for exact-size buffer");
        return NULL;
    }

    // Deque should be functional
    PyObject *obj = PyLong_FromLong(42);
    if (obj == NULL) {
        _PyWSDeque_Fini(&deque);
        PyMem_RawFree(buffer);
        return NULL;
    }

    Py_INCREF(obj);
    _PyWSDeque_Push(&deque, obj);

    PyObject *taken = _PyWSDeque_Take(&deque);
    if (taken != obj) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyMem_RawFree(buffer);
        PyErr_SetString(PyExc_AssertionError,
                        "Deque push/take failed with exact buffer");
        return NULL;
    }
    Py_DECREF(taken);
    Py_DECREF(obj);

    // FiniExternal skips freeing the external buffer itself
    _PyWSDeque_FiniExternal(&deque, buffer);
    PyMem_RawFree(buffer);
    Py_RETURN_NONE;
}

// ============================================================================
// Concurrent Tests
// ============================================================================

typedef struct {
    _PyWSDeque *deque;
    int num_steals;
    int num_successful;
} steal_worker_args;

static void *
steal_worker(void *arg)
{
    steal_worker_args *args = (steal_worker_args *)arg;
    args->num_successful = 0;

    for (int i = 0; i < args->num_steals; i++) {
        void *obj = _PyWSDeque_Steal(args->deque);
        if (obj != NULL) {
            args->num_successful++;
            // In real use, would process obj here
            // For test, just count successes
        }
    }

    return NULL;
}

static PyObject *
test_ws_deque_concurrent_push_steal(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyWSDeque deque;
    _PyWSDeque_Init(&deque);

    const int num_items = 1000;
    const int num_workers = 4;
    const int steals_per_worker = 300;

    // Create test object
    PyObject *obj = PyLong_FromLong(42);
    if (obj == NULL) {
        _PyWSDeque_Fini(&deque);
        return NULL;
    }

    // Start worker threads
    pthread_t workers[num_workers];
    steal_worker_args args[num_workers];

    for (int i = 0; i < num_workers; i++) {
        args[i].deque = &deque;
        args[i].num_steals = steals_per_worker;
        args[i].num_successful = 0;
        pthread_create(&workers[i], NULL, steal_worker, &args[i]);
    }

    // Owner thread: push items
    for (int i = 0; i < num_items; i++) {
        Py_INCREF(obj);
        _PyWSDeque_Push(&deque, obj);
    }

    // Wait for workers
    for (int i = 0; i < num_workers; i++) {
        pthread_join(workers[i], NULL);
    }

    // Count total successful steals
    int total_stolen = 0;
    for (int i = 0; i < num_workers; i++) {
        total_stolen += args[i].num_successful;
    }

    // Remaining items in deque + stolen should equal pushed
    size_t remaining = _PyWSDeque_Size(&deque);

    // Drain remaining items
    int drained = 0;
    PyObject *result;
    while ((result = _PyWSDeque_Take(&deque)) != NULL) {
        Py_DECREF(result);
        drained++;
    }

    // Verify: pushed = stolen + drained
    if (total_stolen + drained != num_items) {
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

// ============================================================================
// Module Registration
// ============================================================================

static PyMethodDef test_methods[] = {
    // Basic operations
    {"test_ws_deque_init_fini", test_ws_deque_init_fini, METH_NOARGS, NULL},
    {"test_ws_deque_push_take_single", test_ws_deque_push_take_single, METH_NOARGS, NULL},
    {"test_ws_deque_push_steal_single", test_ws_deque_push_steal_single, METH_NOARGS, NULL},
    {"test_ws_deque_lifo_order", test_ws_deque_lifo_order, METH_NOARGS, NULL},
    {"test_ws_deque_fifo_order", test_ws_deque_fifo_order, METH_NOARGS, NULL},

    // Edge cases
    {"test_ws_deque_take_empty", test_ws_deque_take_empty, METH_NOARGS, NULL},
    {"test_ws_deque_steal_empty", test_ws_deque_steal_empty, METH_NOARGS, NULL},
    {"test_ws_deque_resize", test_ws_deque_resize, METH_NOARGS, NULL},
    {"test_ws_deque_init_with_undersized_buffer", test_ws_deque_init_with_undersized_buffer, METH_NOARGS, NULL},
    {"test_ws_deque_init_with_exact_buffer", test_ws_deque_init_with_exact_buffer, METH_NOARGS, NULL},

    // Concurrent
    {"test_ws_deque_concurrent_push_steal", test_ws_deque_concurrent_push_steal, METH_NOARGS, NULL},

    {NULL, NULL, 0, NULL}
};

int
_PyTestInternalCapi_Init_WSDeque(PyObject *mod)
{
    if (PyModule_AddFunctions(mod, test_methods) < 0) {
        return -1;
    }
    return 0;
}
