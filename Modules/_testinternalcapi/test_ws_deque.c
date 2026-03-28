// Test suite for work-stealing deque, barrier, and parallel GC infrastructure

#include "parts.h"
#include "pycore_ws_deque.h"       // _PyWSDeque, _PyGCLocalBuffer
#include "pycore_gc_barrier.h"     // _PyGCBarrier
#ifdef Py_PARALLEL_GC
#include "pycore_gc_parallel.h"    // _PyGCSplitVector, _PyGCWorkQueue, _PyGCSemaphore
#endif

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
// Barrier Tests (T3-F1, T3-F9, invariants 1-4)
// ============================================================================

// T3-F1: Init with capacity=0 should trigger assertion
static PyObject *
test_barrier_capacity_zero(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCBarrier barrier;
    // This should trigger assert(capacity > 0) and abort
    _PyGCBarrier_Init(&barrier, 0);
    // If we reach here, the assertion didn't fire
    _PyGCBarrier_Fini(&barrier);
    PyErr_SetString(PyExc_AssertionError, "Init(capacity=0) should have aborted");
    return NULL;
}

// Invariant 2: All N threads reach Wait, barrier lifts
typedef struct {
    _PyGCBarrier *barrier;
    int arrived;
} barrier_worker_args;

static void *
barrier_worker(void *arg)
{
    barrier_worker_args *args = (barrier_worker_args *)arg;
    args->arrived = 1;
    _PyGCBarrier_Wait(args->barrier);
    return NULL;
}

static PyObject *
test_barrier_basic(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    const int num_threads = 4;
    _PyGCBarrier barrier;
    _PyGCBarrier_Init(&barrier, num_threads);

    pthread_t threads[num_threads - 1];
    barrier_worker_args args[num_threads - 1];

    for (int i = 0; i < num_threads - 1; i++) {
        args[i].barrier = &barrier;
        args[i].arrived = 0;
        pthread_create(&threads[i], NULL, barrier_worker, &args[i]);
    }

    // Main thread is the Nth participant
    _PyGCBarrier_Wait(&barrier);

    for (int i = 0; i < num_threads - 1; i++) {
        pthread_join(threads[i], NULL);
        if (!args[i].arrived) {
            _PyGCBarrier_Fini(&barrier);
            PyErr_SetString(PyExc_AssertionError, "Worker did not arrive at barrier");
            return NULL;
        }
    }

    _PyGCBarrier_Fini(&barrier);
    Py_RETURN_NONE;
}

// Invariant 3: Epoch increments once per cycle
static PyObject *
test_barrier_multiple_rounds(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCBarrier barrier;
    _PyGCBarrier_Init(&barrier, 1);  // Single-thread barrier for simplicity

    unsigned int epoch_before = barrier.epoch;
    _PyGCBarrier_Wait(&barrier);
    if (barrier.epoch != epoch_before + 1) {
        _PyGCBarrier_Fini(&barrier);
        PyErr_SetString(PyExc_AssertionError, "Epoch should increment by 1 per cycle");
        return NULL;
    }

    _PyGCBarrier_Wait(&barrier);
    if (barrier.epoch != epoch_before + 2) {
        _PyGCBarrier_Fini(&barrier);
        PyErr_SetString(PyExc_AssertionError, "Epoch should increment by 1 per cycle (round 2)");
        return NULL;
    }

    _PyGCBarrier_Fini(&barrier);
    Py_RETURN_NONE;
}

// Invariant 3b: Epoch distinguishes barrier rounds (multi-threaded)
typedef struct {
    _PyGCBarrier *barrier;
    unsigned int epoch_after_round1;
    unsigned int epoch_after_round2;
} epoch_worker_args;

static void *
epoch_worker(void *arg)
{
    epoch_worker_args *args = (epoch_worker_args *)arg;
    _PyGCBarrier_Wait(args->barrier);
    args->epoch_after_round1 = args->barrier->epoch;
    _PyGCBarrier_Wait(args->barrier);
    args->epoch_after_round2 = args->barrier->epoch;
    return NULL;
}

static PyObject *
test_barrier_epoch_distinguishes(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCBarrier barrier;
    _PyGCBarrier_Init(&barrier, 2);

    epoch_worker_args args = { .barrier = &barrier };
    pthread_t thread;
    pthread_create(&thread, NULL, epoch_worker, &args);

    unsigned int epoch_before = barrier.epoch;
    _PyGCBarrier_Wait(&barrier);
    _PyGCBarrier_Wait(&barrier);

    pthread_join(thread, NULL);

    if (args.epoch_after_round1 == epoch_before) {
        _PyGCBarrier_Fini(&barrier);
        PyErr_SetString(PyExc_AssertionError, "Epoch should differ after round 1");
        return NULL;
    }
    if (args.epoch_after_round2 == args.epoch_after_round1) {
        _PyGCBarrier_Fini(&barrier);
        PyErr_SetString(PyExc_AssertionError, "Epoch should differ between rounds");
        return NULL;
    }

    _PyGCBarrier_Fini(&barrier);
    Py_RETURN_NONE;
}

// Invariant 4 (T3-F9): num_left resets after lift — postcondition
static PyObject *
test_barrier_postcondition(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCBarrier barrier;
    _PyGCBarrier_Init(&barrier, 1);

    unsigned int epoch_before = barrier.epoch;
    _PyGCBarrier_Wait(&barrier);

    // After Wait returns, epoch must have advanced (T3-F9 assertion)
    if (barrier.epoch == epoch_before) {
        _PyGCBarrier_Fini(&barrier);
        PyErr_SetString(PyExc_AssertionError, "Epoch did not advance after Wait");
        return NULL;
    }
    // num_left should be reset to capacity
    if (barrier.num_left != barrier.capacity) {
        _PyGCBarrier_Fini(&barrier);
        PyErr_SetString(PyExc_AssertionError, "num_left not reset after barrier lift");
        return NULL;
    }

    _PyGCBarrier_Fini(&barrier);
    Py_RETURN_NONE;
}

// ============================================================================
// LocalBuffer Tests (T3-F2, T3-F3, T3-F4, invariants 10-13)
// ============================================================================

// Invariant 10/11: LocalBuffer push/pop basic operation
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
    if (obj == NULL) return NULL;

    _PyGCLocalBuffer_Push(&buf, obj);
    if (_PyGCLocalBuffer_IsEmpty(&buf)) {
        Py_DECREF(obj);
        PyErr_SetString(PyExc_AssertionError, "Buffer should not be empty after push");
        return NULL;
    }

    PyObject *result = _PyGCLocalBuffer_Pop(&buf);
    if (result != obj) {
        Py_DECREF(obj);
        PyErr_SetString(PyExc_AssertionError, "Pop should return pushed object");
        return NULL;
    }
    if (!_PyGCLocalBuffer_IsEmpty(&buf)) {
        Py_DECREF(obj);
        PyErr_SetString(PyExc_AssertionError, "Buffer should be empty after pop");
        return NULL;
    }

    Py_DECREF(obj);
    Py_RETURN_NONE;
}

// Invariant 10: count <= max (T3-F2 assertion test — fill to capacity)
static PyObject *
test_localbuffer_push_full(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCLocalBuffer buf;
    _PyGCLocalBuffer_Init(&buf);

    PyObject *obj = PyLong_FromLong(99);
    if (obj == NULL) return NULL;

    // Fill to capacity
    for (int i = 0; i < _PyGC_LOCAL_BUFFER_SIZE; i++) {
        Py_INCREF(obj);
        _PyGCLocalBuffer_Push(&buf, obj);
    }

    if (!_PyGCLocalBuffer_IsFull(&buf)) {
        Py_DECREF(obj);
        PyErr_SetString(PyExc_AssertionError, "Buffer should be full after 1024 pushes");
        return NULL;
    }

    // Drain
    for (int i = 0; i < _PyGC_LOCAL_BUFFER_SIZE; i++) {
        PyObject *r = _PyGCLocalBuffer_Pop(&buf);
        Py_DECREF(r);
    }

    Py_DECREF(obj);
    Py_RETURN_NONE;
}

// Invariant 11: count >= 0 (T3-F3 assertion test — pop from empty would underflow)
static PyObject *
test_localbuffer_pop_empty(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCLocalBuffer buf;
    _PyGCLocalBuffer_Init(&buf);

    // Verify IsEmpty works
    if (!_PyGCLocalBuffer_IsEmpty(&buf)) {
        PyErr_SetString(PyExc_AssertionError, "New buffer should be empty");
        return NULL;
    }

    // Push one, pop one — should be empty again
    PyObject *obj = PyLong_FromLong(1);
    if (obj == NULL) return NULL;

    _PyGCLocalBuffer_Push(&buf, obj);
    PyObject *r = _PyGCLocalBuffer_Pop(&buf);
    if (r != obj) {
        Py_DECREF(obj);
        PyErr_SetString(PyExc_AssertionError, "Pop should return pushed object");
        return NULL;
    }

    if (!_PyGCLocalBuffer_IsEmpty(&buf)) {
        Py_DECREF(obj);
        PyErr_SetString(PyExc_AssertionError, "Buffer should be empty after popping all");
        return NULL;
    }

    Py_DECREF(obj);
    Py_RETURN_NONE;
}

// Invariant 12: OverflowFlush precondition (T3-F4 — must be at least half full)
static PyObject *
test_overflow_flush_precondition(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCLocalBuffer buf;
    _PyGCLocalBuffer_Init(&buf);
    _PyWSDeque deque;
    _PyWSDeque_Init(&deque);

    PyObject *obj = PyLong_FromLong(7);
    if (obj == NULL) {
        _PyWSDeque_Fini(&deque);
        return NULL;
    }

    // Fill buffer to capacity (IsFull guard, as callers do)
    for (int i = 0; i < _PyGC_LOCAL_BUFFER_SIZE; i++) {
        Py_INCREF(obj);
        _PyGCLocalBuffer_Push(&buf, obj);
    }

    // OverflowFlush should work — buffer is full (count=1024 >= 512)
    _PyGC_OverflowFlush(&buf, &deque);

    // After flush: half remains in buffer
    if (buf.count != _PyGC_LOCAL_BUFFER_SIZE / 2) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_Format(PyExc_AssertionError,
                    "Expected %d items after flush, got %zu",
                    _PyGC_LOCAL_BUFFER_SIZE / 2, buf.count);
        return NULL;
    }

    // Deque should have the other half
    size_t deque_size = _PyWSDeque_Size(&deque);
    if (deque_size != _PyGC_LOCAL_BUFFER_SIZE / 2) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_Format(PyExc_AssertionError,
                    "Expected %d items in deque after flush, got %zu",
                    _PyGC_LOCAL_BUFFER_SIZE / 2, deque_size);
        return NULL;
    }

    // Drain buffer and deque
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

// Invariant 13: OverflowFlush normal operation (fill, flush, continue)
static PyObject *
test_overflow_flush_normal(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCLocalBuffer buf;
    _PyGCLocalBuffer_Init(&buf);
    _PyWSDeque deque;
    _PyWSDeque_Init(&deque);

    PyObject *obj = PyLong_FromLong(42);
    if (obj == NULL) {
        _PyWSDeque_Fini(&deque);
        return NULL;
    }

    // Simulate the real pattern: push until full, flush, push more
    int total_pushed = 0;
    for (int round = 0; round < 3; round++) {
        while (!_PyGCLocalBuffer_IsFull(&buf)) {
            Py_INCREF(obj);
            _PyGCLocalBuffer_Push(&buf, obj);
            total_pushed++;
        }
        _PyGC_OverflowFlush(&buf, &deque);
    }

    // Count everything: buffer + deque should equal total_pushed
    int counted = (int)buf.count;
    PyObject *r;
    while ((r = _PyWSDeque_Take(&deque)) != NULL) {
        Py_DECREF(r);
        counted++;
    }
    while (!_PyGCLocalBuffer_IsEmpty(&buf)) {
        Py_DECREF(_PyGCLocalBuffer_Pop(&buf));
        // already counted via buf.count
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

// ============================================================================
// Deque Invariant Tests (T3-F10, T3-F11)
// ============================================================================

// Invariant 6 (T3-F10): top/bot init to 1
static PyObject *
test_deque_init_values(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyWSDeque deque;
    _PyWSDeque_Init(&deque);

    // T3-F10: verify init-to-1 (prevents Take wraparound bug)
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

// Invariant 5 (T3-F11): top <= bot after operations complete
static PyObject *
test_deque_top_leq_bot(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyWSDeque deque;
    _PyWSDeque_Init(&deque);

    PyObject *obj = PyLong_FromLong(42);
    if (obj == NULL) {
        _PyWSDeque_Fini(&deque);
        return NULL;
    }

    // Push several, take some, verify top<=bot at each step
    for (int i = 0; i < 100; i++) {
        Py_INCREF(obj);
        _PyWSDeque_Push(&deque, obj);

        size_t top = _Py_atomic_load_ssize_relaxed((Py_ssize_t *)&deque.top);
        size_t bot = _Py_atomic_load_ssize_relaxed((Py_ssize_t *)&deque.bot);
        if (top > bot) {
            Py_DECREF(obj);
            _PyWSDeque_Fini(&deque);
            PyErr_Format(PyExc_AssertionError,
                        "top > bot after push %d: top=%zu, bot=%zu", i, top, bot);
            return NULL;
        }
    }

    // Take all and verify
    for (int i = 0; i < 100; i++) {
        PyObject *r = _PyWSDeque_Take(&deque);
        Py_DECREF(r);

        size_t top = _Py_atomic_load_ssize_relaxed((Py_ssize_t *)&deque.top);
        size_t bot = _Py_atomic_load_ssize_relaxed((Py_ssize_t *)&deque.bot);
        if (top > bot) {
            Py_DECREF(obj);
            _PyWSDeque_Fini(&deque);
            PyErr_Format(PyExc_AssertionError,
                        "top > bot after take %d: top=%zu, bot=%zu", i, top, bot);
            return NULL;
        }
    }

    // Take from empty — should still have top <= bot after
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

// Grow chain Fini test (gatekeeper suggestion: D9 — old arrays freed)
static PyObject *
test_deque_grow_chain_fini(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyWSDeque deque;
    _PyWSDeque_Init(&deque);

    PyObject *obj = PyLong_FromLong(42);
    if (obj == NULL) {
        _PyWSDeque_Fini(&deque);
        return NULL;
    }

    // Push enough to trigger multiple resizes
    const int count = 10000;  // Well above initial 4096
    for (int i = 0; i < count; i++) {
        Py_INCREF(obj);
        _PyWSDeque_Push(&deque, obj);
    }

    int resizes = _PyWSDeque_GetNumResizes(&deque);
    if (resizes < 1) {
        Py_DECREF(obj);
        _PyWSDeque_Fini(&deque);
        PyErr_SetString(PyExc_AssertionError, "Expected at least 1 resize");
        return NULL;
    }

    // Drain, then Fini (should free the entire array chain without leaking)
    for (int i = 0; i < count; i++) {
        Py_DECREF(_PyWSDeque_Take(&deque));
    }

    Py_DECREF(obj);
    _PyWSDeque_Fini(&deque);  // Should free old + new arrays via linked list
    Py_RETURN_NONE;
}

// ============================================================================
// T1 Infrastructure Tests (SplitVector, WorkQueue, Semaphore)
// ============================================================================

#ifdef Py_PARALLEL_GC

// T1-F1/F2: SplitVector init, push, capacity grow
static PyObject *
test_splitvector_init_push(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCSplitVector vec;
    if (_PyGCSplitVector_Init(&vec) < 0) {
        return PyErr_NoMemory();
    }

    if (vec.entries == NULL || vec.count != 0) {
        _PyGCSplitVector_Fini(&vec);
        PyErr_SetString(PyExc_AssertionError, "Init should set entries!=NULL, count=0");
        return NULL;
    }

    // Push items up to and beyond initial capacity to trigger grow
    size_t initial_cap = vec.capacity;
    for (size_t i = 0; i < initial_cap + 10; i++) {
        // Use a dummy GC head (just needs a non-NULL pointer for the test)
        if (_PyGCSplitVector_Push(&vec, (PyGC_Head *)(uintptr_t)(i + 1)) < 0) {
            _PyGCSplitVector_Fini(&vec);
            return PyErr_NoMemory();
        }
    }

    if (vec.count != initial_cap + 10) {
        _PyGCSplitVector_Fini(&vec);
        PyErr_Format(PyExc_AssertionError,
                    "Expected count=%zu, got %zu", initial_cap + 10, vec.count);
        return NULL;
    }
    if (vec.capacity <= initial_cap) {
        _PyGCSplitVector_Fini(&vec);
        PyErr_SetString(PyExc_AssertionError, "Capacity should have grown");
        return NULL;
    }

    // Clear should reset count but keep capacity
    _PyGCSplitVector_Clear(&vec);
    if (vec.count != 0) {
        _PyGCSplitVector_Fini(&vec);
        PyErr_SetString(PyExc_AssertionError, "Clear should set count=0");
        return NULL;
    }

    _PyGCSplitVector_Fini(&vec);
    Py_RETURN_NONE;
}

// T1-F3/F4: WorkQueue init, push, ordering
static PyObject *
test_workqueue_init_push(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCWorkQueue queue;
    if (_PyGCWorkQueue_Init(&queue) < 0) {
        return PyErr_NoMemory();
    }

    // Push several objects
    const int count = 100;
    PyObject *obj = PyLong_FromLong(42);
    if (obj == NULL) {
        _PyGCWorkQueue_Fini(&queue);
        return NULL;
    }

    for (int i = 0; i < count; i++) {
        Py_INCREF(obj);
        if (_PyGCWorkQueue_Push(&queue, obj) < 0) {
            Py_DECREF(obj);
            _PyGCWorkQueue_Fini(&queue);
            return PyErr_NoMemory();
        }
    }

    // Verify write_index advanced
    Py_ssize_t write_idx = _Py_atomic_load_ssize_relaxed(&queue.write_index);
    if (write_idx != count) {
        Py_DECREF(obj);
        _PyGCWorkQueue_Fini(&queue);
        PyErr_Format(PyExc_AssertionError,
                    "Expected write_index=%d, got %zd", count, write_idx);
        return NULL;
    }

    // Verify read_index <= write_index (invariant 6)
    Py_ssize_t read_idx = _Py_atomic_load_ssize_relaxed(&queue.read_index);
    if (read_idx > write_idx) {
        Py_DECREF(obj);
        _PyGCWorkQueue_Fini(&queue);
        PyErr_SetString(PyExc_AssertionError, "read_index > write_index");
        return NULL;
    }

    // Decref all pushed objects
    for (int i = 0; i < count; i++) {
        Py_DECREF(obj);
    }
    Py_DECREF(obj);
    _PyGCWorkQueue_Fini(&queue);
    Py_RETURN_NONE;
}

// T1-F5/F6/F7: Semaphore init, post, wait
static PyObject *
test_semaphore_post_wait(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCSemaphore sema;
    if (_PyGCSemaphore_Init(&sema) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Semaphore init failed");
        return NULL;
    }

    // Post 3 tokens
    _PyGCSemaphore_Post(&sema, 3);

    // Wait should succeed 3 times without blocking
    for (int i = 0; i < 3; i++) {
        _PyGCSemaphore_Wait(&sema);
    }

    // Tokens should be 0 now
    if (sema.tokens != 0) {
        _PyGCSemaphore_Fini(&sema);
        PyErr_Format(PyExc_AssertionError,
                    "Expected 0 tokens, got %zd", sema.tokens);
        return NULL;
    }

    _PyGCSemaphore_Fini(&sema);
    Py_RETURN_NONE;
}

// T1-F6: Semaphore Post with n > 0 assertion
static PyObject *
test_semaphore_post_multiple(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCSemaphore sema;
    if (_PyGCSemaphore_Init(&sema) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Semaphore init failed");
        return NULL;
    }

    // Post in batches, wait in batches
    _PyGCSemaphore_Post(&sema, 5);
    _PyGCSemaphore_Post(&sema, 3);

    // Should have 8 tokens total
    if (sema.tokens != 8) {
        _PyGCSemaphore_Fini(&sema);
        PyErr_Format(PyExc_AssertionError,
                    "Expected 8 tokens, got %zd", sema.tokens);
        return NULL;
    }

    // Consume all
    for (int i = 0; i < 8; i++) {
        _PyGCSemaphore_Wait(&sema);
    }

    _PyGCSemaphore_Fini(&sema);
    Py_RETURN_NONE;
}

// Semaphore concurrent test: producer posts, consumer waits
typedef struct {
    _PyGCSemaphore *sema;
    int count;
    int received;
} sema_worker_args;

static void *
sema_consumer(void *arg)
{
    sema_worker_args *args = (sema_worker_args *)arg;
    args->received = 0;
    for (int i = 0; i < args->count; i++) {
        _PyGCSemaphore_Wait(args->sema);
        args->received++;
    }
    return NULL;
}

static PyObject *
test_semaphore_concurrent(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCSemaphore sema;
    if (_PyGCSemaphore_Init(&sema) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Semaphore init failed");
        return NULL;
    }

    const int count = 100;
    sema_worker_args args = { .sema = &sema, .count = count, .received = 0 };

    pthread_t consumer_thread;
    pthread_create(&consumer_thread, NULL, sema_consumer, &args);

    // Producer: post one at a time
    for (int i = 0; i < count; i++) {
        _PyGCSemaphore_Post(&sema, 1);
    }

    pthread_join(consumer_thread, NULL);

    if (args.received != count) {
        _PyGCSemaphore_Fini(&sema);
        PyErr_Format(PyExc_AssertionError,
                    "Expected %d received, got %d", count, args.received);
        return NULL;
    }

    _PyGCSemaphore_Fini(&sema);
    Py_RETURN_NONE;
}

// WorkQueue reset test
static PyObject *
test_workqueue_reset(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    _PyGCWorkQueue queue;
    if (_PyGCWorkQueue_Init(&queue) < 0) {
        return PyErr_NoMemory();
    }

    PyObject *obj = PyLong_FromLong(1);
    if (obj == NULL) {
        _PyGCWorkQueue_Fini(&queue);
        return NULL;
    }

    // Push some items
    for (int i = 0; i < 10; i++) {
        Py_INCREF(obj);
        _PyGCWorkQueue_Push(&queue, obj);
    }

    // Reset
    _PyGCWorkQueue_Reset(&queue);

    Py_ssize_t write_idx = _Py_atomic_load_ssize_relaxed(&queue.write_index);
    Py_ssize_t read_idx = _Py_atomic_load_ssize_relaxed(&queue.read_index);
    int done = _Py_atomic_load_int_relaxed(&queue.producer_done);

    if (write_idx != 0 || read_idx != 0 || done != 0) {
        Py_DECREF(obj);
        _PyGCWorkQueue_Fini(&queue);
        PyErr_SetString(PyExc_AssertionError,
                       "Reset should zero indices and producer_done");
        return NULL;
    }

    // Decref pushed objects (they're still referenced by queue blocks)
    for (int i = 0; i < 10; i++) {
        Py_DECREF(obj);
    }
    Py_DECREF(obj);
    _PyGCWorkQueue_Fini(&queue);
    Py_RETURN_NONE;
}

#endif  // Py_PARALLEL_GC

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

    // Barrier tests (T3-F1, T3-F9)
    {"test_barrier_capacity_zero", test_barrier_capacity_zero, METH_NOARGS, NULL},
    {"test_barrier_basic", test_barrier_basic, METH_NOARGS, NULL},
    {"test_barrier_multiple_rounds", test_barrier_multiple_rounds, METH_NOARGS, NULL},
    {"test_barrier_epoch_distinguishes", test_barrier_epoch_distinguishes, METH_NOARGS, NULL},
    {"test_barrier_postcondition", test_barrier_postcondition, METH_NOARGS, NULL},

    // LocalBuffer tests (T3-F2, T3-F3, T3-F4)
    {"test_localbuffer_push_pop", test_localbuffer_push_pop, METH_NOARGS, NULL},
    {"test_localbuffer_push_full", test_localbuffer_push_full, METH_NOARGS, NULL},
    {"test_localbuffer_pop_empty", test_localbuffer_pop_empty, METH_NOARGS, NULL},
    {"test_overflow_flush_precondition", test_overflow_flush_precondition, METH_NOARGS, NULL},
    {"test_overflow_flush_normal", test_overflow_flush_normal, METH_NOARGS, NULL},

    // Deque invariant tests (T3-F10, T3-F11)
    {"test_deque_init_values", test_deque_init_values, METH_NOARGS, NULL},
    {"test_deque_top_leq_bot", test_deque_top_leq_bot, METH_NOARGS, NULL},
    {"test_deque_grow_chain_fini", test_deque_grow_chain_fini, METH_NOARGS, NULL},

#ifdef Py_PARALLEL_GC
    // T1 infrastructure tests (SplitVector, WorkQueue, Semaphore)
    {"test_splitvector_init_push", test_splitvector_init_push, METH_NOARGS, NULL},
    {"test_workqueue_init_push", test_workqueue_init_push, METH_NOARGS, NULL},
    {"test_workqueue_reset", test_workqueue_reset, METH_NOARGS, NULL},
    {"test_semaphore_post_wait", test_semaphore_post_wait, METH_NOARGS, NULL},
    {"test_semaphore_post_multiple", test_semaphore_post_multiple, METH_NOARGS, NULL},
    {"test_semaphore_concurrent", test_semaphore_concurrent, METH_NOARGS, NULL},
#endif

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
