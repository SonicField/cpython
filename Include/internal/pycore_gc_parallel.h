// Parallel garbage collector for GIL-enabled Python.
// This provides parallel reference subtraction and marking for gc.c.

#ifndef Py_INTERNAL_GC_PARALLEL_H
#define Py_INTERNAL_GC_PARALLEL_H

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include "Python.h"
#include "pycore_condvar.h"        // PyMUTEX_T, PyCOND_T
#include "pycore_gc.h"             // PyGC_Head
#include "pycore_gc_barrier.h"     // _PyGCBarrier
#include "pycore_pythread.h"       // PyThread_handle_t
#include "pycore_ws_deque.h"       // _PyWSDeque, _PyGCLocalBuffer

// This implementation is used by GIL-enabled builds configured with
// parallel GC support. Free-threaded builds use a separate collector.

#if defined(Py_PARALLEL_GC) && !defined(Py_GIL_DISABLED)

#if SIZEOF_VOID_P < 8
#  error "Parallel GC requires 64-bit platform (SIZEOF_VOID_P >= 8)"
#endif

#if defined(__GNUC__) || defined(__clang__)
#  define _PyGC_PREFETCH(ptr, locality) \
        __builtin_prefetch((ptr), 0, (locality))
#elif defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
#  include <intrin.h>
#  define _PyGC_PREFETCH(ptr, locality) \
        _mm_prefetch((const char*)(ptr), (locality))
#else
#  define _PyGC_PREFETCH(ptr, locality) ((void)(ptr))
#endif

#define _PyGC_PREFETCH_T0(ptr) _PyGC_PREFETCH((ptr), 3)

// update_refs records these boundaries for parallel subtract_refs.
#define _PyGC_SPLIT_VECTOR_INITIAL_CAPACITY 128

typedef struct {
    PyGC_Head **entries;
    size_t count;
    size_t capacity;
} _PyGCSplitVector;

// Workers wait on per-worker (wake_mutex, wake_cond) for dispatch, then
// execute the phase selected by the main thread.

typedef enum {
    _PyGC_PHASE_IDLE,               // Waiting for work
    _PyGC_PHASE_SUBTRACT_REFS,      // Decrement gc_refs for internal refs
    _PyGC_PHASE_MARK,               // Parallel marking
} _PyGCPhase;

typedef enum {
    _PyGC_ASSERT_NONE,
    _PyGC_ASSERT_LIVE_REFERENT,
    _PyGC_ASSERT_POSITIVE_REFS,
} _PyGCDeferredAssertion;

typedef struct _PyParallelGCState _PyParallelGCState;

typedef struct {
    _PyWSDeque deque;
    _PyGCLocalBuffer local_buffer;

    PyGC_Head *slice_start;
    PyGC_Head *slice_end;

    _PyParallelGCState *par_gc;
    PyThread_handle_t thread;
    int should_exit;
    int error;
    _PyGCPhase phase;

    // Object assertions must run on the collecting thread.  Their diagnostic
    // acquires the GIL to format the object representation.
    _PyGCDeferredAssertion deferred_assertion;
    PyObject *assertion_object;
    PyObject *assertion_referent;

    PyMUTEX_T wake_mutex;
    PyCOND_T wake_cond;
    int wake_flag;
} _PyParallelGCWorker;

struct _PyParallelGCState {
    size_t num_workers;

    _PyGCSplitVector split_vector;

    // Workers must finish startup before Start returns.
    _PyGCBarrier startup_barrier;

    int num_workers_active;
    int enabled;

    PyMUTEX_T done_mutex;
    PyCOND_T done_cond;
    int workers_done_count;  // Protected by done_mutex.

    _PyParallelGCWorker workers[];
};

// Decrement visitors are compared by address when frame traversal decides
// whether an embedded stack reference contributes to an object's refcount.
PyAPI_FUNC(int) _PyGC_VisitDecref(PyObject *op, void *parent);
PyAPI_FUNC(int) _PyGC_ParallelVisitDecref(PyObject *op, void *arg);

// API Functions

// Initialize parallel GC with num_workers worker threads
// Returns 0 on success, -1 on error (with exception set)
PyAPI_FUNC(int) _PyGC_ParallelInit(
    PyInterpreterState *interp, size_t num_workers);

// Shutdown parallel GC and clean up all worker threads
PyAPI_FUNC(void) _PyGC_ParallelFini(PyInterpreterState *interp);

// Start worker threads (called after initialization)
PyAPI_FUNC(int) _PyGC_ParallelStart(PyInterpreterState *interp);

// Stop worker threads (but don't destroy state - can restart later)
PyAPI_FUNC(void) _PyGC_ParallelStop(PyInterpreterState *interp);

// Quiesce and restart the worker pool around fork().
PyAPI_FUNC(void) _PyGC_ParallelBeforeFork(PyInterpreterState *interp);
PyAPI_FUNC(void) _PyGC_ParallelAfterFork(PyInterpreterState *interp);
PyAPI_FUNC(void) _PyGC_ParallelAfterForkChild(PyInterpreterState *interp);

// Check if parallel GC is enabled
PyAPI_FUNC(int) _PyGC_ParallelIsEnabled(PyInterpreterState *interp);

// Get current configuration
PyAPI_FUNC(PyObject *) _PyGC_ParallelGetConfig(PyInterpreterState *interp);

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
    PyInterpreterState *interp
);

// Split Vector Operations

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

#endif  // Py_PARALLEL_GC && !Py_GIL_DISABLED

#ifdef __cplusplus
}
#endif

#endif // Py_INTERNAL_GC_PARALLEL_H
