// Barrier synchronization for parallel GC worker threads.
//
// 3.12-compatible version: uses PyMUTEX_T/PyCOND_T types from
// pycore_condvar.h with direct platform calls for operations
// (the PyMUTEX_LOCK etc. macros live in Python/condvar.h which
// is not available to Include/internal/ headers).

#ifndef Py_INTERNAL_GC_BARRIER_H
#define Py_INTERNAL_GC_BARRIER_H

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include "pycore_condvar.h"  // PyMUTEX_T, PyCOND_T
#include <assert.h>          // assert

// =============================================================================
// Mutex/Condvar Operation Macros for Parallel GC
// =============================================================================
// pycore_condvar.h defines the types (PyMUTEX_T, PyCOND_T) but not operations.
// The operation macros (PyMUTEX_LOCK, PyCOND_WAIT, etc.) live in
// Python/condvar.h which is only includable from Python/*.c files.
// We define our own portable wrappers here for POSIX and Windows.

#ifdef _POSIX_THREADS
#include <pthread.h>

#define _PyGC_MUTEX_INIT(mut)       pthread_mutex_init((mut), NULL)
#define _PyGC_MUTEX_FINI(mut)       pthread_mutex_destroy((mut))
#define _PyGC_MUTEX_LOCK(mut)       pthread_mutex_lock((mut))
#define _PyGC_MUTEX_UNLOCK(mut)     pthread_mutex_unlock((mut))
#define _PyGC_COND_INIT(cond)       pthread_cond_init((cond), NULL)
#define _PyGC_COND_FINI(cond)       pthread_cond_destroy((cond))
#define _PyGC_COND_WAIT(cond, mut)  pthread_cond_wait((cond), (mut))
#define _PyGC_COND_BROADCAST(cond)  pthread_cond_broadcast((cond))

#elif defined(NT_THREADS)

// Windows: PyMUTEX_T and PyCOND_T differ based on _PY_EMULATED_WIN_CV.
// On Vista+ (non-emulated): PyMUTEX_T is SRWLOCK, PyCOND_T is CONDITION_VARIABLE.
// On XP (emulated): PyMUTEX_T is CRITICAL_SECTION, PyCOND_T is struct with sem.
// We support both variants.

#if _PY_EMULATED_WIN_CV

// Emulated (XP-compatible): CriticalSection + Semaphore
#define _PyGC_MUTEX_INIT(mut)       (InitializeCriticalSection((mut)), 0)
#define _PyGC_MUTEX_FINI(mut)       DeleteCriticalSection((mut))
#define _PyGC_MUTEX_LOCK(mut)       EnterCriticalSection((mut))
#define _PyGC_MUTEX_UNLOCK(mut)     LeaveCriticalSection((mut))

// For the emulated PyCOND_T, we need the semaphore-based operations.
// However, PyCOND_BROADCAST in the emulated case uses the 'waiting' counter.
// The barrier guarantees correct usage (mutex held during signal/broadcast),
// so we can safely use the semaphore approach.
static inline void
_PyGC_COND_INIT_emulated(PyCOND_T *cv) {
    cv->sem = CreateSemaphore(NULL, 0, 100000, NULL);
    cv->waiting = 0;
}
#define _PyGC_COND_INIT(cond)       _PyGC_COND_INIT_emulated((cond))
#define _PyGC_COND_FINI(cond)       CloseHandle((cond)->sem)

static inline void
_PyGC_COND_WAIT_emulated(PyCOND_T *cv, PyMUTEX_T *cs) {
    cv->waiting++;
    LeaveCriticalSection(cs);
    WaitForSingleObjectEx(cv->sem, INFINITE, FALSE);
    EnterCriticalSection(cs);
}
#define _PyGC_COND_WAIT(cond, mut)  _PyGC_COND_WAIT_emulated((cond), (mut))

static inline void
_PyGC_COND_BROADCAST_emulated(PyCOND_T *cv) {
    int waiting = cv->waiting;
    if (waiting > 0) {
        cv->waiting = 0;
        ReleaseSemaphore(cv->sem, waiting, NULL);
    }
}
#define _PyGC_COND_BROADCAST(cond)  _PyGC_COND_BROADCAST_emulated((cond))

#else /* !_PY_EMULATED_WIN_CV -- Vista+ native primitives */

// Native (Vista+): SRWLOCK + CONDITION_VARIABLE
#define _PyGC_MUTEX_INIT(mut)       InitializeSRWLock((mut))
#define _PyGC_MUTEX_FINI(mut)       ((void)0)  // SRWLOCK doesn't need cleanup
#define _PyGC_MUTEX_LOCK(mut)       AcquireSRWLockExclusive((mut))
#define _PyGC_MUTEX_UNLOCK(mut)     ReleaseSRWLockExclusive((mut))
#define _PyGC_COND_INIT(cond)       InitializeConditionVariable((cond))
#define _PyGC_COND_FINI(cond)       ((void)0)  // CONDITION_VARIABLE doesn't need cleanup
#define _PyGC_COND_WAIT(cond, mut)  SleepConditionVariableSRW((cond), (mut), INFINITE, 0)
#define _PyGC_COND_BROADCAST(cond)  WakeAllConditionVariable((cond))

#endif /* _PY_EMULATED_WIN_CV */

#else
#error "Parallel GC requires either POSIX threads or NT threads"
#endif

// =============================================================================
// Barrier Synchronization for Parallel GC
// =============================================================================
//
// A barrier for synchronizing N threads in parallel garbage collection.
// All N threads must reach the barrier before it is lifted, unblocking all
// threads simultaneously.
//
// This is shared between both GIL and FTP parallel GC implementations.
// Uses CPython's portable mutex/condvar types (PyMUTEX_T/PyCOND_T from
// pycore_condvar.h) with direct platform calls for cross-platform
// compatibility (POSIX and Windows via NT_THREADS).

typedef struct {
    // Number of threads left to reach the barrier before it can be lifted
    unsigned int num_left;

    // Total number of threads managed by the barrier
    unsigned int capacity;

    // The epoch advances once all threads reach the barrier; it
    // disambiguates spurious wakeups from true wakeups that happen once all
    // threads have reached the barrier.
    unsigned int epoch;

    PyMUTEX_T lock;
    PyCOND_T cond;
} _PyGCBarrier;

// Initialize barrier for capacity threads
static inline void
_PyGCBarrier_Init(_PyGCBarrier *barrier, unsigned int capacity)
{
    assert(capacity > 0);  // T3-F1: capacity=0 causes unsigned underflow in Wait
    barrier->capacity = capacity;
    barrier->num_left = capacity;
    barrier->epoch = 0;
    _PyGC_MUTEX_INIT(&barrier->lock);
    _PyGC_COND_INIT(&barrier->cond);
}

// Finalize barrier resources
static inline void
_PyGCBarrier_Fini(_PyGCBarrier *barrier)
{
    _PyGC_COND_FINI(&barrier->cond);
    _PyGC_MUTEX_FINI(&barrier->lock);
}

// Wait at barrier - blocks until all threads arrive
static inline void
_PyGCBarrier_Wait(_PyGCBarrier *barrier)
{
    _PyGC_MUTEX_LOCK(&barrier->lock);

    unsigned int current_epoch = barrier->epoch;
    barrier->num_left--;

    if (barrier->num_left == 0) {
        // Last thread to arrive - lift the barrier
        barrier->epoch++;
        barrier->num_left = barrier->capacity;
        _PyGC_COND_BROADCAST(&barrier->cond);
    } else {
        // Wait until the barrier is lifted
        while (barrier->epoch == current_epoch) {
            _PyGC_COND_WAIT(&barrier->cond, &barrier->lock);
        }
    }

    assert(barrier->epoch != current_epoch);  // T3-F9: barrier actually lifted
    _PyGC_MUTEX_UNLOCK(&barrier->lock);
}

#ifdef __cplusplus
}
#endif

#endif // Py_INTERNAL_GC_BARRIER_H
