// Copyright (c) Meta Platforms, Inc. and affiliates.
// Ported to CPython by Alex Turner

#ifndef Py_INTERNAL_GC_BARRIER_H
#define Py_INTERNAL_GC_BARRIER_H

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include "pycore_condvar.h"  // PyMUTEX_T, PyCOND_T

// =============================================================================
// Mutex/Condvar Operation Macros for Parallel GC
// =============================================================================
// pycore_condvar.h defines the types (PyMUTEX_T, PyCOND_T) but not operations.
// These macros provide portable operations for POSIX and Windows.

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
// Windows: use native SRWLOCK and CONDITION_VARIABLE
// Note: PyMUTEX_T is SRWLOCK, PyCOND_T is CONDITION_VARIABLE in pycore_condvar.h

#define _PyGC_MUTEX_INIT(mut)       InitializeSRWLock((mut))
#define _PyGC_MUTEX_FINI(mut)       ((void)0)  // SRWLOCK doesn't need cleanup
#define _PyGC_MUTEX_LOCK(mut)       AcquireSRWLockExclusive((mut))
#define _PyGC_MUTEX_UNLOCK(mut)     ReleaseSRWLockExclusive((mut))
#define _PyGC_COND_INIT(cond)       InitializeConditionVariable((cond))
#define _PyGC_COND_FINI(cond)       ((void)0)  // CONDITION_VARIABLE doesn't need cleanup
#define _PyGC_COND_WAIT(cond, mut)  SleepConditionVariableSRW((cond), (mut), INFINITE, 0)
#define _PyGC_COND_BROADCAST(cond)  WakeAllConditionVariable((cond))

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
// Uses CPython's portable mutex/condvar wrappers (PyMUTEX_T/PyCOND_T) for
// cross-platform compatibility (POSIX and Windows via NT_THREADS).

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
_PyGCBarrier_Init(_PyGCBarrier *barrier, int capacity)
{
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

    _PyGC_MUTEX_UNLOCK(&barrier->lock);
}

#ifdef __cplusplus
}
#endif

#endif // Py_INTERNAL_GC_BARRIER_H
