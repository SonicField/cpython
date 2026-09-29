// Barrier synchronization for parallel GC worker threads.

#ifndef Py_INTERNAL_GC_BARRIER_H
#define Py_INTERNAL_GC_BARRIER_H

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include "pycore_condvar.h"  // PyMUTEX_T, PyCOND_T
#include <assert.h>

// pycore_condvar.h defines the types but not their operations.

#ifdef _POSIX_THREADS
#include <pthread.h>

#define _PyGC_SYNC_CHECK(call) \
    do { \
        if ((call) != 0) { \
            Py_FatalError(#call " failed"); \
        } \
    } while (0)
#define _PyGC_MUTEX_INIT(mut) \
    _PyGC_SYNC_CHECK(pthread_mutex_init((mut), NULL))
#define _PyGC_MUTEX_FINI(mut) \
    _PyGC_SYNC_CHECK(pthread_mutex_destroy((mut)))
#define _PyGC_MUTEX_LOCK(mut) \
    _PyGC_SYNC_CHECK(pthread_mutex_lock((mut)))
#define _PyGC_MUTEX_UNLOCK(mut) \
    _PyGC_SYNC_CHECK(pthread_mutex_unlock((mut)))
#define _PyGC_COND_INIT(cond) \
    _PyGC_SYNC_CHECK(pthread_cond_init((cond), NULL))
#define _PyGC_COND_FINI(cond) \
    _PyGC_SYNC_CHECK(pthread_cond_destroy((cond)))
#define _PyGC_COND_WAIT(cond, mut) \
    _PyGC_SYNC_CHECK(pthread_cond_wait((cond), (mut)))
#define _PyGC_COND_SIGNAL(cond) \
    _PyGC_SYNC_CHECK(pthread_cond_signal((cond)))
#define _PyGC_COND_BROADCAST(cond) \
    _PyGC_SYNC_CHECK(pthread_cond_broadcast((cond)))

#elif defined(NT_THREADS)
// PyMUTEX_T is SRWLOCK and PyCOND_T is CONDITION_VARIABLE on Windows.

#define _PyGC_MUTEX_INIT(mut)       InitializeSRWLock((mut))
#define _PyGC_MUTEX_FINI(mut)       ((void)0)
#define _PyGC_MUTEX_LOCK(mut)       AcquireSRWLockExclusive((mut))
#define _PyGC_MUTEX_UNLOCK(mut)     ReleaseSRWLockExclusive((mut))
#define _PyGC_COND_INIT(cond)       InitializeConditionVariable((cond))
#define _PyGC_COND_FINI(cond)       ((void)0)
#define _PyGC_COND_WAIT(cond, mut) \
    do { \
        if (!SleepConditionVariableSRW((cond), (mut), INFINITE, 0)) { \
            Py_FatalError("SleepConditionVariableSRW failed"); \
        } \
    } while (0)
#define _PyGC_COND_SIGNAL(cond)     WakeConditionVariable((cond))
#define _PyGC_COND_BROADCAST(cond)  WakeAllConditionVariable((cond))

#else
#error "Parallel GC requires either POSIX threads or NT threads"
#endif

// Reusable barrier used while starting GIL-build collector helpers.

typedef struct {
    unsigned int num_left;
    unsigned int capacity;

    // The epoch advances once all threads reach the barrier; it
    // disambiguates spurious wakeups from true wakeups that happen once all
    // threads have reached the barrier.
    unsigned int epoch;

    PyMUTEX_T lock;
    PyCOND_T cond;
} _PyGCBarrier;

static inline void
_PyGCBarrier_Init(_PyGCBarrier *barrier, unsigned int capacity)
{
    assert(capacity > 0);
    barrier->capacity = capacity;
    barrier->num_left = capacity;
    barrier->epoch = 0;
    _PyGC_MUTEX_INIT(&barrier->lock);
    _PyGC_COND_INIT(&barrier->cond);
}

static inline void
_PyGCBarrier_Fini(_PyGCBarrier *barrier)
{
    _PyGC_COND_FINI(&barrier->cond);
    _PyGC_MUTEX_FINI(&barrier->lock);
}

// Block until all participating threads arrive.
static inline void
_PyGCBarrier_Wait(_PyGCBarrier *barrier)
{
    _PyGC_MUTEX_LOCK(&barrier->lock);

    unsigned int current_epoch = barrier->epoch;
    barrier->num_left--;

    if (barrier->num_left == 0) {
        barrier->epoch++;
        barrier->num_left = barrier->capacity;
        _PyGC_COND_BROADCAST(&barrier->cond);
    } else {
        while (barrier->epoch == current_epoch) {
            _PyGC_COND_WAIT(&barrier->cond, &barrier->lock);
        }
    }

    assert(barrier->epoch != current_epoch);
    _PyGC_MUTEX_UNLOCK(&barrier->lock);
}

#ifdef __cplusplus
}
#endif

#endif  // Py_INTERNAL_GC_BARRIER_H
