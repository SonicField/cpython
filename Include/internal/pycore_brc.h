#ifndef Py_INTERNAL_BRC_H
#define Py_INTERNAL_BRC_H

#include <stdint.h>
#include "pycore_llist.h"           // struct llist_node
#include "pycore_object_stack.h"    // _PyObjectStack

#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

#ifdef Py_GIL_DISABLED

// Enable sharded BRC for reduced contention on cross-thread decrefs.
// When enabled, each bucket has multiple shards indexed by the decrefing
// thread's ID, reducing mutex contention by a factor of _Py_BRC_NUM_SHARDS.
#define Py_BRC_SHARDED 1

// Enable fast decref path using atomic ADD when object is already queued/merged.
// When disabled, always uses CAS loop for all shared decrefs.
// This optimisation avoids CAS retry loops for the common case.
#define Py_BRC_FAST_DECREF 1

// Prime number to avoid correlations with memory addresses.
#define _Py_BRC_NUM_BUCKETS 257

#ifdef Py_BRC_SHARDED
// Prime number for shard count to avoid correlations with thread IDs.
#define _Py_BRC_NUM_SHARDS 11

// Per-shard mutex within a bucket
struct _brc_shard {
    PyMutex mutex;
};

// Hash table bucket with sharding
struct _brc_bucket {
    // Array of shards, each with its own mutex.
    // Shard index = decrefing_thread_id % _Py_BRC_NUM_SHARDS
    struct _brc_shard shards[_Py_BRC_NUM_SHARDS];

    // Linked list of _PyThreadStateImpl objects hashed to this bucket.
    // Protected by ALL shard mutexes (must hold all for modification).
    struct llist_node root;
};

#else  /* !Py_BRC_SHARDED */

// Hash table bucket (original non-sharded design)
struct _brc_bucket {
    // Mutex protects both the bucket and thread state queues in this bucket.
    PyMutex mutex;

    // Linked list of _PyThreadStateImpl objects hashed to this bucket.
    struct llist_node root;
};

#endif  /* Py_BRC_SHARDED */

// Per-interpreter biased reference counting state
struct _brc_state {
    // Hash table of thread states by thread-id. Thread states within a bucket
    // are chained using a doubly-linked list.
    struct _brc_bucket table[_Py_BRC_NUM_BUCKETS];
};

#ifdef Py_BRC_SHARDED

// Per-thread biased reference counting state (sharded)
struct _brc_thread_state {
    // Linked-list of thread states per hash bucket
    struct llist_node bucket_node;

    // Thread-id as determined by _PyThread_Id()
    uintptr_t tid;

    // Objects with refcounts to be merged, one queue per shard.
    // Queue[i] is protected by bucket->shards[i].mutex
    _PyObjectStack queues[_Py_BRC_NUM_SHARDS];

    // Bitmap tracking which queues are non-empty.
    // Bit i set iff queues[i] is non-empty.
    uint16_t non_empty_shards;

    // Local stack of objects to be merged (not accessed by other threads)
    _PyObjectStack local_objects_to_merge;
};

#else  /* !Py_BRC_SHARDED */

// Per-thread biased reference counting state (original)
struct _brc_thread_state {
    // Linked-list of thread states per hash bucket
    struct llist_node bucket_node;

    // Thread-id as determined by _PyThread_Id()
    uintptr_t tid;

    // Objects with refcounts to be merged (protected by bucket mutex)
    _PyObjectStack objects_to_merge;

    // Local stack of objects to be merged (not accessed by other threads)
    _PyObjectStack local_objects_to_merge;
};

#endif  /* Py_BRC_SHARDED */

// Initialize/finalize the per-thread biased reference counting state
void _Py_brc_init_thread(PyThreadState *tstate);
void _Py_brc_remove_thread(PyThreadState *tstate);

// Initialize per-interpreter state
void _Py_brc_init_state(PyInterpreterState *interp);

void _Py_brc_after_fork(PyInterpreterState *interp);

// Enqueues an object to be merged by it's owning thread (tid). This
// steals a reference to the object.
void _Py_brc_queue_object(PyObject *ob);

// Merge the refcounts of queued objects for the current thread.
void _Py_brc_merge_refcounts(PyThreadState *tstate);

#endif /* Py_GIL_DISABLED */

#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_BRC_H */
