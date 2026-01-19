# BRC Sharding Proposal: Fixing Cross-Thread Decref Scalability

**Date**: 2026-01-19
**Author**: Dr Alex Turner / Claude
**Status**: Proposal

## Executive Summary

We have identified a fundamental scalability bottleneck in free-threading Python's Biased Reference Counting (BRC) mechanism. Cross-thread decrefs serialize on a per-bucket mutex, causing severe performance degradation when multiple threads decref objects owned by the same thread. This affects GC cleanup and any producer-consumer workload.

We propose a simple, low-risk fix: shard each BRC bucket by the decrefing thread's ID. This reduces contention by a factor of N (the shard count) while preserving the existing safety guarantees.

## Problem Statement

### Background: Biased Reference Counting

Free-threading Python uses biased reference counting (BRC) to handle reference counts efficiently:

- Each object has an owning thread (stored in `ob_tid`)
- The owner uses fast local operations for refcount changes
- Non-owners must queue decrefs for the owner to process later

When Thread A decrefs an object owned by Thread B:
1. Thread A locks B's bucket mutex
2. Thread A finds B's thread state in a linked list
3. Thread A pushes the object to B's merge queue
4. Thread A unlocks

### The Scalability Problem

The bucket mutex creates a serialization point. When many threads decref objects owned by the same thread, they all contend on one mutex.

**Concrete scenario**: GC cleanup with 4 application threads and 8 cleanup workers:
- Application threads created 500k objects
- Objects are distributed across ~4 buckets (one per creator)
- 8 cleanup workers all try to decref objects from the same creators
- All 8 workers serialize on the same 4 mutexes

### Impact

This affects:
1. **GC cleanup**: Parallel cleanup provides no speedup (our investigation's origin)
2. **Producer-consumer patterns**: 1 producer, N consumers → N-way mutex contention
3. **Shared caches**: Cache-filling thread owns objects, many threads access
4. **Thread pools**: Dispatcher creates work items, workers consume

## Investigation

### Initial Observation

Parallel GC cleanup with `cleanup_workers=4` showed no improvement over serial cleanup (`cleanup_workers=0`). Phase timing showed cleanup taking ~65ms regardless of worker count.

### Profiling

`perf` analysis revealed:
- Cleanup takes ~65ms wall-clock time
- Cleanup functions show <1% CPU time

This contradiction (wall-clock time without CPU time) indicates blocking, not computation.

### Root Cause Identification

Tracing the code path:

```
Py_DECREF(op)  [object not owned by current thread]
  → _Py_DecRefShared(op)
    → _Py_brc_queue_object(op)
      → PyMutex_Lock(&bucket->mutex)  ← CONTENTION POINT
      → find_thread_state(bucket, ob_tid)
      → _PyObjectStack_Push(&tstate->brc.objects_to_merge, ob)
      → PyMutex_Unlock(&bucket->mutex)
```

Every cross-thread decref locks the bucket mutex. With K creator threads and N consumer threads where K << N, effective parallelism ≈ K.

### Empirical Confirmation

We created a benchmark (`test_brc_contention.py`) to isolate the effect:

**Test 2: Cross-thread fresh (single producer, N consumers release final refs)**
```
Consumers= 1:    7.37 ms, throughput=13.56 M/s
Consumers= 2:    9.29 ms, throughput=10.77 M/s
Consumers= 4:   13.38 ms, throughput= 7.47 M/s
Consumers= 8:   29.49 ms, throughput= 3.39 M/s
Consumers=16:  100.00 ms, throughput= 1.00 M/s
```

**Result: 13x slower with 16 consumers vs 1!** Negative scaling confirmed.

**Test 3: Pre-merged objects (already through BRC)**
```
Consumers= 1:    3.89 ms, throughput=25.71 M/s
Consumers= 2:    2.14 ms, throughput=46.84 M/s
Consumers= 4:    1.70 ms, throughput=58.95 M/s
Consumers= 8:    2.33 ms, throughput=42.99 M/s
Consumers=16:    3.27 ms, throughput=30.58 M/s
```

**Result: 2.3x faster with 4 consumers.** Positive scaling when mutex is bypassed.

### Why the Mutex Exists

The mutex serves critical safety functions:

1. **Protect linked list traversal**: `find_thread_state` walks a linked list. Concurrent modification (thread exit) could cause corruption.

2. **Synchronize with thread exit**: When a thread exits, it must safely remove itself from the bucket. The mutex ensures producers either see the thread (and queue to it) or don't see it (and merge directly). No in-between state.

3. **Protect queue operations**: The `_PyObjectStack` is not thread-safe.

A completely lock-free design would require RCU or hazard pointers, which Python doesn't have.

## Proposed Solution: Sharded BRC

### Concept

Instead of one mutex per bucket, use N mutexes (shards) per bucket. The shard is selected by the **decrefing thread's** ID, not the owning thread's ID.

```
Current:  ob_tid % 257 → bucket → mutex → queue
Proposed: ob_tid % 257 → bucket → (my_tid % 11) → shard → mutex → queue
```

With 11 shards:
- 48 threads decrefing objects from 1 owner
- Spread across 11 shards
- ~4.4 threads per mutex instead of 48

### Data Structures

```c
#define _Py_BRC_NUM_SHARDS 11  // Prime for good distribution

struct _brc_shard {
    PyMutex mutex;
};

struct _brc_bucket {
    struct _brc_shard shards[_Py_BRC_NUM_SHARDS];
    struct llist_node root;  // Linked list of thread states
};

struct _brc_thread_state {
    struct llist_node bucket_node;
    uintptr_t tid;
    _PyObjectStack queues[_Py_BRC_NUM_SHARDS];  // One queue per shard
    _PyObjectStack local_objects_to_merge;
};
```

### Producer Path (Queue Object)

```c
void _Py_brc_queue_object(PyObject *ob)
{
    PyInterpreterState *interp = _PyInterpreterState_GET();
    uintptr_t ob_tid = _Py_atomic_load_uintptr(&ob->ob_tid);
    uintptr_t my_tid = _Py_ThreadId();

    if (ob_tid == 0) {
        Py_DECREF(ob);
        return;
    }

    struct _brc_bucket *bucket = get_bucket(interp, ob_tid);
    int shard_idx = my_tid % _Py_BRC_NUM_SHARDS;
    struct _brc_shard *shard = &bucket->shards[shard_idx];

    PyMutex_Lock(&shard->mutex);

    _PyThreadStateImpl *tstate = find_thread_state(bucket, ob_tid);
    if (tstate == NULL) {
        // Thread exited, merge directly
        Py_ssize_t refcount = _Py_ExplicitMergeRefcount(ob, -1);
        PyMutex_Unlock(&shard->mutex);
        if (refcount == 0) {
            _Py_Dealloc(ob);
        }
        return;
    }

    if (_PyObjectStack_Push(&tstate->brc.queues[shard_idx], ob) < 0) {
        PyMutex_Unlock(&shard->mutex);
        // OOM fallback - same as current
        _PyEval_StopTheWorld(interp);
        Py_ssize_t refcount = _Py_ExplicitMergeRefcount(ob, -1);
        _PyEval_StartTheWorld(interp);
        if (refcount == 0) {
            _Py_Dealloc(ob);
        }
        return;
    }

    _Py_set_eval_breaker_bit(&tstate->base, _PY_EVAL_EXPLICIT_MERGE_BIT);
    PyMutex_Unlock(&shard->mutex);
}
```

### Consumer Path (Merge Refcounts)

```c
void _Py_brc_merge_refcounts(PyThreadState *tstate)
{
    struct _brc_thread_state *brc = &((_PyThreadStateImpl *)tstate)->brc;
    struct _brc_bucket *bucket = get_bucket(tstate->interp, brc->tid);

    // Merge from all shards
    for (int i = 0; i < _Py_BRC_NUM_SHARDS; i++) {
        PyMutex_Lock(&bucket->shards[i].mutex);
        _PyObjectStack_Merge(&brc->local_objects_to_merge, &brc->queues[i]);
        PyMutex_Unlock(&bucket->shards[i].mutex);
    }

    // Process locally
    merge_queued_objects(&brc->local_objects_to_merge);
}
```

### Thread Shutdown

```c
void _Py_brc_remove_thread(PyThreadState *tstate)
{
    struct _brc_thread_state *brc = &((_PyThreadStateImpl *)tstate)->brc;
    struct _brc_bucket *bucket = get_bucket(tstate->interp, brc->tid);

    bool empty = false;
    while (!empty) {
        merge_queued_objects(&brc->local_objects_to_merge);

        // Lock ALL shards
        for (int i = 0; i < _Py_BRC_NUM_SHARDS; i++) {
            PyMutex_Lock(&bucket->shards[i].mutex);
        }

        // Check if all queues are empty
        empty = true;
        for (int i = 0; i < _Py_BRC_NUM_SHARDS; i++) {
            if (brc->queues[i].head != NULL) {
                empty = false;
                _PyObjectStack_Merge(&brc->local_objects_to_merge,
                                     &brc->queues[i]);
            }
        }

        if (empty) {
            llist_remove(&brc->bucket_node);
        }

        // Unlock ALL shards (reverse order to avoid deadlock patterns)
        for (int i = _Py_BRC_NUM_SHARDS - 1; i >= 0; i--) {
            PyMutex_Unlock(&bucket->shards[i].mutex);
        }
    }

    assert(brc->local_objects_to_merge.head == NULL);
    for (int i = 0; i < _Py_BRC_NUM_SHARDS; i++) {
        assert(brc->queues[i].head == NULL);
    }
}
```

### Thread Registration

```c
void _Py_brc_init_thread(PyThreadState *tstate)
{
    struct _brc_thread_state *brc = &((_PyThreadStateImpl *)tstate)->brc;
    uintptr_t tid = _Py_ThreadId();

    struct _brc_bucket *bucket = get_bucket(tstate->interp, tid);

    // Need to lock all shards to safely add to linked list
    for (int i = 0; i < _Py_BRC_NUM_SHARDS; i++) {
        PyMutex_Lock(&bucket->shards[i].mutex);
    }

    brc->tid = tid;
    llist_insert_tail(&bucket->root, &brc->bucket_node);

    for (int i = _Py_BRC_NUM_SHARDS - 1; i >= 0; i--) {
        PyMutex_Unlock(&bucket->shards[i].mutex);
    }
}
```

## Analysis

### Contention Reduction

With S shards and N threads contending on one bucket:
- Current: N threads on 1 mutex
- Proposed: ~N/S threads per mutex

With S=11 and N=48: ~4.4 threads per mutex instead of 48.

### Memory Overhead

Per thread state:
- Current: 1 queue head pointer (8 bytes)
- Proposed: 11 queue head pointers (88 bytes)

Per bucket:
- Current: 1 mutex
- Proposed: 11 mutexes

Total additional memory per interpreter:
- 257 buckets × 10 additional mutexes × ~40 bytes = ~100KB
- Negligible for any real application

### Performance Impact

**Improved cases:**
- Any workload with concentrated object ownership
- Producer-consumer patterns
- GC cleanup (our original problem)

**Potentially impacted:**
- Thread registration/shutdown: Must lock all shards
- Consumer merge: Must iterate all shards

Both of these are rare operations. Thread registration/shutdown happens once per thread lifetime. Consumer merge happens when the eval breaker fires, which is infrequent compared to queue operations.

### Safety

The design preserves all safety properties:
- Mutex protects linked list traversal
- Mutex synchronizes with thread exit
- Mutex protects queue operations
- No new race conditions introduced

The only change is finer-grained locking, which is a well-understood transformation.

### Why Prime Numbers?

Using 11 (prime) instead of 8 (power of 2) for shard count avoids correlations with memory addresses or thread ID patterns. This matches the existing choice of 257 (prime) for bucket count.

## Verification Plan

### Unit Tests

1. Existing BRC tests must pass
2. Add stress test with many threads and concentrated ownership
3. Add test for thread registration/shutdown under contention

### Benchmarks

1. **BRC contention benchmark** (`test_brc_contention.py`):
   - Cross-thread fresh: Should now scale positively
   - Pre-merged: Should remain unchanged

2. **GC cleanup benchmark**:
   - `cleanup_workers=4` should now show improvement over `cleanup_workers=0`

3. **Normal operation benchmark**:
   - Distributed workloads should show no regression
   - Single-threaded performance should be unchanged

### Expected Results

| Benchmark | Current | With Sharding |
|-----------|---------|---------------|
| Cross-thread (16 consumers) | 1.0 M/s | ~10+ M/s |
| GC cleanup (4 workers) | No speedup | ~2-4x speedup |
| Normal operation | Baseline | No regression |

## Implementation Plan

1. **Phase 1**: Implement sharded BRC
   - Modify data structures
   - Update producer/consumer/shutdown paths
   - Run existing tests

2. **Phase 2**: Benchmark
   - Run BRC contention benchmark
   - Run GC cleanup benchmark
   - Run normal operation benchmarks

3. **Phase 3**: Tune
   - Experiment with different shard counts (7, 11, 13, 17)
   - Measure memory vs performance trade-off

4. **Phase 4**: Upstream discussion
   - Present findings to CPython core team
   - Discuss integration path

## Appendix: Benchmark Code

```python
#!/usr/bin/env python3
"""
BRC Contention Benchmark - test_brc_contention.py
"""

import gc
import threading
import time

gc.disable()

def benchmark_cross_thread_decref(num_objects, num_consumers):
    """
    Single producer creates all objects first, then distributes to consumers.
    All decrefs are cross-thread and should hit BRC.
    """
    objects = [object() for _ in range(num_objects)]
    chunk_size = num_objects // num_consumers
    chunks = [objects[i*chunk_size:(i+1)*chunk_size] for i in range(num_consumers)]
    objects.clear()

    times = []
    def worker(chunk):
        t0 = time.perf_counter()
        chunk.clear()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(chunks[i],))
               for i in range(num_consumers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total_time = time.perf_counter() - t0

    return {
        'total_time': total_time,
        'throughput': num_objects / total_time if total_time > 0 else 0,
    }

def main():
    num_objects = 100_000
    consumer_counts = [1, 2, 4, 8, 16]

    print("BRC Contention Benchmark")
    print("=" * 60)
    print(f"Objects per test: {num_objects:,}")
    print()

    print("Cross-thread fresh (producer creates, consumers decref)")
    print("-" * 60)
    for nc in consumer_counts:
        result = benchmark_cross_thread_decref(num_objects, nc)
        print(f"  Consumers={nc:2d}: {result['total_time']*1000:7.2f} ms, "
              f"throughput={result['throughput']/1e6:.2f} M/s")

if __name__ == "__main__":
    main()
```

## Optimisation: Non-Empty Shard Bitmap

### Motivation

The consumer path iterates all 11 shards, locking each one even if empty. With a bitmap tracking non-empty shards, we can skip empty shards entirely.

### Data Structure

```c
struct _brc_thread_state {
    struct llist_node bucket_node;
    uintptr_t tid;
    _PyObjectStack queues[_Py_BRC_NUM_SHARDS];
    _PyObjectStack local_objects_to_merge;
    _Py_atomic_uint16_t non_empty_shards;  // Bit i set iff queue[i] non-empty
};
```

### Producer Path (Updated)

```c
void _Py_brc_queue_object(PyObject *ob)
{
    // ... get bucket, shard_idx, shard, tstate ...

    PyMutex_Lock(&shard->mutex);

    if (_PyObjectStack_Push(&tstate->brc.queues[shard_idx], ob) < 0) {
        // OOM handling...
        return;
    }

    // Set bit under lock - atomic OR for thread-safety with consumer reads
    _Py_atomic_or_uint16(&tstate->brc.non_empty_shards, (1 << shard_idx));

    _Py_set_eval_breaker_bit(&tstate->base, _PY_EVAL_EXPLICIT_MERGE_BIT);
    PyMutex_Unlock(&shard->mutex);
}
```

### Consumer Path (Updated)

```c
void _Py_brc_merge_refcounts(PyThreadState *tstate)
{
    struct _brc_thread_state *brc = &((_PyThreadStateImpl *)tstate)->brc;
    struct _brc_bucket *bucket = get_bucket(tstate->interp, brc->tid);

    // Atomic read - relaxed is fine, we check again under lock
    uint16_t to_check = _Py_atomic_load_uint16_relaxed(&brc->non_empty_shards);

    if (to_check == 0) {
        return;  // Fast path: nothing to merge
    }

    // Only lock shards with set bits
    for (int i = 0; i < _Py_BRC_NUM_SHARDS; i++) {
        if (!(to_check & (1 << i))) {
            continue;
        }

        PyMutex_Lock(&bucket->shards[i].mutex);

        // Merge queue
        _PyObjectStack_Merge(&brc->local_objects_to_merge, &brc->queues[i]);

        // Clear bit under lock - queue is now empty
        _Py_atomic_and_uint16(&brc->non_empty_shards, ~(1 << i));

        PyMutex_Unlock(&bucket->shards[i].mutex);
    }

    merge_queued_objects(&brc->local_objects_to_merge);
}
```

### Correctness

**Producer-consumer synchronisation:**
- Producer sets bit under shard mutex
- Consumer clears bit under same shard mutex
- No race on individual bits

**Missed-item race:**
1. Consumer reads bitmap = `0b0010`
2. Producer locks shard 5, adds item, sets bit 5 → bitmap = `0b100010`
3. Consumer processes shard 1, clears bit 1 → bitmap = `0b100000`
4. Consumer returns (didn't check shard 5)
5. But producer set eval breaker → consumer runs again
6. Consumer reads bitmap = `0b100000`, processes shard 5

**Invariant:** If queue[i] is non-empty, either bit i is set OR the eval breaker will trigger another pass.

### Cost

| Scenario | Without Bitmap | With Bitmap |
|----------|---------------|-------------|
| Nothing to merge | 11 lock/unlock | 1 atomic read, return |
| 1 shard active | 11 lock/unlock | 1 atomic read + 1 lock/unlock |
| k shards active | 11 lock/unlock | 1 atomic read + k lock/unlock |

For the common case of 1-2 active shards, this eliminates 80-90% of lock operations.

### Thread Shutdown (Updated)

```c
void _Py_brc_remove_thread(PyThreadState *tstate)
{
    // ... existing drain and unlink logic ...

    // Clear bitmap before final unlink
    _Py_atomic_store_uint16(&brc->non_empty_shards, 0);

    // ... unlink from bucket under all shard locks ...
}
```

## References

- `Python/brc.c`: Current BRC implementation
- `Include/internal/pycore_brc.h`: BRC data structures
- `Objects/object.c`: `_Py_ExplicitMergeRefcount` implementation
- `Python/ceval_gil.c`: Consumer invocation via eval breaker
