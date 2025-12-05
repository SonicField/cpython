# Designing a Parallel Garbage Collector for CPython

**TL;DR**: We ported the parallel GC from CinderX to upstream CPython, adapting it for the GIL-based runtime. This post explains the design, where we diverged from CinderX, optimizations we added, and lessons learned from CPython's incremental collector. This implementation targets GIL Python; Free Threaded Python will require a different approach next year.

---

## Scope and Constraints

### GIL Python Only

This parallel GC implementation is designed for **GIL-based CPython** (the default build). It is mutually exclusive with Free Threaded Python (`--disable-gil`), which uses a completely different GC architecture.

Why? The GIL provides a critical simplification: we know that application code is paused during collection. The parallel workers only race with each other, not with the application. Free Threaded Python removes this guarantee, requiring a fundamentally different concurrent collector design.

```
Build configuration:
  --with-parallel-gc      Enable parallel GC (GIL builds only)
  --disable-gil           Free Threaded Python (incompatible with parallel GC)
```

### Next Year: Free Threaded Python

The Free Threaded Python project (PEP 703) removes the GIL, enabling true multi-threaded execution. This creates new GC challenges:

- Application threads can mutate the object graph during collection
- Reference counts change concurrently
- Write barriers or snapshot-at-beginning semantics may be required

We plan to explore parallel/concurrent GC for Free Threaded Python in 2025, likely requiring techniques like:
- Concurrent marking with SATB barriers
- Incremental update protocols
- Hazard pointers or epoch-based reclamation

---

## Design Heritage: CinderX

This implementation is ported from [CinderX](https://github.com/facebookincubator/cinder), Meta's performance-oriented fork of CPython. CinderX introduced parallel GC to accelerate Instagram's Python workloads.

### Core Algorithm

The algorithm follows the standard parallel mark-sweep pattern:

1. **Root identification**: Find objects with external references (gc_refs > 0)
2. **Work distribution**: Assign roots to worker threads
3. **Parallel marking**: Workers traverse object graphs, marking reachable objects
4. **Work stealing**: Idle workers steal from busy workers' queues
5. **Sweep**: Collect unmarked objects (still serial)

### Chase-Lev Work-Stealing Deque

The heart of the design is the Chase-Lev work-stealing deque, as described in:

- ["Dynamic Circular Work-Stealing Deque"](https://dl.acm.org/doi/10.1145/1073970.1073974) (Chase & Lev, 2005)
- ["Correct and Efficient Work-Stealing for Weak Memory Models"](https://dl.acm.org/doi/10.1145/2442516.2442524) (Lê et al., 2013)

Each worker has a local deque:
- **Owner operations** (push/pop from bottom): Lock-free, LIFO for cache locality
- **Steal operations** (pop from top): Lock-free CAS, FIFO for fairness

```c
// Worker pushes discovered objects to local deque
_PyWSDeque_Push(&worker->deque, child_object);

// Worker pops from own deque (fast path)
PyObject *obj = _PyWSDeque_Take(&worker->deque);

// When local deque empty, steal from others
PyObject *stolen = _PyWSDeque_Steal(&other_worker->deque);
```

### Atomic Marking

Objects are marked using atomic compare-and-swap on the `_gc_prev` field:

```c
static inline int
gc_try_mark_reachable_atomic(PyGC_Head *gc)
{
    uintptr_t prev = _Py_atomic_load_uintptr_relaxed(&gc->_gc_prev);

    if (!(prev & _PyGC_PREV_MASK_COLLECTING)) {
        return 0;  // Already marked by another worker
    }

    uintptr_t new_prev = prev & ~_PyGC_PREV_MASK_COLLECTING;

    // CAS: Only succeed if value hasn't changed
    return _Py_atomic_compare_exchange_uintptr(&gc->_gc_prev, &prev, new_prev);
}
```

This ensures each object is processed exactly once, even with concurrent workers.

---

## Divergence from CinderX

While the core algorithm matches CinderX, we made several adaptations for upstream CPython:

### 1. Atomic Abstractions

CinderX uses raw C11 atomics. We use CPython's portable atomic wrappers:

| CinderX | CPython |
|---------|---------|
| `atomic_load_explicit(..., relaxed)` | `_Py_atomic_load_uintptr_relaxed()` |
| `atomic_compare_exchange_weak()` | `_Py_atomic_compare_exchange_uintptr()` |
| `atomic_thread_fence()` | `_Py_atomic_fence_seq_cst()` |

This ensures portability across platforms where C11 atomics may not be available or may behave differently.

### 2. Barrier Synchronization

CinderX uses a custom barrier implementation. We implemented barriers using CPython's `PyMUTEX_T` and `PyCOND_T` primitives for consistency with the rest of the runtime:

```c
typedef struct {
    unsigned int num_left;
    unsigned int capacity;
    unsigned int epoch;  // Disambiguates spurious wakeups
    PyMUTEX_T lock;
    PyCOND_T cond;
} _PyGCBarrier;
```

### 3. Thread Lifecycle

CinderX manages threads differently due to Instagram's deployment model. Our implementation:

- Creates worker threads when `gc.enable_parallel(N)` is called
- Destroys threads when `gc.disable_parallel()` is called
- Workers wait at a barrier between collections (no busy-waiting)

This matches the explicit control pattern expected for manual GC in AI/ML workloads.

### 4. Build System Integration

Parallel GC is an optional feature enabled via configure:

```bash
./configure --with-parallel-gc
```

The implementation is guarded by `#ifdef Py_PARALLEL_GC` throughout.

---

## Optimizations Beyond CinderX

Through profiling and experimentation, we added several optimizations not present in the original CinderX implementation:

### 1. Thread-Local Memory Pools

**Problem**: Each collection was calling `calloc()` to allocate deque backing arrays. With multiple workers, this created allocator contention and cache pollution.

**Solution**: Pre-allocate 2MB memory pools per worker at initialization time:

```c
// At gc.enable_parallel() time:
size_t pool_bytes = sizeof(_PyWSArray) + sizeof(uintptr_t) * 262144;
worker->local_pool = PyMem_RawCalloc(1, pool_bytes);

// During collection, use pre-allocated buffer:
_PyWSDeque_InitWithBuffer(&worker->deque, worker->local_pool, pool_bytes, size);
```

This eliminates allocation overhead during the hot path. The pool handles up to 262K objects per worker before falling back to dynamic allocation.

### 2. Static Slicing

**Problem**: Original round-robin root distribution scattered related objects across workers, hurting cache locality.

**Solution**: Assign contiguous slices of the GC list to each worker:

```c
// Each worker gets objects [i*N/W, (i+1)*N/W)
size_t objs_per_slice = total_objects / num_workers;

// Worker 0: objects 0 to objs_per_slice-1
// Worker 1: objects objs_per_slice to 2*objs_per_slice-1
// etc.
```

Objects allocated together tend to reference each other. Static slicing keeps these on the same worker, reducing work stealing overhead.

### 3. Threshold Tuning

**Problem**: Original threshold (too low) triggered parallel GC on small heaps where overhead dominated.

**Solution**: Based on systematic benchmarking, we raised the threshold:

```c
// Parallel GC only activates for large heaps
const size_t MIN_TOTAL_OBJECTS = 500000;  // 500K objects minimum
const size_t MIN_OBJECTS_PER_WORKER = 1000;
```

Below 500K objects, the incremental collector is faster due to lower overhead.

---

## Lessons from the Incremental Collector

CPython 3.13 introduced an incremental GC that spreads collection work across multiple pauses. Studying this implementation informed several design decisions:

### 1. Adaptive Thresholds

The incremental collector uses `work_to_do` budgeting to limit work per pause. Similarly, our threshold tuning ensures parallel GC only activates when the heap is large enough to benefit.

### 2. Tuple Untracking

The incremental collector untracks tuples that only reference immutable objects. We inherit this optimization—untracked tuples are never added to worker deques.

### 3. Serial Remains Faster for Small Heaps

The most important lesson: for typical Python workloads with modest heaps, the incremental collector's simple serial loop outperforms parallel GC. Parallelism has overhead (atomics, barriers, work stealing) that only pays off at scale.

This is why parallel GC targets a specific niche: **large heaps with manual GC**, particularly AI/ML workloads.

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                        gc.collect()                              │
├─────────────────────────────────────────────────────────────────┤
│  Check: heap_size >= 500K objects?                              │
│    No  → Use incremental collector (serial)                    │
│    Yes → Use parallel collector                                 │
├─────────────────────────────────────────────────────────────────┤
│                     Parallel Collector                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Worker 0   │  │  Worker 1   │  │  Worker N   │             │
│  │  ┌───────┐  │  │  ┌───────┐  │  │  ┌───────┐  │             │
│  │  │ Deque │  │  │  │ Deque │  │  │  │ Deque │  │             │
│  │  └───────┘  │  │  └───────┘  │  │  └───────┘  │             │
│  │     ↑↓      │  │     ↑↓      │  │     ↑↓      │             │
│  │  Push/Pop   │  │  Push/Pop   │  │  Push/Pop   │             │
│  │  (local)    │  │  (local)    │  │  (local)    │             │
│  └──────┼──────┘  └──────┼──────┘  └──────┼──────┘             │
│         └────────────────┼────────────────┘                     │
│                    Work Stealing                                │
│              (CAS-based, lock-free)                            │
├─────────────────────────────────────────────────────────────────┤
│  Synchronization: Barriers at start/end of marking phase       │
├─────────────────────────────────────────────────────────────────┤
│  Marking: Atomic CAS on _gc_prev COLLECTING flag               │
└─────────────────────────────────────────────────────────────────┘
```

---

## API

```python
import gc

# Enable parallel GC with N worker threads
gc.enable_parallel(num_workers=4)

# Disable parallel GC (destroys worker threads)
gc.disable_parallel()

# Check configuration
gc.get_parallel_config()
# {'enabled': True, 'num_workers': 4, 'min_gen': 0}

# Get statistics from last collection
gc.get_parallel_stats()
# {'roots_found': 1234, 'collections_succeeded': 1,
#  'workers': [{'objects_marked': 5000, 'steal_successes': 42}, ...]}
```

---

## Performance Characteristics

Based on systematic experimentation:

| Scenario | Parallel vs Incremental |
|----------|------------------------|
| < 100K objects | 0.5-0.9x (slower) |
| 100K-300K objects | 0.9-1.1x (break-even) |
| 300K-500K objects | 1.1-1.3x (slight win) |
| > 500K objects | **1.3-1.8x (clear win)** |

Best results: **1.81x speedup** on neural network-like graph structures with layered connectivity.

---

## Future Work

### 2025: Free Threaded Python

The removal of the GIL creates new challenges and opportunities:

1. **Concurrent marking**: Mark objects while application runs
2. **Write barriers**: Track mutations during collection
3. **Incremental updates**: Handle reference count changes
4. **Memory ordering**: Ensure visibility across threads

We'll explore designs from other language runtimes (Go, Java G1/ZGC, .NET) for inspiration.

### Other Improvements

- **Parallel sweep**: Currently serial; could parallelize for large heaps
- **NUMA awareness**: Pin workers to cores for better memory locality
- **Dynamic worker count**: Adjust based on heap size and CPU availability

---

## Conclusion

Parallel GC for CPython is a targeted optimization for large-heap workloads, particularly AI/ML applications with manual GC timing. By porting and adapting CinderX's design, adding thread-local pools and static slicing, and carefully tuning thresholds based on empirical data, we achieve 1.3-1.8x speedups on heaps with 500K+ objects.

This implementation is specific to GIL Python. Free Threaded Python will require a fundamentally different concurrent collector, which we plan to explore in 2025.
