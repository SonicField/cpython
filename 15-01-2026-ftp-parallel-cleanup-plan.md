# FTP Parallel GC: Parallel Cleanup Phase Plan

## Progress

### Completed

**Phase 1: parallel_cleanup config option** (commit b4205ccaec8)
- Added `parallel_cleanup` field to `GCConfig` struct
- Added `gc.set_parallel_config(parallel_cleanup=True/False)` API
- Config controls whether parallel finalize/delete is used

**Phase 2: parallel finalize_garbage** (commit 2459024d4ac)
- Implemented using existing thread pool infrastructure
- Added `_PyGC_WORK_FINALIZE` work type to `pycore_gc_ft_parallel.h`
- Added `finalize_pool_work()` worker function with batch-based distribution (BATCH_SIZE=64)
- Added `_PyGC_ParallelFinalizeWithPool()` API function
- Uses `_PyGC_TrySetBit()` for atomic FINALIZED flag setting
- Falls back to serial for < 1000 objects

**Benchmark additions** (commit ca14a4173e5)
- Added `--finalizers` flag to gc_benchmark.py
- Tests worst-case (all objects have `__del__`) vs common case (no finalizers)

**Phase 3: parallel delete_garbage** (commit c72b791cc14)
- Implemented using existing thread pool infrastructure
- Added `_PyGC_WORK_DELETE` work type to `pycore_gc_ft_parallel.h`
- Added `delete_pool_work()` worker function with batch-based distribution (BATCH_SIZE=64)
- Added `_PyGC_ParallelDeleteWithPool()` API function
- Added `parallel_cleanup_enabled` field to `_gc_runtime_state`
- Uses atomic counter for accurate collected statistics
- Falls back to serial for DEBUG_SAVEALL mode or < 1000 objects

### Performance Notes

With trivial finalizers (worst case):
- Serial finalization is faster due to barrier synchronisation overhead (~9ms)
- Parallel finalize adds overhead when finalizers are trivial (empty `__del__`)

With no finalizers (common case):
- No impact - parallel finalize path is not taken when no tp_finalize exists

With parallel delete_garbage:
- Shows ~10% overhead for simple objects due to barrier synchronization
- Benchmark results (100k-500k objects): 0.72x-0.96x (slower than serial)
- Operations (tp_clear + Py_DECREF) are very fast per-object
- Memory-bound workload doesn't scale well with parallelism

### Analysis

The cleanup phase parallelization shows overhead rather than speedup because:
1. **Barrier synchronization overhead**: ~9ms for start/done barriers
2. **Simple per-object work**: tp_clear and Py_DECREF are very fast (hundreds of ns)
3. **Memory-bound**: Each object access is a cache miss, limiting scaling
4. **Atomic contention**: Even with batching, atomic operations add overhead

Parallel cleanup may benefit workloads with:
- Very large heaps (millions of objects)
- Complex tp_clear implementations
- Objects with significant cleanup work

## Context

With the benchmark fixes, we now measure real garbage collection:
- gc_benchmark.py: 1.47x mean speedup with parallel mark phase
- gc_locality_benchmark.py: 1.43x speedup with real garbage collection (100k objects)

Phase timing from locality benchmark (500k heap, 20% garbage):
- update_refs: 3.71ms
- mark_heap: 0.47ms
- cleanup: 40.74ms
- total: 44.92ms

**The cleanup phase is 90% of total GC time** and is currently serial.

## Current Cleanup Implementation

The cleanup phase in `gc_free_threading.c` consists of:

```
gc_collect_internal()
├── (stop-the-world mark phase)
├── StartTheWorld()                    # Resume other threads
├── call_weakref_callbacks(state)      # Serial, with world running
├── finalize_garbage(state)            # Serial, calls tp_finalize (__del__)
├── StopTheWorld()                     # Pause again
├── handle_resurrected_objects()       # Must be serial
├── clear_weakrefs(state)              # Could parallelise
├── StartTheWorld()                    # Resume again
├── delete_garbage(state)              # Serial, calls tp_clear
└── handle_legacy_finalizers()         # Append to gc.garbage list
```

### finalize_garbage() (lines 1937-1952)

```c
WORKSTACK_FOR_EACH(&state->unreachable, op) {
    if (!_PyGC_FINALIZED(op)) {
        destructor finalize = Py_TYPE(op)->tp_finalize;
        if (finalize != NULL) {
            _PyGC_SET_FINALIZED(op);
            finalize(op);
        }
    }
}
```

Iterates the unreachable worklist and calls `tp_finalize` (Python `__del__`).

### delete_garbage() (lines 1956-2001)

```c
// First: decref pending objects
while ((op = worklist_pop(&state->objs_to_decref)) != NULL) {
    Py_DECREF(op);
}

// Second: clear and decref unreachable objects
while ((op = worklist_pop(&state->unreachable)) != NULL) {
    gc_clear_unreachable(op);
    if (_PyObject_GC_IS_TRACKED(op)) {
        state->collected++;
        inquiry clear = Py_TYPE(op)->tp_clear;
        if (clear != NULL) {
            (void) clear(op);
        }
    }
    Py_DECREF(op);
}
```

## Why Parallel Cleanup is Feasible in FTP

1. **No GIL**: FTP has no Global Interpreter Lock, so multiple threads can execute Python code concurrently
2. **Thread-safe tp_finalize**: In FTP, `__del__` methods are already designed for concurrent execution
3. **Thread-safe tp_clear**: Clearing container references is thread-safe in FTP
4. **World is running**: Both `finalize_garbage` and `delete_garbage` execute with the world running (other threads active)

## Proposed Approach

### Phase 1: Parallel finalize_garbage

**Approach**: Convert worklist iteration to parallel work distribution.

1. Before calling finalize_garbage, convert the unreachable worklist to an array
2. Partition the array among N workers
3. Each worker calls tp_finalize on its partition
4. Wait for all workers to complete

```c
static void
finalize_garbage_parallel(struct collection_state *state, int num_workers)
{
    // 1. Convert worklist to array for random access
    size_t count = workstack_count(&state->unreachable);
    if (count == 0 || num_workers <= 1) {
        finalize_garbage(state);  // Fall back to serial
        return;
    }

    PyObject **objects = PyMem_Malloc(count * sizeof(PyObject *));
    size_t i = 0;
    PyObject *op;
    WORKSTACK_FOR_EACH(&state->unreachable, op) {
        objects[i++] = op;
    }

    // 2. Spawn workers to process partitions
    struct finalize_work {
        PyObject **objects;
        size_t start;
        size_t end;
    };

    // Use existing worker pool from parallel mark phase
    // Each worker iterates its partition and calls tp_finalize

    // 3. Wait for completion

    PyMem_Free(objects);
}
```

**Challenges**:
- _PyGC_SET_FINALIZED needs atomic write (already uses ob_gc_bits which is atomic in FTP)
- tp_finalize can trigger arbitrary Python code - but this is already the case
- Objects might get resurrected - handled by handle_resurrected_objects which runs after

### Phase 2: Parallel delete_garbage

**Approach**: Similar partition-based parallelism.

1. Convert unreachable worklist to array
2. Partition among workers
3. Each worker:
   - Clears the unreachable flag (gc_clear_unreachable)
   - Calls tp_clear
   - Calls Py_DECREF
4. Use atomic increment for state->collected counter

```c
static void
delete_garbage_parallel(struct collection_state *state, int num_workers)
{
    // Similar structure to finalize_garbage_parallel

    // Key differences:
    // - state->collected needs atomic increment
    // - gcstate->garbage list access (for DEBUG_SAVEALL) needs locking
    //   or should be disabled in parallel mode
}
```

**Challenges**:
- `state->collected++` needs `_Py_atomic_add_ssize`
- gc.garbage list append is not thread-safe - could:
  - Skip SAVEALL in parallel mode
  - Use a per-worker collection merged after
  - Use a lock (acceptable since DEBUG_SAVEALL is rare)

### Phase 3: objs_to_decref parallel processing

The first loop in delete_garbage processes `objs_to_decref` worklist.
This is purely Py_DECREF calls and could also be parallelised.

## Shared Infrastructure

Both phases need:

1. **Array conversion**: Convert worklist to array for partitioning
2. **Worker pool reuse**: Use the same worker pool as parallel mark
3. **Barrier synchronisation**: Wait for all workers before proceeding

The existing parallel GC infrastructure in gc_free_threading_parallel.c provides:
- Worker spawning via pthread_create
- Barrier via parallel_gc_barrier()
- Work stealing queue (though not needed for simple partitioning)

## Implementation Steps

### Step 1: Add array conversion utility

```c
// Convert worklist to array, return count
static size_t
worklist_to_array(struct workstack *ws, PyObject ***out_array);
```

### Step 2: Implement parallel finalize_garbage

1. Add finalize_garbage_parallel() function
2. Modify gc_collect_internal to call parallel version when enabled
3. Test with gc_benchmark.py and gc_locality_benchmark.py
4. Verify object resurrection still works correctly

### Step 3: Implement parallel delete_garbage

1. Add delete_garbage_parallel() function
2. Handle atomic collected counter
3. Handle DEBUG_SAVEALL case (lock or disable)
4. Test thoroughly

### Step 4: Add cleanup phase timing

1. Add cleanup_finalize_ns and cleanup_clear_ns timing
2. Report in gc.get_parallel_stats()
3. Measure speedup from parallelisation

## Expected Impact

With cleanup taking 90% of GC time:
- If we achieve 4x speedup on cleanup with 8 workers: cleanup goes from 40ms to 10ms
- Total GC time goes from 45ms to ~15ms (3x overall speedup)
- More realistic: 2x speedup on cleanup (cache effects, sync overhead)
- Total would be ~25ms (1.8x overall speedup, up from current 1.43x)

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| tp_finalize ordering dependencies | Document that __del__ order is undefined |
| Resurrection race conditions | handle_resurrected_objects runs in stop-the-world |
| Thread safety bugs in tp_clear | Extensive testing, fallback to serial |
| Performance regression on small heaps | Threshold: only parallelise above N objects |

## Testing Plan

1. Run gc_benchmark.py with all heap types
2. Run gc_locality_benchmark.py with various sizes
3. Run Python test suite with parallel cleanup enabled
4. Stress test with concurrent threads doing allocation
5. Test resurrection scenarios (objects resurrected in __del__)

## Open Questions

1. Should we use work-stealing or simple partitioning? (Recommend: partitioning first)
2. What's the minimum object count to enable parallel cleanup? (Suggest: 10000)
3. Should cleanup workers be separate from mark workers? (Recommend: reuse pool)
