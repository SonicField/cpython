# Parallel deduce_unreachable_heap Design

## Current Serial Implementation

`deduce_unreachable_heap()` has 4 phases that walk all heap pages:

```c
// Phase 1: Compute ref differences (external refs = refcount - internal refs)
gc_visit_heaps(interp, &update_refs, &state->base);

// Phase 2: Visit thread stacks for deferred references
gc_visit_thread_stacks(interp, state);

// Phase 3: Transitively mark reachable objects
gc_visit_heaps(interp, &mark_heap_visitor, &state->base);

// Phase 4: Identify unreachable, push to worklist
gc_visit_heaps(interp, &scan_heap_visitor, &state->base);
```

All 4 phases are currently serial, walking ALL heap pages sequentially.

## Analysis of Each Phase

### Phase 1: update_refs

**What it does:**
- For each object, compute `gc_refs = refcount - internal_references`
- Uses `ob_tid` to store gc_refs (repurposed during GC)
- Calls `tp_traverse(op, visit_decref, NULL)` which decrements gc_refs on referenced objects

**Cross-object dependencies:**
- `visit_decref` modifies OTHER objects' `ob_tid` (decrements gc_refs)
- Multiple workers processing objects A and B that both reference C will race on C's ob_tid

**Solution: Atomic gc_refs operations**
```c
static inline void gc_decref_atomic(PyObject *op) {
    _Py_atomic_add_ssize(&op->ob_tid, -1);
}

static inline void gc_add_refs_atomic(PyObject *op, Py_ssize_t refs) {
    _Py_atomic_add_ssize(&op->ob_tid, refs);
}
```

### Phase 2: gc_visit_thread_stacks

**What it does:**
- Walks all thread states and their stacks
- For deferred references, calls `gc_add_refs(obj, 1)`

**Parallelization:**
- Could be parallel across threads, but likely small work
- Keep serial for simplicity (thread count << object count)

### Phase 3: mark_heap_visitor

**What it does:**
- For each object with gc_refs > 0, transitively mark as reachable
- Clears `_PyGC_BITS_UNREACHABLE` flag on reachable objects
- Uses `tp_traverse` to propagate reachability

**Cross-object dependencies:**
- `visit_clear_unreachable` modifies OTHER objects' gc bits
- Multiple workers may try to clear same object's unreachable bit

**Solution: Already have atomic bit operations**
- `gc_clear_unreachable` → use `_PyGC_AtomicClearBit()`
- Work-stealing for transitive marking (like mark_alive)

### Phase 4: scan_heap_visitor

**What it does:**
- For each unreachable object, push to worklist
- For each reachable object, restore ob_tid and clear alive bit
- Updates counters (long_lived_total)

**Cross-object dependencies:**
- `worklist_push` is not thread-safe (linked list)
- Counter updates need to be atomic

**Solution: Per-worker worklists + merge**
```c
typedef struct {
    struct worklist unreachable;  // Per-worker unreachable list
    struct worklist legacy;       // Per-worker legacy finalizer list
    size_t long_lived_count;      // Per-worker counter
} _PyGCScanWorkerState;
```

## Implementation Plan

### Step 1: Atomic gc_refs operations

Add to `gc_free_threading.c`:
```c
static inline void gc_decref_atomic(PyObject *op) {
    _Py_atomic_add_ssize(&op->ob_tid, -1);
}

static inline void gc_add_refs_atomic(PyObject *op, Py_ssize_t refs) {
    _Py_atomic_add_ssize(&op->ob_tid, refs);
}

static inline void gc_maybe_init_refs_atomic(PyObject *op) {
    // Use CAS to atomically set unreachable bit and init ob_tid
    if (_PyGC_TrySetBit(op, _PyGC_BITS_UNREACHABLE)) {
        // We set the bit, now init ob_tid
        _Py_atomic_store_ssize(&op->ob_tid, 0);
    }
}
```

### Step 2: Parallel update_refs

Create `update_refs_parallel()`:
1. Distribute pages across workers (like page bucket assignment)
2. Each worker calls `update_refs` on its pages
3. Use atomic gc_refs operations for cross-object refs

```c
static int update_refs_parallel(PyInterpreterState *interp,
                                struct collection_state *state,
                                int num_workers);
```

### Step 3: Parallel mark_heap_visitor

Similar to parallel mark_alive:
1. Initial phase: each worker scans its pages for roots (gc_refs > 0)
2. Propagation phase: work-stealing for transitive marking
3. Use atomic bit operations for unreachable flag

### Step 4: Parallel scan_heap_visitor

1. Each worker has local worklists
2. Workers scan their assigned pages
3. Push unreachable objects to local lists
4. Merge all worker lists at end

### Step 5: Integration

Modify `deduce_unreachable_heap()`:
```c
static int deduce_unreachable_heap(...) {
    int workers = _PyGC_ShouldUseParallel(interp, estimated_objects);

    if (workers > 1) {
        // Parallel path
        update_refs_parallel(interp, state, workers);
        gc_visit_thread_stacks(interp, state);  // Keep serial
        mark_heap_parallel(interp, state, workers);
        scan_heap_parallel(interp, state, workers);
    } else {
        // Existing serial path
        gc_visit_heaps(interp, &update_refs, &state->base);
        gc_visit_thread_stacks(interp, state);
        gc_visit_heaps(interp, &mark_heap_visitor, &state->base);
        gc_visit_heaps(interp, &scan_heap_visitor, &state->base);
    }
}
```

## Expected Benefits

1. **Better scaling** - Currently only mark_alive is parallel; this parallelizes 3 more phases
2. **Higher sweet spot** - Should see benefits with more workers on larger heaps
3. **Reduced STW time** - All phases run with world stopped; parallelizing reduces pause time

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Atomic overhead on gc_refs | Only affects update_refs; rest uses existing atomics |
| Work imbalance across pages | Page assignment already handles this |
| Correctness regressions | Extensive testing with gc tests and stress tests |
| Memory ordering bugs | Use proven patterns from mark_alive |

## Estimated Complexity

- Atomic gc_refs: Small (~20 lines)
- Parallel update_refs: Medium (~100 lines)
- Parallel mark_heap: Medium (~150 lines, similar to mark_alive)
- Parallel scan_heap: Medium (~100 lines)
- Integration: Small (~30 lines)

Total: ~400 lines of new code
