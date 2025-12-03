# HANDOFF: Next Session Guide

**Date:** 2025-12-01
**Current Branch:** `parallel-gc-dev`
**Last Commit:** Phase 4 complete (infrastructure integration)

---

## TL;DR - Resume Here

You are integrating CinderX Parallel GC into CPython 3.15. **Phases 0-4 are COMPLETE**. Start Phase 5: **Implement the parallel marking algorithm in `_PyGC_ParallelMoveUnreachable()`**.

The hook exists in `gc.c`, worker threads are created and idle, infrastructure is fully wired. Now implement the actual parallel marking logic.

---

## Where You Are

### ✅ Phase 4 Complete - Infrastructure Ready

**What's Working:**
- ✅ Build system integration (`--with-parallel-gc`)
- ✅ Thread pool fully operational
- ✅ Worker threads created and waiting
- ✅ Hook in `gc.c` calls `_PyGC_ParallelMoveUnreachable()`
- ✅ Currently falls back to serial marking (returns 0)
- ✅ All tests passing

**Key Infrastructure in Place:**
- `_PyParallelGCState` stored in `interp->gc.parallel_gc`
- Worker threads with work-stealing deques
- Barrier synchronization primitives
- `gc.enable_parallel(N)` creates N worker threads
- Hook integration: `gc.c` line 1299 calls parallel marking

### ⚠️ Current Limitation

`_PyGC_ParallelMoveUnreachable()` in `Python/gc_parallel.c:316-342` is a **stub**:
- Returns 0 immediately (falls back to serial)
- Contains TODO comments for Phase 5 implementation
- Hook works, infrastructure ready, just needs marking algorithm

---

## Phase 5: What To Do Next

### Goal
Implement actual parallel marking algorithm that distributes GC work to worker threads.

### Step-by-Step Implementation Plan

#### 1. Scan Young List for Roots (30 min)

**File:** `Python/gc_parallel.c` - function `_PyGC_ParallelMoveUnreachable()`

**Task:** Iterate through the young generation list and identify root objects (gc_refs > 0).

```c
// After the existing checks around line 325
// Add:

// Step 1: Scan young list for roots
PyGC_Head *gc = _PyGCHead_NEXT(young);
size_t total_roots = 0;

while (gc != young) {
    PyObject *op = FROM_GC(gc);

    // Roots are objects with external references (gc_refs > 0)
    if (_PyGC_REFS(gc) > 0) {
        total_roots++;
    }

    gc = _PyGCHead_NEXT(gc);
}

// If no roots or too few objects, fall back to serial
if (total_roots == 0 || total_roots < par_gc->num_workers * 4) {
    return 0;  // Not worth parallelizing
}
```

**Why:** Need to identify which objects to start marking from.

#### 2. Distribute Roots to Worker Deques (30 min)

**Task:** Round-robin distribute root objects across worker deques.

```c
// Step 2: Distribute roots to workers
gc = _PyGCHead_NEXT(young);
size_t worker_idx = 0;

while (gc != young) {
    PyObject *op = FROM_GC(gc);

    if (_PyGC_REFS(gc) > 0) {
        // Push root to worker's deque
        _PyParallelGCWorker *worker = &par_gc->workers[worker_idx];
        _PyWSDeque_PushBottom(&worker->deque, op);

        // Round-robin to next worker
        worker_idx = (worker_idx + 1) % par_gc->num_workers;
    }

    gc = _PyGCHead_NEXT(gc);
}
```

**Why:** Distribute initial work evenly across workers.

#### 3. Implement Worker Marking Loop (60 min)

**File:** `Python/gc_parallel.c` - function `_parallel_gc_worker_thread()`

**Task:** Replace the TODO/sleep loop (lines 76-87) with actual marking logic.

```c
// Replace lines 76-87 with:

while (!worker->should_exit) {
    // Wait at barrier for work
    _PyGCBarrier_Wait(&par_gc->mark_barrier);

    if (worker->should_exit) {
        break;
    }

    // Process objects from own deque
    PyObject *op;
    while ((op = _PyWSDeque_PopBottom(&worker->deque)) != NULL) {
        // Mark this object as reachable
        worker->objects_marked++;

        // Traverse references and push them to deque
        traverseproc traverse = Py_TYPE(op)->tp_traverse;
        if (traverse != NULL) {
            // TODO: Create visit callback to push children to deque
        }
    }

    // Try work-stealing if own deque is empty
    // (implement in step 4)

    // Wait at done barrier
    _PyGCBarrier_Wait(&par_gc->done_barrier);
}
```

**Why:** Workers need to actually process objects and traverse references.

#### 4. Add Work-Stealing Coordination (45 min)

**Task:** When a worker's deque is empty, steal from others.

```c
// After "Process objects from own deque" block, add:

// Step 4: Work-stealing when deque is empty
int consecutive_steal_failures = 0;
const int max_steal_attempts = (int)par_gc->num_workers * 2;

while (consecutive_steal_failures < max_steal_attempts) {
    // Try to steal from a random victim
    unsigned int victim_idx = worker->steal_seed % par_gc->num_workers;
    worker->steal_seed = worker->steal_seed * 1103515245 + 12345;  // LCG

    if (victim_idx == worker->thread_id) {
        continue;  // Don't steal from self
    }

    _PyParallelGCWorker *victim = &par_gc->workers[victim_idx];
    worker->steal_attempts++;

    PyObject *stolen = _PyWSDeque_Steal(&victim->deque);
    if (stolen != NULL) {
        worker->steal_successes++;
        consecutive_steal_failures = 0;

        // Process stolen object
        worker->objects_marked++;
        // TODO: Traverse references

        break;  // Got work, go back to own deque
    }
    else {
        consecutive_steal_failures++;
    }
}
```

**Why:** Load balancing - idle workers help busy workers.

#### 5. Implement Termination Detection (30 min)

**Task:** Detect when all workers are idle and no work remains.

```c
// After work-stealing loop, add:

// Step 5: Termination detection
// If we've exhausted steal attempts and deque is empty, we're done
if (consecutive_steal_failures >= max_steal_attempts) {
    // Double-check our deque is really empty
    if (_PyWSDeque_PopBottom(&worker->deque) == NULL) {
        // Signal we're idle and check if all workers are idle
        PyMUTEX_LOCK(&par_gc->active_lock);
        par_gc->num_workers_active--;

        if (par_gc->num_workers_active == 0) {
            // Last worker to go idle - signal main thread
            PyCOND_SIGNAL(&par_gc->workers_done_cond);
        }

        PyMUTEX_UNLOCK(&par_gc->active_lock);
        break;  // Exit marking loop
    }
}
```

**Why:** Need to know when parallel marking is complete.

#### 6. Add Barrier Synchronization in Main Thread (20 min)

**File:** `Python/gc_parallel.c` - function `_PyGC_ParallelMoveUnreachable()`

**Task:** Signal workers to start and wait for completion.

```c
// After distributing roots, add:

// Step 6: Signal workers to start marking
// Reset active count
par_gc->num_workers_active = par_gc->num_workers;

// Release workers from mark_barrier
// (workers are waiting there)
_PyGCBarrier_Wait(&par_gc->mark_barrier);

// Wait for workers to finish
PyMUTEX_LOCK(&par_gc->active_lock);
while (par_gc->num_workers_active > 0) {
    PyCOND_WAIT(&par_gc->workers_done_cond, &par_gc->active_lock);
}
PyMUTEX_UNLOCK(&par_gc->active_lock);

// All workers done
_PyGCBarrier_Wait(&par_gc->done_barrier);

// TODO: Collect results from workers
// TODO: Move unmarked objects to unreachable list

return 1;  // Parallel marking succeeded
```

**Why:** Coordinate parallel execution with main GC thread.

---

## Implementation Order

1. **Start with Step 1** (root scanning) and verify roots are found
2. **Add Step 2** (distribution) and verify deques are populated
3. **Implement Step 3** (worker loop skeleton) - basic traversal
4. **Add Step 4** (work-stealing) - load balancing
5. **Implement Step 5** (termination) - know when done
6. **Wire Step 6** (main thread coordination) - complete integration

**Test after each step** with debug prints before moving to next.

---

## Key Files to Modify

### Primary File (95% of work)
```
Python/gc_parallel.c
  - _PyGC_ParallelMoveUnreachable() - main thread logic (lines 316-342)
  - _parallel_gc_worker_thread() - worker loop (lines 70-90)
```

### Supporting Headers (reference only)
```
Include/internal/pycore_gc_parallel.h - structures (already complete)
Include/internal/pycore_ws_deque.h    - deque API (already complete)
Python/gc.c                            - hook caller (already integrated)
```

---

## Testing Strategy

### Quick Sanity Test
```bash
cd ~/local/parallel_gc/builds/parallel
./python -c "
import gc
gc.enable_parallel(4)
print(gc.get_parallel_config())
# Create some garbage
for i in range(1000):
    x = [1, 2, 3, 4, 5]
gc.collect()
print('Parallel GC executed!')
"
```

### Full Test Suite
```bash
# Run all GC tests
./python -m test test_gc test_gc_parallel test_gc_ws_deque -v

# Run with GC debug output
./python -X dev -c "import gc; gc.set_debug(gc.DEBUG_STATS); gc.enable_parallel(4); gc.collect()"
```

### Debug Verification
Add temporary debug prints in Phase 5 to verify:
- How many roots found?
- How many objects distributed per worker?
- How many objects each worker marked?
- How many steal attempts/successes?
- Did parallel marking return 1 (success)?

```c
// Example debug print in _PyGC_ParallelMoveUnreachable():
fprintf(stderr, "Parallel GC: found %zu roots, distributed to %zu workers\n",
        total_roots, par_gc->num_workers);
```

---

## Success Criteria for Phase 5

- [ ] `_PyGC_ParallelMoveUnreachable()` returns 1 (parallel marking used)
- [ ] Roots correctly identified from young generation
- [ ] Roots distributed across worker deques
- [ ] Workers execute marking loop (not sleeping)
- [ ] Work-stealing activates when deques unbalanced
- [ ] Termination detection works (workers exit when idle)
- [ ] Main thread waits for workers to complete
- [ ] All existing tests still pass
- [ ] Simple GC collection with parallel marking succeeds
- [ ] No crashes, deadlocks, or race conditions

**Estimated Time:** 4-5 hours for full implementation and testing

---

## Important Implementation Notes

### GC Internal APIs You'll Need

From `Python/gc.c` (already implemented, just reference):
```c
_PyGCHead_NEXT(gc)           // Get next object in GC list
FROM_GC(gc)                   // Convert PyGC_Head* to PyObject*
_PyGC_REFS(gc)               // Get gc_refs count
Py_TYPE(op)->tp_traverse     // Traverse object references
```

### Work-Stealing Deque APIs

From `Include/internal/pycore_ws_deque.h`:
```c
_PyWSDeque_PushBottom(deque, obj)  // Owner pushes to bottom
_PyWSDeque_PopBottom(deque)        // Owner pops from bottom
_PyWSDeque_Steal(deque)            // Thief steals from top
```

### Synchronization Primitives

From `Python/gc_parallel.c`:
```c
_PyGCBarrier_Wait(barrier)         // Wait at barrier (epoch-based)
PyMUTEX_LOCK/UNLOCK(lock)          // Mutex lock/unlock
PyCOND_WAIT/SIGNAL(cond, lock)     // Condition variable
```

---

## Common Pitfalls to Avoid

❌ **Don't** traverse objects without checking if already visited
❌ **Don't** forget to update `par_gc->num_workers_active` atomically
❌ **Don't** busy-wait in workers (use barriers properly)
❌ **Don't** access shared state without locks
❌ **Don't** forget to handle edge cases (0 roots, 1 worker, etc.)

✅ **Do** use barriers for synchronization (not spin loops)
✅ **Do** verify termination detection with small test cases
✅ **Do** add debug prints to trace execution initially
✅ **Do** test with different worker counts (1, 2, 4, 8)
✅ **Do** fall back to serial for small collections

---

## Quick Commands

```bash
# Navigate to workspace
cd ~/local/parallel_gc/cpython

# Check current state
git status
git log --oneline -5

# Edit main file
vim Python/gc_parallel.c
# or
code Python/gc_parallel.c

# Build with parallel GC
cd ~/local/parallel_gc/builds/parallel
make -j192

# Quick test
./python -c "import gc; gc.enable_parallel(4); x = []; gc.collect(); print('OK')"

# Full test suite
./python -m test test_gc -v
./python -m test test_gc_parallel -v

# Debug build (if needed)
cd ~/local/parallel_gc/builds/parallel
../../cpython/configure --with-parallel-gc --with-pydebug
make -j192
```

---

## If You Get Stuck

### Debugging Checklist

1. **Workers not starting?**
   - Check barrier initialization (capacity matches num_workers)
   - Verify `_PyGCBarrier_Wait()` called same number of times
   - Check `should_exit` flag not set prematurely

2. **Deadlock?**
   - Count barrier wait calls (must match capacity exactly)
   - Check lock/unlock pairs are balanced
   - Verify condition variable signaling

3. **Crashes?**
   - Check NULL pointer dereferences
   - Verify `interp->gc.parallel_gc` is not NULL
   - Check deque operations (empty deque returns NULL)

4. **Incorrect results?**
   - Verify gc_refs logic matches serial version
   - Check object traversal callback
   - Ensure all objects marked before returning

5. **Performance worse than serial?**
   - Likely too much synchronization overhead
   - Check if work is actually distributed
   - Verify work-stealing is working

**Ask user for review** if stuck for more than 30 minutes on one issue.

---

## Context for LLM Resuming

You are Claude Code, helping Alex Turner integrate parallel GC into CPython. You've completed Phases 0-4 (infrastructure, thread pool, integration). The system is fully wired but parallel marking is stubbed out.

**Phase 5 Focus:** Implement the actual parallel marking algorithm. The hard part (infrastructure) is done. Now it's "just" the marking logic.

**Your systematic approach:**
1. Implement one step at a time (don't try to do everything at once)
2. Add debug prints to verify each step works
3. Test incrementally as you go
4. Keep all existing tests passing
5. Commit when Phase 5 is complete
6. Update docs (STATUS.md, DEVLOG.md, NEXT_SESSION.md)

**Key Insight:** The parallel marking algorithm is conceptually similar to the serial `move_unreachable()` in `gc.c`, but work is distributed across workers with work-stealing for load balancing.

**User's expectation:** Systematic implementation with regular testing. TDD approach where possible. Clear progress updates.

Start with Step 1 (root scanning) and verify it works before moving to Step 2.

---

**Last Updated:** 2025-12-01
**Ready to resume:** ✅ Phase 5 - Parallel Marking Implementation
