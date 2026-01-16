# Async Cleanup Implementation Plan

## Goal

Replace parallel cleanup (which is slower due to FTP ref counting) with async cleanup:
- gc.collect() returns after mark phase
- delete_garbage runs in single background worker
- `collecting` flag prevents new GC until cleanup completes

## Cleanup Tasks

### 1. Remove ephemeral files from git
- `15-01-2026-ftp-parallel-cleanup-plan.md` - delete from git

### 2. Remove parallel cleanup code

**Files to modify:**

- `Include/internal/pycore_gc_ft_parallel.h`:
  - Remove `_PyGC_WORK_FINALIZE` and `_PyGC_WORK_DELETE` from enum
  - Remove finalize/delete fields from `_PyGCWorkDescriptor`
  - Remove `_PyGC_ParallelFinalizeWithPool` and `_PyGC_ParallelDeleteWithPool` declarations

- `Include/internal/pycore_interp_structs.h`:
  - Remove `parallel_cleanup_enabled` field

- `Python/gc_free_threading_parallel.c`:
  - Remove `finalize_pool_work()` function
  - Remove `delete_pool_work()` function
  - Remove cases from `thread_pool_do_work()` switch
  - Remove `_PyGC_ParallelFinalizeWithPool()` function
  - Remove `_PyGC_ParallelDeleteWithPool()` function

- `Python/gc_free_threading.c`:
  - Remove `finalize_garbage_parallel()` function
  - Remove `delete_garbage_parallel()` function
  - Remove parallel cleanup branches in gc_collect_internal()

- `Modules/gcmodule.c`:
  - Remove `parallel_cleanup_enabled` setting
  - Remove `parallel_cleanup` from gc.set_parallel_config()

## Implementation Tasks

### 3. Add async cleanup infrastructure

**Add to `_gc_runtime_state` in pycore_interp_structs.h:**
```c
PyObject **async_cleanup_objects;      /* Objects pending async cleanup */
Py_ssize_t async_cleanup_count;        /* Number of objects */
```

### 4. Add async cleanup work type

**In pycore_gc_ft_parallel.h:**
```c
_PyGC_WORK_ASYNC_CLEANUP,  /* Single-threaded async cleanup */
```

### 5. Implement async cleanup worker

**In gc_free_threading_parallel.c:**
```c
static void
async_cleanup_work(_PyGCThreadPool *pool, int worker_id)
{
    // Only worker 0 does cleanup
    if (worker_id != 0) return;

    // Process all objects
    for (Py_ssize_t i = 0; i < count; i++) {
        PyObject *op = objects[i];
        gc_clear_unreachable(op);
        inquiry clear = Py_TYPE(op)->tp_clear;
        if (clear) clear(op);
        Py_DECREF(op);
    }

    // Clear collecting flag
    _Py_atomic_store_int(&gcstate->collecting, 0);
}
```

### 6. Modify gc_collect_internal

Instead of calling delete_garbage:
```c
if (parallel_gc_enabled) {
    // Queue async cleanup
    queue_async_cleanup(state);
    // Don't clear collecting - background will do it
} else {
    delete_garbage(state);
}
```

### 7. Update gc_collect_main

When parallel GC enabled:
- Don't set `collecting = 0` at end
- Background worker sets it when cleanup completes

## Testing

1. Basic functionality: gc.collect() works
2. Verify gc.collect() returns quickly (mark phase time only)
3. Verify memory is freed (eventually)
4. Verify new GC blocked while cleanup running
5. Run test suite

## Notes

- This file is ephemeral - do NOT commit to git
- Delete after implementation complete
