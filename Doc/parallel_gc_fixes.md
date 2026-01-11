# Parallel GC Bug Fixes - Session Notes

## Overview

This document details two critical bugs fixed in the FTP parallel GC implementation during refactoring.

---

## Bug 1: UNREACHABLE Bit Not Cleared for ALIVE Objects

### Symptom
```
Python/gc_free_threading.c:1101: validate_refcounts: Assertion "!gc_is_unreachable(op)" failed: object should not be marked as unreachable yet
```

Objects (typically function objects) had the UNREACHABLE bit set at the start of a new GC cycle, when no object should have this bit set yet.

### Root Cause

The GC has two marking phases:
1. **gc_mark_alive_from_roots**: Marks objects reachable from known roots (sysdict, builtins, stacks) with the ALIVE bit
2. **deduce_unreachable_heap (update_refs)**: Sets UNREACHABLE bit on all tracked objects, then clears it for objects found reachable via reference counting

The problem: When `gc_set_alive` or `_PyGC_TryMarkAlive` marked an object as ALIVE, they did NOT clear the UNREACHABLE bit. If an object had a leftover UNREACHABLE bit from a previous cycle (e.g., due to a race or edge case), it would persist.

### Fix

Modified `gc_set_alive` (serial) and `_PyGC_TryMarkAlive` (parallel) to also clear the UNREACHABLE bit when marking an object ALIVE:

**gc_free_threading.c:210-215:**
```c
static inline void
gc_set_alive(PyObject *op)
{
    gc_set_bit(op, _PyGC_BITS_ALIVE);
    // Also clear UNREACHABLE to prevent mishandling if it was left set
    // from a previous GC cycle
    gc_clear_bit(op, _PyGC_BITS_UNREACHABLE);
}
```

**pycore_gc_ft_parallel.h:82-96:**
```c
static inline int
_PyGC_TryMarkAlive(PyObject *op)
{
    // Fast path check...
    int marked = _PyGC_TrySetBit(op, _PyGC_BITS_ALIVE);
    if (marked) {
        // Also clear UNREACHABLE to prevent mishandling if it was left set
        _PyGC_AtomicClearBit(op, _PyGC_BITS_UNREACHABLE);
    }
    return marked;
}
```

---

## Bug 2: ob_tid Worklist Corruption During Serial Post-Processing

### Symptom
```
Python/gc_free_threading.c:1124: validate_refcounts: Assertion "op->ob_tid == 0" failed: merged objects should have ob_tid == 0
```

A merged dict object (class `__dict__`) had `ob_tid` pointing to a Node object, which is a worklist next pointer that wasn't cleared.

### Background: ob_tid Usage in GC

In free-threading Python, `ob_tid` serves multiple purposes:
- **Normal operation**: Stores the thread ID of the owning thread
- **Merged objects**: Must be 0 (refcount is centralised, no owning thread)
- **GC worklists**: Used as the "next" pointer in linked lists of unreachable objects

The worklist operations:
- `worklist_push(op)`: Sets `op->ob_tid = worklist_head`, then `worklist_head = op`
- `worklist_pop()`: Returns head, sets `op->ob_tid = 0`
- `worklist_remove()`: Unlinks from list, sets `op->ob_tid = 0`

### Root Cause

The parallel `scan_heap` phase processes objects in parallel:
1. `par_merge_refcount(op, 1)` - Merges refcount, sets `ob_tid = 0`
2. `scan_worklist_push(op)` - Adds to worklist, sets `ob_tid = next`

After parallel scan completes, a **serial post-processing loop** walks the worklist to call `disable_deferred_refcounting` on each object (lines 1684-1698):

```c
PyObject *op = (PyObject *)state->unreachable.head;
while (op != NULL) {
    PyObject *next = (PyObject *)op->ob_tid;  // Save next before call
    disable_deferred_refcounting(op);          // BUG: may clear ob_tid!
    op = next;
}
```

The problem: `disable_deferred_refcounting` calls `merge_refcount` for objects with deferred refcount:

```c
static void disable_deferred_refcounting(PyObject *op)
{
    if (_PyObject_HasDeferredRefcount(op)) {
        op->ob_gc_bits &= ~_PyGC_BITS_DEFERRED;
        op->ob_ref_shared -= _Py_REF_SHARED(_Py_REF_DEFERRED, 0);
        merge_refcount(op, 0);  // <-- Sets ob_tid = 0!
        ...
    }
}
```

`merge_refcount` unconditionally sets `ob_tid = 0`:
```c
static Py_ssize_t merge_refcount(PyObject *op, Py_ssize_t extra)
{
    ...
    op->ob_tid = 0;  // <-- Corrupts worklist link!
    op->ob_ref_local = 0;
    op->ob_ref_shared = _Py_REF_SHARED(refcount, _Py_REF_MERGED);
    return refcount;
}
```

### The Corruption Sequence

1. After parallel scan: Worklist is `A -> B -> C -> NULL` (using `ob_tid` as next pointers)
2. Serial post-processing walks the list with `next = op->ob_tid` saved first
3. If `A` has deferred refcount, `disable_deferred_refcounting(A)` sets `A->ob_tid = 0`
4. Worklist structure is now broken: `head -> A` but `A->ob_tid = 0` instead of `B`
5. Later iteration (e.g., `finalize_garbage`) using `WORKSTACK_FOR_EACH` stops at `A`
6. `B` and `C` are never processed, never popped
7. `B->ob_tid` still points to `C` at end of GC cycle
8. Next GC cycle: `validate_refcounts` sees merged object `B` with non-zero `ob_tid`

### Fix

Created `disable_deferred_refcounting_on_worklist()` that does NOT call `merge_refcount`:

```c
// Version of disable_deferred_refcounting for objects already on worklists.
// These objects have already been merged by par_merge_refcount, and their
// ob_tid is used as the worklist next pointer. We must NOT clear ob_tid.
static void
disable_deferred_refcounting_on_worklist(PyObject *op)
{
    if (_PyObject_HasDeferredRefcount(op)) {
        op->ob_gc_bits &= ~_PyGC_BITS_DEFERRED;
        op->ob_ref_shared -= _Py_REF_SHARED(_Py_REF_DEFERRED, 0);
        // NOTE: Do NOT call merge_refcount here! The object is already merged
        // (by par_merge_refcount in parallel scan_heap), and ob_tid is used
        // as the worklist next pointer. Calling merge_refcount would set
        // ob_tid = 0, corrupting the worklist.

        _PyObject_DisablePerThreadRefcounting(op);
    }

    // Handle generators and frame objects (same as original)
    if (PyGen_CheckExact(op) || PyCoro_CheckExact(op) || PyAsyncGen_CheckExact(op)) {
        frame_disable_deferred_refcounting(&((PyGenObject *)op)->gi_iframe);
    }
    else if (PyFrame_Check(op)) {
        frame_disable_deferred_refcounting(((PyFrameObject *)op)->f_frame);
    }
}
```

Updated the serial post-processing to use the worklist-safe version:
```c
// Use the worklist-safe version that preserves ob_tid (worklist link).
PyObject *op = (PyObject *)state->unreachable.head;
while (op != NULL) {
    PyObject *next = (PyObject *)op->ob_tid;
    disable_deferred_refcounting_on_worklist(op);
    op = next;
}
```

### Why This Works

The objects on the worklist were already merged by `par_merge_refcount` during parallel scan. At that point:
- `ob_tid` was set to 0
- `ob_ref_shared` has the `_Py_REF_MERGED` flag
- `ob_ref_local` is 0

Then `scan_worklist_push` set `ob_tid` to the next pointer.

The worklist-safe version:
- Clears the `_PyGC_BITS_DEFERRED` flag from `ob_gc_bits`
- Adjusts `ob_ref_shared` for the deferred count
- Does NOT touch `ob_tid` (preserving the worklist link)
- Does NOT call `merge_refcount` (the object is already merged)

---

## Verification

Both fixes verified with:
- `gc-stress` test: 10 simple tests + stress test pass
- `gc-ftp` test: 30/30 tests pass with no TSan warnings
- `benchmark-quick`: All benchmarks complete without assertion failures
- Large heap benchmarks: 500K-5M objects, 1-8 workers, 80% survivor rate

---

## Key Lessons

1. **ob_tid has multiple uses**: Thread ID, merged indicator (0), and GC worklist link. Code that modifies `ob_tid` must be aware of the current context.

2. **Worklist invariants**: Objects on worklists have `ob_tid` as the next pointer. Any function that might set `ob_tid = 0` must not be called on worklist objects without care.

3. **Parallel/serial interaction**: When parallel code creates data structures (worklists) that serial code later processes, both must agree on invariants. The parallel code set up objects as merged with `ob_tid` as next pointer; the serial code must preserve that.

4. **GC bit flags can persist**: The UNREACHABLE bit should be cleared at the end of each cycle, but defensive code should clear it when setting ALIVE to handle edge cases.
