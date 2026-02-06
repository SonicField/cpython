# Abstract Phase Reporting Design

## Problem Statement

The GIL and FTP (Free-Threading) parallel garbage collectors report timing using different phase names:

**GIL Build Phases:**
| Phase | Description |
|-------|-------------|
| `update_refs_ns` | Copy refcounts to gc_refs, build split vector |
| `mark_alive_ns` | Pre-mark reachable from interpreter roots |
| `subtract_refs_ns` | Decrement gc_refs for internal references |
| `mark_ns` | Work-stealing parallel mark to find unreachable |
| `cleanup_ns` | Finalization, weakrefs, deallocation (combined) |
| `total_ns` | Sum of all phases |

**FTP Build Phases:**
- STW barriers: `stw0_ns`, `stw1_ns`, `stw2_ns`, `stw3_ns`
- Mark phases: `update_refs_ns`, `mark_heap_ns`, `scan_heap_ns`
- Resurrection: `disable_deferred_ns`, `find_weakrefs_ns`, `objs_decref_ns`, `weakref_callbacks_ns`, `finalize_ns`, `resurrection_ns`
- Cleanup: `freelists_ns`, `clear_weakrefs_ns`, `cleanup_ns`

Without abstraction, benchmarks must hard-code collector-specific phase names, causing incorrect metrics when running on the wrong build type.

## Design Goals

1. **Unified abstraction**: Provide comparable phases across both collectors
2. **No information loss**: Keep detailed phases available for debugging
3. **Benchmark compatibility**: Single code path for STW pause calculation
4. **C-side implementation**: Abstract phases computed in the collector, not Python code

## Abstract Phase Categories

Both collectors export these abstract phases alongside their detailed phases:

| Phase | What It Measures | Definition |
|-------|-----------------|------------|
| `scan_mark_ns` | Core graph traversal to identify reachable/unreachable objects | GIL: update_refs + mark_alive + subtract_refs + mark; FTP: stw0 through scan_heap |
| `finalization_ns` | Weakref callbacks, finaliser handling, resurrection | GIL: 0 (not separately tracked); FTP: disable_deferred through resurrection |
| `dealloc_ns` | Final deallocation and cleanup | GIL: cleanup_ns; FTP: freelists through cleanup |
| `stw_pause_ns` | Total time threads are stopped (cannot make progress) | GIL: update_refs + cleanup (serial portions); FTP: stw0 + stw2 |
| `total_ns` | Complete collection time | Sum of all phases |

## Implementation

Abstract phases are computed in the C code and exported alongside detailed phases in `phase_timing`.

### GIL Build (gc_parallel.c)

```c
// Abstract phases - provide a common interface across GIL and FTP collectors.
int64_t scan_mark = update_refs_ns + mark_alive_ns + subtract_refs_ns + mark_ns;
int64_t finalization = 0;  // Not separately tracked in GIL build
int64_t dealloc = cleanup_ns;
int64_t stw_pause = update_refs_ns + cleanup_ns;  // Serial phases only

ADD_TIMING("scan_mark_ns", scan_mark);
ADD_TIMING("finalization_ns", finalization);
ADD_TIMING("dealloc_ns", dealloc);
ADD_TIMING("stw_pause_ns", stw_pause);
```

### FTP Build (gc_free_threading_parallel.c)

```c
// Abstract phases - provide a common interface across GIL and FTP collectors.
int64_t scan_mark = stw0_ns + merge_refs_ns + delayed_frees_ns +
                    mark_alive_ns + bucket_assign_ns +
                    update_refs_ns + mark_heap_ns + scan_heap_ns;
int64_t finalization = disable_deferred_ns + find_weakrefs_ns +
                       stw1_ns + objs_decref_ns +
                       weakref_callbacks_ns + finalize_ns +
                       stw2_ns + resurrection_ns;
int64_t dealloc = freelists_ns + clear_weakrefs_ns + stw3_ns + cleanup_ns;
int64_t stw_pause = stw0_ns + stw2_ns;  // Main STW barriers

ADD_TIMING("scan_mark_ns", scan_mark);
ADD_TIMING("finalization_ns", finalization);
ADD_TIMING("dealloc_ns", dealloc);
ADD_TIMING("stw_pause_ns", stw_pause);
```

## STW Pause Definition

The key metric for parallel GC is STW pause reduction:

- **GIL Build**: Time when Python threads cannot make progress.
  - `update_refs_ns` (serial, with GIL)
  - `cleanup_ns` (serial, with GIL)
  - The mark phases run with GIL but workers do parallel work, so Python threads are paused but GC work progresses.

- **FTP Build**: Time during Stop-The-World barriers when all threads are frozen.
  - STW0: Initial barrier for merge_refs and mark
  - STW2: Barrier for resurrection handling

The abstract `stw_pause_ns` captures the portions where threads are truly blocked, enabling comparison between serial and parallel collection.

## Benchmark Usage

```python
# In PauseTracker.gc_callback:
if self.parallel_enabled:
    stats = get_parallel_stats()
    if 'phase_timing' in stats:
        pt = stats['phase_timing']
        # Use the abstract stw_pause_ns phase (works for both GIL and FTP builds)
        stw_ns = pt.get('stw_pause_ns', 0)
        self.stw_pauses_ms.append(stw_ns / 1e6)
```

## Verification

The implementation is correct if:
1. GIL build reports non-zero `stw_pause_ns` in benchmarks
2. Parallel mode shows lower `stw_pause_ns` than serial mode
3. `total_ns` matches regardless of build type
4. Detailed phases still available for collector-specific debugging

## Future Work

1. **Split GIL cleanup_ns**: Separate finalization from deallocation to report `finalization_ns` accurately
2. **Per-phase parallelism metrics**: Report how much of each phase ran in parallel
3. **Worker utilisation**: Track time workers spent idle vs working
