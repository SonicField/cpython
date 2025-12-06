# Parallel GC Benchmark Results - Initial Findings

## Summary

This document summarizes initial benchmark results comparing the GIL-based and Free-Threading (FTP) parallel garbage collector implementations.

## Test Environment

- **Build Types Tested**: GIL with --with-parallel-gc, FTP with --disable-gil
- **Benchmark Tool**: `Lib/test/gc_benchmark.py` (unified for both builds)
- **Heap Types**: chain, tree, wide_tree, graph, layered, independent
- **Metrics**: GC time (ms), speedup vs serial (1 worker)

## Key Results

### GIL-based Parallel GC

| Configuration | Serial (ms) | Parallel 4W (ms) | Speedup |
|--------------|-------------|------------------|---------|
| independent_500k_s80 | 242 | 175 | **1.29x** |
| independent_1M_s80 | 642 | 458 | **1.36x** |
| wide_tree_500k_s80 | 299 | 223 | **1.38x** |
| wide_tree_1M_s80 | 748 | 662 | **1.15x** |
| chain_50k_s50 | 13 | 14 | 0.92x |

**Conclusions for GIL build:**
- Parallel GC shows 1.15x-1.38x speedup for large heaps (500k+ objects)
- Small heaps (50k) show overhead, not speedup
- Best performance with parallelizable structures (independent, wide_tree)
- Chain structures (sequential) do not benefit

### Free-Threading (FTP) Parallel GC

| Configuration | Serial (ms) | Parallel 4W (ms) | Speedup |
|--------------|-------------|------------------|---------|
| independent_500k_s80 | 313 | 453 | **0.71x** |
| independent_1M_s80 | 655 | 659 | **1.01x** |
| wide_tree_500k_s80 | 263 | 404 | **0.65x** |
| chain_50k_s50 | 21 | 29 | 0.73x |

**Conclusions for FTP build:**
- Parallel GC generally SLOWER than serial even for large heaps
- Best case: 1.01x (break-even) at 1M objects
- Atomic CAS overhead not amortized by parallelism gains

### Per-Worker Work Distribution (FTP)

Test with 100k objects, 4 workers:
```
Per-worker distribution: [56083, 32688, 7915, 9228]
Mean per worker: 26,478
Imbalance (CoV): 0.86
```

Worker 0 does 53% of the work, indicating poor work distribution.

## Analysis

### Why is FTP Parallel GC slower?

1. **Atomic CAS Overhead**
   - Every object marking requires atomic compare-and-swap
   - `_PyGC_TryMarkAlive()` uses CAS loop for each object
   - This per-object overhead is significant

2. **Work Distribution Imbalance**
   - Root-based distribution leads to uneven work
   - One worker often gets most of the reachable objects
   - Work-stealing helps but doesn't fully compensate

3. **Thread Coordination Overhead**
   - Barrier synchronization adds overhead
   - Thread pool coordination costs

### Why does GIL Parallel GC work better?

1. **No Atomic CAS Required**
   - Under GIL, only one thread executes Python at a time
   - Marking can use non-atomic operations
   - Significantly lower per-object overhead

2. **Better Work Distribution**
   - Static slicing based on GC list position
   - Temporal locality preserved
   - More balanced work across workers

## Recommendations

### For GIL-based Builds
- Enable parallel GC for heaps > 500k objects
- Use 4 workers as default (diminishing returns beyond)
- Default threshold: 10,000-50,000 roots

### For Free-Threading Builds
- **Parallel GC currently NOT recommended**
- Overhead exceeds parallelism benefits
- Needs optimization of atomic operations
- Consider page-based work distribution instead of root propagation

## Future Work

1. **Reduce FTP Atomic Overhead**
   - Batch marking to reduce CAS frequency
   - Epoch-based marking instead of per-object CAS

2. **Improve Work Distribution**
   - Page-based distribution shows better balance
   - Consider hybrid approach for FTP

3. **Profile-Guided Threshold**
   - Dynamic threshold based on heap characteristics
   - Consider graph structure when deciding serial vs parallel

## Files

- Benchmark script: `Lib/test/gc_benchmark.py`
- Existing benchmarks: `~/claude_docs/parallel_gc/benchmarks/`
- Plan document: `PARALLEL_GC_BENCHMARK_PLAN.md`
