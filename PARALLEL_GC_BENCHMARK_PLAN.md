# Parallel GC Performance Analysis Plan

## Overview

Comprehensive performance analysis of parallel garbage collection across both:
- **GIL-based parallel GC** (`gc_parallel.c`) - for standard CPython builds
- **Free-threading parallel GC** (`gc_free_threading_parallel.c`) - for Py_GIL_DISABLED builds

## Dimensions of Analysis

### Dimension 1: GC Worker Count
- **Values**: 1, 2, 4, 8 workers
- **Purpose**: Measure scaling efficiency with thread count
- **Expected**: Speedup should increase with workers up to a point, then plateau/decrease due to contention

### Dimension 2: Heap Type (Object Graph Structure)
| Type | Description | Parallelizability |
|------|-------------|-------------------|
| `chain` | Single linked list (worst case) | Very low - sequential dependencies |
| `tree` | Balanced tree (branching) | Medium - limited by tree depth |
| `wide_tree` | Many roots × shallow children | High - independent subtrees |
| `graph` | Complex with cross-references | Medium - some stealing opportunities |
| `layered` | Generations with inter-layer refs | Medium - layer boundaries |
| `independent` | Many isolated objects | Highest - fully parallel |

### Dimension 3: Heap Size (Object Count)
- **Values**: 10k, 50k, 100k, 500k, 1M objects
- **Purpose**: Find crossover point where parallel overhead is amortized
- **Expected**: Small heaps favor serial; large heaps favor parallel

### Dimension 4: Survivor Ratio
- **Values**: 0%, 25%, 50%, 75%, 100% survival
- **Purpose**: Measure impact of GC workload type
- **0%**: All objects are garbage (marking is minimal)
- **100%**: All objects survive (full marking required)
- **Expected**: Higher survival ratio → more marking work → better parallel speedup

### Dimension 5: Multi-threaded Object Creation [Free-threading only]
- **Values**: 1, 2, 4, 8 threads creating objects
- **Purpose**: Test real-world scenario where objects are created across threads
- **Expected**: Objects may be distributed across per-thread heaps, affecting page distribution

## Benchmark Implementation Plan

### Phase 1: Benchmark Framework (Day 1)

1. **Create benchmark harness** (`Lib/test/gc_benchmark.py`)
   - Command-line interface for all dimensions
   - JSON output for analysis
   - Automatic detection of GIL vs free-threading build
   - Warmup runs and statistical sampling (median of N runs)

2. **Implement heap generators**
   - `create_chain(n)` - linked list of n objects
   - `create_tree(n, branching)` - balanced tree
   - `create_wide_tree(roots, children_per)` - many independent subtrees
   - `create_graph(n, edge_prob)` - random graph with edge probability
   - `create_layered(layers, objects_per)` - generational layers
   - `create_independent(n)` - isolated objects

3. **Implement survivor control**
   - Create objects with "keep" list and "garbage" list
   - Control ratio by adjusting list sizes

### Phase 2: GIL-based Parallel GC Benchmarks (Day 2)

1. **Build GIL Python with parallel GC**
   ```bash
   ./configure --with-parallel-gc --with-pydebug
   make -j8
   ```

2. **Run benchmark matrix**
   - Workers × Heap type × Heap size × Survivor ratio
   - Store raw results in JSON

3. **Analyze results**
   - Generate speedup heatmaps
   - Identify optimal worker counts per scenario
   - Find minimum heap size for parallel benefit

### Phase 3: Free-threading Parallel GC Benchmarks (Day 3)

1. **Build free-threading Python**
   ```bash
   ./configure --disable-gil --with-pydebug
   make -j8
   ```

2. **Run benchmark matrix**
   - All 5 dimensions (including multi-threaded creation)
   - Compare barrier-based thread pool vs spawn-per-collection

3. **Analyze results**
   - Speedup vs GIL-based version
   - Impact of atomic CAS overhead
   - Multi-threaded creation impact

### Phase 4: Comparative Analysis (Day 4)

1. **Cross-build comparison**
   - GIL vs Free-threading on same workload
   - Identify scenarios where each excels

2. **Visualization**
   - Heatmaps: speedup[workers, heap_size]
   - Line plots: speedup vs heap_size for each heap_type
   - Bar charts: comparing GIL vs free-threading

3. **Write analysis report**
   - Key findings
   - Recommendations for threshold tuning
   - Future optimization opportunities

## Benchmark Script Outline

```python
#!/usr/bin/env python3
"""
Parallel GC Benchmark Suite

Usage:
    python gc_benchmark.py --workers 1,2,4,8 --heap-type all --heap-size 100k,500k
    python gc_benchmark.py --full-matrix --output results.json
"""

import gc
import time
import json
import argparse
import sys
from typing import List, Dict, Any

# Heap type generators
def create_chain(n: int) -> tuple:
    """Create linked list - worst case for parallelism."""
    ...

def create_wide_tree(roots: int, children: int) -> tuple:
    """Create many independent subtrees - best case."""
    ...

# Benchmark runner
def run_benchmark(
    heap_type: str,
    heap_size: int,
    survivor_ratio: float,
    num_workers: int,
    num_runs: int = 5
) -> Dict[str, Any]:
    """Run single benchmark configuration."""
    ...

# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', default='1,2,4')
    parser.add_argument('--heap-type', default='chain,wide_tree')
    parser.add_argument('--heap-size', default='100000')
    parser.add_argument('--survivor-ratio', default='1.0')
    parser.add_argument('--output', default='results.json')
    ...
```

## Expected Deliverables

1. `Lib/test/gc_benchmark.py` - Benchmark harness
2. `Lib/test/gc_heap_generators.py` - Heap creation utilities
3. `results/gil_parallel_gc_results.json` - Raw GIL benchmark data
4. `results/ft_parallel_gc_results.json` - Raw free-threading data
5. `PARALLEL_GC_PERFORMANCE_ANALYSIS.md` - Final analysis report

## Success Criteria

- [ ] Benchmark covers all 5 dimensions
- [ ] At least 3 runs per configuration for statistical validity
- [ ] Identify configurations where parallel > serial
- [ ] Quantify overhead of atomic operations (free-threading)
- [ ] Recommend default threshold values
- [ ] Document when to enable/disable parallel GC

## Schedule

| Day | Task |
|-----|------|
| 1 | Implement benchmark framework + heap generators |
| 2 | Run GIL-based parallel GC benchmarks |
| 3 | Run free-threading parallel GC benchmarks |
| 4 | Analysis, visualization, final report |

## Notes

- All benchmarks should be run with `gc.disable()` to control collection timing
- Use `time.perf_counter()` for high-resolution timing
- Run benchmarks on dedicated machine (no background processes)
- Document CPU model, core count, memory for reproducibility

---

## Phase 5: FTP Parallel GC Optimization (Current Work)

Initial benchmarks revealed that FTP parallel GC is **slower** than serial due to
per-object atomic operation and memory fence overhead. This phase addresses those issues.

### Completed Optimizations

1. **Replace CAS loops with fetch-or/fetch-and** ✅
   - `_PyGC_TrySetBit()` now uses single `atomic_or` instead of CAS loop
   - `_PyGC_TryClearBit()` now uses single `atomic_and` instead of CAS loop
   - Result: Work distribution improved from CoV 0.86 to 0.08

### Identified Overhead Sources

1. **Work-stealing deque fences** (PRIMARY BOTTLENECK)
   - `_PyWSDeque_Take()` has `fence_seq_cst()` on every pop (~20-100 cycles)
   - `_PyWSDeque_Push()` has `fence_release()` on every push
   - For 100k objects with 5 children each: 600k fence operations!

2. **Atomic marking**
   - `_PyGC_TryMarkAlive()` uses atomic fetch-or per object
   - Always pays atomic RMW cost even for already-marked objects

### Optimization Plan

#### Step 1: Batched Local Work Buffer (NEXT)

**Goal**: Amortize deque fence overhead over batches of 1024 objects.

**Design**:
```c
typedef struct {
    PyObject *buffer[1024];    // Local non-atomic buffer
    size_t head;               // Push index
    size_t tail;               // Pop index
    _PyWSDeque *deque;         // Work-stealing deque for overflow/stealing
} _PyGCLocalWorkQueue;
```

**Work flow**:
1. Push children to local buffer (zero fences)
2. When buffer full (1024 items): flush batch to deque (1 fence for 1024)
3. Pop from local buffer (zero fences)
4. When buffer empty: pull batch from deque (1 fence for batch)
5. When deque empty: steal batch from other workers (steal up to half, max 512)

**Expected benefit**: Reduce fence operations from O(objects) to O(objects/1024)

**Files to modify**:
- `Include/internal/pycore_gc_ft_parallel.h` - Add `_PyGCLocalWorkQueue` struct
- `Python/gc_free_threading_parallel.c` - Update `thread_pool_do_work()` and `propagate_pool_visitproc()`

#### Step 2: Performance Analysis

After implementing batched work:
- Run benchmark suite: `./python Lib/test/gc_benchmark.py --workers 1,4 --heap-size 100k,500k,1M`
- Compare speedup before/after
- Measure work distribution (should remain balanced)
- Profile to verify fence reduction

#### Step 3: Check-First Marking Optimization

**Goal**: Skip atomic RMW for already-marked objects.

**Current code**:
```c
static inline int _PyGC_TryMarkAlive(PyObject *op) {
    return _PyGC_TrySetBit(op, _PyGC_BITS_ALIVE);
}
```

**Optimized code**:
```c
static inline int _PyGC_TryMarkAlive(PyObject *op) {
    // Fast path: relaxed load to check if already marked (very cheap)
    if (_Py_atomic_load_uint8_relaxed(&op->ob_gc_bits) & _PyGC_BITS_ALIVE) {
        return 0;  // Already marked, skip atomic RMW entirely
    }
    // Slow path: atomic set for unmarked objects
    return _PyGC_TrySetBit(op, _PyGC_BITS_ALIVE);
}
```

**Expected benefit**:
- Type objects, builtins, common strings get marked once
- Every subsequent reference just does cheap relaxed load
- Likely significant win for realistic workloads with shared objects

**Files to modify**:
- `Include/internal/pycore_gc_ft_parallel.h` - Update `_PyGC_TryMarkAlive()`

#### Step 4: Final Performance Analysis

- Run full benchmark matrix
- Compare GIL vs FTP parallel GC performance
- Document recommendations for when to enable parallel GC
- Update threshold defaults if needed

### Success Criteria for Phase 5

- [ ] Batched work reduces fence operations by ~1000x
- [ ] FTP parallel GC achieves speedup > 1.0 for large heaps (500k+ objects)
- [ ] Check-first optimization reduces atomic RMW count significantly
- [ ] Work distribution remains balanced (CoV < 0.2)
- [ ] No correctness regressions (all GC tests pass)
