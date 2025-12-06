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
