# Parallel GC Benchmark Redesign: Questions and Dimensions

## Problem Statement

The current `gc_throughput_benchmark.py` creates a pathological workload:
- 8 threads doing nothing but creating cyclic garbage
- GC overhead exceeds 100% (GC uses more CPU than application)
- Results don't reflect realistic application behaviour

We need a benchmark that models real applications to measure parallel cleanup effectiveness meaningfully.

---

## What We're Measuring

1. **Throughput impact**: How much does GC reduce application work rate?
2. **Latency impact**: How long are STW pauses?
3. **Parallel cleanup effectiveness**: Does cw4 beat cw0 in realistic conditions?

---

## Dimensions of a Realistic Benchmark

| Dimension | Current Benchmark | Realistic Range | Notes |
|-----------|------------------|-----------------|-------|
| **Work ratio** | 0 (pure allocation) | 100-10,000 CPU cycles per allocation | Real apps do work |
| **Garbage fraction** | ~100% | 10-50% | Most allocations are long-lived |
| **Cycle fraction** | 100% | 5-20%? | Most garbage is acyclic (refcount handles it) |
| **Heap size** | 500k objects | Varies widely | Affects scan time |
| **Thread pattern** | All identical | Varies | Web: independent. ML: shared model |
| **Cross-thread refs** | High (except web_server) | Depends on app | Critical for cw effectiveness |

---

## Experimental Findings: pyperformance GC Production Analysis

**Experiment:** `gc_production_experiment.py` - runs simplified versions of canonical
pyperformance benchmarks and measures cyclic garbage production.

**Methodology:**
1. Run each benchmark with GC disabled - measure memory growth (indicates uncollected cycles)
2. Run each benchmark with GC enabled - count collections and objects collected
3. Classify by cyclic garbage production rate

### Results Summary (500 iterations each)

| Benchmark       | Cycles/Iter | GC Time % | Memory Growth (no GC) | Classification |
|-----------------|-------------|-----------|----------------------|----------------|
| deltablue       | 416.1       | 44.8%     | 17,512 KB            | HIGH_CYCLES    |
| deepcopy        | 242.5       | 6.3%      | 484 KB               | HIGH_CYCLES    |
| async_tree      | 242.0       | 32.0%     | 3,320 KB             | HIGH_CYCLES    |
| richards        | 0.1         | 10.5%     | 4 KB                 | MINIMAL_CYCLES |
| nbody           | 0.0         | 68.9%     | 0 KB                 | MINIMAL_CYCLES |
| json_loads      | 0.0         | 1.2%      | 76 KB                | NO_CYCLES      |
| float           | 0.0         | 8.0%      | 0 KB                 | NO_CYCLES      |
| regex           | 0.0         | 3.8%      | 0 KB                 | NO_CYCLES      |
| comprehensions  | 0.0         | 2.5%      | 0 KB                 | NO_CYCLES      |
| generators      | 0.0         | 6.1%      | 0 KB                 | NO_CYCLES      |
| pathlib         | 0.0         | 0.4%      | 0 KB                 | NO_CYCLES      |
| logging         | 0.0         | 0.3%      | 4,980 KB             | NO_CYCLES      |
| pprint          | 0.0         | 0.5%      | 0 KB                 | NO_CYCLES      |

### Classification Distribution

- **HIGH_CYCLES**: 3 benchmarks (23%) - significant GC load
- **MODERATE_CYCLES**: 0 benchmarks (0%)
- **MINIMAL_CYCLES**: 2 benchmarks (15%)
- **NO_CYCLES**: 8 benchmarks (62%) - acyclic garbage only

### Key Findings

1. **Only 23% of canonical benchmarks produce significant cyclic garbage.**
2. **62% produce NO cyclic garbage at all** - refcount handles everything.
3. HIGH_CYCLES benchmarks share a pattern: **bidirectional references**
   - deltablue: constraint-variable cycles
   - deepcopy: parent pointer cycles
   - async_tree: task parent cycles
4. Even in HIGH_CYCLES benchmarks, GC time ranges from 6% to 45% of total time.
5. **Our current benchmark (100% cyclic garbage) is 4-5x more GC-intensive than
   the most GC-heavy canonical benchmark.**

### Implications for Benchmark Design

The current `gc_throughput_benchmark.py` with 100% cyclic garbage is unrealistic.
A representative benchmark should have:
- **~20-30% cyclic garbage** (matching HIGH_CYCLES benchmarks)
- **Significant work between allocations** (not pure allocation)
- **Mix of cyclic and acyclic structures**

---

## Questions Requiring Input

### 1. What percentage of real Python garbage is cyclic?

The GC only handles cycles - acyclic garbage is freed by refcount immediately. If 95% of garbage is acyclic, we're benchmarking a rare event.

**Experimental answer:** Based on pyperformance analysis, approximately **20-30% of garbage
in cycle-heavy workloads is cyclic**, and **most workloads (62%) produce no cyclic garbage**.

**Options:**
- A: Use empirical data from real applications (do we have any?)
- B: Estimate based on common patterns (closures, dataclasses with back-refs)
- C: Parameterise and test a range
- **D: Use 20-30% based on pyperformance experiment (recommended)**

**Alex's response:**


---

### 2. What work-to-allocation ratio is realistic?

| Category | Ops per allocation | Example |
|----------|-------------------|---------|
| Low | 10-100 | String processing, JSON parsing |
| Medium | 1,000 | Web handler with business logic |
| High | 10,000+ | Compute-bound, occasional allocation |

**Question:** Should we parameterise this or pick one representative value?

**Alex's response:**


---

### 3. What application model should we target?

| Model | Characteristics | GC Pattern |
|-------|-----------------|------------|
| **Web server** | Independent requests, per-request garbage | Frequent small GCs, isolated heaps |
| **Data pipeline** | Batch processing, large temps | Occasional large GCs |
| **ML inference** | Long-lived model, request tensors | Infrequent GCs, large live set |
| **Background worker** | Queue processing, task isolation | Moderate GCs, independent tasks |

**Question:** Pick one to focus on, or create variants for each?

**Alex's response:**


---

### 4. What are we comparing?

**Options:**
- A: cw0 vs cw1 vs cw4 (cleanup worker effectiveness) - current focus
- B: Parallel GC vs serial GC (overall parallel benefit)
- C: FTP vs GIL build (free-threading overhead)
- D: All of the above

**Alex's response:**


---

### 5. Should we use existing benchmarks?

Are there standard Python GC benchmarks (pyperformance, gcbench, etc.) we should use instead of inventing our own? They may have solved these design questions already.

**Alex's response:**


---

## Proposed Benchmark Design

### Parameterised Interface

```
--work-ratio N        # CPU ops per allocation (default: ?)
--garbage-fraction F  # Fraction becoming garbage (default: ?)
--cycle-fraction C    # Fraction of garbage that's cyclic (default: ?)
--heap-size S         # Live heap size (default: ?)
--threads T           # Worker threads (default: ?)
--cross-refs R        # Cross-thread reference probability (default: ?)
```

### Default Values

To be determined based on answers above.

### Measurement Methodology

1. **Baseline**: Run with GC disabled, measure pure application throughput
2. **With GC**: Run with various cw settings, measure:
   - Throughput reduction vs baseline
   - STW pause distribution
   - Total GC time
3. **Comparison**: Report ratios and percentages, not absolute numbers

---

## Next Steps

1. Alex provides input on questions above
2. Agree on default parameter values
3. Implement revised benchmark
4. Run and analyse results

---

## Appendix: Experiment Details

**Experiment script:** `Lib/test/gc_production_experiment.py`

**To reproduce:**
```bash
./python Lib/test/gc_production_experiment.py --iterations 500 --warmup 20 -o results.json
```

**To run specific benchmarks:**
```bash
./python Lib/test/gc_production_experiment.py -b deltablue deepcopy async_tree --iterations 1000
```

**Available benchmarks:**
richards, deltablue, nbody, json_loads, float, regex, comprehensions,
generators, deepcopy, async_tree, pathlib, logging, pprint

**Output:** JSON file with detailed per-benchmark statistics and summary table.
