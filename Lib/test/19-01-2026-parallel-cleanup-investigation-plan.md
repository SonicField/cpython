# Parallel Cleanup Investigation Plan

## Problem Statement

The `cleanup_workers` feature (Phase 4) is functionally correct but shows inconsistent performance:
- Sometimes parallel cleanup is faster than serial
- Sometimes it's slower
- The cleanup phase time (~15-32ms) remains roughly constant regardless of worker count
- Throughput benchmarks show lower overall STW overhead but higher max pauses

**Hypothesis:** The cleanup phase is contention-bound, not CPU-bound. The parallelisation overhead exceeds the benefit.

## Investigation Goals

1. Identify where time is spent during parallel cleanup
2. Identify contention points (locks, atomics, cache lines)
3. Understand mimalloc's behaviour under parallel free
4. Develop informed optimisation strategies

## Phase 1: Profiling with perf

### 1.1 CPU Profile

Capture where CPU cycles are spent during parallel cleanup:

```bash
# Build with frame pointers for better stacks
./configure --disable-gil --with-lto CFLAGS="-O3 -fno-omit-frame-pointer" && make -j

# Run benchmark under perf
perf record -g --call-graph dwarf ./python Lib/test/gc_benchmark.py \
    --workers 8 --cleanup-workers 4 --heap-type wide_tree \
    --heap-size 500k --survivor-ratio 0.0 --iterations 10

# Analyse
perf report --hierarchy --sort dso,symbol
```

**Questions to answer:**
- What percentage of time is in `mi_free()` vs Python's cleanup code?
- Is time spent in spinlocks or atomics?
- Are there unexpected hotspots?

### 1.2 Lock Contention Profile

```bash
perf record -e sched:sched_switch,sched:sched_wakeup -g ./python ...
perf lock record ./python ...
perf lock report
```

**Questions to answer:**
- Are workers spending time waiting for locks?
- Which locks are contended?

### 1.3 Cache Miss Analysis

```bash
perf stat -e cache-misses,cache-references,L1-dcache-load-misses ./python ...
```

**Questions to answer:**
- Is cache thrashing a factor?
- Does the pointer-sorting help or hurt cache behaviour?

## Phase 2: eBPF Deep Dive

### 2.1 Custom Probes on Cleanup Functions

Create probes to measure time spent in specific functions:

```python
# cleanup_probe.py (bcc script)
from bcc import BPF

program = """
#include <uapi/linux/ptrace.h>

BPF_HASH(start, u32);
BPF_HISTOGRAM(mi_free_us);
BPF_HISTOGRAM(cleanup_work_us);

int trace_mi_free_entry(struct pt_regs *ctx) {
    u32 pid = bpf_get_current_pid_tgid();
    u64 ts = bpf_ktime_get_ns();
    start.update(&pid, &ts);
    return 0;
}

int trace_mi_free_return(struct pt_regs *ctx) {
    u32 pid = bpf_get_current_pid_tgid();
    u64 *tsp = start.lookup(&pid);
    if (tsp) {
        u64 delta = (bpf_ktime_get_ns() - *tsp) / 1000;
        mi_free_us.increment(bpf_log2l(delta));
        start.delete(&pid);
    }
    return 0;
}
"""
```

### 2.2 Worker Thread Analysis

Track which worker does how much work and how long they wait:

```python
# Track per-worker statistics
# - Objects processed per worker
# - Time spent processing vs waiting
# - Atomic operation counts
```

### 2.3 Memory Access Patterns

Use `perf mem` or eBPF to understand memory access patterns:
- Are workers accessing the same cache lines?
- Is there false sharing in the work descriptor?

## Phase 3: Mimalloc Investigation

### 3.1 Understand Mimalloc's Thread Model

Key questions:
- How does mimalloc handle frees from non-owning threads?
- What's the overhead of cross-thread frees?
- Does mimalloc batch frees or process them immediately?

### 3.2 Test Mimalloc Isolation

Create a micro-benchmark that isolates mimalloc behaviour:

```c
// test_mi_free_parallel.c
// Allocate objects on thread A, free on threads B,C,D
// Measure throughput vs serial free
```

This will tell us if the contention is in mimalloc or in our coordination code.

## Phase 4: Optimisation Strategies

Based on investigation findings, potential strategies include:

### 4.1 Chunking Strategy
- Current: Contiguous chunks by address
- Alternative: Interleaved to distribute cross-thread frees
- Alternative: Per-page grouping (mimalloc pages, not OS pages)

### 4.2 Work Stealing
- Instead of static partitioning, use work stealing
- Workers that finish early steal from others
- May reduce tail latency

### 4.3 Batched Frees
- Collect objects to free in thread-local buffers
- Flush to mimalloc in batches
- May amortise lock overhead

### 4.4 Delayed Cleanup
- Queue objects for cleanup
- Process in background thread(s)
- Completely decouple from GC cycle

### 4.5 Mimalloc Page Awareness
- Group objects by their mimalloc page
- Free all objects from same page together
- Reduces cross-page lock contention

## Execution Plan

| Step | Activity | Output |
|------|----------|--------|
| 1 | perf CPU profile | Hotspot identification |
| 2 | perf lock analysis | Contention points |
| 3 | perf cache analysis | Memory behaviour |
| 4 | eBPF mi_free timing | Per-call distribution |
| 5 | eBPF worker analysis | Load balance insight |
| 6 | Mimalloc micro-benchmark | Isolate allocator overhead |
| 7 | Synthesise findings | Root cause identification |
| 8 | Prototype optimisations | Performance validation |

## Success Criteria

1. Parallel cleanup with N workers should be faster than serial
2. Speedup should scale (imperfectly) with worker count
3. No increase in max pause times
4. Consistent behaviour across heap types

## Notes

- Build must include debug symbols for profiling: `-g`
- Frame pointers needed for stack traces: `-fno-omit-frame-pointer`
- May need to disable some optimisations for clearer profiles
- eBPF requires root or CAP_BPF capability
