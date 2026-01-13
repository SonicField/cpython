#!/usr/bin/env python3
"""
Profile subtract_refs to understand parallelism bottleneck.

This script investigates why parallel subtract_refs with 8 workers achieves
only ~1.7x speedup instead of closer to 8x.

Key measurements:
1. Serial subtract_refs time (when parallel is disabled)
2. Parallel subtract_refs time (with timing breakdown)
3. Per-worker timing and load balance
4. Phase breakdown: update_refs, mark_alive, subtract_refs, mark, cleanup
"""

import gc
import sys
import time
import random
import argparse
import statistics
from typing import List, Dict, Any, Optional

# =============================================================================
# Build Detection
# =============================================================================

def detect_build():
    """Detect whether we're running on GIL or FTP build."""
    try:
        gil_enabled = sys._is_gil_enabled()
        return "ftp" if not gil_enabled else "gil"
    except AttributeError:
        return "gil"

def is_parallel_gc_available():
    """Check if parallel GC is available in this build."""
    try:
        config = gc.get_parallel_config()
        return config.get('available', False)
    except AttributeError:
        return False

BUILD_TYPE = detect_build()
PARALLEL_GC_AVAILABLE = is_parallel_gc_available()

print(f"Build type: {BUILD_TYPE.upper()}")
print(f"Parallel GC available: {PARALLEL_GC_AVAILABLE}")

# =============================================================================
# Node Class
# =============================================================================

class Node:
    """Generic node for building object graphs."""
    __slots__ = ['refs', 'data', '__weakref__']

    def __init__(self):
        self.refs = []
        self.data = None

# =============================================================================
# Heap Generators
# =============================================================================

def create_chain(n: int) -> List[Node]:
    """Create a single linked list chain."""
    nodes = [Node() for _ in range(n)]
    for i in range(n - 1):
        nodes[i].refs.append(nodes[i + 1])
    return nodes

def create_tree(n: int, branching: int = 2) -> List[Node]:
    """Create a balanced tree."""
    import math
    depth = max(1, int(math.log(n * (branching - 1) + 1, branching)))

    nodes = []
    level = [Node()]
    nodes.extend(level)

    for d in range(depth - 1):
        next_level = []
        for parent in level:
            for _ in range(branching):
                if len(nodes) >= n:
                    break
                child = Node()
                parent.refs.append(child)
                next_level.append(child)
                nodes.append(child)
        level = next_level
        if len(nodes) >= n:
            break

    return nodes[:n]

def create_independent(n: int) -> List[Node]:
    """Create isolated objects with no references."""
    return [Node() for _ in range(n)]

HEAP_GENERATORS = {
    'chain': lambda n: create_chain(n),
    'tree': lambda n: create_tree(n, branching=2),
    'independent': lambda n: create_independent(n),
}

# =============================================================================
# Profiling Functions
# =============================================================================

def run_serial_collection(heap_type: str, heap_size: int, survivor_ratio: float = 0.5):
    """Run a single serial collection and return timing."""
    gc.disable()
    try:
        gc.disable_parallel()
    except:
        pass  # Not available

    random.seed(42)
    objects = HEAP_GENERATORS[heap_type](heap_size)

    # Apply survivor ratio
    keep = None
    if survivor_ratio < 1.0:
        num_keep = int(len(objects) * survivor_ratio)
        random.shuffle(objects)
        keep = objects[:num_keep]
    else:
        keep = objects
    objects = None

    # Time the collection
    start = time.perf_counter()
    collected = gc.collect()
    elapsed = time.perf_counter() - start

    del keep
    gc.collect()
    gc.enable()

    return {
        'time_ms': elapsed * 1000,
        'collected': collected,
    }


def run_parallel_collection(heap_type: str, heap_size: int, num_workers: int,
                            survivor_ratio: float = 0.5):
    """Run a single parallel collection and return timing with phase breakdown."""
    gc.disable()
    try:
        gc.enable_parallel(num_workers)
    except RuntimeError:
        # Already enabled, reconfigure by disabling and re-enabling
        gc.disable_parallel()
        gc.enable_parallel(num_workers)

    random.seed(42)
    objects = HEAP_GENERATORS[heap_type](heap_size)

    # Apply survivor ratio
    keep = None
    if survivor_ratio < 1.0:
        num_keep = int(len(objects) * survivor_ratio)
        random.shuffle(objects)
        keep = objects[:num_keep]
    else:
        keep = objects
    objects = None

    # Time the collection
    start = time.perf_counter()
    collected = gc.collect()
    elapsed = time.perf_counter() - start

    # Get stats
    stats = gc.get_parallel_stats()
    phase_timing = stats.get('phase_timing', {})
    workers = stats.get('workers', [])

    del keep
    gc.collect()
    gc.enable()

    return {
        'time_ms': elapsed * 1000,
        'collected': collected,
        'phase_timing': phase_timing,
        'workers': workers,
    }


def profile_subtract_refs(heap_type: str, heap_size: int, num_workers: int,
                          survivor_ratio: float = 0.5, iterations: int = 5,
                          warmup: int = 2):
    """Profile subtract_refs phase in detail."""

    print(f"\n{'='*80}")
    print(f"Profiling: {heap_type}_{heap_size//1000}k with {num_workers} workers, {survivor_ratio*100:.0f}% survivors")
    print(f"{'='*80}")

    # Warmup
    print(f"\nWarming up ({warmup} iterations)...", end="", flush=True)
    for _ in range(warmup):
        run_parallel_collection(heap_type, heap_size, num_workers, survivor_ratio)
    print(" done")

    # Serial runs
    print(f"\nSerial collections ({iterations} iterations):")
    serial_times = []
    for i in range(iterations):
        result = run_serial_collection(heap_type, heap_size, survivor_ratio)
        serial_times.append(result['time_ms'])
        print(f"  Run {i+1}: {result['time_ms']:.1f}ms")

    serial_mean = statistics.mean(serial_times)
    serial_std = statistics.stdev(serial_times) if len(serial_times) > 1 else 0
    print(f"  Mean: {serial_mean:.1f}ms (+/- {serial_std:.1f}ms)")

    # Parallel runs with detailed phase breakdown
    print(f"\nParallel collections ({iterations} iterations):")
    parallel_results = []
    for i in range(iterations):
        result = run_parallel_collection(heap_type, heap_size, num_workers, survivor_ratio)
        parallel_results.append(result)

        pt = result['phase_timing']
        upd = pt.get('update_refs_ns', 0) / 1e6
        alive = pt.get('mark_alive_ns', 0) / 1e6
        sub = pt.get('subtract_refs_ns', 0) / 1e6
        mark = pt.get('mark_ns', 0) / 1e6
        cleanup = pt.get('cleanup_ns', 0) / 1e6
        total = pt.get('total_ns', 0) / 1e6

        print(f"  Run {i+1}: {result['time_ms']:.1f}ms")
        print(f"         Phases: upd={upd:.1f}ms alive={alive:.1f}ms sub={sub:.1f}ms mark={mark:.1f}ms clean={cleanup:.1f}ms")

    parallel_times = [r['time_ms'] for r in parallel_results]
    parallel_mean = statistics.mean(parallel_times)
    parallel_std = statistics.stdev(parallel_times) if len(parallel_times) > 1 else 0

    print(f"  Mean: {parallel_mean:.1f}ms (+/- {parallel_std:.1f}ms)")

    # Speedup
    speedup = serial_mean / parallel_mean if parallel_mean > 0 else 0
    print(f"\n*** Speedup: {speedup:.2f}x ({serial_mean:.1f}ms / {parallel_mean:.1f}ms) ***")

    # Phase breakdown analysis
    print(f"\nPhase Analysis (mean across {iterations} runs):")

    def mean_phase(key):
        values = [r['phase_timing'].get(key, 0) for r in parallel_results]
        return statistics.mean(values) / 1e6  # ns to ms

    upd_mean = mean_phase('update_refs_ns')
    alive_mean = mean_phase('mark_alive_ns')
    sub_mean = mean_phase('subtract_refs_ns')
    mark_mean = mean_phase('mark_ns')
    cleanup_mean = mean_phase('cleanup_ns')
    total_mean = mean_phase('total_ns')

    print(f"  update_refs:  {upd_mean:>8.1f}ms ({upd_mean/total_mean*100:>5.1f}%)")
    print(f"  mark_alive:   {alive_mean:>8.1f}ms ({alive_mean/total_mean*100:>5.1f}%)")
    print(f"  subtract_refs:{sub_mean:>8.1f}ms ({sub_mean/total_mean*100:>5.1f}%)")
    print(f"  mark:         {mark_mean:>8.1f}ms ({mark_mean/total_mean*100:>5.1f}%)")
    print(f"  cleanup:      {cleanup_mean:>8.1f}ms ({cleanup_mean/total_mean*100:>5.1f}%)")
    print(f"  total:        {total_mean:>8.1f}ms")

    # Per-worker analysis (from last run)
    if parallel_results and parallel_results[-1]['workers']:
        print(f"\nPer-Worker Analysis (last run):")
        workers = parallel_results[-1]['workers']
        print(f"  {'Worker':>8} | {'Time(ms)':>10} | {'Objects':>10} | {'Traversals':>12}")
        print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")

        worker_times = []
        for i, w in enumerate(workers):
            work_time = w.get('work_time_ns', 0) / 1e6
            objects = w.get('objects_in_segment', 0)
            traversals = w.get('traversals_performed', 0)
            worker_times.append(work_time)
            print(f"  {i:>8} | {work_time:>10.2f} | {objects:>10} | {traversals:>12}")

        if worker_times:
            min_time = min(worker_times)
            max_time = max(worker_times)
            mean_time = statistics.mean(worker_times)
            load_imbalance = max_time / min_time if min_time > 0 else 0

            print(f"\n  Worker time: min={min_time:.2f}ms, max={max_time:.2f}ms, mean={mean_time:.2f}ms")
            print(f"  Load imbalance: {load_imbalance:.2f}x (max/min)")

            # Theoretical speedup analysis
            print(f"\n  Theoretical Analysis:")
            print(f"    - Serial time (estimated): {serial_mean:.1f}ms")
            print(f"    - Parallel time: {parallel_mean:.1f}ms")
            print(f"    - Max worker time: {max_time:.1f}ms (bottleneck)")
            print(f"    - Barrier overhead: {sub_mean - max_time:.1f}ms")
            print(f"    - Theoretical max speedup: {serial_mean / max_time:.2f}x (limited by slowest worker)")
            print(f"    - Actual speedup: {speedup:.2f}x")

    return {
        'heap_type': heap_type,
        'heap_size': heap_size,
        'num_workers': num_workers,
        'survivor_ratio': survivor_ratio,
        'serial_mean_ms': serial_mean,
        'parallel_mean_ms': parallel_mean,
        'speedup': speedup,
        'phase_breakdown': {
            'update_refs_ms': upd_mean,
            'mark_alive_ms': alive_mean,
            'subtract_refs_ms': sub_mean,
            'mark_ms': mark_mean,
            'cleanup_ms': cleanup_mean,
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Profile subtract_refs bottleneck')
    parser.add_argument('--heap-type', '-t', type=str, default='chain',
                        choices=['chain', 'tree', 'independent'],
                        help='Heap type (default: chain)')
    parser.add_argument('--heap-size', '-s', type=str, default='500k',
                        help='Heap size with K/M suffix (default: 500k)')
    parser.add_argument('--workers', '-w', type=int, default=8,
                        help='Number of workers (default: 8)')
    parser.add_argument('--survivor-ratio', '-r', type=float, default=0.5,
                        help='Survivor ratio (default: 0.5)')
    parser.add_argument('--iterations', '-n', type=int, default=5,
                        help='Number of timed iterations (default: 5)')
    parser.add_argument('--warmup', type=int, default=2,
                        help='Number of warmup iterations (default: 2)')

    args = parser.parse_args()

    # Parse heap size
    size_str = args.heap_size.upper()
    if size_str.endswith('K'):
        heap_size = int(size_str[:-1]) * 1000
    elif size_str.endswith('M'):
        heap_size = int(size_str[:-1]) * 1000000
    else:
        heap_size = int(size_str)

    if not PARALLEL_GC_AVAILABLE:
        print("ERROR: Parallel GC not available in this build")
        return

    profile_subtract_refs(
        heap_type=args.heap_type,
        heap_size=heap_size,
        num_workers=args.workers,
        survivor_ratio=args.survivor_ratio,
        iterations=args.iterations,
        warmup=args.warmup
    )


if __name__ == '__main__':
    main()
