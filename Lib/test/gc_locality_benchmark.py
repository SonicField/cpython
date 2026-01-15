#!/usr/bin/env python3
"""
High-Locality GC Benchmark

Tests parallel GC performance on memory-local heap layouts where
cache effects dominate and parallelisation is hardest.

This is the WORST CASE for parallel GC - sequential chains where
objects are allocated contiguously and traversal is cache-friendly.

Usage:
    python gc_locality_benchmark.py
    python gc_locality_benchmark.py --size 500000 --workers 8
"""

import gc
import time
import argparse
import statistics


class Node:
    """Minimal node for chain/tree structures."""
    __slots__ = ['next', 'value']

    def __init__(self, value=0):
        self.next = None
        self.value = value


def build_sequential_chain(n):
    """
    Build a sequential chain of N objects.

    This is the worst case for parallel GC because:
    1. Objects are allocated contiguously (good cache locality)
    2. Traversal is inherently sequential (each node -> next node)
    3. Only one worker can do useful work at a time
    4. Serial GC benefits from prefetching; parallel cannot
    """
    root = Node(0)
    current = root
    for i in range(1, n):
        new_node = Node(i)
        current.next = new_node
        current = new_node
    return [root]  # Return in a list to keep root alive


def run_gc_timing(heap, generation=2, iterations=5):
    """Run GC and return timing statistics."""
    times = []
    for _ in range(iterations):
        gc.collect()  # Clear any pending garbage
        start = time.perf_counter()
        gc.collect(generation=generation)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return {
        'min': min(times),
        'max': max(times),
        'mean': statistics.mean(times),
        'stdev': statistics.stdev(times) if len(times) > 1 else 0,
        'all': times,
    }


def run_benchmark(size, workers, iterations=5, warmup=2):
    """Run the high-locality benchmark."""
    print(f"High-Locality GC Benchmark")
    print(f"=" * 60)
    print(f"Heap size: {size:,} objects")
    print(f"Workers: {workers}")
    print(f"Iterations: {iterations} (after {warmup} warmup)")
    print()

    # Build the heap
    print("Building sequential chain...")
    gc.disable()
    heap = build_sequential_chain(size)
    gc.enable()

    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        gc.collect(generation=2)

    # Serial benchmark
    print()
    print("Running SERIAL benchmark...")
    gc.disable_parallel()
    serial_stats = run_gc_timing(heap, iterations=iterations)
    print(f"  Min: {serial_stats['min']:.2f}ms")
    print(f"  Mean: {serial_stats['mean']:.2f}ms")
    print(f"  Max: {serial_stats['max']:.2f}ms")

    # Parallel benchmark
    print()
    print(f"Running PARALLEL benchmark ({workers} workers)...")
    gc.enable_parallel(num_workers=workers)
    parallel_stats = run_gc_timing(heap, iterations=iterations)
    print(f"  Min: {parallel_stats['min']:.2f}ms")
    print(f"  Mean: {parallel_stats['mean']:.2f}ms")
    print(f"  Max: {parallel_stats['max']:.2f}ms")

    # Get phase timing from last collection
    try:
        stats = gc.get_parallel_stats()
        phase_timing = stats.get('phase_timing', {})
        print()
        print("Phase timing (last collection):")
        for phase, ns in phase_timing.items():
            if ns != 0:
                print(f"  {phase}: {ns/1e6:.2f}ms")
    except AttributeError:
        pass

    # Comparison
    print()
    print("=" * 60)
    speedup = serial_stats['mean'] / parallel_stats['mean']
    if speedup >= 1.0:
        print(f"RESULT: {speedup:.2f}x SPEEDUP (parallel is faster)")
    else:
        print(f"RESULT: {1/speedup:.2f}x SLOWDOWN (parallel is slower)")
    print(f"Serial mean: {serial_stats['mean']:.2f}ms")
    print(f"Parallel mean: {parallel_stats['mean']:.2f}ms")
    print(f"Overhead: {parallel_stats['mean'] - serial_stats['mean']:.2f}ms")

    return {
        'serial': serial_stats,
        'parallel': parallel_stats,
        'speedup': speedup,
    }


def main():
    parser = argparse.ArgumentParser(description="High-Locality GC Benchmark")
    parser.add_argument('--size', '-s', type=int, default=500000,
                        help='Number of objects in chain (default: 500000)')
    parser.add_argument('--workers', '-w', type=int, default=8,
                        help='Number of parallel workers (default: 8)')
    parser.add_argument('--iterations', '-i', type=int, default=5,
                        help='Number of timed iterations (default: 5)')
    parser.add_argument('--warmup', type=int, default=2,
                        help='Number of warmup iterations (default: 2)')
    args = parser.parse_args()

    # Check if parallel GC is available
    try:
        gc.get_parallel_config()
    except AttributeError:
        print("ERROR: Parallel GC not available in this build")
        return 1

    run_benchmark(
        size=args.size,
        workers=args.workers,
        iterations=args.iterations,
        warmup=args.warmup,
    )
    return 0


if __name__ == '__main__':
    exit(main())
