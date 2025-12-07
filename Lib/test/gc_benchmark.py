#!/usr/bin/env python3
"""
Parallel GC Benchmark Suite - Unified for GIL and Free-Threading Builds

This benchmark measures parallel GC performance across 5 dimensions:
1. Worker count (1, 2, 4, 8)
2. Heap type (chain, tree, wide_tree, graph, layered, independent)
3. Heap size (10k, 50k, 100k, 500k, 1M objects)
4. Survivor ratio (0%, 25%, 50%, 75%, 100%)
5. Multi-threaded creation (FTP only: 1, 2, 4, 8 threads)

Usage:
    python gc_benchmark.py --quick             # Quick test
    python gc_benchmark.py --workers 1,2,4    # Specific worker counts
    python gc_benchmark.py --heap-type chain  # Specific heap type
    python gc_benchmark.py --full-matrix -o results.json  # Full benchmark
"""

import gc
import sys
import time
import json
import random
import argparse
import statistics
import threading
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor

# =============================================================================
# Build Detection
# =============================================================================

def detect_build():
    """Detect whether we're running on GIL or FTP (Free-Threading) build."""
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
# GC Control - Unified API
# =============================================================================

def enable_parallel_gc(num_workers: int):
    """Enable parallel GC with specified worker count."""
    try:
        # FTP build uses keyword argument
        if BUILD_TYPE == "ftp":
            gc.enable_parallel(num_workers=num_workers)
        else:
            gc.enable_parallel(num_workers)
    except RuntimeError as e:
        if "not available" in str(e).lower():
            raise
        # Already enabled, reconfigure
        pass

def disable_parallel_gc():
    """Disable parallel GC."""
    try:
        gc.disable_parallel()
    except (RuntimeError, AttributeError):
        pass  # Already disabled or not available

def set_parallel_threshold(threshold: int):
    """Set the parallel GC threshold (minimum roots to use parallel)."""
    try:
        gc.set_parallel_threshold(threshold)
    except AttributeError:
        pass  # Not available

# =============================================================================
# Node Class for Object Graphs
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
    """
    Create a single linked list chain: A -> B -> C -> ... -> Z

    Parallelizability: VERY LOW
    - Sequential dependencies mean workers can't split the work
    - This is the worst case for parallel GC
    """
    nodes = [Node() for _ in range(n)]
    for i in range(n - 1):
        nodes[i].refs.append(nodes[i + 1])
    return nodes

def create_tree(n: int, branching: int = 2) -> List[Node]:
    """
    Create a balanced tree with given branching factor.

    Parallelizability: MEDIUM
    - Limited by tree depth
    - Workers can process different subtrees
    """
    import math
    # Calculate depth needed for ~n nodes
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

def create_wide_tree(roots: int, children_per_root: int) -> List[Node]:
    """
    Create many independent subtrees - best case for parallelism.

    Parallelizability: HIGH
    - Each root is completely independent
    - Workers can process different roots in parallel
    """
    nodes = []
    for _ in range(roots):
        root = Node()
        nodes.append(root)
        for _ in range(children_per_root):
            child = Node()
            root.refs.append(child)
            nodes.append(child)
    return nodes

def create_graph(n: int, edge_prob: float = 0.3) -> List[Node]:
    """
    Create a random graph with cross-references (can have cycles).

    Parallelizability: MEDIUM
    - Some work-stealing opportunities
    - Cross-references create dependencies
    """
    random.seed(42)  # Reproducible
    nodes = [Node() for _ in range(n)]

    # Add random edges
    edges_per_node = max(1, int(n * edge_prob / n * 3))  # ~3 edges on average
    for node in nodes:
        for _ in range(random.randint(1, edges_per_node)):
            target = random.choice(nodes)
            node.refs.append(target)

    return nodes

def create_layered(layers: int, nodes_per_layer: int) -> List[Node]:
    """
    Create a neural network-like structure with sparse connections between layers.

    Each node connects to cbrt(nodes_per_layer) random nodes in the previous layer.
    This models a dense interconnected system while scaling as O(n^(4/3)) instead
    of O(n²) for fully-connected layers.

    Parallelizability: MEDIUM
    - Layer boundaries provide some parallelism
    - But connections between layers create dependencies
    """
    random.seed(42)  # Reproducible

    all_nodes = []
    prev_layer = None

    # Each node connects to cbrt(layer_size) nodes in previous layer
    # Cube root keeps it dense but scales as O(n^(4/3)) instead of O(n^(3/2))
    connections_per_node = max(1, int(nodes_per_layer ** (1/3)))

    for _ in range(layers):
        layer = [Node() for _ in range(nodes_per_layer)]
        all_nodes.extend(layer)

        if prev_layer:
            # Use random.choices for batch sampling (faster than per-element)
            for node in layer:
                node.refs = random.choices(prev_layer, k=connections_per_node)

        prev_layer = layer

    return all_nodes

def create_independent(n: int) -> List[Node]:
    """
    Create many isolated objects with no references.

    Parallelizability: HIGHEST
    - No dependencies at all
    - Workers can process any object independently
    """
    return [Node() for _ in range(n)]

# Map heap type names to generator functions
HEAP_GENERATORS = {
    'chain': lambda n: create_chain(n),
    'tree': lambda n: create_tree(n, branching=2),
    'wide_tree': lambda n: create_wide_tree(n // 11, 10),  # ~n nodes total
    'graph': lambda n: create_graph(n, edge_prob=0.3),
    'layered': lambda n: create_layered(10, n // 10),  # 10 layers
    'independent': lambda n: create_independent(n),
}

# =============================================================================
# Multi-threaded Object Creation (FTP only)
# =============================================================================

def create_objects_multithreaded(heap_type: str, total_objects: int,
                                  num_threads: int) -> List[Node]:
    """
    Create objects across multiple threads.

    This simulates real-world scenarios where objects are created by
    different threads and may be distributed across per-thread heaps.
    """
    if num_threads <= 1:
        return HEAP_GENERATORS[heap_type](total_objects)

    objects_per_thread = total_objects // num_threads
    all_nodes = []
    lock = threading.Lock()

    def create_worker(thread_id: int):
        nodes = HEAP_GENERATORS[heap_type](objects_per_thread)
        with lock:
            all_nodes.extend(nodes)

    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=create_worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return all_nodes

# =============================================================================
# Benchmark Result Data Structures
# =============================================================================

@dataclass
class RunResult:
    """Result of a single benchmark run."""
    time_ms: float
    objects_collected: int

@dataclass
class BenchmarkResult:
    """Results from a single benchmark configuration."""
    # Configuration
    build_type: str
    heap_type: str
    heap_size: int
    survivor_ratio: float
    num_workers: int
    creation_threads: int  # FTP only

    # Mode (serial or parallel)
    mode: str

    # Timing statistics (milliseconds)
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float

    # Collection statistics
    mean_collected: float

    # Raw data
    runs: List[RunResult] = field(default_factory=list)

    @property
    def times_ms(self) -> List[float]:
        return [r.time_ms for r in self.runs]

@dataclass
class ComparisonResult:
    """Comparison between serial and parallel for same configuration."""
    config: Dict[str, Any]
    serial_result: BenchmarkResult
    parallel_result: BenchmarkResult

    @property
    def speedup(self) -> float:
        if self.parallel_result.mean_ms == 0:
            return 0.0
        return self.serial_result.mean_ms / self.parallel_result.mean_ms

    @property
    def significant(self) -> bool:
        # Simple significance test
        if len(self.serial_result.runs) < 2 or len(self.parallel_result.runs) < 2:
            return False
        se = ((self.serial_result.std_ms**2 / len(self.serial_result.runs) +
               self.parallel_result.std_ms**2 / len(self.parallel_result.runs)) ** 0.5)
        return abs(self.serial_result.mean_ms - self.parallel_result.mean_ms) > 2 * se if se > 0 else False

# =============================================================================
# Benchmark Runner
# =============================================================================

class GCBenchmark:
    """Main benchmark runner."""

    def __init__(self, warmup: int = 3, iterations: int = 5, seed: int = 42):
        self.warmup = warmup
        self.iterations = iterations
        self.seed = seed

    def run_single(self, heap_type: str, heap_size: int, survivor_ratio: float,
                   num_workers: int, creation_threads: int = 1,
                   parallel: bool = True) -> BenchmarkResult:
        """Run a single benchmark configuration."""

        mode = "parallel" if parallel else "serial"
        runs = []

        # Configure GC
        gc.disable()
        if parallel and num_workers > 1:
            enable_parallel_gc(num_workers)
            set_parallel_threshold(100)  # Low threshold for benchmarking
        else:
            disable_parallel_gc()

        try:
            # Warmup
            for _ in range(self.warmup):
                random.seed(self.seed)
                if BUILD_TYPE == "ftp" and creation_threads > 1:
                    objects = create_objects_multithreaded(heap_type, heap_size, creation_threads)
                else:
                    objects = HEAP_GENERATORS[heap_type](heap_size)

                # Apply survivor ratio
                if survivor_ratio < 1.0:
                    num_keep = int(len(objects) * survivor_ratio)
                    if num_keep > 0:
                        random.shuffle(objects)
                        objects = objects[:num_keep]
                    else:
                        objects = []

                gc.collect()
                del objects
                gc.collect()

            # Timed runs
            for _ in range(self.iterations):
                random.seed(self.seed)
                if BUILD_TYPE == "ftp" and creation_threads > 1:
                    objects = create_objects_multithreaded(heap_type, heap_size, creation_threads)
                else:
                    objects = HEAP_GENERATORS[heap_type](heap_size)

                # Apply survivor ratio
                keep_refs = None
                if survivor_ratio < 1.0:
                    num_keep = int(len(objects) * survivor_ratio)
                    if num_keep > 0:
                        random.shuffle(objects)
                        keep_refs = objects[:num_keep]
                    objects = None  # Release original list
                else:
                    keep_refs = objects

                # Time the GC
                start = time.perf_counter()
                collected = gc.collect()
                elapsed = time.perf_counter() - start

                runs.append(RunResult(
                    time_ms=elapsed * 1000,
                    objects_collected=collected
                ))

                # Cleanup
                del keep_refs
                gc.collect()

        finally:
            gc.enable()

        # Calculate statistics
        times = [r.time_ms for r in runs]
        collected = [r.objects_collected for r in runs]
        sorted_times = sorted(times)

        return BenchmarkResult(
            build_type=BUILD_TYPE,
            heap_type=heap_type,
            heap_size=heap_size,
            survivor_ratio=survivor_ratio,
            num_workers=num_workers,
            creation_threads=creation_threads,
            mode=mode,
            mean_ms=statistics.mean(times),
            std_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
            min_ms=min(times),
            max_ms=max(times),
            p50_ms=sorted_times[len(times) // 2],
            p95_ms=sorted_times[int(len(times) * 0.95)] if len(times) >= 20 else max(times),
            mean_collected=statistics.mean(collected),
            runs=runs
        )

    def compare(self, heap_type: str, heap_size: int, survivor_ratio: float,
                num_workers: int, creation_threads: int = 1) -> ComparisonResult:
        """Compare serial vs parallel for a configuration."""

        config = {
            'heap_type': heap_type,
            'heap_size': heap_size,
            'survivor_ratio': survivor_ratio,
            'num_workers': num_workers,
            'creation_threads': creation_threads,
        }

        # Run serial (1 worker or parallel disabled)
        serial_result = self.run_single(
            heap_type, heap_size, survivor_ratio,
            num_workers=1, creation_threads=creation_threads,
            parallel=False
        )

        # Run parallel
        parallel_result = self.run_single(
            heap_type, heap_size, survivor_ratio,
            num_workers=num_workers, creation_threads=creation_threads,
            parallel=True
        )

        return ComparisonResult(
            config=config,
            serial_result=serial_result,
            parallel_result=parallel_result
        )

# =============================================================================
# Matrix Runner
# =============================================================================

def parse_list_arg(arg: str) -> List[str]:
    """Parse comma-separated argument."""
    return [x.strip() for x in arg.split(',')]

def parse_int_list(arg: str) -> List[int]:
    """Parse comma-separated integers with K/M suffix support."""
    result = []
    for x in arg.split(','):
        x = x.strip().upper()
        if x.endswith('K'):
            result.append(int(x[:-1]) * 1000)
        elif x.endswith('M'):
            result.append(int(x[:-1]) * 1000000)
        else:
            result.append(int(x))
    return result

def parse_float_list(arg: str) -> List[float]:
    """Parse comma-separated floats."""
    return [float(x.strip()) for x in arg.split(',')]

def run_benchmark_matrix(
    workers: List[int] = [1, 2, 4],
    heap_types: List[str] = ['chain', 'wide_tree'],
    heap_sizes: List[int] = [100000],
    survivor_ratios: List[float] = [0.5],
    creation_threads: List[int] = [1],
    warmup: int = 3,
    iterations: int = 5,
    output_file: Optional[str] = None,
    verbose: bool = True
) -> List[ComparisonResult]:
    """Run the full benchmark matrix."""

    if not PARALLEL_GC_AVAILABLE:
        print("ERROR: Parallel GC not available in this build")
        print("Rebuild with --with-parallel-gc (GIL) or --disable-gil (FTP)")
        return []

    results = []
    benchmark = GCBenchmark(warmup=warmup, iterations=iterations)

    # Header
    print("=" * 80)
    print(f"Parallel GC Benchmark - {BUILD_TYPE.upper()} Build")
    print("=" * 80)
    print(f"Workers: {workers}")
    print(f"Heap types: {heap_types}")
    print(f"Heap sizes: {heap_sizes}")
    print(f"Survivor ratios: {survivor_ratios}")
    if BUILD_TYPE == "ftp":
        print(f"Creation threads: {creation_threads}")
    print(f"Warmup: {warmup}, Iterations: {iterations}")
    print()

    total_configs = (len(workers) * len(heap_types) * len(heap_sizes) *
                    len(survivor_ratios) * len(creation_threads))
    current = 0

    print(f"{'Config':<40} | {'Serial':>10} | {'Parallel':>10} | {'Speedup':>10}")
    print("-" * 80)

    for heap_type in heap_types:
        for heap_size in heap_sizes:
            for survivor_ratio in survivor_ratios:
                for num_workers in workers:
                    for ct in creation_threads:
                        if BUILD_TYPE == "gil" and ct > 1:
                            continue  # Skip multi-thread creation for GIL build

                        current += 1
                        config_str = f"{heap_type}_{heap_size//1000}k_s{int(survivor_ratio*100)}_w{num_workers}"
                        if BUILD_TYPE == "ftp" and ct > 1:
                            config_str += f"_t{ct}"

                        if verbose:
                            print(f"[{current}/{total_configs}] {config_str:<36}", end=" | ", flush=True)

                        result = benchmark.compare(
                            heap_type=heap_type,
                            heap_size=heap_size,
                            survivor_ratio=survivor_ratio,
                            num_workers=num_workers,
                            creation_threads=ct
                        )
                        results.append(result)

                        if verbose:
                            speedup_str = f"{result.speedup:.2f}x"
                            if result.speedup > 1.1:
                                speedup_str = f"\033[32m{speedup_str}\033[0m"  # Green
                            elif result.speedup < 0.9:
                                speedup_str = f"\033[31m{speedup_str}\033[0m"  # Red

                            print(f"{result.serial_result.mean_ms:>10.2f} | "
                                  f"{result.parallel_result.mean_ms:>10.2f} | "
                                  f"{speedup_str:>10}")

    print("-" * 80)

    # Summary
    if results:
        speedups = [r.speedup for r in results if r.speedup > 0]
        significant = [r for r in results if r.significant]

        print(f"\nSummary:")
        print(f"  Total configurations: {len(results)}")
        print(f"  Mean speedup: {statistics.mean(speedups):.2f}x")
        print(f"  Min speedup:  {min(speedups):.2f}x")
        print(f"  Max speedup:  {max(speedups):.2f}x")
        print(f"  Significant:  {len(significant)}/{len(results)}")

    # Save results
    if output_file:
        output = {
            'metadata': {
                'build_type': BUILD_TYPE,
                'workers': workers,
                'heap_types': heap_types,
                'heap_sizes': heap_sizes,
                'survivor_ratios': survivor_ratios,
                'creation_threads': creation_threads,
                'warmup': warmup,
                'iterations': iterations,
            },
            'results': [
                {
                    'config': r.config,
                    'speedup': r.speedup,
                    'significant': r.significant,
                    'serial': asdict(r.serial_result),
                    'parallel': asdict(r.parallel_result),
                }
                for r in results
            ]
        }
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to: {output_file}")

    print("=" * 80)
    return results

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Parallel GC Benchmark Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test
  %(prog)s --quick

  # Test specific worker counts
  %(prog)s --workers 1,2,4,8

  # Test specific heap types
  %(prog)s --heap-type chain,wide_tree,independent

  # Full matrix with output
  %(prog)s --full-matrix -o results.json

  # Custom configuration
  %(prog)s --workers 2,4 --heap-type graph --heap-size 100k,500k --survivor-ratio 0.5,0.8
"""
    )

    parser.add_argument('--workers', '-w', type=str, default='1,2,4',
                        help='Comma-separated worker counts (default: 1,2,4)')
    parser.add_argument('--heap-type', '-t', type=str, default='chain,wide_tree',
                        help='Comma-separated heap types (default: chain,wide_tree)')
    parser.add_argument('--heap-size', '-s', type=str, default='100k',
                        help='Comma-separated heap sizes with K/M suffix (default: 100k)')
    parser.add_argument('--survivor-ratio', '-r', type=str, default='0.5',
                        help='Comma-separated survivor ratios (default: 0.5)')
    parser.add_argument('--creation-threads', '-c', type=str, default='1',
                        help='FTP only: Comma-separated creation thread counts (default: 1)')
    parser.add_argument('--warmup', type=int, default=3,
                        help='Warmup iterations (default: 3)')
    parser.add_argument('--iterations', '-n', type=int, default=5,
                        help='Timed iterations (default: 5)')
    parser.add_argument('--output', '-o', type=str,
                        help='Output JSON file')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Quick test: 2 warmup, 3 iterations, small matrix')
    parser.add_argument('--full-matrix', '-f', action='store_true',
                        help='Run full benchmark matrix')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')

    args = parser.parse_args()

    if args.quick:
        args.warmup = 2
        args.iterations = 3
        args.workers = '1,2,4'
        args.heap_type = 'chain,wide_tree'
        args.heap_size = '50k'
        args.survivor_ratio = '0.5'

    if args.full_matrix:
        args.workers = '1,2,4,8'
        args.heap_type = 'chain,tree,wide_tree,graph,layered,independent'
        args.heap_size = '10k,50k,100k,500k'
        args.survivor_ratio = '0.0,0.25,0.5,0.75,1.0'
        if BUILD_TYPE == "ftp":
            args.creation_threads = '1,2,4'

    run_benchmark_matrix(
        workers=parse_int_list(args.workers),
        heap_types=parse_list_arg(args.heap_type),
        heap_sizes=parse_int_list(args.heap_size),
        survivor_ratios=parse_float_list(args.survivor_ratio),
        creation_threads=parse_int_list(args.creation_threads),
        warmup=args.warmup,
        iterations=args.iterations,
        output_file=args.output,
        verbose=not args.quiet
    )

if __name__ == '__main__':
    main()
