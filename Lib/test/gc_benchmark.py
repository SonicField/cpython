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
from typing import List, Dict, Any, Optional

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

# =============================================================================
# Node Classes for Object Graphs
# =============================================================================

class Node:
    """Generic node for building object graphs."""
    __slots__ = ['refs', 'data', '__weakref__']

    def __init__(self):
        self.refs = []
        self.data = None


class FinalizerNode:
    """Node with a finalizer (__del__ method)."""
    __slots__ = ['refs', 'data', '__weakref__']

    def __init__(self):
        self.refs = []
        self.data = None

    def __del__(self):
        pass  # Presence of __del__ is what matters


class ContainerNode:
    """Node using __dict__ with list and dict children - models real objects."""

    def __init__(self):
        self.children_list = []
        self.children_dict = {}
        self.parent_ref = None  # Will hold weakref for some instances

# =============================================================================
# Heap Generators - Isolated Cyclic Clusters
# =============================================================================
#
# All heap generators return List[List[Node]] - a list of independent cyclic
# clusters. This allows survivor_ratio to work correctly by discarding complete
# clusters, ensuring discarded objects are truly unreachable and can be collected.
#
# For FTP (free-threading), GC only collects cyclic garbage - reference counting
# handles acyclic structures. Creating isolated cycles is essential for meaningful
# GC benchmarks.

DEFAULT_CLUSTER_SIZE = 100  # Nodes per cluster


def create_chain(n: int, cluster_size: int = DEFAULT_CLUSTER_SIZE,
                 node_class: type = None) -> List[List]:
    """
    Create isolated circular chains: A -> B -> C -> ... -> Z -> A

    Each cluster is a closed loop, so discarding a cluster creates cyclic garbage.

    Parallelizability: VERY LOW within cluster
    - Sequential dependencies within each chain
    - But clusters can be processed independently
    """
    if node_class is None:
        node_class = Node
    clusters = []
    num_clusters = max(1, n // cluster_size)

    for _ in range(num_clusters):
        nodes = [node_class() for _ in range(cluster_size)]
        # Make circular
        for i in range(cluster_size):
            nodes[i].refs.append(nodes[(i + 1) % cluster_size])
        clusters.append(nodes)

    return clusters


def create_tree(n: int, cluster_size: int = DEFAULT_CLUSTER_SIZE,
                node_class: type = None) -> List[List]:
    """
    Create isolated cyclic trees - each tree has back-references to root.

    Each cluster is a tree where leaves reference back to root, creating cycles.

    Parallelizability: MEDIUM
    - Tree structure within cluster
    - Clusters are independent
    """
    if node_class is None:
        node_class = Node
    clusters = []
    num_clusters = max(1, n // cluster_size)
    branching = 2

    for _ in range(num_clusters):
        nodes = []
        root = node_class()
        nodes.append(root)

        # Build tree
        level = [root]
        while len(nodes) < cluster_size:
            next_level = []
            for parent in level:
                for _ in range(branching):
                    if len(nodes) >= cluster_size:
                        break
                    child = node_class()
                    parent.refs.append(child)
                    # Back-reference to root creates cycle
                    child.refs.append(root)
                    next_level.append(child)
                    nodes.append(child)
            if not next_level:
                break
            level = next_level

        clusters.append(nodes)

    return clusters


def create_wide_tree(n: int, cluster_size: int = DEFAULT_CLUSTER_SIZE,
                     node_class: type = None) -> List[List]:
    """
    Create isolated wide trees with cyclic back-references.

    Each cluster has one root with many children, all children ref back to root.

    Parallelizability: HIGH
    - Simple structure, easy to traverse
    - Clusters are independent
    """
    if node_class is None:
        node_class = Node
    clusters = []
    num_clusters = max(1, n // cluster_size)

    for _ in range(num_clusters):
        nodes = []
        root = node_class()
        nodes.append(root)

        for _ in range(cluster_size - 1):
            child = node_class()
            root.refs.append(child)
            child.refs.append(root)  # Back-reference creates cycle
            nodes.append(child)

        clusters.append(nodes)

    return clusters


def create_graph(n: int, cluster_size: int = DEFAULT_CLUSTER_SIZE,
                 node_class: type = None) -> List[List]:
    """
    Create isolated random graphs with internal cycles.

    Each cluster is a fully-connected random graph with many cycles.

    Parallelizability: MEDIUM
    - Random structure within cluster
    - Clusters are independent
    """
    if node_class is None:
        node_class = Node
    clusters = []
    num_clusters = max(1, n // cluster_size)

    for _ in range(num_clusters):
        nodes = [node_class() for _ in range(cluster_size)]

        # Add random edges within cluster
        for node in nodes:
            for _ in range(random.randint(1, 3)):
                target = random.choice(nodes)
                node.refs.append(target)

        clusters.append(nodes)

    return clusters

def create_layered(n: int, cluster_size: int = DEFAULT_CLUSTER_SIZE,
                   node_class: type = None) -> List[List]:
    """
    Create isolated layered networks with cycles.

    Each cluster is a mini neural-network-like structure with back-references
    from the last layer to the first, creating cycles.

    Parallelizability: MEDIUM
    - Layered structure within cluster
    - Clusters are independent
    """
    if node_class is None:
        node_class = Node
    clusters = []
    num_clusters = max(1, n // cluster_size)
    layers_per_cluster = 4
    nodes_per_layer = cluster_size // layers_per_cluster

    for _ in range(num_clusters):
        all_nodes = []
        first_layer = None
        prev_layer = None

        for layer_idx in range(layers_per_cluster):
            layer = [node_class() for _ in range(nodes_per_layer)]
            all_nodes.extend(layer)

            if first_layer is None:
                first_layer = layer

            if prev_layer:
                # Connect to previous layer
                for node in layer:
                    node.refs.append(random.choice(prev_layer))

            prev_layer = layer

        # Back-reference from last layer to first creates cycles
        if prev_layer and first_layer:
            for node in prev_layer:
                node.refs.append(random.choice(first_layer))

        clusters.append(all_nodes)

    return clusters


def create_independent(n: int, cluster_size: int = DEFAULT_CLUSTER_SIZE,
                       node_class: type = None) -> List[List]:
    """
    Create isolated self-referencing clusters.

    Each cluster contains nodes that reference each other in a cycle.
    This is the simplest cyclic structure - each node refs the next.

    Parallelizability: HIGHEST
    - Simple structure, minimal traversal
    - Clusters are independent
    """
    if node_class is None:
        node_class = Node
    clusters = []
    num_clusters = max(1, n // cluster_size)

    for _ in range(num_clusters):
        nodes = [node_class() for _ in range(cluster_size)]
        # Simple cycle: each node refs the next
        for i in range(cluster_size):
            nodes[i].refs.append(nodes[(i + 1) % cluster_size])
        clusters.append(nodes)

    return clusters


def create_ai_workload(n: int, cluster_size: int = DEFAULT_CLUSTER_SIZE) -> List[List[Node]]:
    """
    Create isolated AI-workload-like clusters with cycles.

    Each cluster models a mini ML computation graph:
    - ContainerNode parents with list and dict children
    - 10% of children have finalizers
    - Cross-references within cluster create cycles

    Parallelizability: MEDIUM
    - Complex structure within cluster
    - Clusters are independent
    """

    clusters = []
    num_clusters = max(1, n // cluster_size)

    for _ in range(num_clusters):
        all_nodes = []

        # Create parent-child structure with cycles
        num_parents = cluster_size // 6  # Each parent has ~5 children

        parents = []
        for _ in range(num_parents):
            parent = ContainerNode()
            parents.append(parent)
            all_nodes.append(parent)

            # Add 3-5 children
            for j in range(random.randint(3, 5)):
                if random.random() < 0.1:
                    child = FinalizerNode()
                else:
                    child = Node()

                all_nodes.append(child)

                if random.random() < 0.5:
                    parent.children_list.append(child)
                else:
                    parent.children_dict[f"child_{j}"] = child

                # Back-reference to parent creates cycle
                child.refs.append(parent)

        # Cross-references between parents (more cycles)
        for parent in parents:
            if parents:
                parent.children_list.append(random.choice(parents))

        clusters.append(all_nodes)

    return clusters


# Module-level node class selection (set by CLI --finalizers flag)
_node_class = Node


def set_node_class(use_finalizers: bool):
    """Set the node class to use for heap generation."""
    global _node_class
    _node_class = FinalizerNode if use_finalizers else Node


def get_node_class():
    """Get the current node class."""
    return _node_class


# Map heap type names to generator functions
# All generators return List[List[Node]] - a list of independent cyclic clusters
# Note: We use get_node_class() to get late binding of the node class
HEAP_GENERATORS = {
    'chain': lambda n: create_chain(n, node_class=get_node_class()),
    'tree': lambda n: create_tree(n, node_class=get_node_class()),
    'wide_tree': lambda n: create_wide_tree(n, node_class=get_node_class()),
    'graph': lambda n: create_graph(n, node_class=get_node_class()),
    'layered': lambda n: create_layered(n, node_class=get_node_class()),
    'independent': lambda n: create_independent(n, node_class=get_node_class()),
    'ai_workload': lambda n: create_ai_workload(n),  # Uses its own mixed types
}

# =============================================================================
# Multi-threaded Object Creation (FTP only)
# =============================================================================

def create_objects_multithreaded(heap_type: str, total_objects: int,
                                  num_threads: int) -> List[List[Node]]:
    """
    Create objects across multiple threads.

    This simulates real-world scenarios where objects are created by
    different threads and may be distributed across per-thread heaps.

    Returns a list of clusters (each cluster is a list of nodes).
    """
    if num_threads <= 1:
        return HEAP_GENERATORS[heap_type](total_objects)

    objects_per_thread = total_objects // num_threads
    all_clusters = []
    lock = threading.Lock()

    def create_worker(thread_id: int):
        clusters = HEAP_GENERATORS[heap_type](objects_per_thread)
        with lock:
            all_clusters.extend(clusters)

    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=create_worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return all_clusters

# =============================================================================
# Benchmark Result Data Structures
# =============================================================================

@dataclass
class RunResult:
    """Result of a single benchmark run."""
    time_ms: float
    objects_collected: int
    # Phase timing from gc.get_parallel_stats() (nanoseconds)
    phase_timing: Dict[str, int] = field(default_factory=dict)

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

    # Phase timing statistics (nanoseconds) - averaged across runs
    # Keys depend on build: GIL has subtract_refs_ns/mark_ns, FTP has update_refs_ns/mark_heap_ns
    mean_phase_timing: Dict[str, float] = field(default_factory=dict)

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
        if parallel:
            enable_parallel_gc(num_workers)
        else:
            disable_parallel_gc()

        try:
            # Warmup
            for _ in range(self.warmup):
                random.seed(self.seed)
                if BUILD_TYPE == "ftp" and creation_threads > 1:
                    clusters = create_objects_multithreaded(heap_type, heap_size, creation_threads)
                else:
                    clusters = HEAP_GENERATORS[heap_type](heap_size)

                # Apply survivor ratio - discard complete clusters
                # This ensures discarded objects are truly unreachable (cyclic garbage)
                if survivor_ratio < 1.0:
                    num_keep = int(len(clusters) * survivor_ratio)
                    if num_keep > 0:
                        random.shuffle(clusters)
                        clusters = clusters[:num_keep]
                    else:
                        clusters = []

                gc.collect()
                clusters = None  # Release references
                gc.collect()

            # Timed runs
            for _ in range(self.iterations):
                random.seed(self.seed)
                if BUILD_TYPE == "ftp" and creation_threads > 1:
                    clusters = create_objects_multithreaded(heap_type, heap_size, creation_threads)
                else:
                    clusters = HEAP_GENERATORS[heap_type](heap_size)

                # Apply survivor ratio - discard complete clusters
                # This ensures discarded objects are truly unreachable (cyclic garbage)
                keep_refs = None
                if survivor_ratio < 1.0:
                    num_keep = int(len(clusters) * survivor_ratio)
                    if num_keep > 0:
                        random.shuffle(clusters)
                        keep_refs = clusters[:num_keep]
                else:
                    keep_refs = clusters
                clusters = None  # Release original list (must be after keep_refs assignment)

                # Time the GC
                start = time.perf_counter()
                collected = gc.collect()
                elapsed = time.perf_counter() - start

                # Get phase timing from parallel stats
                phase_timing = {}
                if parallel:
                    try:
                        stats = gc.get_parallel_stats()
                        if 'phase_timing' in stats:
                            phase_timing = stats['phase_timing']
                    except AttributeError:
                        pass  # API not available

                runs.append(RunResult(
                    time_ms=elapsed * 1000,
                    objects_collected=collected,
                    phase_timing=phase_timing
                ))

                # Cleanup: release all references and collect garbage to reset state
                keep_refs = None
                gc.collect()

        finally:
            gc.enable()

        # Calculate statistics
        times = [r.time_ms for r in runs]
        collected = [r.objects_collected for r in runs]
        sorted_times = sorted(times)

        # Aggregate phase timing (mean across runs)
        mean_phase_timing = {}
        if runs and runs[0].phase_timing:
            # Get keys from first run
            keys = runs[0].phase_timing.keys()
            for key in keys:
                values = [r.phase_timing.get(key, 0) for r in runs if r.phase_timing]
                if values:
                    mean_phase_timing[key] = statistics.mean(values)

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
            mean_phase_timing=mean_phase_timing,
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

        # Run parallel (skip if num_workers < 2)
        if num_workers < 2:
            parallel_result = serial_result  # Use serial as "parallel" for 1-worker case
        else:
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
    verbose: bool = True,
    finalizers: bool = False
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
    print(f"Finalizers: {'enabled' if finalizers else 'disabled'}")
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

                            # Per-iteration details
                            print(f"    Serial runs:   ", end="")
                            print("  ".join(f"{r.time_ms:.1f}" for r in result.serial_result.runs))

                            print(f"    Parallel runs: ", end="")
                            par_details = []
                            for r in result.parallel_result.runs:
                                pt = r.phase_timing
                                if pt:
                                    total = pt.get('total_ns', 1) / 1e6  # avoid div by zero
                                    if BUILD_TYPE == "gil":
                                        upd = pt.get('update_refs_ns', 0) / 1e6
                                        alive = pt.get('mark_alive_ns', 0) / 1e6
                                        sub = pt.get('subtract_refs_ns', 0) / 1e6
                                        mark = pt.get('mark_ns', 0) / 1e6
                                        cleanup = pt.get('cleanup_ns', 0) / 1e6
                                        par_details.append(f"{r.time_ms:.0f}[upd:{upd:.0f}+alive:{alive:.0f}+sub:{sub:.0f}+mark:{mark:.0f}+clean:{cleanup:.0f}]")
                                    else:
                                        # FTP: show key phases with percentages
                                        # Group: mark_alive | parallel_phases | cleanup_phases
                                        mk_alive = pt.get('mark_alive_ns', 0) / 1e6
                                        # Parallel phases
                                        upd = pt.get('update_refs_ns', 0) / 1e6
                                        mk_heap = pt.get('mark_heap_ns', 0) / 1e6
                                        scan = pt.get('scan_heap_ns', 0) / 1e6
                                        # Cleanup phases (often substantial)
                                        cleanup = pt.get('cleanup_ns', 0) / 1e6
                                        finalize = pt.get('finalize_ns', 0) / 1e6
                                        resurrection = pt.get('resurrection_ns', 0) / 1e6
                                        weakrefs = pt.get('find_weakrefs_ns', 0) / 1e6
                                        # Calculate percentage of total
                                        parallel_phases = upd + mk_heap + scan
                                        cleanup_phases = cleanup + finalize + resurrection + weakrefs
                                        # Format: total[alive|par|cleanup] with percentages
                                        alive_pct = int(mk_alive / total * 100) if total > 0 else 0
                                        par_pct = int(parallel_phases / total * 100) if total > 0 else 0
                                        clean_pct = int(cleanup_phases / total * 100) if total > 0 else 0
                                        par_details.append(f"{r.time_ms:.0f}[{alive_pct}%|{par_pct}%|{clean_pct}%]")
                                else:
                                    par_details.append(f"{r.time_ms:.1f}")
                            print("  ".join(par_details))

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

        # Phase timing summary (for parallel runs only)
        parallel_with_timing = [r for r in results if r.parallel_result.mean_phase_timing]
        if parallel_with_timing:
            print(f"\nPhase Timing (parallel runs, mean across iterations, in ms):")
            # Header based on build type
            if BUILD_TYPE == "gil":
                print(f"  {'Config':<25} | {'upd_refs':>8} | {'mk_alive':>8} | {'sub_refs':>8} | {'mark':>8} | {'cleanup':>8} | {'total':>8}")
                print(f"  {'-' * 25}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}")
            else:
                # FTP: Show all significant phases grouped logically
                print(f"  {'Config':<18} | {'mk_alive':>7} | {'upd_ref':>7} | {'mk_heap':>7} | {'scan':>7} | {'cleanup':>7} | {'final':>7} | {'resur':>7} | {'weakrf':>7} | {'total':>7}")
                print(f"  {'-' * 18}-+-{'-' * 7}-+-{'-' * 7}-+-{'-' * 7}-+-{'-' * 7}-+-{'-' * 7}-+-{'-' * 7}-+-{'-' * 7}-+-{'-' * 7}-+-{'-' * 7}")

            for r in parallel_with_timing:
                pt = r.parallel_result.mean_phase_timing
                config_str = f"{r.config['heap_type']}_{r.config['heap_size']//1000}k_w{r.config['num_workers']}"

                # Get phase values (convert ns to ms for readability)
                if BUILD_TYPE == "gil":
                    upd = pt.get('update_refs_ns', 0) / 1e6
                    alive = pt.get('mark_alive_ns', 0) / 1e6
                    sub = pt.get('subtract_refs_ns', 0) / 1e6
                    mark = pt.get('mark_ns', 0) / 1e6
                    cleanup = pt.get('cleanup_ns', 0) / 1e6
                    total = pt.get('total_ns', 0) / 1e6
                    print(f"  {config_str:<25} | {upd:>8.1f} | {alive:>8.1f} | {sub:>8.1f} | {mark:>8.1f} | {cleanup:>8.1f} | {total:>8.1f}")
                else:
                    # FTP: Extract all key phases
                    mk_alive = pt.get('mark_alive_ns', 0) / 1e6
                    upd = pt.get('update_refs_ns', 0) / 1e6
                    mk_heap = pt.get('mark_heap_ns', 0) / 1e6
                    scan = pt.get('scan_heap_ns', 0) / 1e6
                    cleanup = pt.get('cleanup_ns', 0) / 1e6
                    finalize = pt.get('finalize_ns', 0) / 1e6
                    resurrection = pt.get('resurrection_ns', 0) / 1e6
                    weakrefs = pt.get('find_weakrefs_ns', 0) / 1e6
                    total = pt.get('total_ns', 0) / 1e6
                    print(f"  {config_str:<18} | {mk_alive:>7.1f} | {upd:>7.1f} | {mk_heap:>7.1f} | {scan:>7.1f} | {cleanup:>7.1f} | {finalize:>7.1f} | {resurrection:>7.1f} | {weakrefs:>7.1f} | {total:>7.1f}")

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
                'finalizers': finalizers,
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
    parser.add_argument('--finalizers', action='store_true',
                        help='Use nodes with __del__ finalizers (tests cleanup phase)')

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

    # Set node class based on --finalizers flag
    set_node_class(args.finalizers)

    run_benchmark_matrix(
        workers=parse_int_list(args.workers),
        heap_types=parse_list_arg(args.heap_type),
        heap_sizes=parse_int_list(args.heap_size),
        survivor_ratios=parse_float_list(args.survivor_ratio),
        creation_threads=parse_int_list(args.creation_threads),
        warmup=args.warmup,
        iterations=args.iterations,
        output_file=args.output,
        verbose=not args.quiet,
        finalizers=args.finalizers
    )

if __name__ == '__main__':
    main()
