#!/usr/bin/env python3
"""
Parallel GC Performance Benchmark Suite

A unified benchmark for measuring parallel GC performance. Focuses on realistic
workloads based on pyperformance benchmarks. Works with both GIL and
free-threading Python builds.

Usage:
    python gc_perf_benchmark.py                    # Standard suite (~5 min)
    python gc_perf_benchmark.py --quick            # Quick sanity check (~1 min)
    python gc_perf_benchmark.py --full             # Full suite (~15 min)
    python gc_perf_benchmark.py --json             # Output as JSON
    python gc_perf_benchmark.py --include-synthetic # Also run synthetic stress tests
"""

import gc
import sys
import time
import random
import argparse
import threading
import statistics
import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from datetime import datetime
import pickle

# =============================================================================
# Build Detection
# =============================================================================

def detect_build() -> str:
    """Detect whether we're running on GIL or FTP build."""
    try:
        gil_enabled = sys._is_gil_enabled()
        return "ftp" if not gil_enabled else "gil"
    except AttributeError:
        return "gil"


def is_parallel_gc_available() -> bool:
    """Check if parallel GC is available."""
    try:
        config = gc.get_parallel_config()
        return config.get('available', False)
    except AttributeError:
        return False


def get_cpu_count() -> int:
    """Get number of CPUs available."""
    try:
        import os
        return os.cpu_count() or 4
    except Exception:
        return 4


BUILD_TYPE = detect_build()
PARALLEL_GC_AVAILABLE = is_parallel_gc_available()
CPU_COUNT = get_cpu_count()

# =============================================================================
# GC Control
# =============================================================================

def enable_parallel_gc(num_workers: int):
    """Enable parallel GC with specified worker count."""
    try:
        if BUILD_TYPE == "ftp":
            gc.enable_parallel(num_workers=num_workers)
        else:
            gc.enable_parallel(num_workers)
    except (RuntimeError, AttributeError):
        pass


def disable_parallel_gc():
    """Disable parallel GC."""
    try:
        gc.disable_parallel()
    except (RuntimeError, AttributeError):
        pass


def get_parallel_stats() -> Dict[str, Any]:
    """Get parallel GC statistics if available."""
    try:
        return gc.get_parallel_stats()
    except AttributeError:
        return {}

# =============================================================================
# Result Data Structures
# =============================================================================

@dataclass
class CollectionResult:
    """Result from a single collection benchmark (measures GC collection time)."""
    heap_type: str
    heap_size: int
    serial_time_ms: float
    parallel_time_ms: float
    serial_stdev: float = 0.0
    parallel_stdev: float = 0.0
    num_runs: int = 1

    @property
    def speedup(self) -> float:
        if self.parallel_time_ms > 0:
            return self.serial_time_ms / self.parallel_time_ms
        return 1.0


@dataclass
class BenchmarkRun:
    """Result from a single benchmark run."""
    throughput: float  # workloads/sec or objects/sec
    gc_time_ms: float
    gc_overhead_pct: float
    stw_pause_ms: float  # Mean STW pause
    stw_max_ms: float
    collections: int
    duration_sec: float
    phase_timing: Dict[str, float] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Aggregated results from multiple runs of a benchmark."""
    name: str
    description: str
    mode: str  # "serial" or "parallel-N"
    runs: List[BenchmarkRun] = field(default_factory=list)

    @property
    def throughput_mean(self) -> float:
        return statistics.mean(r.throughput for r in self.runs) if self.runs else 0

    @property
    def throughput_stdev(self) -> float:
        if len(self.runs) < 2:
            return 0
        return statistics.stdev(r.throughput for r in self.runs)

    @property
    def throughput_best(self) -> float:
        return max(r.throughput for r in self.runs) if self.runs else 0

    @property
    def throughput_worst(self) -> float:
        return min(r.throughput for r in self.runs) if self.runs else 0

    @property
    def stw_pause_mean(self) -> float:
        return statistics.mean(r.stw_pause_ms for r in self.runs) if self.runs else 0

    @property
    def stw_pause_stdev(self) -> float:
        if len(self.runs) < 2:
            return 0
        return statistics.stdev(r.stw_pause_ms for r in self.runs)

    @property
    def stw_pause_max(self) -> float:
        return max(r.stw_max_ms for r in self.runs) if self.runs else 0

    @property
    def gc_overhead_mean(self) -> float:
        return statistics.mean(r.gc_overhead_pct for r in self.runs) if self.runs else 0

    @property
    def total_duration(self) -> float:
        return sum(r.duration_sec for r in self.runs)

    @property
    def total_collections(self) -> int:
        return sum(r.collections for r in self.runs)


@dataclass
class ComparisonResult:
    """Comparison between serial and parallel for a benchmark."""
    benchmark_name: str
    serial: BenchmarkResult
    parallel: BenchmarkResult

    @property
    def speedup(self) -> float:
        if self.serial.throughput_mean == 0:
            return 0
        return self.parallel.throughput_mean / self.serial.throughput_mean

    @property
    def speedup_best(self) -> float:
        if self.serial.throughput_best == 0:
            return 0
        return self.parallel.throughput_best / self.serial.throughput_best

    @property
    def speedup_worst(self) -> float:
        if self.serial.throughput_worst == 0:
            return 0
        return self.parallel.throughput_worst / self.serial.throughput_worst


@dataclass
class SuiteResult:
    """Results from running the full benchmark suite."""
    build_type: str
    parallel_gc_available: bool
    num_workers: int
    timestamp: str
    duration_per_benchmark: float = 30.0
    num_runs: int = 3
    num_threads: int = 4
    heap_size: int = 500000
    realistic: Optional[ComparisonResult] = None
    synthetic_by_heap: Dict[str, ComparisonResult] = field(default_factory=dict)
    collection_results: List[CollectionResult] = field(default_factory=list)
    phase_timing: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def geometric_mean_speedup(self) -> float:
        """Geometric mean of speedups across all synthetic heap types."""
        if not self.synthetic_by_heap:
            return 1.0
        speedups = [r.speedup for r in self.synthetic_by_heap.values() if r.speedup > 0]
        if not speedups:
            return 1.0
        product = 1.0
        for s in speedups:
            product *= s
        return product ** (1.0 / len(speedups))

    @property
    def geometric_mean_stw_reduction(self) -> float:
        """Geometric mean of STW pause reduction ratios across all synthetic heap types."""
        if not self.synthetic_by_heap:
            return 1.0
        ratios = []
        for r in self.synthetic_by_heap.values():
            if r.serial.stw_pause_mean > 0 and r.parallel.stw_pause_mean > 0:
                ratios.append(r.parallel.stw_pause_mean / r.serial.stw_pause_mean)
        if not ratios:
            return 1.0
        product = 1.0
        for ratio in ratios:
            product *= ratio
        return product ** (1.0 / len(ratios))

# =============================================================================
# Realistic Workloads (pyperformance-style)
# =============================================================================

def workload_richards() -> None:
    """Richards benchmark - OS task scheduler simulation."""
    BUFSIZE = 4

    class Packet:
        __slots__ = ['link', 'ident', 'kind', 'datum', 'data']
        def __init__(self, link, ident, kind):
            self.link = link
            self.ident = ident
            self.kind = kind
            self.datum = 0
            self.data = [0] * BUFSIZE

    class TaskRec:
        __slots__ = ['pending', 'work_in', 'device_in', 'control', 'count']
        def __init__(self):
            self.pending = None
            self.work_in = None
            self.device_in = None
            self.control = 1
            self.count = 10000

    class Task:
        __slots__ = ['link', 'ident', 'priority', 'input', 'handle', 'task_holding', 'task_waiting']
        def __init__(self, ident, priority, input_queue, handle):
            self.link = None
            self.ident = ident
            self.priority = priority
            self.input = input_queue
            self.handle = handle
            self.task_holding = False
            self.task_waiting = False

    tasks = []
    packets = []

    for i in range(6):
        rec = TaskRec()
        t = Task(i, i * 10, None, rec)
        if tasks:
            t.link = tasks[-1]
        tasks.append(t)

        for j in range(3):
            pkt = Packet(None, i, j)
            if packets:
                pkt.link = packets[-1]
            packets.append(pkt)

    for _ in range(100):
        for t in tasks:
            if packets:
                pkt = packets.pop()
                old_input = t.input
                t.input = pkt
                pkt.link = old_input


def workload_deltablue() -> None:
    """DeltaBlue benchmark - constraint solver with bidirectional references."""
    class Variable:
        __slots__ = ['value', 'constraints', 'determined_by', 'walk_strength', 'stay', 'mark']
        def __init__(self, value=0):
            self.value = value
            self.constraints = []
            self.determined_by = None
            self.walk_strength = 0
            self.stay = True
            self.mark = 0

    class Constraint:
        __slots__ = ['strength', 'variables']
        def __init__(self, strength, variables):
            self.strength = strength
            self.variables = variables
            for v in variables:
                v.constraints.append(self)

    variables = [Variable(i) for i in range(50)]
    constraints = []

    for i in range(len(variables) - 1):
        c = Constraint(i % 5, [variables[i], variables[i + 1]])
        constraints.append(c)

    for i in range(0, len(variables) - 2, 2):
        c = Constraint((i + 1) % 5, [variables[i], variables[i + 2]])
        constraints.append(c)


def workload_deepcopy() -> None:
    """Deep copy benchmark - creates cyclic garbage."""
    class Node:
        def __init__(self, value):
            self.value = value
            self.children = []
            self.parent = None

    def build_tree(depth, breadth):
        root = Node(0)
        level = [root]
        for d in range(depth):
            next_level = []
            for parent in level:
                for b in range(breadth):
                    child = Node(d * breadth + b)
                    child.parent = parent
                    parent.children.append(child)
                    next_level.append(child)
            level = next_level
        return root

    tree = build_tree(4, 3)
    for _ in range(5):
        copied = deepcopy(tree)
        del copied


def workload_pickle_copy() -> None:
    """Pickle copy benchmark - serialization creates temporary objects."""
    class DataNode:
        def __init__(self, data):
            self.data = data
            self.refs = []

    nodes = [DataNode(list(range(100))) for _ in range(20)]
    for i, node in enumerate(nodes):
        node.refs = [nodes[(i + 1) % len(nodes)], nodes[(i + 2) % len(nodes)]]

    for _ in range(3):
        data = pickle.dumps(nodes)
        restored = pickle.loads(data)
        del restored


def workload_async_tree() -> None:
    """Async tree benchmark - simulates async task trees."""
    class AsyncTask:
        __slots__ = ['parent', 'children', 'result', 'state']
        def __init__(self, parent=None):
            self.parent = parent
            self.children = []
            self.result = None
            self.state = 'pending'
            if parent:
                parent.children.append(self)

    def build_task_tree(depth, breadth):
        root = AsyncTask()
        level = [root]
        for _ in range(depth):
            next_level = []
            for parent in level:
                for _ in range(breadth):
                    child = AsyncTask(parent)
                    next_level.append(child)
            level = next_level
        return root

    for _ in range(10):
        tree = build_task_tree(4, 3)
        del tree


def workload_nbody() -> None:
    """N-body simulation - minimal cycles, compute heavy."""
    PI = 3.14159265358979323
    SOLAR_MASS = 4 * PI * PI
    DAYS_PER_YEAR = 365.24

    bodies = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, SOLAR_MASS],
        [4.84143144246472090e+00, -1.16032004402742839e+00, -1.03622044471123109e-01,
         1.66007664274403694e-03 * DAYS_PER_YEAR, 7.69901118419740425e-03 * DAYS_PER_YEAR,
         -6.90460016972063023e-05 * DAYS_PER_YEAR, 9.54791938424326609e-04 * SOLAR_MASS],
        [8.34336671824457987e+00, 4.12479856412430479e+00, -4.03523417114321381e-01,
         -2.76742510726862411e-03 * DAYS_PER_YEAR, 4.99852801234917238e-03 * DAYS_PER_YEAR,
         2.30417297573763929e-05 * DAYS_PER_YEAR, 2.85885980666130812e-04 * SOLAR_MASS],
    ]

    dt = 0.01
    for _ in range(100):
        for i, body1 in enumerate(bodies):
            for body2 in bodies[i + 1:]:
                dx = body1[0] - body2[0]
                dy = body1[1] - body2[1]
                dz = body1[2] - body2[2]
                dist = (dx * dx + dy * dy + dz * dz) ** 0.5
                mag = dt / (dist * dist * dist)
                body1[3] -= dx * body2[6] * mag
                body1[4] -= dy * body2[6] * mag
                body1[5] -= dz * body2[6] * mag
                body2[3] += dx * body1[6] * mag
                body2[4] += dy * body1[6] * mag
                body2[5] += dz * body1[6] * mag


def workload_comprehensions() -> None:
    """List/dict/set comprehensions - no cycles."""
    data = list(range(1000))

    result1 = [x * 2 for x in data if x % 3 == 0]
    result2 = {x: x ** 2 for x in data if x % 5 == 0}
    result3 = {x for x in data if x % 7 == 0}

    nested = [[y * x for y in range(10)] for x in range(100)]

    del result1, result2, result3, nested


# Workload registry with cycle characteristics
REALISTIC_WORKLOADS: Dict[str, Callable] = {
    'deltablue': workload_deltablue,      # HIGH_CYCLES
    'deepcopy': workload_deepcopy,        # HIGH_CYCLES
    'pickle_copy': workload_pickle_copy,  # HIGH_CYCLES
    'async_tree': workload_async_tree,    # HIGH_CYCLES
    'richards': workload_richards,        # MINIMAL_CYCLES
    'nbody': workload_nbody,              # MINIMAL_CYCLES
    'comprehensions': workload_comprehensions,  # NO_CYCLES
}

# =============================================================================
# Synthetic Heap Generators
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
        self.parent_ref = None


def create_chain(n: int, cluster_size: int = DEFAULT_CLUSTER_SIZE,
                 node_class: type = None) -> List[List]:
    """
    Create isolated circular chains: A -> B -> C -> ... -> Z -> A

    Each cluster is a closed loop, so discarding a cluster creates cyclic garbage.
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

    Each cluster is a mini neural-network-like structure with bidirectional
    references between first and last layers, creating cycles.
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

        # Bidirectional references between first and last layer create cycles
        if prev_layer and first_layer:
            for node in prev_layer:
                node.refs.append(random.choice(first_layer))
            for node in first_layer:
                node.refs.append(random.choice(prev_layer))

        clusters.append(all_nodes)

    return clusters


def create_independent(n: int, cluster_size: int = DEFAULT_CLUSTER_SIZE,
                       node_class: type = None) -> List[List]:
    """
    Create isolated self-referencing clusters.

    Each cluster contains nodes that reference each other in a cycle.
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


def create_ai_workload(n: int, cluster_size: int = DEFAULT_CLUSTER_SIZE) -> List[List]:
    """
    Create isolated AI-workload-like clusters with cycles.

    Each cluster models a mini ML computation graph:
    - ContainerNode parents with list and dict children
    - 10% of children have finalizers
    - Cross-references within cluster create cycles
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


def create_web_server(n: int, cluster_size: int = 200) -> List[List]:
    """
    Create isolated web server request-like clusters with NO cross-cluster references.

    Each cluster models a single HTTP request/response lifecycle:
    - A Request object (ContainerNode) with headers, body, session
    - Associated Response object with data
    - Middleware/handler chain with back-references (creates cycles)
    - Database result objects

    CRITICAL: No cross-cluster references. Each request is fully independent.
    """
    clusters = []
    num_clusters = max(1, n // cluster_size)

    for _ in range(num_clusters):
        all_nodes = []

        # Request object - the root of this request's object graph
        request = ContainerNode()
        all_nodes.append(request)

        # Headers dict (simulated as nodes)
        for i in range(5):
            header = Node()
            request.children_dict[f"header_{i}"] = header
            all_nodes.append(header)

        # Request body / parsed data
        body = ContainerNode()
        request.children_list.append(body)
        all_nodes.append(body)

        # Session object with back-reference to request (cycle!)
        session = ContainerNode()
        session.children_list.append(request)  # Back-ref creates cycle
        request.children_dict["session"] = session
        all_nodes.append(session)

        # Session data items
        for i in range(10):
            item = Node()
            session.children_list.append(item)
            all_nodes.append(item)

        # Response object
        response = ContainerNode()
        request.children_dict["response"] = response
        response.children_list.append(request)  # Back-ref creates cycle
        all_nodes.append(response)

        # Response body chunks
        for i in range(8):
            chunk = Node()
            response.children_list.append(chunk)
            all_nodes.append(chunk)

        # Middleware chain (each references next and previous - cycles)
        middleware_chain = []
        for i in range(5):
            mw = ContainerNode()
            middleware_chain.append(mw)
            all_nodes.append(mw)
            if i > 0:
                mw.children_list.append(middleware_chain[i-1])  # Prev
                middleware_chain[i-1].children_list.append(mw)  # Next

        # First middleware attached to request
        if middleware_chain:
            request.children_list.append(middleware_chain[0])
            middleware_chain[0].children_list.append(request)  # Cycle

        # Database query results (attached to response)
        db_results = ContainerNode()
        response.children_dict["db_results"] = db_results
        all_nodes.append(db_results)

        # Result rows
        for i in range(15):
            row = Node()
            db_results.children_list.append(row)
            all_nodes.append(row)

        # Fill remaining cluster size with generic handler objects
        remaining = cluster_size - len(all_nodes)
        for _ in range(max(0, remaining)):
            handler = Node()
            # Reference something in the request graph (creates more cycles)
            handler.refs.append(request)
            request.children_list.append(handler)
            all_nodes.append(handler)

        clusters.append(all_nodes)

    return clusters


# All 8 heap types for collection benchmarks
HEAP_GENERATORS = {
    "chain": create_chain,
    "tree": create_tree,
    "wide_tree": create_wide_tree,
    "graph": create_graph,
    "layered": create_layered,
    "independent": create_independent,
    "ai_workload": create_ai_workload,
    "web_server": create_web_server,
}

# =============================================================================
# Pause Tracking
# =============================================================================

class PauseTracker:
    """Track GC pause times."""

    def __init__(self, parallel_enabled: bool = False):
        self.gc_times_ms: List[float] = []
        self.stw_pauses_ms: List[float] = []
        self.gc_start_time: Optional[float] = None
        self.parallel_enabled = parallel_enabled

    def gc_callback(self, phase: str, info: dict):
        if phase == "start":
            self.gc_start_time = time.perf_counter()
        elif phase == "stop":
            if self.gc_start_time is not None:
                gc_time_ms = (time.perf_counter() - self.gc_start_time) * 1000
                self.gc_times_ms.append(gc_time_ms)

                if self.parallel_enabled:
                    stats = get_parallel_stats()
                    if 'phase_timing' in stats:
                        pt = stats['phase_timing']
                        stw_phases = ['mark_alive_ns', 'update_refs_ns', 'mark_heap_ns',
                                     'scan_heap_ns', 'resurrection_ns']
                        stw_ns = sum(pt.get(p, 0) for p in stw_phases)
                        self.stw_pauses_ms.append(stw_ns / 1e6)
                    else:
                        self.stw_pauses_ms.append(gc_time_ms)
                else:
                    self.stw_pauses_ms.append(gc_time_ms)

    def reset(self):
        self.gc_times_ms.clear()
        self.stw_pauses_ms.clear()

# =============================================================================
# Realistic Benchmark Runner
# =============================================================================

def run_realistic_benchmark(
    duration_sec: float,
    num_threads: int,
    parallel_workers: int = 0,
    verbose: bool = False
) -> BenchmarkRun:
    """
    Run realistic mixed-workload benchmark.

    Args:
        duration_sec: How long to run
        num_threads: Number of worker threads
        parallel_workers: 0 for serial, >0 for parallel GC
        verbose: Include phase timing
    """
    gc.collect()
    gc.disable()

    if parallel_workers > 0:
        enable_parallel_gc(parallel_workers)
    else:
        disable_parallel_gc()

    tracker = PauseTracker(parallel_enabled=parallel_workers > 0)
    gc.callbacks.append(tracker.gc_callback)

    workload_counts: Dict[str, int] = {name: 0 for name in REALISTIC_WORKLOADS}
    workload_list = list(REALISTIC_WORKLOADS.items())
    lock = threading.Lock()
    stop_flag = threading.Event()

    def worker():
        local_counts = {name: 0 for name in REALISTIC_WORKLOADS}
        while not stop_flag.is_set():
            name, func = random.choice(workload_list)
            try:
                func()
                local_counts[name] += 1
            except Exception:
                pass

        with lock:
            for name, count in local_counts.items():
                workload_counts[name] += count

    gc.enable()
    start_time = time.perf_counter()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(num_threads)]
    for t in threads:
        t.start()

    time.sleep(duration_sec)
    stop_flag.set()

    for t in threads:
        t.join(timeout=2.0)

    end_time = time.perf_counter()
    gc.disable()

    gc.callbacks.remove(tracker.gc_callback)

    actual_duration = end_time - start_time
    total_workloads = sum(workload_counts.values())
    throughput = total_workloads / actual_duration

    total_gc_time = sum(tracker.gc_times_ms)
    gc_overhead = (total_gc_time / 1000) / actual_duration * 100

    stw_mean = statistics.mean(tracker.stw_pauses_ms) if tracker.stw_pauses_ms else 0
    stw_max = max(tracker.stw_pauses_ms) if tracker.stw_pauses_ms else 0

    phase_timing = {}
    if verbose and parallel_workers > 0:
        stats = get_parallel_stats()
        if 'phase_timing' in stats:
            phase_timing = {k: v / 1e6 for k, v in stats['phase_timing'].items()}

    gc.enable()

    return BenchmarkRun(
        throughput=throughput,
        gc_time_ms=total_gc_time,
        gc_overhead_pct=gc_overhead,
        stw_pause_ms=stw_mean,
        stw_max_ms=stw_max,
        collections=len(tracker.gc_times_ms),
        duration_sec=actual_duration,
        phase_timing=phase_timing
    )

# =============================================================================
# Synthetic Benchmark Runner
# =============================================================================

def run_synthetic_benchmark(
    duration_sec: float,
    heap_size: int,
    heap_type: str,
    num_threads: int,
    parallel_workers: int = 0,
    verbose: bool = False
) -> BenchmarkRun:
    """
    Run synthetic throughput benchmark with specified heap type.
    """
    gc.collect()
    gc.disable()

    if parallel_workers > 0:
        enable_parallel_gc(parallel_workers)
    else:
        disable_parallel_gc()

    tracker = PauseTracker(parallel_enabled=parallel_workers > 0)
    gc.callbacks.append(tracker.gc_callback)

    heap_generator = HEAP_GENERATORS[heap_type]
    churn_size = heap_size // 100
    objects_per_thread = heap_size // num_threads
    churn_per_thread = churn_size // num_threads

    total_created = [0]
    lock = threading.Lock()
    stop_flag = threading.Event()

    def worker():
        local_heap = heap_generator(objects_per_thread)
        local_created = 0

        while not stop_flag.is_set():
            num_discard = min(len(local_heap), churn_per_thread // 100 + 1)
            if num_discard > 0:
                random.shuffle(local_heap)
                local_heap = local_heap[num_discard:]

            new_clusters = heap_generator(churn_per_thread)
            local_heap.extend(new_clusters)
            local_created += churn_per_thread

        with lock:
            total_created[0] += local_created

    gc.enable()
    start_time = time.perf_counter()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(num_threads)]
    for t in threads:
        t.start()

    time.sleep(duration_sec)
    stop_flag.set()

    for t in threads:
        t.join(timeout=2.0)

    end_time = time.perf_counter()
    gc.disable()

    gc.callbacks.remove(tracker.gc_callback)

    actual_duration = end_time - start_time
    throughput = total_created[0] / actual_duration

    total_gc_time = sum(tracker.gc_times_ms)
    gc_overhead = (total_gc_time / 1000) / actual_duration * 100

    stw_mean = statistics.mean(tracker.stw_pauses_ms) if tracker.stw_pauses_ms else 0
    stw_max = max(tracker.stw_pauses_ms) if tracker.stw_pauses_ms else 0

    phase_timing = {}
    if verbose and parallel_workers > 0:
        stats = get_parallel_stats()
        if 'phase_timing' in stats:
            phase_timing = {k: v / 1e6 for k, v in stats['phase_timing'].items()}

    gc.enable()

    return BenchmarkRun(
        throughput=throughput,
        gc_time_ms=total_gc_time,
        gc_overhead_pct=gc_overhead,
        stw_pause_ms=stw_mean,
        stw_max_ms=stw_max,
        collections=len(tracker.gc_times_ms),
        duration_sec=actual_duration,
        phase_timing=phase_timing
    )

# =============================================================================
# Collection Time Benchmark (measures single GC collection times)
# =============================================================================

class CreationThreadPool:
    """
    Thread pool for object creation that keeps threads alive.

    Keeps threads alive so mimalloc pages remain in live thread heaps
    (not abandoned pool). This matches the old gc_benchmark.py methodology.
    """

    def __init__(self, num_threads: int):
        import queue as queue_module
        self.num_threads = num_threads
        self.task_queue = queue_module.Queue()
        self.result_queue = queue_module.Queue()
        self.shutdown_flag = threading.Event()
        self.threads = []

        for i in range(num_threads):
            t = threading.Thread(target=self._worker, args=(i,), daemon=True)
            t.start()
            self.threads.append(t)

    def _worker(self, thread_id: int):
        """Worker thread that waits for creation tasks."""
        while not self.shutdown_flag.is_set():
            try:
                task = self.task_queue.get(timeout=0.1)
            except:
                continue

            if task is None:
                break

            heap_type, num_objects = task
            clusters = HEAP_GENERATORS[heap_type](num_objects)
            self.result_queue.put(clusters)
            # Release reference immediately to avoid retaining garbage across iterations
            clusters = None
            del clusters
            self.task_queue.task_done()

    def create_objects(self, heap_type: str, total_objects: int) -> List:
        """Create objects using the thread pool."""
        objects_per_thread = total_objects // self.num_threads

        # Submit tasks
        for _ in range(self.num_threads):
            self.task_queue.put((heap_type, objects_per_thread))

        # Wait for completion
        self.task_queue.join()

        # Collect results
        all_clusters = []
        while not self.result_queue.empty():
            clusters = self.result_queue.get()
            all_clusters.extend(clusters)

        return all_clusters

    def shutdown(self):
        """Shutdown the thread pool."""
        self.shutdown_flag.set()
        for _ in range(self.num_threads):
            self.task_queue.put(None)
        for t in self.threads:
            t.join(timeout=1.0)


# Global thread pool (kept alive between runs like old benchmark)
_creation_pool = None


def get_creation_pool(num_threads: int) -> CreationThreadPool:
    """Get or create the global creation thread pool."""
    global _creation_pool
    if _creation_pool is None or _creation_pool.num_threads != num_threads:
        if _creation_pool is not None:
            _creation_pool.shutdown()
        _creation_pool = CreationThreadPool(num_threads)
    return _creation_pool


def run_collection_benchmark(
    heap_size: int,
    heap_type: str,
    num_runs: int,
    parallel_workers: int = 8,
    survivor_ratio: float = 0.8,
    creation_threads: int = 4,
    warmup_runs: int = 3
) -> CollectionResult:
    """
    Measure GC collection time for a given heap type.

    Matches the old gc_benchmark.py methodology exactly:
    - Creates heap across multiple threads using persistent thread pool
    - Keeps threads alive (pages remain in live thread heaps)
    - Uses survivor ratio on complete CLUSTERS (not individual objects)
    - Warmup runs before measurement
    - Serial and parallel measured in separate passes (not interleaved)
    """
    pool = get_creation_pool(creation_threads)
    seed = 42

    def run_batch(use_parallel: bool, num_iterations: int, is_warmup: bool) -> List[float]:
        """Run a batch of iterations and return times in ms."""
        gc.disable()

        if use_parallel:
            enable_parallel_gc(parallel_workers)
        else:
            disable_parallel_gc()

        times = []

        try:
            for _ in range(num_iterations):
                random.seed(seed)

                # Create clusters using thread pool (threads stay alive)
                clusters = pool.create_objects(heap_type, heap_size)

                # Apply survivor ratio by keeping complete CLUSTERS
                keep_refs = None
                if survivor_ratio < 1.0:
                    num_keep = int(len(clusters) * survivor_ratio)
                    if num_keep > 0:
                        random.shuffle(clusters)
                        keep_refs = clusters[:num_keep]
                    else:
                        keep_refs = []
                else:
                    keep_refs = clusters

                clusters = None  # Release original list

                if is_warmup:
                    # Warmup: just collect and cleanup
                    gc.collect()
                    keep_refs = None
                    gc.collect()
                else:
                    # Timed run: measure collection time
                    start = time.perf_counter()
                    gc.collect()
                    elapsed = (time.perf_counter() - start) * 1000
                    times.append(elapsed)

                    # Cleanup
                    keep_refs = None
                    gc.collect()

        finally:
            gc.enable()

        return times

    # Run serial: warmup then measurements
    run_batch(use_parallel=False, num_iterations=warmup_runs, is_warmup=True)
    serial_times = run_batch(use_parallel=False, num_iterations=num_runs, is_warmup=False)

    # Run parallel: warmup then measurements
    run_batch(use_parallel=True, num_iterations=warmup_runs, is_warmup=True)
    parallel_times = run_batch(use_parallel=True, num_iterations=num_runs, is_warmup=False)

    serial_mean = statistics.mean(serial_times)
    parallel_mean = statistics.mean(parallel_times)
    serial_stdev = statistics.stdev(serial_times) if len(serial_times) > 1 else 0
    parallel_stdev = statistics.stdev(parallel_times) if len(parallel_times) > 1 else 0

    return CollectionResult(
        heap_type=heap_type,
        heap_size=heap_size,
        serial_time_ms=serial_mean,
        parallel_time_ms=parallel_mean,
        serial_stdev=serial_stdev,
        parallel_stdev=parallel_stdev,
        num_runs=num_runs
    )

# =============================================================================
# Suite Runner
# =============================================================================

def run_comparison(
    name: str,
    description: str,
    run_fn: Callable[..., BenchmarkRun],
    num_runs: int,
    parallel_workers: int,
    verbose: bool = False,
    **kwargs
) -> ComparisonResult:
    """Run a benchmark in both serial and parallel modes."""

    # Serial runs
    serial_runs = []
    for _ in range(num_runs):
        run = run_fn(parallel_workers=0, verbose=verbose, **kwargs)
        serial_runs.append(run)

    serial_result = BenchmarkResult(
        name=name,
        description=description,
        mode="serial",
        runs=serial_runs
    )

    # Parallel runs
    parallel_runs = []
    for _ in range(num_runs):
        run = run_fn(parallel_workers=parallel_workers, verbose=verbose, **kwargs)
        parallel_runs.append(run)

    parallel_result = BenchmarkResult(
        name=name,
        description=description,
        mode=f"parallel-{parallel_workers}",
        runs=parallel_runs
    )

    return ComparisonResult(
        benchmark_name=name,
        serial=serial_result,
        parallel=parallel_result
    )


def run_suite(
    duration_per_benchmark: float = 30.0,
    num_runs: int = 3,
    num_threads: int = 4,
    parallel_workers: int = 8,
    heap_size: int = 500000,
    verbose: bool = False,
    include_synthetic: bool = True
) -> SuiteResult:
    """Run the full benchmark suite."""

    print(f"Parallel GC Performance Benchmark")
    print(f"=" * 60)
    print(f"Build type: {BUILD_TYPE.upper()}")
    print(f"Parallel GC available: {PARALLEL_GC_AVAILABLE}")
    print(f"CPUs: {CPU_COUNT}")
    print(f"Worker threads: {num_threads}")
    print(f"Parallel GC workers: {parallel_workers}")
    print(f"Duration per benchmark: {duration_per_benchmark}s")
    print(f"Runs per configuration: {num_runs}")
    print(f"Heap size: {heap_size:,}")
    print()

    result = SuiteResult(
        build_type=BUILD_TYPE,
        parallel_gc_available=PARALLEL_GC_AVAILABLE,
        num_workers=parallel_workers,
        timestamp=datetime.now().isoformat(),
        duration_per_benchmark=duration_per_benchmark,
        num_runs=num_runs,
        num_threads=num_threads,
        heap_size=heap_size
    )

    if not PARALLEL_GC_AVAILABLE:
        print("WARNING: Parallel GC not available - running serial only")
        parallel_workers = 0

    # Realistic benchmark (most important)
    print("Running: Realistic Workloads (pyperformance-style)")
    print("-" * 60)
    result.realistic = run_comparison(
        name="realistic",
        description="Mixed workloads based on pyperformance benchmarks",
        run_fn=run_realistic_benchmark,
        num_runs=num_runs,
        parallel_workers=parallel_workers,
        verbose=verbose,
        duration_sec=duration_per_benchmark,
        num_threads=num_threads
    )
    _print_comparison(result.realistic)
    print()

    if include_synthetic:
        # Collection time benchmarks (measures single GC collection time)
        # Run all 8 heap types - these are fast
        all_heap_types = list(HEAP_GENERATORS.keys())
        print("Running: Collection Time Benchmarks (all heap types)")
        print("-" * 60)
        for heap_type in all_heap_types:
            coll = run_collection_benchmark(
                heap_size=heap_size,
                heap_type=heap_type,
                num_runs=num_runs,
                parallel_workers=parallel_workers
            )
            result.collection_results.append(coll)
            print(f"  {heap_type:<12}: {coll.serial_time_ms:6.1f}ms -> {coll.parallel_time_ms:6.1f}ms "
                  f"({coll.speedup:.2f}x)")
        print()

        # Synthetic throughput - use representative subset (takes longer)
        throughput_heap_types = ["chain", "graph", "ai_workload"]
        for heap_type in throughput_heap_types:
            print(f"Running: Synthetic {heap_type} throughput")
            print("-" * 60)
            comp = run_comparison(
                name=f"synthetic_{heap_type}",
                description=f"{heap_type} heap structure",
                run_fn=run_synthetic_benchmark,
                num_runs=num_runs,
                parallel_workers=parallel_workers,
                verbose=verbose,
                duration_sec=duration_per_benchmark,
                heap_size=heap_size,
                heap_type=heap_type,
                num_threads=num_threads
            )
            result.synthetic_by_heap[heap_type] = comp
            _print_comparison(comp)
            print()

        # Print geometric mean summary
        if result.synthetic_by_heap:
            print("Synthetic Summary (geometric mean across heap types)")
            print("-" * 60)
            gm_speedup = result.geometric_mean_speedup
            gm_stw = result.geometric_mean_stw_reduction
            print(f"  Throughput change: {_format_change(gm_speedup)}")
            print(f"  STW pause change: {(gm_stw - 1) * 100:+.0f}%")
            print()

    return result


def _format_change(ratio: float) -> str:
    """Format a ratio as a percentage change with +/- sign."""
    pct = (ratio - 1) * 100
    if pct >= 0:
        return f"+{pct:.1f}%"
    else:
        return f"{pct:.1f}%"


def _print_comparison(comp: ComparisonResult):
    """Print a comparison result."""
    print(f"  Serial:   {comp.serial.throughput_mean:,.0f}/sec (range: {comp.serial.throughput_worst:,.0f} - {comp.serial.throughput_best:,.0f})")
    print(f"  Parallel: {comp.parallel.throughput_mean:,.0f}/sec (range: {comp.parallel.throughput_worst:,.0f} - {comp.parallel.throughput_best:,.0f})")
    print(f"  Throughput change: {_format_change(comp.speedup)}")
    if comp.serial.stw_pause_mean > 0:
        stw_change = (comp.parallel.stw_pause_mean / comp.serial.stw_pause_mean - 1) * 100
        print(f"  STW pause: {comp.serial.stw_pause_mean:.0f}ms -> {comp.parallel.stw_pause_mean:.0f}ms ({stw_change:+.0f}%)")

# =============================================================================
# Output Formatters
# =============================================================================

def format_markdown(result: SuiteResult) -> str:
    """Format results as markdown with aligned tables."""
    lines = []
    lines.append("# Parallel GC Performance Benchmark Results")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Build type: {result.build_type.upper()}")
    lines.append(f"- Parallel GC available: {result.parallel_gc_available}")
    lines.append(f"- Parallel workers: {result.num_workers}")
    lines.append(f"- Worker threads: {result.num_threads}")
    lines.append(f"- Duration per benchmark: {result.duration_per_benchmark}s")
    lines.append(f"- Runs per configuration: {result.num_runs}")
    lines.append(f"- Heap size (synthetic): {result.heap_size:,}")
    lines.append(f"- Timestamp: {result.timestamp}")
    lines.append("")

    # Realistic results (prominently displayed)
    if result.realistic:
        r = result.realistic
        lines.append("## Realistic Workloads (Primary Metric)")
        lines.append("")
        lines.append("Mixed workloads based on pyperformance benchmarks. This is the most")
        lines.append("representative measure of real-world parallel GC benefit.")
        lines.append("")

        # Runtime info
        serial_dur = r.serial.total_duration
        parallel_dur = r.parallel.total_duration
        serial_coll = r.serial.total_collections
        parallel_coll = r.parallel.total_collections
        lines.append(f"Runtime: {serial_dur + parallel_dur:.0f}s total "
                     f"({serial_coll + parallel_coll} collections)")
        lines.append("")

        # Aligned table for realistic results
        lines.append("| Metric           | Serial             | Parallel           | Change     |")
        lines.append("|------------------|--------------------|--------------------|------------|")

        # Throughput with stddev
        s_tp = f"{r.serial.throughput_mean:,.0f} ± {r.serial.throughput_stdev:,.0f}/s"
        p_tp = f"{r.parallel.throughput_mean:,.0f} ± {r.parallel.throughput_stdev:,.0f}/s"
        change_str = f"{(r.speedup - 1) * 100:+.1f}%"
        lines.append(f"| Throughput       | {s_tp:<18} | {p_tp:<18} | {change_str:<10} |")

        if r.serial.stw_pause_mean > 0:
            # STW pause mean with stddev
            s_stw = f"{r.serial.stw_pause_mean:.0f} ± {r.serial.stw_pause_stdev:.0f}ms"
            p_stw = f"{r.parallel.stw_pause_mean:.0f} ± {r.parallel.stw_pause_stdev:.0f}ms"
            stw_change_str = f"{(r.parallel.stw_pause_mean / r.serial.stw_pause_mean - 1) * 100:+.0f}%"
            lines.append(f"| STW pause (mean) | {s_stw:<18} | {p_stw:<18} | {stw_change_str:<10} |")

            # STW pause max
            s_max = f"{r.serial.stw_pause_max:.0f}ms"
            p_max = f"{r.parallel.stw_pause_max:.0f}ms"
            stw_max_change = (r.parallel.stw_pause_max / r.serial.stw_pause_max - 1) * 100 if r.serial.stw_pause_max > 0 else 0
            stw_max_str = f"{stw_max_change:+.0f}%"
            lines.append(f"| STW pause (max)  | {s_max:<18} | {p_max:<18} | {stw_max_str:<10} |")

        # GC overhead (absolute change in percentage)
        s_overhead = f"{r.serial.gc_overhead_mean:.1f}%"
        p_overhead = f"{r.parallel.gc_overhead_mean:.1f}%"
        overhead_change = r.parallel.gc_overhead_mean - r.serial.gc_overhead_mean
        overhead_str = f"{overhead_change:+.1f}%"
        lines.append(f"| GC overhead      | {s_overhead:<18} | {p_overhead:<18} | {overhead_str:<10} |")
        lines.append("")

    # Collection time benchmarks
    if result.collection_results:
        lines.append("## GC Collection Time (500k heap)")
        lines.append("")
        lines.append("Time to collect a single 500,000 object heap. Lower is better.")
        lines.append("")

        # Aligned table - wider heap type column for ai_workload, web_server
        lines.append("| Heap Type    | Serial (ms)        | Parallel (ms)      | Speedup |")
        lines.append("|--------------|--------------------|--------------------|---------|")

        for coll in result.collection_results:
            s_time = f"{coll.serial_time_ms:.1f} ± {coll.serial_stdev:.1f}"
            p_time = f"{coll.parallel_time_ms:.1f} ± {coll.parallel_stdev:.1f}"
            speedup_str = f"{coll.speedup:.2f}x"
            lines.append(f"| {coll.heap_type:<12} | {s_time:<18} | {p_time:<18} | {speedup_str:<7} |")

        # Geometric mean of collection speedups
        if len(result.collection_results) > 1:
            product = 1.0
            for coll in result.collection_results:
                product *= coll.speedup
            gm_speedup = product ** (1.0 / len(result.collection_results))
            gm_str = f"{gm_speedup:.2f}x"
            lines.append(f"| Geomean      |                    |                    | {gm_str:<7} |")
        lines.append("")

    # Synthetic throughput results
    if result.synthetic_by_heap:
        lines.append("## Synthetic Throughput (Per Heap Type)")
        lines.append("")
        lines.append("Steady-state throughput with continuous allocation. Representative subset of heap types.")
        lines.append("")

        # Aligned table
        lines.append("| Heap Type    | Serial             | Parallel           | Throughput | STW Change |")
        lines.append("|--------------|--------------------|--------------------|------------|------------|")

        for heap_type, comp in result.synthetic_by_heap.items():
            s_tp = f"{comp.serial.throughput_mean:,.0f}/s"
            p_tp = f"{comp.parallel.throughput_mean:,.0f}/s"
            tp_change = f"{(comp.speedup - 1) * 100:+.1f}%"
            if comp.serial.stw_pause_mean > 0:
                stw_change = f"{(comp.parallel.stw_pause_mean / comp.serial.stw_pause_mean - 1) * 100:+.0f}%"
            else:
                stw_change = "N/A"
            lines.append(f"| {heap_type:<12} | {s_tp:<18} | {p_tp:<18} | {tp_change:<10} | {stw_change:<10} |")

        # Geometric mean summary
        gm_speedup = f"{(result.geometric_mean_speedup - 1) * 100:+.1f}%"
        gm_stw = f"{(result.geometric_mean_stw_reduction - 1) * 100:+.0f}%"
        lines.append(f"| Geomean      |                    |                    | {gm_speedup:<10} | {gm_stw:<10} |")
        lines.append("")

        # STW Pause times table
        lines.append("### STW Pause Times")
        lines.append("")
        lines.append("| Heap Type    | Serial Mean (ms) | Serial Max (ms) | Parallel Mean (ms) | Parallel Max (ms) |")
        lines.append("|--------------|------------------|-----------------|--------------------|--------------------|")

        for heap_type, comp in result.synthetic_by_heap.items():
            s_mean = f"{comp.serial.stw_pause_mean:.0f}"
            s_max = f"{comp.serial.stw_pause_max:.0f}"
            p_mean = f"{comp.parallel.stw_pause_mean:.0f}"
            p_max = f"{comp.parallel.stw_pause_max:.0f}"
            lines.append(f"| {heap_type:<12} | {s_mean:<16} | {s_max:<15} | {p_mean:<18} | {p_max:<18} |")

        lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    if result.realistic:
        pct = (result.realistic.speedup - 1) * 100
        if pct > 0:
            lines.append(f"Parallel GC provides {pct:.1f}% throughput improvement on realistic workloads.")
        else:
            lines.append(f"Parallel GC shows {pct:.1f}% throughput change on realistic workloads.")

    if result.collection_results:
        product = 1.0
        for coll in result.collection_results:
            product *= coll.speedup
        gm_coll = product ** (1.0 / len(result.collection_results))
        lines.append(f"GC collection time improved by {gm_coll:.2f}x (geometric mean across heap types).")

    return "\n".join(lines)


def format_json(result: SuiteResult) -> str:
    """Format results as JSON."""
    def comparison_to_dict(comp: Optional[ComparisonResult]) -> Optional[Dict]:
        if comp is None:
            return None
        stw_change = 0
        if comp.serial.stw_pause_mean > 0:
            stw_change = (comp.parallel.stw_pause_mean / comp.serial.stw_pause_mean - 1) * 100
        return {
            "name": comp.benchmark_name,
            "serial": {
                "throughput_mean": comp.serial.throughput_mean,
                "throughput_stdev": comp.serial.throughput_stdev,
                "stw_pause_mean": comp.serial.stw_pause_mean,
                "stw_pause_stdev": comp.serial.stw_pause_stdev,
                "stw_pause_max": comp.serial.stw_pause_max,
                "gc_overhead": comp.serial.gc_overhead_mean,
                "total_duration_sec": comp.serial.total_duration,
                "total_collections": comp.serial.total_collections,
            },
            "parallel": {
                "throughput_mean": comp.parallel.throughput_mean,
                "throughput_stdev": comp.parallel.throughput_stdev,
                "stw_pause_mean": comp.parallel.stw_pause_mean,
                "stw_pause_stdev": comp.parallel.stw_pause_stdev,
                "stw_pause_max": comp.parallel.stw_pause_max,
                "gc_overhead": comp.parallel.gc_overhead_mean,
                "total_duration_sec": comp.parallel.total_duration,
                "total_collections": comp.parallel.total_collections,
            },
            "throughput_change_pct": (comp.speedup - 1) * 100,
            "stw_change_pct": stw_change,
        }

    def collection_to_dict(coll: CollectionResult) -> Dict:
        return {
            "heap_type": coll.heap_type,
            "heap_size": coll.heap_size,
            "serial_time_ms": coll.serial_time_ms,
            "serial_stdev": coll.serial_stdev,
            "parallel_time_ms": coll.parallel_time_ms,
            "parallel_stdev": coll.parallel_stdev,
            "speedup": coll.speedup,
            "num_runs": coll.num_runs,
        }

    synthetic_dict = {}
    for heap_type, comp in result.synthetic_by_heap.items():
        synthetic_dict[heap_type] = comparison_to_dict(comp)

    collection_list = [collection_to_dict(c) for c in result.collection_results]

    # Geometric mean of collection speedups
    gm_collection = 1.0
    if result.collection_results:
        product = 1.0
        for coll in result.collection_results:
            product *= coll.speedup
        gm_collection = product ** (1.0 / len(result.collection_results))

    data = {
        "configuration": {
            "build_type": result.build_type,
            "parallel_gc_available": result.parallel_gc_available,
            "num_workers": result.num_workers,
            "num_threads": result.num_threads,
            "duration_per_benchmark": result.duration_per_benchmark,
            "num_runs": result.num_runs,
            "heap_size": result.heap_size,
        },
        "timestamp": result.timestamp,
        "realistic": comparison_to_dict(result.realistic),
        "collection_results": collection_list,
        "synthetic_by_heap": synthetic_dict,
        "summary": {
            "realistic_throughput_change_pct": (result.realistic.speedup - 1) * 100 if result.realistic else 0,
            "collection_speedup_geomean": gm_collection,
            "synthetic_throughput_geomean_pct": (result.geometric_mean_speedup - 1) * 100,
            "synthetic_stw_geomean_pct": (result.geometric_mean_stw_reduction - 1) * 100,
        },
    }

    return json.dumps(data, indent=2)

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Parallel GC Performance Benchmark Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s                      # Standard suite (~5 min)
    %(prog)s --quick              # Quick sanity check (~1 min)
    %(prog)s --full               # Full suite (~15 min)
    %(prog)s --json -o results.json
    %(prog)s --include-synthetic  # Also run stress tests
"""
    )

    parser.add_argument('--quick', '-q', action='store_true',
                        help='Quick sanity check (~1 min)')
    parser.add_argument('--full', '-f', action='store_true',
                        help='Full benchmark suite (~15 min)')
    parser.add_argument('--duration', '-d', type=float, default=30.0,
                        help='Duration per benchmark in seconds (default: 30)')
    parser.add_argument('--runs', '-r', type=int, default=3,
                        help='Number of runs per configuration (default: 3)')
    parser.add_argument('--threads', '-t', type=int, default=4,
                        help='Number of worker threads (default: 4)')
    parser.add_argument('--workers', '-w', type=int, default=8,
                        help='Number of parallel GC workers (default: 8)')
    parser.add_argument('--heap-size', '-s', type=int, default=500000,
                        help='Heap size for synthetic benchmarks (default: 500000)')
    parser.add_argument('--json', '-j', action='store_true',
                        help='Output as JSON instead of markdown')
    parser.add_argument('--output', '-o', type=str,
                        help='Output file (default: stdout)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Include phase timing details')
    parser.add_argument('--include-synthetic', action='store_true',
                        help='Include synthetic stress test benchmarks')

    args = parser.parse_args()

    # Adjust parameters for quick/full modes
    if args.quick:
        args.duration = 10.0
        args.runs = 2
    elif args.full:
        args.duration = 60.0
        args.runs = 5

    # Run the suite
    result = run_suite(
        duration_per_benchmark=args.duration,
        num_runs=args.runs,
        num_threads=args.threads,
        parallel_workers=args.workers,
        heap_size=args.heap_size,
        verbose=args.verbose,
        include_synthetic=args.include_synthetic
    )

    # Format output
    if args.json:
        output = format_json(result)
    else:
        output = format_markdown(result)

    # Write output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"\nResults saved to: {args.output}")
    else:
        print("\n" + "=" * 60)
        print(output)


if __name__ == '__main__':
    main()
