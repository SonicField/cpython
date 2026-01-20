#!/usr/bin/env python3
"""
GC Production Experiment

Measures cyclic garbage production across canonical pyperformance benchmarks.

Purpose:
    Determine which standard benchmarks produce cyclic garbage (requiring GC)
    versus acyclic garbage (freed by refcount). This informs realistic GC
    benchmark design.

Methodology:
    1. Run each benchmark with GC disabled - measure baseline performance
    2. Run each benchmark with GC enabled - count collections and objects
    3. Compare to determine GC's actual role in each workload

Output:
    Rigorous, verifiable data on:
    - Collections triggered per benchmark iteration
    - Objects collected (cyclic garbage) per collection
    - Time spent in GC as percentage of total
    - Memory growth with GC disabled (indicates uncollected cycles)

Usage:
    ./python gc_production_experiment.py
    ./python gc_production_experiment.py --iterations 10 --output results.json
"""

import gc
import sys
import time
import json
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Callable, Any, Optional


# =============================================================================
# GC Instrumentation
# =============================================================================

@dataclass
class GCStats:
    """Statistics from a single benchmark run."""
    benchmark_name: str
    iterations: int

    # Timing
    total_time_s: float = 0.0
    gc_time_s: float = 0.0

    # Collections
    gc_collections: int = 0
    objects_collected: int = 0
    uncollectable: int = 0

    # Per-generation stats
    gen0_collections: int = 0
    gen1_collections: int = 0
    gen2_collections: int = 0

    # Memory (if available)
    peak_memory_kb: int = 0

    # Derived metrics (filled in after run)
    gc_overhead_percent: float = 0.0
    objects_per_collection: float = 0.0
    collections_per_iteration: float = 0.0
    cyclic_garbage_per_iteration: float = 0.0


class GCTracker:
    """Track GC activity during benchmark execution."""

    def __init__(self):
        self.collections = 0
        self.collected = 0
        self.uncollectable = 0
        self.gc_time_s = 0.0
        self._gc_start = None
        self._gen_collections = [0, 0, 0]

    def gc_callback(self, phase: str, info: dict):
        if phase == "start":
            self._gc_start = time.perf_counter()
        elif phase == "stop":
            if self._gc_start is not None:
                self.gc_time_s += time.perf_counter() - self._gc_start
            self.collections += 1
            self.collected += info.get('collected', 0)
            self.uncollectable += info.get('uncollectable', 0)
            gen = info.get('generation', 0)
            if 0 <= gen < 3:
                self._gen_collections[gen] += 1

    def reset(self):
        self.collections = 0
        self.collected = 0
        self.uncollectable = 0
        self.gc_time_s = 0.0
        self._gen_collections = [0, 0, 0]

    def get_stats(self) -> dict:
        return {
            'collections': self.collections,
            'collected': self.collected,
            'uncollectable': self.uncollectable,
            'gc_time_s': self.gc_time_s,
            'gen_collections': self._gen_collections.copy(),
        }


# =============================================================================
# Benchmark Implementations
# =============================================================================
# Simplified versions of canonical pyperformance benchmarks.
# These capture the essential allocation patterns of each benchmark.

def benchmark_richards(iterations: int = 100) -> None:
    """
    Richards benchmark - simulates OS task scheduler.

    Creates linked list structures (Packet chains) and task queues.
    Expected: Creates cyclic structures via task references.
    """
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

    # Simulate richards workload
    for _ in range(iterations):
        tasks = []
        packets = []

        # Create task chain
        for i in range(6):
            rec = TaskRec()
            t = Task(i, i * 10, None, rec)
            if tasks:
                t.link = tasks[-1]  # Creates chain
            tasks.append(t)

            # Create packet chains
            for j in range(3):
                pkt = Packet(None, i, j)
                if packets:
                    pkt.link = packets[-1]
                packets.append(pkt)

        # Simulate work: shuffle packets between tasks
        for _ in range(100):
            for t in tasks:
                if packets:
                    pkt = packets.pop()
                    old_input = t.input
                    t.input = pkt
                    pkt.link = old_input

        # Let structures go out of scope - garbage


def benchmark_deltablue(iterations: int = 100) -> None:
    """
    DeltaBlue benchmark - constraint solver.

    Creates constraint graph with bidirectional references.
    Expected: High cyclic garbage from constraint-variable cycles.
    """
    class Variable:
        __slots__ = ['value', 'constraints', 'determined_by', 'walk_strength', 'stay', 'mark']
        def __init__(self, value):
            self.value = value
            self.constraints = []  # References to Constraint objects
            self.determined_by = None
            self.walk_strength = 0
            self.stay = True
            self.mark = 0

    class Constraint:
        __slots__ = ['strength', 'variables']
        def __init__(self, strength):
            self.strength = strength
            self.variables = []  # References to Variable objects

    class BinaryConstraint(Constraint):
        __slots__ = ['v1', 'v2', 'direction']
        def __init__(self, v1, v2, strength):
            super().__init__(strength)
            self.v1 = v1
            self.v2 = v2
            self.direction = 0
            # Create bidirectional references (cycles!)
            v1.constraints.append(self)
            v2.constraints.append(self)
            self.variables = [v1, v2]

    for _ in range(iterations):
        # Create variable chain
        variables = [Variable(i) for i in range(100)]
        constraints = []

        # Create constraint graph with cycles
        for i in range(len(variables) - 1):
            c = BinaryConstraint(variables[i], variables[i + 1], i % 5)
            constraints.append(c)

        # Add cross-links (more cycles)
        for i in range(0, len(variables) - 10, 10):
            c = BinaryConstraint(variables[i], variables[i + 10], 3)
            constraints.append(c)

        # Simulate constraint satisfaction
        for c in constraints:
            c.v1.value = c.v2.value + 1

        # Let go out of scope - cyclic garbage


def benchmark_nbody(iterations: int = 1000) -> None:
    """
    N-body simulation benchmark.

    Simple numerical computation with minimal object allocation.
    Expected: Very little garbage - mostly numeric operations.
    """
    PI = 3.14159265358979323
    SOLAR_MASS = 4 * PI * PI
    DAYS_PER_YEAR = 365.24

    class Body:
        __slots__ = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'mass']
        def __init__(self, x, y, z, vx, vy, vz, mass):
            self.x = x
            self.y = y
            self.z = z
            self.vx = vx
            self.vy = vy
            self.vz = vz
            self.mass = mass

    # Create solar system
    bodies = [
        Body(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, SOLAR_MASS),  # Sun
        Body(4.84, -1.16, -0.10, 0.001 * DAYS_PER_YEAR, 0.007 * DAYS_PER_YEAR,
             -0.00007 * DAYS_PER_YEAR, 0.0009 * SOLAR_MASS),  # Jupiter
        Body(8.34, 4.12, -0.40, -0.002 * DAYS_PER_YEAR, 0.005 * DAYS_PER_YEAR,
             0.00002 * DAYS_PER_YEAR, 0.0002 * SOLAR_MASS),  # Saturn
    ]

    dt = 0.01
    for _ in range(iterations):
        # Advance simulation - pure numeric, no allocations
        for i, b1 in enumerate(bodies):
            for b2 in bodies[i + 1:]:
                dx = b1.x - b2.x
                dy = b1.y - b2.y
                dz = b1.z - b2.z
                dist = (dx * dx + dy * dy + dz * dz) ** 0.5
                mag = dt / (dist * dist * dist)
                b1.vx -= dx * b2.mass * mag
                b1.vy -= dy * b2.mass * mag
                b1.vz -= dz * b2.mass * mag
                b2.vx += dx * b1.mass * mag
                b2.vy += dy * b1.mass * mag
                b2.vz += dz * b1.mass * mag

        for b in bodies:
            b.x += dt * b.vx
            b.y += dt * b.vy
            b.z += dt * b.vz


def benchmark_json_loads(iterations: int = 100) -> None:
    """
    JSON parsing benchmark.

    Creates nested dict/list structures from parsing.
    Expected: Acyclic garbage (JSON can't represent cycles).
    """
    import json

    # Sample JSON with nested structures
    sample = json.dumps({
        "users": [
            {"id": i, "name": f"user_{i}", "email": f"user{i}@example.com",
             "preferences": {"theme": "dark", "notifications": True},
             "history": [{"action": "login", "time": j} for j in range(10)]}
            for i in range(50)
        ],
        "metadata": {"version": "1.0", "count": 50}
    })

    for _ in range(iterations):
        data = json.loads(sample)
        # Process data slightly
        total = sum(u["id"] for u in data["users"])
        # Let data go out of scope


def benchmark_float(iterations: int = 10000) -> None:
    """
    Floating point benchmark.

    Pure numeric computation.
    Expected: No garbage at all.
    """
    import math

    for _ in range(iterations):
        x = 1.0
        for i in range(1, 100):
            x = (x + math.sin(i) * math.cos(i)) / (1 + abs(math.tan(i / 10)))
        result = x


def benchmark_regex(iterations: int = 100) -> None:
    """
    Regular expression benchmark.

    Compiles and matches regex patterns.
    Expected: Some internal state, likely acyclic.
    """
    import re

    patterns = [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        r'^\d{4}-\d{2}-\d{2}$',
        r'https?://[^\s]+',
        r'\b\d{3}-\d{3}-\d{4}\b',
    ]

    texts = [
        "Contact us at test@example.com or support@company.org",
        "Date: 2024-01-15 and 2024-12-31",
        "Visit https://example.com or http://test.org/path",
        "Call 555-123-4567 or 800-555-0199",
    ] * 10

    compiled = [re.compile(p) for p in patterns]

    for _ in range(iterations):
        for pattern in compiled:
            for text in texts:
                matches = pattern.findall(text)


def benchmark_comprehensions(iterations: int = 100) -> None:
    """
    List/dict/set comprehension benchmark.

    Creates many temporary collections.
    Expected: Acyclic garbage (simple collections).
    """
    for _ in range(iterations):
        # List comprehensions
        squares = [x * x for x in range(1000)]
        evens = [x for x in squares if x % 2 == 0]

        # Nested comprehension
        matrix = [[i * j for j in range(10)] for i in range(10)]

        # Dict comprehension
        square_dict = {x: x * x for x in range(100)}

        # Set comprehension
        unique_mods = {x % 17 for x in range(1000)}

        # Generator consumed
        total = sum(x * x for x in range(100))


def benchmark_generators(iterations: int = 100) -> None:
    """
    Generator benchmark.

    Creates generator objects and iterates them.
    Expected: Generator frames may form cycles with locals.
    """
    def fib_gen(n):
        a, b = 0, 1
        for _ in range(n):
            yield a
            a, b = b, a + b

    def filter_gen(source, predicate):
        for item in source:
            if predicate(item):
                yield item

    def map_gen(source, func):
        for item in source:
            yield func(item)

    for _ in range(iterations):
        # Create generator pipeline
        fib = fib_gen(100)
        evens = filter_gen(fib, lambda x: x % 2 == 0)
        squared = map_gen(evens, lambda x: x * x)

        # Consume
        result = list(squared)

        # Nested generators with captured locals
        data = list(range(50))
        nested = (
            sum(inner)
            for inner in ([x * y for y in range(10)] for x in data)
        )
        total = sum(nested)


def benchmark_deepcopy(iterations: int = 50) -> None:
    """
    Deep copy benchmark.

    Copies complex nested structures.
    Expected: Creates acyclic copies (original cycles broken by copy).
    """
    from copy import deepcopy

    # Create structure with internal references
    class Node:
        __slots__ = ['value', 'children', 'parent']
        def __init__(self, value):
            self.value = value
            self.children = []
            self.parent = None

    # Build tree with parent pointers (cycles!)
    root = Node(0)
    current_level = [root]
    for level in range(4):
        next_level = []
        for parent in current_level:
            for i in range(3):
                child = Node(parent.value * 10 + i)
                child.parent = parent  # Cycle!
                parent.children.append(child)
                next_level.append(child)
        current_level = next_level

    for _ in range(iterations):
        copy = deepcopy(root)
        # Let copy go out of scope - cyclic garbage


def benchmark_pickle_copy(iterations: int = 50) -> None:
    """
    Pickle round-trip benchmark.

    Same structure as deepcopy, but uses pickle.dumps/loads to create copy.
    This mirrors real IPC patterns (multiprocessing, distributed computing).
    Expected: Creates cyclic garbage (pickle preserves object identity/cycles).
    """
    import pickle

    # Build tree with parent pointers (cycles!)
    # Using dicts instead of custom class to avoid pickle limitations with local classes
    def make_node(value):
        return {'value': value, 'children': [], 'parent': None}

    root = make_node(0)
    current_level = [root]
    for level in range(4):
        next_level = []
        for parent in current_level:
            for i in range(3):
                child = make_node(parent['value'] * 10 + i)
                child['parent'] = parent  # Cycle!
                parent['children'].append(child)
                next_level.append(child)
        current_level = next_level

    for _ in range(iterations):
        # Serialize and deserialize - same as IPC pattern
        serialized = pickle.dumps(root)
        copy = pickle.loads(serialized)
        # Let copy go out of scope - cyclic garbage


def benchmark_async_tree(iterations: int = 50) -> None:
    """
    Async tree benchmark (simplified - no actual async).

    Simulates async task tree structure.
    Expected: Task objects may reference each other cyclically.
    """
    class Task:
        __slots__ = ['name', 'children', 'parent', 'result', 'done']
        def __init__(self, name):
            self.name = name
            self.children = []
            self.parent = None
            self.result = None
            self.done = False

    def create_task_tree(depth, breadth, parent=None):
        task = Task(f"task_{depth}_{id(parent)}")
        task.parent = parent  # Creates cycle with parent
        if depth > 0:
            for i in range(breadth):
                child = create_task_tree(depth - 1, breadth, task)
                task.children.append(child)
        return task

    for _ in range(iterations):
        root = create_task_tree(depth=4, breadth=3)

        # Simulate execution
        def execute(task):
            task.result = len(task.children)
            for child in task.children:
                execute(child)
            task.done = True

        execute(root)
        # Let tree go out of scope - cyclic garbage


def benchmark_pathlib(iterations: int = 100) -> None:
    """
    Pathlib benchmark.

    Path manipulation operations.
    Expected: Path objects are simple, likely acyclic.
    """
    from pathlib import PurePath

    base_paths = [
        "/home/user/documents",
        "/var/log/application",
        "/usr/local/bin",
        "/etc/config/app",
    ]

    for _ in range(iterations):
        paths = []
        for base in base_paths:
            p = PurePath(base)
            for i in range(20):
                p = p / f"subdir_{i}" / f"file_{i}.txt"
                paths.append(p)

        # Path operations
        for p in paths:
            _ = p.parent
            _ = p.name
            _ = p.suffix
            _ = p.parts


def benchmark_logging(iterations: int = 100) -> None:
    """
    Logging benchmark.

    Logging operations with formatting.
    Expected: LogRecord objects and handlers may have references.
    """
    import logging
    import io

    # Create logger with null handler
    logger = logging.getLogger('benchmark')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(io.StringIO())
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    for _ in range(iterations):
        for i in range(100):
            logger.debug("Debug message %d with value %s", i, "test")
            logger.info("Info message %d", i)
            if i % 10 == 0:
                logger.warning("Warning at iteration %d", i)

    logger.removeHandler(handler)


def benchmark_pprint(iterations: int = 100) -> None:
    """
    Pretty print benchmark.

    Formats complex structures for printing.
    Expected: Temporary string allocations, acyclic.
    """
    import pprint

    data = {
        "users": [
            {"id": i, "name": f"user_{i}", "nested": {"a": 1, "b": [1, 2, 3]}}
            for i in range(20)
        ],
        "config": {
            "settings": {"theme": "dark", "level": [1, 2, 3, 4, 5] * 10}
        }
    }

    pp = pprint.PrettyPrinter(width=80)

    for _ in range(iterations):
        formatted = pp.pformat(data)
        lines = formatted.split('\n')


# =============================================================================
# Benchmark Registry
# =============================================================================

BENCHMARKS: Dict[str, Callable[[int], None]] = {
    'richards': benchmark_richards,
    'deltablue': benchmark_deltablue,
    'nbody': benchmark_nbody,
    'json_loads': benchmark_json_loads,
    'float': benchmark_float,
    'regex': benchmark_regex,
    'comprehensions': benchmark_comprehensions,
    'generators': benchmark_generators,
    'deepcopy': benchmark_deepcopy,
    'pickle_copy': benchmark_pickle_copy,
    'async_tree': benchmark_async_tree,
    'pathlib': benchmark_pathlib,
    'logging': benchmark_logging,
    'pprint': benchmark_pprint,
}


# =============================================================================
# Experiment Runner
# =============================================================================

def get_memory_usage_kb() -> int:
    """Get current process memory usage in KB."""
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1])
    except:
        pass
    return 0


def run_benchmark_with_gc(
    name: str,
    func: Callable[[int], None],
    iterations: int,
) -> GCStats:
    """Run a benchmark with GC enabled and collect statistics."""

    tracker = GCTracker()
    gc.callbacks.append(tracker.gc_callback)

    try:
        # Enable GC with default thresholds
        gc.enable()
        gc.set_threshold(700, 10, 10)

        # Force collection before benchmark to start clean
        gc.collect()
        gc.collect()
        gc.collect()
        tracker.reset()

        mem_before = get_memory_usage_kb()
        start_time = time.perf_counter()

        # Run benchmark
        func(iterations)

        end_time = time.perf_counter()
        mem_after = get_memory_usage_kb()

        # Final collection to catch remaining garbage
        gc.collect()

        gc_stats = tracker.get_stats()

        stats = GCStats(
            benchmark_name=name,
            iterations=iterations,
            total_time_s=end_time - start_time,
            gc_time_s=gc_stats['gc_time_s'],
            gc_collections=gc_stats['collections'],
            objects_collected=gc_stats['collected'],
            uncollectable=gc_stats['uncollectable'],
            gen0_collections=gc_stats['gen_collections'][0],
            gen1_collections=gc_stats['gen_collections'][1],
            gen2_collections=gc_stats['gen_collections'][2],
            peak_memory_kb=max(mem_before, mem_after),
        )

        # Calculate derived metrics
        if stats.total_time_s > 0:
            stats.gc_overhead_percent = (stats.gc_time_s / stats.total_time_s) * 100
        if stats.gc_collections > 0:
            stats.objects_per_collection = stats.objects_collected / stats.gc_collections
        if iterations > 0:
            stats.collections_per_iteration = stats.gc_collections / iterations
            stats.cyclic_garbage_per_iteration = stats.objects_collected / iterations

        return stats

    finally:
        gc.callbacks.remove(tracker.gc_callback)


def run_benchmark_without_gc(
    name: str,
    func: Callable[[int], None],
    iterations: int,
) -> dict:
    """Run a benchmark with GC disabled to measure memory growth."""

    # Collect everything first
    gc.collect()
    gc.collect()
    gc.collect()

    gc.disable()

    try:
        mem_before = get_memory_usage_kb()
        start_time = time.perf_counter()

        func(iterations)

        end_time = time.perf_counter()
        mem_after = get_memory_usage_kb()

        return {
            'time_s': end_time - start_time,
            'mem_before_kb': mem_before,
            'mem_after_kb': mem_after,
            'mem_growth_kb': mem_after - mem_before,
        }

    finally:
        gc.enable()
        gc.collect()


def run_experiment(
    benchmarks: Optional[List[str]] = None,
    iterations: int = 100,
    warmup_iterations: int = 10,
) -> Dict[str, Any]:
    """Run the full GC production experiment."""

    if benchmarks is None:
        benchmarks = list(BENCHMARKS.keys())

    results = {
        'metadata': {
            'python_version': sys.version,
            'gc_thresholds': gc.get_threshold(),
            'iterations': iterations,
            'warmup_iterations': warmup_iterations,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'benchmarks': {},
    }

    # Check for FTP build
    try:
        ftp = hasattr(sys, '_is_gil_enabled') and not sys._is_gil_enabled()
        results['metadata']['free_threading'] = ftp
    except:
        results['metadata']['free_threading'] = False

    for name in benchmarks:
        if name not in BENCHMARKS:
            print(f"Unknown benchmark: {name}")
            continue

        func = BENCHMARKS[name]
        print(f"\n{'='*60}")
        print(f"Benchmark: {name}")
        print(f"{'='*60}")

        # Warmup
        print(f"  Warming up ({warmup_iterations} iterations)...")
        try:
            func(warmup_iterations)
        except Exception as e:
            print(f"  ERROR during warmup: {e}")
            continue
        gc.collect()

        # Run with GC disabled first (to measure memory growth from cycles)
        print(f"  Running without GC ({iterations} iterations)...")
        try:
            no_gc_result = run_benchmark_without_gc(name, func, iterations)
        except Exception as e:
            print(f"  ERROR: {e}")
            no_gc_result = {'error': str(e)}

        # Run with GC enabled
        print(f"  Running with GC ({iterations} iterations)...")
        try:
            gc_stats = run_benchmark_with_gc(name, func, iterations)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        # Store results
        results['benchmarks'][name] = {
            'with_gc': asdict(gc_stats),
            'without_gc': no_gc_result,
        }

        # Print summary
        print(f"\n  Results:")
        print(f"    Total time: {gc_stats.total_time_s:.3f}s")
        print(f"    GC time: {gc_stats.gc_time_s:.3f}s ({gc_stats.gc_overhead_percent:.1f}%)")
        print(f"    Collections: {gc_stats.gc_collections} "
              f"(gen0={gc_stats.gen0_collections}, gen1={gc_stats.gen1_collections}, gen2={gc_stats.gen2_collections})")
        print(f"    Objects collected: {gc_stats.objects_collected}")
        print(f"    Cyclic garbage/iter: {gc_stats.cyclic_garbage_per_iteration:.1f}")
        print(f"    Memory growth (no GC): {no_gc_result.get('mem_growth_kb', 'N/A')} KB")

        # Classification
        if gc_stats.objects_collected == 0:
            classification = "NO_CYCLES"
        elif gc_stats.cyclic_garbage_per_iteration < 10:
            classification = "MINIMAL_CYCLES"
        elif gc_stats.cyclic_garbage_per_iteration < 100:
            classification = "MODERATE_CYCLES"
        else:
            classification = "HIGH_CYCLES"

        print(f"    Classification: {classification}")
        results['benchmarks'][name]['classification'] = classification

    return results


def print_summary_table(results: Dict[str, Any]) -> None:
    """Print a summary table of all benchmarks."""

    print("\n" + "=" * 80)
    print("SUMMARY: Cyclic Garbage Production by Benchmark")
    print("=" * 80)
    print(f"\n{'Benchmark':<20} {'Cycles/Iter':>12} {'GC Time %':>10} "
          f"{'Collections':>12} {'Classification':<20}")
    print("-" * 80)

    benchmarks = results.get('benchmarks', {})

    # Sort by cyclic garbage production (descending)
    sorted_benchmarks = sorted(
        benchmarks.items(),
        key=lambda x: x[1].get('with_gc', {}).get('cyclic_garbage_per_iteration', 0),
        reverse=True
    )

    for name, data in sorted_benchmarks:
        gc_data = data.get('with_gc', {})
        cycles_per_iter = gc_data.get('cyclic_garbage_per_iteration', 0)
        gc_overhead = gc_data.get('gc_overhead_percent', 0)
        collections = gc_data.get('gc_collections', 0)
        classification = data.get('classification', 'UNKNOWN')

        print(f"{name:<20} {cycles_per_iter:>12.1f} {gc_overhead:>10.1f}% "
              f"{collections:>12} {classification:<20}")

    print("-" * 80)

    # Statistics
    high_cycle = sum(1 for _, d in benchmarks.items() if d.get('classification') == 'HIGH_CYCLES')
    moderate_cycle = sum(1 for _, d in benchmarks.items() if d.get('classification') == 'MODERATE_CYCLES')
    minimal_cycle = sum(1 for _, d in benchmarks.items() if d.get('classification') == 'MINIMAL_CYCLES')
    no_cycle = sum(1 for _, d in benchmarks.items() if d.get('classification') == 'NO_CYCLES')

    print(f"\nClassification Summary:")
    print(f"  HIGH_CYCLES:     {high_cycle} benchmarks (significant GC load)")
    print(f"  MODERATE_CYCLES: {moderate_cycle} benchmarks")
    print(f"  MINIMAL_CYCLES:  {minimal_cycle} benchmarks")
    print(f"  NO_CYCLES:       {no_cycle} benchmarks (acyclic only)")


def main():
    parser = argparse.ArgumentParser(description="GC Production Experiment")
    parser.add_argument('--benchmarks', '-b', nargs='+', default=None,
                        help=f"Benchmarks to run (default: all). Available: {list(BENCHMARKS.keys())}")
    parser.add_argument('--iterations', '-i', type=int, default=100,
                        help="Iterations per benchmark (default: 100)")
    parser.add_argument('--warmup', '-w', type=int, default=10,
                        help="Warmup iterations (default: 10)")
    parser.add_argument('--output', '-o', type=str, default=None,
                        help="Output JSON file for results")
    parser.add_argument('--list', '-l', action='store_true',
                        help="List available benchmarks and exit")
    args = parser.parse_args()

    if args.list:
        print("Available benchmarks:")
        for name in sorted(BENCHMARKS.keys()):
            print(f"  {name}")
        return

    print("=" * 80)
    print("GC Production Experiment")
    print("=" * 80)
    print(f"Python: {sys.version}")
    print(f"Iterations: {args.iterations}")
    print(f"Warmup: {args.warmup}")

    results = run_experiment(
        benchmarks=args.benchmarks,
        iterations=args.iterations,
        warmup_iterations=args.warmup,
    )

    print_summary_table(results)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
