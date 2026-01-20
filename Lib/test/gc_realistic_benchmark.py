#!/usr/bin/env python3
"""
Realistic Multi-Workload GC Benchmark

A throughput benchmark based on canonical pyperformance benchmarks.
Each worker thread randomly selects from a pool of benchmarks and executes them.

This creates a realistic mix of:
- HIGH_CYCLES workloads (deltablue, deepcopy, pickle_copy, async_tree)
- MINIMAL_CYCLES workloads (richards, nbody)
- NO_CYCLES workloads (comprehensions)

Based on experimental analysis of pyperformance benchmarks showing:
- 23% of benchmarks produce significant cyclic garbage
- 62% produce no cyclic garbage
- Realistic GC load is much lower than 100% cyclic garbage

Usage:
    ./python gc_realistic_benchmark.py --duration 30 --threads 4
    ./python gc_realistic_benchmark.py --duration 30 --threads 8 --parallel 8 --cleanup-workers 4
"""

import gc
import sys
import time
import random
import argparse
import threading
import statistics
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional
import pickle


# =============================================================================
# Workload Implementations (from gc_production_experiment.py)
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
        def __init__(self, value):
            self.value = value
            self.constraints = []
            self.determined_by = None
            self.walk_strength = 0
            self.stay = True
            self.mark = 0

    class Constraint:
        __slots__ = ['strength', 'variables']
        def __init__(self, strength):
            self.strength = strength
            self.variables = []

    class BinaryConstraint(Constraint):
        __slots__ = ['v1', 'v2', 'direction']
        def __init__(self, v1, v2, strength):
            super().__init__(strength)
            self.v1 = v1
            self.v2 = v2
            self.direction = 0
            v1.constraints.append(self)
            v2.constraints.append(self)
            self.variables = [v1, v2]

    variables = [Variable(i) for i in range(100)]
    constraints = []

    for i in range(len(variables) - 1):
        c = BinaryConstraint(variables[i], variables[i + 1], i % 5)
        constraints.append(c)

    for i in range(0, len(variables) - 10, 10):
        c = BinaryConstraint(variables[i], variables[i + 10], 3)
        constraints.append(c)

    for c in constraints:
        c.v1.value = c.v2.value + 1


def workload_nbody() -> None:
    """N-body simulation - pure numerical computation."""
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

    bodies = [
        Body(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, SOLAR_MASS),
        Body(4.84, -1.16, -0.10, 0.001 * DAYS_PER_YEAR, 0.007 * DAYS_PER_YEAR,
             -0.00007 * DAYS_PER_YEAR, 0.0009 * SOLAR_MASS),
        Body(8.34, 4.12, -0.40, -0.002 * DAYS_PER_YEAR, 0.005 * DAYS_PER_YEAR,
             0.00002 * DAYS_PER_YEAR, 0.0002 * SOLAR_MASS),
    ]

    dt = 0.01
    for _ in range(1000):
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


def workload_comprehensions() -> None:
    """Comprehension benchmark - creates acyclic collections."""
    squares = [x * x for x in range(1000)]
    evens = [x for x in squares if x % 2 == 0]
    matrix = [[i * j for j in range(10)] for i in range(10)]
    square_dict = {x: x * x for x in range(100)}
    unique_mods = {x % 17 for x in range(1000)}
    total = sum(x * x for x in range(100))


def workload_deepcopy() -> None:
    """Deep copy benchmark - creates cyclic garbage via parent pointers."""
    class Node:
        __slots__ = ['value', 'children', 'parent']
        def __init__(self, value):
            self.value = value
            self.children = []
            self.parent = None

    root = Node(0)
    current_level = [root]
    for level in range(4):
        next_level = []
        for parent in current_level:
            for i in range(3):
                child = Node(parent.value * 10 + i)
                child.parent = parent
                parent.children.append(child)
                next_level.append(child)
        current_level = next_level

    copy = deepcopy(root)


def workload_pickle_copy() -> None:
    """Pickle round-trip benchmark - IPC pattern with cyclic structures."""
    def make_node(value):
        return {'value': value, 'children': [], 'parent': None}

    root = make_node(0)
    current_level = [root]
    for level in range(4):
        next_level = []
        for parent in current_level:
            for i in range(3):
                child = make_node(parent['value'] * 10 + i)
                child['parent'] = parent
                parent['children'].append(child)
                next_level.append(child)
        current_level = next_level

    serialized = pickle.dumps(root)
    copy = pickle.loads(serialized)


def workload_async_tree() -> None:
    """Async tree benchmark - task trees with parent references."""
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
        task.parent = parent
        if depth > 0:
            for i in range(breadth):
                child = create_task_tree(depth - 1, breadth, task)
                task.children.append(child)
        return task

    root = create_task_tree(depth=4, breadth=3)

    def execute(task):
        task.result = len(task.children)
        for child in task.children:
            execute(child)
        task.done = True

    execute(root)


# =============================================================================
# Workload Pool
# =============================================================================

WORKLOADS: Dict[str, Callable[[], None]] = {
    # HIGH_CYCLES - significant cyclic garbage
    'deltablue': workload_deltablue,
    'deepcopy': workload_deepcopy,
    'pickle_copy': workload_pickle_copy,
    'async_tree': workload_async_tree,
    # MINIMAL_CYCLES - little cyclic garbage
    'richards': workload_richards,
    'nbody': workload_nbody,
    # NO_CYCLES - acyclic garbage only
    'comprehensions': workload_comprehensions,
}

WORKLOAD_NAMES = list(WORKLOADS.keys())


# =============================================================================
# GC Tracking
# =============================================================================

@dataclass
class GCTracker:
    """Track GC activity during benchmark."""
    collections: int = 0
    collected: int = 0
    gc_time_s: float = 0.0
    _gc_start: Optional[float] = None

    stw_pauses_ms: List[float] = field(default_factory=list)

    def gc_callback(self, phase: str, info: dict):
        if phase == "start":
            self._gc_start = time.perf_counter()
        elif phase == "stop":
            if self._gc_start is not None:
                pause_s = time.perf_counter() - self._gc_start
                self.gc_time_s += pause_s
                self.stw_pauses_ms.append(pause_s * 1000)
            self.collections += 1
            self.collected += info.get('collected', 0)

    def reset(self):
        self.collections = 0
        self.collected = 0
        self.gc_time_s = 0.0
        self.stw_pauses_ms.clear()


# =============================================================================
# Worker Thread
# =============================================================================

def worker_loop(
    thread_id: int,
    end_time: float,
    workload_names: List[str],
    results: dict,
    rng_seed: int,
):
    """
    Worker thread that randomly selects and executes workloads.

    Args:
        thread_id: Unique thread identifier
        end_time: When to stop (time.perf_counter() value)
        workload_names: List of workload names to choose from
        results: Shared dict to store results
        rng_seed: Random seed for reproducibility
    """
    rng = random.Random(rng_seed + thread_id)
    workload_counts = {name: 0 for name in workload_names}
    total_workloads = 0

    while time.perf_counter() < end_time:
        # Randomly select a workload
        name = rng.choice(workload_names)
        workload_fn = WORKLOADS[name]

        # Execute workload
        workload_fn()

        workload_counts[name] += 1
        total_workloads += 1

    results[thread_id] = {
        'total_workloads': total_workloads,
        'workload_counts': workload_counts,
    }


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_benchmark(
    duration_seconds: float,
    num_threads: int,
    parallel_workers: Optional[int] = None,
    cleanup_workers: int = 0,
    workload_names: Optional[List[str]] = None,
    seed: int = 42,
) -> dict:
    """
    Run the multi-workload benchmark.

    Args:
        duration_seconds: How long to run
        num_threads: Number of worker threads
        parallel_workers: Parallel GC workers (None for serial GC)
        cleanup_workers: Cleanup workers (0=serial, N=async parallel)
        workload_names: Workloads to use (default: all)
        seed: Random seed
    """
    if workload_names is None:
        workload_names = WORKLOAD_NAMES

    # Validate workloads
    for name in workload_names:
        if name not in WORKLOADS:
            raise ValueError(f"Unknown workload: {name}")

    # Set up GC tracking
    tracker = GCTracker()
    gc.callbacks.append(tracker.gc_callback)

    try:
        # Configure GC
        gc.enable()
        gc.set_threshold(700, 10, 10)

        mode = "serial"
        if parallel_workers is not None:
            gc.enable_parallel(num_workers=parallel_workers)
            mode = f"parallel-{parallel_workers}"
            if cleanup_workers > 0:
                gc.set_cleanup_workers(cleanup_workers)
                mode += f"-cw{cleanup_workers}"

        # Force initial collection
        gc.collect()
        gc.collect()
        tracker.reset()

        print(f"Starting benchmark: {num_threads} threads, {duration_seconds}s, {mode}")
        print(f"Workloads: {', '.join(workload_names)}")

        # Start worker threads
        results = {}
        threads = []
        start_time = time.perf_counter()
        end_time = start_time + duration_seconds

        for i in range(num_threads):
            t = threading.Thread(
                target=worker_loop,
                args=(i, end_time, workload_names, results, seed),
            )
            t.start()
            threads.append(t)

        # Wait for completion
        for t in threads:
            t.join()

        elapsed = time.perf_counter() - start_time

        # Final collection
        gc.collect()

        # Aggregate results
        total_workloads = sum(r['total_workloads'] for r in results.values())
        workload_totals = {name: 0 for name in workload_names}
        for r in results.values():
            for name, count in r['workload_counts'].items():
                workload_totals[name] += count

        # Pause statistics
        stw_pauses = tracker.stw_pauses_ms
        if stw_pauses:
            max_pause = max(stw_pauses)
            mean_pause = statistics.mean(stw_pauses)
            p99_idx = int(len(stw_pauses) * 0.99)
            p99_pause = sorted(stw_pauses)[min(p99_idx, len(stw_pauses) - 1)]
        else:
            max_pause = mean_pause = p99_pause = 0.0

        return {
            'mode': mode,
            'duration_s': elapsed,
            'threads': num_threads,
            'total_workloads': total_workloads,
            'workloads_per_second': total_workloads / elapsed,
            'workload_distribution': workload_totals,
            'gc_collections': tracker.collections,
            'gc_collected': tracker.collected,
            'gc_time_s': tracker.gc_time_s,
            'gc_overhead_percent': (tracker.gc_time_s / elapsed) * 100,
            'stw_pause_count': len(stw_pauses),
            'stw_pause_max_ms': max_pause,
            'stw_pause_mean_ms': mean_pause,
            'stw_pause_p99_ms': p99_pause,
        }

    finally:
        gc.callbacks.remove(tracker.gc_callback)
        try:
            gc.set_cleanup_workers(0)
        except (AttributeError, RuntimeError):
            pass
        try:
            gc.disable_parallel()
        except (RuntimeError, AttributeError):
            pass


def print_results(results: dict) -> None:
    """Print formatted results."""
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Mode: {results['mode']}")
    print(f"Duration: {results['duration_s']:.2f}s")
    print(f"Threads: {results['threads']}")
    print()
    print(f"Total workloads completed: {results['total_workloads']:,}")
    print(f"Throughput: {results['workloads_per_second']:,.1f} workloads/sec")
    print()
    print("Workload distribution:")
    for name, count in sorted(results['workload_distribution'].items()):
        pct = (count / results['total_workloads']) * 100 if results['total_workloads'] > 0 else 0
        print(f"  {name}: {count:,} ({pct:.1f}%)")
    print()
    print("GC Statistics:")
    print(f"  Collections: {results['gc_collections']}")
    print(f"  Objects collected: {results['gc_collected']:,}")
    print(f"  GC time: {results['gc_time_s']:.3f}s ({results['gc_overhead_percent']:.1f}%)")
    print()
    print("STW Pauses:")
    print(f"  Count: {results['stw_pause_count']}")
    print(f"  Max: {results['stw_pause_max_ms']:.2f}ms")
    print(f"  Mean: {results['stw_pause_mean_ms']:.2f}ms")
    print(f"  P99: {results['stw_pause_p99_ms']:.2f}ms")


def main():
    parser = argparse.ArgumentParser(description="Realistic Multi-Workload GC Benchmark")
    parser.add_argument('--duration', '-d', type=float, default=30.0,
                        help="Benchmark duration in seconds (default: 30)")
    parser.add_argument('--threads', '-t', type=int, default=4,
                        help="Number of worker threads (default: 4)")
    parser.add_argument('--parallel', '-p', type=int, default=None,
                        help="Parallel GC workers (omit for serial GC)")
    parser.add_argument('--cleanup-workers', '-c', type=int, default=0,
                        help="Cleanup workers (0=serial, N=async parallel)")
    parser.add_argument('--workloads', '-w', nargs='+', default=None,
                        help=f"Workloads to use (default: all). Available: {WORKLOAD_NAMES}")
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument('--compare', action='store_true',
                        help="Compare cw0 vs cw4")
    args = parser.parse_args()

    # Check for FTP build
    try:
        ftp = hasattr(sys, '_is_gil_enabled') and not sys._is_gil_enabled()
    except:
        ftp = False

    print("=" * 70)
    print("Realistic Multi-Workload GC Benchmark")
    print("=" * 70)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Free-threading: {ftp}")
    print()

    if args.compare and args.parallel is not None:
        # Compare cw0 vs cw4
        print("Running comparison: cw0 vs cw4")
        print()

        print("--- cw0 (serial cleanup) ---")
        results_cw0 = run_benchmark(
            duration_seconds=args.duration,
            num_threads=args.threads,
            parallel_workers=args.parallel,
            cleanup_workers=0,
            workload_names=args.workloads,
            seed=args.seed,
        )
        print_results(results_cw0)

        print()
        print("--- cw4 (parallel cleanup) ---")
        results_cw4 = run_benchmark(
            duration_seconds=args.duration,
            num_threads=args.threads,
            parallel_workers=args.parallel,
            cleanup_workers=4,
            workload_names=args.workloads,
            seed=args.seed,
        )
        print_results(results_cw4)

        print()
        print("=" * 70)
        print("COMPARISON")
        print("=" * 70)
        speedup = results_cw4['workloads_per_second'] / results_cw0['workloads_per_second']
        print(f"Throughput: cw0={results_cw0['workloads_per_second']:.1f}, cw4={results_cw4['workloads_per_second']:.1f}")
        print(f"Speedup: {speedup:.2f}x ({(speedup - 1) * 100:+.1f}%)")
        print(f"GC overhead: cw0={results_cw0['gc_overhead_percent']:.1f}%, cw4={results_cw4['gc_overhead_percent']:.1f}%")
        print(f"Max pause: cw0={results_cw0['stw_pause_max_ms']:.1f}ms, cw4={results_cw4['stw_pause_max_ms']:.1f}ms")

    else:
        results = run_benchmark(
            duration_seconds=args.duration,
            num_threads=args.threads,
            parallel_workers=args.parallel,
            cleanup_workers=args.cleanup_workers,
            workload_names=args.workloads,
            seed=args.seed,
        )
        print_results(results)


if __name__ == "__main__":
    main()
