#!/usr/bin/env python3
"""
Parallel GC Throughput Benchmark

Measures real-world throughput impact of parallel GC by:
1. Building an ai_workload style heap to steady-state size
2. Running continuous create/destroy at steady state for duration Y
3. Counting total objects created (higher = better throughput)
4. Tracking ACTUAL STW pause times from phase timing (not total GC time)

Usage:
    python gc_throughput_benchmark.py --heap-size 500k --duration 30
    python gc_throughput_benchmark.py --heap-size 500k --duration 30 --parallel 8
    python gc_throughput_benchmark.py --heap-size 500k --duration 30 --threads 4 --keep-threads
"""

import gc
import sys
import time
import queue
import random
import argparse
import threading
import statistics
from dataclasses import dataclass, field
from typing import List, Optional, Dict

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

BUILD_TYPE = detect_build()

# =============================================================================
# STW Pause Calculation from Phase Timing
# =============================================================================

# Phases that occur during STW #0 (before first StartTheWorld)
STW0_PHASES = [
    'stw0_ns',           # Initial STW overhead
    'async_wait_ns',     # Wait for async operations
    'merge_refs_ns',     # Merge reference lists
    'delayed_frees_ns',  # Process delayed frees
    'mark_alive_ns',     # Mark alive objects (parallel)
    'bucket_assign_ns',  # Assign work buckets
    'update_refs_ns',    # Update references (parallel)
    'mark_heap_ns',      # Mark heap (parallel)
    'scan_heap_ns',      # Scan heap (parallel)
    'disable_deferred_ns',  # Disable deferred operations
    'find_weakrefs_ns',  # Find weak references
]

# Phases that occur during STW #2 (second stop-the-world)
STW2_PHASES = [
    'resurrection_ns',   # Handle resurrection
    'freelists_ns',      # Process freelists
    'clear_weakrefs_ns', # Clear weak references
]

# Concurrent phases (mutators running)
CONCURRENT_PHASES = [
    'objs_decref_ns',        # Decref objects
    'weakref_callbacks_ns',  # Weakref callbacks
    'finalize_ns',           # Run finalizers
    'cleanup_ns',            # Deallocation (concurrent with mutators)
]


def calculate_stw_pause_ms(phase_timing: Dict[str, int]) -> float:
    """
    Calculate actual STW pause time from phase timing.

    STW pause = STW #0 phases + STW #2 phases
    Excludes concurrent phases (objs_decref, weakref_callbacks, finalize, cleanup)
    """
    stw0_ns = sum(phase_timing.get(p, 0) for p in STW0_PHASES)
    stw2_ns = sum(phase_timing.get(p, 0) for p in STW2_PHASES)
    return (stw0_ns + stw2_ns) / 1e6  # Convert to ms


def calculate_concurrent_time_ms(phase_timing: Dict[str, int]) -> float:
    """Calculate time spent in concurrent phases (mutators running)."""
    concurrent_ns = sum(phase_timing.get(p, 0) for p in CONCURRENT_PHASES)
    return concurrent_ns / 1e6


def calculate_total_gc_time_ms(phase_timing: Dict[str, int]) -> float:
    """Calculate total GC time from phase timing."""
    total_ns = phase_timing.get('total_ns', 0)
    if total_ns > 0:
        return total_ns / 1e6
    # Fallback: sum all phases
    all_phases = STW0_PHASES + STW2_PHASES + CONCURRENT_PHASES
    return sum(phase_timing.get(p, 0) for p in all_phases) / 1e6


# =============================================================================
# Pause Tracking via gc.callbacks
# =============================================================================

@dataclass
class PauseTracker:
    """
    Tracks GC timing using gc.callbacks and phase timing.

    Note: gc.callbacks measures total GC time (start to stop), which includes
    concurrent phases. For actual STW pause time, use phase timing from
    gc.get_parallel_stats() (only valid when parallel GC is enabled).
    """
    # Total GC times (from gc.callbacks - includes concurrent phases)
    gc_times_ms: List[float] = field(default_factory=list)
    # Actual STW pause times (from phase timing)
    stw_pauses_ms: List[float] = field(default_factory=list)
    # Concurrent times (from phase timing)
    concurrent_times_ms: List[float] = field(default_factory=list)
    # Object counts
    collected_counts: List[int] = field(default_factory=list)

    gc_start_time: Optional[float] = None
    total_collections: int = 0
    # Whether parallel GC is enabled (phase timing only valid when True)
    parallel_enabled: bool = False

    def gc_callback(self, phase: str, info: dict):
        """Called by gc module at start/stop of each collection."""
        if phase == "start":
            self.gc_start_time = time.perf_counter()
        elif phase == "stop":
            if self.gc_start_time is not None:
                gc_time_ms = (time.perf_counter() - self.gc_start_time) * 1000
                self.gc_times_ms.append(gc_time_ms)
                self.total_collections += 1

                # Track collected count from info dict
                collected = info.get('collected', 0)
                self.collected_counts.append(collected)

                # Try to get phase timing for accurate STW pause
                # Only valid when parallel GC is enabled
                if self.parallel_enabled:
                    try:
                        stats = gc.get_parallel_stats()
                        phase_timing = stats.get('phase_timing', {})
                        if phase_timing:
                            stw_pause = calculate_stw_pause_ms(phase_timing)
                            concurrent = calculate_concurrent_time_ms(phase_timing)
                            # Only record valid (non-negative) values
                            if stw_pause >= 0:
                                self.stw_pauses_ms.append(stw_pause)
                            if concurrent >= 0:
                                self.concurrent_times_ms.append(concurrent)
                    except (AttributeError, TypeError):
                        pass

                self.gc_start_time = None

    def reset(self):
        """Reset tracking for a new measurement period."""
        self.gc_times_ms.clear()
        self.stw_pauses_ms.clear()
        self.concurrent_times_ms.clear()
        self.collected_counts.clear()
        self.gc_start_time = None
        self.total_collections = 0
        # Note: parallel_enabled is NOT reset - it should be set explicitly

    def gc_time_summary(self) -> dict:
        """Return total GC time statistics (includes concurrent phases)."""
        return self._summarise_times(self.gc_times_ms)

    def stw_pause_summary(self) -> dict:
        """Return actual STW pause statistics."""
        return self._summarise_times(self.stw_pauses_ms)

    def concurrent_summary(self) -> dict:
        """Return concurrent phase time statistics."""
        return self._summarise_times(self.concurrent_times_ms)

    def _summarise_times(self, times: List[float]) -> dict:
        """Summarise a list of times."""
        if not times:
            return {
                "count": 0,
                "total_ms": 0,
                "mean_ms": 0,
                "max_ms": 0,
                "p99_ms": 0,
                "p95_ms": 0,
            }

        sorted_times = sorted(times)
        p99_idx = int(len(sorted_times) * 0.99)
        p95_idx = int(len(sorted_times) * 0.95)

        return {
            "count": len(times),
            "total_ms": sum(times),
            "mean_ms": statistics.mean(times),
            "max_ms": max(times),
            "p99_ms": sorted_times[min(p99_idx, len(sorted_times) - 1)],
            "p95_ms": sorted_times[min(p95_idx, len(sorted_times) - 1)],
        }


# =============================================================================
# AI Workload Heap (from gc_benchmark.py)
# =============================================================================

class Node:
    """Generic node for building object graphs."""
    __slots__ = ['refs', 'data', '__weakref__']

    def __init__(self):
        self.refs = []
        self.data = None


class TensorMock:
    """Mock tensor with shape and data references."""
    __slots__ = ['shape', 'data', 'grad', 'requires_grad']

    def __init__(self, shape):
        self.shape = shape
        self.data = [0.0] * min(shape[0] if shape else 1, 10)
        self.grad = None
        self.requires_grad = False


class LayerMock:
    """Mock neural network layer."""
    __slots__ = ['weights', 'bias', 'input_cache', 'output_cache', 'name']

    def __init__(self, name, in_features, out_features):
        self.name = name
        self.weights = TensorMock((in_features, out_features))
        self.bias = TensorMock((out_features,))
        self.input_cache = None
        self.output_cache = None


class AttentionMock:
    """Mock attention mechanism with Q, K, V projections."""
    __slots__ = ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'attention_weights']

    def __init__(self, hidden_size, num_heads):
        head_dim = hidden_size // num_heads
        self.q_proj = LayerMock("q_proj", hidden_size, hidden_size)
        self.k_proj = LayerMock("k_proj", hidden_size, hidden_size)
        self.v_proj = LayerMock("v_proj", hidden_size, hidden_size)
        self.out_proj = LayerMock("out_proj", hidden_size, hidden_size)
        self.attention_weights = None


class TransformerBlockMock:
    """Mock transformer block."""
    __slots__ = ['attention', 'mlp_up', 'mlp_down', 'ln1', 'ln2', 'residual']

    def __init__(self, hidden_size, num_heads):
        self.attention = AttentionMock(hidden_size, num_heads)
        self.mlp_up = LayerMock("mlp_up", hidden_size, hidden_size * 4)
        self.mlp_down = LayerMock("mlp_down", hidden_size * 4, hidden_size)
        self.ln1 = LayerMock("ln1", hidden_size, hidden_size)
        self.ln2 = LayerMock("ln2", hidden_size, hidden_size)
        self.residual = None


class KVCacheMock:
    """Mock KV cache for inference."""
    __slots__ = ['keys', 'values', 'seq_len']

    def __init__(self, batch_size, num_heads, seq_len, head_dim):
        self.keys = [TensorMock((batch_size, num_heads, seq_len, head_dim))]
        self.values = [TensorMock((batch_size, num_heads, seq_len, head_dim))]
        self.seq_len = seq_len


def create_ai_workload(size: int) -> List:
    """
    Create AI/ML workload mimicking transformer inference.
    Returns list of objects for heap retention.
    """
    objects = []

    # Model configuration (scaled by size)
    num_layers = max(1, size // 50000)
    hidden_size = 256
    num_heads = 8
    batch_size = max(1, size // 100000)
    seq_len = 512

    # Create transformer layers
    layers = []
    for i in range(num_layers):
        block = TransformerBlockMock(hidden_size, num_heads)
        layers.append(block)
        objects.append(block)

    # Create KV caches (simulating inference batches)
    num_caches = max(1, size // 10000)
    caches = []
    for _ in range(num_caches):
        cache = KVCacheMock(batch_size, num_heads, seq_len, hidden_size // num_heads)
        caches.append(cache)
        objects.append(cache)

    # Create intermediate activations
    num_activations = size - len(objects)
    for i in range(max(0, num_activations)):
        if i % 5 == 0:
            # Tensor
            obj = TensorMock((batch_size, seq_len, hidden_size))
        elif i % 5 == 1:
            # Attention weights reference
            obj = Node()
            if layers:
                obj.refs.append(random.choice(layers).attention)
        elif i % 5 == 2:
            # Cache reference
            obj = Node()
            if caches:
                obj.refs.append(random.choice(caches))
        elif i % 5 == 3:
            # Layer reference
            obj = Node()
            if layers:
                obj.refs.append(random.choice(layers))
        else:
            # Cross-reference between activations
            obj = Node()
            if objects:
                obj.refs.append(random.choice(objects))
        objects.append(obj)

    return objects


# =============================================================================
# Heap Creation with Cycles
# =============================================================================


def create_cyclic_heap(size: int, cluster_size: int = 100) -> List:
    """
    Create a heap with isolated cyclic clusters.

    Returns a flat list of objects where clusters form internal cycles.
    This ensures the heap has both live objects and potential for cyclic garbage.
    """
    all_objects = []
    num_clusters = max(1, size // cluster_size)

    for _ in range(num_clusters):
        # Create cluster of nodes
        cluster = [Node() for _ in range(cluster_size)]

        # Form cycle: each node refs next, last refs first
        for i in range(cluster_size):
            cluster[i].refs.append(cluster[(i + 1) % cluster_size])

        all_objects.extend(cluster)

    return all_objects


# =============================================================================
# Creation Thread Pool (persistent threads, no abandoned pool)
# =============================================================================

HEAP_GENERATORS = {
    "cyclic": create_cyclic_heap,
    "ai": create_ai_workload,
}


class CreationThreadPool:
    """
    Thread pool for object creation that keeps threads alive.

    Unlike ThreadPoolExecutor which may exit threads, this pool keeps
    threads alive so pages remain in live thread heaps rather than
    going to the abandoned pool.
    """

    def __init__(self, num_threads: int):
        self.num_threads = num_threads
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
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
            except queue.Empty:
                continue

            if task is None:  # Shutdown signal
                break

            task_type, *args = task

            if task_type == "create":
                heap_type, num_objects = args
                clusters = HEAP_GENERATORS[heap_type](num_objects)
                self.result_queue.put(clusters)
                # Release reference immediately to avoid retaining garbage
                clusters = None
                del clusters

            elif task_type == "churn":
                # Churn task: run churn loop for duration
                heap_size, churn_size, end_time, heap_type, result_dict = args
                self._run_churn_loop(thread_id, heap_size, churn_size,
                                     end_time, heap_type, result_dict)

            self.task_queue.task_done()

    def _run_churn_loop(self, thread_id: int, heap_size: int, churn_size: int,
                        end_time: float, heap_type: str, result_dict: dict):
        """Run churn loop within persistent thread."""
        rng = random.Random(42 + thread_id)

        # Build thread-local heap
        heap = HEAP_GENERATORS[heap_type](heap_size)

        objects_created = 0
        cycles = 0

        while time.perf_counter() < end_time:
            # Create batch of new objects that form cycles
            new_objects = [Node() for _ in range(churn_size)]

            # Create cycles within the batch
            for i in range(len(new_objects)):
                new_objects[i].refs.append(new_objects[(i + 1) % len(new_objects)])

            # Add some refs to our local heap
            for i, obj in enumerate(new_objects):
                if heap and i % 5 == 0:
                    obj.refs.append(rng.choice(heap))

            # Add to heap temporarily, then remove to create garbage
            heap.extend(new_objects)

            # Remove some objects to create garbage
            num_to_remove = min(churn_size, len(heap) - heap_size)
            if num_to_remove > 0:
                del heap[-num_to_remove:]

            objects_created += churn_size
            cycles += 1

        # Store results
        result_dict[thread_id] = {"objects_created": objects_created, "cycles": cycles}

    def create_objects(self, heap_type: str, total_objects: int) -> List:
        """Create objects using the thread pool."""
        objects_per_thread = total_objects // self.num_threads

        # Submit tasks
        for _ in range(self.num_threads):
            self.task_queue.put(("create", heap_type, objects_per_thread))

        # Wait for completion
        self.task_queue.join()

        # Collect results
        all_clusters = []
        while not self.result_queue.empty():
            clusters = self.result_queue.get()
            all_clusters.extend(clusters)

        return all_clusters

    def run_churn(self, heap_size_per_thread: int, churn_size_per_thread: int,
                  end_time: float, heap_type: str) -> dict:
        """Run churn loop across all threads."""
        result_dict = {}

        # Submit churn tasks
        for _ in range(self.num_threads):
            self.task_queue.put(("churn", heap_size_per_thread, churn_size_per_thread,
                                end_time, heap_type, result_dict))

        # Wait for completion
        self.task_queue.join()

        return result_dict

    def shutdown(self):
        """Shutdown the thread pool."""
        self.shutdown_flag.set()
        # Send shutdown signals
        for _ in range(self.num_threads):
            self.task_queue.put(None)
        for t in self.threads:
            t.join(timeout=1.0)


# Global thread pool (created on demand)
_creation_pool: Optional[CreationThreadPool] = None


def get_creation_pool(num_threads: int) -> CreationThreadPool:
    """Get or create the global creation thread pool."""
    global _creation_pool
    if _creation_pool is None or _creation_pool.num_threads != num_threads:
        if _creation_pool is not None:
            _creation_pool.shutdown()
        _creation_pool = CreationThreadPool(num_threads)
    return _creation_pool


def shutdown_creation_pool():
    """Shutdown the global creation thread pool."""
    global _creation_pool
    if _creation_pool is not None:
        _creation_pool.shutdown()
        _creation_pool = None


# =============================================================================
# Throughput Benchmark
# =============================================================================

def run_throughput_benchmark(
    heap_size: int,
    duration_seconds: float,
    parallel_workers: Optional[int],
    num_threads: Optional[int] = None,
    keep_threads: bool = False,
    heap_type: str = "cyclic",
    gc_threshold: int = 500,
    seed: int = 42,
    pre_stw_wait: bool = False,
) -> dict:
    """
    Run throughput benchmark:
    1. Build heap to steady state
    2. Run create/destroy loop for duration
    3. Count objects created
    4. Track ACTUAL STW pause times (not total GC time)

    Args:
        heap_size: Target heap size in objects
        duration_seconds: How long to run the benchmark
        parallel_workers: Number of parallel GC workers (None for serial)
        num_threads: Number of threads for object creation
        keep_threads: If True, keep creation threads alive (no abandoned pool)
        heap_type: "cyclic" or "ai"
        gc_threshold: GC threshold for triggering collections
        seed: Random seed for reproducibility
        pre_stw_wait: If True, wait for cleanup before STW (better for multi-threaded)
    """
    random.seed(seed)

    # Set up pause tracking
    tracker = PauseTracker()
    gc.callbacks.append(tracker.gc_callback)

    try:
        # Configure GC with lower threshold for more frequent collections
        gc.enable()
        gc.set_threshold(gc_threshold, 5, 5)
        if parallel_workers is not None:
            if BUILD_TYPE == "ftp":
                gc.enable_parallel(num_workers=parallel_workers)
                # Set async wait mode (only available in FTP builds)
                try:
                    gc.set_async_wait_stw(not pre_stw_wait)
                except AttributeError:
                    pass
            else:
                gc.enable_parallel(parallel_workers)
            mode = f"parallel-{parallel_workers}"
            if pre_stw_wait:
                mode += "-prestw"
            tracker.parallel_enabled = True
        else:
            try:
                gc.disable_parallel()
            except (RuntimeError, AttributeError):
                pass
            mode = "serial"
            tracker.parallel_enabled = False

        # Add threading info to mode
        if num_threads:
            thread_mode = "pool" if keep_threads else "abandoned"
            mode += f"-{num_threads}t-{thread_mode}"

        print(f"Building initial heap ({heap_size} objects, {heap_type})...")

        # Calculate churn parameters
        churn_size = min(5000, max(1000, heap_size // 100))

        if num_threads:
            heap_size_per_thread = heap_size // num_threads
            churn_size_per_thread = max(100, churn_size // num_threads)

            print(f"Starting steady-state benchmark ({duration_seconds}s, {mode})...")
            print(f"  Total heap size: {heap_size} ({heap_size_per_thread} per thread)")
            print(f"  Churn per cycle: {churn_size} ({churn_size_per_thread} per thread)")
            print(f"  Threads: {num_threads} ({'persistent pool' if keep_threads else 'exit after duration'})")

            tracker.reset()

            start_time = time.perf_counter()
            end_time = start_time + duration_seconds

            if keep_threads:
                # Use persistent thread pool - threads stay alive
                pool = get_creation_pool(num_threads)
                worker_results = pool.run_churn(
                    heap_size_per_thread, churn_size_per_thread,
                    end_time, heap_type
                )
            else:
                # Use regular threading - threads exit, pages go to abandoned pool
                worker_results = {}
                threads = []
                for thread_id in range(num_threads):
                    t = threading.Thread(
                        target=_worker_churn_loop,
                        args=(thread_id, heap_size_per_thread, churn_size_per_thread,
                              end_time, worker_results, heap_type)
                    )
                    t.start()
                    threads.append(t)

                for t in threads:
                    t.join()

            elapsed = time.perf_counter() - start_time

            # Aggregate results from all threads
            objects_created = sum(r["objects_created"] for r in worker_results.values())
            cycles = sum(r["cycles"] for r in worker_results.values())

        else:
            # Single-threaded execution
            gc.disable()  # Don't count setup GC
            heap = HEAP_GENERATORS[heap_type](heap_size)
            gc.collect()  # Clean up any garbage from setup
            gc.enable()

            print(f"Starting steady-state benchmark ({duration_seconds}s, {mode})...")
            print(f"  Heap size: {heap_size}, Churn per cycle: {churn_size}")

            tracker.reset()
            objects_created = 0
            cycles = 0

            start_time = time.perf_counter()
            end_time = start_time + duration_seconds

            while time.perf_counter() < end_time:
                # Create batch of new objects that form cycles
                new_objects = [Node() for _ in range(churn_size)]

                # Create cycles within the batch
                for i in range(len(new_objects)):
                    new_objects[i].refs.append(new_objects[(i + 1) % len(new_objects)])

                # Add some refs to the heap
                for i, obj in enumerate(new_objects):
                    if heap and i % 5 == 0:
                        obj.refs.append(random.choice(heap))

                # Add to heap temporarily, then remove to create garbage
                heap.extend(new_objects)

                # Remove some objects to create garbage
                num_to_remove = min(churn_size, len(heap) - heap_size)
                if num_to_remove > 0:
                    del heap[-num_to_remove:]

                objects_created += churn_size
                cycles += 1

            elapsed = time.perf_counter() - start_time

        # Final collection to ensure consistent state
        gc.collect()

        # Get summaries
        gc_time_summary = tracker.gc_time_summary()
        stw_summary = tracker.stw_pause_summary()
        concurrent_summary = tracker.concurrent_summary()

        # Get phase timing from last collection (only valid in parallel mode)
        phase_timing = {}
        if tracker.parallel_enabled:
            try:
                stats = gc.get_parallel_stats()
                phase_timing = stats.get('phase_timing', {})
            except (AttributeError, TypeError):
                pass

        # Calculate throughput
        throughput = objects_created / elapsed

        # Determine which pause data to use
        # Prefer STW pause if available, fall back to total GC time
        if stw_summary["count"] > 0:
            pause_summary = stw_summary
            pause_type = "stw"
        else:
            pause_summary = gc_time_summary
            pause_type = "total"

        return {
            "mode": mode,
            "heap_size": heap_size,
            "duration_seconds": elapsed,
            "objects_created": objects_created,
            "cycles": cycles,
            "throughput_per_second": throughput,
            # STW pause metrics (actual latency impact)
            "stw_pause_count": stw_summary["count"],
            "stw_pause_total_ms": stw_summary["total_ms"],
            "stw_pause_mean_ms": stw_summary["mean_ms"],
            "stw_pause_max_ms": stw_summary["max_ms"],
            "stw_pause_p99_ms": stw_summary["p99_ms"],
            "stw_pause_p95_ms": stw_summary["p95_ms"],
            # Total GC time metrics (throughput impact)
            "gc_time_count": gc_time_summary["count"],
            "gc_time_total_ms": gc_time_summary["total_ms"],
            "gc_time_mean_ms": gc_time_summary["mean_ms"],
            "gc_time_max_ms": gc_time_summary["max_ms"],
            # Concurrent time (runs while mutators active)
            "concurrent_total_ms": concurrent_summary["total_ms"],
            "concurrent_mean_ms": concurrent_summary["mean_ms"],
            # Overhead calculations
            "stw_overhead_percent": (stw_summary["total_ms"] / (elapsed * 1000)) * 100 if stw_summary["count"] > 0 else 0,
            "gc_overhead_percent": (gc_time_summary["total_ms"] / (elapsed * 1000)) * 100,
            # Phase timing from last collection
            "phase_timing": phase_timing,
            # Which pause type was used
            "pause_type": pause_type,
        }

    finally:
        # Clean up
        gc.callbacks.remove(tracker.gc_callback)
        try:
            gc.disable_parallel()
        except (RuntimeError, AttributeError):
            pass


def _worker_churn_loop(
    thread_id: int,
    heap_size: int,
    churn_size: int,
    end_time: float,
    results: dict,
    heap_type: str,
):
    """
    Worker function for non-persistent threads.
    Threads exit after duration, pages go to abandoned pool.
    """
    rng = random.Random(42 + thread_id)

    # Build thread-local heap
    heap = HEAP_GENERATORS[heap_type](heap_size)

    objects_created = 0
    cycles = 0

    while time.perf_counter() < end_time:
        # Create batch of new objects that form cycles
        new_objects = [Node() for _ in range(churn_size)]

        # Create cycles within the batch
        for i in range(len(new_objects)):
            new_objects[i].refs.append(new_objects[(i + 1) % len(new_objects)])

        # Add some refs to our local heap
        for i, obj in enumerate(new_objects):
            if heap and i % 5 == 0:
                obj.refs.append(rng.choice(heap))

        # Add to heap temporarily, then remove to create garbage
        heap.extend(new_objects)

        # Remove some objects to create garbage
        num_to_remove = min(churn_size, len(heap) - heap_size)
        if num_to_remove > 0:
            del heap[-num_to_remove:]

        objects_created += churn_size
        cycles += 1

    # Store results for this thread
    results[thread_id] = {"objects_created": objects_created, "cycles": cycles}


def format_results(results: dict) -> str:
    """Format results for display."""
    lines = [
        f"Mode: {results['mode']}",
        f"Duration: {results['duration_seconds']:.2f}s",
        f"Objects created: {results['objects_created']:,}",
        f"Throughput: {results['throughput_per_second']:,.0f} objects/sec",
        f"",
    ]

    # STW pause data only available in parallel mode
    if results['stw_pause_count'] > 0:
        lines.extend([
            f"STW Pauses (actual latency impact):",
            f"  Count: {results['stw_pause_count']}",
            f"  Total: {results['stw_pause_total_ms']:.1f}ms",
            f"  Mean: {results['stw_pause_mean_ms']:.2f}ms",
            f"  Max: {results['stw_pause_max_ms']:.2f}ms",
            f"  P99: {results['stw_pause_p99_ms']:.2f}ms",
            f"  P95: {results['stw_pause_p95_ms']:.2f}ms",
            f"  Overhead: {results['stw_overhead_percent']:.2f}%",
            f"",
        ])

    lines.extend([
        f"Total GC Time (throughput impact):",
        f"  Count: {results['gc_time_count']}",
        f"  Total: {results['gc_time_total_ms']:.1f}ms",
        f"  Mean: {results['gc_time_mean_ms']:.2f}ms",
        f"  Max: {results['gc_time_max_ms']:.2f}ms",
    ])

    if results.get('concurrent_total_ms', 0) > 0:
        lines.append(f"  Concurrent: {results['concurrent_total_ms']:.1f}ms total")

    lines.append(f"  Overhead: {results['gc_overhead_percent']:.2f}%")

    # Add phase timing if available (from last collection)
    phase_timing = results.get('phase_timing', {})
    if phase_timing:
        lines.append("")
        lines.append("Phase timing (last collection):")

        # Group phases by category
        stw0_total = sum(phase_timing.get(p, 0) for p in STW0_PHASES) / 1e6
        stw2_total = sum(phase_timing.get(p, 0) for p in STW2_PHASES) / 1e6
        concurrent_total = sum(phase_timing.get(p, 0) for p in CONCURRENT_PHASES) / 1e6

        lines.append(f"  STW #0: {stw0_total:.2f}ms")
        lines.append(f"  STW #2: {stw2_total:.2f}ms")
        lines.append(f"  Concurrent: {concurrent_total:.2f}ms")
        lines.append(f"  Total pause: {stw0_total + stw2_total:.2f}ms")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Parallel GC Throughput Benchmark")
    parser.add_argument("--heap-size", "-s", type=str, default="500k",
                        help="Steady-state heap size (e.g., 100k, 500k, 1M)")
    parser.add_argument("--heap-type", "-t", type=str, default="cyclic",
                        choices=["cyclic", "ai"],
                        help="Heap structure: cyclic (circular chains) or ai (transformer-style)")
    parser.add_argument("--duration", "-d", type=float, default=30.0,
                        help="Benchmark duration in seconds")
    parser.add_argument("--parallel", "-p", type=int, default=None,
                        help="Number of parallel GC workers (omit for serial)")
    parser.add_argument("--compare", "-c", action="store_true",
                        help="Run both serial and parallel, compare results")
    parser.add_argument("--workers", "-w", type=int, default=8,
                        help="Workers to use in compare mode")
    parser.add_argument("--threads", type=int, default=None,
                        help="Number of threads for object creation")
    parser.add_argument("--keep-threads", "-k", action="store_true",
                        help="Keep creation threads alive (no abandoned pool)")
    parser.add_argument("--pre-stw-wait", action="store_true",
                        help="Wait for cleanup before STW (not in STW)")
    args = parser.parse_args()

    # Parse heap size
    size_str = args.heap_size.lower()
    if size_str.endswith("m"):
        heap_size = int(float(size_str[:-1]) * 1_000_000)
    elif size_str.endswith("k"):
        heap_size = int(float(size_str[:-1]) * 1_000)
    else:
        heap_size = int(size_str)

    print(f"=" * 70)
    print(f"Parallel GC Throughput Benchmark - {BUILD_TYPE.upper()} Build")
    print(f"=" * 70)

    # Check if parallel GC is available
    try:
        config = gc.get_parallel_config()
        parallel_available = config.get('available', False)
    except AttributeError:
        parallel_available = False

    print(f"Parallel GC available: {parallel_available}")
    print()

    try:
        if args.compare and parallel_available:
            # Run both serial and parallel
            print("Running serial benchmark...")
            serial_results = run_throughput_benchmark(
                heap_size=heap_size,
                duration_seconds=args.duration,
                parallel_workers=None,
                num_threads=args.threads,
                keep_threads=args.keep_threads,
                heap_type=args.heap_type,
            )
            print(format_results(serial_results))
            print()

            print(f"Running parallel benchmark ({args.workers} workers)...")
            parallel_results = run_throughput_benchmark(
                heap_size=heap_size,
                duration_seconds=args.duration,
                parallel_workers=args.workers,
                num_threads=args.threads,
                keep_threads=args.keep_threads,
                heap_type=args.heap_type,
                pre_stw_wait=args.pre_stw_wait,
            )
            print(format_results(parallel_results))
            print()

            # Comparison
            throughput_speedup = parallel_results["throughput_per_second"] / serial_results["throughput_per_second"]

            # Total GC time reduction
            if serial_results["gc_time_mean_ms"] > 0:
                gc_time_reduction = 1 - (parallel_results["gc_time_mean_ms"] / serial_results["gc_time_mean_ms"])
            else:
                gc_time_reduction = 0

            print("=" * 70)
            print("COMPARISON")
            print("=" * 70)
            print(f"Throughput: {throughput_speedup:.2f}x ({'+' if throughput_speedup >= 1 else ''}{(throughput_speedup-1)*100:.1f}%)")
            print()

            # STW pause data only available for parallel mode
            if parallel_results["stw_pause_count"] > 0:
                print("STW Pause (parallel only - latency impact):")
                print(f"  Mean: {parallel_results['stw_pause_mean_ms']:.1f}ms")
                print(f"  Max: {parallel_results['stw_pause_max_ms']:.1f}ms")
                print(f"  P99: {parallel_results['stw_pause_p99_ms']:.1f}ms")
                print()

            print("Total GC Time (throughput impact):")
            print(f"  Mean: {serial_results['gc_time_mean_ms']:.1f}ms -> {parallel_results['gc_time_mean_ms']:.1f}ms ({gc_time_reduction * 100:+.1f}%)")
            print(f"  Total: {serial_results['gc_time_total_ms']:.0f}ms -> {parallel_results['gc_time_total_ms']:.0f}ms")
        else:
            # Single run
            results = run_throughput_benchmark(
                heap_size=heap_size,
                duration_seconds=args.duration,
                parallel_workers=args.parallel,
                num_threads=args.threads,
                keep_threads=args.keep_threads,
                heap_type=args.heap_type,
                pre_stw_wait=args.pre_stw_wait,
            )
            print(format_results(results))
    finally:
        shutdown_creation_pool()


if __name__ == "__main__":
    main()
