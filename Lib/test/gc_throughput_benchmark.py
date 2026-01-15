#!/usr/bin/env python3
"""
Parallel GC Throughput Benchmark

Measures real-world throughput impact of parallel GC by:
1. Building an ai_workload style heap to steady-state size
2. Running continuous create/destroy at steady state for duration Y
3. Counting total objects created (higher = better throughput)
4. Tracking GC pause times via gc.callbacks

Usage:
    python gc_throughput_benchmark.py --heap-size 500k --duration 30
    python gc_throughput_benchmark.py --heap-size 500k --duration 30 --parallel 8
"""

import gc
import sys
import time
import random
import argparse
import statistics
from dataclasses import dataclass, field
from typing import List, Optional

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
# Pause Tracking via gc.callbacks
# =============================================================================

@dataclass
class PauseTracker:
    """Tracks GC pause times using gc.callbacks."""
    pauses_ms: List[float] = field(default_factory=list)
    gc_start_time: Optional[float] = None
    total_collections: int = 0

    def gc_callback(self, phase: str, info: dict):
        """Called by gc module at start/stop of each collection."""
        if phase == "start":
            self.gc_start_time = time.perf_counter()
        elif phase == "stop":
            if self.gc_start_time is not None:
                pause_ms = (time.perf_counter() - self.gc_start_time) * 1000
                self.pauses_ms.append(pause_ms)
                self.total_collections += 1
                self.gc_start_time = None

    def reset(self):
        """Reset tracking for a new measurement period."""
        self.pauses_ms.clear()
        self.gc_start_time = None
        self.total_collections = 0

    def summary(self) -> dict:
        """Return pause statistics."""
        if not self.pauses_ms:
            return {
                "count": 0,
                "total_ms": 0,
                "mean_ms": 0,
                "max_ms": 0,
                "p99_ms": 0,
                "p95_ms": 0,
            }

        sorted_pauses = sorted(self.pauses_ms)
        p99_idx = int(len(sorted_pauses) * 0.99)
        p95_idx = int(len(sorted_pauses) * 0.95)

        return {
            "count": len(self.pauses_ms),
            "total_ms": sum(self.pauses_ms),
            "mean_ms": statistics.mean(self.pauses_ms),
            "max_ms": max(self.pauses_ms),
            "p99_ms": sorted_pauses[min(p99_idx, len(sorted_pauses) - 1)],
            "p95_ms": sorted_pauses[min(p95_idx, len(sorted_pauses) - 1)],
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
# Throughput Benchmark
# =============================================================================

def run_throughput_benchmark(
    heap_size: int,
    duration_seconds: float,
    parallel_workers: Optional[int],
    survivor_ratio: float = 0.8,
    gc_threshold: int = 500,
    seed: int = 42,
) -> dict:
    """
    Run throughput benchmark:
    1. Build heap to steady state
    2. Run create/destroy loop for duration
    3. Count objects created
    4. Track pause times
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
            else:
                gc.enable_parallel(parallel_workers)
            mode = f"parallel-{parallel_workers}"
        else:
            try:
                gc.disable_parallel()
            except (RuntimeError, AttributeError):
                pass
            mode = "serial"

        print(f"Building initial heap ({heap_size} objects)...")

        # Phase 1: Build initial heap
        gc.disable()  # Don't count setup GC
        heap = create_ai_workload(heap_size)
        gc.collect()  # Clean up any garbage from setup
        gc.enable()

        # Calculate churn parameters
        # Small churn size = more cycles = more GC events = better measurement
        # Target: ~1000-5000 objects per cycle to trigger frequent GC
        churn_size = min(5000, max(1000, heap_size // 100))

        print(f"Starting steady-state benchmark ({duration_seconds}s, {mode})...")
        print(f"  Heap size: {heap_size}, Churn per cycle: {churn_size}")

        # Phase 2: Steady-state throughput measurement
        tracker.reset()
        objects_created = 0
        cycles = 0

        start_time = time.perf_counter()
        end_time = start_time + duration_seconds

        while time.perf_counter() < end_time:
            # Create batch of new objects with references to heap
            # This creates "garbage" - objects that will become unreachable
            new_objects = []
            for i in range(churn_size):
                obj = Node()
                if heap and i % 3 == 0:
                    obj.refs.append(random.choice(heap))
                if new_objects and i % 2 == 0:
                    obj.refs.append(random.choice(new_objects))
                new_objects.append(obj)

            # Add to heap temporarily, then remove to create garbage pattern
            heap.extend(new_objects)

            # Remove some objects to create garbage (like real workload)
            num_to_remove = min(churn_size, len(heap) - heap_size)
            if num_to_remove > 0:
                # Remove from the end (the newly added objects become garbage)
                del heap[-num_to_remove:]

            objects_created += churn_size
            cycles += 1

        elapsed = time.perf_counter() - start_time

        # Final collection to ensure consistent state
        gc.collect()

        # Get pause summary
        pause_summary = tracker.summary()

        # Calculate throughput
        throughput = objects_created / elapsed

        return {
            "mode": mode,
            "heap_size": heap_size,
            "duration_seconds": elapsed,
            "objects_created": objects_created,
            "cycles": cycles,
            "throughput_per_second": throughput,
            "pause_count": pause_summary["count"],
            "pause_total_ms": pause_summary["total_ms"],
            "pause_mean_ms": pause_summary["mean_ms"],
            "pause_max_ms": pause_summary["max_ms"],
            "pause_p99_ms": pause_summary["p99_ms"],
            "pause_p95_ms": pause_summary["p95_ms"],
            "gc_overhead_percent": (pause_summary["total_ms"] / (elapsed * 1000)) * 100,
        }

    finally:
        # Clean up
        gc.callbacks.remove(tracker.gc_callback)
        try:
            gc.disable_parallel()
        except (RuntimeError, AttributeError):
            pass


def format_results(results: dict) -> str:
    """Format results for display."""
    lines = [
        f"Mode: {results['mode']}",
        f"Duration: {results['duration_seconds']:.2f}s",
        f"Objects created: {results['objects_created']:,}",
        f"Throughput: {results['throughput_per_second']:,.0f} objects/sec",
        f"",
        f"GC Pauses:",
        f"  Count: {results['pause_count']}",
        f"  Total: {results['pause_total_ms']:.1f}ms",
        f"  Mean: {results['pause_mean_ms']:.2f}ms",
        f"  Max: {results['pause_max_ms']:.2f}ms",
        f"  P99: {results['pause_p99_ms']:.2f}ms",
        f"  P95: {results['pause_p95_ms']:.2f}ms",
        f"  Overhead: {results['gc_overhead_percent']:.2f}%",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Parallel GC Throughput Benchmark")
    parser.add_argument("--heap-size", "-s", type=str, default="500k",
                        help="Steady-state heap size (e.g., 100k, 500k, 1M)")
    parser.add_argument("--duration", "-d", type=float, default=30.0,
                        help="Benchmark duration in seconds")
    parser.add_argument("--parallel", "-p", type=int, default=None,
                        help="Number of parallel workers (omit for serial)")
    parser.add_argument("--survivor-ratio", "-r", type=float, default=0.8,
                        help="Fraction of heap that survives each cycle")
    parser.add_argument("--compare", "-c", action="store_true",
                        help="Run both serial and parallel, compare results")
    parser.add_argument("--workers", "-w", type=int, default=8,
                        help="Workers to use in compare mode")
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

    if args.compare and parallel_available:
        # Run both serial and parallel
        print("Running serial benchmark...")
        serial_results = run_throughput_benchmark(
            heap_size=heap_size,
            duration_seconds=args.duration,
            parallel_workers=None,
            survivor_ratio=args.survivor_ratio,
        )
        print(format_results(serial_results))
        print()

        print(f"Running parallel benchmark ({args.workers} workers)...")
        parallel_results = run_throughput_benchmark(
            heap_size=heap_size,
            duration_seconds=args.duration,
            parallel_workers=args.workers,
            survivor_ratio=args.survivor_ratio,
        )
        print(format_results(parallel_results))
        print()

        # Comparison
        speedup = parallel_results["throughput_per_second"] / serial_results["throughput_per_second"]
        pause_reduction = 1 - (parallel_results["pause_mean_ms"] / serial_results["pause_mean_ms"]) if serial_results["pause_mean_ms"] > 0 else 0

        print("=" * 70)
        print("COMPARISON")
        print("=" * 70)
        print(f"Throughput speedup: {speedup:.2f}x")
        print(f"Mean pause reduction: {pause_reduction * 100:.1f}%")
        print(f"Max pause: {serial_results['pause_max_ms']:.1f}ms (serial) -> {parallel_results['pause_max_ms']:.1f}ms (parallel)")
        print(f"GC overhead: {serial_results['gc_overhead_percent']:.2f}% (serial) -> {parallel_results['gc_overhead_percent']:.2f}% (parallel)")
    else:
        # Single run
        results = run_throughput_benchmark(
            heap_size=heap_size,
            duration_seconds=args.duration,
            parallel_workers=args.parallel,
            survivor_ratio=args.survivor_ratio,
        )
        print(format_results(results))


if __name__ == "__main__":
    main()
