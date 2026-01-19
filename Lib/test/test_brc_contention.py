#!/usr/bin/env python3
"""
Benchmark to test BRC (Biased Reference Counting) contention hypothesis.

Hypothesis: Cross-thread decrefs serialize on BRC bucket mutexes when objects
haven't been merged yet. After merge, decrefs use pure atomic operations.

Test cases:
1. Producer-consumer with fresh objects (never shared) - should hit BRC
2. Producer-consumer with pre-shared objects (already merged) - should be fast
3. Self-owned objects (same thread creates and decrefs) - should be fast

Run with: ./python Lib/test/test_brc_contention.py
"""

import gc
import sys
import threading
import time

# Disable GC during benchmark to isolate BRC effects
gc.disable()


def create_objects(n):
    """Create n simple objects."""
    return [object() for _ in range(n)]


def create_containers(n, items_per_container=10):
    """Create n list containers, each with items_per_container objects."""
    return [[object() for _ in range(items_per_container)] for _ in range(n)]


def benchmark_producer_consumer_fresh(num_objects, num_consumers):
    """
    Simpler test without Queue overhead.
    Producer creates objects, stores in shared list.
    Consumers pop from list and decref.
    Uses a simple lock for list access to isolate BRC effects.
    """
    import random

    # Producer creates all objects upfront
    objects = [object() for _ in range(num_objects)]
    random.shuffle(objects)  # Randomize to avoid cache effects

    # Use indices to avoid list locking - each consumer gets a range
    chunk_size = num_objects // num_consumers
    results = []

    def worker(start, end):
        t0 = time.perf_counter()
        # Each worker clears their portion by setting to None
        for i in range(start, end):
            obj = objects[i]
            objects[i] = None  # Release reference, triggers decref
            del obj
        t1 = time.perf_counter()
        results.append(('time', t1 - t0, end - start))

    t0 = time.perf_counter()
    threads = []
    for i in range(num_consumers):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_consumers - 1 else num_objects
        t = threading.Thread(target=worker, args=(start, end))
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total_time = time.perf_counter() - t0

    total_consumer_time = sum(r[1] for r in results)
    total_consumed = sum(r[2] for r in results)

    return {
        'producer_time': 0,  # Not meaningful in this version
        'total_consumer_time': total_consumer_time,
        'avg_consumer_time': total_consumer_time / num_consumers,
        'objects_consumed': total_consumed,
        'throughput': total_consumed / total_time if total_time > 0 else 0,
    }


def benchmark_self_owned(num_objects, num_threads):
    """
    Each thread creates and decrefs its own objects.
    No cross-thread decrefs, so no BRC contention.
    """
    times = []

    def worker():
        t0 = time.perf_counter()
        for _ in range(num_objects // num_threads):
            obj = object()
            del obj
        t1 = time.perf_counter()
        times.append(t1 - t0)

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total_time = time.perf_counter() - t0

    return {
        'total_time': total_time,
        'avg_thread_time': sum(times) / len(times),
        'throughput': num_objects / total_time,
    }


def benchmark_cross_thread_decref(num_objects, num_consumers):
    """
    Single producer creates all objects first, then distributes to consumers.
    All decrefs are cross-thread and should hit BRC.
    Consumer clears the chunk list to release final references.
    """
    # Create all objects on main thread
    objects = [object() for _ in range(num_objects)]

    # Split among consumers - give each a copy they can clear
    chunk_size = num_objects // num_consumers
    chunks = [objects[i*chunk_size:(i+1)*chunk_size] for i in range(num_consumers)]

    # Clear original list so chunks hold the only references
    objects.clear()

    times = []
    def worker(chunk):
        t0 = time.perf_counter()
        # Clear the chunk list - this releases the final reference to each object
        # Each object was created by main thread, so decref goes through BRC
        chunk.clear()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(chunks[i],)) for i in range(num_consumers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total_time = time.perf_counter() - t0

    return {
        'total_time': total_time,
        'avg_thread_time': sum(times) / len(times) if times else 0,
        'throughput': num_objects / total_time if total_time > 0 else 0,
    }


def benchmark_pre_merged(num_objects, num_consumers):
    """
    Objects are created, shared across threads (forcing merge), then decrefd.
    Post-merge decrefs should be fast (pure atomic).
    """
    # Create objects on main thread
    objects = [object() for _ in range(num_objects)]

    # Force sharing: pass each object to another thread and back
    # This should trigger merge
    def touch_objects(objs):
        for obj in objs:
            _ = obj  # Just reference it

    # Have a worker thread touch all objects
    t = threading.Thread(target=touch_objects, args=(objects,))
    t.start()
    t.join()

    # Now decref from multiple consumers - objects should be merged
    chunk_size = num_objects // num_consumers
    chunks = [objects[i*chunk_size:(i+1)*chunk_size] for i in range(num_consumers)]

    times = []
    def worker(chunk):
        t0 = time.perf_counter()
        for obj in chunk:
            del obj
        t1 = time.perf_counter()
        times.append(t1 - t0)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(chunks[i],)) for i in range(num_consumers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total_time = time.perf_counter() - t0

    objects.clear()

    return {
        'total_time': total_time,
        'avg_thread_time': sum(times) / len(times) if times else 0,
        'throughput': num_objects / total_time if total_time > 0 else 0,
    }


def main():
    num_objects = 100_000
    consumer_counts = [1, 2, 4, 8, 16]

    print("BRC Contention Benchmark")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"Objects per test: {num_objects:,}")
    print()

    # Test 1: Self-owned (baseline - no cross-thread)
    print("Test 1: Self-owned (each thread creates/decrefs own objects)")
    print("-" * 60)
    for nc in consumer_counts:
        result = benchmark_self_owned(num_objects, nc)
        print(f"  Threads={nc:2d}: {result['total_time']*1000:7.2f} ms, "
              f"throughput={result['throughput']/1e6:.2f} M/s")
    print()

    # Test 2: Cross-thread fresh (should hit BRC)
    print("Test 2: Cross-thread fresh (producer creates, consumers decref)")
    print("-" * 60)
    for nc in consumer_counts:
        result = benchmark_cross_thread_decref(num_objects, nc)
        print(f"  Consumers={nc:2d}: {result['total_time']*1000:7.2f} ms, "
              f"throughput={result['throughput']/1e6:.2f} M/s")
    print()

    # Test 3: Pre-merged (should be fast)
    print("Test 3: Pre-merged (objects shared before decref)")
    print("-" * 60)
    for nc in consumer_counts:
        result = benchmark_pre_merged(num_objects, nc)
        print(f"  Consumers={nc:2d}: {result['total_time']*1000:7.2f} ms, "
              f"throughput={result['throughput']/1e6:.2f} M/s")
    print()

    # Test 4: Producer-consumer queue pattern
    print("Test 4: Producer-consumer queue (realistic pattern)")
    print("-" * 60)
    for nc in consumer_counts:
        result = benchmark_producer_consumer_fresh(num_objects, nc)
        print(f"  Consumers={nc:2d}: producer={result['producer_time']*1000:6.2f} ms, "
              f"throughput={result['throughput']/1e6:.2f} M/s")
    print()


if __name__ == "__main__":
    main()
