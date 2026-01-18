#!/usr/bin/env python3
"""
Analyse how object creation thread count affects GC performance.

This script investigates the hypothesis that multi-threaded object creation
causes different heap distribution (across mimalloc thread-local heaps) that
makes parallel GC more expensive.

IMPORTANT: For accurate serial vs parallel comparison, use --subprocess mode
to avoid stale stats contamination between runs.

Usage:
    ./python Lib/test/gc_creation_analysis.py --creation-threads --heap ai_workload
    ./python Lib/test/gc_creation_analysis.py --chain-vs-clusters
    ./python Lib/test/gc_creation_analysis.py --all-phases --heap ai_workload
"""
import sys
import argparse
import subprocess
import threading
import random


# =============================================================================
# Node Classes (matching gc_benchmark.py)
# =============================================================================

class Node:
    """Node class matching gc_benchmark.py for comparison."""
    __slots__ = ['refs', 'data', '__weakref__']

    def __init__(self):
        self.refs = []
        self.data = None


class ContainerNode:
    """Node using __dict__ with list and dict children - models real objects."""

    def __init__(self):
        self.children_list = []
        self.children_dict = {}
        self.parent_ref = None


class FinalizerNode(Node):
    """Node with a __del__ finalizer."""

    def __del__(self):
        pass  # Just having __del__ is enough


# =============================================================================
# Heap Generators
# =============================================================================

def create_chain(n, cluster_size=100):
    """Create isolated circular chains (single long chain per cluster)."""
    clusters = []
    num_clusters = max(1, n // cluster_size)

    for _ in range(num_clusters):
        nodes = [Node() for _ in range(cluster_size)]
        for i in range(cluster_size):
            nodes[i].refs.append(nodes[(i + 1) % cluster_size])
        clusters.append(nodes)

    return clusters


def create_ai_workload(n, cluster_size=100):
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
        num_parents = cluster_size // 6  # Each parent has ~5 children

        parents = []
        for _ in range(num_parents):
            parent = ContainerNode()
            parents.append(parent)
            all_nodes.append(parent)

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

                child.refs.append(parent)

        for parent in parents:
            if parents:
                parent.children_list.append(random.choice(parents))

        clusters.append(all_nodes)

    return clusters


def create_clusters(n, cluster_size=100):
    """Create isolated Node clusters (like benchmark chain but clustered)."""
    return create_chain(n, cluster_size)


HEAP_GENERATORS = {
    'chain': create_chain,
    'clusters': create_clusters,
    'ai_workload': create_ai_workload,
}


# =============================================================================
# Multi-threaded Creation
# =============================================================================

def create_heap_single_thread(heap_type, n, cluster_size=100):
    """Create heap with single thread."""
    generator = HEAP_GENERATORS.get(heap_type, create_ai_workload)
    return generator(n, cluster_size)


def create_heap_multi_thread(heap_type, n, num_threads, cluster_size=100,
                              keep_threads_alive=False):
    """Create heap with multiple threads (distributes across mimalloc heaps).

    Args:
        keep_threads_alive: If True, returns (clusters, threads) and threads
            stay alive until caller joins them. This keeps heap pages in live
            thread heaps instead of abandoned pool.
    """
    chunk_size = n // num_threads
    results = [None] * num_threads
    generator = HEAP_GENERATORS.get(heap_type, create_ai_workload)

    if keep_threads_alive:
        # Use barriers to synchronise: threads wait after creating objects
        import threading as _threading
        creation_done = _threading.Barrier(num_threads + 1)  # +1 for main
        release_barrier = _threading.Barrier(num_threads + 1)

        def create_chunk_and_wait(tid, res):
            res[tid] = generator(chunk_size, cluster_size)
            creation_done.wait()  # Signal creation done
            release_barrier.wait()  # Wait for GC to complete

        threads = []
        for tid in range(num_threads):
            t = _threading.Thread(target=create_chunk_and_wait, args=(tid, results))
            threads.append(t)
            t.start()

        # Wait for all threads to finish creating objects
        creation_done.wait()

        # Flatten clusters - objects now in live thread heaps (not abandoned)
        all_clusters = []
        for chunk in results:
            if chunk:
                all_clusters.extend(chunk)

        # Return clusters and a release function
        def release_threads():
            release_barrier.wait()
            for t in threads:
                t.join()

        return all_clusters, release_threads
    else:
        # Original behavior: threads exit, pages go to abandoned pool
        def create_chunk(tid, res):
            res[tid] = generator(chunk_size, cluster_size)

        threads = []
        for tid in range(num_threads):
            t = threading.Thread(target=create_chunk, args=(tid, results))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        # Flatten all clusters
        all_clusters = []
        for chunk in results:
            if chunk:
                all_clusters.extend(chunk)

        results = None
        threads = None

        return all_clusters, None


# =============================================================================
# GC Testing
# =============================================================================

PHASES = [
    ('mark_alive', 'mark_alive_ns'),
    ('update_refs', 'update_refs_ns'),
    ('mark_heap', 'mark_heap_ns'),
    ('scan_heap', 'scan_heap_ns'),
    ('find_weakrefs', 'find_weakrefs_ns'),
    ('cleanup', 'cleanup_ns'),
    ('finalize', 'finalize_ns'),
    ('resurrection', 'resurrection_ns'),
]


def run_subprocess_test(mode, workers, creation_threads, size, heap_type='ai_workload',
                        survivors=False, cluster_size=100, keep_threads_alive=False):
    """Run a GC test in a subprocess for clean state.

    Args:
        keep_threads_alive: If True, creation threads stay alive during GC,
            keeping pages in live thread heaps. If False (default), threads
            exit after creation, putting pages in abandoned pool.
    """
    if survivors:
        release_code = "keep = clusters; clusters = None"
    else:
        release_code = "clusters = None"

    # Thread creation mode
    if keep_threads_alive:
        thread_mode = "pool"
    else:
        thread_mode = "abandon"

    script = f'''
import gc
import sys
import random
import threading

random.seed(42)  # Reproducible

class Node:
    __slots__ = ['refs', 'data', '__weakref__']
    def __init__(self):
        self.refs = []
        self.data = None

class ContainerNode:
    def __init__(self):
        self.children_list = []
        self.children_dict = {{}}
        self.parent_ref = None

class FinalizerNode(Node):
    def __del__(self):
        pass

def create_chain(n, cluster_size):
    clusters = []
    num_clusters = max(1, n // cluster_size)
    for _ in range(num_clusters):
        nodes = [Node() for _ in range(cluster_size)]
        for i in range(cluster_size):
            nodes[i].refs.append(nodes[(i + 1) % cluster_size])
        clusters.append(nodes)
    return clusters

def create_ai_workload(n, cluster_size):
    clusters = []
    num_clusters = max(1, n // cluster_size)
    for _ in range(num_clusters):
        all_nodes = []
        num_parents = cluster_size // 6
        parents = []
        for _ in range(num_parents):
            parent = ContainerNode()
            parents.append(parent)
            all_nodes.append(parent)
            for j in range(random.randint(3, 5)):
                if random.random() < 0.1:
                    child = FinalizerNode()
                else:
                    child = Node()
                all_nodes.append(child)
                if random.random() < 0.5:
                    parent.children_list.append(child)
                else:
                    parent.children_dict[f"child_{{j}}"] = child
                child.refs.append(parent)
        for parent in parents:
            if parents:
                parent.children_list.append(random.choice(parents))
        clusters.append(all_nodes)
    return clusters

GENERATORS = {{"chain": create_chain, "ai_workload": create_ai_workload, "clusters": create_chain}}

mode = "{mode}"
workers = {workers}
creation_threads = {creation_threads}
size = {size}
heap_type = "{heap_type}"
cluster_size = {cluster_size}
thread_mode = "{thread_mode}"

if mode == "parallel":
    gc.enable_parallel(num_workers=workers)
else:
    try:
        gc.disable_parallel()
    except:
        pass

gc.collect()
gc.collect()

generator = GENERATORS.get(heap_type, create_ai_workload)

release_threads = None  # Function to release pool threads

if creation_threads == 1:
    clusters = generator(size, cluster_size)
elif thread_mode == "pool":
    # Pool mode: threads stay alive during GC (pages in live heaps)
    chunk_size = size // creation_threads
    results = [None] * creation_threads
    creation_done = threading.Barrier(creation_threads + 1)
    release_barrier = threading.Barrier(creation_threads + 1)

    def create_chunk_and_wait(tid, res):
        res[tid] = generator(chunk_size, cluster_size)
        creation_done.wait()
        release_barrier.wait()

    threads = []
    for tid in range(creation_threads):
        t = threading.Thread(target=create_chunk_and_wait, args=(tid, results))
        threads.append(t)
        t.start()

    creation_done.wait()
    clusters = []
    for chunk in results:
        if chunk:
            clusters.extend(chunk)

    def release_threads_func():
        release_barrier.wait()
        for t in threads:
            t.join()
    release_threads = release_threads_func
else:
    # Abandon mode: threads exit, pages go to abandoned pool
    chunk_size = size // creation_threads
    results = [None] * creation_threads
    def create_chunk(tid, res):
        res[tid] = generator(chunk_size, cluster_size)
    threads = []
    for tid in range(creation_threads):
        t = threading.Thread(target=create_chunk, args=(tid, results))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    clusters = []
    for chunk in results:
        if chunk:
            clusters.extend(chunk)
    results = None
    threads = None

{release_code}
collected = gc.collect()
stats = gc.get_parallel_stats()
pt = stats.get("phase_timing", {{}})

# Release pool threads after GC if applicable
if release_threads is not None:
    release_threads()

for key, val in sorted(pt.items()):
    print(f"{{key}}={{val / 1e6:.3f}}")
print(f"collected={{collected}}")
print(f"enabled={{stats.get('enabled', False)}}")
'''
    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Subprocess error: {result.stderr}")
        return {}

    data = {}
    for line in result.stdout.strip().split('\n'):
        if '=' in line:
            key, val = line.split('=', 1)
            try:
                data[key] = float(val)
            except ValueError:
                data[key] = val
    return data


def test_creation_threads_impact(size, heap_type, gc_workers, creation_thread_counts,
                                  survivor_ratio=0.8):
    """Test how creation thread count affects parallel GC performance."""
    survivors = survivor_ratio >= 1.0
    survivor_pct = int(survivor_ratio * 100)

    print("=" * 80)
    print(f"CREATION THREAD IMPACT ON PARALLEL GC")
    print(f"Heap: {heap_type}, Size: {size:,}, Workers: {gc_workers}, Survivors: {survivor_pct}%")
    print("=" * 80)

    results = []
    for ct in creation_thread_counts:
        print(f"\n--- Creation threads: {ct} ---")

        serial = run_subprocess_test('serial', gc_workers, ct, size, heap_type, survivors)
        parallel = run_subprocess_test('parallel', gc_workers, ct, size, heap_type, survivors)

        s_total = serial.get('total_ns', 1)
        p_total = parallel.get('total_ns', 1)
        s_mark = serial.get('mark_alive_ns', 0)
        p_mark = parallel.get('mark_alive_ns', 0)
        speedup = s_total / p_total if p_total > 0 else 0

        print(f"  Serial:   total={s_total:.1f}ms, mark_alive={s_mark:.1f}ms")
        print(f"  Parallel: total={p_total:.1f}ms, mark_alive={p_mark:.1f}ms")
        print(f"  Speedup: {speedup:.2f}x")

        results.append({
            'creation_threads': ct,
            'serial_total': s_total,
            'parallel_total': p_total,
            'serial_mark': s_mark,
            'parallel_mark': p_mark,
            'speedup': speedup,
        })

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Create T':>10} | {'Serial':>10} | {'Parallel':>10} | {'Speedup':>10}")
    print("-" * 50)
    for r in results:
        print(f"{r['creation_threads']:>10} | {r['serial_total']:>10.1f} | "
              f"{r['parallel_total']:>10.1f} | {r['speedup']:>10.2f}x")

    return results


def compare_chain_vs_clusters(size, gc_workers):
    """Compare chain structure vs clusters to show parallel GC sensitivity to structure."""
    print("=" * 80)
    print("CHAIN vs CLUSTERS COMPARISON")
    print(f"Size: {size:,}, Workers: {gc_workers}, 100% survivors")
    print("=" * 80)
    print()
    print("This test shows that heap STRUCTURE (not object type) determines")
    print("whether parallel GC helps or hurts performance.")
    print()

    for heap_type in ['chain', 'clusters']:
        print(f"--- {heap_type.upper()} ---")

        serial = run_subprocess_test('serial', gc_workers, 1, size, heap_type, survivors=True)
        parallel = run_subprocess_test('parallel', gc_workers, 1, size, heap_type, survivors=True)

        s_total = serial.get('total_ns', 1)
        p_total = parallel.get('total_ns', 1)
        s_mark = serial.get('mark_alive_ns', 0)
        p_mark = parallel.get('mark_alive_ns', 0)
        speedup = s_total / p_total if p_total > 0 else 0

        print(f"  Serial:   total={s_total:.1f}ms, mark_alive={s_mark:.1f}ms")
        print(f"  Parallel: total={p_total:.1f}ms, mark_alive={p_mark:.1f}ms")
        print(f"  Speedup: {speedup:.2f}x")
        if speedup < 1.0:
            print(f"  ** PARALLEL IS {1/speedup:.1f}x SLOWER **")
        print()


def show_all_phases(size, creation_threads, gc_workers, heap_type, survivors=False):
    """Show all phases for a single run (useful for debugging)."""
    print("=" * 80)
    print(f"ALL PHASES (serial vs parallel) - {'100% survivors' if survivors else '100% garbage'}")
    print(f"Heap: {heap_type}, Size: {size:,}, Creation threads: {creation_threads}")
    print(f"GC workers: {gc_workers}")
    print("=" * 80)

    serial = run_subprocess_test('serial', gc_workers, creation_threads, size, heap_type, survivors)
    parallel = run_subprocess_test('parallel', gc_workers, creation_threads, size, heap_type, survivors)

    all_keys = sorted(set(serial.keys()) | set(parallel.keys()))

    print(f"\n{'Phase':<25} | {'Serial':>10} | {'Parallel':>10} | {'Diff':>10}")
    print("-" * 65)

    for key in all_keys:
        if key in ('collected', 'enabled'):
            continue
        s = serial.get(key, 0)
        p = parallel.get(key, 0)
        if isinstance(s, str) or isinstance(p, str):
            continue
        diff = p - s
        marker = "**" if abs(diff) > 1 else ""
        name = key.replace('_ns', '')
        print(f"{name:<25} | {s:>10.2f} | {p:>10.2f} | {diff:>+10.2f} {marker}")


def compare_abandoned_vs_pool(size, gc_workers, creation_threads_list, heap_type):
    """Compare abandoned threads vs pool threads for parallel GC performance.

    This test shows the impact of the abandoned pool fix:
    - Abandoned: Creation threads exit, pages go to abandoned pool
    - Pool: Creation threads stay alive, pages remain in live heaps
    """
    print("=" * 80)
    print("ABANDONED vs POOL THREADS COMPARISON")
    print(f"Heap: {heap_type}, Size: {size:,}, Workers: {gc_workers}, 100% survivors")
    print("=" * 80)
    print()
    print("This test compares two multi-thread creation modes:")
    print("  - ABANDON: Threads exit after creation -> pages in abandoned pool")
    print("  - POOL:    Threads stay alive during GC -> pages in live heaps")
    print()

    results = []
    for ct in creation_threads_list:
        if ct == 1:
            continue  # Single thread has no abandoned pool behavior
        print(f"--- Creation threads: {ct} ---")

        abandon = run_subprocess_test('parallel', gc_workers, ct, size, heap_type,
                                       survivors=True, keep_threads_alive=False)
        pool = run_subprocess_test('parallel', gc_workers, ct, size, heap_type,
                                    survivors=True, keep_threads_alive=True)

        a_total = abandon.get('total_ns', 1)
        p_total = pool.get('total_ns', 1)
        a_scan = abandon.get('scan_heap_ns', 0)
        p_scan = pool.get('scan_heap_ns', 0)

        print(f"  Abandon: total={a_total:.1f}ms, scan_heap={a_scan:.1f}ms")
        print(f"  Pool:    total={p_total:.1f}ms, scan_heap={p_scan:.1f}ms")

        if a_total > 0 and p_total > 0:
            ratio = a_total / p_total
            if ratio > 1.0:
                print(f"  -> Abandon is {ratio:.1f}x SLOWER (abandoned pool overhead)")
            else:
                print(f"  -> Pool is {1/ratio:.1f}x SLOWER")
        print()

        results.append({
            'creation_threads': ct,
            'abandon_total': a_total,
            'pool_total': p_total,
            'abandon_scan': a_scan,
            'pool_scan': p_scan,
        })

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Create T':>10} | {'Abandon':>10} | {'Pool':>10} | {'Ratio':>10}")
    print("-" * 50)
    for r in results:
        ratio = r['abandon_total'] / r['pool_total'] if r['pool_total'] > 0 else 0
        print(f"{r['creation_threads']:>10} | {r['abandon_total']:>10.1f} | "
              f"{r['pool_total']:>10.1f} | {ratio:>10.2f}x")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Analyse GC creation thread impact',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test creation thread impact with ai_workload heap (primary use case)
    ./python Lib/test/gc_creation_analysis.py --creation-threads --heap ai_workload

    # Compare chain vs clusters to show structure sensitivity
    ./python Lib/test/gc_creation_analysis.py --chain-vs-clusters

    # Compare abandoned vs pool threads (shows parallel GC fix impact)
    ./python Lib/test/gc_creation_analysis.py --abandon-vs-pool --heap ai_workload

    # Show all phases for debugging
    ./python Lib/test/gc_creation_analysis.py --all-phases --heap ai_workload --survivors
""")
    parser.add_argument('--threads', type=int, default=1,
                        help='Number of threads for object creation')
    parser.add_argument('--size', type=int, default=400000,
                        help='Number of objects to create')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel GC workers')
    parser.add_argument('--heap', type=str, default='ai_workload',
                        choices=['chain', 'clusters', 'ai_workload'],
                        help='Heap structure type (default: ai_workload)')
    parser.add_argument('--creation-threads', action='store_true',
                        help='Test creation thread impact (primary analysis)')
    parser.add_argument('--chain-vs-clusters', action='store_true',
                        help='Compare chain vs clusters structure sensitivity')
    parser.add_argument('--abandon-vs-pool', action='store_true',
                        help='Compare abandoned threads vs pool threads')
    parser.add_argument('--all-phases', action='store_true',
                        help='Show all phases (subprocess mode)')
    parser.add_argument('--survivors', action='store_true',
                        help='Keep all objects alive (100%% survivors, no garbage)')

    args = parser.parse_args()

    if args.chain_vs_clusters:
        compare_chain_vs_clusters(args.size, args.workers)
    elif args.abandon_vs_pool:
        compare_abandoned_vs_pool(args.size, args.workers, [2, 4, 8], args.heap)
    elif args.creation_threads:
        test_creation_threads_impact(args.size, args.heap, args.workers, [1, 2, 4, 8])
    elif args.all_phases:
        show_all_phases(args.size, args.threads, args.workers, args.heap, args.survivors)
    else:
        # Default: show creation thread impact with ai_workload
        test_creation_threads_impact(args.size, args.heap, args.workers, [1, 2, 4])


if __name__ == '__main__':
    main()
