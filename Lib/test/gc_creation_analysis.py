#!/usr/bin/env python3
"""
Analyse how object creation thread count affects GC performance.

This script investigates the hypothesis that multi-threaded object creation
causes different heap distribution (across mimalloc thread-local heaps) that
makes parallel GC more expensive.

IMPORTANT: For accurate serial vs parallel comparison, use --subprocess mode
to avoid stale stats contamination between runs.

Usage:
    ./python Lib/test/gc_creation_analysis.py --compare --workers 8
    ./python Lib/test/gc_creation_analysis.py --serial-vs-parallel --subprocess
    ./python Lib/test/gc_creation_analysis.py --all-phases
"""
import gc
import sys
import argparse
import subprocess
import threading


class Node:
    """Node class matching gc_benchmark.py for comparison."""
    __slots__ = ['refs', 'data', '__weakref__']

    def __init__(self):
        self.refs = []
        self.data = None


def create_cyclic_chain_single_thread(n, use_nodes=False):
    """Create cyclic chain with single thread (actual GC work)."""
    if use_nodes:
        objects = [Node() for _ in range(n)]
        for i in range(n - 1):
            objects[i].refs.append(objects[i + 1])
        objects[-1].refs.append(objects[0])
    else:
        objects = [{"id": i} for i in range(n)]
        for i in range(n - 1):
            objects[i]["next"] = objects[i + 1]
        objects[-1]["next"] = objects[0]
    return objects


def create_cyclic_chain_multi_thread(n, num_threads, use_nodes=False):
    """Create cyclic chain with multiple threads (mimics benchmark behaviour)."""
    chunk_size = n // num_threads
    results = [None] * num_threads

    def create_chunk(tid, res):
        if use_nodes:
            chunk = [Node() for _ in range(chunk_size)]
            for i in range(chunk_size - 1):
                chunk[i].refs.append(chunk[i + 1])
        else:
            chunk = [{"id": tid * chunk_size + i} for i in range(chunk_size)]
            for i in range(chunk_size - 1):
                chunk[i]["next"] = chunk[i + 1]
        res[tid] = chunk

    threads = []
    for tid in range(num_threads):
        t = threading.Thread(target=create_chunk, args=(tid, results))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    all_objects = []
    for chunk in results:
        if chunk:
            all_objects.extend(chunk)

    if use_nodes:
        for i in range(len(all_objects) - 1):
            if not all_objects[i].refs or all_objects[i].refs[-1] is not all_objects[i + 1]:
                all_objects[i].refs.append(all_objects[i + 1])
        all_objects[-1].refs.append(all_objects[0])
    else:
        for i in range(len(all_objects) - 1):
            if "next" not in all_objects[i]:
                all_objects[i]["next"] = all_objects[i + 1]
        all_objects[-1]["next"] = all_objects[0]

    # Clean up intermediates
    results = None
    threads = None

    return all_objects


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


def enable_parallel_gc(num_workers):
    """Enable parallel GC with specified worker count."""
    try:
        gc.enable_parallel(num_workers=num_workers)
        return True
    except (RuntimeError, AttributeError) as e:
        print(f"Warning: Could not enable parallel GC: {e}")
        return False


def disable_parallel_gc():
    """Disable parallel GC."""
    try:
        gc.disable_parallel()
    except (RuntimeError, AttributeError):
        pass


def run_gc_test(creation_threads, object_count, verbose=True):
    """Run a GC test with specified creation thread count."""
    gc.collect()
    gc.collect()

    if creation_threads == 1:
        objects = create_cyclic_chain_single_thread(object_count)
    else:
        objects = create_cyclic_chain_multi_thread(object_count, creation_threads)

    # Use = None instead of del for proper cleanup
    objects = None

    collected = gc.collect()
    stats = gc.get_parallel_stats()
    pt = stats.get('phase_timing', {})
    parallel_enabled = stats.get('enabled', False)
    num_workers = stats.get('num_workers', 0)

    total = pt.get('total_ns', 0) / 1e6

    results = {
        'creation_threads': creation_threads,
        'object_count': object_count,
        'collected': collected,
        'total_ms': total,
        'parallel_enabled': parallel_enabled,
        'num_workers': num_workers,
        'phases': {},
        'raw_phases': pt,
    }

    for name, key in PHASES:
        val = pt.get(key, 0) / 1e6
        pct = val / total * 100 if total > 0 else 0
        results['phases'][name] = {'ms': val, 'pct': pct}

    if verbose:
        gc_mode = f"Parallel ({num_workers} workers)" if parallel_enabled else "Serial"
        print(f"Creation threads: {creation_threads}")
        print(f"GC mode: {gc_mode}")
        print(f"Object count: {object_count:,}")
        print(f"Collected: {collected:,}")
        print(f"Total: {total:.1f}ms")
        print()

        for name, key in PHASES:
            val = results['phases'][name]['ms']
            pct = int(results['phases'][name]['pct'])
            print(f"  {name:15}: {val:7.1f}ms ({pct:3}%)")
        print("---")

    return results


def compare_creation_threads(object_count, thread_counts):
    """Compare GC behaviour across different creation thread counts."""
    print("=" * 70)
    print("Comparing GC behaviour with different object creation thread counts")
    print("=" * 70)
    print(f"Object count: {object_count:,}")
    print()

    all_results = []
    for threads in thread_counts:
        print(f"\n=== Creation threads: {threads} ===")
        results = run_gc_test(threads, object_count)
        all_results.append(results)

    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)

    header = f"{'Phase':<15}"
    for r in all_results:
        header += f" | {r['creation_threads']:>1}T (ms)"
    print(header)
    print("-" * len(header))

    phases = ['mark_alive', 'update_refs', 'mark_heap', 'scan_heap',
              'find_weakrefs', 'cleanup', 'finalize', 'resurrection']

    for phase in phases:
        row = f"{phase:<15}"
        for r in all_results:
            val = r['phases'][phase]['ms']
            row += f" | {val:>8.1f}"
        print(row)

    print("-" * len(header))
    row = f"{'TOTAL':<15}"
    for r in all_results:
        row += f" | {r['total_ms']:>8.1f}"
    print(row)


def run_subprocess_test(mode, workers, creation_threads, size):
    """Run a GC test in a subprocess for clean state."""
    script = f'''
import gc
import sys
import threading

mode = "{mode}"
workers = {workers}
creation_threads = {creation_threads}
size = {size}

if mode == "parallel":
    gc.enable_parallel(num_workers=workers)
else:
    try:
        gc.disable_parallel()
    except:
        pass

gc.collect()
gc.collect()

if creation_threads == 1:
    objects = [{{"id": i}} for i in range(size)]
    for i in range(size - 1):
        objects[i]["next"] = objects[i + 1]
    objects[-1]["next"] = objects[0]
else:
    chunk_size = size // creation_threads
    results = [None] * creation_threads
    def create_chunk(tid, res):
        chunk = [{{"id": tid * chunk_size + i}} for i in range(chunk_size)]
        for i in range(chunk_size - 1):
            chunk[i]["next"] = chunk[i + 1]
        res[tid] = chunk
    threads = []
    for tid in range(creation_threads):
        t = threading.Thread(target=create_chunk, args=(tid, results))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    objects = []
    for chunk in results:
        if chunk:
            objects.extend(chunk)
    for i in range(len(objects) - 1):
        if "next" not in objects[i]:
            objects[i]["next"] = objects[i + 1]
    objects[-1]["next"] = objects[0]
    results = None
    threads = None

objects = None
collected = gc.collect()
stats = gc.get_parallel_stats()
pt = stats.get("phase_timing", {{}})

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


def compare_serial_vs_parallel_subprocess(object_count, creation_threads_list, gc_workers):
    """Compare serial vs parallel GC using subprocesses for clean state."""
    print("=" * 80)
    print("SERIAL vs PARALLEL GC COMPARISON (subprocess isolation)")
    print(f"Object count: {object_count:,}, Parallel GC workers: {gc_workers}")
    print("=" * 80)

    key_phases = ['mark_alive_ns', 'cleanup_ns', 'finalize_ns', 'resurrection_ns', 'total_ns']

    results = []
    for ct in creation_threads_list:
        print(f"\n--- Creation threads: {ct} ---")

        serial = run_subprocess_test('serial', gc_workers, ct, object_count)
        parallel = run_subprocess_test('parallel', gc_workers, ct, object_count)

        s_collected = int(serial.get('collected', 0))
        p_collected = int(parallel.get('collected', 0))
        print(f"Collected: Serial={s_collected:,}, Parallel={p_collected:,}")
        print()

        print(f"{'Phase':<15} | {'Serial':>10} | {'Parallel':>10} | {'Diff':>10}")
        print("-" * 55)

        for key in key_phases:
            name = key.replace('_ns', '')
            s = serial.get(key, 0)
            p = parallel.get(key, 0)
            diff = p - s
            print(f"{name:<15} | {s:>10.2f} | {p:>10.2f} | {diff:>+10.2f}")

        s_total = serial.get('total_ns', 1)
        p_total = parallel.get('total_ns', 1)
        speedup = s_total / p_total if p_total > 0 else 0
        print(f"\nSpeedup: {speedup:.2f}x (serial/parallel)")

        results.append({
            'creation_threads': ct,
            'serial_ms': s_total,
            'parallel_ms': p_total,
            'speedup': speedup,
        })

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Create T':<10} | {'Serial':>10} | {'Parallel':>10} | {'Speedup':>10}")
    print("-" * 50)
    for r in results:
        print(f"{r['creation_threads']:<10} | {r['serial_ms']:>10.1f} | {r['parallel_ms']:>10.1f} | {r['speedup']:>10.2f}x")


def show_all_phases(object_count, creation_threads, gc_workers):
    """Show all phases for a single run (useful for debugging)."""
    print("=" * 80)
    print("ALL PHASES (serial vs parallel)")
    print(f"Object count: {object_count:,}, Creation threads: {creation_threads}")
    print(f"GC workers: {gc_workers}")
    print("=" * 80)

    serial = run_subprocess_test('serial', gc_workers, creation_threads, object_count)
    parallel = run_subprocess_test('parallel', gc_workers, creation_threads, object_count)

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


def main():
    parser = argparse.ArgumentParser(description='Analyse GC creation thread impact')
    parser.add_argument('--threads', type=int, default=1,
                        help='Number of threads for object creation')
    parser.add_argument('--size', type=int, default=400000,
                        help='Number of objects to create')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel GC workers')
    parser.add_argument('--compare', action='store_true',
                        help='Compare different creation thread counts')
    parser.add_argument('--serial-vs-parallel', action='store_true',
                        help='Compare serial vs parallel GC')
    parser.add_argument('--subprocess', action='store_true',
                        help='Use subprocess isolation for clean state')
    parser.add_argument('--all-phases', action='store_true',
                        help='Show all phases (subprocess mode)')

    args = parser.parse_args()

    if args.all_phases:
        show_all_phases(args.size, args.threads, args.workers)
    elif args.serial_vs_parallel:
        if args.subprocess:
            compare_serial_vs_parallel_subprocess(args.size, [1, 2, 4], args.workers)
        else:
            print("WARNING: Using --subprocess is recommended for accurate comparison")
            compare_serial_vs_parallel_subprocess(args.size, [1, 2, 4], args.workers)
    elif args.compare:
        if args.workers > 0:
            enable_parallel_gc(args.workers)
            print(f"Parallel GC enabled with {args.workers} workers\n")
        compare_creation_threads(args.size, [1, 2, 4])
    else:
        if args.workers > 0:
            enable_parallel_gc(args.workers)
        run_gc_test(args.threads, args.size)


if __name__ == '__main__':
    main()
