#!/usr/bin/env python3
"""
Test script to capture atomic operation patterns for specific heap types.
Runs GC collections on each heap type individually to compare atomic patterns.
"""

import gc
import sys

# Heap generators from the benchmark
def make_chain(n):
    """Linear chain: each object references the next."""
    head = {}
    current = head
    for _ in range(n - 1):
        new = {}
        current['next'] = new
        current = new
    return [head]

def make_independent(n):
    """Many small independent clusters (100 objects each)."""
    cluster_size = 100
    clusters = []
    for _ in range(n // cluster_size):
        head = {}
        current = head
        for _ in range(cluster_size - 1):
            new = {}
            current['next'] = new
            current = new
        clusters.append(head)
    return clusters

def make_tree(n):
    """Binary tree structure."""
    def build_tree(depth):
        if depth == 0:
            return {}
        return {'left': build_tree(depth - 1), 'right': build_tree(depth - 1)}
    # Approximate depth for n objects: 2^depth - 1 = n, so depth = log2(n+1)
    import math
    depth = int(math.log2(n + 1))
    return [build_tree(depth)]

def make_web_server(n):
    """Simulates web server: mix of long-lived and short-lived objects."""
    sessions = []
    num_sessions = n // 50  # 50 objects per session
    for _ in range(num_sessions):
        session = {'user': {}, 'request': {}, 'response': {}}
        session['cache'] = [{'data': {}} for _ in range(47)]
        sessions.append(session)
    return sessions

HEAP_TYPES = {
    'chain': make_chain,
    'independent': make_independent,
    'tree': make_tree,
    'web_server': make_web_server,
}

def run_collection_test(heap_type, heap_size=100000, num_runs=3):
    """Run GC collections on specific heap type and print stats."""
    print(f"\n{'='*60}")
    print(f"Heap type: {heap_type} ({heap_size:,} objects, {num_runs} runs)")
    print('='*60)

    generator = HEAP_TYPES[heap_type]

    for run in range(num_runs):
        # Clear everything
        gc.collect()
        gc.collect()
        gc.collect()

        # Enable parallel GC
        gc.enable_parallel(8)

        # Create heap
        print(f"\n--- Run {run + 1} ---")
        heap = generator(heap_size)

        # Force collection - this should print atomic stats
        gc.collect()

        # Clear heap
        del heap

    # Disable parallel
    gc.disable_parallel()

def main():
    print("Parallel GC Atomic Pattern Analysis")
    print("="*60)
    print(f"Python: {sys.version}")
    print(f"GIL disabled: {getattr(sys, '_is_gil_enabled', lambda: True)() == False}")

    # Check parallel GC is available
    if not hasattr(gc, 'enable_parallel'):
        print("ERROR: gc.enable_parallel not available")
        return 1

    heap_size = 100000  # Smaller for faster iteration

    # Test heap types - focus on independent (problem case) vs chain/web_server (speedup cases)
    for heap_type in ['independent', 'chain', 'web_server']:
        run_collection_test(heap_type, heap_size, num_runs=3)

    print("\n" + "="*60)
    print("Analysis complete")
    return 0

if __name__ == '__main__':
    sys.exit(main())
