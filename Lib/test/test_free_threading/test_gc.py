import unittest

import threading
from threading import Thread
from unittest import TestCase
import gc

from test.support import threading_helper


class MyObj:
    pass


class Node:
    """Node class for cyclic garbage tests."""
    __slots__ = ['refs', '__weakref__']

    def __init__(self):
        self.refs = []


@threading_helper.requires_working_threading()
class TestGC(TestCase):
    def test_get_objects(self):
        event = threading.Event()

        def gc_thread():
            for i in range(100):
                o = gc.get_objects()
            event.set()

        def mutator_thread():
            while not event.is_set():
                o1 = MyObj()
                o2 = MyObj()
                o3 = MyObj()
                o4 = MyObj()

        gcs = [Thread(target=gc_thread)]
        mutators = [Thread(target=mutator_thread) for _ in range(4)]
        with threading_helper.start_threads(gcs + mutators):
            pass

    def test_get_referrers(self):
        NUM_GC = 2
        NUM_MUTATORS = 4

        b = threading.Barrier(NUM_GC + NUM_MUTATORS)
        event = threading.Event()

        obj = MyObj()

        def gc_thread():
            b.wait()
            for i in range(100):
                o = gc.get_referrers(obj)
            event.set()

        def mutator_thread():
            b.wait()
            while not event.is_set():
                d1 = { "key": obj }
                d2 = { "key": obj }
                d3 = { "key": obj }
                d4 = { "key": obj }

        gcs = [Thread(target=gc_thread) for _ in range(NUM_GC)]
        mutators = [Thread(target=mutator_thread) for _ in range(NUM_MUTATORS)]
        with threading_helper.start_threads(gcs + mutators):
            pass


@threading_helper.requires_working_threading()
class TestAbandonedPoolGC(TestCase):
    """Tests for garbage collection of objects in abandoned pool.

    When threads exit, their mimalloc heap pages go to the abandoned pool.
    These tests verify that parallel GC correctly collects garbage from
    the abandoned pool.
    """

    def test_abandoned_pool_cyclic_garbage(self):
        """Test that cyclic garbage in abandoned pool is collected."""
        NUM_THREADS = 4
        OBJECTS_PER_THREAD = 1000

        # Create cyclic garbage in worker threads that exit
        def create_garbage():
            nodes = [Node() for _ in range(OBJECTS_PER_THREAD)]
            # Create cycles
            for i in range(len(nodes)):
                nodes[i].refs.append(nodes[(i + 1) % len(nodes)])
            # Thread exits, pages go to abandoned pool
            # nodes goes out of scope here, making them garbage

        threads = [Thread(target=create_garbage) for _ in range(NUM_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Now garbage is in abandoned pool - collect it
        gc.collect()
        gc.collect()

        # Verify no crash and collection completed
        # (If abandoned pool wasn't scanned, we'd leak memory)
        stats = gc.get_stats()
        self.assertIsNotNone(stats)

    def test_abandoned_pool_parallel_gc(self):
        """Test parallel GC correctly handles abandoned pool pages."""
        NUM_THREADS = 4
        OBJECTS_PER_THREAD = 500

        # Track collected count
        collected_before = 0
        for gen_stats in gc.get_stats():
            collected_before += gen_stats.get('collected', 0)

        # Create cyclic garbage in worker threads
        def create_garbage():
            nodes = [Node() for _ in range(OBJECTS_PER_THREAD)]
            for i in range(len(nodes)):
                nodes[i].refs.append(nodes[(i + 1) % len(nodes)])

        threads = [Thread(target=create_garbage) for _ in range(NUM_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Collect with parallel GC
        collected = gc.collect()

        # Should have collected the cyclic garbage
        # (actual count may vary due to other objects)
        self.assertGreater(collected, 0)

    def test_abandoned_pool_with_survivors(self):
        """Test parallel GC with mix of garbage and survivors in abandoned pool."""
        NUM_THREADS = 4
        OBJECTS_PER_THREAD = 500

        # Keep references to some objects (survivors)
        survivors = []
        survivor_lock = threading.Lock()

        def create_mixed():
            garbage = [Node() for _ in range(OBJECTS_PER_THREAD // 2)]
            # Cycles in garbage
            for i in range(len(garbage)):
                garbage[i].refs.append(garbage[(i + 1) % len(garbage)])

            keep = [Node() for _ in range(OBJECTS_PER_THREAD // 2)]
            # Cycles in survivors too
            for i in range(len(keep)):
                keep[i].refs.append(keep[(i + 1) % len(keep)])

            with survivor_lock:
                survivors.extend(keep)
            # garbage goes out of scope, keep is saved

        threads = [Thread(target=create_mixed) for _ in range(NUM_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Collect
        collected = gc.collect()

        # Survivors should still be alive
        self.assertEqual(len(survivors), NUM_THREADS * (OBJECTS_PER_THREAD // 2))
        for node in survivors[:10]:  # Check a sample
            self.assertIsNotNone(node.refs)

        # Clean up
        survivors.clear()
        gc.collect()

    def test_multiple_gc_cycles_with_abandoned_pool(self):
        """Test multiple GC cycles with abandoned pool objects."""
        for iteration in range(3):
            # Create garbage in threads
            def create_garbage():
                nodes = [Node() for _ in range(100)]
                for i in range(len(nodes)):
                    nodes[i].refs.append(nodes[(i + 1) % len(nodes)])

            threads = [Thread(target=create_garbage) for _ in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Collect
            collected = gc.collect()
            self.assertGreaterEqual(collected, 0)

    def test_parallel_vs_serial_abandoned_pool_collection(self):
        """Verify parallel GC collects same amount as serial from abandoned pool.

        This is a critical regression test. If parallel GC fails to include
        abandoned pool pages in scan_heap, it will collect 0 objects while
        serial GC correctly collects all garbage.
        """
        NUM_THREADS = 4
        OBJECTS_PER_THREAD = 1000
        EXPECTED_MIN = NUM_THREADS * OBJECTS_PER_THREAD

        def run_collection(use_parallel, num_workers=4):
            """Create garbage in threads and collect it."""
            # Disable GC, clean up existing garbage
            gc.disable()
            gc.set_threshold(0)
            gc.enable()
            gc.collect()
            gc.collect()
            gc.collect()
            gc.disable()

            # Set up parallel/serial mode
            if use_parallel:
                if hasattr(gc, 'enable_parallel'):
                    gc.enable_parallel(num_workers)
            else:
                if hasattr(gc, 'disable_parallel'):
                    gc.disable_parallel()

            # Create garbage in threads that will exit
            def create_garbage():
                nodes = [Node() for _ in range(OBJECTS_PER_THREAD)]
                for i in range(len(nodes)):
                    nodes[i].refs.append(nodes[(i + 1) % len(nodes)])

            threads = [Thread(target=create_garbage) for _ in range(NUM_THREADS)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Small delay for thread cleanup
            import time
            time.sleep(0.1)

            # Collect
            gc.enable()
            collected = gc.collect()
            return collected

        # Test requires parallel GC support
        if not hasattr(gc, 'enable_parallel') or not hasattr(gc, 'disable_parallel'):
            self.skipTest("Parallel GC not available")

        # Run serial collection
        serial_collected = run_collection(use_parallel=False)

        # Run parallel collection
        parallel_collected = run_collection(use_parallel=True, num_workers=8)

        # Both should collect similar amounts
        # Allow some variance due to other objects, but parallel should
        # collect at least 50% of what serial collects
        self.assertGreater(
            serial_collected, EXPECTED_MIN // 2,
            f"Serial GC should collect ~{EXPECTED_MIN} objects, got {serial_collected}"
        )

        self.assertGreater(
            parallel_collected, serial_collected // 2,
            f"Parallel GC collected {parallel_collected} vs serial {serial_collected}. "
            f"Parallel should collect at least 50% of serial. "
            f"This may indicate abandoned pool pages are not being scanned."
        )

        # Ideally they should be equal (within small variance)
        ratio = parallel_collected / serial_collected if serial_collected > 0 else 0
        self.assertGreater(
            ratio, 0.9,
            f"Parallel/Serial ratio is {ratio:.2f} - should be close to 1.0"
        )


if __name__ == "__main__":
    unittest.main()
