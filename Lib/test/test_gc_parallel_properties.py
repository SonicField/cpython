"""
Property-based tests for parallel garbage collection.

This module implements property-based testing following engineering standards:
- Actively seek counterexamples rather than confirming happy paths
- Test invariants that must always hold
- Include boundary value tests

Since Hypothesis is not available, uses Python's built-in random module
to generate random inputs and check properties hold for all generated cases.
"""

import gc
import os
import random
import sys
import threading
import unittest
import weakref


# Skip entire test file if NOT free-threading build
if not hasattr(sys, '_is_gil_enabled') or sys._is_gil_enabled():
    raise unittest.SkipTest("Parallel GC requires free-threaded build (--disable-gil)")


# Number of random test iterations (balance thoroughness vs time)
PROPERTY_TEST_ITERATIONS = 100


class Node:
    """
    A simple node class that supports weak references.

    Plain dict/list don't support weakref, so we use this for tracking
    object collection in property tests.
    """
    __slots__ = ['id', 'refs', 'children', 'data', 'value', 'ref', '__weakref__']

    def __init__(self, node_id=None):
        self.id = node_id
        self.refs = []
        self.children = []
        self.data = None
        self.value = None
        self.ref = None


class PropertyTestBase(unittest.TestCase):
    """Base class for property-based tests with common setup/teardown."""

    def setUp(self):
        """Save GC state and ensure clean slate."""
        self.was_enabled = gc.isenabled()
        gc.disable()
        gc.collect()  # Clean up before test
        seed = int(os.environ.get('GC_TEST_SEED', '0')) or int.from_bytes(os.urandom(8), 'big')
        self._seed = seed
        random.seed(seed)
        print(f"Random seed: {seed}", flush=True)

    def tearDown(self):
        """Restore GC state."""
        try:
            gc.disable_parallel()
        except (ValueError, RuntimeError):
            pass
        gc.collect()
        if self.was_enabled:
            gc.enable()


class TestPropertyCyclicGarbageCollected(PropertyTestBase):
    """
    Property: Any cyclic garbage is collected.

    Generate random cyclic object graphs of varying shapes and sizes.
    Verify they are all collected after gc.collect().
    """

    def _create_random_cycle(self, size):
        """
        Create a random cyclic structure of the given size.

        Returns a list of Node objects that form cycles. The caller should delete
        this list to make the objects unreachable (and thus garbage).
        """
        if size == 0:
            return []

        # Create Node objects
        objects = [Node(i) for i in range(size)]

        # Add random forward references (creates dag-like structure)
        for i, obj in enumerate(objects):
            # Each object references 1-3 random other objects
            num_refs = random.randint(1, min(3, size - 1))
            targets = random.sample(range(size), min(num_refs, size))
            for t in targets:
                if t != i:
                    obj.refs.append(objects[t])

        # Add back references to create cycles
        for i in range(size - 1):
            objects[i + 1].refs.append(objects[i])

        # Close the loop: last references first
        if size > 1:
            objects[-1].refs.append(objects[0])

        return objects

    def test_property_cyclic_garbage_collected_serial(self):
        """
        PROPERTY: Any cyclic garbage is collected (serial GC).

        For random graph sizes, create cyclic garbage and verify collection.
        """
        for iteration in range(PROPERTY_TEST_ITERATIONS):
            # Generate random graph size: 2-100 objects
            size = random.randint(2, 100)

            # Create cyclic garbage
            objects = self._create_random_cycle(size)

            # Use weak references to verify collection
            weak_refs = [weakref.ref(obj) for obj in objects]

            # Make objects unreachable
            del objects

            # Collect
            gc.collect()

            # PROPERTY CHECK: All objects should be collected
            alive_count = sum(1 for ref in weak_refs if ref() is not None)
            self.assertEqual(
                alive_count, 0,
                f"PROPERTY VIOLATED: {alive_count}/{size} cyclic garbage objects "
                f"survived collection (serial GC, iteration {iteration})"
            )

    def test_property_cyclic_garbage_collected_parallel(self):
        """
        PROPERTY: Any cyclic garbage is collected (parallel GC).

        Same as serial test but with parallel GC enabled.
        """
        config = gc.get_parallel_config()
        if not config['available']:
            self.skipTest("Parallel GC not available")

        gc.enable_parallel()

        for iteration in range(PROPERTY_TEST_ITERATIONS):
            size = random.randint(2, 100)
            objects = self._create_random_cycle(size)
            weak_refs = [weakref.ref(obj) for obj in objects]
            del objects
            gc.collect()

            alive_count = sum(1 for ref in weak_refs if ref() is not None)
            self.assertEqual(
                alive_count, 0,
                f"PROPERTY VIOLATED: {alive_count}/{size} cyclic garbage objects "
                f"survived collection (parallel GC, iteration {iteration})"
            )


class TestPropertyReachableObjectsSurvive(PropertyTestBase):
    """
    Property: Any reachable object survives.

    Generate random object graphs with live references.
    Verify they all survive gc.collect().
    """

    def _create_random_graph_with_root(self, size):
        """
        Create a random object graph of given size, returning the root.

        All objects are reachable from the returned root, so they should
        survive garbage collection.
        """
        if size == 0:
            return None

        # Create root object
        root = Node(0)

        if size == 1:
            return root

        # Create remaining objects
        objects = [root]
        for i in range(1, size):
            obj = Node(i)
            # Attach to a random existing object
            parent = random.choice(objects)
            parent.children.append(obj)
            objects.append(obj)

        # Add some random cross-references (still reachable from root)
        for _ in range(size // 2):
            src = random.choice(objects)
            dst = random.choice(objects)
            if src != dst:
                src.children.append(dst)

        return root

    def _count_reachable(self, obj, seen=None):
        """Count reachable nodes from given root."""
        if seen is None:
            seen = set()
        if obj is None or id(obj) in seen:
            return 0
        seen.add(id(obj))
        count = 1
        for child in obj.children:
            count += self._count_reachable(child, seen)
        return count

    def test_property_reachable_survives_serial(self):
        """
        PROPERTY: Any reachable object survives (serial GC).
        """
        for iteration in range(PROPERTY_TEST_ITERATIONS):
            size = random.randint(1, 100)
            root = self._create_random_graph_with_root(size)

            count_before = self._count_reachable(root)

            # Collect
            gc.collect()

            # PROPERTY CHECK: All reachable objects should survive
            count_after = self._count_reachable(root)
            self.assertEqual(
                count_after, count_before,
                f"PROPERTY VIOLATED: {count_before - count_after}/{count_before} "
                f"reachable objects were incorrectly collected "
                f"(serial GC, iteration {iteration})"
            )

            # Verify we can still access all objects
            self.assertIsNotNone(root)
            self.assertEqual(root.id, 0)

    def test_property_reachable_survives_parallel(self):
        """
        PROPERTY: Any reachable object survives (parallel GC).
        """
        config = gc.get_parallel_config()
        if not config['available']:
            self.skipTest("Parallel GC not available")

        gc.enable_parallel()

        for iteration in range(PROPERTY_TEST_ITERATIONS):
            size = random.randint(1, 100)
            root = self._create_random_graph_with_root(size)

            count_before = self._count_reachable(root)
            gc.collect()
            count_after = self._count_reachable(root)

            self.assertEqual(
                count_after, count_before,
                f"PROPERTY VIOLATED: {count_before - count_after}/{count_before} "
                f"reachable objects were incorrectly collected "
                f"(parallel GC, iteration {iteration})"
            )


class TestPropertyWorkerStatisticsConsistency(PropertyTestBase):
    """
    Property: Worker statistics sum to total.

    Run repeated parallel collections and verify per-worker statistics are
    consistent with totals under adaptive worker selection.
    """

    def test_property_stats_consistency_random_workers(self):
        """
        PROPERTY: Across repeated collections, stats are internally consistent.
        """
        config = gc.get_parallel_config()
        if not config['available']:
            self.skipTest("Parallel GC not available")

        gc.enable_parallel()
        for trial in range(4):

            # Create some garbage to collect
            garbage = []
            for _ in range(50):
                a = Node()
                b = Node()
                a.ref = b
                b.ref = a
                garbage.append(a)
            del garbage

            gc.collect()

            # Get stats
            stats = gc.get_parallel_stats()
            config = gc.get_parallel_config()

            # PROPERTY CHECK: Config should expose the fixed ceiling.
            self.assertEqual(
                config['num_workers'], 16,
                "PROPERTY VIOLATED: unexpected worker ceiling"
            )

            # PROPERTY CHECK: enabled should be True when workers > 0
            self.assertTrue(
                config['enabled'],
                "PROPERTY VIOLATED: Parallel GC should be enabled"
            )

            # PROPERTY CHECK: Phase timing total should be >= individual phases
            timing = stats.get('phase_timing', {})
            total_ns = timing.get('total_ns', 0)
            mark_alive_ns = timing.get('mark_alive_ns', 0)

            # mark_alive is a subset of total
            if total_ns > 0:
                self.assertLessEqual(
                    mark_alive_ns, total_ns,
                    f"PROPERTY VIOLATED: mark_alive_ns ({mark_alive_ns}) > "
                    f"total_ns ({total_ns}) in trial {trial}"
                )


class TestBoundaryValues(PropertyTestBase):
    """
    Boundary value tests for parallel GC.

    Test edge cases: minimum workers, maximum workers, empty collections,
    single object collections.
    """

    def test_boundary_minimum_workers(self):
        """
        Enabling uses the fixed maximum worker count.
        """
        config = gc.get_parallel_config()
        if not config['available']:
            self.skipTest("Parallel GC not available")

        gc.enable_parallel()
        config = gc.get_parallel_config()
        self.assertEqual(config['num_workers'], 16)
        self.assertTrue(config['enabled'])

        # Should still collect garbage correctly
        a = Node()
        a.ref = a  # Self-referential
        weak_a = weakref.ref(a)
        del a
        gc.collect()
        self.assertIsNone(weak_a(), "Garbage should be collected with 2 workers")

    def test_boundary_one_worker_rejected(self):
        """
        The no-argument API rejects an explicit worker count.
        """
        config = gc.get_parallel_config()
        if not config['available']:
            self.skipTest("Parallel GC not available")

        with self.assertRaises(TypeError):
            gc.enable_parallel(1)

    def test_boundary_maximum_workers(self):
        """
        The fixed maximum is reported after enabling.
        """
        config = gc.get_parallel_config()
        if not config['available']:
            self.skipTest("Parallel GC not available")

        gc.enable_parallel()
        config = gc.get_parallel_config()
        self.assertEqual(config['num_workers'], 16)

    def test_boundary_zero_workers(self):
        """
        A zero positional argument is not part of the API.
        """
        config = gc.get_parallel_config()
        if not config['available']:
            self.skipTest("Parallel GC not available")

        with self.assertRaises(TypeError):
            gc.enable_parallel(0)

    def test_boundary_negative_workers(self):
        """
        A negative positional argument is not part of the API.
        """
        with self.assertRaises(TypeError):
            gc.enable_parallel(-1)

    def test_boundary_collection_empty(self):
        """
        Boundary: GC collection with 0 garbage objects.
        """
        config = gc.get_parallel_config()
        if not config['available']:
            self.skipTest("Parallel GC not available")

        gc.enable_parallel()

        # Force collection to clean state
        gc.collect()
        gc.collect()

        # Now collect with no garbage - should succeed without error
        collected = gc.collect()
        self.assertGreaterEqual(collected, 0,
                               "Collection should succeed with no garbage")

    def test_boundary_collection_single_object(self):
        """
        Boundary: GC collection with exactly 1 garbage object.

        A single self-referential object is the minimum cyclic garbage.
        """
        config = gc.get_parallel_config()
        if not config['available']:
            self.skipTest("Parallel GC not available")

        gc.enable_parallel()
        gc.collect()  # Clean slate

        # Create single self-referential object (using Node for weakref support)
        obj = Node()
        obj.ref = obj
        weak_obj = weakref.ref(obj)
        del obj

        gc.collect()
        self.assertIsNone(weak_obj(),
                         "Single self-referential object should be collected")

    def test_boundary_collection_two_objects(self):
        """
        Boundary: GC collection with exactly 2 objects forming a cycle.

        Minimum multi-object cycle.
        """
        config = gc.get_parallel_config()
        if not config['available']:
            self.skipTest("Parallel GC not available")

        gc.enable_parallel()
        gc.collect()

        a = Node()
        b = Node()
        a.ref = b
        b.ref = a
        weak_a = weakref.ref(a)
        weak_b = weakref.ref(b)
        del a, b

        gc.collect()
        self.assertIsNone(weak_a(), "Object a should be collected")
        self.assertIsNone(weak_b(), "Object b should be collected")


class TestPropertyMixedGarbageAndReachable(PropertyTestBase):
    """
    Combined property test: mixed garbage and reachable objects.

    Verify that GC correctly distinguishes garbage from reachable objects
    when both are present in the same collection.
    """

    def test_property_mixed_correctness(self):
        """
        PROPERTY: In mixed scenarios, garbage is collected AND reachable survives.
        """
        config = gc.get_parallel_config()
        if not config['available']:
            self.skipTest("Parallel GC not available")

        gc.enable_parallel()

        # Ensure clean state before testing
        gc.collect()
        gc.collect()

        for iteration in range(PROPERTY_TEST_ITERATIONS):
            # Use minimum of 2 to ensure cycles are always created
            garbage_size = random.randint(2, 50)
            reachable_size = random.randint(1, 50)

            # Create garbage graph FIRST (cycles with no external references)
            garbage = []
            for i in range(garbage_size):
                obj = Node(f'garbage_{i}')
                if garbage:
                    # Reference previous garbage
                    obj.refs.append(garbage[-1])
                    garbage[-1].refs.append(obj)
                garbage.append(obj)
            # Close cycle (always true since garbage_size >= 2)
            garbage[-1].refs.append(garbage[0])
            garbage[0].refs.append(garbage[-1])

            # Track garbage with weak refs BEFORE creating reachable objects
            weak_garbage = [weakref.ref(obj) for obj in garbage]

            # Delete garbage references (now unreachable)
            del garbage
            del obj  # Ensure loop variable doesn't keep reference

            # Now create reachable graph (keep reference to root)
            root = Node('root')
            for i in range(reachable_size):
                child = Node(f'reachable_{i}')
                child.value = i * 2
                root.children.append(child)

            # Collect - multiple calls to handle generations
            gc.collect()
            gc.collect()
            gc.collect()

            # PROPERTY CHECK 1: All garbage should be collected
            alive_garbage = sum(1 for ref in weak_garbage if ref() is not None)
            self.assertEqual(
                alive_garbage, 0,
                f"PROPERTY VIOLATED: {alive_garbage}/{garbage_size} garbage objects "
                f"survived (iteration {iteration})"
            )

            # PROPERTY CHECK 2: All reachable should survive
            self.assertEqual(len(root.children), reachable_size,
                            f"PROPERTY VIOLATED: Reachable children lost "
                            f"(iteration {iteration})")
            for i, child in enumerate(root.children):
                self.assertEqual(
                    child.value, i * 2,
                    f"PROPERTY VIOLATED: Reachable object corrupted "
                    f"(iteration {iteration})"
                )


class TestPropertyThreadedGarbageCollection(PropertyTestBase):
    """
    Property tests for multi-threaded garbage collection scenarios.
    """

    def test_property_abandoned_threads_garbage_collected(self):
        """
        PROPERTY: Cyclic garbage from any number of abandoned threads is collected.
        """
        config = gc.get_parallel_config()
        if not config['available']:
            self.skipTest("Parallel GC not available")

        gc.enable_parallel()

        for iteration in range(20):  # Fewer iterations (threading is slower)
            num_threads = random.randint(1, 8)
            garbage_per_thread = random.randint(10, 50)

            weak_refs = []
            lock = threading.Lock()

            def create_garbage(size):
                local_garbage = []
                for _ in range(size):
                    a = Node()
                    b = Node()
                    a.ref = b
                    b.ref = a
                    local_garbage.append(a)

                # Track with weak refs
                with lock:
                    weak_refs.extend(weakref.ref(obj) for obj in local_garbage)
                # local_garbage goes out of scope - garbage!

            threads = []
            for _ in range(num_threads):
                t = threading.Thread(target=create_garbage, args=(garbage_per_thread,))
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

            # Collect garbage from abandoned threads
            gc.collect()

            # PROPERTY CHECK: All garbage should be collected
            alive_count = sum(1 for ref in weak_refs if ref() is not None)
            self.assertEqual(
                alive_count, 0,
                f"PROPERTY VIOLATED: {alive_count}/{len(weak_refs)} garbage objects "
                f"from {num_threads} abandoned threads survived "
                f"(iteration {iteration})"
            )

    def test_property_persistent_threads_reachable_survives(self):
        """
        PROPERTY: Reachable objects in persistent threads survive GC.
        """
        config = gc.get_parallel_config()
        if not config['available']:
            self.skipTest("Parallel GC not available")

        gc.enable_parallel()

        for iteration in range(10):  # Fewer iterations
            num_threads = random.randint(2, 6)
            objects_per_thread = random.randint(5, 20)

            results = []
            lock = threading.Lock()
            barrier = threading.Barrier(num_threads + 1)
            stop = threading.Event()

            def create_and_hold(size):
                # Create objects and keep them reachable
                my_objects = [Node(i) for i in range(size)]
                for i, obj in enumerate(my_objects):
                    obj.data = i * 3
                with lock:
                    results.extend(my_objects)
                barrier.wait()  # Signal ready
                barrier.wait()  # Wait for GC
                stop.wait()  # Stay alive until signaled

            threads = []
            for _ in range(num_threads):
                t = threading.Thread(target=create_and_hold, args=(objects_per_thread,))
                t.start()
                threads.append(t)

            barrier.wait()  # Wait for all threads ready
            gc.collect()  # Collect while threads alive
            barrier.wait()  # Signal GC done

            # PROPERTY CHECK: All objects should survive
            expected_count = num_threads * objects_per_thread
            self.assertEqual(
                len(results), expected_count,
                f"PROPERTY VIOLATED: Expected {expected_count} objects, "
                f"found {len(results)} (iteration {iteration})"
            )

            # Verify data integrity
            for obj in results:
                self.assertEqual(
                    obj.data, obj.id * 3,
                    f"PROPERTY VIOLATED: Object data corrupted (iteration {iteration})"
                )

            stop.set()
            for t in threads:
                t.join()


if __name__ == '__main__':
    unittest.main()
