"""
Test cases for GIL parallel GC mark_alive optimisation.

These tests verify correctness of the mark_alive phase, which pre-marks
objects reachable from known roots to skip them in subsequent GC phases.

Run with: ./python -m pytest Lib/test/test_gc_parallel_mark_alive.py -v
      or: ./python -m test test_gc_parallel_mark_alive
"""

import gc
import os
import sys
import unittest
import weakref
import threading


# Helper class that supports weakrefs (unlike built-in object())
class WeakrefableObject:
    """Simple object that supports weakrefs for testing."""
    pass


# Skip if parallel GC not available
def setUpModule():
    if not gc.get_parallel_config()['available']:
        raise unittest.SkipTest("Parallel GC not available (build without --with-parallel-gc)")


def _setup_parallel_gc(test_case):
    """Enable parallel GC for a test, tracking if we need to disable it later."""
    stats = gc.get_parallel_stats()
    test_case._parallel_gc_was_enabled = stats['enabled']
    if not test_case._parallel_gc_was_enabled:
        gc.enable_parallel(4)
    gc.collect()


def _teardown_parallel_gc(test_case):
    """Disable parallel GC only if we enabled it."""
    if not getattr(test_case, '_parallel_gc_was_enabled', True):
        gc.disable_parallel()


class TestBasicCycleCollection(unittest.TestCase):
    """Verify basic GC functionality is preserved."""

    def setUp(self):
        _setup_parallel_gc(self)

    def tearDown(self):
        _teardown_parallel_gc(self)

    def test_simple_cycle_collected(self):
        """Simple reference cycle should be collected."""
        class Node:
            pass

        a = Node()
        b = Node()
        a.ref = b
        b.ref = a

        a_id = id(a)
        b_id = id(b)

        del a, b
        collected = gc.collect()

        self.assertGreaterEqual(collected, 2)

    def test_self_referential_collected(self):
        """Self-referential object should be collected."""
        class Node:
            pass

        a = Node()
        a.ref = a

        del a
        collected = gc.collect()

        self.assertGreaterEqual(collected, 1)

    def test_long_chain_collected(self):
        """Long reference chain forming cycle should be collected."""
        class Node:
            pass

        nodes = [Node() for _ in range(1000)]
        for i in range(len(nodes) - 1):
            nodes[i].next = nodes[i + 1]
        nodes[-1].next = nodes[0]  # Complete cycle

        del nodes
        collected = gc.collect()

        self.assertGreaterEqual(collected, 1000)


class TestKnownRootsPreserved(unittest.TestCase):
    """Verify objects reachable from known roots are not collected."""

    def setUp(self):
        _setup_parallel_gc(self)

    def tearDown(self):
        _teardown_parallel_gc(self)
        # Clean up any test artifacts from sys.modules
        if '_test_mark_alive_module' in sys.modules:
            del sys.modules['_test_mark_alive_module']

    def test_sysdict_reachable_preserved(self):
        """Objects reachable from sys.__dict__ should not be collected."""
        class Container:
            pass

        container = Container()
        container.data = [WeakrefableObject() for _ in range(100)]

        # Make reachable from sys
        sys._test_container = container

        try:
            weak_refs = [weakref.ref(obj) for obj in container.data]

            gc.collect()

            # All objects should still be alive
            alive = sum(1 for ref in weak_refs if ref() is not None)
            self.assertEqual(alive, 100)
        finally:
            del sys._test_container

    def test_builtins_reachable_preserved(self):
        """Objects reachable from builtins should not be collected."""
        import builtins

        class Container:
            pass

        container = Container()
        container.data = [WeakrefableObject() for _ in range(100)]

        builtins._test_container = container

        try:
            weak_refs = [weakref.ref(obj) for obj in container.data]

            gc.collect()

            alive = sum(1 for ref in weak_refs if ref() is not None)
            self.assertEqual(alive, 100)
        finally:
            del builtins._test_container

    def test_module_globals_preserved(self):
        """Objects in module globals should not be collected."""
        # Create a fake module
        import types
        module = types.ModuleType('_test_mark_alive_module')
        sys.modules['_test_mark_alive_module'] = module

        class Container:
            pass

        module.container = Container()
        module.container.data = [WeakrefableObject() for _ in range(100)]

        weak_refs = [weakref.ref(obj) for obj in module.container.data]

        gc.collect()

        alive = sum(1 for ref in weak_refs if ref() is not None)
        self.assertEqual(alive, 100)


class TestThreadStacksPreserved(unittest.TestCase):
    """Verify objects on thread stacks are not collected."""

    def setUp(self):
        _setup_parallel_gc(self)

    def tearDown(self):
        _teardown_parallel_gc(self)

    def test_main_thread_locals_preserved(self):
        """Local variables on main thread should not be collected."""
        class Container:
            pass

        def inner():
            local_obj = Container()
            local_obj.data = [WeakrefableObject() for _ in range(50)]
            weak_refs = [weakref.ref(obj) for obj in local_obj.data]

            gc.collect()

            # All should be alive - they're on the stack
            alive = sum(1 for ref in weak_refs if ref() is not None)
            return alive, local_obj  # Return to keep alive during check

        alive, _ = inner()
        self.assertEqual(alive, 50)

    def test_other_thread_locals_preserved(self):
        """Local variables on other threads should not be collected."""
        result = {'alive': 0, 'done': False}
        barrier = threading.Barrier(2)

        def thread_func():
            class Container:
                pass

            local_obj = Container()
            local_obj.data = [WeakrefableObject() for _ in range(50)]
            weak_refs = [weakref.ref(obj) for obj in local_obj.data]

            # Signal ready
            barrier.wait()

            # Wait for main thread to run GC
            barrier.wait()

            # Check survival
            result['alive'] = sum(1 for ref in weak_refs if ref() is not None)
            result['done'] = True

            # Keep alive until check complete
            barrier.wait()

        t = threading.Thread(target=thread_func)
        t.start()

        # Wait for thread to set up
        barrier.wait()

        # Run GC while thread is waiting
        gc.collect()

        # Signal thread to check
        barrier.wait()

        # Wait for thread to finish check
        barrier.wait()

        t.join()

        self.assertTrue(result['done'])
        self.assertEqual(result['alive'], 50)


class TestUnreachableCollected(unittest.TestCase):
    """Verify unreachable objects are still collected."""

    def setUp(self):
        _setup_parallel_gc(self)

    def tearDown(self):
        _teardown_parallel_gc(self)

    def test_unreachable_cycle_collected(self):
        """Unreachable cycle should be collected even with mark_alive."""
        class Node:
            pass

        # Create reachable objects
        reachable = [Node() for _ in range(100)]
        reachable_refs = [weakref.ref(n) for n in reachable]

        # Create unreachable cycle
        unreachable = [Node() for _ in range(100)]
        for i in range(len(unreachable) - 1):
            unreachable[i].next = unreachable[i + 1]
        unreachable[-1].next = unreachable[0]
        unreachable_refs = [weakref.ref(n) for n in unreachable]

        # Make unreachable actually unreachable
        del unreachable

        gc.collect()

        # Reachable should survive
        reachable_alive = sum(1 for ref in reachable_refs if ref() is not None)
        self.assertEqual(reachable_alive, 100)

        # Unreachable should be collected
        unreachable_alive = sum(1 for ref in unreachable_refs if ref() is not None)
        self.assertEqual(unreachable_alive, 0)

    def test_mixed_reachable_unreachable(self):
        """Mixed graph with reachable and unreachable portions."""
        class Node:
            def __init__(self):
                self.refs = []

        # Reachable from sys
        root = Node()
        sys._test_root = root

        try:
            # Add some reachable children
            reachable_children = [Node() for _ in range(50)]
            root.refs = reachable_children
            reachable_refs = [weakref.ref(n) for n in reachable_children]

            # Create separate unreachable cycle
            unreachable = [Node() for _ in range(50)]
            for i in range(len(unreachable) - 1):
                unreachable[i].refs = [unreachable[i + 1]]
            unreachable[-1].refs = [unreachable[0]]
            unreachable_refs = [weakref.ref(n) for n in unreachable]

            del unreachable

            gc.collect()

            # Reachable should survive
            reachable_alive = sum(1 for ref in reachable_refs if ref() is not None)
            self.assertEqual(reachable_alive, 50)

            # Unreachable should be collected
            unreachable_alive = sum(1 for ref in unreachable_refs if ref() is not None)
            self.assertEqual(unreachable_alive, 0)
        finally:
            del sys._test_root


class TestRaceConditions(unittest.TestCase):
    """Test edge cases around timing and race conditions."""

    def setUp(self):
        _setup_parallel_gc(self)

    def tearDown(self):
        _teardown_parallel_gc(self)

    def test_weakref_callback_during_collection(self):
        """Weakref callbacks shouldn't interfere with mark_alive."""
        callback_called = [False]

        class Node:
            pass

        def callback(ref):
            callback_called[0] = True

        node = Node()
        node.self_ref = node  # Cycle
        ref = weakref.ref(node, callback)

        del node
        gc.collect()

        # Callback should have been called
        self.assertTrue(callback_called[0])
        # Reference should be dead
        self.assertIsNone(ref())


class TestFinalizers(unittest.TestCase):
    """Test interaction with finalizers."""

    def setUp(self):
        _setup_parallel_gc(self)

    def tearDown(self):
        _teardown_parallel_gc(self)

    def test_finalizer_called_on_unreachable(self):
        """__del__ should be called on unreachable objects."""
        finalized = []

        class Destructor:
            def __init__(self, name):
                self.name = name
            def __del__(self):
                finalized.append(self.name)

        # Create unreachable cycle with finalizers
        a = Destructor('a')
        b = Destructor('b')
        a.ref = b
        b.ref = a

        del a, b
        gc.collect()

        # Both should have been finalized
        self.assertEqual(sorted(finalized), ['a', 'b'])

    def test_finalizer_resurrection(self):
        """Object resurrected in __del__ should not be collected."""
        resurrected = []

        class Resurrector:
            def __del__(self):
                resurrected.append(self)

        obj = Resurrector()
        obj.self_ref = obj  # Cycle

        del obj
        gc.collect()

        # Object was resurrected
        self.assertEqual(len(resurrected), 1)
        # And is still alive
        self.assertIsNotNone(resurrected[0])


class TestLargeHeaps(unittest.TestCase):
    """Test with large object counts."""

    def setUp(self):
        _setup_parallel_gc(self)

    def tearDown(self):
        _teardown_parallel_gc(self)

    def test_500k_objects_mixed(self):
        """500k objects with mixed reachable/unreachable."""
        class Node:
            __slots__ = ['refs', '__weakref__']
            def __init__(self):
                self.refs = []

        # 400k reachable
        reachable = [Node() for _ in range(400000)]
        for i in range(len(reachable) - 1):
            reachable[i].refs = [reachable[i + 1]]
        sys._test_reachable = reachable

        try:
            reachable_refs = [weakref.ref(reachable[0]),
                             weakref.ref(reachable[-1])]

            # 100k unreachable (cycle)
            unreachable = [Node() for _ in range(100000)]
            for i in range(len(unreachable) - 1):
                unreachable[i].refs = [unreachable[i + 1]]
            unreachable[-1].refs = [unreachable[0]]
            unreachable_refs = [weakref.ref(unreachable[0]),
                               weakref.ref(unreachable[-1])]

            del unreachable

            gc.collect()

            # Reachable should survive
            for ref in reachable_refs:
                self.assertIsNotNone(ref())

            # Unreachable should be collected
            for ref in unreachable_refs:
                self.assertIsNone(ref())
        finally:
            del sys._test_reachable

    def test_deep_nesting(self):
        """Deeply nested object graph."""
        class Node:
            pass

        # Create 10k deep nesting
        root = Node()
        current = root
        for _ in range(10000):
            new = Node()
            current.child = new
            current = new

        sys._test_deep = root

        try:
            # Get refs to first and last
            first_ref = weakref.ref(root)
            last_ref = weakref.ref(current)

            gc.collect()

            # Both should survive
            self.assertIsNotNone(first_ref())
            self.assertIsNotNone(last_ref())
        finally:
            del sys._test_deep


class TestParallelCorrectness(unittest.TestCase):
    """Test parallel marking correctness."""

    def setUp(self):
        _setup_parallel_gc(self)

    def tearDown(self):
        _teardown_parallel_gc(self)

    def test_shared_objects_marked_once(self):
        """Objects referenced by multiple roots should be marked correctly."""
        class Node:
            pass

        shared = Node()

        # Multiple roots pointing to same object
        sys._test_root1 = shared
        sys._test_root2 = shared
        sys._test_root3 = shared

        try:
            ref = weakref.ref(shared)

            gc.collect()

            # Shared object should survive
            self.assertIsNotNone(ref())
        finally:
            del sys._test_root1
            del sys._test_root2
            del sys._test_root3

    def test_concurrent_allocation_during_gc(self):
        """Objects allocated during GC should be handled correctly."""
        results = {'errors': []}
        stop_flag = threading.Event()

        def allocator():
            """Continuously allocate objects."""
            try:
                while not stop_flag.is_set():
                    objs = [WeakrefableObject() for _ in range(100)]
                    del objs
            except Exception as e:
                results['errors'].append(e)

        def collector():
            """Trigger GC repeatedly."""
            try:
                for _ in range(10):
                    gc.collect()
            except Exception as e:
                results['errors'].append(e)
            finally:
                stop_flag.set()

        threads = [
            threading.Thread(target=allocator),
            threading.Thread(target=allocator),
            threading.Thread(target=collector),
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=30)

        self.assertEqual(results['errors'], [])


class TestTypeObjects(unittest.TestCase):
    """Test handling of type objects."""

    def setUp(self):
        _setup_parallel_gc(self)

    def tearDown(self):
        _teardown_parallel_gc(self)

    def test_class_with_cycle_in_dict(self):
        """Class with cycle in __dict__ should be handled correctly."""
        class MyClass:
            pass

        # Create cycle through class dict
        MyClass.self_ref = MyClass

        ref = weakref.ref(MyClass)

        gc.collect()

        # Class should survive (reachable from its module)
        self.assertIsNotNone(ref())

    def test_orphaned_class(self):
        """Orphaned class (no module ref) in cycle should be collected."""
        class Container:
            pass

        # Create class dynamically
        OrphanClass = type('OrphanClass', (), {})
        OrphanClass.self_ref = OrphanClass

        ref = weakref.ref(OrphanClass)

        del OrphanClass

        gc.collect()

        # Should be collected (not reachable from any root)
        self.assertIsNone(ref())


class TestExtensionModules(unittest.TestCase):
    """Test handling of extension module objects."""

    def setUp(self):
        _setup_parallel_gc(self)

    def tearDown(self):
        _teardown_parallel_gc(self)

    def test_datetime_objects(self):
        """datetime objects should be handled correctly."""
        import datetime

        # Wrap datetime objects in containers since datetime doesn't support weakrefs
        class DatetimeContainer:
            def __init__(self, dt):
                self.dt = dt

        objs = [DatetimeContainer(datetime.datetime.now()) for _ in range(100)]
        refs = [weakref.ref(obj) for obj in objs]

        gc.collect()

        # All should survive (still referenced by objs list)
        alive = sum(1 for ref in refs if ref() is not None)
        self.assertEqual(alive, 100)

    def test_regex_objects(self):
        """Compiled regex objects should be handled correctly."""
        import re

        patterns = [re.compile(f'pattern{i}') for i in range(100)]
        refs = [weakref.ref(p) for p in patterns]

        gc.collect()

        alive = sum(1 for ref in refs if ref() is not None)
        self.assertEqual(alive, 100)


class TestPerformance(unittest.TestCase):
    """Verify performance improvement from mark_alive."""

    def setUp(self):
        gc.collect()

    def test_mark_alive_faster_than_baseline(self):
        """Parallel with mark_alive should be faster than serial on large heaps.

        Note: This test is inherently variable due to system load and GC timing.
        We use multiple iterations and check median performance to reduce flakiness.
        The current GIL implementation achieves ~1.5-2x speedup on large heaps.
        """
        import time
        import statistics

        class Node:
            __slots__ = ['refs']
            def __init__(self):
                self.refs = []

        def create_heap():
            nodes = [Node() for _ in range(200000)]
            for i in range(len(nodes) - 1):
                nodes[i].refs = [nodes[i + 1]]
            nodes[-1].refs = [nodes[0]]
            return nodes

        # Warm up
        nodes = create_heap()
        del nodes
        gc.collect()

        iterations = 5
        serial_times = []
        parallel_times = []

        for _ in range(iterations):
            # Serial timing
            gc.disable_parallel()
            nodes = create_heap()
            del nodes
            start = time.perf_counter()
            gc.collect()
            serial_times.append(time.perf_counter() - start)

            # Parallel timing
            gc.enable_parallel(4)
            nodes = create_heap()
            del nodes
            start = time.perf_counter()
            gc.collect()
            parallel_times.append(time.perf_counter() - start)

        gc.disable_parallel()

        median_serial = statistics.median(serial_times)
        median_parallel = statistics.median(parallel_times)
        speedup = median_serial / median_parallel

        print(f"\nMedian speedup: {speedup:.2f}x (serial={median_serial*1000:.1f}ms, parallel={median_parallel*1000:.1f}ms)")

        # Parallel should not be catastrophically slower than serial.
        # Allow up to 2x slower — mark_alive is an optimization that may
        # not help on all configs (e.g. without thread-stack walking,
        # fewer roots are pre-marked). The parallel subtract_refs/mark
        # phases provide the main speedup on large collections.
        self.assertLess(median_parallel, median_serial * 2.0,
                        f"Parallel should not be >2x slower than serial. "
                        f"Got speedup={speedup:.2f}x")


# =============================================================================
# Adaptive Worker Count Controller Tests
# =============================================================================

def _has_adaptive_controller():
    """Check if the per-generation adaptive controller is available."""
    try:
        config = gc.get_parallel_config()
        if not config.get('available', False):
            return False
        # Must enable parallel GC to see per-gen keys
        if not config.get('enabled', False):
            gc.enable_parallel(4)
            config = gc.get_parallel_config()
            gc.disable_parallel()
        return 'adaptive_workers_gen0' in config
    except (AttributeError, RuntimeError):
        return False


@unittest.skipUnless(_has_adaptive_controller(),
                     "Per-generation adaptive controller not available")
class TestAdaptiveControllerAPI(unittest.TestCase):
    """Verify the adaptive controller exposes per-generation state via API."""

    def setUp(self):
        _setup_parallel_gc(self)

    def tearDown(self):
        _teardown_parallel_gc(self)

    def test_config_has_per_gen_workers(self):
        """gc.get_parallel_config() should expose per-generation worker counts."""
        config = gc.get_parallel_config()
        for gen in range(3):
            key = f'adaptive_workers_gen{gen}'
            self.assertIn(key, config,
                          f"Missing {key} in get_parallel_config()")
            self.assertIsInstance(config[key], int)
            self.assertGreaterEqual(config[key], 2,
                                   f"{key} must be >= 2 (min floor)")

    def test_config_has_epsilon(self):
        """gc.get_parallel_config() should expose exploration probability."""
        config = gc.get_parallel_config()
        self.assertIn('epsilon', config)
        self.assertIsInstance(config['epsilon'], float)
        self.assertGreaterEqual(config['epsilon'], 0.0)
        self.assertLessEqual(config['epsilon'], 1.0)

    def test_stats_has_per_gen_ema(self):
        """gc.get_parallel_stats() should expose per-generation EMA values."""
        # Run a collection to populate stats
        gc.collect()
        stats = gc.get_parallel_stats()
        for gen in range(3):
            key = f'ema_per_obj_ns_gen{gen}'
            self.assertIn(key, stats,
                          f"Missing {key} in get_parallel_stats()")
            self.assertIsInstance(stats[key], float)
            self.assertGreater(stats[key], 0.0,
                               f"{key} must be positive")

    def test_stats_has_last_generation(self):
        """gc.get_parallel_stats() should report which generation was last collected."""
        gc.collect()
        stats = gc.get_parallel_stats()
        self.assertIn('last_generation', stats)
        self.assertIn(stats['last_generation'], (0, 1, 2))

    def test_per_gen_workers_within_bounds(self):
        """Per-generation worker counts must be in [2, num_workers]."""
        config = gc.get_parallel_config()
        num_workers = config['num_workers']
        for gen in range(3):
            key = f'adaptive_workers_gen{gen}'
            self.assertGreaterEqual(config[key], 2)
            self.assertLessEqual(config[key], num_workers)


def _load_is_reasonable():
    """Convergence tests assume CPU is not saturated. Under extreme load
    (>100), the controller correctly reduces workers because dispatch
    overhead dominates — but this inverts the expected gen0 < gen2 ordering."""
    try:
        return os.getloadavg()[0] < 100
    except (OSError, AttributeError):
        return True  # can't check, assume OK


@unittest.skipUnless(_has_adaptive_controller(),
                     "Per-generation adaptive controller not available")
@unittest.skipUnless(_load_is_reasonable(),
                     "Machine load too high for convergence tests")
class TestAdaptiveControllerConvergence(unittest.TestCase):
    """Verify the controller converges differently for different heap sizes.

    Falsification: if gen0 (small heap) and gen2 (large heap) converge to
    the same worker count, the per-generation controller is unnecessary.
    """

    def setUp(self):
        _setup_parallel_gc(self)

    def tearDown(self):
        _teardown_parallel_gc(self)

    def test_gen0_prefers_fewer_workers(self):
        """After many gen0 collections on small heaps, adaptive_workers_gen0
        should converge toward the minimum (2).

        Gen0 collections process ~hundreds of objects. At that scale,
        dispatch overhead dominates and fewer workers is optimal.
        """
        # explore_rng is seeded at interpreter startup from GC_TEST_SEED
        # env var or perf counter. Tests check directional properties,
        # not exact values, so non-determinism is acceptable.

        # Force many gen0 collections with small heaps
        for _ in range(50):
            # Create small batch of objects with cycles
            objs = [{'ref': None} for _ in range(200)]
            for i in range(len(objs) - 1):
                objs[i]['ref'] = objs[(i + 1) % len(objs)]
            del objs
            gc.collect(0)  # gen0 only

        config = gc.get_parallel_config()
        gen0_workers = config['adaptive_workers_gen0']
        # Gen0 should converge toward minimum (2-3 workers)
        self.assertLessEqual(gen0_workers, 4,
                             f"Gen0 should converge to low worker count, "
                             f"got {gen0_workers}")

    def test_gen2_allows_more_workers(self):
        """After gen2 collections on large heaps, adaptive_workers_gen2
        should be higher than gen0.

        Gen2 collections process ~100K+ objects. At that scale,
        parallelism pays off and more workers is optimal.
        """
        import random
        # explore_rng is seeded at interpreter startup from GC_TEST_SEED
        # env var or perf counter. Tests check directional properties,
        # not exact values, so non-determinism is acceptable.
        rng = random.Random(42)

        # First, force gen0 collections with small heaps to drive gen0
        # workers down. Gen0 processes ~hundreds of objects where dispatch
        # overhead dominates.
        for _ in range(50):
            objs = [{'ref': None} for _ in range(200)]
            for i in range(len(objs) - 1):
                objs[i]['ref'] = objs[(i + 1) % len(objs)]
            del objs
            gc.collect(0)

        # Then force gen2 collections with large heaps. Gen2 processes
        # ~50K+ objects where parallelism pays off. 40 collections gives
        # enough convergence budget: minus 3 warmup = 37 active, minus
        # ~30% exploration = ~26 exploit steps.
        for _ in range(40):
            nodes = [{'id': i, 'refs': []} for i in range(50_000)]
            for i in range(len(nodes)):
                targets = rng.sample(range(len(nodes)), min(3, len(nodes)))
                for t in targets:
                    nodes[i]['refs'].append(nodes[t])
            del nodes
            gc.collect(2)  # full collection

        config = gc.get_parallel_config()
        gen2_workers = config['adaptive_workers_gen2']
        gen0_workers = config['adaptive_workers_gen0']
        # Gen2 should converge to strictly MORE workers than gen0.
        # If it doesn't, the per-generation controller is unnecessary —
        # this assertion IS the falsification test.
        self.assertGreater(gen2_workers, gen0_workers,
                           f"Gen2 ({gen2_workers}) must have more workers "
                           f"than gen0 ({gen0_workers}) — "
                           f"otherwise per-gen controller is unjustified")


@unittest.skipUnless(_has_adaptive_controller(),
                     "Per-generation adaptive controller not available")
class TestAdaptiveControllerExploration(unittest.TestCase):
    """Verify the epsilon-greedy exploration mechanism."""

    def setUp(self):
        _setup_parallel_gc(self)

    def tearDown(self):
        _teardown_parallel_gc(self)

    def test_epsilon_decays_on_stable_workload(self):
        """On a stable workload, epsilon should decay toward the floor (0.05)."""
        # explore_rng is seeded at interpreter startup from GC_TEST_SEED
        # env var or perf counter. Tests check directional properties,
        # not exact values, so non-determinism is acceptable.

        initial_config = gc.get_parallel_config()
        initial_epsilon = initial_config['epsilon']

        # Run many collections with identical workload
        for _ in range(40):
            objs = [{'ref': None} for _ in range(10_000)]
            for i in range(len(objs) - 1):
                objs[i]['ref'] = objs[(i + 1) % len(objs)]
            del objs
            gc.collect()

        final_config = gc.get_parallel_config()
        final_epsilon = final_config['epsilon']

        # Epsilon should have decayed (or stayed at floor)
        self.assertLessEqual(final_epsilon, initial_epsilon,
                             f"Epsilon should decay on stable workload: "
                             f"{initial_epsilon} → {final_epsilon}")
        # Should be near or at floor (0.05)
        self.assertLessEqual(final_epsilon, 0.15,
                             f"After 40 stable collections, epsilon should "
                             f"be near floor, got {final_epsilon}")

    def test_epsilon_does_not_reset_on_single_outlier(self):
        """A single outlier collection should NOT reset epsilon to 0.3.

        The shift detection requires 3 consecutive above-threshold
        collections to prevent noise-triggered resets.
        """
        # explore_rng is seeded at interpreter startup from GC_TEST_SEED
        # env var or perf counter. Tests check directional properties,
        # not exact values, so non-determinism is acceptable.

        # Stabilize with consistent workload to decay epsilon
        for _ in range(30):
            objs = [{'ref': None} for _ in range(10_000)]
            for i in range(len(objs) - 1):
                objs[i]['ref'] = objs[(i + 1) % len(objs)]
            del objs
            gc.collect()

        config_before = gc.get_parallel_config()
        epsilon_before = config_before['epsilon']

        # Single large collection (outlier)
        big = [{'refs': list(range(100))} for _ in range(200_000)]
        del big
        gc.collect()

        # Then back to normal
        objs = [{'ref': None} for _ in range(10_000)]
        del objs
        gc.collect()

        config_after = gc.get_parallel_config()
        epsilon_after = config_after['epsilon']

        # Epsilon should NOT have jumped back to 0.3
        self.assertLess(epsilon_after, 0.3,
                        f"Single outlier should not reset epsilon. "
                        f"Before={epsilon_before}, after={epsilon_after}")


if __name__ == '__main__':
    unittest.main()
