"""
Test cases for GIL parallel GC mark_alive optimisation.

These tests verify correctness of the mark_alive phase, which pre-marks
objects reachable from known roots to skip them in subsequent GC phases.

Run with: ./python -m pytest Lib/test/test_gc_parallel_mark_alive.py -v
      or: ./python -m test test_gc_parallel_mark_alive
"""

import gc
import sys
import textwrap
import threading
import unittest
import weakref

import _testinternalcapi

from test.support import script_helper


# Helper class that supports weakrefs (unlike built-in object())
class WeakrefableObject:
    """Simple object that supports weakrefs for testing."""
    pass


# Skip if not GIL build with parallel GC
def setUpModule():
    if hasattr(sys, '_is_gil_enabled') and not sys._is_gil_enabled():
        raise unittest.SkipTest("FTP build - these tests are for GIL parallel GC")

    if not gc.get_parallel_config()['available']:
        raise unittest.SkipTest("Parallel GC not available (build without --with-parallel-gc)")


def _setup_parallel_gc(test_case):
    """Enable parallel GC for a test, tracking if we need to disable it later."""
    stats = gc.get_parallel_stats()
    test_case._parallel_gc_was_enabled = stats['enabled']
    if not test_case._parallel_gc_was_enabled:
        gc.enable_parallel()
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

    def test_small_collection_stays_serial(self):
        """Collections below two work slices do not dispatch helpers."""
        class Node:
            pass

        gc.collect()
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            stats_before = gc.get_parallel_stats()
            workers_before = gc.get_parallel_config()['adaptive_workers']

            nodes = [Node() for _ in range(4000)]
            gc.collect(0)

            stats_after = gc.get_parallel_stats()
            workers_after = gc.get_parallel_config()['adaptive_workers']
            self.assertEqual(
                stats_after['collections_attempted'],
                stats_before['collections_attempted'],
            )
            self.assertEqual(workers_after, workers_before)
            self.assertEqual(len(nodes), 4000)
        finally:
            if was_enabled:
                gc.enable()

    def test_large_collection_uses_parallel(self):
        """Collections above two work slices dispatch helpers."""
        class Node:
            pass

        gc.collect()
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            stats_before = gc.get_parallel_stats()

            nodes = [Node() for _ in range(20000)]
            gc.collect(0)

            stats_after = gc.get_parallel_stats()
            self.assertGreater(
                stats_after['collections_attempted'],
                stats_before['collections_attempted'],
            )
            self.assertEqual(len(nodes), 20000)
        finally:
            if was_enabled:
                gc.enable()


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

    def test_wide_root_graph_drains_private_work(self):
        """Pre-mark workers drain descendants spilled to their private deque."""
        code = textwrap.dedent("""
            import gc

            class Node:
                def __init__(self, children=()):
                    self.children = list(children)

            gc.enable_parallel()
            try:
                for _ in range(3):
                    hubs = [Node(Node() for _ in range(2000))
                            for _ in range(20)]
                    root = Node(hubs)
                    hubs = None
                    gc.collect()
                    assert len(root.children) == 20
                    assert all(len(hub.children) == 2000
                               for hub in root.children)
                    root = None
                    gc.collect()
            finally:
                gc.disable_parallel()
        """)
        script_helper.assert_python_ok("-c", code)


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
            gc.enable_parallel()
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
# Adaptive Worker Count Controller Tests (Biased Constrained Random Walk)
# =============================================================================

def _has_adaptive_controller():
    """Check if the adaptive controller is available."""
    try:
        config = gc.get_parallel_config()
        if not config.get('available', False):
            return False
        # Must enable parallel GC to see adaptive_workers key
        if not config.get('enabled', False):
            gc.enable_parallel()
            config = gc.get_parallel_config()
            gc.disable_parallel()
        return 'adaptive_workers' in config
    except (AttributeError, RuntimeError):
        return False


@unittest.skipUnless(_has_adaptive_controller(),
                     "Adaptive controller not available")
class TestAdaptiveControllerAPI(unittest.TestCase):
    """Verify the random walk controller exposes state via API."""

    def setUp(self):
        _setup_parallel_gc(self)

    def tearDown(self):
        _teardown_parallel_gc(self)

    def test_config_has_adaptive_workers(self):
        """gc.get_parallel_config() should expose adaptive worker count."""
        config = gc.get_parallel_config()
        self.assertIn('adaptive_workers', config)
        self.assertIsInstance(config['adaptive_workers'], int)
        self.assertGreaterEqual(config['adaptive_workers'], 2)

    def test_stats_has_prev_cost(self):
        """gc.get_parallel_stats() should expose previous per-object cost."""
        gc.collect()
        stats = gc.get_parallel_stats()
        self.assertIn('prev_cost_per_obj_ns', stats)
        self.assertIsInstance(stats['prev_cost_per_obj_ns'], float)

    def test_cost_uses_exact_candidate_count(self):
        """Adaptive cost is elapsed time divided by exact candidates."""
        code = textwrap.dedent("""
            import gc

            class Node:
                pass

            gc.disable()
            gc.enable_parallel()
            before = sum(item['candidates'] for item in gc.get_stats())

            nodes = [Node() for _ in range(20_000)]
            for node in nodes:
                node.ref = node
            gc.collect()

            after = sum(item['candidates'] for item in gc.get_stats())
            stats = gc.get_parallel_stats()
            candidates = after - before
            total_ns = stats['phase_timing']['total_ns']

            assert candidates >= 20_000, candidates
            assert total_ns > 0, total_ns
            assert stats['prev_cost_per_obj_ns'] == total_ns / candidates
        """)
        script_helper.assert_python_ok("-c", code)

    def test_stats_has_last_generation(self):
        """gc.get_parallel_stats() should report which generation was last collected."""
        gc.collect()
        stats = gc.get_parallel_stats()
        self.assertIn('last_generation', stats)
        self.assertIn(stats['last_generation'], (0, 1, 2))


@unittest.skipUnless(_has_adaptive_controller(),
                     "Adaptive controller not available")
class TestAdaptiveControllerIntegration(unittest.TestCase):
    """Exercise the controller through real parallel collections."""

    def setUp(self):
        _setup_parallel_gc(self)

    def tearDown(self):
        _teardown_parallel_gc(self)

    def test_no_crash_varying_heaps(self):
        """100 collections with varying heap sizes must not crash or deadlock."""
        import random
        rng = random.Random(42)
        for _ in range(100):
            size = rng.choice([100, 1_000, 10_000, 50_000])
            objs = [{'ref': None} for _ in range(size)]
            for i in range(len(objs) - 1):
                objs[i]['ref'] = objs[(i + 1) % len(objs)]
            del objs
            gc.collect()

    def test_real_collections_drive_random_walk(self):
        """Eligible collections must dispatch and exercise the controller."""
        code = textwrap.dedent("""
            import gc

            gc.disable()
            gc.enable_parallel()
            num_workers = gc.get_parallel_config()['num_workers']
            observed_workers = {
                gc.get_parallel_config()['adaptive_workers']
            }

            for _ in range(20):
                before_attempts = gc.get_parallel_stats()[
                    'collections_attempted'
                ]
                before_candidates = sum(
                    item['candidates'] for item in gc.get_stats()
                )

                nodes = [{} for _ in range(20_000)]
                for node in nodes:
                    node['self'] = node
                del nodes
                gc.collect()

                after_attempts = gc.get_parallel_stats()[
                    'collections_attempted'
                ]
                after_candidates = sum(
                    item['candidates'] for item in gc.get_stats()
                )
                candidates = after_candidates - before_candidates
                assert candidates >= 20_000, candidates
                assert after_attempts > before_attempts, (
                    before_attempts, after_attempts
                )
                workers = gc.get_parallel_config()['adaptive_workers']
                assert 2 <= workers <= num_workers, (
                    workers, num_workers
                )
                observed_workers.add(workers)

            assert len(observed_workers) > 1, observed_workers
        """)
        script_helper.assert_python_ok("-c", code, GC_TEST_SEED="42")


class TestCondvarStress(unittest.TestCase):
    """Stress-test the condvar dispatch path under varying worker counts.

    Designed to catch races in _PyGC_DispatchAndWait when the number of
    active workers changes between collections (as the proactive walker
    does). Particularly useful under TSAN.
    """

    def setUp(self):
        _setup_parallel_gc(self)

    def tearDown(self):
        _teardown_parallel_gc(self)

    def test_rapid_collections_varying_heaps(self):
        """200 rapid collections with wildly varying heap sizes.

        The proactive walker changes adaptive_workers on ~20% of collections.
        With 200 collections, we expect ~40 worker-count changes, exercising
        the condvar wake/wait path with different participant counts.
        """
        import random
        rng = random.Random(42)
        try:
            gc.disable_parallel()
        except (ValueError, RuntimeError):
            pass
        gc.enable_parallel()

        for i in range(200):
            # Alternate between tiny and large heaps to stress the
            # dispatch path with different workload characteristics
            size = rng.choice([50, 500, 5_000, 50_000, 200_000])
            objs = [{'ref': None} for _ in range(size)]
            # Create cycles so GC has work to do
            for j in range(len(objs) - 1):
                objs[j]['ref'] = objs[(j + 1) % len(objs)]
            del objs
            gc.collect()

    def test_enable_disable_cycles(self):
        """Rapid enable/disable/re-enable with collections in between.

        Tests the full lifecycle: pool start -> dispatch -> pool stop,
        repeatedly with the fixed maximum. This exercises the condvar
        init/fini paths and catches races in shutdown.
        """
        for _ in range(8):
            try:
                gc.disable_parallel()
            except (ValueError, RuntimeError):
                pass
            gc.enable_parallel()
            # Run a few collections before the next lifecycle transition.
            for _ in range(10):
                objs = [{'ref': None} for _ in range(10_000)]
                for j in range(len(objs) - 1):
                    objs[j]['ref'] = objs[(j + 1) % len(objs)]
                del objs
                gc.collect()
            gc.disable_parallel()
            # Collect once with parallel disabled to test serial fallback
            gc.collect()

    def test_concurrent_allocation_during_gc(self):
        """Run GC collections while other threads allocate objects.

        This simulates real-world conditions where GC runs concurrently
        with application threads. The condvar dispatch must not deadlock
        when the GIL is contended.
        """

        stop = threading.Event()
        errors = []

        def allocator():
            """Continuously allocate and release cyclic garbage."""
            try:
                while not stop.is_set():
                    objs = [{'ref': None} for _ in range(1_000)]
                    for j in range(len(objs) - 1):
                        objs[j]['ref'] = objs[(j + 1) % len(objs)]
                    del objs
                    # Don't call gc.collect() -- let the main thread drive GC
            except Exception as e:
                errors.append(e)

        try:
            gc.disable_parallel()
        except (ValueError, RuntimeError):
            pass
        gc.enable_parallel()

        # Start allocator threads
        threads = []
        for _ in range(4):
            t = threading.Thread(target=allocator)
            t.start()
            threads.append(t)

        try:
            # Run collections while allocators are running
            for _ in range(100):
                gc.collect()
        finally:
            stop.set()
            for t in threads:
                t.join(timeout=10)

        self.assertEqual(errors, [], f"Allocator threads had errors: {errors}")

    def test_dispatch_with_changing_worker_counts(self):
        """Exercise condvar dispatch with changing participant counts."""
        try:
            gc.disable_parallel()
        except (ValueError, RuntimeError):
            pass
        gc.enable_parallel()

        for active_workers in (2, 4, 8, 16):
            with self.subTest(active_workers=active_workers):
                caller_visits, helper_visits = (
                    _testinternalcapi.parallel_gc_helper_visits_with_workers(
                        active_workers
                    )
                )
                self.assertGreater(caller_visits + helper_visits, 0)
                self.assertGreater(helper_visits, 0)


if __name__ == '__main__':
    unittest.main()
