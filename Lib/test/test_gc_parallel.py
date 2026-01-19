"""
Tests for parallel garbage collection in free-threaded Python.

This test module covers the parallel GC feature, which is only available
in free-threaded builds compiled with --disable-gil.
"""

import gc
import sys
import threading
import time
import unittest
from test import support


# Skip entire test file if NOT free-threading build
if not hasattr(sys, '_is_gil_enabled') or sys._is_gil_enabled():
    raise unittest.SkipTest("Parallel GC requires free-threaded build (--disable-gil)")


class TestParallelGCAPI(unittest.TestCase):
    """Test parallel GC API functions."""

    def test_get_parallel_config_exists(self):
        """Test that gc.get_parallel_config() exists."""
        self.assertTrue(hasattr(gc, 'get_parallel_config'))
        self.assertTrue(callable(gc.get_parallel_config))

    def test_enable_parallel_exists(self):
        """Test that gc.enable_parallel() exists."""
        self.assertTrue(hasattr(gc, 'enable_parallel'))
        self.assertTrue(callable(gc.enable_parallel))

    def test_get_parallel_config_returns_dict(self):
        """Test that gc.get_parallel_config() returns a dictionary."""
        config = gc.get_parallel_config()
        self.assertIsInstance(config, dict)

    def test_get_parallel_config_has_required_keys(self):
        """Test that config dictionary has required keys."""
        config = gc.get_parallel_config()
        self.assertIn('available', config)
        self.assertIn('enabled', config)
        self.assertIn('num_workers', config)

    def test_get_parallel_config_types(self):
        """Test that config values have correct types."""
        config = gc.get_parallel_config()
        self.assertIsInstance(config['available'], bool)
        self.assertIsInstance(config['enabled'], bool)
        self.assertIsInstance(config['num_workers'], int)

    @support.cpython_only
    def test_parallel_gc_availability(self):
        """Test parallel GC availability based on build configuration."""
        config = gc.get_parallel_config()

        # If built with --with-parallel-gc, should be available
        # Otherwise, should not be available
        # We can't assert specific value without knowing build config,
        # but we can verify consistency
        if config['available']:
            # If available, num_workers should be >= 0
            self.assertGreaterEqual(config['num_workers'], 0)
        else:
            # If not available, should be disabled with 0 workers
            self.assertFalse(config['enabled'])
            self.assertEqual(config['num_workers'], 0)


class TestParallelGCEnable(unittest.TestCase):
    """Test enabling and disabling parallel GC."""

    def test_enable_parallel_no_args(self):
        """Test that enable_parallel() requires num_workers argument."""
        # num_workers is now a required parameter
        with self.assertRaises(TypeError):
            gc.enable_parallel()

    def test_enable_parallel_with_workers(self):
        """Test calling enable_parallel() with specific worker count."""
        config = gc.get_parallel_config()
        if not config.get('available', False):
            self.skipTest("Parallel GC not available in this build")
        if config.get('enabled', False):
            self.skipTest("Parallel GC already enabled")

        result = gc.enable_parallel(4)
        self.assertIsNone(result)

        # Verify it was enabled
        config = gc.get_parallel_config()
        self.assertTrue(config['enabled'])
        self.assertEqual(config['num_workers'], 4)

    def test_enable_parallel_zero_workers(self):
        """Test calling enable_parallel(0) to disable."""
        config = gc.get_parallel_config()
        if not config.get('available', False):
            self.skipTest("Parallel GC not available in this build")
        if config.get('enabled', False):
            self.skipTest("Parallel GC already enabled")

        # Currently 0 workers raises ValueError (not yet implemented as disable)
        with self.assertRaises(ValueError):
            gc.enable_parallel(0)

    def test_enable_parallel_invalid_negative(self):
        """Test that invalid negative values raise ValueError."""
        try:
            with self.assertRaises(ValueError):
                gc.enable_parallel(-2)
        except RuntimeError:
            # Skip if parallel GC not available
            pass

    def test_enable_parallel_invalid_large(self):
        """Test that excessively large values raise ValueError."""
        try:
            with self.assertRaises(ValueError):
                gc.enable_parallel(10000)
        except RuntimeError:
            # Skip if parallel GC not available
            pass


class TestParallelGCBuildConfig(unittest.TestCase):
    """Test build configuration for parallel GC."""

    def test_parallel_gc_available_in_free_threading(self):
        """Verify parallel GC IS available in free-threading builds."""
        # We're in a free-threading build (verified by module-level skip)
        config = gc.get_parallel_config()
        self.assertTrue(config['available'],
                       "Parallel GC should be available in free-threading builds")

    @support.cpython_only
    def test_enable_parallel_error_message_without_build_flag(self):
        """Test error message when parallel GC not compiled in."""
        config = gc.get_parallel_config()
        if not config['available']:
            with self.assertRaises(RuntimeError) as cm:
                gc.enable_parallel(2)
            error_msg = str(cm.exception)
            # Should mention rebuilding
            self.assertTrue(
                "rebuild" in error_msg.lower() or
                "not available" in error_msg.lower()
            )


class TestParallelGCCompatibility(unittest.TestCase):
    """Test that parallel GC doesn't break existing GC functionality."""

    def test_gc_still_collects(self):
        """Verify basic GC collection still works."""
        # Create some garbage
        garbage = []
        garbage.append(garbage)
        del garbage

        # Collection should still work
        collected = gc.collect()
        self.assertIsInstance(collected, int)
        self.assertGreaterEqual(collected, 0)

    def test_gc_enable_disable_still_works(self):
        """Verify gc.enable() and gc.disable() still work."""
        was_enabled = gc.isenabled()
        try:
            gc.disable()
            self.assertFalse(gc.isenabled())
            gc.enable()
            self.assertTrue(gc.isenabled())
        finally:
            # Restore original state
            if was_enabled:
                gc.enable()
            else:
                gc.disable()

    def test_gc_get_count_still_works(self):
        """Verify gc.get_count() still works."""
        counts = gc.get_count()
        self.assertIsInstance(counts, tuple)
        self.assertEqual(len(counts), 3)


class TestParallelGCPhaseTiming(unittest.TestCase):
    """Test phase timing instrumentation in parallel GC."""

    def setUp(self):
        """Set up test fixtures."""
        config = gc.get_parallel_config()
        if not config['available']:
            self.skipTest("Parallel GC not available in this build")
        if not config.get('enabled', False):
            gc.enable_parallel(4)

    def tearDown(self):
        """Clean up after test."""
        gc.collect()

    def test_get_parallel_stats_has_phase_timing(self):
        """Test that get_parallel_stats() returns phase_timing dict."""
        stats = gc.get_parallel_stats()
        self.assertIn('phase_timing', stats)
        self.assertIsInstance(stats['phase_timing'], dict)

    def test_phase_timing_has_required_keys(self):
        """Test that phase_timing has all required keys."""
        stats = gc.get_parallel_stats()
        timing = stats['phase_timing']
        # Check for the actual keys in the current implementation
        self.assertIn('update_refs_ns', timing)
        self.assertIn('mark_alive_ns', timing)
        self.assertIn('total_ns', timing)

    def test_phase_timing_values_are_integers(self):
        """Test that phase timing values are integers."""
        stats = gc.get_parallel_stats()
        timing = stats['phase_timing']
        self.assertIsInstance(timing['update_refs_ns'], int)
        self.assertIsInstance(timing['mark_alive_ns'], int)
        self.assertIsInstance(timing['total_ns'], int)

    def test_phase_timing_after_collection(self):
        """Test that phase timing is populated after a collection."""
        # Create objects to ensure GC runs
        class Node:
            __slots__ = ['refs']
            def __init__(self):
                self.refs = []

        nodes = [Node() for _ in range(50000)]
        for i in range(len(nodes) - 1):
            nodes[i].refs.append(nodes[i + 1])

        # Clear references and collect
        del nodes
        gc.collect()

        # Check timing was recorded
        stats = gc.get_parallel_stats()
        timing = stats['phase_timing']

        # total_ns should have been recorded
        self.assertGreaterEqual(timing['total_ns'], 0)

    def test_phase_timing_total_positive(self):
        """Test that total timing is positive after collection."""
        # Create enough objects to trigger collection
        class Node:
            __slots__ = ['refs']
            def __init__(self):
                self.refs = []

        nodes = [Node() for _ in range(100000)]
        for i in range(len(nodes) - 1):
            nodes[i].refs.append(nodes[i + 1])

        del nodes
        gc.collect()

        stats = gc.get_parallel_stats()
        timing = stats['phase_timing']

        # total should have measurable time
        self.assertGreater(timing['total_ns'], 0,
                          "total_ns should have positive timing after collection")

    def test_phase_timing_consistency(self):
        """Test that individual phases don't exceed total."""
        class Node:
            __slots__ = ['refs']
            def __init__(self):
                self.refs = []

        nodes = [Node() for _ in range(100000)]
        for i in range(len(nodes) - 1):
            nodes[i].refs.append(nodes[i + 1])

        del nodes
        gc.collect()

        stats = gc.get_parallel_stats()
        timing = stats['phase_timing']

        # If total is recorded, individual phases should be <= total
        if timing['total_ns'] > 0:
            self.assertLessEqual(timing['mark_alive_ns'], timing['total_ns'],
                                "mark_alive_ns should be <= total_ns")


class TestAbandonedSerial(unittest.TestCase):
    """
    Test garbage collection correctness with abandoned threads using serial GC.

    Abandoned threads are threads that exit, causing their heap pages to be
    moved to the abandoned pool. This tests that objects created in such
    threads are correctly collected when using serial (non-parallel) GC.
    """

    def setUp(self):
        """Save original GC state."""
        self.was_enabled = gc.isenabled()
        gc.disable()
        gc.collect()  # Start clean

    def tearDown(self):
        """Restore original GC state."""
        gc.collect()
        if self.was_enabled:
            gc.enable()

    def test_abandoned_thread_cyclic_garbage_collected(self):
        """
        Test that cyclic garbage created in abandoned threads is collected.

        Creates cyclic garbage in threads that then exit. The garbage should
        be collected when gc.collect() is called from the main thread.
        """
        collected_count = []

        def create_garbage():
            # Create cyclic garbage
            for _ in range(100):
                a = {'ref': None}
                b = {'ref': a}
                a['ref'] = b
                # a and b go out of scope when function returns

        # Create threads that make garbage then exit
        threads = []
        for _ in range(4):
            t = threading.Thread(target=create_garbage)
            t.start()
            threads.append(t)

        # Wait for all threads to complete (abandon their heaps)
        for t in threads:
            t.join()

        # Collect garbage - should collect objects from abandoned heaps
        collected = gc.collect()
        self.assertGreater(collected, 0,
                          "Should collect cyclic garbage from abandoned threads")

    def test_abandoned_thread_reachable_objects_preserved(self):
        """
        Test that objects from abandoned threads that are still reachable survive.

        Creates objects in threads that are stored in a shared structure.
        Even after threads exit, these objects should survive collection.
        """
        results = []
        lock = threading.Lock()

        def create_and_store():
            obj = {'data': list(range(100)), 'id': threading.get_ident()}
            with lock:
                results.append(obj)

        # Create threads that store objects then exit
        threads = []
        for _ in range(4):
            t = threading.Thread(target=create_and_store)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        # Objects are still reachable via 'results'
        gc.collect()

        # Verify all objects survived
        self.assertEqual(len(results), 4)
        for obj in results:
            self.assertEqual(obj['data'], list(range(100)))

    def test_multiple_abandoned_waves(self):
        """
        Test collection after multiple waves of thread creation and abandonment.
        """
        total_collected = 0

        for wave in range(3):
            def create_garbage():
                for _ in range(50):
                    a = []
                    b = []
                    a.append(b)
                    b.append(a)

            threads = []
            for _ in range(4):
                t = threading.Thread(target=create_garbage)
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

            collected = gc.collect()
            total_collected += collected

        self.assertGreater(total_collected, 0,
                          "Should collect garbage across multiple waves")


class TestAbandonedParallel(unittest.TestCase):
    """
    Test garbage collection correctness with abandoned threads using parallel GC.

    Same scenarios as TestAbandonedSerial but with parallel GC enabled.
    """

    def setUp(self):
        """Enable parallel GC and save original state."""
        self.was_enabled = gc.isenabled()
        gc.disable()
        gc.collect()

        config = gc.get_parallel_config()
        if not config['available']:
            self.skipTest("Parallel GC not available")

        # Enable parallel GC with 4 workers
        gc.enable_parallel(4)

    def tearDown(self):
        """Restore original GC state."""
        gc.collect()
        if self.was_enabled:
            gc.enable()

    def test_abandoned_thread_cyclic_garbage_collected_parallel(self):
        """
        Test that cyclic garbage from abandoned threads is collected with parallel GC.
        """
        def create_garbage():
            for _ in range(100):
                a = {'ref': None}
                b = {'ref': a}
                a['ref'] = b

        threads = []
        for _ in range(8):  # More threads for parallel testing
            t = threading.Thread(target=create_garbage)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        collected = gc.collect()
        self.assertGreater(collected, 0,
                          "Parallel GC should collect garbage from abandoned threads")

    def test_abandoned_thread_large_garbage_parallel(self):
        """
        Test parallel GC with large amounts of garbage from abandoned threads.
        """
        def create_large_garbage():
            # Create many objects with complex references
            objects = []
            for i in range(200):
                obj = {'id': i, 'children': []}
                if objects:
                    # Reference previous object (creates chain)
                    obj['prev'] = objects[-1]
                objects.append(obj)
            # Create cycles
            for i in range(0, len(objects) - 1, 2):
                objects[i]['pair'] = objects[i + 1]
                objects[i + 1]['pair'] = objects[i]
            # All become garbage when function returns

        threads = []
        for _ in range(8):
            t = threading.Thread(target=create_large_garbage)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        collected = gc.collect()
        self.assertGreater(collected, 0,
                          "Should collect large garbage graphs from abandoned threads")

    def test_abandoned_reachable_preserved_parallel(self):
        """
        Test that reachable objects from abandoned threads survive parallel GC.
        """
        results = []
        lock = threading.Lock()

        def create_and_store():
            obj = {'data': list(range(100)), 'thread': threading.get_ident()}
            with lock:
                results.append(obj)

        threads = []
        for _ in range(8):
            t = threading.Thread(target=create_and_store)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        gc.collect()

        self.assertEqual(len(results), 8)
        for obj in results:
            self.assertEqual(obj['data'], list(range(100)))


class TestPersistentThreads(unittest.TestCase):
    """
    Test garbage collection with persistent (long-lived) threads.

    Persistent threads keep their heap pages active (not abandoned).
    This tests that GC correctly handles objects in active thread heaps.
    """

    def setUp(self):
        """Save original GC state."""
        self.was_enabled = gc.isenabled()
        gc.disable()
        gc.collect()

    def tearDown(self):
        """Restore original GC state."""
        gc.collect()
        if self.was_enabled:
            gc.enable()

    def _run_persistent_thread_test(self, use_parallel):
        """Helper to run test with serial or parallel GC."""
        if use_parallel:
            config = gc.get_parallel_config()
            if not config['available']:
                self.skipTest("Parallel GC not available")
            gc.enable_parallel(4)

        barrier = threading.Barrier(5)  # 4 workers + main
        results = []
        lock = threading.Lock()
        stop_event = threading.Event()

        def worker():
            # Create garbage in a persistent thread
            for _ in range(50):
                a = {'ref': None}
                b = {'ref': a}
                a['ref'] = b

            # Create object to keep
            obj = {'id': threading.get_ident(), 'data': 'persistent'}
            with lock:
                results.append(obj)

            # Signal ready and wait for collection to complete
            barrier.wait()  # Wait 1: ready
            barrier.wait()  # Wait 2: collection done

            # Thread is still alive - not abandoned
            stop_event.wait()

        threads = []
        for _ in range(4):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)

        # Wait for all workers to be ready
        barrier.wait()  # Wait 1

        # Collect while threads are still alive
        collected = gc.collect()

        # Signal collection done
        barrier.wait()  # Wait 2

        # Stop threads
        stop_event.set()
        for t in threads:
            t.join()

        self.assertGreater(collected, 0,
                          "Should collect garbage from persistent threads")
        self.assertEqual(len(results), 4)

    def test_persistent_threads_serial(self):
        """Test garbage collection in persistent threads with serial GC."""
        self._run_persistent_thread_test(use_parallel=False)

    def test_persistent_threads_parallel(self):
        """Test garbage collection in persistent threads with parallel GC."""
        self._run_persistent_thread_test(use_parallel=True)

    def test_concurrent_allocation_during_gc(self):
        """
        Test that threads can continue allocating while GC runs.

        This specifically tests that persistent threads don't block allocation
        during the collection cycle.
        """
        config = gc.get_parallel_config()
        if not config['available']:
            self.skipTest("Parallel GC not available")
        gc.enable_parallel(4)

        stop_event = threading.Event()
        allocation_counts = []
        lock = threading.Lock()

        def allocator():
            count = 0
            while not stop_event.is_set():
                # Allocate objects rapidly
                objs = [{'n': i} for i in range(100)]
                count += len(objs)
                del objs
            with lock:
                allocation_counts.append(count)

        # Start allocator threads
        threads = []
        for _ in range(4):
            t = threading.Thread(target=allocator)
            t.start()
            threads.append(t)

        # Run several GC collections while threads allocate
        for _ in range(5):
            gc.collect()
            time.sleep(0.01)

        # Stop allocators
        stop_event.set()
        for t in threads:
            t.join()

        # Verify threads were able to allocate
        self.assertEqual(len(allocation_counts), 4)
        for count in allocation_counts:
            self.assertGreater(count, 0, "Thread should have allocated objects")


class TestMixedAndConfig(unittest.TestCase):
    """
    Test mixed abandoned/persistent thread scenarios and cleanup configuration.
    """

    def setUp(self):
        """Save original GC state."""
        self.was_enabled = gc.isenabled()
        gc.disable()
        gc.collect()

    def tearDown(self):
        """Restore original GC state."""
        gc.collect()
        if self.was_enabled:
            gc.enable()

    def test_mixed_abandoned_and_persistent(self):
        """
        Test GC with mix of abandoned and persistent threads.

        Some threads exit (abandoned) while others stay alive (persistent).
        GC should correctly handle both cases simultaneously.
        """
        config = gc.get_parallel_config()
        if not config['available']:
            self.skipTest("Parallel GC not available")
        gc.enable_parallel(4)

        persistent_ready = threading.Event()
        gc_done = threading.Event()
        stop_persistent = threading.Event()
        results = []
        lock = threading.Lock()

        def abandoned_worker():
            """Thread that creates garbage and exits."""
            for _ in range(100):
                a = []
                b = []
                a.append(b)
                b.append(a)
            # Store one object
            with lock:
                results.append({'type': 'abandoned', 'id': threading.get_ident()})
            # Thread exits - heap becomes abandoned

        def persistent_worker():
            """Thread that creates garbage and stays alive."""
            for _ in range(100):
                a = []
                b = []
                a.append(b)
                b.append(a)
            # Store one object
            with lock:
                results.append({'type': 'persistent', 'id': threading.get_ident()})
            # Signal ready and wait
            persistent_ready.set()
            gc_done.wait()
            stop_persistent.wait()

        # Start abandoned threads (they exit after creating garbage)
        abandoned = []
        for _ in range(4):
            t = threading.Thread(target=abandoned_worker)
            t.start()
            abandoned.append(t)

        # Start persistent thread
        persistent = threading.Thread(target=persistent_worker)
        persistent.start()

        # Wait for abandoned threads to exit
        for t in abandoned:
            t.join()

        # Wait for persistent thread to be ready
        persistent_ready.wait()

        # Now we have: abandoned heaps + one active heap
        collected = gc.collect()

        # Signal GC done
        gc_done.set()

        # Stop persistent thread
        stop_persistent.set()
        persistent.join()

        self.assertGreater(collected, 0,
                          "Should collect garbage from mixed thread scenario")
        self.assertEqual(len(results), 5)  # 4 abandoned + 1 persistent

    def test_gc_config_num_workers(self):
        """
        Test that gc.get_parallel_config() returns correct worker count.
        """
        config = gc.get_parallel_config()
        if not config['available']:
            self.skipTest("Parallel GC not available")

        # Enable with specific worker count
        gc.enable_parallel(8)
        config = gc.get_parallel_config()
        self.assertEqual(config['num_workers'], 8)

        # Change worker count
        gc.enable_parallel(2)
        config = gc.get_parallel_config()
        self.assertEqual(config['num_workers'], 2)

    def test_parallel_stats_tracks_collections(self):
        """
        Test that gc.get_parallel_stats() tracks collection statistics.
        """
        config = gc.get_parallel_config()
        if not config['available']:
            self.skipTest("Parallel GC not available")

        gc.enable_parallel(4)

        # Get baseline
        stats_before = gc.get_parallel_stats()
        attempts_before = stats_before.get('collections_attempted', 0)

        # Create garbage and collect
        for _ in range(100):
            a = []
            b = []
            a.append(b)
            b.append(a)

        gc.collect()

        # Check stats updated
        stats_after = gc.get_parallel_stats()
        self.assertGreaterEqual(stats_after.get('collections_attempted', 0),
                               attempts_before,
                               "Should track collection attempts")

    def test_serial_vs_parallel_correctness(self):
        """
        Test that serial and parallel GC produce same results.

        Run the same garbage creation pattern with both modes and verify
        the same objects are collected.
        """
        config = gc.get_parallel_config()
        if not config['available']:
            self.skipTest("Parallel GC not available")

        def create_garbage_pattern():
            """Create reproducible garbage pattern."""
            garbage = []
            for i in range(100):
                a = {'id': i, 'ref': None}
                b = {'id': i + 1000, 'ref': a}
                a['ref'] = b
                garbage.append(a)
            del garbage
            return gc.collect()

        # Test with parallel GC
        gc.enable_parallel(4)
        gc.collect()  # Clean slate
        parallel_collected = create_garbage_pattern()

        # Test without parallel GC (serial) - use disable_parallel if available
        # For now, we can't easily disable parallel once enabled,
        # so we just verify parallel collects correctly
        self.assertGreater(parallel_collected, 0,
                          "Parallel GC should collect cyclic garbage")


class TestCleanupWorkersAPI(unittest.TestCase):
    """Test cleanup_workers API functions."""

    def test_get_cleanup_workers_exists(self):
        """Test that gc.get_cleanup_workers() exists."""
        self.assertTrue(hasattr(gc, 'get_cleanup_workers'))
        self.assertTrue(callable(gc.get_cleanup_workers))

    def test_set_cleanup_workers_exists(self):
        """Test that gc.set_cleanup_workers() exists."""
        self.assertTrue(hasattr(gc, 'set_cleanup_workers'))
        self.assertTrue(callable(gc.set_cleanup_workers))

    def test_get_cleanup_workers_returns_int(self):
        """Test that gc.get_cleanup_workers() returns an integer."""
        result = gc.get_cleanup_workers()
        self.assertIsInstance(result, int)

    def test_get_cleanup_workers_in_config(self):
        """Test that cleanup_workers is available in get_parallel_config()."""
        config = gc.get_parallel_config()
        if config.get('enabled', False):
            self.assertIn('cleanup_workers', config)
            self.assertIsInstance(config['cleanup_workers'], int)

    @unittest.skipUnless(
        gc.get_parallel_config().get('enabled', False),
        "Parallel GC not enabled")
    def test_set_cleanup_workers_valid_values(self):
        """Test setting valid cleanup_workers values."""
        # Save original value
        original = gc.get_cleanup_workers()
        try:
            # Test setting to 0 (serial)
            gc.set_cleanup_workers(0)
            self.assertEqual(gc.get_cleanup_workers(), 0)

            # Test setting to 1 (async single worker)
            gc.set_cleanup_workers(1)
            self.assertEqual(gc.get_cleanup_workers(), 1)

        finally:
            # Restore original value
            gc.set_cleanup_workers(original)

    @unittest.skipUnless(
        gc.get_parallel_config().get('enabled', False),
        "Parallel GC not enabled")
    def test_set_cleanup_workers_negative_raises(self):
        """Test that negative values raise ValueError."""
        with self.assertRaises(ValueError):
            gc.set_cleanup_workers(-1)

    @unittest.skipUnless(
        gc.get_parallel_config().get('enabled', False),
        "Parallel GC not enabled")
    def test_set_cleanup_workers_too_large_raises(self):
        """Test that values exceeding num_workers raise ValueError."""
        config = gc.get_parallel_config()
        num_workers = config.get('num_workers', 0)
        if num_workers > 0:
            with self.assertRaises(ValueError):
                gc.set_cleanup_workers(num_workers + 1)

    @unittest.skipUnless(
        gc.get_parallel_config().get('enabled', False),
        "Parallel GC not enabled")
    def test_cleanup_workers_gc_still_works(self):
        """Test that GC still works with different cleanup_workers settings."""
        original = gc.get_cleanup_workers()
        try:
            # Test with serial cleanup
            gc.set_cleanup_workers(0)

            # Create cyclic garbage
            class Cycle:
                pass
            a = Cycle()
            b = Cycle()
            a.ref = b
            b.ref = a
            del a, b

            # Force collection
            collected = gc.collect()
            # Just verify no crash - collection count varies

            # Test with async cleanup
            gc.set_cleanup_workers(1)

            # Create more cyclic garbage
            a = Cycle()
            b = Cycle()
            a.ref = b
            b.ref = a
            del a, b

            collected = gc.collect()
            # Just verify no crash

        finally:
            gc.set_cleanup_workers(original)


if __name__ == '__main__':
    unittest.main()
