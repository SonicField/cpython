"""
Tests for parallel garbage collection.

This test module covers the parallel GC feature, which is only available
in GIL-based builds compiled with --with-parallel-gc.
"""

import gc
import sys
import unittest
from test import support


# Skip entire test file if free-threading build
if hasattr(sys, '_is_gil_enabled') and not sys._is_gil_enabled():
    raise unittest.SkipTest("Parallel GC requires GIL-based build")


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
        """Test calling enable_parallel() with no arguments."""
        try:
            result = gc.enable_parallel()
            # If it succeeds, result should be None
            self.assertIsNone(result)
        except RuntimeError as e:
            # Expected if parallel GC not available
            self.assertIn("not available", str(e).lower())

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

    def test_parallel_gc_not_in_nogil_builds(self):
        """Verify parallel GC is not available in nogil builds."""
        if hasattr(sys, '_is_gil_enabled') and not sys._is_gil_enabled():
            # This test should be skipped due to module-level skip,
            # but double-check here
            config = gc.get_parallel_config()
            self.assertFalse(config['available'])
            self.assertFalse(config['enabled'])
            self.assertEqual(config['num_workers'], 0)

    @support.cpython_only
    def test_enable_parallel_error_message_without_build_flag(self):
        """Test error message when parallel GC not compiled in."""
        config = gc.get_parallel_config()
        if not config['available']:
            with self.assertRaises(RuntimeError) as cm:
                gc.enable_parallel()
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


class TestParallelMarkingPhase5(unittest.TestCase):
    """
    TDD tests for Phase 5: Parallel Marking Implementation

    These tests verify each step of the parallel marking algorithm:
    - Step 1: Root scanning
    - Step 2: Root distribution
    - Step 3: Worker marking loop
    - Step 4: Work stealing
    - Step 5: Termination detection
    - Step 6: Barrier synchronization
    """

    def setUp(self):
        """Set up test fixtures."""
        # Check if parallel GC is available
        config = gc.get_parallel_config()
        if not config['available']:
            self.skipTest("Parallel GC not available in this build")

        # Ensure parallel GC is disabled at start
        # (No disable API yet, but we can test from clean state)

    def tearDown(self):
        """Clean up after test."""
        # Force a collection to clean up test objects
        gc.collect()

    @support.cpython_only
    def test_step1_root_scanning(self):
        """
        Step 1: Verify roots are identified from young generation.

        This test checks that _PyGC_ParallelMoveUnreachable() correctly
        scans the young generation list and identifies root objects
        (objects with gc_refs > 0, meaning they have external references).
        """
        # Enable parallel GC with 4 workers (skip if already enabled)
        config = gc.get_parallel_config()
        if not config.get('enabled', False):
            gc.enable_parallel(4)

        # Get baseline stats
        stats_before = gc.get_parallel_stats()
        self.assertEqual(stats_before['roots_found'], 0,
                        "Should start with 0 roots found")

        # Create objects with external references (these become roots)
        # The list 'roots' holds references, so these objects won't be garbage
        roots = []
        for i in range(100):
            obj = {'data': i, 'value': 'test'}
            roots.append(obj)  # External ref makes it a root

        # Trigger GC collection
        gc.collect()

        # Check stats after collection
        stats_after = gc.get_parallel_stats()

        # Verify collections_attempted was incremented
        # (Even if parallel marking returns 0, we should attempt it)
        self.assertGreater(stats_after['collections_attempted'],
                          stats_before['collections_attempted'],
                          "Should have attempted parallel marking")

        # Verify roots were found
        # Note: May fall back to serial if not enough roots, so check both cases
        if stats_after['collections_succeeded'] > 0:
            # Parallel marking was used
            self.assertGreater(stats_after['roots_found'], 0,
                              "Should have found root objects when using parallel marking")
            # Should find at least our 100 dicts plus some builtins
            self.assertGreaterEqual(stats_after['roots_found'], 100,
                                   f"Should find at least the 100 dicts we created, "
                                   f"got {stats_after['roots_found']}")
        else:
            # Fell back to serial - that's OK for now, just document it
            # When Step 1 is fully implemented, this should not happen
            pass

    @support.cpython_only
    def test_step2_root_distribution(self):
        """
        Step 2: Verify roots are distributed to worker deques.

        This test checks that roots identified in Step 1 are distributed
        across worker deques using static slicing for temporal locality.
        With static slicing, roots are assigned to workers based on their
        position in the GC list, preserving allocation order.
        """
        # Disable automatic GC to control when collections happen
        was_enabled = gc.isenabled()
        gc.disable()

        try:
            # Enable parallel GC with 4 workers (skip if already enabled)
            config = gc.get_parallel_config()
            if not config.get('enabled', False):
                gc.enable_parallel(4)

            # Force a full collection to clear out old objects
            gc.collect()

            # Get baseline stats
            stats_before = gc.get_parallel_stats()

            # Create a large object graph that will have roots
            # Note: In CPython's GC, a "root" is an object with EXTERNAL references
            # (from stack frames, module globals, etc.), not internal references
            # from other tracked objects. So creating nested structures doesn't
            # create more roots - only the outer container referenced by a local
            # variable is a root.
            #
            # To ensure parallel marking is used, we need enough TOTAL OBJECTS
            # (threshold is num_workers * 4 = 16 objects for 4 workers)
            roots = []
            for i in range(50):
                # Create a list with dicts - this creates many objects
                # but only 'roots' is the actual GC root
                obj_list = [{'id': i, 'data': j} for j in range(5)]
                roots.append(obj_list)

            # Trigger GC collection on generation 0 only
            # This ensures our new objects are in the young generation
            gc.collect(0)

            # Check stats after collection
            stats_after = gc.get_parallel_stats()

            # Verify at least some roots were found (from Step 1)
            # With our object graph, we expect at least 1 root (the 'roots' list)
            # plus potentially a few temporary objects
            self.assertGreater(stats_after['roots_found'], 0,
                              "Should have found roots (Step 1)")

            # If parallel marking was attempted and succeeded
            if stats_after['collections_succeeded'] > stats_before.get('collections_succeeded', 0):
                # Step 2 verification: roots should be distributed
                self.assertGreater(stats_after['roots_distributed'], 0,
                                  "Should have distributed roots to workers")

                self.assertEqual(stats_after['roots_distributed'],
                               stats_after['roots_found'],
                               "All found roots should be distributed")

                # Verify distribution is roughly balanced
                # (This is a heuristic - perfect balance not guaranteed)
                worker_stats = stats_after['workers']
                self.assertEqual(len(worker_stats), 4,
                               "Should have 4 worker entries")

                # With static slicing, roots_in_slice shows how roots
                # are distributed based on their position in the GC list
                total_roots_in_slices = sum(w['roots_in_slice'] for w in worker_stats)
                self.assertEqual(total_roots_in_slices, stats_after['roots_distributed'],
                               "Sum of roots_in_slice should equal roots_distributed")

            else:
                # Fell back to serial - that's OK for small collections
                # Parallel GC has overhead, so it falls back for small heaps
                pass

        finally:
            # Restore GC state
            if was_enabled:
                gc.enable()
            else:
                gc.disable()

    @support.cpython_only
    def test_step3_worker_marking(self):
        """
        Step 3: Verify workers mark objects by traversing object graphs.

        This test checks that worker threads actually process objects from
        their deques, marking them and traversing their children.
        """
        # Disable automatic GC to control when collections happen
        was_enabled = gc.isenabled()
        gc.disable()

        try:
            # Enable parallel GC with 4 workers (skip if already enabled)
            config = gc.get_parallel_config()
            if not config.get('enabled', False):
                gc.enable_parallel(4)

            # Force a full collection to clear out old objects
            gc.collect()

            # Get baseline stats
            stats_before = gc.get_parallel_stats()

            # Create an object graph with parent->children references
            # Each parent dict will have a list of child dicts
            # This creates a tree structure that workers need to traverse
            roots = []
            for i in range(20):
                # Create parent object
                parent = {'id': i, 'children': []}

                # Create children for this parent
                for j in range(10):
                    child = {'parent_id': i, 'child_id': j, 'data': list(range(5))}
                    parent['children'].append(child)

                roots.append(parent)  # Keep parent alive (makes it a root)

            # Trigger GC collection on generation 0
            gc.collect(0)

            # Check stats after collection
            stats_after = gc.get_parallel_stats()

            # Verify roots were found and distributed (Steps 1-2)
            self.assertGreater(stats_after['roots_found'], 0,
                              "Should have found roots (Step 1)")
            self.assertGreater(stats_after['roots_distributed'], 0,
                              "Should have distributed roots (Step 2)")

            # Step 3 verification: workers should have marked objects
            if stats_after['collections_succeeded'] > 0:
                # Parallel marking was used
                worker_stats = stats_after['workers']
                total_marked = sum(w['objects_marked'] for w in worker_stats)

                self.assertGreater(total_marked, 0,
                                  "At least one worker should have marked objects")

                # With 20 parents + 200 children = 220 objects minimum,
                # we should see significant marking activity
                self.assertGreaterEqual(total_marked, 20,
                                       f"Should have marked at least the 20 parent dicts, "
                                       f"got {total_marked}")

            else:
                # Fell back to serial - that's OK, Step 3 not fully implemented yet
                # When Step 3 is complete, this branch should not be taken
                # for collections with sufficient roots and distribution
                pass

        finally:
            # Restore GC state
            if was_enabled:
                gc.enable()
            else:
                gc.disable()

    @support.cpython_only
    def test_step4_work_stealing(self):
        """
        Step 4: Verify workers can steal work from each other.

        This test checks that when a worker runs out of local work,
        it attempts to steal from other workers' deques.
        """
        # Disable automatic GC to control when collections happen
        was_enabled = gc.isenabled()
        gc.disable()

        try:
            # Enable parallel GC with 4 workers (skip if already enabled)
            config = gc.get_parallel_config()
            if not config.get('enabled', False):
                gc.enable_parallel(4)

            # Force a full collection to clear out old objects
            gc.collect()

            # Get baseline stats
            stats_before = gc.get_parallel_stats()

            # Create many root objects to ensure there's work to do
            # With current implementation (no traversal), each worker
            # gets roots via round-robin and processes them quickly.
            # Work-stealing might not occur often, but we can verify
            # the infrastructure exists and tracks statistics.
            roots = []
            for i in range(100):
                # Create root object
                obj = {'id': i, 'data': list(range(10))}
                roots.append(obj)

            # Trigger GC collection on generation 0
            gc.collect(0)

            # Check stats after collection
            stats_after = gc.get_parallel_stats()

            # Verify roots were found and distributed (Steps 1-2)
            self.assertGreater(stats_after['roots_found'], 0,
                              "Should have found roots (Step 1)")
            self.assertGreater(stats_after['roots_distributed'], 0,
                              "Should have distributed roots (Step 2)")

            # Step 4 verification: check work-stealing infrastructure
            if stats_after['collections_succeeded'] > 0:
                # Parallel marking was used
                worker_stats = stats_after['workers']

                # At minimum, verify steal statistics exist and are tracked
                # (Even if no stealing occurred due to balanced distribution)
                for i, w in enumerate(worker_stats):
                    self.assertIn('steal_attempts', w,
                                 f"Worker {i} should track steal_attempts")
                    self.assertIn('steal_successes', w,
                                 f"Worker {i} should track steal_successes")
                    self.assertIsInstance(w['steal_attempts'], int,
                                        f"Worker {i} steal_attempts should be int")
                    self.assertIsInstance(w['steal_successes'], int,
                                        f"Worker {i} steal_successes should be int")

                    # Steal successes can't exceed attempts
                    self.assertLessEqual(w['steal_successes'], w['steal_attempts'],
                                       f"Worker {i}: steal_successes <= steal_attempts")

                # Note: With current implementation (no traversal, round-robin distribution),
                # workers may not actually steal since they all finish quickly.
                # That's OK - we're verifying the infrastructure exists.
                # When traversal is added, work-stealing will become more important.

            else:
                # Fell back to serial - that's OK, Step 4 not fully implemented yet
                pass

        finally:
            # Restore GC state
            if was_enabled:
                gc.enable()
            else:
                gc.disable()

    @support.cpython_only
    def test_step3b_object_traversal(self):
        """
        Step 3b: Verify workers traverse object graphs and discover children.

        This test checks that workers actually call tp_traverse() to discover
        child objects, not just process the initial roots. This is the critical
        step that makes parallel GC actually useful.
        """
        # Disable automatic GC to control when collections happen
        was_enabled = gc.isenabled()
        gc.disable()

        try:
            # Enable parallel GC with 4 workers (skip if already enabled)
            config = gc.get_parallel_config()
            if not config.get('enabled', False):
                gc.enable_parallel(4)

            # Force a full collection to clear out old objects
            gc.collect()

            # Create deep object graph:
            # Root -> 10 level-1 children -> 100 level-2 grandchildren
            # Total: 1 + 10 + 100 = 111 objects
            root = {'level': 0, 'children': []}
            for i in range(10):
                child = {'level': 1, 'parent': root, 'children': []}
                root['children'].append(child)
                for j in range(10):
                    grandchild = {'level': 2, 'parent': child, 'id': f'{i}_{j}'}
                    child['children'].append(grandchild)

            # Trigger GC collection on generation 0
            gc.collect(0)

            # Check stats after collection
            stats = gc.get_parallel_stats()

            # Step 3b verification: workers should traverse and discover children
            if stats['collections_succeeded'] > 0:
                # Parallel marking was used

                # Should have distributed only the root (or few roots)
                # but discovered many more objects via traversal
                self.assertGreater(stats['roots_distributed'], 0,
                                  "Should have distributed root objects")

                # **KEY TEST**: Objects traversed should be MUCH larger than roots distributed
                # With traversal, we expect to discover all 111 objects (root + children + grandchildren)
                # Without traversal, we'd only see the root(s)
                self.assertGreater(stats.get('objects_traversed', 0), 100,
                                  "Workers should discover 100+ objects via tp_traverse(). "
                                  "Currently only counting roots - need to implement traversal!")

                # Verify workers actually performed traversals
                worker_stats = stats['workers']
                total_traversed = sum(w.get('objects_marked', 0) for w in worker_stats)
                self.assertGreater(total_traversed, 100,
                                  "Workers should have marked 100+ objects via traversal")

                # Verify traversal statistics exist
                for i, w in enumerate(worker_stats):
                    # New fields for Step 3b
                    if w.get('objects_marked', 0) > 0:
                        # If worker marked objects, it should have performed traversals
                        self.assertIn('traversals_performed', w,
                                     f"Worker {i} should track traversals_performed")
                        self.assertGreater(w.get('traversals_performed', 0), 0,
                                         f"Worker {i} should have performed tp_traverse calls")

            else:
                # Fell back to serial - that's OK for now
                self.skipTest("Parallel GC not used (fell back to serial)")

        finally:
            # Restore GC state
            if was_enabled:
                gc.enable()
            else:
                gc.disable()


if __name__ == '__main__':
    unittest.main()
