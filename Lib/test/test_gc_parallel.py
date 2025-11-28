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
        try:
            result = gc.enable_parallel(4)
            self.assertIsNone(result)
        except RuntimeError as e:
            # Expected if parallel GC not available
            self.assertIn("not available", str(e).lower())

    def test_enable_parallel_zero_workers(self):
        """Test calling enable_parallel(0) to disable."""
        try:
            result = gc.enable_parallel(0)
            self.assertIsNone(result)
        except RuntimeError as e:
            # Expected if parallel GC not available
            self.assertIn("not available", str(e).lower())

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


if __name__ == '__main__':
    unittest.main()
