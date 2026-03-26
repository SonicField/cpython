"""
Tests for work-stealing deque data structure.

This test module wraps C-level tests in _testinternalcapi for the work-stealing
deque implementation (_PyWSDeque), which is used by the parallel GC.
"""

import unittest
from test import support

# Skip if we can't import _testinternalcapi
try:
    import _testinternalcapi
except ImportError:
    raise unittest.SkipTest("_testinternalcapi module not available")


class TestWorkStealingDeque(unittest.TestCase):
    """Test work-stealing deque basic operations."""

    def test_init_fini(self):
        """Test deque initialization and finalization."""
        _testinternalcapi.test_ws_deque_init_fini()

    def test_push_take_single(self):
        """Test pushing and taking a single element."""
        _testinternalcapi.test_ws_deque_push_take_single()

    def test_push_steal_single(self):
        """Test pushing and stealing a single element."""
        _testinternalcapi.test_ws_deque_push_steal_single()

    def test_lifo_order(self):
        """Test LIFO ordering for owner (push/take)."""
        _testinternalcapi.test_ws_deque_lifo_order()

    def test_fifo_order(self):
        """Test FIFO ordering for workers (push/steal)."""
        _testinternalcapi.test_ws_deque_fifo_order()


class TestWorkStealingDequeEdgeCases(unittest.TestCase):
    """Test work-stealing deque edge cases."""

    def test_take_empty(self):
        """Test taking from empty deque."""
        _testinternalcapi.test_ws_deque_take_empty()

    def test_steal_empty(self):
        """Test stealing from empty deque."""
        _testinternalcapi.test_ws_deque_steal_empty()

    def test_resize(self):
        """Test deque automatic resizing."""
        _testinternalcapi.test_ws_deque_resize()

    def test_init_with_undersized_buffer(self):
        """Test InitWithBuffer falls back to malloc when buffer is too small."""
        _testinternalcapi.test_ws_deque_init_with_undersized_buffer()

    def test_init_with_exact_buffer(self):
        """Test InitWithBuffer succeeds with correctly sized buffer."""
        _testinternalcapi.test_ws_deque_init_with_exact_buffer()


class TestWorkStealingDequeConcurrent(unittest.TestCase):
    """Test work-stealing deque concurrent operations."""

    @support.requires_fork()
    def test_concurrent_push_steal(self):
        """Test concurrent push (owner) and steal (workers)."""
        # This test uses pthreads internally
        _testinternalcapi.test_ws_deque_concurrent_push_steal()


if __name__ == '__main__':
    unittest.main()
