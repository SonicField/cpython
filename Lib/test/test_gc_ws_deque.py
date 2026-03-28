"""
Tests for work-stealing deque data structure.

This test module wraps C-level tests in _testinternalcapi for the work-stealing
deque implementation (_PyWSDeque), which is used by the parallel GC.
"""

import subprocess
import sys
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


class TestBarrier(unittest.TestCase):
    """Test GC barrier synchronization (T3-F1, T3-F9)."""

    def test_basic(self):
        """All N threads reach barrier, barrier lifts."""
        _testinternalcapi.test_barrier_basic()

    def test_multiple_rounds(self):
        """Epoch increments once per barrier cycle."""
        _testinternalcapi.test_barrier_multiple_rounds()

    def test_epoch_distinguishes(self):
        """Epoch distinguishes barrier rounds (multi-threaded)."""
        _testinternalcapi.test_barrier_epoch_distinguishes()

    def test_postcondition(self):
        """After Wait, epoch advanced and num_left reset (T3-F9)."""
        _testinternalcapi.test_barrier_postcondition()

    @unittest.skipUnless(hasattr(sys, 'gettotalrefcount') or support.Py_DEBUG,
                         "assert() only fires in debug builds")
    def test_capacity_zero(self):
        """Init with capacity=0 triggers assertion (T3-F1).

        Falsifiability: removing assert(capacity > 0) from _PyGCBarrier_Init
        causes this test to hang (Wait decrements num_left=0 to UINT_MAX).
        """
        code = "import _testinternalcapi; _testinternalcapi.test_barrier_capacity_zero()"
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, timeout=5
        )
        # Assertion failure causes SIGABRT (return code -6 on POSIX)
        self.assertNotEqual(result.returncode, 0,
                           "Init(capacity=0) should abort via assertion")


class TestLocalBuffer(unittest.TestCase):
    """Test GC local work buffer (T3-F2, T3-F3, T3-F4)."""

    def test_push_pop(self):
        """Basic push/pop operations."""
        _testinternalcapi.test_localbuffer_push_pop()

    def test_push_full(self):
        """Fill buffer to capacity (T3-F2 bounds)."""
        _testinternalcapi.test_localbuffer_push_full()

    def test_pop_empty(self):
        """Pop until empty (T3-F3 bounds)."""
        _testinternalcapi.test_localbuffer_pop_empty()

    def test_overflow_flush_precondition(self):
        """OverflowFlush with full buffer (T3-F4 precondition)."""
        _testinternalcapi.test_overflow_flush_precondition()

    def test_overflow_flush_normal(self):
        """OverflowFlush normal cycle: fill, flush, continue."""
        _testinternalcapi.test_overflow_flush_normal()


class TestDequeInvariants(unittest.TestCase):
    """Test deque structural invariants (T3-F10, T3-F11)."""

    def test_init_values(self):
        """top and bot initialized to 1, not 0 (T3-F10)."""
        _testinternalcapi.test_deque_init_values()

    def test_top_leq_bot(self):
        """top <= bot after all operations complete (T3-F11)."""
        _testinternalcapi.test_deque_top_leq_bot()

    def test_grow_chain_fini(self):
        """Fini frees entire array chain after resize (D9)."""
        _testinternalcapi.test_deque_grow_chain_fini()


class TestParallelGCLifecycle(unittest.TestCase):
    """Test parallel GC enable/disable consistency (T1-F10, T1-F11)."""

    def test_enable_disable_consistency(self):
        """Enable → config shows enabled. Disable → config shows disabled."""
        import gc
        if not hasattr(gc, 'enable_parallel'):
            self.skipTest("Parallel GC not available")

        gc.enable_parallel(2)
        try:
            config = gc.get_parallel_config()
            self.assertTrue(config['enabled'], "Should be enabled after enable_parallel")
            self.assertEqual(config['num_workers'], 2)

            # Run a collection to exercise the parallel path
            gc.collect()
        finally:
            gc.disable_parallel()

        config = gc.get_parallel_config()
        self.assertFalse(config['enabled'], "Should be disabled after disable_parallel")


class TestSplitVector(unittest.TestCase):
    """Test GC split vector (T1-F1, T1-F2)."""

    def test_init_push(self):
        """Init, push beyond capacity (triggers grow), clear."""
        if not hasattr(_testinternalcapi, 'test_splitvector_init_push'):
            self.skipTest("Py_PARALLEL_GC not enabled")
        _testinternalcapi.test_splitvector_init_push()


class TestWorkQueue(unittest.TestCase):
    """Test GC work queue (T1-F3, T1-F4)."""

    def test_init_push(self):
        """Init, push items, verify write_index and ordering."""
        if not hasattr(_testinternalcapi, 'test_workqueue_init_push'):
            self.skipTest("Py_PARALLEL_GC not enabled")
        _testinternalcapi.test_workqueue_init_push()

    def test_reset(self):
        """Push items, reset, verify indices zeroed."""
        if not hasattr(_testinternalcapi, 'test_workqueue_reset'):
            self.skipTest("Py_PARALLEL_GC not enabled")
        _testinternalcapi.test_workqueue_reset()


class TestSemaphore(unittest.TestCase):
    """Test GC semaphore (T1-F5, T1-F6, T1-F7)."""

    def test_post_wait(self):
        """Post N tokens, wait N times, verify consumed."""
        if not hasattr(_testinternalcapi, 'test_semaphore_post_wait'):
            self.skipTest("Py_PARALLEL_GC not enabled")
        _testinternalcapi.test_semaphore_post_wait()

    def test_post_multiple(self):
        """Post in batches, verify total tokens correct."""
        if not hasattr(_testinternalcapi, 'test_semaphore_post_multiple'):
            self.skipTest("Py_PARALLEL_GC not enabled")
        _testinternalcapi.test_semaphore_post_multiple()

    def test_concurrent(self):
        """Producer posts, consumer waits — concurrent correctness."""
        if not hasattr(_testinternalcapi, 'test_semaphore_concurrent'):
            self.skipTest("Py_PARALLEL_GC not enabled")
        _testinternalcapi.test_semaphore_concurrent()


if __name__ == '__main__':
    unittest.main()
