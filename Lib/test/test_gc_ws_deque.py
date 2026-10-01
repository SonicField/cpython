"""
Tests for work-stealing deque data structure.

This test module wraps C-level tests in _testinternalcapi for the work-stealing
deque implementation (_PyWSDeque), which is used by the parallel GC.
"""

import subprocess
import sys
import unittest
from test import support
from test.support import threading_helper

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

    @threading_helper.requires_working_threading()
    def test_concurrent_push_steal(self):
        """Test concurrent push (owner) and steal (workers)."""
        _testinternalcapi.ws_deque_concurrent_push_steal()


@unittest.skipUnless(hasattr(_testinternalcapi, 'barrier_basic'),
                     "barriers are not supported on this platform")
class TestBarrier(unittest.TestCase):
    """Test GC barrier synchronization (T3-F1, T3-F9)."""

    @threading_helper.requires_working_threading()
    def test_basic(self):
        """All N threads reach barrier, barrier lifts."""
        _testinternalcapi.barrier_basic()

    def test_multiple_rounds(self):
        """Epoch increments once per barrier cycle."""
        _testinternalcapi.test_barrier_multiple_rounds()

    @threading_helper.requires_working_threading()
    def test_epoch_distinguishes(self):
        """Epoch distinguishes barrier rounds (multi-threaded)."""
        _testinternalcapi.barrier_epoch_distinguishes()

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
        code = (
            "from test import support; "
            "support.SuppressCrashReport().__enter__(); "
            "import _testinternalcapi; "
            "_testinternalcapi.unsafe_barrier_capacity_zero()"
        )
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
        if not gc.get_parallel_config()['available']:
            self.skipTest("Parallel GC not available")

        gc.enable_parallel()
        try:
            config = gc.get_parallel_config()
            self.assertTrue(config['enabled'], "Should be enabled after enable_parallel")
            self.assertEqual(config['num_workers'], 16)

            # Run a collection to exercise the parallel path
            gc.collect()
        finally:
            gc.disable_parallel()

        config = gc.get_parallel_config()
        self.assertFalse(config['enabled'], "Should be disabled after disable_parallel")


@unittest.skipUnless(support.Py_PARALLEL_GC, "requires parallel GC build")
class TestAdaptiveWorkerController(unittest.TestCase):

    def test_accepts_improvement_and_rejects_regression(self):
        _testinternalcapi.test_gc_random_walk_accept_reject()


class TestParallelExecution(unittest.TestCase):

    @unittest.skipUnless(support.Py_PARALLEL_GC,
                         "requires parallel GC build")
    def test_helper_threads_traverse_objects(self):
        import gc

        gc.enable_parallel()
        try:
            caller_visits, helper_visits = (
                _testinternalcapi.parallel_gc_helper_visits()
            )
        finally:
            gc.disable_parallel()

        self.assertGreater(caller_visits + helper_visits, 0)
        self.assertGreater(helper_visits, 0)

    @unittest.skipUnless(
        hasattr(_testinternalcapi, 'parallel_gc_stackref_visits'),
        "requires a GIL parallel GC build",
    )
    def test_embedded_stackref_visits(self):
        _testinternalcapi.parallel_gc_stackref_visits()


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

    @threading_helper.requires_working_threading()
    def test_concurrent(self):
        """Producer posts, consumer waits — concurrent correctness."""
        if not hasattr(_testinternalcapi, 'semaphore_concurrent'):
            self.skipTest("Py_PARALLEL_GC not enabled")
        _testinternalcapi.semaphore_concurrent()


if __name__ == '__main__':
    unittest.main()
