"""Tests for the work-stealing deque and related GC primitives."""

import subprocess
import sys
import unittest
import _testcapi
from test import support
from test.support import threading_helper

try:
    import _testinternalcapi
except ImportError:
    raise unittest.SkipTest("_testinternalcapi module not available")


class TestWorkStealingDeque(unittest.TestCase):
    def test_init_fini(self):
        _testinternalcapi.test_ws_deque_init_fini()

    def test_push_take_single(self):
        _testinternalcapi.test_ws_deque_push_take_single()

    def test_push_steal_single(self):
        _testinternalcapi.test_ws_deque_push_steal_single()

    def test_lifo_order(self):
        _testinternalcapi.test_ws_deque_lifo_order()

    def test_fifo_order(self):
        _testinternalcapi.test_ws_deque_fifo_order()


class TestWorkStealingDequeEdgeCases(unittest.TestCase):
    def test_take_empty(self):
        _testinternalcapi.test_ws_deque_take_empty()

    def test_steal_empty(self):
        _testinternalcapi.test_ws_deque_steal_empty()

    def test_resize(self):
        _testinternalcapi.test_ws_deque_resize()

    @support.nomemtest
    def test_grow_oom(self):
        _testcapi.call_with_nomemory(
            1, 2, _testinternalcapi.ws_deque_grow_oom)


class TestWorkStealingDequeConcurrent(unittest.TestCase):
    @threading_helper.requires_working_threading()
    def test_concurrent_push_steal(self):
        _testinternalcapi.test_ws_deque_concurrent_push_steal()


@unittest.skipUnless(hasattr(_testinternalcapi, 'test_barrier_basic'),
                     "barriers are not supported on this platform")
class TestBarrier(unittest.TestCase):
    @threading_helper.requires_working_threading()
    def test_basic(self):
        _testinternalcapi.test_barrier_basic()

    def test_multiple_rounds(self):
        _testinternalcapi.test_barrier_multiple_rounds()

    @threading_helper.requires_working_threading()
    def test_epoch_distinguishes(self):
        _testinternalcapi.test_barrier_epoch_distinguishes()

    def test_postcondition(self):
        _testinternalcapi.test_barrier_postcondition()

    @support.requires_subprocess()
    @unittest.skipUnless(hasattr(sys, 'gettotalrefcount') or support.Py_DEBUG,
                         "assert() only fires in debug builds")
    def test_capacity_zero(self):
        code = (
            "import _testinternalcapi; "
            "_testinternalcapi.unsafe_barrier_capacity_zero()"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, timeout=5
        )
        self.assertNotEqual(result.returncode, 0,
                           "Init(capacity=0) should abort via assertion")


class TestLocalBuffer(unittest.TestCase):
    def test_push_pop(self):
        _testinternalcapi.test_localbuffer_push_pop()

    def test_push_full(self):
        _testinternalcapi.test_localbuffer_push_full()

    def test_overflow_flush_precondition(self):
        _testinternalcapi.test_overflow_flush_precondition()

    def test_overflow_flush_normal(self):
        _testinternalcapi.test_overflow_flush_normal()


class TestDequeInvariants(unittest.TestCase):
    def test_init_values(self):
        _testinternalcapi.test_deque_init_values()

    def test_top_leq_bot(self):
        _testinternalcapi.test_deque_top_leq_bot()

    def test_grow_chain_fini(self):
        _testinternalcapi.test_deque_grow_chain_fini()


@unittest.skipUnless(hasattr(_testinternalcapi, 'test_splitvector_init_push'),
                     "parallel GC is not enabled")
class TestParallelGCInfrastructure(unittest.TestCase):
    def test_splitvector_init_push(self):
        _testinternalcapi.test_splitvector_init_push()

    def test_embedded_stackref_visits(self):
        _testinternalcapi.parallel_gc_stackref_visits()


if __name__ == '__main__':
    unittest.main()
