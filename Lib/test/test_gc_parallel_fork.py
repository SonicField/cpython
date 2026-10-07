"""Fork lifecycle tests for the parallel garbage collectors."""

import gc
import os
import sys
import traceback
import unittest
import _interpreters
import _testinternalcapi

from test import support
from test.support import warnings_helper


GIL_BUILD = sys._is_gil_enabled()


def tearDownModule():
    support.reap_children()


@unittest.skipUnless(support.has_fork_support, "requires working os.fork()")
class ParallelGCForkTests(unittest.TestCase):
    CHILD_TIMEOUT = 10.0
    INITIAL_WORKERS = 4
    LARGE_HEAP = 20_000

    def setUp(self):
        config = gc.get_parallel_config()
        if not config.get("available", False):
            self.skipTest("parallel GC is not available")

        self.gc_was_enabled = gc.isenabled()
        self.addCleanup(self._restore_gc)
        gc.disable()
        gc.disable_parallel()
        gc.collect()
        gc.enable_parallel()
        self._collect_large_cycle()

        stats = gc.get_parallel_stats()
        self.assertGreater(stats["prev_cost_per_obj_ns"], 0.0)

    def _restore_gc(self):
        if hasattr(_testinternalcapi, "parallel_gc_set_startup_failure"):
            _testinternalcapi.parallel_gc_set_startup_failure("none")
        try:
            gc.disable_parallel()
        except RuntimeError:
            pass
        if self.gc_was_enabled:
            gc.enable()

    def _make_large_cycle(self):
        class Node:
            pass

        nodes = [Node() for _ in range(self.LARGE_HEAP)]
        for node in nodes:
            node.ref = node

    def _collect_large_cycle(self):
        self._make_large_cycle()
        gc.collect(0)

    def _assert_fresh_child_controller(self):
        config = gc.get_parallel_config()
        self.assertTrue(config["enabled"])
        if GIL_BUILD:
            self.assertFalse(config["pool_active"])
            self.assertFalse(config["startup_failed"])
        self.assertEqual(config["num_workers"], 16)
        self.assertEqual(config["adaptive_workers"], self.INITIAL_WORKERS)

        stats = gc.get_parallel_stats()
        self.assertEqual(stats["prev_cost_per_obj_ns"], 0.0)

    def _assert_child_can_collect(self):
        self._collect_large_cycle()
        if GIL_BUILD:
            self.assertTrue(gc.get_parallel_config()["pool_active"])
        stats = gc.get_parallel_stats()
        self.assertGreater(stats["prev_cost_per_obj_ns"], 0.0)

    def _check_child_after_fork(self):
        self._assert_fresh_child_controller()
        self._assert_child_can_collect()

    def _check_failed_child_after_fork(self):
        config = gc.get_parallel_config()
        self.assertFalse(config["enabled"])
        self.assertFalse(config["pool_active"])
        self.assertTrue(config["startup_failed"])

        self._collect_large_cycle()
        self.assertFalse(gc.get_parallel_config()["pool_active"])

        gc.enable_parallel()
        self.assertFalse(gc.get_parallel_config()["pool_active"])
        self._assert_child_can_collect()

    def _run_child(self, action):
        try:
            action()
        except BaseException:
            traceback.print_exc()
            os._exit(1)
        os._exit(0)

    def _wait_for_child(self, pid):
        support.wait_process(pid, exitcode=0, timeout=self.CHILD_TIMEOUT)

    def _fork_from_start_callback(self):
        fork_result = []

        def callback(phase, info):
            if phase == "start":
                fork_result.append(os.fork())

        gc.callbacks.append(callback)
        try:
            gc.collect(0)
        finally:
            gc.callbacks.remove(callback)

        self.assertEqual(len(fork_result), 1)
        pid = fork_result[0]
        if pid == 0:
            self._run_child(self._check_child_after_fork)
        self._wait_for_child(pid)

    @warnings_helper.ignore_fork_in_thread_deprecation_warnings()
    def test_child_rebuilds_pool_parent_preserves_state(self):
        parent_config = gc.get_parallel_config()
        parent_stats = gc.get_parallel_stats()

        for _ in range(2):
            pid = os.fork()
            if pid == 0:
                self._run_child(self._check_child_after_fork)

            self._wait_for_child(pid)
            config = gc.get_parallel_config()
            stats = gc.get_parallel_stats()
            self.assertEqual(
                config["adaptive_workers"],
                parent_config["adaptive_workers"],
            )
            self.assertEqual(
                stats["prev_cost_per_obj_ns"],
                parent_stats["prev_cost_per_obj_ns"],
            )

    @warnings_helper.ignore_fork_in_thread_deprecation_warnings()
    @unittest.skipUnless(GIL_BUILD, "GIL lazy pool only")
    def test_armed_child_stays_armed_until_eligible_collection(self):
        gc.disable_parallel()
        gc.enable_parallel()
        self.assertFalse(gc.get_parallel_config()["pool_active"])

        pid = os.fork()
        if pid == 0:
            self._run_child(self._check_child_after_fork)

        self._wait_for_child(pid)
        config = gc.get_parallel_config()
        self.assertTrue(config["enabled"])
        self.assertFalse(config["pool_active"])

    @warnings_helper.ignore_fork_in_thread_deprecation_warnings()
    @unittest.skipUnless(GIL_BUILD, "GIL lazy pool only")
    def test_failed_child_does_not_retry_until_explicit_enable(self):
        gc.disable_parallel()
        gc.enable_parallel()
        _testinternalcapi.parallel_gc_set_startup_failure("thread", 0)
        with self.assertWarns(RuntimeWarning):
            self._collect_large_cycle()
        _testinternalcapi.parallel_gc_set_startup_failure("none")

        pid = os.fork()
        if pid == 0:
            self._run_child(self._check_failed_child_after_fork)

        self._wait_for_child(pid)
        config = gc.get_parallel_config()
        self.assertFalse(config["enabled"])
        self.assertTrue(config["startup_failed"])

    @warnings_helper.ignore_fork_in_thread_deprecation_warnings()
    @unittest.skipUnless(GIL_BUILD, "GIL lazy pool only")
    def test_disabled_child_stays_disabled(self):
        gc.disable_parallel()

        pid = os.fork()
        if pid == 0:
            self._run_child(
                lambda: self.assertEqual(
                    gc.get_parallel_config(),
                    {
                        "available": True,
                        "enabled": False,
                        "num_workers": 0,
                        "pool_active": False,
                        "startup_failed": False,
                    },
                )
            )

        self._wait_for_child(pid)
        self.assertFalse(gc.get_parallel_config()["enabled"])

    @warnings_helper.ignore_fork_in_thread_deprecation_warnings()
    @unittest.skipUnless(GIL_BUILD, "GIL lazy pool only")
    def test_fork_from_start_callback_while_armed(self):
        gc.disable_parallel()
        gc.enable_parallel()

        self._fork_from_start_callback()
        config = gc.get_parallel_config()
        self.assertTrue(config["enabled"])
        self.assertFalse(config["pool_active"])

    @warnings_helper.ignore_fork_in_thread_deprecation_warnings()
    @unittest.skipUnless(GIL_BUILD, "GIL lazy pool only")
    def test_fork_from_start_callback_while_active(self):
        parent_config = gc.get_parallel_config()
        parent_stats = gc.get_parallel_stats()

        self._fork_from_start_callback()
        config = gc.get_parallel_config()
        stats = gc.get_parallel_stats()
        self.assertTrue(config["pool_active"])
        self.assertEqual(
            config["adaptive_workers"], parent_config["adaptive_workers"]
        )
        self.assertEqual(
            stats["prev_cost_per_obj_ns"],
            parent_stats["prev_cost_per_obj_ns"],
        )

    def _fork_from_finalizer(self, generation=2):
        fork_result = []

        class ForkFromFinalizer:
            def __del__(self):
                fork_result.append(os.fork())

        victim = ForkFromFinalizer()
        victim.cycle = victim
        del victim

        gc.collect(generation)
        self.assertEqual(len(fork_result), 1)
        pid = fork_result[0]

        if pid == 0:
            self._run_child(self._check_child_after_fork)

        self._wait_for_child(pid)

    @warnings_helper.ignore_fork_in_thread_deprecation_warnings()
    def test_fork_from_finalizer_excludes_inherited_collection(self):
        self._fork_from_finalizer()

    @warnings_helper.ignore_fork_in_thread_deprecation_warnings()
    @unittest.skipUnless(GIL_BUILD, "GIL lazy pool only")
    def test_fork_from_finalizer_while_armed(self):
        gc.disable_parallel()
        gc.enable_parallel()
        gc.collect(0)

        self._fork_from_finalizer(0)
        config = gc.get_parallel_config()
        self.assertTrue(config["enabled"])
        self.assertFalse(config["pool_active"])

    @unittest.skip(
        "upstream CPython crashes when a main-interpreter fork cleans up "
        "a legacy subinterpreter, even without parallel GC"
    )
    @warnings_helper.ignore_fork_in_thread_deprecation_warnings()
    def test_child_discards_subinterpreter_pool_before_cleanup(self):
        # CPython deletes non-main interpreters in PyOS_AfterFork_Child(). A
        # parallel-GC pool belonging to one of those interpreters must be
        # abandoned rather than joined because its helpers exist only in the
        # parent. This remains skipped until the upstream feature-off control
        # can survive the same lifecycle in both GIL and free-threaded builds.
        gc.disable_parallel()
        interp = _interpreters.create("legacy")
        try:
            _interpreters.exec(interp, "import gc; gc.disable(); gc.enable_parallel()")

            pid = os.fork()
            if pid == 0:
                os._exit(0)
            self._wait_for_child(pid)
        finally:
            _interpreters.exec(interp, "import gc; gc.disable_parallel()")
            _interpreters.destroy(interp)

if __name__ == "__main__":
    unittest.main()
