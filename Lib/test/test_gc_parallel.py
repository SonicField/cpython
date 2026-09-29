"""Tests for the public parallel garbage collector interface."""

import gc
import os
import sysconfig
import unittest
import warnings
import weakref

from test import support
from test.support import script_helper


class Node:
    pass


class ParallelGCAPITests(unittest.TestCase):

    def test_configuration(self):
        config = gc.get_parallel_config()
        self.assertEqual(config["available"], support.Py_PARALLEL_GC)
        self.assertIsInstance(config["enabled"], bool)
        self.assertIsInstance(config["num_workers"], int)

        if not support.Py_PARALLEL_GC:
            self.assertFalse(config["enabled"])
            self.assertEqual(config["num_workers"], 0)

    @unittest.skipIf(support.Py_PARALLEL_GC,
                     "requires a build without parallel GC")
    def test_unavailable(self):
        with self.assertRaisesRegex(RuntimeError, "not available"):
            gc.enable_parallel(2)
        with self.assertRaisesRegex(RuntimeError, "not available"):
            gc.disable_parallel()

    @unittest.skipIf(support.Py_PARALLEL_GC,
                     "requires a build without parallel GC")
    def test_unavailable_startup_options(self):
        _, _, stderr = script_helper.assert_python_failure(
            "-X", "parallel_gc=2", "-c", "pass")
        self.assertIn(b"parallel GC is not supported", stderr)

        _, _, stderr = script_helper.assert_python_failure(
            "-c", "pass", PYTHON_PARALLEL_GC="2", __isolated=False)
        self.assertIn(b"parallel GC is not supported", stderr)

        script_helper.assert_python_ok(
            "-X", "parallel_gc=0", "-c", "pass")
        script_helper.assert_python_ok(
            "-c", "pass", PYTHON_PARALLEL_GC="0", __isolated=False)


@unittest.skipUnless(support.Py_PARALLEL_GC,
                     "requires a parallel-GC build")
class ParallelGCLifecycleTests(unittest.TestCase):

    def setUp(self):
        gc.disable_parallel()

    def tearDown(self):
        gc.disable_parallel()

    def test_worker_count(self):
        for workers in (2, 4):
            with self.subTest(workers=workers):
                gc.enable_parallel(workers)
                config = gc.get_parallel_config()
                self.assertTrue(config["enabled"])
                self.assertEqual(config["num_workers"], workers)

        gc.disable_parallel()
        self.assertEqual(
            gc.get_parallel_config(),
            {"available": True, "enabled": False, "num_workers": 0},
        )

    def test_same_worker_count_is_noop(self):
        gc.enable_parallel(2)
        gc.enable_parallel(2)
        self.assertEqual(gc.get_parallel_config()["num_workers"], 2)

    def test_invalid_worker_counts(self):
        for workers in (-1, 0, 1, 65):
            with self.subTest(workers=workers):
                with self.assertRaises(ValueError):
                    gc.enable_parallel(workers)

    def test_collect_cycle(self):
        first = Node()
        second = Node()
        first.other = second
        second.other = first
        refs = weakref.ref(first), weakref.ref(second)
        del first, second

        gc.enable_parallel(2)
        gc.collect()
        self.assertEqual([ref() for ref in refs], [None, None])

    @unittest.skipUnless(hasattr(os, "fork"), "requires fork")
    def test_fork_restarts_workers(self):
        gc.enable_parallel(2)
        with warnings.catch_warnings(category=DeprecationWarning,
                                     action="ignore"):
            pid = os.fork()
        if pid == 0:
            try:
                if gc.get_parallel_config()["enabled"]:
                    os._exit(1)
                gc.enable_parallel(2)
                gc.collect()
            except BaseException:
                os._exit(1)
            os._exit(0)

        support.wait_process(pid, exitcode=0)
        self.assertEqual(
            gc.get_parallel_config(),
            {"available": True, "enabled": True, "num_workers": 2},
        )
        gc.collect()

    def test_subinterpreter_lifecycle(self):
        from concurrent import interpreters

        interp = interpreters.create()
        self.addCleanup(interp.close)
        interp.exec("""import gc
gc.enable_parallel(2)
cycle = []
cycle.append(cycle)
del cycle
gc.collect()
assert gc.get_parallel_config()["enabled"]
""")


@unittest.skipUnless(support.Py_PARALLEL_GC,
                     "requires a parallel-GC build")
class ParallelGCStartupTests(unittest.TestCase):
    CODE = "import gc; assert gc.get_parallel_config()['num_workers'] == 2"

    def test_xoption(self):
        script_helper.assert_python_ok(
            "-X", "parallel_gc=2", "-c", self.CODE)

    def test_environment(self):
        script_helper.assert_python_ok(
            "-c", self.CODE,
            PYTHON_PARALLEL_GC="2", __isolated=False)

    def test_invalid_environment_worker_counts(self):
        for value in ("-1", "1", "65", "not-an-integer"):
            with self.subTest(value=value):
                script_helper.assert_python_failure(
                    "-c", "pass", PYTHON_PARALLEL_GC=value,
                    __isolated=False)

    def test_xoption_overrides_environment(self):
        script_helper.assert_python_ok(
            "-X", "parallel_gc=2", "-c", self.CODE,
            PYTHON_PARALLEL_GC="4", __isolated=False)

    def test_invalid_worker_counts(self):
        for value in ("-1", "1", "65", "not-an-integer"):
            with self.subTest(value=value):
                script_helper.assert_python_failure(
                    "-X", f"parallel_gc={value}", "-c", "pass")

    def test_sysconfig(self):
        self.assertEqual(sysconfig.get_config_var("Py_PARALLEL_GC"), 1)


if __name__ == "__main__":
    unittest.main()
