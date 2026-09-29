"""Tests for the free-threaded parallel garbage collector."""

import gc
import sys
import threading
import unittest
import weakref

from test.support import threading_helper


if not hasattr(sys, "_is_gil_enabled") or sys._is_gil_enabled():
    raise unittest.SkipTest("requires a free-threaded build")

if not gc.get_parallel_config()["available"]:
    raise unittest.SkipTest("parallel GC is not available")


class GCTestObject:
    __slots__ = ("data", "ref", "__weakref__")

    def __init__(self, **data):
        self.data = data
        self.ref = None


class ParallelGCTest(unittest.TestCase):
    def setUp(self):
        self.gc_was_enabled = gc.isenabled()
        config = gc.get_parallel_config()
        self.old_workers = config["num_workers"] if config["enabled"] else 0
        gc.disable()
        gc.collect()
        gc.enable_parallel(4)

    def tearDown(self):
        gc.disable_parallel()
        if self.old_workers:
            gc.enable_parallel(self.old_workers)
        if self.gc_was_enabled:
            gc.enable()
        else:
            gc.disable()


@threading_helper.requires_working_threading()
class TestCrossThreadReferences(ParallelGCTest):
    def test_cross_thread_refs_survive(self):
        shared = []
        lock = threading.Lock()

        def allocate(thread_id):
            objects = [
                GCTestObject(thread=thread_id, index=index)
                for index in range(100)
            ]
            with lock:
                shared.extend(objects)

        threads = [
            threading.Thread(target=allocate, args=(thread_id,))
            for thread_id in range(4)
        ]
        with threading_helper.start_threads(threads):
            pass

        refs = [weakref.ref(obj) for obj in shared]
        gc.collect()
        self.assertTrue(all(ref() is not None for ref in refs))

    def test_cross_thread_cycles_collected(self):
        first = []
        second = []
        refs = []
        lock = threading.Lock()
        barrier = threading.Barrier(2)

        def make_half_cycle(own, other):
            obj = GCTestObject()
            own.append(obj)
            with lock:
                refs.append(weakref.ref(obj))
            barrier.wait()
            obj.ref = other[0]
            barrier.wait()

        threads = [
            threading.Thread(target=make_half_cycle, args=(first, second)),
            threading.Thread(target=make_half_cycle, args=(second, first)),
        ]
        with threading_helper.start_threads(threads):
            pass

        first.clear()
        second.clear()
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))


@threading_helper.requires_working_threading()
class TestConcurrentCollection(ParallelGCTest):
    def test_concurrent_collect(self):
        errors = []
        lock = threading.Lock()

        def collect(thread_id):
            try:
                for iteration in range(20):
                    objects = [
                        GCTestObject(
                            thread=thread_id,
                            iteration=iteration,
                            index=index,
                        )
                        for index in range(100)
                    ]
                    for index, obj in enumerate(objects):
                        obj.ref = objects[(index + 1) % len(objects)]
                    gc.collect()
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [
            threading.Thread(target=collect, args=(thread_id,))
            for thread_id in range(4)
        ]
        with threading_helper.start_threads(threads):
            pass
        self.assertEqual(errors, [])

    def test_allocation_during_collection(self):
        start = threading.Barrier(5)
        errors = []
        lock = threading.Lock()

        def allocate():
            try:
                start.wait()
                for _ in range(100):
                    [GCTestObject(index=index) for index in range(500)]
            except Exception as exc:
                with lock:
                    errors.append(exc)

        def collect():
            try:
                start.wait()
                for _ in range(100):
                    gc.collect()
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=allocate) for _ in range(4)]
        threads.append(threading.Thread(target=collect))
        with threading_helper.start_threads(threads):
            pass

        self.assertEqual(errors, [])

    def test_reachable_graph_survives(self):
        objects = [GCTestObject(index=index) for index in range(1000)]
        for index, obj in enumerate(objects):
            obj.ref = objects[(index * 17 + 3) % len(objects)]

        refs = [weakref.ref(obj) for obj in objects]

        threads = [threading.Thread(target=gc.collect) for _ in range(4)]
        with threading_helper.start_threads(threads):
            pass

        self.assertTrue(all(ref() is not None for ref in refs))
        self.assertEqual(
            [obj.data["index"] for obj in objects],
            list(range(len(objects))),
        )


class TestThreadPoolLifecycle(unittest.TestCase):
    def tearDown(self):
        gc.disable_parallel()

    def test_worker_count(self):
        gc.enable_parallel(4)
        config = gc.get_parallel_config()
        self.assertTrue(config["enabled"])
        self.assertEqual(config["num_workers"], 4)

    def test_enable_disable_cycle(self):
        gc.enable_parallel(2)
        gc.collect()
        gc.disable_parallel()
        gc.collect()

    @threading_helper.requires_working_threading()
    def test_concurrent_enable_disable(self):
        errors = []
        barrier = threading.Barrier(4)

        def reconfigure():
            try:
                barrier.wait()
                for _ in range(20):
                    gc.enable_parallel(2)
                    gc.get_parallel_config()
                    gc.collect()
                    gc.disable_parallel()
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=reconfigure) for _ in range(4)]
        with threading_helper.start_threads(threads):
            pass
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
