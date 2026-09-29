"""Graph invariants shared by both parallel collector implementations."""

import gc
import threading
import unittest
import weakref
import _testinternalcapi

from test.support import threading_helper


if not gc.get_parallel_config()["available"]:
    raise unittest.SkipTest("parallel GC is not available")


class Node:
    __slots__ = ("children", "value", "__weakref__")

    def __init__(self, value):
        self.children = []
        self.value = value


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

    @staticmethod
    def make_ring(size):
        nodes = [Node(i) for i in range(size)]
        for index, node in enumerate(nodes):
            node.children.append(nodes[(index + 1) % size])
        return nodes

    def assert_collected(self, objects):
        refs = [weakref.ref(obj) for obj in objects]
        del objects
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))

    def test_cycles_are_collected(self):
        for size in (2, 3, 16, 127):
            with self.subTest(size=size):
                self.assert_collected(self.make_ring(size))

    def test_reachable_graph_survives(self):
        nodes = [Node(i) for i in range(128)]
        for index in range(1, len(nodes)):
            nodes[index - 1].children.append(nodes[index])
        for index in range(0, len(nodes), 7):
            nodes[index].children.append(nodes[index // 2])

        refs = [weakref.ref(node) for node in nodes]
        gc.collect()

        self.assertTrue(all(ref() is not None for ref in refs))
        self.assertEqual([node.value for node in nodes], list(range(128)))

    def test_reachable_graph_crosses_split_boundaries(self):
        nodes = [Node(i) for i in range(20_000)]
        for index in range(1, len(nodes)):
            nodes[index - 1].children.append(nodes[index])
        nodes[-1].children.append(nodes[0])

        root = nodes[0]
        refs = [weakref.ref(node) for node in nodes]
        del nodes
        gc.collect()

        self.assertTrue(all(ref() is not None for ref in refs))
        self.assertEqual(root.value, 0)
        del root
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))

    def test_mixed_reachable_and_unreachable_graphs(self):
        live = self.make_ring(64)
        garbage = self.make_ring(64)
        garbage_refs = [weakref.ref(node) for node in garbage]
        del garbage

        gc.collect()

        self.assertTrue(all(ref() is None for ref in garbage_refs))
        self.assertEqual([node.value for node in live], list(range(64)))

    def test_cross_split_cycle_is_collected(self):
        self.assert_collected(self.make_ring(20_000))

    def test_helper_thread_traverses_objects(self):
        caller_visits, helper_visits = (
            _testinternalcapi.parallel_gc_helper_visits()
        )
        self.assertGreater(caller_visits + helper_visits, 0)
        self.assertGreater(helper_visits, 0)

    def test_builtin_container_cycles(self):
        refs = []
        for _ in range(100):
            node = Node(0)
            cycle = [node]
            cycle.append({"cycle": cycle})
            node.children.append(cycle)
            refs.append(weakref.ref(node))
        del node, cycle
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))

    def test_repeated_collections_and_reconfiguration(self):
        for workers in (2, 4, 2):
            gc.enable_parallel(workers)
            for _ in range(5):
                self.assert_collected(self.make_ring(1024))

    def test_supported_worker_counts(self):
        for workers in (2, 4, 8):
            with self.subTest(workers=workers):
                gc.enable_parallel(workers)
                config = gc.get_parallel_config()
                self.assertTrue(config["enabled"])
                self.assertEqual(config["num_workers"], workers)
                self.assert_collected(self.make_ring(257))

    @threading_helper.requires_working_threading()
    def test_cycles_from_exited_threads(self):
        refs = []
        lock = threading.Lock()

        def allocate():
            nodes = self.make_ring(64)
            with lock:
                refs.extend(weakref.ref(node) for node in nodes)

        threads = [threading.Thread(target=allocate) for _ in range(8)]
        with threading_helper.start_threads(threads):
            pass

        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))


if __name__ == "__main__":
    unittest.main()
