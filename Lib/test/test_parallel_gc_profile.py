import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

from test import support


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "Tools"
    / "build"
    / "parallel_gc_profile.py"
)


def load_profile_module():
    spec = importlib.util.spec_from_file_location(
        "parallel_gc_profile", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeGC:
    def __init__(self, *, available=True, fail_enable=False, fail_collect=False):
        self.available = available
        self.fail_enable = fail_enable
        self.fail_collect = fail_collect
        self.enabled = False
        self.collections = 0

    def get_parallel_config(self):
        return {
            "available": self.available,
            "enabled": self.enabled,
            "num_workers": 16 if self.enabled else 0,
        }

    def get_parallel_stats(self):
        return {
            "collections_succeeded": self.collections,
            "phase_timing": {
                "total_ns": self.collections,
            },
        }

    def enable_parallel(self):
        if self.fail_enable:
            raise RuntimeError("enable failed")
        self.enabled = True

    def collect(self):
        if self.fail_collect:
            raise RuntimeError("collect failed")
        self.collections += 1
        return 0

    def disable(self):
        pass

    def enable(self):
        pass


class ParallelGCProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = load_profile_module()

    def test_unavailable_collector_skips_without_enabling(self):
        fake_gc = FakeGC(available=False)

        result = self.profile.run_profile_workload(
            gc_module=fake_gc,
            node_count=64,
            collections=2,
        )

        self.assertFalse(result["ran"])
        self.assertFalse(fake_gc.enabled)
        self.assertEqual(fake_gc.collections, 0)

    def test_workload_enables_collector_and_completes_collections(self):
        fake_gc = FakeGC()

        result = self.profile.run_profile_workload(
            gc_module=fake_gc,
            node_count=64,
            collections=2,
        )

        self.assertTrue(result["ran"])
        self.assertTrue(result["enabled"])
        self.assertEqual(result["requested_nodes"], 64)
        self.assertEqual(result["collections"], 2)
        self.assertTrue(result["parallel_path_used"])
        self.assertGreaterEqual(fake_gc.collections, 2)

    def test_live_graph_contains_every_exact_container_layout(self):
        roots = self.profile.build_live_graph(64)
        (
            nodes,
            tuples,
            unicode_dicts,
            general_dicts,
            split_nodes,
            split_dicts,
        ) = roots

        self.assertTrue(all(type(node.refs) is list for node in nodes))
        self.assertTrue(all(type(item) is tuple for item in tuples))
        self.assertTrue(all(type(item) is dict for item in unicode_dicts))
        self.assertTrue(all(type(item) is dict for item in general_dicts))
        self.assertTrue(all(type(item) is dict for item in split_dicts))
        expected_layouts = min(
            64,
            self.profile.CONTAINER_SAMPLE_COUNT,
        )
        self.assertEqual(len(tuples), expected_layouts)
        self.assertEqual(len(unicode_dicts), expected_layouts)
        self.assertEqual(len(general_dicts), expected_layouts)
        self.assertEqual(len(split_dicts), expected_layouts)
        self.assertTrue(all(
            type(next(iter(item))) is self.profile.ProfileNode
            for item in general_dicts
        ))
        self.assertEqual(len(split_nodes), len(split_dicts))

        try:
            import _testinternalcapi
        except ImportError:
            pass
        else:
            self.assertTrue(all(
                _testinternalcapi.has_split_table(item)
                for item in split_dicts
            ))

    def test_enable_failure_propagates(self):
        fake_gc = FakeGC(fail_enable=True)

        with self.assertRaisesRegex(RuntimeError, "enable failed"):
            self.profile.run_profile_workload(
                gc_module=fake_gc,
                node_count=64,
                collections=2,
            )

    def test_collection_failure_propagates(self):
        fake_gc = FakeGC(fail_collect=True)

        with self.assertRaisesRegex(RuntimeError, "collect failed"):
            self.profile.run_profile_workload(
                gc_module=fake_gc,
                node_count=64,
                collections=2,
            )

    def test_script_reports_current_build(self):
        result = subprocess.run(
            [
                sys.executable,
                SCRIPT,
                "--json",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        report = json.loads(result.stdout)

        self.assertEqual(report["gil_enabled"], sys._is_gil_enabled())
        if support.Py_PARALLEL_GC:
            self.assertTrue(report["ran"])
            self.assertTrue(report["enabled"])
            self.assertEqual(
                report["requested_nodes"],
                self.profile.DEFAULT_NODE_COUNT,
            )
            self.assertEqual(
                report["collections"],
                self.profile.DEFAULT_COLLECTIONS,
            )
            self.assertTrue(report["parallel_path_used"])
        else:
            self.assertFalse(report["ran"])
            self.assertFalse(report["enabled"])


if __name__ == "__main__":
    unittest.main()
