#!/usr/bin/env python3
"""Exercise parallel GC while collecting PGO profile data."""

import argparse
import gc
import json
import sys

DEFAULT_NODE_COUNT = 9_000
DEFAULT_COLLECTIONS = 1
CONTAINER_SAMPLE_COUNT = 8


class ProfileNode:
    __slots__ = ("refs",)

    def __init__(self):
        self.refs = []


class SplitNode:
    pass


def build_live_graph(node_count):
    nodes = [ProfileNode() for _ in range(node_count)]
    for index, node in enumerate(nodes):
        node.refs.extend((nodes[(index + 1) % node_count], node))

    layout_nodes = nodes[:min(node_count, CONTAINER_SAMPLE_COUNT)]
    tuples = [(node, node.refs) for node in layout_nodes]
    unicode_dicts = [
        {"node": node, "tuple": item}
        for node, item in zip(layout_nodes, tuples)
    ]
    general_dicts = [
        {node: item}
        for node, item in zip(layout_nodes, tuples)
    ]

    template = SplitNode()
    template.node = None
    template.payload = None
    split_nodes = []
    split_dicts = []
    for node, item in zip(layout_nodes, tuples):
        split_node = SplitNode()
        split_node.node = node
        split_node.payload = item
        split_nodes.append(split_node)
        split_dicts.append(split_node.__dict__)

    return [
        nodes,
        tuples,
        unicode_dicts,
        general_dicts,
        split_nodes,
        split_dicts,
    ]


def build_unreachable_graph(node_count):
    nodes = [ProfileNode() for _ in range(node_count)]
    for index, node in enumerate(nodes):
        node.refs.append(nodes[(index + 1) % node_count])


def parallel_progress(stats, *, gil_enabled):
    if gil_enabled:
        return stats.get("collections_succeeded", 0)
    timing = stats.get("phase_timing", {})
    return timing.get("total_ns", 0)


def run_profile_workload(
    *,
    gc_module=gc,
    node_count=DEFAULT_NODE_COUNT,
    collections=DEFAULT_COLLECTIONS,
):
    if node_count <= 0:
        raise ValueError("node_count must be positive")
    if collections <= 0:
        raise ValueError("collections must be positive")

    get_config = getattr(gc_module, "get_parallel_config", None)
    config = get_config() if get_config is not None else {}
    if not config.get("available", False):
        return {
            "ran": False,
            "enabled": False,
            "gil_enabled": sys._is_gil_enabled(),
            "requested_nodes": node_count,
            "collections": 0,
            "parallel_path_used": False,
        }

    gc_module.disable()
    try:
        gc_module.enable_parallel()
        config = gc_module.get_parallel_config()
        if not config.get("enabled", False):
            raise RuntimeError("parallel GC did not become enabled")

        gil_enabled = sys._is_gil_enabled()
        stats_before = gc_module.get_parallel_stats()
        progress_before = parallel_progress(
            stats_before,
            gil_enabled=gil_enabled,
        )
        roots = build_live_graph(node_count)
        gc_module.collect()
        garbage_nodes = max(2, node_count // 4)
        for _ in range(collections):
            build_unreachable_graph(garbage_nodes)
            gc_module.collect()

        # Keep the live graph reachable through every measured collection.
        if len(roots) != 6:
            raise RuntimeError("parallel GC profile graph changed shape")

        stats_after = gc_module.get_parallel_stats()
        progress_after = parallel_progress(
            stats_after,
            gil_enabled=gil_enabled,
        )
        if gil_enabled:
            parallel_path_used = progress_after > progress_before
        else:
            parallel_path_used = progress_after > 0
        if not parallel_path_used:
            raise RuntimeError("parallel GC profile workload stayed serial")
    finally:
        gc_module.enable()

    return {
        "ran": True,
        "enabled": True,
        "gil_enabled": sys._is_gil_enabled(),
        "requested_nodes": node_count,
        "collections": collections,
        "parallel_path_used": parallel_path_used,
        "parallel_config": config,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Parallel-GC PGO training workload"
    )
    parser.add_argument("--nodes", type=int, default=DEFAULT_NODE_COUNT)
    parser.add_argument(
        "--collections", type=int, default=DEFAULT_COLLECTIONS
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = run_profile_workload(
        node_count=args.nodes,
        collections=args.collections,
    )
    if args.json:
        print(json.dumps(result, sort_keys=True))
    elif result["ran"]:
        build = "GIL" if result["gil_enabled"] else "free-threaded"
        print(
            f"Trained {build} parallel GC with "
            f"{result['requested_nodes']} nodes and "
            f"{result['collections']} collections"
        )
    else:
        print("Parallel GC is unavailable; profile workload skipped")


if __name__ == "__main__":
    main()
