#!/usr/bin/env python3
"""WS3 Phase 2 method-B probe: minimal in-process gen_simple timing.

v2 — match Tools/benchmark_phoenix.py:1719 _worker_jit warmup pattern exactly:
  3 warmup calls of bench_gen_simple(N_ITER), then 5 measure calls.
  cinderjit.auto() (compile_after_n_calls=1000) — only inner gen() exceeds
  threshold (87k calls per outer), outer bench_gen_simple stays interpreter
  (3 warmup + 5 measure = 8 outer calls < 1000).

v1 used 1100 outer warmup calls which crossed threshold and JIT-compiled
bench_gen_simple itself, producing 16% speedup vs harness ABBA — calibration
divergence per testkeeper 08:05:15Z.

Usage:
  PYTHONJITAUTO=1 ./python_bench scripts/probe_gen_simple_minimal.py jit
  /data/users/alexturner/phoenix/cpython-vanilla/python scripts/probe_gen_simple_minimal.py vanilla

Output line: PROBE_RESULT mode=<jit|vanilla> ms=<float>
Caller computes ratio = vanilla_ms / jit_ms.
"""

import sys
import time

def bench_gen_simple(n_iter):
    def gen(n):
        for i in range(n):
            yield i
    total = 0
    for _ in range(n_iter // 100):
        for v in gen(100):
            total += v
    return total

N_ITER = 8_700_000        # Tools/benchmark_phoenix.py:1698
N_WARMUP = 3              # Tools/benchmark_phoenix.py:1719
N_MEASURE = 5             # Tools/benchmark_phoenix.py:1720

mode = sys.argv[1] if len(sys.argv) > 1 else "vanilla"

if mode == "jit":
    import _cinderx  # noqa: F401  -- triggers JIT init
    try:
        import cinderjit
        if hasattr(cinderjit, "auto"):
            cinderjit.auto()  # compile_after_n_calls=1000
        if hasattr(cinderjit, "enable_specialized_opcodes"):
            cinderjit.enable_specialized_opcodes()
    except ImportError:
        pass

# Warmup (matches harness _worker_jit n_warmup=3)
for _ in range(N_WARMUP):
    bench_gen_simple(N_ITER)

# Measure (matches harness _worker_jit n_measure=5)
times_ms = []
for _ in range(N_MEASURE):
    t0 = time.perf_counter()
    bench_gen_simple(N_ITER)
    t1 = time.perf_counter()
    times_ms.append((t1 - t0) * 1000.0)

mean_ms = sum(times_ms) / len(times_ms)
print(f"PROBE_RESULT mode={mode} ms={mean_ms:.2f}")
print(f"PROBE_DETAIL runs={times_ms}")
