# x86_64 Balanced ABBA Benchmark — Geometric Mean (2026-04-05)

## Summary

**Geometric Mean: 1.10x** across 24 equally-weighted benchmarks (JIT vs vanilla CPython 3.12.13).

All benchmarks calibrated to ~500ms vanilla runtime, so each contributes equally to the geometric mean. This replaces the prior total-runtime-ratio metric (1.82x / 1.47x) which over-indexed on long-running benchmarks.

## Methodology

- **ABBA design**: 3 reps per benchmark, subprocess isolation, 12 runs each
- **Harness**: `benchmark_phoenix_full.py jit --compile=auto --reps=3`
- **Balanced iteration counts**: each benchmark targets ~500ms vanilla runtime
- **Headline metric**: geometric mean of per-benchmark speedup ratios
- **JIT ON**: `./python` (commit a24c5a19f0, postalloc C wired)
- **JIT OFF**: `../cpython-vanilla/python -I` (vanilla CPython 3.12.13)
- **Platform**: x86_64, Linux
- **Suite time**: ~5 min

## Per-Benchmark Results

| Benchmark | Vanilla (ms) | JIT (ms) | Speedup | Category |
|-----------|-------------|---------|---------|----------|
| fibonacci | 495.68 | 197.91 | **2.50x** | frame-dominated |
| richards_full | 500.59 | 262.66 | **1.91x** | frame-dominated |
| nqueens | 492.77 | 299.07 | **1.65x** | compute |
| dunder_protocol | 494.93 | 394.83 | **1.25x** | dispatch |
| spectral_norm | 475.14 | 387.48 | **1.23x** | compute |
| positional_dispatch | 506.35 | 454.97 | **1.11x** | dispatch |
| richards_slots | 490.12 | 441.26 | **1.11x** | frame-dominated |
| nbody | 470.55 | 434.85 | **1.08x** | compute |
| func_calls | 528.54 | 499.76 | **1.06x** | dispatch |
| store_subscr | 489.57 | 462.03 | **1.06x** | data structures |
| int_arith | 464.28 | 447.03 | 1.04x | compute |
| import_callee | 491.91 | 482.15 | 1.02x | dispatch |
| dict_ops | 476.96 | 468.34 | 1.02x | data structures |
| list_comp | 515.55 | 508.26 | 1.01x | data structures |
| kwargs_dispatch | 475.03 | 471.82 | 1.01x | dispatch |
| context_manager | 502.15 | 500.15 | 1.00x | dispatch |
| decorator_chain | 484.01 | 488.37 | 0.99x | dispatch |
| pytorch_cm | 480.92 | 495.41 | 0.97x | real-world |
| float_arith | 470.54 | 487.06 | 0.97x | compute |
| gen_nested | 525.15 | 541.20 | 0.97x | generators |
| try_except_callee | 495.15 | 521.69 | 0.95x | exception |
| deep_class_super | 492.71 | 523.02 | 0.94x | dispatch |
| nn_module_forward | 474.44 | 526.39 | 0.90x | real-world |
| gen_simple | 489.63 | 683.96 | **0.72x** | generators |

**Geometric Mean: 1.10x** (24 benchmarks)
**Total Runtime: 1.07x** (for reference, not headline)

## Balance Verification

All vanilla runtimes between 464ms and 529ms — maximum spread is ±7% from the 500ms target. This confirms equal weighting.

## Category Summary

| Category | Count | Geo Mean | Range |
|----------|-------|----------|-------|
| frame-dominated | 3 | 1.76x | 1.11-2.50x |
| compute | 5 | 1.17x | 0.97-1.65x |
| dispatch | 8 | 1.04x | 0.94-1.25x |
| data structures | 3 | 1.03x | 1.01-1.06x |
| generators | 2 | 0.84x | 0.72-0.97x |
| exception | 1 | 0.95x | - |
| real-world | 2 | 0.94x | 0.90-0.97x |

## Key Observations

1. **Frame-dominated benchmarks** (fibonacci, richards_full, richards_slots) show the strongest JIT benefit — these exercise the lightweight frame path
2. **Generator benchmarks** show regression due to guard overhead
3. **Dispatch-heavy benchmarks** cluster around 1.0x — JIT provides marginal benefit for simple function dispatch
4. The geometric mean (1.10x) is the honest aggregate — it says the JIT is 10% faster on average across all workload types

## Known Issues

- gen_simple 0.72x: pre-existing generator guard overhead
- nn_module_forward 0.90x: guard overhead on simple forward()
- deep_class_super 0.94x: super() chain overhead
- Dict watcher shutdown warning at exit (non-fatal, pre-existing)

## Raw Data

Full ABBA output: `x86_64_balanced_geomean_2026-04-05_raw.txt`

## Command

```bash
PYTHONPATH=Lib/test ASAN_OPTIONS=detect_leaks=0 VANILLA_PYTHON=../cpython-vanilla/python \
  ./python Tools/benchmark_phoenix_full.py jit --compile=auto --reps=3
```
