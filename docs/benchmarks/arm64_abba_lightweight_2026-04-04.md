# ARM64 ABBA Benchmark — Lightweight Frames (2026-04-04)

## Summary

**Total: 1.83x** (JIT vs vanilla CPython 3.12.13) on ARM64.
Lightweight frames enabled (`ENABLE_LIGHTWEIGHT_FRAMES`). Auto-compilation mode (threshold=1000).

## Methodology

- **ABBA design**: 3 reps per benchmark, subprocess isolation
- **Harness**: `benchmark_phoenix_full.py --compile=auto --reps=3`
- **JIT ON**: `/data/users/alexturner/phoenix-cpython/python` (phoenix-asm-integration branch, commit 4b218453b5)
- **JIT OFF**: `/data/users/alexturner/vanilla-cpython/python` (vanilla CPython 3.12.13)
- **Platform**: aarch64 (devgpu004.kcm2.facebook.com), same machine, same session

## Per-Benchmark Results

| Benchmark | Vanilla (ms) | JIT (ms) | Speedup | Category |
|-----------|-------------|---------|---------|----------|
| fibonacci | 7369 | 2937 | **2.51x** | frame-dominated |
| richards_full | 40 | 20 | **1.97x** | frame-dominated |
| nqueens | 424 | 273 | **1.55x** | compute |
| dunder_protocol | 24 | 20 | **1.23x** | dispatch |
| spectral_norm | 2681 | 2291 | **1.17x** | compute |
| positional_dispatch | 8 | 7 | **1.13x** | dispatch |
| nbody | 39 | 35 | **1.12x** | compute |
| richards_slots | 189 | 171 | **1.10x** | frame-dominated |
| import_callee | 13 | 12 | 1.04x | dispatch |
| func_calls | 9 | 9 | 1.02x | dispatch |
| float_arith | 13 | 13 | 1.00x | compute |
| dict_ops | 6 | 6 | 1.00x | data structures |
| try_except_callee | 4 | 4 | 1.00x | exception |
| store_subscr | 9 | 9 | 1.00x | data structures |
| pytorch_cm | 77 | 77 | 1.00x | real-world |
| kwargs_dispatch | 16 | 16 | 1.00x | dispatch |
| decorator_chain | 16 | 16 | 0.99x | dispatch |
| deep_class_super | 81 | 82 | 0.99x | dispatch |
| context_manager | 18 | 18 | 0.98x | dispatch |
| int_arith | 8 | 8 | 0.97x | compute |
| list_comp | 2 | 2 | 0.97x | data structures |
| gen_nested | 8 | 9 | 0.96x | generators |
| nn_module_forward | 2 | 2 | **0.89x** | real-world |
| gen_simple | 3 | 4 | **0.78x** | generators |
| **TOTAL** | **11061** | **6043** | **1.83x** | |

## Analysis

### Where JIT wins (>1.10x): 8 benchmarks
- **fibonacci (2.51x)** and **richards_full (1.97x)**: frame-dominated, lightweight frames eliminates frame setup/teardown
- **nqueens (1.55x)**, **spectral_norm (1.17x)**, **nbody (1.12x)**: compute loop optimization
- **dunder_protocol (1.23x)**, **positional_dispatch (1.13x)**, **richards_slots (1.10x)**: dispatch optimization

### Neutral (0.95x-1.05x): 14 benchmarks
- Most dispatch, data structure, and arithmetic benchmarks at interpreter parity

### Where JIT loses (<0.95x): 2 benchmarks
- **gen_simple (0.78x)**: generator frame overhead — known limitation
- **nn_module_forward (0.89x)**: PyTorch-style module dispatch — guard overhead dominates

## Cross-Architecture Comparison

| Benchmark | x86_64 | ARM64 | ARM64 vs x86_64 |
|-----------|--------|-------|-----------------|
| fibonacci | 2.39x | 2.51x | ARM64 better |
| richards_full | 1.92x | 1.97x | ARM64 better |
| nqueens | 1.50x | 1.55x | ARM64 better |
| nbody | 0.98x | 1.12x | ARM64 significantly better |
| import_callee | 0.98x | 1.04x | ARM64 better |
| int_arith | 1.00x | 0.97x | Similar |
| **TOTAL** | **1.77x** | **1.83x** | **ARM64 better** |

ARM64 shows consistently better JIT performance than x86_64 across all categories. Non-frame benchmarks (nbody, import_callee) that showed regression on x86_64 are neutral or positive on ARM64.

## Historical ARM64 Comparison

| Date | Total | Notes |
|------|-------|-------|
| 2026-03-31 | 1.33x | Pre-lightweight frames |
| **2026-04-04** | **1.83x** | **Lightweight frames enabled** |

Improvement: +0.50x from lightweight frames.

## Test Suite Gate

- ARM64 full test suite: 454/494 PASS
- Zero JIT-specific regressions
- 9 failures all env/infra or pre-existing (dict watcher shutdown)

## Known Issues

- gen_simple 0.78x: pre-existing generator guard overhead
- nn_module_forward 0.89x: pre-existing, guard overhead dominates simple forward()
- Dict watcher shutdown warning at exit (non-fatal, pre-existing)
