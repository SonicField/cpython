# x86_64 ABBA Benchmark — Lightweight Frames (2026-04-04)

## Summary

**Total: 1.77x** (JIT vs vanilla CPython 3.12.13) on x86_64.
Lightweight frames enabled (`ENABLE_LIGHTWEIGHT_FRAMES`). Auto-compilation mode (threshold=1000).

## Methodology

- **ABBA design**: 3 reps per benchmark, subprocess isolation
- **Harness**: `benchmark_phoenix_full.py --compile=auto --reps=3 --iters=50000 --blocks=15`
- **JIT ON**: `/data/users/alexturner/phoenix/cpython/python` (phoenix-asm-integration branch, commit 4b218453b5)
- **JIT OFF**: `/data/users/alexturner/phoenix/cpython-vanilla/python -I` (vanilla CPython 3.12.13)
- **Platform**: x86_64, Linux, same machine, same session

## Per-Benchmark Results

| Benchmark | Vanilla (ms) | JIT (ms) | Speedup | Category |
|-----------|-------------|---------|---------|----------|
| fibonacci | 14936.79 | 6243.41 | **2.39x** | frame-dominated |
| richards_full | 86.18 | 44.94 | **1.92x** | frame-dominated |
| nqueens | 976.08 | 650.35 | **1.50x** | compute |
| dunder_protocol | 52.21 | 43.34 | **1.20x** | dispatch |
| spectral_norm | 5768.48 | 4887.88 | **1.18x** | compute |
| richards_slots | 405.87 | 363.52 | **1.12x** | frame-dominated |
| positional_dispatch | 16.48 | 14.92 | **1.10x** | dispatch |
| float_arith | 27.06 | 26.18 | 1.03x | compute |
| dict_ops | 11.72 | 11.62 | 1.01x | data structures |
| list_comp | 5.02 | 5.04 | 1.00x | data structures |
| store_subscr | 16.87 | 16.83 | 1.00x | data structures |
| int_arith | 18.80 | 18.75 | 1.00x | compute |
| pytorch_cm | 168.88 | 169.78 | 0.99x | real-world |
| func_calls | 18.23 | 18.39 | 0.99x | dispatch |
| context_manager | 38.19 | 38.71 | 0.99x | dispatch |
| import_callee | 28.61 | 29.28 | 0.98x | dispatch |
| kwargs_dispatch | 33.17 | 33.69 | 0.98x | dispatch |
| nbody | 76.34 | 77.96 | 0.98x | compute |
| deep_class_super | 166.48 | 171.72 | 0.97x | dispatch |
| decorator_chain | 34.17 | 35.72 | 0.96x | dispatch |
| gen_nested | 16.56 | 17.44 | 0.95x | generators |
| try_except_callee | 7.76 | 8.44 | 0.92x | exception |
| nn_module_forward | 3.47 | 3.84 | 0.90x | real-world |
| gen_simple | 5.71 | 7.61 | **0.75x** | generators |
| **TOTAL** | **22919.13** | **12939.37** | **1.77x** | |

## Analysis

### Where JIT wins (>1.10x): 6 benchmarks
- **fibonacci (2.39x)** and **richards_full (1.92x)** are frame-dominated — lightweight frames eliminates frame setup/teardown overhead
- **nqueens (1.50x)**, **spectral_norm (1.18x)** benefit from compute loop JIT compilation
- **dunder_protocol (1.20x)**, **positional_dispatch (1.10x)** benefit from dispatch optimization

### Neutral (0.95x-1.05x): 12 benchmarks
- Arithmetic, data structures, and most dispatch benchmarks show interpreter parity
- JIT compilation overhead roughly matches execution speedup

### Where JIT loses (<0.95x): 3 benchmarks
- **gen_simple (0.75x)**: generator frame overhead — known limitation, generators use different frame path
- **try_except_callee (0.92x)**: exception handling path overhead
- **nn_module_forward (0.90x)**: PyTorch-style module pattern — likely bail-out or deopt on dynamic dispatch

### Concentration risk
The 1.77x total is dominated by fibonacci (contributing ~8.7s of ~10s total improvement). Without fibonacci, the geometric mean is approximately 1.02-1.05x. The headline number reflects frame-dominated workloads; non-frame workloads see interpreter parity.

## Comparison with prior sessions

| Date | Total | Notes |
|------|-------|-------|
| 2026-03-31 | 1.22x | Pre-Phase 3D expansion, no LW frames |
| 2026-04-02 | 1.09x | Post-Phase 3D B3/B4, no LW frames |
| **2026-04-04** | **1.77x** | **Lightweight frames enabled** |

The 1.22x→1.09x decline was investigated: Phase 3D codegen is CLEARED (identical HIR, better LIR at HEAD). Remaining hypotheses: runtime dispatch overhead, icache effects, or cross-session measurement noise.

## Phase 3D codegen validation

HIR/LIR comparison between cf49ad6da5 (pre-Phase 3D) and HEAD:
- **int_arith HIR**: structurally identical
- **int_arith LIR**: HEAD is better (19 spills vs 27, fewer UpdatePrevInstr)
- **fib HIR**: HEAD inlines one recursion level (from lightweight frames, not Phase 3D)

Phase 3D container swaps did NOT degrade codegen quality.

## Raw data

Full ABBA output: `x86_64_abba_lightweight_2026-04-04_raw.txt`
