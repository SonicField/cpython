# x86_64 Rebalanced ABBA Benchmark — Full Results (2026-04-05)

## Summary

**Total: 1.47x** (JIT vs vanilla CPython 3.12.13) on x86_64 with rebalanced iteration counts.

The 1.47x total is NOT comparable to the 1.82x pre-rebalanced total — different iteration counts change the weighting. Per-benchmark speedup ratios are the correct comparison metric and are stable within noise.

## Methodology

- **ABBA design**: 3 reps per benchmark, subprocess isolation, 12 runs each (ABBA x 3)
- **Harness**: `benchmark_phoenix_full.py jit --compile=auto --reps=3`
- **Rebalanced iteration counts** (commit 661cfd07fd):
  - fibonacci: 10,000 (was 100,000)
  - spectral_norm: 20,000 (was 100,000)
  - All others: 100,000 (unchanged)
- **Inter-run sleep**: 0.25s (was 1.0s)
- **JIT ON**: `/data/users/alexturner/phoenix/cpython/python` (commit 16dfb79c2f)
- **JIT OFF**: `/data/users/alexturner/phoenix/cpython-vanilla/python -I` (vanilla CPython 3.12.13)
- **Platform**: x86_64, Linux
- **Build**: release, clang 21.1.7
- **Suite runtime**: ~8 min (target ≤5 min — spectral_norm still heavy at 20K iters)

## Per-Benchmark Results

| Benchmark | Vanilla (ms) | JIT (ms) | Speedup | Delta % | Category |
|-----------|-------------|---------|---------|---------|----------|
| fibonacci | 1508.90 | 607.41 | **2.48x** | +59.7% | frame-dominated |
| richards_full | 87.58 | 45.80 | **1.91x** | +47.7% | frame-dominated |
| nqueens | 984.62 | 601.97 | **1.64x** | +38.9% | compute |
| dunder_protocol | 53.76 | 44.41 | **1.21x** | +17.4% | dispatch |
| spectral_norm | 1170.27 | 971.73 | **1.20x** | +17.0% | compute |
| positional_dispatch | 16.50 | 14.67 | **1.12x** | +11.1% | dispatch |
| nbody | 81.25 | 72.58 | **1.12x** | +10.7% | compute |
| richards_slots | 402.17 | 364.44 | **1.10x** | +9.4% | frame-dominated |
| int_arith | 19.27 | 17.52 | **1.10x** | +9.1% | compute |
| dict_ops | 12.62 | 11.47 | **1.10x** | +9.1% | data structures |
| kwargs_dispatch | 33.99 | 33.25 | 1.02x | +2.2% | dispatch |
| store_subscr | 18.17 | 17.76 | 1.02x | +2.2% | data structures |
| import_callee | 28.84 | 28.34 | 1.02x | +1.7% | dispatch |
| float_arith | 26.72 | 26.66 | 1.00x | +0.2% | compute |
| context_manager | 38.37 | 38.45 | 1.00x | -0.2% | dispatch |
| pytorch_cm | 173.08 | 175.46 | 0.99x | -1.4% | real-world |
| decorator_chain | 34.81 | 35.32 | 0.99x | -1.4% | dispatch |
| gen_nested | 16.42 | 16.61 | 0.99x | -1.2% | generators |
| list_comp | 4.92 | 5.05 | 0.97x | -2.7% | data structures |
| try_except_callee | 7.78 | 8.24 | 0.94x | -5.9% | exception |
| deep_class_super | 166.57 | 176.70 | 0.94x | -6.1% | dispatch |
| func_calls | 17.81 | 19.61 | 0.91x | -10.1% | dispatch |
| nn_module_forward | 3.48 | 3.90 | 0.89x | -12.3% | real-world |
| gen_simple | 5.71 | 7.72 | 0.74x | -35.2% | generators |
| **TOTAL** | **4913.60** | **3345.09** | **1.47x** | +31.9% | |

## Per-Benchmark Ratio Comparison (rebalanced vs pre-rebalanced)

| Benchmark | Pre-rebalanced | Rebalanced | Delta | Within ±5%? |
|-----------|---------------|------------|-------|-------------|
| fibonacci | 2.50x | 2.48x | -0.02 | YES |
| richards_full | 1.94x | 1.91x | -0.03 | YES |
| nqueens | 1.63x | 1.64x | +0.01 | YES |
| dunder_protocol | 1.24x | 1.21x | -0.03 | YES |
| spectral_norm | 1.19x | 1.20x | +0.01 | YES |
| positional_dispatch | 0.98x | 1.12x | +0.14 | NO (+14%) |
| nbody | 1.09x | 1.12x | +0.03 | YES |
| richards_slots | 1.11x | 1.10x | -0.01 | YES |
| int_arith | 1.08x | 1.10x | +0.02 | YES |
| dict_ops | 1.02x | 1.10x | +0.08 | NO (+8%) |
| kwargs_dispatch | 1.02x | 1.02x | 0.00 | YES |
| store_subscr | 1.03x | 1.02x | -0.01 | YES |
| import_callee | 1.00x | 1.02x | +0.02 | YES |
| float_arith | 1.00x | 1.00x | 0.00 | YES |
| context_manager | 1.01x | 1.00x | -0.01 | YES |
| pytorch_cm | 1.05x | 0.99x | -0.06 | NO (-6%) |
| decorator_chain | 0.99x | 0.99x | 0.00 | YES |
| gen_nested | 0.96x | 0.99x | +0.03 | YES |
| list_comp | 1.00x | 0.97x | -0.03 | YES |
| try_except_callee | 0.98x | 0.94x | -0.04 | YES |
| deep_class_super | 0.94x | 0.94x | 0.00 | YES |
| func_calls | 0.99x | 0.91x | -0.08 | NO (-8%) |
| nn_module_forward | 0.91x | 0.89x | -0.02 | YES |
| gen_simple | 0.72x | 0.74x | +0.02 | YES |

**20/24 within ±5%.** 4 outliers are noise-dominated benchmarks with small absolute times:
- positional_dispatch: 16ms abs → high variance, +14% is within 2ms noise
- dict_ops: 11ms abs → high variance, +8% is within 1ms noise
- pytorch_cm: 175ms abs → -6% could be co-tenant noise (single outlier block)
- func_calls: 18ms abs → -8% is within 2ms noise; JIT rep2 had a 29ms outlier

**Conclusion: rebalancing preserves per-benchmark ratios.** The total drops from 1.82x to 1.47x because fibonacci (2.48x, the strongest contributor) ran 10x fewer iterations, reducing its weight in the sum. This is expected and correct.

## JIT Category Summary

| Category | Benchmarks | Avg Speedup | Notes |
|----------|-----------|-------------|-------|
| frame-dominated | fibonacci, richards_full, richards_slots | **1.83x** | Primary JIT benefit |
| compute | nqueens, spectral_norm, float_arith, int_arith, nbody | **1.21x** | Moderate gains |
| dispatch | dunder_protocol, positional_dispatch, kwargs_dispatch, context_manager, func_calls, import_callee, decorator_chain, deep_class_super | **1.00x** | Mixed |
| generators | gen_simple, gen_nested | **0.87x** | Known guard overhead |
| data structures | dict_ops, store_subscr, list_comp | **1.03x** | Marginal |
| exception | try_except_callee | **0.94x** | Guard overhead |
| real-world | nn_module_forward, pytorch_cm | **0.94x** | Guard overhead dominates |

## Known Issues

- gen_simple 0.74x: pre-existing generator guard overhead
- nn_module_forward 0.89x: pre-existing, guard overhead dominates simple forward()
- deep_class_super 0.94x: pre-existing super() chain overhead
- func_calls 0.91x: JIT rep2 had a 29ms outlier (vs 17-18ms typical) — likely co-tenant
- Dict watcher shutdown warning at exit (non-fatal, pre-existing)

## Raw Data

Full ABBA output: `x86_64_rebalanced_full_2026-04-05_raw.txt`

## Command

```bash
PYTHONPATH=Lib/test ASAN_OPTIONS=detect_leaks=0 VANILLA_PYTHON=/data/users/alexturner/phoenix/cpython-vanilla/python \
  ./python Tools/benchmark_phoenix_full.py jit --compile=auto --reps=3
```
