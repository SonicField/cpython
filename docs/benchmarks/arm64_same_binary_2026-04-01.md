# ARM64 Same-Binary JIT vs Interpreter Benchmark

- **Date:** 2026-04-01
- **Platform:** aarch64 (devgpu004.kcm2.facebook.com)
- **Binary:** ~/local/phoenix-cpython/python (Phoenix CPython 3.12.13, Clang 21.1.7)
- **Build flags:** no LTO, no PGO, RelWithDebInfo (-O2)
- **Methodology:** Same binary, JIT-on (post force_compile/auto-compile) vs JIT-off (pre-compilation). 3 runs per measurement, min time reported.
- **JIT verification:** cinderjit.is_jit_compiled() = True for all benchmarks. 16 PHX compilation messages. 7 compiled functions.
- **compile_after_n_calls:** 1000
- **Raw output:** devgpu004:/tmp/jit_vs_interp_results.txt
- **Script:** /tmp/benchmark_jit_vs_interp.py on devgpu004

## Purpose

Eliminates version/compiler confounds from prior benchmarks that compared Phoenix 3.12.13 (Clang) vs cinderx_dev 3.12.9 (GCC). Both measurements use the identical binary.

## Tolerance Check: PASS

- Geometric mean: **1.11x** (>= 1.0x required)
- All benchmarks >= 1.00x (no regressions)
- All functions verified compiled via cinderjit.is_jit_compiled()

## Results (6 benchmarks)

| Benchmark | Interp ms | JIT ms | Speedup | Compiled |
|---|---|---|---|---|
| fibonacci | 504.2 | 503.6 | 1.00x | True |
| int_arith | 6.2 | 4.8 | 1.31x | True |
| float_arith | 7.0 | 6.7 | 1.05x | True |
| func_calls | 4.6 | 3.8 | 1.22x | True |
| dict_ops | 3.5 | 3.4 | 1.02x | True |
| list_comp | 2.1 | 2.0 | 1.08x | True |

**Geomean: 1.11x**

## Methodology Caveats

1. **fibonacci 1.00x is a measurement artifact:** fib() auto-compiles during initial warmup (10 iterations = ~109K recursive calls, exceeding threshold=1000). Both "cold" and "warm" measurements are post-compilation. A proper measurement requires independent function copies to prevent auto-compilation of the cold copy.

2. **6 benchmarks, not 21:** The full benchmark_phoenix.py has 21 JIT benchmarks. This comparison covers 6 representative ones. Full 21-benchmark comparison requires a --no-jit harness flag (requested).

3. **Not ABBA interleaved:** Measurements are sequential (cold first, warm second). Thermal drift is a potential confound, though the measurements are short enough (~seconds each) to minimize this.

## JIT Compilation Verification

Separate diagnostic confirmed JIT compilation is functional on ARM64:

```
fib(20) cold: 1834.7 us
fib(20) warm: 502.1 us → 3.65x speedup (single function, isolated test)
add_simple compiled after 2000 calls: True
compiled_functions: 2
```

This test (with _cinderx imported first) shows auto-compilation triggering correctly at threshold=1000.

## Comparison with Prior Results

| Date | Methodology | Geomean | Notes |
|---|---|---|---|
| 2026-03-31 | Phoenix 3.12.13 vs cinderx_dev 3.12.9 | 1.14x | Cross-version, cross-compiler |
| 2026-04-01 | Same binary JIT-on vs JIT-off | 1.11x | Apples-to-apples, 6 benchmarks |

The 1.14x → 1.11x delta is within expected range given different benchmark sets and methodology improvements.

## Session Context

- ARM64 benchmark crashes: ALL 3 FIXED this session (stale binary + FP V-bit encoding)
- Single-root-cause hypothesis: FALSIFIED (2/3 stale binary, 1/3 real V-bit bug)
- JIT compilation initially questioned by Alex — confirmed working after proper _cinderx import
- cinderjit module requires `import _cinderx` first (lazy registration)
