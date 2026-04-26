# cinderx_dev Oracle Environment Specification

Canonical procedure for using cinderx_dev (devgpu004:~/local/cinderx_dev/cinderx)
as Phoenix-vs-Cinder perf oracle. Closes the env-equivalence gap pythia
D-1777065234 #3 flagged + assigned to theologian (deferred 2026-04-22, closed
2026-04-26 post pythia #146 + supervisor 01:30:42Z directive).

## Why This Spec Exists

cinderx_dev oracle was attempted 2026-04-26T01:16:27Z to discriminate Phoenix
W27c-introduced perf regressions from Cinder-inherited ones. Attempt FAILED:
- cinderx_dev `-X jit` showed ZERO speedup vs no-JIT for 3 benches (JIT didn't
  activate at warmup threshold, OR `-X jit` is wrong invocation, OR
  interpreter-overhead dominated).
- ARM64-only cinderx_dev cross-arched against x86_64 Phoenix readings further
  confounded the comparison.

Result: oracle could not discriminate. Phoenix-vs-Cinder verdict on perf
remained unresolved. PROVISIONAL-PRE-EXISTING labeling could not promote to
PRE-EXISTING per Alex feedback_assume_phoenix_regression.md.

This spec resolves the env-equivalence gap so future oracle invocations
discriminate cleanly.

## Required Environment Equivalence

### 1. Build Equivalence

cinderx_dev binary on devgpu004 must match Phoenix build flags as closely as
possible:
- **Compiler**: clang (Phoenix uses clang 21 per memory; verify cinderx_dev
  also clang 21+)
- **Optimization**: `-O2` (RelWithDebInfo) — matches `cinderx_dev build flags
  on devgpu004: NO PGO, NO LTO, RelWithDebInfo (-O2)` per project memory
- **Phoenix x86_64 build**: also `-O2` non-LTO for fair comparison (LTO
  build is separate baseline; cinderx_dev has none, so use non-LTO Phoenix
  for oracle)
- **Architecture**: prefer SAME-arch comparison. devgpu004 is ARM64; if
  Phoenix x86_64 readings are the regression source, build Phoenix on ARM64
  via nbs-remote-session devgpu004 + scripts/build_phoenix.sh and compare
  ARM64 Phoenix vs ARM64 cinderx_dev. Cross-arch comparison is INVALID.

### 2. JIT Activation Verification

**v2 (2026-04-26 post-execution): Phoenix and cinderx_dev have different
JIT-activation API names + import patterns. Both require explicit module
load before force_compile.**

Phoenix-side (verified working at push 60):
```bash
PYTHONJITALL=1 PYTHONPATH=/path/to/benchmark phoenix_python -c "
import _cinderx           # MUST import first (loads C extension)
import cinderjit          # only available after _cinderx import
import benchmark_phoenix
cinderjit.force_compile(benchmark_phoenix.bench_int_arith)
assert cinderjit.is_jit_compiled(benchmark_phoenix.bench_int_arith)
"
```

cinderx_dev-side (verified working 2026-04-26T02:39Z):
```bash
LD_LIBRARY_PATH=/home/alexturner/local/cinderx_dev/python-3.12 \
PYTHONPATH=/home/alexturner/local/cinderx_dev/cinderx/cinderx/PythonLib:/path/to/benchmark \
~/local/cinderx_dev/python-3.12/python -c "
from cinderx import jit   # cinderx package, jit submodule
import benchmark_phoenix
jit.force_compile(benchmark_phoenix.bench_int_arith)
assert jit.is_jit_compiled(benchmark_phoenix.bench_int_arith)
"
```

**Critical v2 corrections from v1:**
- Phoenix: NOT `import cinderjit` directly — requires `import _cinderx` first
  to load the C extension that registers cinderjit module. v1 spec was wrong.
- cinderx_dev: NOT `import cinderjit` — uses `from cinderx import jit`. Module
  layout differs from Phoenix.
- cinderx_dev binary path: `~/local/cinderx_dev/python-3.12/python` (NOT
  `~/local/cinderx_dev/cinderx/python` as v1 assumed).
- cinderx_dev requires LD_LIBRARY_PATH to find libpython3.12.so.1.0 +
  PYTHONPATH to find cinderx package.
- `-X jit` flag: insufficient (didn't activate JIT in 2026-04-26T01:16Z dry-run).
  Always use force_compile + is_jit_compiled to verify activation.

If neither force_compile nor is_jit_compiled returns True, the oracle reading
is INVALID (JIT didn't run; comparison is interpreter-vs-something, not
JIT-vs-JIT).

### 3. Test-Harness Equivalence

Same benchmark_phoenix.py harness (copy to devgpu004:/tmp/oracle_tools/),
same `--reps=3` ABBA, same vanilla baseline (`../cpython-vanilla/python` on
devgpu004 — note W-ARM64-VANILLA-INFRA debt: vanilla path may not exist on
devgpu004, blocking ABBA; use targeted bench-only invocation with
explicit timing if vanilla missing).

For targeted bench timing without vanilla baseline:
```bash
JIT_ENABLE=1 cinderx_dev_python -c "
import time, benchmark_phoenix
benchmark_phoenix.bench_int_arith(50000)  # warmup
times = []
for _ in range(5):
    t0 = time.perf_counter()
    benchmark_phoenix.bench_int_arith(50000)
    times.append(time.perf_counter() - t0)
print(f'median={sorted(times)[2]*1e3:.1f}ms')
"
```

Compare median directly between Phoenix-JIT and cinderx_dev-JIT (no vanilla
needed for inheritance discrimination).

### 4. Warmup Threshold

cinderx_dev auto-compile threshold may differ from Phoenix (Phoenix=1000 per
memory; cinderx_dev unknown). Use `cinderjit.force_compile` (Method 1) to
bypass threshold uncertainty. If force_compile unavailable, run
`bench(n_iter=50000)` first to amortize warmup, then time subsequent runs.

## Oracle Invocation Procedure

For ANY perf-regression discrimination question (Phoenix-introduced vs
Cinder-inherited):

1. **Same-arch verify**: confirm Phoenix and cinderx_dev are same arch
   (ARM64 most common since cinderx_dev is on devgpu004). If Phoenix is
   x86_64-only and bench shows regression, build Phoenix ARM64 first.
2. **JIT activation verify**: run Method 1 (force_compile + is_jit_compiled)
   on cinderx_dev. If verification fails, abort oracle — fix env first.
3. **Targeted bench timing**: run targeted bench (Method 3) on both
   Phoenix-JIT and cinderx_dev-JIT, 5-run median.
4. **Discrimination**:
   - cinderx_dev-JIT median ≈ Phoenix-JIT median → INHERITED (close as
     not-Phoenix-introduced).
   - cinderx_dev-JIT median significantly faster than Phoenix-JIT median
     → PHOENIX-INTRODUCED (open bisect against cinderx_dev oracle as
     known-clean reference).

## Failure Modes (Observed 2026-04-26T01:16:27Z + 02:33Z dry-run)

- **JIT didn't activate** (-X jit insufficient): use force_compile + verify.
- **Cross-arch confound** (ARM64 cinderx_dev vs x86_64 Phoenix): build
  same-arch first.
- **Vanilla baseline missing on devgpu004** (W-ARM64-VANILLA-INFRA debt):
  use targeted bench median instead of ABBA harness.
- **Wrong binary path** (v1 assumed `cinderx/python`): actual is
  `python-3.12/python`.
- **Missing shared library** (libpython3.12.so.1.0): set LD_LIBRARY_PATH.
- **Wrong import name** (v1 assumed cinderjit on cinderx_dev): use
  `from cinderx import jit`.
- **Phoenix import order** (v1 assumed `import cinderjit` works directly):
  must `import _cinderx` first.

## Application Result (2026-04-26T02:39Z)

Oracle executed try_except_callee + nn_module_forward + bench_int_arith
discrimination per W-PERF-PRE-W27C-BISECT (push 60 PROVISIONAL labels):

| Bench                   | cinderx_dev | Phoenix-ARM64-p60 | Verdict           |
|-------------------------|-------------|-------------------|-------------------|
| bench_int_arith         | 4.1ms       | 3.7ms             | Phoenix 10% faster |
| bench_try_except_callee | 1.6ms       | 1.3ms             | Phoenix 19% faster |
| bench_deep_class        | 1.1ms       | 1.0ms             | Phoenix 9% faster  |

Verdict: try_except_callee + nn_module_forward (= bench_deep_class)
sub-1.0x-vs-vanilla = INHERITED FROM CINDER. Phoenix not regression
source. PROVISIONAL → CONFIRMED-INHERITED. W-PERF-PRE-W27C-BISECT
CLOSED-by-oracle-confirmation 2026-04-26T02:40:16Z.

If future oracle confirms INHERITED, PROVISIONAL → CONFIRMED-INHERITED.
If future oracle confirms PHOENIX-INTRODUCED, open bisect to localize commit.
