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

cinderx_dev JIT activation cannot be assumed from `-X jit` flag alone.
Verification REQUIRED:

```bash
# Method 1: explicit compile + verify
PYTHONPATH=/path/to/benchmark cinderx_dev_python -c "
import cinderjit, benchmark_phoenix
cinderjit.force_compile(benchmark_phoenix.bench_int_arith)
assert cinderjit.is_jit_compiled(benchmark_phoenix.bench_int_arith), 'JIT not active'
"
```

If `cinderjit.force_compile` or `cinderjit.is_jit_compiled` are not exposed
in cinderx_dev (different API), use Cinder-equivalent:
```bash
# Method 2: cinder API (verify via cinder.is_jit_compiled or _cinder)
cinderx_dev_python -c "import _cinderjit; print(_cinderjit.get_runtime_helper_addr())"
```

If neither method confirms JIT activation, the oracle reading is INVALID
(JIT didn't run; comparison is interpreter-vs-Phoenix-JIT, not
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

## Failure Modes (Observed 2026-04-26T01:16:27Z)

- **JIT didn't activate** (-X jit insufficient): use Method 1 force_compile.
- **Cross-arch confound** (ARM64 cinderx_dev vs x86_64 Phoenix): build
  same-arch first.
- **Vanilla baseline missing on devgpu004** (W-ARM64-VANILLA-INFRA debt):
  use targeted bench median instead of ABBA harness.

## Application

Next oracle invocation: try_except_callee + nn_module_forward
discrimination per W-PERF-PRE-W27C-BISECT (push 60 PROVISIONAL-PRE-EXISTING
labels). Apply this spec to obtain valid Phoenix-vs-Cinder verdict.

If oracle confirms INHERITED, PROVISIONAL → PRE-EXISTING (true).
If oracle confirms PHOENIX-INTRODUCED, open bisect to localize commit.
