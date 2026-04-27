# gen_simple regression investigation — 2026-04-27

**Status:** ROOT-CAUSE-CLASS LOCALIZED. Phoenix JIT generator-compilation
cost. Bisect-by-commit + cinderx_dev oracle queued for next-session
fix-class workstream.

**Trigger:** pythia #195 2026-04-27T15:10:01Z (4-commit convergent-
negative pattern: BLME b2 / kwnames / preloader_ / push-59 baseline all
report `gen_simple` at ~0.73x without root-cause investigation).

**Authorization:** supervisor 15:10:44Z (evidence-class, distinct from
spec amendment). Per `feedback_assume_phoenix_regression.md` (Alex
2026-04-24): every regression presumed Phoenix-introduced;
`feedback_falsifier_convergent_negative.md`: convergent-negative
delta needs `investigated-no-quick-fix` framing, not pre-existing
escape.

## Workload

`Tools/benchmark_phoenix.py:bench_gen_simple` (line 777):

```python
def bench_gen_simple(n_iter):
    """Simple generator iteration."""
    def gen(n):
        for i in range(n):
            yield i
    total = 0
    for _ in range(n_iter // 100):
        for v in gen(100):
            total += v
    return total
```

ABBA-calibrated `n_iter = 8_700_000` per `_PER_BENCH_ITERS`
(benchmark_phoenix.py:1698). 87,000 outer iterations × 1 `gen()`
call each = 87,000 `gen()` calls per measured run; comfortably
crosses the 1000-call auto-compile threshold (per MEMORY:
`Auto-compilation threshold: 1000 calls (matches cinderx_dev)`).

## Three-config focused timing (n_iter=8,700,000, 3-rep median)

| Config | Median | vs vanilla | vs Phoenix interpreter |
|---|---|---|---|
| Vanilla CPython 3.12 | 505 ms | 1.00x | — |
| Phoenix interpreter (no `cinderjit.auto`) | 472 ms | **0.94x (FASTER)** | 1.00x |
| Phoenix + `cinderjit.auto` (= ABBA condition) | 670 ms | **1.33x (SLOWER)** | 1.42x |

Phoenix's interpreter beats vanilla CPython by 6% on `gen_simple`.
Phoenix's JIT path loses 33% vs vanilla AND 42% vs Phoenix's own
interpreter on the same workload.

The 0.73x ABBA ratio reproduced consistently across the four observer
commits is **the Phoenix JIT generator-compilation cost itself**, not
baseline noise and not a property of those four commits' code paths
(BLME mutating-pass, KW-arg dispatch field, builder-state plumbing —
none touch generator FRAME ops or generator codegen).

## Falsification of the 'pre-existing structural causal absence' framing

The dismissal applied to BLME b2 / kwnames / preloader_ ABBA reports
('gen_simple regression is pre-existing, structural causal absence
because this commit doesn't touch generator code') was structurally
INCONSISTENT:

- The 4 commits don't touch generator code → they CAN'T have introduced
  the regression. **Correct conclusion** for those commits in
  isolation.
- BUT 'pre-existing' as a meta-classification implied the regression
  was inherited (pre-Phoenix) noise, requiring no Phoenix-side root
  cause. **Incorrect conclusion** at the meta level — the regression
  IS Phoenix-introduced, just upstream of the four observed commits.

The accurate framing per theologian 15:13:57Z: 'investigated; the 4
commits are observers of an upstream Phoenix-introduced JIT-generator-
codegen cost; structural causal absence for the 4 commits is accurate
at the per-commit layer; the meta-framing required investigation, not
dismissal.'

This satisfies `feedback_falsifier_convergent_negative.md` discipline.

## Mechanism (hypothesis, post-investigation)

Phoenix `cinderjit.auto` JIT-compiles the inner `gen()` closure once
the call count crosses the 1000-call threshold (~the 12th outer
iteration). The JIT'd generator runs slower per-yield than the
interpreter on this workload — likely candidates:

- Phoenix's JIT generator codegen has per-yield overhead
  (frame-setup / yield-resume bookkeeping) heavier than the
  interpreter's generator dispatch
- Auto-compile threshold + per-call counting overhead amortised across
  too few iterations to recoup

Not yet falsified or root-caused at the Phoenix-source layer — bisect
or perf-record is the next step.

## Next-step options for fix-class workstream (next session)

(a) **cinderx_dev oracle bisect** (Tier-1 falsifier per
    `feedback_assume_phoenix_regression.md`). Run `bench_gen_simple`
    at `n_iter=8_700_000` under cinderx_dev (the upstream Cinder
    JIT). If cinderx_dev shows the same ~1.33x cost, this is
    inherited Cinder behavior (not Phoenix-introduced); workstream
    pivots to architectural-cost framing. If cinderx_dev is faster,
    Phoenix regressed at a specific commit — git bisect against
    cinderx_dev as oracle finds the introducing commit. Either is
    decisive evidence for next-session scoping. **Recommended first
    step.**

(b) **Profile + perf-record** the JIT'd `gen()` execution vs Phoenix
    interpreter's generator dispatch. Identifies the per-yield
    hotspot at function granularity. Useful regardless of (a)
    outcome.

(c) **Park** as 'investigated-no-quick-fix' if (a) shows
    architectural cost and (b) shows no localizable hotspot. Add
    `bench_gen_simple` to a known-cost benchmark list; ABBA gates
    no longer flag the steady-state ratio as a regression.

Theologian 15:13:57Z recommends (a) first; supervisor 15:14:01Z
defers to next-session.

## Cascade (per supervisor 15:14:01Z)

Going-forward ABBA-report convention: `gen_simple` is no longer
'pre-existing structural causal absence'. New framing:

> gen_simple X.XXx — Phoenix JIT generator-compilation regression,
> root-cause class identified (see docs/benchmarks/
> gen_simple_investigation_2026-04-27.md), bisect queued for
> next-session fix-class workstream.

Until the bisect lands, this artifact is the workstream-tag the
gatekeeper >5% BLOCK rule (15:12:00Z) accepts as the evidence-class
root-cause citation that distinguishes the regression from a generic
pre-existing dismissal (per `feedback_no_workarounds.md` — the
workstream IS the corrective; ABBA dismissal was the workaround that
masked it).

## Reproduction

`/tmp/gen_simple_timing2.py` (preserved here for ease of re-run):

```python
import time
import sys
sys.path.insert(0, 'Tools')
from benchmark_phoenix import bench_gen_simple

# Simulate ABBA: enable cinderjit auto-compile if available
try:
    import _cinderx
    import cinderjit
    cinderjit.auto()
    print("JIT enabled (auto)")
except ImportError:
    print("No JIT")

# Warm up
for _ in range(3):
    bench_gen_simple(100_000)

# Run with ABBA-calibrated n_iter
N = 8_700_000
results = []
for _ in range(3):
    t0 = time.perf_counter_ns()
    bench_gen_simple(N)
    t1 = time.perf_counter_ns()
    results.append((t1 - t0) / 1_000_000)

results.sort()
print(f"gen_simple({N}) median={results[1]:.2f}ms ({results})")
```

Run as:
```bash
../cpython-vanilla/python /tmp/gen_simple_timing2.py    # vanilla
./python_bench /tmp/gen_simple_timing2.py               # Phoenix + JIT
```

For Phoenix interpreter only, drop the `cinderjit.auto()` call (or
delete the entire `try` block).
