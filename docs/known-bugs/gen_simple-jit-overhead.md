# gen_simple — JIT generator-creation overhead (Phoenix-stable known-cost)

**Status:** KNOWN-COST, fix-class workstream queued post-terminal-goal
pure-C completion (per Alex priority order). NOT a recent regression;
stable across 11+ days, 6+ sessions, multiple unrelated commit
surfaces. Documented per supervisor 2026-04-27T15:15:20Z disposition
following pythia #195 catch on prior "pre-existing structural causal
absence" framing.

## Symptom

`Tools/benchmark_phoenix.py jit --only=gen_simple` reports Phoenix
(JIT_ON, `cinderjit.auto`) at ~0.70-0.76x of vanilla CPython on the
4-bench ABBA harness — i.e. JIT is ~30-37% slower than vanilla on
this single benchmark, while other benchmarks in the same run land at
positive speedup (fibonacci ~2.27x, nqueens ~1.59x, func_calls
~1.00x).

## Workload shape

`bench_gen_simple(n_iter=8_700_000)` per-rep (`PER_BENCH_ITERS` in
`Tools/benchmark_phoenix.py`):

```python
def bench_gen_simple(n_iter):
    def gen(n):
        for i in range(n):
            yield i
    total = 0
    for _ in range(n_iter // 100):    # 87,000 outer iterations
        for v in gen(100):            # 87,000 generator creations
            total += v                # 8,700,000 inner yields
    return total
```

87,000 short-lived generators are created and exhausted per rep. The
inner `gen()` function runs ~100 yields each before the generator
object is discarded.

## Reproduction (testkeeper 2026-04-27T15:14:16Z, `/tmp/gen_simple_match_harness.py`)

3-warmup + 5-measure pattern matching `_worker_jit` in
`Tools/benchmark_phoenix.py`, `n_iter=8_700_000`, `mean_ms` of 5
measurements:

| Configuration | mean_ms |
|---|---|
| vanilla CPython `-S` | 476.1 |
| Phoenix `JIT_ENABLE=0 PYTHONJITAUTO=0 -S` | 491.2 |
| Phoenix `cinderjit.auto()` `-S` | 633.2 |

`cinderjit.auto()` is the cost driver. Phoenix-with-JIT-disabled is
within 3% of vanilla (harmless CPython-branch overhead). Phoenix-JIT
adds 142ms (29% on top of JIT_OFF) on this workload.

## Hypothesis (mechanism candidate, not yet confirmed)

The JIT-compiled outer loop pays per-generator-creation prologue +
epilogue cost on each of the 87,000 generators. Vanilla CPython's
interpreter has a more compact path for short-lived generator
dispatch. Specific candidates worth investigating when fix-class
workstream opens:

- JIT generator frame setup vs interpreter `_PyEvalFramePushAndInit`
- Reference-count / LOAD_DEREF cost in compiled generator frames
- `GEN_START` opcode emit weight relative to interpreter
- Inliner heuristic: does Phoenix inline `gen(100)` into the outer
  loop? If not, every `gen(100)` call is full prologue.

## Stability evidence (NOT a recent regression)

| Date | Benchmark log | gen_simple ratio |
|---|---|---|
| 2026-04-16 | `docs/benchmarks/abba_x86_64_20260416.md` | 0.73x |
| 2026-04-16 | `docs/benchmarks/abba_arm64_20260416.md` | 0.76x |
| 2026-04-22 | `docs/benchmarks/abba_push45_20260422_040152.md` | 0.72x |
| 2026-04-22 | `docs/benchmarks/abba_pre_push46_20260422_044949.md` | 0.72-0.74x |
| 2026-04-24 | `docs/benchmarks/abba_w27c2a_2026-04-24.md` | 0.74x |
| 2026-04-25 | `docs/benchmarks/abba_24bench_push59_2026-04-25.md` | 0.70x |
| 2026-04-27 | `docs/benchmarks/abba_blme_b2_2026-04-27.md` | 0.74x |
| 2026-04-27 | `docs/benchmarks/abba_kwnames_probe_2026-04-27.md` | 0.75x |
| 2026-04-27 | `docs/benchmarks/abba_preloader_probe_2026-04-27.md` | 0.73x |

Ratio range 0.70-0.76x over 11+ days, 9 logged runs, 6+ commit
surfaces. The variance is consistent with same-session ABBA noise
floor (~3-7% per `feedback_abba_cross_session.md`); no commit in this
window introduced or removed the cost.

## Anti-pattern guard

When per-commit ABBA reports `gen_simple` in this 0.70-0.76x range,
gate reports MUST cite this doc + use the framing
`Phoenix-stable known-cost, fix-class workstream queued
post-terminal-goal` rather than the prior dismissal language
`pre-existing, structural causal absence` (which sounded like
no-action accounting and was caught by pythia #195
2026-04-27T15:10:01Z + medic 15:15:15Z motivated-reasoning warning).

If `gen_simple` drops BELOW 0.70x or rises ABOVE 0.76x in a single
ABBA, that IS a new signal — investigate the proximate commit, do
NOT cite this doc.

## Resumption gate (when fix-class work opens)

1. Re-baseline `gen_simple` standalone (vanilla / Phoenix-JIT-off /
   Phoenix-JIT-on) to confirm the ~0.73x ratio still holds; if not,
   reframe.
2. Profile JIT-compiled `bench_gen_simple` under `perf record` or
   `cProfile` (note cProfile distorts vanilla baseline by ~3.8x; use
   `perf` for clean comparison).
3. Diff JIT-emitted code for `bench_gen_simple` outer loop vs
   interpreter dispatch — look for per-iteration generator-frame
   prologue weight.
4. Falsify each hypothesis-class candidate above; document which is
   load-bearing.
5. Heavy-tier instrumentation (analogous to W-PYTORCH-CM (ii)
   tp_alloc-watchpoint pattern) only after lighter probes converge or
   exhaust.

## Related

- `feedback_assume_phoenix_regression.md` — Alex 2026-04-24 directive:
  all bugs presumed Phoenix-introduced. Satisfied here (cost IS
  Phoenix's, just not recent).
- `feedback_falsifier_convergent_negative.md` — convergent-negative
  ABBA delta needs `investigated-no-quick-fix` framing, not
  `pre-existing` escape. This doc IS that framing.
- `feedback_no_workarounds.md` — documenting + deferring fix is NOT a
  workaround; only bail-out / deopt is the prohibited pattern.
  Deferred-fix-with-workstream-anchor is transparent accounting.
- pythia #195 (D-1777300601 2026-04-27T15:10:01Z) — caught the
  framing drift across 4 commits.
- medic 2026-04-27T15:15:15Z [MEDIC-WARNING] — caught generalist
  follow-up framing as motivated reasoning vs pythia.
