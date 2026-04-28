# WS3 gen_simple regression bisect — Phase 1 + Phase 2 result

**Date:** 2026-04-28
**Anchor evidence:** testkeeper 03:41:37Z (in-tree bd00b75500 measurement),
07:21:46Z (Phase 1 bisect), 07:58:35Z (Phase 2A harness sub-bisect),
07:59:52Z (Phase 2 ABBA at 314d0f2310), 08:08:16Z (method-B v2 calibration),
08:10:53Z (33bbbe6c36 probe), 08:36:18Z (Step 0 force_compile falsifier at
e557787b1f), 09:20:36Z (auto-compile re-dump, X1 refutation),
theologian 07:46:45Z + 08:02:51Z + 08:11:47Z + 08:34:37Z + 09:18:34Z
methodology dispositions.

## Summary

The gen_simple JIT regression (~25-33% slowdown vs vanilla CPython) is
present in both `force_compile` and `auto-compile` paths at HEAD, with
ratios consistently in the ~0.68-0.78x band across all measured points
in the dd70e1cf89..HEAD span.

Commit `33bbbe6c36` ("Enable auto-compilation via func watcher + counting
trampoline", 2026-03-30) is the first **auto-compile-observable** commit —
it enables the path that exposes the regression to ABBA/auto-compile
measurement, but does NOT introduce or activate the slow JIT codegen
itself. Direct `force_compile` measurement at `e557787b1f` (testkeeper
08:36:18Z) yields ratio 0.683x — same band as post-`33bbbe6c36`.

This corrects the initial framing of `33bbbe6c36` as "activation commit",
which rested on a structural inference that pre-activation ratio must be
~1.0x because auto-compile vectorcall hooks were off. Step 0 falsified
that inference with direct measurement.

**Force-compile vs auto-compile path divergence (testkeeper 09:20:36Z):**
the `force_compile`-path HIR shows generic `InPlaceOp<Add>` (no
specialization) for `bench_gen_simple`'s `total += v`. The `auto-compile`
path correctly emits `GuardType<PyLong>` + `LongInPlaceOp<Add>` because
adaptive interpreter specialization runs first and provides the
specialized opcode. The two paths produce structurally different HIR for
the same source — `force_compile`-path evidence cannot be generalized to
auto-compile production codegen. The "slow codegen pre-exists in
extraction-era cluster" claim, while still true for the `force_compile`
path at `e557787b1f`, does not directly indict auto-compile production
codegen. W-PHASE2-CODEGEN-SLOW continues investigation in the production
hot path (inner `gen()` generator, JIT-compiled per-CodeObject counter
with 696k inner calls per ABBA measurement; `bench_gen_simple` itself
stays in the interpreter at 8 outer calls < threshold 1000).

## Bisect range

- **Original framing:** `dd70e1cf89..first-ABBA-commit`
  (testkeeper 03:41:37Z; theologian 05:59:02Z TWO-PHASE BISECT spec)
- **Refined first-ABBA upper bound:** `cf49ad6da5` (where 0.75x was
  measured per `docs/benchmarks/x86_64_abba_2026-03-31.md`), NOT
  `2e15d1e70d` (which only added the doc, +6 commits later)

## Phase 1: stability bisect

- **Range:** `dd70e1cf89..2e15d1e70d` (45 commits)
- **Predicate:** `PYTHONJITAUTO=1 timeout 30 ./python -c 'import _cinderx; print("OK")'`
  exits 0 (theologian 06:58:47Z corrected from `cinderjit.auto()` after generalist
  06:58:03Z surfaced not-yet-landed-API conflation)
- **Driver:** `scripts/bisect_phase1_stability.sh`
- **Result:** `e11fc09f6e8c4a5a1de562efec3f0c79ba87e3ac` (2026-03-30 07:48:05,
  +37min after dd70e1cf89). Commit body: 'Phase 2B: JIT initializes
  successfully on import _cinderx'. Three fixes inside: PyModule_Create
  in place of `_Ci_CreateBuiltinModule`; skip auto-compile scheduling;
  drop `_cinderx-lib.cpp` + `async_lazy_value.cpp` from build.
- **Wallclock:** 5 iterations, 22 revisions, ~10min.
- **Side-finding:** intermediate commits `33bbbe6c`, `82501bcc`, `b972354e`
  exhibited shutdown-time SIGSEGV (init OK then crash); fixed before
  Phase 1 GOOD endpoint. Logged as W-SHUTDOWN-CRASH-INTERMEDIATE candidate
  (low-priority/historical, no current impact). Supervisor 07:23:17Z ACK'd
  but did not open as active workstream.

## W-DD70E1CF-INIT-SEGV closure

Resolved-via-Phase-1-discovery (supervisor 07:23:17Z disposition). The three
fixes inside `e11fc09f6e` are the standalone resolution of the dd70e1cf89
`jit::initialize()` segfault — root cause identified at Phase-2B authoring
time, fix landed sub-1-hour gap. Not pre-existing parked debt.

## Phase 2A: harness sub-bisect

- **Range:** `e11fc09f6e..cf49ad6da5` (38 commits)
- **Driver:** `scripts/bisect_phase2a_harness.sh`
- **Predicate:** `./python_bench Tools/benchmark_phoenix.py --help` exit 0
  within 30s (necessary AND sufficient for ABBA setup phase per theologian
  07:46:45Z)
- **Result:** `314d0f23109d14b85a6283d0f526c3b4245edf22` (2026-03-30 10:51:39,
  +3h03min after `e11fc09f6e`). Commit body: 'Guard dict watcher dealloc and
  restore uncompiled vectorcalls on shutdown'.
- **Wallclock:** 6 iterations, 38 revisions, ~10min.
- **Why needed:** `import _cinderx` succeeds at `e11fc09f6e` (Phase 1 GOOD)
  but the `Tools/benchmark_phoenix.py` harness segfaults during its setup
  phase on commits `e11fc09f6e..b972354eaa` — JIT-init-stable does not
  imply harness-compatible. Mechanism attribution by testkeeper at
  07:45:55Z + 07:53:45Z was retracted as forensic-log-first violation
  (medic 07:53:05Z + 07:55:31Z catches; testkeeper 07:57:04Z full
  concession). Path A predicate is empirical, mechanism-independent.

## Phase 2 ABBA at `314d0f2310`

- **Result:** ratio 0.738x (already-regressed). Phase 2 perf-bisect range
  `314d0f2310..cf49ad6da5` is structurally empty — every commit measures bad.
- **Implication:** regression entered in the harness-broken cluster
  `e11fc09f6e..314d0f2310` (22 commits), where ABBA cannot run.

## Phase 2 method-B probe (in-process timing)

Theologian 08:02:51Z authorized C-then-B path: hypothesis-driven probe of
`33bbbe6c36` ('Enable auto-compilation') with minimal in-process timing
(no harness).

- **Probe:** `scripts/probe_gen_simple_minimal.py`. Replicates
  `Tools/benchmark_phoenix.py:777` `bench_gen_simple` verbatim. v1 used
  1100 outer warmup calls of `bench_gen_simple(100)` which crossed
  `compile_after_n_calls=1000` and JIT-compiled the outer function
  (calibration divergence 21% vs ABBA, 542ms vs 641ms). v2 matches harness
  warmup pattern exactly: 3 warmup + 5 measure × `bench_gen_simple(8.7M)`.
- **v2 calibration at `314d0f2310`:** ratio 0.776x vs ABBA 0.738x (5.2%
  delta, within ±10% window).
- **`33bbbe6c36` probe:** ratio 0.713x. BAD by 0.85x threshold.

## Result

`33bbbe6c36` is the first **auto-compile-observable** commit, NOT the
introducing/activating commit. The slow JIT codegen for `gen_simple`
pre-exists in `e11fc09f6e..e557787b1f` (~11 commits, all pre-`33bbbe6c36`)
and is directly measurable via `force_compile` per testkeeper 08:36:18Z
(W-PHASE2-CODEGEN-SLOW Step 0).

| Commit | Description | Ratio (vanilla / jit) | Source |
|---|---|---|---|
| `e557787b1f` | Pre-`33bbbe6c36`, auto-compile off | 0.683x | force_compile probe |
| `33bbbe6c36` | Enable auto-compilation (first auto-observable) | 0.713x | method-B v2 |
| `314d0f2310` | First harness-stable post-`33bbbe6c36` | 0.776x / 0.738x | method-B v2 / ABBA |
| `cf49ad6da5` | First-ABBA upper bound | 0.75x | ABBA 2026-03-31 |

Magnitude is stable across the 11-commit pre-`33bbbe6c36` cluster + the
22-commit post-`33bbbe6c36` span, ~0.68-0.78x range. Falsifies any
'gradual progression' or 'late-introduction' hypothesis. The slow codegen
substrate is present at extraction time and does not vary across the
measured 33-commit span.

## Out-of-scope for bisect

The 'what slow-code-shape' question requires Debug-First profiling
(perf record + cinderx-disassemble on current-tree gen_simple JIT compile).
Theologian 08:11:47Z RECOMMEND: profile HEAD over bisecting the
pre-`33bbbe6c36` cluster — fix-relevant info comes from characterizing the
slow code-shape, not identifying which extraction-era infrastructure commit
introduced it.

W-PHASE2-CODEGEN-SLOW opened (supervisor 08:13:49Z). Step 0 falsifier
landed (testkeeper 08:36:18Z) — slow codegen confirmed pre-existing in
extraction-era cluster. Step 1+ profiling work in flight per theologian
08:34:37Z spec.

## Artifacts

- Phase 1 driver: `scripts/bisect_phase1_stability.sh`
- Phase 2A harness driver: `scripts/bisect_phase2a_harness.sh`
- Phase 2 method-B v2 probe: `scripts/probe_gen_simple_minimal.py`
- Bisect logs: `/tmp/phx-bisect-staging/{bisect_full.log,bisect_phase2a.log,
  abba_validate_314d0f.log,calibrate_v2_314d0f.log,probe_33bbbe6c.log}`
- Worktree: `/tmp/phx-bisect-phase1` (preserved at `33bbbe6c36`)
