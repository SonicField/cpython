# Load-Only Failure Classification Rule

**Authored:** 2026-05-12 by generalist (per supervisor 18:09:43Z dispatch, codification
batch item 3 during c7-close ABBA wait window). Discharges queue item 8 from supervisor
14:21:53Z disposition of pythia #356 (3).

**Class:** post-c2 codification batch item 3.

---

## Problem

`test.test_multiprocessing_fork.test_threads` failed once during the c2 Tier 1 gate
(testkeeper 12:06:17Z) under suite contention (`-j8 --timeout=120`). N-run isolation
follow-up (testkeeper 12:26:56Z, 5 reps each on parent + c2) showed 0/5 fail on both
binaries in isolation, leading to reclassification as "environmental flake / suite-race"
(supervisor 12:27:34Z disposition).

Pythia #356 (3) substantive (theologian/supervisor 14:13:32Z window) flagged the
asymmetry: isolation reproduces neither the load conditions (`-j8`) nor the timeout
budget (`--timeout=120`) of the original failure. Isolation-passing is consistent with
both "environmental flake" and "introduced regression that surfaces only under
contention." The flake reclassification therefore inverts `feedback_assume_phoenix_regression`
(Alex 2026-04-24): "All bugs presumed Phoenix-introduced." Once a load-only failure is
classified as flake without a controlled load experiment, every subsequent recurrence
gets the same flake treatment, and a real load-class regression hides indefinitely.

Supervisor 14:21:53Z codified the corrective rule and queued it as queue item 8 for
post-c2 codification.

## Rule

**For any test that has previously failed under suite-load (i.e., during a CPython
parallel test run with `-j N >= 4`) AND was subsequently classified as "environmental
flake" via N-run isolation — if the same test re-fails under suite-load on a subsequent
c-series commit, the default classification is INTRODUCED, not flake.**

The flake classification is forfeited on first re-occurrence under load. To restore the
flake classification, a *load-faithful* falsifier is required:

- Run the test ≥5x under the SAME load conditions that produced the failure (e.g.,
  `-j8 --timeout=120` for the test_multiprocessing_fork class).
- Both parent commit and accused commit must be tested under those same conditions.
- If parent ≥1/5 fails AND accused ≥1/5 fails with similar rate AND the per-run failure
  fingerprint matches → flake-confirmed under load (re-classification permitted).
- If parent 0/5 + accused ≥1/5 → genuine introduced regression, halt push and
  root-cause.
- If parent ≥1/5 + accused 0/5 → curious; do not block, surface for analysis.

The N-run-in-isolation falsifier is permitted as a SUPPLEMENTARY signal but is no
longer sufficient on its own once the test has a load-failure history.

## Currently affected tests

| test | first load-fail | flake-classification origin | next load-fail re-classification rule |
|---|---|---|---|
| `test.test_multiprocessing_fork.test_threads` | 2026-05-12 12:06:17Z (c2 1419c1261e Tier 1, `-j8 --timeout=120`) | testkeeper 12:26:56Z N-run isolation (parent 0/5, c2 0/5) → supervisor 12:27:34Z reclassified as flake | next load-fail on c-series = INTRODUCED until load-faithful falsifier clears |

The CPython "PreExistingProven" set (~8 stdlib modules: `test_gdb.test_misc`,
`test_gdb.test_pretty_print`, `test_cmd_line`, `test_pdb`, `test_peg_generator`,
`test_posixpath`, `test_urllib`, `test_urllib2`) is NOT subject to this rule — those
fail consistently across c-series and predate the c-series window. They remain
non-blocking per the gate-script's stdlib-failure policy.

The ARM64-specific PreExistingProven set (per c3+c4+c5+c6+c7 ARM64 Tier 1 reports —
`test_importlib`, `test_pdb`, `test_posixpath`, `test_unittest`, `test_venv`,
`test_gdb.test_misc`, `test_gdb.test_pretty_print`) similarly predates c-series and is
excluded from this rule.

## Why default-introduced (not default-flake)

Per `feedback_assume_phoenix_regression`: "All bugs presumed Phoenix-introduced … Cinder
runs IG prod for years." The Phoenix C/C++ JIT touches code paths that any
multi-threaded / fork-based test could exercise indirectly. A load-class regression in
the JIT (lock contention, race in JIT compilation cache, signal-safety issue) would
plausibly surface under `-j N` test contention before isolation. Treating the surface
as flake throws away the signal that `-j N` provides.

The cost of "default-introduced" is one load-faithful falsifier run per re-occurrence
(~5min per binary × 2 binaries = ~10min). The cost of "default-flake" if the regression
is real is undetected production risk that compounds across c-series — the
asymmetry favors INTRODUCED.

## Anchors

- supervisor 12:27:34Z 2026-05-12: original flake reclassification (D-1778588856
  vicinity)
- pythia #356 (3) (~14:13Z): substantive flag on isolation-not-reproducing-load
- supervisor 14:21:53Z 2026-05-12: codification queue item 8 commitment
- supervisor 18:09:43Z 2026-05-12: this codification dispatch
- feedback_assume_phoenix_regression (Alex 2026-04-24)
- feedback_within_window_not_preexisting (medic 2026-05-05): related discipline on
  pre-existing-vs-introduced burden of proof
- testkeeper 12:14:53Z + 12:26:56Z: original c2 falsifier sequence (the precedent
  this rule corrects)
