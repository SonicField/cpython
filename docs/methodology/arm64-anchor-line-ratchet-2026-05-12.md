# ARM64 Anchor-Line Ratchet (FRESH-EYES v2 A3 codification)

**Authored:** 2026-05-12 by generalist (per supervisor 18:09:43Z dispatch, codification batch
during c7-close ABBA wait window). Discharges supervisor 17:02:10Z 2026-05-11 commitment
(D-1778518969) for v2 A3 codification + supervisor 02:54Z 2026-05-12 v2.x amendment moratorium
exception for in-tree codification (no new amendments to v2 spec; this is a methodology doc
documenting the agreed mechanism).

**Class:** post-c2 codification batch item 1.

---

## Problem

`feedback_no_perf_drift` (Alex 2026-04-29) requires gatekeeper to BLOCK on monotone
per-bench drift across 5+ batches OR >7% cumulative drift, per-batch-pass notwithstanding.
The rule is implemented as a check on `gatekeeper-fresh-eyes-checklist-v2-2026-05-11.md`
Step 4 (line 102): "ARM64 anchor-commit codification: TBD post-ratify per A3 resolution path."

The TBD has been outstanding since v2 adoption 2026-05-11. Today's c7-close ABBA (testkeeper
17:46Z dispatch, ~60-80min per arch-leg) is the first concrete drift probe across the
c4→c7 inline-conversion waiver chain (per pythia #360 + supervisor 17:26:03Z reframing).

Without an anchor, "drift since when?" has no canonical answer; ratchet thresholds become
unfalsifiable.

## Mechanism

**Anchor commit:** `1419c1261e` (Phase 5.B c2 — substrate-fallback boundary, last
substrate-class commit before the inline-conversion waiver chain c4→c7 began). Anchor is
re-named whenever a strict-equivalence ABBA establishes a new floor.

**Anchor ratio:** the geo-mean ratio (Phoenix-JIT vs vanilla CPython) measured at the
anchor commit, in the canonical 25-bench `--compile=auto --reps=3` ABBA. ARM64 + x86_64
each carry their own anchor ratio.

**Ratchet rule (per-batch, gatekeeper-applied):**

1. Every N=3-5 commits OR at end-of-batch boundary (whichever comes first), run a
   strict-equivalence ABBA: same-session, reps≥3 (5 for F3-strict per librarian 17:31:17Z),
   ±2pp band per leg, anchor (binary A) vs current tip (binary B).
2. Compute `delta_ratio = tip_geomean − anchor_geomean`.
3. PASS conditions:
   - `delta_ratio ≥ -0.02` (2pp band) AND
   - per-bench >5% drops are flake-falsified via N-run isolation (per testkeeper 14:12:33Z
     precedent on store_subscr/printer_coverage/nbody outliers).
4. BLOCK conditions (per `feedback_no_perf_drift`):
   - `delta_ratio < -0.07` (>7pp cumulative drop) — single-batch BLOCK regardless of trend.
   - Monotone drop across ≥5 consecutive batch ABBAs — BLOCK on trend even when per-batch
     passes the 7% threshold.

**Anchor advance protocol:**

- After a substrate-class push that lands a measured ABBA improvement, the team MAY
  re-anchor (move the anchor commit forward) to lock in the gain.
- Re-anchor authorization: supervisor + gatekeeper joint sign-off via chat post citing
  the ABBA log path + new anchor SHA.
- Anchor REGRESSION (moving anchor back to an earlier commit) is BLOCKED — once a floor
  is set, the team holds or improves it, never retreats it.

## Current state

| arch | anchor SHA | anchor geo-mean | last measured | status |
|---|---|---|---|---|
| x86_64 | 1419c1261e (c2) | 1.07x (testkeeper 13:45:49Z) | 2026-05-12 | active anchor |
| ARM64 | 1419c1261e ≡ d5fd126080 (c2 cherry-pick) | 1.09x (testkeeper 13:53:25Z) | 2026-05-12 | active anchor |

Note ARM64 arch operates at a different absolute ratio than x86_64 (1.09x vs 1.07x as of
c2 anchor). The supervisor 17:02:10Z 2026-05-11 concern about "ARM64 absolute 1.03x ratchet"
referenced an earlier anchor state; since c2 the ARM64 ratio has been at 1.09x. Current
ratchet target for ARM64 is "hold ≥1.07x" per checklist v2 line 87 — i.e., do not allow
the gain back from 1.03x → 1.07x→1.09x to erode.

## Application timeline

The c7-close ABBA dispatched at testkeeper 17:46Z (re-launched 17:45Z with corrected `jit`
subcommand per librarian 17:54:08Z) is the first concrete A3 application:

- Anchor: 1419c1261e (c2)
- Tip: 0a67941bef (c7)
- Methodology: F3 strict, reps=5, ±2pp band, sequential c2-vs-vanilla then c7-vs-vanilla,
  both x86_64 + ARM64
- Output: `docs/benchmarks/abba_x86_2026-05-12.md` + `docs/benchmarks/abba_arm64_2026-05-12.md`

Result determines:
- PASS (delta within band) → c8 unblocks; anchor MAY advance to c7 if gain measured.
- BLOCK (delta beyond band) → bisect inside c4→c7 to attribute, halt c8 dispatch until
  fix.

## Why a doc and not a v2 amendment

Per supervisor 02:54Z 2026-05-12 informal v2.x moratorium (D-1778554524). A3 was already
queued as a v2 amendment when the moratorium took effect; this doc captures the mechanism
that v2 already cited but left as TBD, without re-opening the amendment cadence. If/when
the moratorium lifts, the relevant text from this doc can be folded into checklist v2
Step 4 directly.

## Anchors

- supervisor 17:02:10Z 2026-05-11: ARM64 ratchet → v2 A3 commitment (D-1778518969)
- supervisor 17:15:59Z 2026-05-11: "ARM64 absolute 1.03x ratchet concern → v2 A3
  anchor-commit codification" (push 5.B c1 caveat)
- gatekeeper-fresh-eyes-checklist-v2-2026-05-11.md line 87, line 102: cites A3 resolution
  path
- pythia #360 + supervisor 17:26:03Z: per-batch ABBA at end-of-batch (3-5 commits) cadence
  for header-inline class
- pythia #361 + supervisor 18:04:09Z: queue item 9 hook (a2e8808ade) is the
  structural-enforcement complement to this performance-side ratchet
- librarian 17:31:17Z: F3 strict reps=5 ±2pp protocol for ABBA
- feedback_no_perf_drift (Alex 2026-04-29): 7% cumulative drift BLOCK threshold
- testkeeper 14:12:33Z: per-bench N-run isolation precedent for outlier flake-vs-real
- supervisor 02:54Z 2026-05-12: v2.x moratorium scope (D-1778554524)
