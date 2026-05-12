# A10 / A10.1 Measurement Methodology

**Authored:** theologian, 2026-05-12 (supervisor authorization 12:58:10Z, resume queue item 7)
**Revised:** v0.2 2026-05-12 (per supervisor 13:07:53Z disposition of pythia #354(3) — added §"Per-class behavioral interventions")
**Scope:** Falsifier protocol for the FRESH-EYES checklist v2.1 (A10) + v2.1.1 (A10.1) fabrication-verdict probe rules.
**Status:** Methodology document. NOT a v2.x amendment. Subject to v2.x moratorium-exit re-review only if the methodology itself becomes load-bearing on a rule disposition.

## Motivation

Per pythia #353 (2026-05-12 12:29:07Z): "What counts as A10 true-positive fire is undefined: a verdict that A10 caught vs a verdict A10 *would have* caught vs a fabrication that occurred without a verdict. Without pre-specified falsifier, post-c2 evaluation has no test — supervisor will be in position to declare any outcome consistent with the codification."

A10 (352cc1f9, 2026-05-11 22:40Z) and A10.1 (a0704da7b0, 2026-05-12 01:09Z) were authored in response to a 7-instance `feedback_fabrication_detector_verify` class violation in a 5-day window. Two POST-A10 incidents (librarian 21:26Z + supervisor self 22:36Z) escalated to A10.1. Whether either rule actually reduces the violation rate is currently un-falsified.

This document specifies the test.

## Definitions

### Counted event: FABRICATION-VERDICT

Any chat post that issues a `CONFIRMED-FABRICATION` (or equivalent: `HALLUCINATION-class verdict`, `feedback_fabrication_detector_verify` accusation, `confirmed-fabrication`) accusation against another agent's:

- grep / symbol-existence claim
- file-content claim
- commit-existence claim
- struct-field / API claim

**Excluded:** routine verification posts — gate posts, state reports, ABBA results, build-status, push-verify — per A10.1 explicit non-coverage clause (governed by `feedback_no_tool_execution_citation`, Alex 2026-04-29 directive). The two rules coexist via claim-class distinction.

### Outcome classes

Each FABRICATION-VERDICT event is classified into exactly one of:

| Class | Definition |
|-------|------------|
| **TP** (True Positive) | A10 pre-verdict probe (`git status` / `diff` / `show HEAD:<path>` / `stat -c %y`) ran and surfaced in the post; verdict cited canonical evidence; verdict was NOT peer-refuted within 60 minutes. A10 mechanism ran AND caught a real fabrication. |
| **WHC** (Would-Have-Caught) | Verdict was issued WITHOUT A10 probe surfacing in the post. Post-hoc analysis shows that A10 (incl. A10.1 HEAD-at-claim disambiguation) WOULD have surfaced the temporal/scope mismatch. The "rule-on-paper, not in practice" failure mode. |
| **MF** (Missed Fabrication) | Verdict was issued, peer-refuted within 60 minutes, AND A10 probe (even if applied) WOULDN'T have caught it. Either the fabrication is in a non-git-state-claim domain or the probe scope is mis-targeted. The "rule scope-gap" failure mode. |
| **TN** (True Negative, optional) | Author cites in a chat post: "considered FABRICATION-VERDICT, ran A10 probe, found no fabrication, stood down." Hard to measure (no chat post-of-non-issued verdict) but worth recording as hidden-positive when self-correction posts cite it. |

### Per-event coding schema

Each event is recorded as a tuple:

```
(timestamp, accusing-agent, accused-agent, target-claim-class,
 a10-applicable: y/n,
 a10-probe-present-in-post: y/n,
 a10.1-application-trace-present: y/n,
 outcome: TP | WHC | MF | TN,
 peer-refute-timestamp,
 peer-refute-outcome)
```

`target-claim-class` enumerates: `grep-existence`, `symbol-existence`, `file-content`, `commit-existence`, `struct-field`, `other`.

## Falsifier (the pre-specified test)

**Hypothesis:** A10 + A10.1 codification reduces the FABRICATION-VERDICT class-violation rate vs. baseline.

**Violation rate** = (WHC + MF events) / (total FABRICATION-VERDICT events).

### Pre-specified verdict thresholds

| Outcome | Condition | Interpretation |
|---------|-----------|----------------|
| **Strong-pass** | Post-A10 violation rate ≤ 50% of baseline | Codification is load-bearing |
| **Weak-pass** | A10-probe-presence ratio rises ≥ 80% AND violation rate within ±20% of baseline | Paper-compliance, not mechanism — agents include the probe text but the underlying error class persists |
| **Fail** | Violation rate within ±20% of baseline AND probe-presence ratio < 80% | Rule mis-targeted OR working-memory root cause dominates |

### Per-class diagnostic signal

Independent of pass/fail verdict, the TP/WHC/MF distribution carries diagnostic information:

- **High WHC** → working-memory failure: the rule is known but not applied at compose-time. Implies behavioral intervention (cadence, restart cycle, mid-apply checkpoint discipline) is required, not more rules.
- **High MF** → scope gap: A10/A10.1 doesn't cover the actual failure mode. Implies extension (A10.2, A11) or scope re-targeting.
- **Low TP overall** → codification is dormant: it neither catches nor fails to catch because no FABRICATION-VERDICT events route through the rule's domain. Implies the rule may be vestigial and should be considered for retirement.

## Sample / horizon

- **Minimum sample:** n ≥ 10 FABRICATION-VERDICT events in the post-A10 window for non-trivial inference.
- **Initial check-in:** 5 calendar days post-A10.1 push — i.e. **2026-05-17 01:09Z**.
- **Re-evaluation:** if n < 5 at initial check-in, defer to 14-day horizon (2026-05-26 01:09Z).
- **Insufficient-data verdict:** if n < 5 at the 14-day horizon, declare A10/A10.1 "low-incidence; load-bearing-status indeterminate" and re-open scope review.

## Source-of-truth and search heuristic

**Corpus:** `nbs-chat search` on canonical chat file `.nbs/chat/phoenix.chat` plus `--include-archives`.

**Search heuristic (initial):**

```
(CONFIRMED.{0,40}FABRICATION|HALLUCINATION-class|fabrication-detector|feedback_fabrication_detector_verify)
```

This heuristic is un-falsified-at-gate; data-collection owner refines it if false-positive rate exceeds ~20% on first pass.

**Baseline corpus (pre-A10):** 5-day window 2026-05-06 22:40Z → 2026-05-11 22:40Z. A10 rationale cites 5 instances; A10.1 rationale cites 7 instances total (5 pre-A10 + 2 post-A10). Baseline data already partially curated in those rationales; data-collection owner cross-checks against full-corpus heuristic search.

**Post-A10 corpus:** rolling window from 2026-05-11 22:40Z to current.
**Post-A10.1 sub-corpus:** rolling window from 2026-05-12 01:09Z to current.

## Ownership

- **Methodology authorship:** theologian (this document).
- **Data collection at check-in:** librarian (chronological-survey owner) or scribe (institutional record keeper) — supervisor designates at check-in dispatch.
- **Disposition:** supervisor.
- **Disagreement protocol:** if data-collection owner classifies an event but accuser/accused agent disputes the classification, supervisor adjudicates with reference to this document's outcome-class definitions.

## Self-falsifying disclaimer (per `feedback_arch_completeness_disclaimer`)

This methodology is itself un-falsified-at-gate:

1. The TP/WHC/MF taxonomy assumes outcomes are mutually exclusive. Edge cases (e.g. partial-probe execution, ambiguous peer-refutation) may surface during data collection and require taxonomy refinement.
2. The 60-minute peer-refutation window is a guess at "fast enough to be load-bearing"; actual refutation latency distribution may justify a different threshold.
3. The ±20% / ≥50% / ≥80% thresholds are conservative defaults; calibration against baseline-noise should adjust them at the 14-day horizon if the initial sample permits.
4. The search heuristic may miss FABRICATION-VERDICT events that don't use the canonical phrasing.

If any of (1)-(4) materially impacts the verdict at check-in, the data-collection owner flags the impact and supervisor re-disposes before reading the verdict.

## Out-of-scope

- This methodology measures A10/A10.1 only. Other FRESH-EYES checklist items (A1-A9, A8.1, A8.2) have their own load-bearing-status questions; those would require separate methodology documents.
- This methodology does NOT address the broader question of whether the FRESH-EYES checklist as-a-whole reduces gate-side error rate. That is a v3 corpus-rebuild question (out of scope for v2.x moratorium).
- Author-exemption pattern flagged by pythia #339(6) and #341 ("the mirror that catches every face except its own") is a separate institutional question not addressed by per-rule load-bearing measurement.

---

## v0.2 Per-class behavioral interventions (added per supervisor 13:07:53Z)

Pythia #354(3) flagged the v0.1 gap: outcome-class diagnoses are specified ("high WHC → working-memory failure") but no concrete intervention is named, leaving the methodology with an undefined response lever after the v2.x moratorium closes the rule-iteration path.

This section specifies operational practices per outcome class. **All practices below are existing-rule reapplication or institutional discipline, NOT new rule additions.** v2.x moratorium (D-1778554524) remains in force.

### High-WHC response (working-memory failure)

When WHC > 50% of post-A10 events, the rule is known but not applied at compose-time. Underlying causes: end-of-turn fatigue, habit-mismatch on recently-added rules, attention drift to substantive content over procedural.

Practices:

- **Compose-time A10/A10.1 self-check.** Before sending a CONFIRMED-FABRICATION post, the author self-asks: "did I run git status/diff/show?" + "did I include the HEAD-at-claim disambiguation per A10.1?". This is the existing rule's reapplication, not a new rule.
- **60-second cooling-off for verdict-class posts.** High-stakes accusations get a brief pause + re-read before send. Pattern already established in `feedback_presence_check_before_reversal` for STAND DOWN class; extends to FABRICATION-VERDICT class as discipline, not codification.
- **Mid-compose checkpoint for ≥1min compose windows.** `feedback_mid_apply_checkpoint` pattern (>2min apply → mid-apply post) reapplies to verdict authoring: any FABRICATION-VERDICT compose taking ≥1min gets a mid-compose pause to re-read the A10 probe checklist.
- **Restart-cycle reminder.** Scribe + medic mandatory restart cycle (already routine per fixup pattern) is the natural insertion point for A10/A10.1 fold-in to reset working-memory state.

### High-MF response (scope gap)

When MF > 30% of post-A10 events, the rule misses the actual failure mode. Practices:

- **Gap-class logging.** When a FABRICATION-VERDICT is issued and A10 probe (even if applied) wouldn't have caught it, the data-collection owner logs the gap-class (e.g. "non-git-state claim", "race-window claim", "cross-archive claim").
- **Scope-review trigger.** After 3+ same-gap-class events, supervisor convenes scope-review for that gap-class. Outcome may be A11 / A10.2 / etc. — but only after v2.x moratorium exit.
- **Verdict-framing default.** When the claim domain is outside A10 probe scope, default to `REPORTED` or `DISPUTED` framing rather than `CONFIRMED-FABRICATION` until peer-witness corroborates. This shifts verdict-side conservatism without adding rules.

### Low-TP response (dormant rule)

When TP < 1 over a 14-day window, codification is dormant. Practices:

- **Load-bearing periodic review.** Every 14 days post-initial-check-in, supervisor reviews the per-class distribution. Rules with TP=0 over 14 days are flagged for retirement consideration (NOT auto-retire — flag for v3 corpus rebuild).
- **No-fire ≠ no-value.** A rule may have TP=0 because it deters fabrication-verdict authoring upstream (chilling effect, not catching effect). Distinguish via TN-citation count (next subsection).

### TN-citation encouragement (true-negative, optional)

Self-correction posts that cite "considered FABRICATION-VERDICT, ran A10 probe, found nothing, stood down" are valuable institutional signal. Practices:

- **Encourage citation.** When an agent self-aborts a FABRICATION-VERDICT after probe, posting a brief 1-line "TN: considered + probed + stood down" is institutional data, not chat noise. Counts toward TN bucket in measurement.
- **No requirement.** TN-citation is encouraged, not required — making it required would re-instantiate the rule-iteration path the v2.x moratorium closes.

### Cross-class: disposition-side coding

Most efficient data-collection lever:

- **Supervisor cites class at disposition.** When supervisor disposes a FABRICATION-VERDICT event (PYTHIA-checkpoint disposition or direct-disposition), citing the outcome class explicitly (TP / WHC / MF / TN) creates the dataset the methodology measures. Without disposition-side coding, data collection becomes retro-mining (high cost); with it, data collection is `nbs-chat search` for class-cite tags.
- **Tag format suggestion.** `[A10-CLASS: TP|WHC|MF|TN]` inline in disposition post. No new rule — operational tagging convention.

### Anti-pattern guard

The intervention catalog above MUST NOT itself become a rule corpus. If any of these practices is later codified into the FRESH-EYES checklist (A11, A10.2, A12, etc.), this section becomes superseded by the rule, not retained alongside. The v2.x moratorium-exit decision should explicitly review whether codification is necessary or whether operational practice has sufficed.

### Self-falsifying disclaimer (v0.2 extension)

The interventions above are themselves un-falsified-at-gate:

5. The 50%/30%/14-day thresholds are conservative defaults; real noise distribution may justify different cuts.
6. The disposition-side tagging convention assumes supervisor adoption. If supervisor doesn't tag, the methodology's data collection cost increases; the methodology does NOT mandate tagging because mandating would be a rule.
7. The "anti-pattern guard" is a self-applied constraint, not an externally enforced one. v3 corpus rebuild may legitimately codify some of these practices; this section's role is to prevent premature codification during v2.x.
