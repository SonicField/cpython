## W-PERF-PRE-W27C-BISECT — Scope Doc

**Author:** generalist (initial pre-stage 2026-05-01); theologian (revisions 2026-05-05 post-5.A2-Cfinal-push)
**Owner:** theologian (assigned 2026-05-01 16:35:17Z by supervisor; project_w_perf_owner.md)
**Date:** 2026-05-01 (substantive revision 2026-05-05 post-Cfinal-push 0d7d31c450)
**Class:** post-Phase-3D performance optimization workstream (Alex directive 2026-04-14: "Performance optimization comes after Phase 3D is complete")
**Trigger:** 5.A2 close cap-check ABBA (testkeeper 12:32:36Z) + 91f1702c8a same-session re-baseline (14:30:25Z) surfaced 3-4 carrier classes that satisfy investigated-no-quick-fix per feedback_gatekeeper_block_preexisting_dismissal but require perf-counter instrumentation to localize.
**P10 falsifier inheritance (D-1777161632, librarian 16:51:23Z):** closures-class carriers (try_except_callee, nn_module_forward) remain PROVISIONAL until ALL team-specified P10 in-context full-24-bench falsifiers run. Per-cycle 24-bench ABBA produces evidence data points but does NOT satisfy P10 spec.

---

## §1 — Carrier classes (post-Cfinal 0d7d31c450 push, 2026-05-05)

ARM64 + x86 dual-arch ABBA at Cfinal 0d7d31c450 (testkeeper 11:36:57Z x86, 13:01:13Z ARM64) revised carrier set:

| Bench | x86 Δ | ARM64 Δ | Class | Status |
|-------|-------|---------|-------|--------|
| gen_simple | -12.2% | -14.0% | Dual-arch consistent — strongest carrier signal | CONFIRMED carrier |
| nn_module_forward | -13.6% | -5.8% | Arch-asymmetric (ARM64 less severe); long-stable institutional per D-1775716079 / D-1777161632 | PROVISIONAL — pending P10 |
| try_except_callee | -11.4% | -4.3% | Arch-asymmetric (ARM64 within noise); long-stable institutional | PROVISIONAL — pending P10 |
| nbody | -9.6% | +3.8% | Arch-asymmetric (POSITIVE on ARM64) — strong x86-codegen-specific signal | PROVISIONAL — same-session falsifier required (§3a) |
| list_comp | -0.2% | +5.3% | Recovered to within noise (was -6pp in 5.A2 mid-chain); positive both arches | DROPPED from scope |

**Carrier-set summary (post-revision):**
- 1 CONFIRMED dual-arch carrier (gen_simple).
- 3 PROVISIONAL arch-asymmetric carriers (nn_module_forward, try_except_callee, nbody) — x86-only or x86-severe pattern.
- 1 DROPPED (list_comp recovered).

**Arch-asymmetry as evidence-class:** the consistent x86-severe / ARM64-mild pattern across nn_module_forward + try_except_callee + nbody is structural evidence for x86-codegen-specific perturbation (inlining-heuristic-boundary / register-allocation / i-cache-layout class per feedback_bit_identical_not_innocence_proof.md). Same mechanism class as the F3 bit-identical-codegen-perturbation lesson. gen_simple's dual-arch consistency suggests a different (algorithmic / data-layout) class, not codegen-perturbation.

---

## §2 — Why post-Phase-3D, not now

1. **Alex directive 2026-04-14:** performance optimization comes after Phase 3D is complete. Deferral is policy-driven, not technical.
2. **Phase 5 in flight:** 5.A2 close + 5.A3 (function.cpp 220 LOC STRUCT-tier) + 5.B/5.C/5.D/5.E remaining. Mid-flight perf instrumentation would compete with substrate-conversion attention.
3. **Instrumentation cost:** per-class perf-counter setup is ~2-4hr per class (vtune/perf annotate / linux perf record + flame-graph + per-emit-method timing). Heterogeneous carriers may not share a single instrumentation pass.
4. **Net win preserved:** geomean 1.07x parity holds across ~80 commits per cap-check; mixed-direction offsets (positional_dispatch +12.7%, int_arith +5.9%, func_calls +3.9%) absorb the per-bench drops at aggregate. Production-impact gating not blocked.

---

## §3 — Entry criteria (when this workstream activates)

Tracking item, not commit-now. Activates when:
- Phase 3D arc closes (Phase 5 + Phase 6 complete; ZERO C++ in Phoenix-target)
- AND at least one of: (a) Alex directive activating perf-arc, (b) any per-bench monotone trend exceeds 4-batch + cumulative 7%+ drop per feedback_no_perf_drift, (c) geomean falls below 1.05x.

Until then: per-cycle ABBA records the per-bench data; cap-check fires per 20-commit cadence; this scope doc collects carrier evidence as new classes surface.

---

## §3a — Same-session falsifier spec (cross-session methodology gap remediation)

**Background (theologian 2026-05-05 12:45:38Z, gatekeeper 12:45:53Z endorse):** the F3b verdict that cleared 5.A2 from nbody-carrier status was a cross-session comparison (F3 ABBA 2026-05-05 vs 209d307cd6 ABBAs 2026-05-01). Per feedback_abba_cross_session, cross-session variance is ±7pp; the C3 -15.1% vs Cfinal -9.6% delta of 5.5pp lies INSIDE that band. The data is consistent with EITHER C3 contamination OR a real cross-session shift. F3b uniquely supports neither. Additionally, Cfinal IS the last commit of the suspect 5.A2 chain — it sits inside the suspect window per feedback_within_window_not_preexisting (codified D-1777983027); within-window stability is not pre-existing proof.

**Net:** 5.A2-not-carrier status is PROVISIONAL not FALSIFIED for nbody. Same-session falsifier required to convert to FALSIFIED.

**Falsifier spec (single same-session window):**
1. Build 91f1702c8a (canonical baseline, x86_64).
2. Run 24-bench --reps=5 ABBA at 91f1702c8a vs canonical vanilla (md5 fcb1dddcbf5d1edbf54c478e705deccc).
3. Build 209d307cd6 (X3b root-cause-fix, post-instrumentation-revert).
4. Run 24-bench --reps=5 ABBA at 209d307cd6.
5. Build 0d7d31c450 (Cfinal).
6. Run 24-bench --reps=5 ABBA at Cfinal.
7. All three ABBAs in single uninterrupted window on same host (no concurrent benches).
8. Cost estimate: ~2hr (3x build ~10min + 3x ABBA ~30min + transitions).

**Decision rule:**
- Cfinal nbody Δ within ±2pp of 209d307cd6 nbody Δ (same-session): contamination at C3 confirmed evidence-class. Reclassify 5.A2-not-carrier as FALSIFIED. Drop nbody.
- Cfinal nbody Δ differs from 209d307cd6 nbody Δ by >5pp (same-session): real 5.A2 carrier confirmed. Reclassify nbody as CONFIRMED carrier. Bisect within 5.A2 chain (C0..Cfinal) for sub-attribution.
- 2-5pp delta: ambiguous (within ±5pp same-session noise floor); add 4th repeat ABBA at 209d307cd6 + Cfinal alternation for variance estimate.

**Activation:** when W-PERF batch enters per §3 entry criteria, OR when supervisor authorizes same-session falsifier as standalone budget pre-Phase-3D-close (cost ~2hr; non-blocking for current commit cadence).

---

## §3b — Class-rule for upcoming M3-class waves (5.A3 / 5.D / 5.E)

**Risk:** 5.A2's same-session-falsifier-gap will recur at 3-5x surface in 5.A3 (function.cpp), 5.D (generator.cpp), 5.E (bridge-delete). Each touches >20 caller sites with the same inline-relocation / register-allocation / i-cache-layout perturbation class (per feedback_bit_identical_not_innocence_proof.md). Without scope-doc-level methodology fix, each wave will (a) defer carriers to W-PERF on cross-session match grounds, and (b) cumulative perf debt will distribute invisibly until the eventual Phase-3D-close geomean ABBA against 91f1702c8a baseline surfaces it all at once with no per-batch attribution chain.

**Class-rule (theologian 2026-05-05 post-Cfinal-push):** for any M3-class wave (>20 caller-site inline-relocation), the gate falsifier-set MUST include either:
- (a) **Same-session F3-strict at wave-close:** rebuild canonical baseline (91f1702c8a) + first-wave-commit + last-wave-commit in single ABBA window. Cost ~2hr. OR
- (b) **Investigated-no-quick-fix with explicit cross-session-flag:** push under PROVISIONAL classification with mandatory same-session falsifier deferred to W-PERF batch with named owner + budgeted timeline (NOT open-ended).

Path (b) is acceptable for cadence preservation but accumulates W-PERF debt; path (a) is more expensive per-wave but resolves carrier status at gate-time. Wave-class chooses per supervisor disposition; this rule prevents silent absorption.

**Falsifies the F3b precedent for M3-class waves only:** F3b (single re-run match) remains valid for non-M3-class commits (sentinel changes, single-method conversions, doc-only).

---

## §4 — Per-class instrumentation hint (for future reference)

| Class | Instrumentation hint | Expected localization granularity |
|-------|----------------------|-----------------------------------|
| gen_simple (dual-arch consistent) | `perf record -F 5000 ./python_bench gen_simple.py` + flame-graph; cross-arch flame-graph diff to isolate algorithmic vs codegen-perturbation class; compare cinderx_dev baseline | per-call-site or per-data-structure (consistent-arch suggests algorithmic, not codegen-layout) |
| nn_module_forward (long-stable, arch-asymmetric) | torch-specific bench profiling (already-known hot path); per-arch flame-graph diff to confirm x86-codegen-specific class; compare deopt-rate vs cinderx_dev | per-call-site within nn_module hot loop; x86-specific i-cache / regalloc perturbation suspected |
| try_except_callee (long-stable, arch-asymmetric) | exception-handling path profiling; per-arch flame-graph diff; compare deopt-rate vs pre-Phase-3D; check exception-handler regalloc spill differential | per-exception-handler emission site; x86-specific perturbation suspected |
| nbody (PROVISIONAL, x86-only, arch-asymmetric) | Same-session F3-strict per §3a is the falsifier; if confirmed real-carrier, bisect within 5.A2 chain (C0..Cfinal, 6-commit window) for sub-attribution; per-arch flame-graph diff to isolate inline-relocation impact | per-relocated-method or per-i-cache-line; ARM64 +3.8% suggests dependency on x86 inlining-heuristic boundary specifically |

---

## §5 — References

**Original scope (2026-05-01):**
- supervisor 13:27:40Z — orphan benches defer + W-PERF-PRE-W27C-BISECT batch
- supervisor 15:36:29Z + gatekeeper 15:36:45Z — list_comp investigated-no-quick-fix close
- theologian 15:36:13Z — list_comp structural equivalence inspection
- testkeeper 14:30:25Z — 91f1702c8a same-session re-baseline ABBA
- testkeeper 12:32:36Z + 13:13:58Z — post-C3 cap-check ABBA (geomean 1.07x parity)
- librarian D-1775716079 + D-1777161632 — nn_module_forward / try_except_callee 15-day institutional history
- Alex directive 2026-04-14 — performance optimization comes after Phase 3D

**Post-Cfinal-push revision (2026-05-05):**
- supervisor 16:35:17Z (2026-05-01) — W-PERF owner = theologian
- librarian 16:51:23Z (2026-05-01) D-1777161632 — P10 in-context full-24-bench falsifier inheritance
- testkeeper 11:36:57Z (2026-05-05) — F3 24-bench ABBA at Cfinal x86_64 (geomean 1.08x, nbody -9.6%, list_comp recovery)
- testkeeper 13:01:13Z (2026-05-05) — ARM64 24-bench ABBA at Cfinal (geomean 1.12x, arch-asymmetry data: nbody +3.8%, gen_simple -14.0%)
- theologian 12:45:38Z (2026-05-05) — cross-session methodology gap walk-back, 5.A2-not-carrier reclassified PROVISIONAL
- gatekeeper 12:45:53Z (2026-05-05) — endorse W-PERF post-push provisional reclassification
- D-1777985197 (superseded by D-1777985267 attribution-correction) — scribe log of W-PERF scope walk-back
- feedback_abba_cross_session.md — ±7pp cross-session variance band
- feedback_within_window_not_preexisting.md — within-window stability is not pre-existing proof when window starts at suspected introducer
- feedback_bit_identical_not_innocence_proof.md — F3-strict required for M3-class >20-caller-site relocation

---

Scope doc revised post-Cfinal-push (2026-05-05). Tracking-only; no commit/build/push expected from this doc until §3 entry criteria met. Same-session falsifier per §3a is the next deterministic action when budget allows; class-rule §3b governs upcoming 5.A3 / 5.D / 5.E methodology.
