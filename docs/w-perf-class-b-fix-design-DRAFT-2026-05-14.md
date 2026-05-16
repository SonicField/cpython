# W-PERF Class B Fix-Design DRAFT — 2026-05-14

**Status:** DRAFT (theologian, late-session 2026-05-14). Awaiting supervisor review + generalist code-class implementation next session.
**Anchors:**
- Pre-c-series: b252ca0425 (testkeeper 21:00:11Z 24-bench dual-arch ABBA)
- HEAD post Phase A: 5830b49f19 (gate scaffolding LANDED)
- Class A subset: ef5e125b2b LANDED (1.32x x86 / 1.34x ARM64)
- c2 fixed anchor: 1419c1261e per [c2_fixed_abba_anchor]

## 1. Class B carrier set (per supervisor 21:01:18Z)

| Carrier | pre-c-series x86 / ARM64 | HEAD x86 / ARM64 | c-series amp x86 / ARM64 | Pre-existing leverage x86 |
|---|---|---|---|---|
| kwargs_dispatch | -4.6 / +4.1 | -8.0 / +4.1 | -3.4 / 0.0 | 4.6pp |
| nn_module_forward | -8.6 / -13.7 | -13.2 / -15.5 | -4.6 / -1.8 | 8.6pp |
| gen_simple | -11.7 / -19.3 | -13.3 / -22.4 | -1.6 / -3.1 | 11.7pp |

Class B characterization: x86-codegen-perturbation, distributed (no single-symbol >5% in profiler). PRIMARY fix-target = Phoenix-pre-c-series codegen layout. SECONDARY = c-series x86-amp bisect (lower leverage).

Total addressable: ~25pp x86 if all 3 pre-existing fixed; ~7-12pp from c-series amp bisect.

## 2. Hypothesis decomposition (Phoenix-codegen-layout perturbation classes)

Per [class_of_bug_audit] each hypothesis is a falsifiable class with a named test.

**H1 — Function-ordering / hot-cold split.** Phoenix codegen emits JIT'd functions in different memory order than baseline (CinderX/CPython), perturbing i-cache + branch predictor. Falsifier: `nm -nS` sorted-symbol diff between Phoenix-baseline and Phoenix-c-series shows ordering changes correlated with carrier degradation. Negative if symbol order is bit-identical.

**H2 — Inline placement.** Phoenix may not inline (or differently inlines) hot helpers — e.g., dict_get fast-path for kwargs_dispatch, frame_push for gen_simple, _Py_TYPE checks for nn_module_forward. Falsifier: perf hot-symbol list shows uninlined helper appearing in carrier-degraded run but not baseline (or vice versa). Negative if inline-set is identical.

**H3 — Branch direction / static prediction.** Phoenix codegen emits branches with reversed taken/fallthrough direction vs baseline, causing branch-pred regressions on hot paths. Falsifier: `perf stat -e branch-misses` shows >X% increase in carrier hot path; `objdump -d` of JIT'd region shows opposite branch direction. Negative if branch-mispred rate is within noise.

**H4 — Register allocation pressure.** Phoenix's regalloc spills/fills in hot loops differently. Falsifier: `perf stat -e l1d_loads,l1d_load_misses,dtlb_loads` shows increased spill traffic in carrier hot path. Negative if memory traffic is within noise.

**H5 — Memory layout / structure-access pattern.** Different field offsets, indirect-call patterns, or vtable layout cause load-class regressions. Falsifier: `perf stat -e l1i_load_misses,iTLB_load_misses` correlated with structure-access perturbation; HIR/LIR diff shows different load patterns for hot fields.

## 3. Per-carrier hypothesis mapping (best-fit by carrier characteristics)

### kwargs_dispatch (-4.6pp pre-existing, +3.4pp c-series x86 amp)
- **Hot path:** kwargs dict lookup + frame_push for callee. Distributed (dict_get + frame setup).
- **H2 candidate:** Phoenix may not inline dict_get fast-path (Cix_PyDict_LoadGlobal-class). Test: perf hot-symbol on kwargs_dispatch.
- **H3 candidate:** kwargs presence check (kw==NULL fast vs slow). Test: branch direction on kw-NULL check site.
- **Lower-likelihood:** H4 (kwargs is short-lived, low regalloc pressure).

### nn_module_forward (-8.6pp pre-existing, +4.6pp c-series x86 amp)
- **Hot path:** PyTorch nn.Module __call__ — frequent object dispatch, vtable-heavy, _Py_TYPE + tp_call + frame_push.
- **H4 candidate:** regalloc spill in dispatch loop (high register pressure from method resolution).
- **H5 candidate:** vtable layout / indirect-call cache misses (PyTorch nn.Module subclass dispatch).
- **H2 candidate:** _Py_TYPE check inlining vs C-call.

### gen_simple (-11.7pp pre-existing, +1.6pp c-series x86 amp)
- **Hot path:** generator iter / send / yield state machine. frame_save/restore on yield.
- **H1 candidate:** generator function emission order vs caller (i-cache locality between caller and generator body).
- **H2 candidate:** generator yield/send fast-path (inlined vs out-of-line).
- **H5 candidate:** _PyInterpreterFrame layout perturbation in generator save/restore.

Per [class_of_bug_audit]: each carrier may have a DIFFERENT root cause. No single fix expected to address all 3.

## 4. Investigation tooling (next-session generalist starting kit)

### 4.1 perf-counter spec per hypothesis
| Hypothesis | perf event spec |
|---|---|
| H1 (i-cache) | `perf stat -e L1-icache-load-misses,iTLB-load-misses` |
| H2 (inline) | `perf record -g` + `perf report --stdio` (hot-symbol list) |
| H3 (branch-pred) | `perf stat -e branch-misses,branch-loads` |
| H4 (regalloc) | `perf stat -e l1d_loads,l1d_load_misses,dtlb_loads` |
| H5 (memory layout) | `perf stat -e L1-icache-load-misses,iTLB-load-misses,dTLB-load-misses` |
| H-outside-hot-fn (data-cache meta-class, added 2026-05-16) | `perf record -g --call-graph fp -e L1-dcache-load-misses,L1-dcache-stores,LLC-load-misses,dTLB-load-misses` + caller-frame sample-filter |

#### 4.1.1 H-outside-hot-fn meta-class (amendment 2026-05-16)

Introduced post-original-scope (supervisor 12:16:39Z PRIMARY swap, codified at sup 16:30:46Z disposition (a)) as a meta-class identifying "issue is OUTSIDE the named hot function" — applicable across data-cache substrate classes when caller-frame attribution (path (b)) outranks in-function cost attribution (path (a)).

Event-set composition vs original §4.1:
- `L1-dcache-load-misses` ↔ H4 `l1d_load_misses` (syntax variant, same hw counter)
- `dTLB-load-misses` ↔ H5 `dTLB-load-misses` (verbatim match)
- `L1-dcache-stores` — NEW (write-traffic visibility; data-cache-class extension not in original §4.1)
- `LLC-load-misses` — NEW (memory-tier visibility; data-cache-class extension not in original §4.1)

NEW events are coherent data-cache-class extensions under the meta-class, not a fresh hypothesis. They were authorized per librarian 15:19:43Z PARTIAL MATCH evidence + supervisor 15:20:19Z disposition (a) (symmetric-rule, NOT META-RULE re-open).

Falsifier shape: hot-symbol diff PLUS caller-frame attribution PLUS data-cache event correlation. Distinct from H1-H5 single-instrument falsifiers in that the predicate is "perturbation visible at caller-frame samples, not at named hot function samples."

Hypothesis-set authority: §4.1 (H1-H5 + H-outside-hot-fn) is canonical; `.nbs/scribe/next-session-entry.md` ACTION 1b inherits this enumeration and does not extend beyond it.

### 4.2 HIR/LIR diff range per carrier
- Generate HIR/LIR for hot function in each carrier under both anchors (b252ca0425 vs 5830b49f19)
- `diff` HIR/LIR; flag any non-bit-identical region for hypothesis-mapping
- Per [bit_identical_not_innocence_proof]: bit-identical HIR can still perturb codegen via inlining/regalloc — need asm-level diff too

### 4.3 Compiled asm diff
- `objdump -d ./python | sed -n '/<JIT_function_name>/,/^$/p'` — extract JIT'd region
- diff between anchors; correlate with carrier hot path

### 4.4 Function-ordering diff
- `nm -nS ./python | sort -k1` — symbol order at memory address
- diff between anchors; flag any reordering of hot symbols

### 4.5 Controlled-anchor protocol
- ALL ABBAs same-session (per [feedback_abba_cross_session])
- Reps: 3-rep ABBA (per Phoenix gate convention)
- Vanilla anchor: per CLAUDE.md per-arch md5 check (per [feedback_vanilla_md5_precheck])

## 5. Falsifier composition (per-carrier minimum gate)

For each carrier fix-design that emerges from §2-§3 hypothesis testing:
- F-perf: per-bench 3-rep ABBA dual-arch, post-fix vs pre-fix; carrier-specific recovery >2pp = positive; <0.5pp = falsified
- F-regression: 4-bench M3 dual-arch must NOT regress >2% on non-carrier benches
- F-arm64-asymmetry: ARM64 must NOT lose any ARM64-positive c-series effects (e.g., try_except_callee +3.2pp ARM64) — fix-design must preserve those wins
- F-cross-bench: gen_simple negative-control — kwargs_dispatch fix MUST NOT regress gen_simple
- F-substrate-stress per [bit_identical_not_innocence_proof]: any code-path fix touching shared codegen path requires F3-strict ABBA + n=N stress

## 6. Risks (above-line; workers triage in implementation)

**R1.** Distributed perturbation = no single-symbol fix; may need 3-5 separate fixes across the 3 carriers. Implementation cost is multiplicative.

**R2.** Some Phoenix-codegen-perturbations may be ARM64-positive (e.g., try_except_callee +3.2pp ARM64 from c-series). Class B fixes risk reverting ARM64 wins on Class A symptoms. Cross-arch falsifier mandatory (F-arm64-asymmetry).

**R3.** Pre-c-series Phoenix-baseline regression source (the BULK leverage, ~25pp) lives in PRE-c-series Phoenix codegen. Bisecting WITHIN c-series captures only the +1.6 to +4.6pp x86 amplification. Investigation must distinguish "fix Phoenix-baseline" from "fix c-series amp" — they may be the same root cause (codegen-layout pre-existing, c-series accidentally amplified) or independent (codegen-layout pre-existing, c-series adds new perturbation source).

**R4.** Generator state machine (gen_simple) is the highest-leverage carrier (11.7pp) AND the highest-complexity (state machine vs. straight-line callee). Implementation risk highest here. Recommend kwargs_dispatch first (lowest complexity, validates investigation methodology) → nn_module_forward (medium complexity, vtable analysis) → gen_simple (highest complexity, deepest investigation).

**R5.** Audit cost per [class_of_bug_audit]: each carrier needs per-field consumer-semantic enumeration. 3 carriers × 5 hypotheses = 15 falsifier-tests minimum at investigation-class, before any code-class commit.

## 7. Prioritization recommendation

| Order | Carrier | Leverage (x86 pre-existing) | Complexity | Rationale |
|---|---|---|---|---|
| 1 | kwargs_dispatch | 4.6pp | low | Validates investigation methodology; shortest hot path |
| 2 | nn_module_forward | 8.6pp | medium | Vtable analysis builds on kwargs methodology |
| 3 | gen_simple | 11.7pp | high | Highest leverage, deepest investigation; do last when methodology is proven |

c-series x86-amp bisect (SECONDARY per supervisor 21:01:18Z): defer until pre-c-series fixes land OR investigation reveals shared root cause.

## 8. Next-session generalist starting kit (handoff)

**Cross-workstream note:** Class A Phase B (inline body, deferred) has TWO prerequisites:
  - F-1 (gate-correctness instrumentation, codegen-time): debug counter + sample-verify gate boolean correctly identifies eligible frame set. See chat 2026-05-15T00:05Z. Per librarian 00:04:37Z + pythia #409 — Phase A negative control proves asm-path-unchanged but NOT gate-correct.
  - F0 (runtime branch-rate measurement, conditional on F-1 PASS): hasRtfsFunction=true rate. See chat 2026-05-14T22:33:56Z.

If Phase B is dispatched first next-session, run F-1 → F0 → Phase B in that order per [bit_identical_not_innocence_proof].

1. Re-run testkeeper 24-bench dual-arch ABBA at HEAD (5830b49f19 + any post-Phase-A landings) to confirm Class B carrier deltas haven't shifted
2. Start with kwargs_dispatch (§7 priority 1):
   - Run perf-counter battery from §4.1 on kwargs_dispatch, both anchors
   - Generate HIR/LIR + asm diff for kwargs_dispatch hot function (§4.2-4.4)
   - Map findings to H1-H5 per §3
   - Post hypothesis-class verdict to chat (which H is supported by data)
3. Theologian on-deck for hypothesis review + fix-design code-class drafting per generalist's empirical findings

## 9. Unfalsified-at-gate

Forecast: ~25pp x86 addressable across 3 carriers IF all hypotheses fix-able. Realistic recovery: 30-60% of forecast (pre-existing regressions can have multiple root causes per carrier; some may be irreducible Phoenix-fork divergence from CinderX baseline).

Sub-2pp realized recovery on kwargs_dispatch fix → revisit hypothesis decomposition before nn_module_forward / gen_simple.
