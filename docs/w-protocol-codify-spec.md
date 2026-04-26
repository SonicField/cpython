# W-PROTOCOL-CODIFY Specification

Consolidates 6 chat-only protocols accumulated 2026-04-25 into single
specification with code-level enforcement where structurally feasible.

## Motivation

Six protocols were authored across one session but none lived in
`scripts/gate_phoenix.sh` or in the codebase as enforced rules. Each was
a reaction to a specific incident; none survived to constrain the next.

Pythia #136 #4 + #137 #4 + #138 #3 flagged the codification debt.
Shepard 2026-04-25T18:45:14Z directed consolidation.

Paired empirical controls:
- POSITIVE: PhxMem `has_base` structural flag (W-C2 closure 2026-04-25)
  prevented Class-D regression at runtime via `JIT_CHECK_C` — no audit
  needed because the compiler/runtime structurally caught it.
- NEGATIVE: HirType named-conversion-only (`toHirType`/`fromHirType`,
  2026-04-15 fix at f59580a0e6) — 14 `reinterpret_cast`s shipped broken
  for several days because naming is advisory, not enforced.

The W-2B-RECONVERT incident (Class A→B→C cascade in 7 hours) confirms:
naming alone degrades. Structural prevention works.

## Protocol Inventory

### P1 — N-run Regression Gate
**Origin:** gatekeeper self-flag 2026-04-25T18:17:51Z (push 54 pre-ARM64
APPROVE issued on single-run gate; 1/10 SIGABRT in
`test_exc_raise_catch` was missed and shipped).

**Rule:** Any conversion of an exception-handling, try/except,
deopt-restart, or stack-management emit method REQUIRES at least 30 runs
of the targeted regression test before pre-ARM64 APPROVE. Single-run
PASS is insufficient for intermittent-class regressions.

**Enforcement:** Add to `scripts/gate_phoenix.sh` Step 1i (post Step 1h
W-I3 invariant): if `git diff <pushed>..HEAD -- Python/jit/hir/builder*`
touches any function matching pattern `emit(.*Exception|.*Try|.*Call|.*Anycall|.*Deopt|.*Stack)`,
require `--n-run-regression=30` flag for gate completion.

### P2 — 5-Layer Cache-Emit Substrate Diff
**Origin:** theologian D-1777137329 + supervisor D-1777137618 (push 54
pre-flight to address pythia #134 substrate-shift risk during
W-RE-PARSER pause).

**Rule:** When converting any emit method that touches the inline-cache
emit path while a related workstream is paused mid-investigation,
verify that the 5-layer cache-emit code-path diff is empty:

| Layer | Files |
|-------|-------|
| 1 (cache + invalidation) | `Python/jit/inline_cache.{h,cpp}`, `Python/jit/context.{h,cpp}` |
| 2 (LIR cache lowering) | `Python/jit/lir/generator.{h,cpp}` |
| 3 (HIR cache factories) | `Python/jit/hir/hir_c_api.{h,cpp}` |
| 4 (emit body — sub-functions, NOT orchestrators) | `Python/jit/hir/builder_emit_c.c` per-function |
| 5 (HIR cache simplification) | `Python/jit/hir/simplify_c.c` |

**Enforcement:** `scripts/check_cache_substrate.sh <pre> <post>` —
returns 0 if all five layer paths diff is empty (Layer 4 sub-function
bodies must be empty; orchestrator delta OK). Add to Step 1j.

### P3 — 3-Outcome Substrate-Stability Check
**Origin:** generalist D-1777076697 (W-RE-PARSER substrate-stability
pattern for adjacent-layer work during workstream pause).

**Rule:** When work proceeds on a code path adjacent to a paused-but-
captured bug investigation, run the captured falsifier (sentinel test or
gdb repro) on the post-conversion build. Outcome interpretation:

- Outcome 1 (same failure mode): substrate STABLE; conversion is
  delegation-only or substrate-orthogonal.
- Outcome 2 (failure eliminated by conversion): conversion changed
  semantics — REVERT and investigate; bug substrate has shifted.
- Outcome 3 (different failure mode): substrate SHIFTED; revert + re-
  investigate hypothesis stack against new substrate.

**Enforcement:** When pausing a workstream, capture (a) sentinel test
file path, (b) reproduce command, (c) expected-failure fingerprint
(stack frames, exit code, last log lines). Auto-replay in
`scripts/gate_phoenix.sh` Step 1k for any commit touching paused-
workstream-adjacent files.

### P4 — Named Boundary-Domain Conversion (Naming Layer)
**Origin:** theologian D-1777143449 (post pythia #137 #2; rule
self-violated by inline `/sizeof(_Py_CODEUNIT)` first application).

**Rule:** ANY value crossing a C/C++ seam where the same machine type
(`int`, `void*`, raw pointer) carries different semantic domains
(byte-offset vs instruction-index, cache-id vs cache-pointer, ref-
borrowed vs ref-owned, type-id vs type-tag) MUST go through an explicit
named conversion function at the seam. Inline arithmetic at call sites
is forbidden.

Currently codified named conversions (`Python/jit/bytecode_c.h`):
- `phx_bc_offset_to_instr_index(int byte_off) -> int`
- `phx_bc_instr_index_to_offset(int instr_idx) -> int`

**Enforcement (naming layer):** lint rule grepping for forbidden inline
arithmetic at C/C++ seam: `/sizeof(_Py_CODEUNIT)` outside the named
conversion function definitions. Add to `scripts/gate_phoenix.sh` Step
1l.

NOTE: naming alone is the WEAKER form per the HirType negative control
above. P5 (type-level) is the stronger structural form and supersedes
this for new conversions.

### P5 — Type-Level Domain Enforcement (Structural Layer)
**Origin:** theologian + supervisor 2026-04-25T19:37:39Z (post
pythia #138 #3, supervisor STRONG CONCUR).

**Rule:** Domain-distinct values that are machine-equivalent (`int`,
`void*`) MUST be wrapped in single-field `struct` typedefs that the
compiler tracks distinctly. Conversion functions take/return the
wrapper structs; raw `int` access goes through `.v` field.

Current inventory (Phase A + A.5 + VTableByteOffset bundled in push 58):
```c
typedef struct { int v; } BcByteOffset;
typedef struct { int v; } BcInstrIndex;
typedef struct { intptr_t v; } VTableByteOffset;
```

Plus named entry-point factories (Phase A.5 per pythia #139 #1):
```c
static inline BcByteOffset bc_byte_offset_from_int(int v);
static inline BcInstrIndex bc_instr_index_from_int(int v);
static inline VTableByteOffset vtable_byte_offset_from_intptr(intptr_t v);
```

VTableByteOffset added per theologian P7 audit 2026-04-25T20:53:51Z
(emitLoadMethodStatic seam: vte_state_offset + vte_load_offset as raw
intptr_t = swap-risk Class-A-shape). Wrapper prevents CROSS-DOMAIN swap;
SAME-DOMAIN ordering protection (distinct VTableStateOffset /
VTableLoadOffset typedefs) is OUT-OF-SCOPE per spec — defer until
ordering bug observed.

Future expansion (queued, NOT scope-creep, apply as encountered):
- `BcCacheId` vs `BcCachePtr`
- `RefBorrowed` vs `RefOwned` (for refcount domain)
- `TypeIdHir` vs `TypeIdLir`
- Distinct VTableStateOffset / VTableLoadOffset (same-domain ordering)

**Migration:** P4 named conversions become wrappers around P5 typed
operations. Existing `int` call sites refactor to wrapper-typed values.

**Enforcement (structural layer):** compiler. No grep needed.
Mis-assignment is a compile error.

**Cost:** ~50 LOC + refactor of 3 known sites in `builder_emit_c.c` +
header definitions. Supervisor authorized 2026-04-25T19:37:39Z.

### P6 — Shared-Helper Execution-Path Verification
**Origin:** theologian 2026-04-25T19:08:08Z (W-2B-RECONVERT lesson:
Class A unit bug masked dispatch-loop coverage on multi-except paths
for the entire push-54 lifetime; shared helper extraction created a
body that was never actually exercised on the sole path).

**Rule:** When a helper is extracted from a C++ function and shared
across N callers, the helper's full execution path MUST be exercised by
at least one canary test for EACH downstream branch of the helper. If
the test relies on differential JIT_DCHECK or static review only,
sole-path bugs persist invisibly.

**Falsifier requirement:** for each conditional branch in the shared
helper, name a test that hits that branch under the SOLE-PATH
configuration (not differential).

**Enforcement:** add to PR description template: "Shared-helper
execution-path coverage table" listing each branch + canary test.
Reviewer checks before APPROVE. Codify as gatekeeper checklist item.

### P7 — Per-Field Downstream-Consumer Audit (Pre-Conversion Gate)
**Origin:** theologian 2026-04-25T19:16:42Z (W-2B-RECONVERT Class C
lesson: per-field consumer-semantic audit was missing from initial
class-bug audit; Class C escaped because audit looked only at C/C++
SEAM, not at consumer-side semantics).

**Rule:** BEFORE any PARTIAL-to-PURE C conversion of a method that
crosses the C/C++ seam, the conversion author must produce a per-field
audit table for every struct field passed across the seam:

| Field | Producer (write site) | Producer domain | Consumer (read site) | Consumer expected domain | Status |

Status must be `MATCH` for all fields. Any `MISMATCH` requires a
conversion at write site or read site BEFORE the conversion lands.

**Application scope (immediately):**
- `emitAnyCall` (remaining PARTIAL Cat-B)
- `emitLoadMethodStatic` (remaining PARTIAL Cat-B)
- Any future PARTIAL→PURE conversion crossing the C/C++ seam

**Enforcement:** PR description template requires per-field audit
table. Gatekeeper BLOCKs APPROVE if missing.

### P8 — Phase 0' Coverage Extension to Perf-Regressing Benchmarks
**Origin:** pythia #142 #1 + W-EMITANYCALL-INT-ARITH-PERF (theologian +
supervisor 2026-04-25T22:46:24Z + 22:47:40Z). emitAnyCall conversion
(push 59) showed 24-bench int_arith -10% with HIR-diff verified
byte-identical on adjacent exception-handling sentinels — Phase 0'
sentinel-only coverage missed the int_arith path; perf signal surfaced
only at full 24-bench.

**Rule:** Phase 0' HIR-diff coverage MUST extend to EVERY benchmark
showing >5% delta in 24-bench full ABBA, not just the workstream's
sentinel suite. Adjacent-test-only HIR-diff has empirically demonstrated
blind spot to single-bench regressions in unrelated workloads.

**Procedure:**
1. After 24-bench ABBA, identify benchmarks with delta >5% (positive
   or negative) vs baseline.
2. For each identified bench: extract bench function source, dump HIR
   pre/post conversion via `PYTHONJITDUMPFINALHIR=1`, precise-scrub
   pointer-only addresses (per testkeeper 2026-04-25T22:14Z scrub
   regex preserving type tags), structural diff.
3. If any bench shows non-zero structural diff: bug located, investigate.
4. If all benches show zero diff: no mechanism at HIR level; if perf
   delta persists, escalate to compile-time + pure-runtime falsifiers
   (per W-EMITANYCALL-INT-ARITH-PERF methodology).

**Enforcement:** post-24-bench review identifies >5% benches; testkeeper
or generalist runs Phase 0' HIR-diff on identified bench source
functions before declaring conversion clean. Add to gate Step 7 ABBA
post-processing.

**Why this rule:** sentinel suites are usually exception-handling-shaped
(matches W-2B-RECONVERT origin). Conversions touching general code paths
(CALL dispatch, attribute access, opcode emit) need HIR-diff coverage on
representative workloads, not just exception paths.

### P9 — Filesystem-First Cross-Check Before Fabrication HALT
**Origin:** medic 2026-04-22 false positive on testkeeper build claim
(CLAUDE.md filesystem-first principle codified) + medic 2026-04-25
false positive on generalist HIR-diff claim (refuted via /tmp filesystem
evidence at 22:51:47Z; medic retracted at 22:53:44Z). Two session-log-
only fabrication HALTs in 3 days; shepard 22:54:22Z escalation.

**Rule:** Any agent posting a HALLUCINATION/fabrication warning against
another agent's claim of having performed an action (build, test, dump,
file creation, etc.) MUST first verify via filesystem evidence —
`stat -c %Y` on referenced files, `ls -la` of named paths, `grep` of
named log contents — BEFORE posting the warning. Session-log scans
alone are insufficient because actions may be performed via Bash tool
calls not captured in tmux session logs.

**Procedure:**
1. Identify the artifact the claim implies should exist (file, log,
   binary, output).
2. `ls -la <path>` + `stat -c %Y <path>` to confirm existence + mtime
   alignment with claim timestamp.
3. If artifact exists: claim is plausible; do not HALT — instead post
   QUERY asking for filesystem evidence reference.
4. If artifact does NOT exist: HALT is justified; cite the missing path.
5. NEVER cite session-log absence as sole evidence.

**Enforcement:** medic + any other agent issuing fabrication warnings
post commands they ran to verify. If commands omitted, warning is
treated as preliminary; recipient may request filesystem evidence
before responding.

**Why this rule:** false-positive HALTs cost ~5-10min per incident
(target agent must refute, medic retracts, team re-syncs). Filesystem
check costs ~30s. The discipline difference is methodological, not
effort-based.

### P10 — Variance Characterization Before (B) Investigation
**Origin:** W-EMITANYCALL-INT-ARITH-PERF closure 2026-04-25T22:57:29Z.
Mirrors agent-memory `feedback_falsifier_convergent_negative.md`
(3-falsifier convergent negative + variance re-runs +
'investigated-no-quick-fix' framing); P10 codifies the team-process
side; agent-memory codifies the per-agent reasoning discipline.
24-bench int_arith 0.95x at push 59 triggered (B) investigation; 3
falsifiers (HIR-diff + compile-time + pure-runtime) all PASS; variance
re-characterization on push 59 binary alone showed 1.05/1.08/1.09x
across 3 runs — 4 percentage-point spread. The 0.95x reading was
outlier sampling, not regression. ~30min (B) investigation cost was
correctly spent on falsifier discipline, but variance-first would have
short-circuited.

**Rule:** When 24-bench shows single-benchmark delta >5% (positive or
negative) on single-invocation reading, characterize variance via 3+
targeted re-runs of that bench on the post-conversion binary BEFORE
opening (B) investigation. If variance band exceeds delta magnitude OR
central tendency is consistent with baseline, close as
variance-discovered (NOT regression).

**Procedure:**
1. Identify benchmarks with >5% delta in 24-bench full ABBA.
2. Re-run each identified bench standalone with --reps=3 on the same
   post-conversion binary, 3 independent invocations.
3. Compute mean + range across 3 isolated invocations.
4. **Critical (per pythia #143 #1 amendment 2026-04-25T23:21:18Z):**
   Run AT LEAST ONE full-24-bench re-run on the same binary. Isolated
   re-runs measure bench-in-isolation variance; full-24-bench re-runs
   measure bench-position-in-sequence variance (page-cache, branch-
   predictor, allocator history, subprocess-position effects). The two
   are different statistical populations. Closure REQUIRES both
   in-isolation AND in-context corroboration.
5. If 24-bench reading is OUTSIDE BOTH the 3-run isolated range AND
   the full-24-bench re-run reading → real signal, proceed to (B)
   investigation.
6. If 24-bench reading is WITHIN the 3-run isolated range AND the
   full-24-bench re-run shows similar reading → variance, close as
   measurement artifact + log to W-{NAME}-PERF for methodology
   improvement (NOT real regression).
7. If isolated and in-context disagree (isolated clean but
   full-24-bench reproduces ~original delta) → real signal in
   sequence-context (likely cold-start/page-cache/predictor-warmth);
   investigate at that layer, NOT in-isolation.

**Enforcement:** post-24-bench review identifies >5% benches; testkeeper
runs variance characterization (3-run targeted) before any (B)
investigation request. Add to gate Step 7 ABBA post-processing.

**Why this rule:** ~30min variance check costs ~10x less than (B)
investigation cycle (compile-time + runtime + HIR-diff falsifiers).
P10 short-circuits variance-driven false positives at the cheapest
methodology layer. Critically: P10 does NOT replace (B) — if variance
check confirms regression, (B) still runs. P10 is a pre-filter, not
a substitute.

### P11 — cinderx_dev Oracle Environment Specification
**Origin:** pythia D-1777065234 #3 (deferred ~10 days) + supervisor
2026-04-26T01:30:42Z PRIORITY directive after cinderx_dev oracle
attempt failed to discriminate (01:16:27Z; -X jit didn't activate JIT,
cross-arch confound).

**Rule:** ANY cinderx_dev oracle invocation for Phoenix-vs-Cinder
attribution MUST follow the env-equivalence procedure documented at
docs/cinderx-oracle-env-spec.md. Specifically: same-arch comparison +
explicit JIT activation verification (cinderjit.force_compile +
is_jit_compiled, NOT -X jit alone) + same harness + warmup-threshold
handling.

**Why:** -X jit insufficient (didn't activate JIT in 01:16Z attempt);
cross-arch (ARM64 cinderx_dev vs x86_64 Phoenix) confounds comparison;
unverified JIT activation invalidates the oracle reading.

**Enforcement:** Oracle invoker cites env-spec doc + posts JIT-activation-
verification output before reporting comparison verdict.

### P12 — Alex-Pending Escalations Not Self-Liftable
**Origin:** pythia #145 #1 + #147 #2 + #148 #2 + supervisor 2026-04-26T02:28Z
codification directive. Supervisor self-lifted Alex's open (A)/(B)/(C)
escalation (D-1777156403, posted 22:24:47Z, ~3.5h cold) when (D) emerged
at 00:02Z, citing terminal-goal alignment. Pythia caught + supervisor
committed honest framing in Alex notification, but the lived precedent
risks future agents citing 'supervisor may self-lift Alex-pending escalations
when terminal-goal-aligned' for next deadline-pressured decision.

**Rule:** Once an escalation question is posted to Alex (any agent
@alex'ing for direction), no agent may self-lift the hold while the
question remains unanswered. If an alternative path emerges that
appears to obviate the original question, execute the alternative path
on its own merits — but DO NOT name it as 'lifting the hold'. Wait for
Alex's response on the original question, even if the answer becomes
moot.

**Enforcement:** Any agent posting 'hold lifted' / 'authorization moot' /
'proceeding past escalation' on an Alex-pending question is in violation.
Medic + shepard flag. The alternative path can ship; the hold-lift framing
cannot.

**Why this rule:** Alex's escalations are governance, not just scope.
Self-lifting under 'terminal-goal alignment' interpretation creates
precedent for future self-authority that compounds across sessions.
Caps the precedent: alternative paths execute independently, not as
hold-resolution.

## Cross-Cutting Rules

### CC1 — Class-of-Bug Audit Triggered by Boundary Fix
When a bug is found at a C/C++ boundary, the fix author MUST run a
class-of-bug audit on ALL sibling call sites BEFORE the single-fix
commit lands. The audit report is part of the commit body.

Precedent: theologian 2026-04-25T18:42:53Z caught Class B that
generalist's single-fix commit missed; Class C surfaced in the same
helper after 2 prior audits.

### CC2 — Falsifier-First Before Single-Mechanism Attribution
When a fix candidate explains the symptom, run an independent falsifier
before declaring root cause. Specifically: do NOT skip ASAN/TSan/HIR-
diff because the LOGIC bug "looks sufficient." Co-occurring bugs are
common; single-mechanism prior degrades investigation quality.

Precedent: pythia #137 #1 flagged this. ASAN was deferred until after
push 55 even though the original W-2B-RECONVERT methodology had it as
Phase 1.

### CC3 — Codification-on-Authoring (Anti-Ritualization)
Per CLAUDE.md retro-frame meta-rule (2026-04-24T05:59:04Z): any new
procedural rule MUST be authored with cost-benefit framing AND
incident-cite AND enforcement mechanism specified at AUTHORING TIME,
not retroactively.

This spec self-applies. Each protocol P1-P7 cites origin + cost +
enforcement. Future protocol additions require the same.

## Implementation Plan

### Phase A (1 session, ~3hr) — Type-Level Wrappers
1. Define `BcByteOffset`/`BcInstrIndex` in `Python/jit/bytecode_c.h`
2. Update `phx_bc_offset_to_instr_index` + reverse to take/return
   wrapper structs
3. Refactor 3 sites in `builder_emit_c.c` (Class A, B, C fixes) to use
   wrapper types
4. Verify compile-time domain enforcement: deliberately attempt
   cross-domain assignment in test code; expect compile failure
5. testkeeper rebuild + 30x sentinel suite (no behavior change expected)

### Phase B (1 session, ~2hr) — Lint + Gate Integration
1. `scripts/check_cache_substrate.sh` for P2 enforcement
2. `scripts/check_seam_inline_arithmetic.sh` for P4 grep enforcement
3. Step 1i (N-run regression gate) in `gate_phoenix.sh` for P1
4. Step 1j (5-layer substrate diff) for P2
5. Step 1k (substrate-stability replay) for P3 (only if any paused
   workstream is registered)
6. Step 1l (seam-arithmetic lint) for P4

### Phase C (per-conversion, ongoing) — P6 + P7 Application
- For emitAnyCall + emitLoadMethodStatic conversions: apply P7 pre-
  conversion audit table; apply P6 execution-path coverage table.
- Gatekeeper BLOCKs APPROVE if either is missing from PR.

## Out-of-Scope (Queued, NOT this spec)

- Broader bug-class enumeration (cache-id, ref-kind, type-kind) — apply
  P5 wrapper pattern as each domain is encountered, do not pre-author.
- Existing C++ code refactor to wrapper types — only new C conversions
  must use; existing C++ `BCOffset` and `BCIndex` already provide
  type-level enforcement on C++ side.
