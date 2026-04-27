# W42 — Refcount-Instruction Correctness Verifier (C-port refcount_pass)

**Status:** ACTIVATED 2026-04-23T18:51:08Z per supervisor — threshold
MET (4 pre-existing refcount-correctness bugs surfaced this window:
await + controlflow + comparisons + gen_chain). Filed
2026-04-23T18:38:14Z as background candidate; activated per pythia
#101 spawn-vs-drain concern + lowered 3-bug threshold + 4-bug actual
count.

**Owner:** TBD.

**Estimated cost:** ~2-4 sessions (~8-16 hours) — substantial
infrastructure work.

**Activation justification:** 4 refcount-correctness bugs per pythia
#100/#101 within 95-commit window 05e2b8821a..4a01bfa3d1:
- W22 yield-from (refcount_pass n_in==0 NULL deref) — FIXED 66850a4ba1
- W40 controlflow (match-statement over-decref) — FIXED f47bcc9a8a
- W41 comparisons (match-sequence guard over-decref) — FIXED f47bcc9a8a
- await SIGSEGV — separate deopt v27-not-live, refcount-correctness root
- gen_chain SIGSEGV — separate hir_remove_unreachable_blocks_c NULL,
  same window pre-existing-class

Surface continues to grow per Phase 3D burndown adding C-port (Tier 7
in flight). Reactive 5-bug threshold deferred infrastructure too far;
3 customer-visible incidents = real cost. Active mitigation now.

---

## 1. Problem statement

Phoenix Phase 3D ported `refcount_insertion.cpp` and
`refcount_env.cpp` to C (`refcount_pass_c.c`, `refcount_env_c.c`).
The port surface verification mechanism is **structural**:
- Build-clean (compiles)
- W21 golden trip-wire (HIR shape comparison, structural)
- Wiring gate (force_compile sole-path)
- Differential JIT_DCHECK (per memory `project_wiring_methodology.md`)
  — but **NEVER EXTENDED to refcount_pass** per pythia #100
  2026-04-23

W22 + 3 residual bugs (W39 await + W40 controlflow + W41 comparisons)
empirically demonstrate the gap: build-time crash class is detectable
(NULL deref), but runtime refcount-correctness class (over/under-decref
producing PyObject corruption) surfaces only when:
- The downstream consumer (deopt, PyObject_Repr, gen_chain runtime)
  reads the corrupted state, AND
- The reader's fault address happens to be in a stack frame that
  surfaces a symptom rather than silent corruption

Per generalist 2026-04-23T18:15:49Z: "manual GDB on individual SIGSEGVs
without [the verifier] means hours-per-bug; with the verifier, single
fix surface."

## 2. Scope

### 2.1 In-scope

- C-port refcount instruction correctness vs C++ authoritative
  reference (`refcount_insertion.cpp` at git rev `1d2b9a737e^`)
- Per-emit verification: for each HIR instruction processed by
  `phx_rc_run`, compare the emitted refcount instruction list (Incref,
  Decref, BorrowFrom) against what the C++ baseline emits for the
  same input
- Coverage: at minimum the 8 W22-cluster HIR opcodes (YieldFrom,
  YieldValue, Send, GetYieldFromIter, MatchSequence, UnpackSequence,
  CallEx, CallMethod). Extension to all HIR opcodes via
  fixture-generated coverage.

### 2.2 Out-of-scope

- W21 golden's structural concerns (HIR shape) — already covered
- Wiring-gate sole-path coverage — already covered (post-W32 repair)
- Differential JIT_DCHECK for analysis passes — already covered

## 3. Verification mechanism options

### Option A — Inline differential JIT_DCHECK extension

Per `project_wiring_methodology.md` analysis-pass pattern: at each
`phx_rc_process_instr` call site, run the C++ baseline's
corresponding code path on the same HIR instruction in parallel,
compare emitted refcount-instruction vector. JIT_DCHECK fires on
divergence.

Cost: ~4-8 hours infrastructure. Live coverage on every JIT compile.

Risk: requires keeping `refcount_insertion.cpp` linked even after
Tier 7 burndown. Conflicts with ZERO C++ goal.

### Option B — Offline fixture-driven verifier

Build a fixture suite of HIR-instruction inputs (per opcode, per
shape variant), run both C-port and C++ baseline through them,
compare emitted refcount-instruction lists. Run as gate item, not
inline.

Cost: ~6-12 hours fixture coverage.

Advantage: doesn't conflict with Tier 7 (C++ baseline binary
maintained as test-only artifact, similar to `python_baseline`
methodology).

### Option C — Invariant-class assertion in C-port

Identify N invariants the refcount port must satisfy (e.g.,
"every Incref must be paired with a matching Decref or
ReleaseRef on every CFG path"; "n_in==0 blocks must populate
in-state from authoritative liveness, not heuristic"; etc.).
Embed as JIT_CHECK assertions in C-port.

Cost: ~3-6 hours per invariant identified.

Risk: invariant identification is non-trivial; missing invariants
leave gaps.

### Recommendation

**Option B (offline fixture-driven verifier).** Compatible with
Tier 7 ZERO C++ goal; avoids inline runtime cost; provides
auditable test fixtures. Defers Option A until post-Tier-7 if
inline coverage is needed.

Option C is complementary: identify 5-10 invariants from W22 +
W39/W40/W41 root causes + embed as JIT_CHECK at the same time.

## 4. Acceptance criteria

1. Fixture coverage for ≥8 W22-cluster HIR opcodes (YieldFrom,
   YieldValue, Send, GetYieldFromIter, MatchSequence,
   UnpackSequence, CallEx, CallMethod).
2. Verifier detects: over-decref (decref > N for ref
   originating with N), under-decref (decref < N), wrong-target
   decref (decref'd reg ≠ instruction operand), missing decref on
   exit.
3. Verifier RE-CATCHES W22 + W39/W40/W41 root causes when applied
   to pre-fix C-port code (confirms it would have caught these
   pre-emptively).
4. Verifier integration into `gate_phoenix.sh` as standard gate
   item (~5min add per push).
5. Optional: 5-10 JIT_CHECK invariants embedded in C-port from
   W22-cluster root-cause analysis.

## 5. Cross-link

- Pythia identification: pythia #100 2026-04-23
- Theologian framing: 2026-04-23T18:10:43Z + 2026-04-23T18:11:43Z
- Generalist recommendation: 2026-04-23T18:15:49Z
- Supervisor filing: 2026-04-23T18:38:14Z
- Methodology precedent: project_wiring_methodology.md (analysis-pass
  differential JIT_DCHECK)
- Memory: feedback_no_workarounds.md + feedback_gdb_first.md (root
  causes via GDB; verifier prevents regression)
- Empirical surface: W22 + W39 await + W40 controlflow + W41
  comparisons (Phoenix Phase 3D refcount C-port)
- Related workstreams: W33 (zero-bridge verifier) + W34 (__static__
  retroactive test pass) — sibling correctness verifiers
