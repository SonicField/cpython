# W22 — Liveness Fix Falsification Gate Spec

**Status:** ACTIVE — gate spec for (A) ROOT-CAUSE liveness fix per
generalist [chat L2986] root-cause finding + theologian [chat L2988]
recommendation + supervisor [chat L2989] concur. Awaiting Alex
ratification of (A) approach per supervisor escalation [chat L2993].

**Owner:** generalist (impl) → testkeeper (gate execution).

---

## 1. Background

Per generalist L2986: deterministic x86_64 RelWithDebInfo reproducer
(/tmp/repro_min3.py) crashes with `JIT: deopt.cpp:478 Assertion
failed: it != reg_idx.end() — register v11 not live` at iter 1000
(auto-compile threshold).

Root cause: `YieldFrom` DeoptBase in bb 2 of `gen_yieldfrom`'s HIR
has NO `LiveValues<>` clause despite its `FrameState` referencing
v11 + v20. All other DeoptBase instrs in the same function DO have
`LiveValues<>` populated.

Bug location: `refcount_pass_c.c phx_rc_fill_deopt_live_regs` —
emplaces only regs in `env->live_regs` into the deopt instr's
live_regs. v11 is missing from `env->live_regs` by the time
YieldFrom in bb 2 is processed.

Bug pre-dates Phase A C-port. Phase A C-port faithfully reproduced
the inherited CinderX bug (validates pythia #87
differential-JIT_DCHECK-necessary-not-sufficient framing per
librarian L2989).

## 2. Fix direction (A) ROOT-CAUSE

Per Alex no-workarounds directive + CLAUDE.md "Debug-First for
Unknown Crashes" + memory `feedback_no_workarounds.md`: fix the
underlying liveness analysis bug, NOT a defensive walk in the
consumer.

Investigation candidates (per generalist L2986 + theologian L2988):
- Liveness backward-flow termination treatment of loop bb 1↔bb 2
  cycle.
- FrameState localsplus walk in liveness vs in fill_deopt_live_regs
  differing.
- Visit order: hir_c_visit_uses → visitUsesDeopt walks FrameState
  localsplus, but the resulting reg may not propagate back across
  the cycle.

Investigation methodology (per CLAUDE.md Debug-First protocol step
2): "Set watchpoints on the corrupted memory to find the EXACT code
path that creates the bad state." Watchpoint v11's liveness state
between bb 1 and bb 2 transitions; identify the exact pass +
location that drops v11 from live_regs.

## 3. Falsification gate (3 criteria — ALL must pass)

### Criterion 1: Positive — repro must NOT crash post-fix

```
./python /tmp/repro_min3.py
```

Expected: 1100 iterations complete with no JIT_CHECK fire, no
SIGSEGV, no AssertionError. Exit 0.

If `repro_min3.py` still crashes → fix is INCOMPLETE; root cause
not actually fixed.

### Criterion 2: Synthetic mutation in LIVENESS path must reproduce

Mutate the LIVENESS analysis code (NOT the consumer
`phx_rc_fill_deopt_live_regs`) in a way that should drop v11 from
live_regs at the same spot. Re-run repro_min3.py; must crash with
the same `register v11 not live` assertion.

Purpose: proves the fix is in the liveness path (root cause), not in
the consumer (workaround). If mutation in LIVENESS doesn't reproduce
the crash, then the original fix may have been in the consumer
(masquerading as liveness fix); revisit.

Implementation: temporarily revert one statement in the liveness
fix; build; run repro. Restore after verification.

### Criterion 3: Differential JIT_DCHECK must STILL pass post-fix

Run the standard differential JIT_DCHECK suite (Phase A C-port
verification). C-side `liveness_c.c` and C++ `LivenessAnalysis`
must converge on the FIXED behavior; both produce identical
liveness sets that include v11 across bb 1↔bb 2 cycle for
gen_yieldfrom.

If differential JIT_DCHECK fails post-fix → C and C++ have diverged.
Either the C-side fix didn't propagate to C++, or vice versa. Fix
both consistently.

## 4. Regression test staging (testkeeper task per L2992)

Stage `/tmp/repro_min3.py` as a permanent Phoenix test:
`Lib/test/test_phoenix_w22_yieldfrom_liveness.py`. Test should:
1. Define the gen_yieldfrom + f2 functions.
2. Call f2 1100 times.
3. Verify no JIT_CHECK fires (assert no AssertionError raised; assert
   sys.exc_info() is empty post-loop).
4. Run with `JIT_ENABLE=1` + `force_compile` AND with
   auto-compilation threshold path.

Add to gate_phoenix.sh Step 4 (CPython tests) and to manual force_
compile wiring smoke (post-W32 4a01bfa3d1 repair).

## 5. Pre-fix prerequisites

Before generalist commits the (A) liveness fix:
1. **Alex ratification of (A) approach** per supervisor escalation
   [chat L2993] item (1).
2. **Watchpoint plan** per CLAUDE.md Debug-First step 2 (generalist
   per shepard L2992).
3. **W22 regression test staged** by testkeeper per L2992.
4. **Falsification gate criteria 1-3 documented** (this doc).

After fix commits:
1. Run criterion 1 (positive repro).
2. Run criterion 2 (synthetic mutation).
3. Run criterion 3 (differential JIT_DCHECK).
4. ALL THREE PASS → W22 RESOLVED. HALT lifted.
5. Update memory `feedback_no_preexisting.md` with
   'gate-repair-surfaces-hidden-crashes' subclass lesson per
   theologian L2979.

## 6. Cross-link

- Empirical: generalist L2986 root-cause + repro_min3.py
- Theologian recommendation: [chat L2988]
- Supervisor concur: [chat L2989]
- Alex confirmation HALT correct: [chat L2982]
- Memory: `feedback_no_workarounds.md` + `feedback_no_preexisting.md`
- CLAUDE.md: Debug-First for Unknown Crashes (Alex standing
  directive) + No Pre-existing Dismissal (2026-04-17 + 2026-04-22)
- Pythia validation: #87 (re-issue 2026-04-23) — differential
  JIT_DCHECK necessary-but-not-sufficient
- Librarian validation: [chat L2989]
- W31/W32 cross-link: --wiring infra repair surfaced this crash
  class (W32 4a01bfa3d1 verification triplet PASS per testkeeper
  L2978)
