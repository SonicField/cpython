# W31 — `GetRPOTraversal(start)` C-port Falsification Required

**Status:** FILED FORWARD-LOOKING (no current workstream — flag for any
future C-port of cfg.cpp Cat-B `GetRPOTraversal(BasicBlock*)` /
`GetPostOrderTraversal(BasicBlock*)` overloads). Per supervisor [chat
L2900] + pythia #91 #2.

**Owner:** TBD (whichever workstream first proposes C-porting these
overloads, NOT just header-inlining them).

---

## 1. Background

Batch 2-F (cfg.cpp → cfg.h header-inline) preserves the existing C++
bodies verbatim — NO sole-path C-port of `GetRPOTraversal(start)` /
`GetPostOrderTraversal(start)` Cat-B overloads. The C-side
`hir_cfg_get_rpo_c` is invoked only for the `const` overloads (Cat-A
delegations), not for the (start) overloads which keep their C++
recursive-postorder logic.

If a future workstream proposes C-porting either (start) overload to
mirror the C-side `phx_rc_get_rpo` pattern, **falsification is
mandatory** — both overloads exhibit the bug class historically
demonstrated by R3b bug #11.

## 2. Bug #11 RPO divergence precedent

Per memory `project_r3b_wiring_bugs.md`:

R3b refcount_insertion C-port wired its own C RPO traversal that
diverged from the C++ `CFG::GetRPOTraversal()` ordering on certain CFG
shapes. The divergence was silent under differential JIT_DCHECK
(C-output matched C++-output **for parallel runs** because both consumed
the same input CFG independently) but caused bulk failures the moment
the C path was wired as the sole producer of RPO ordering for downstream
consumers. The fix introduced the C-RPO-bridge wiring so the C-side
delegates back to C++ `GetRPOTraversal()` for the canonical ordering.

**Lesson:** RPO/postorder ordering is a class of computation where:

1. Differential JIT_DCHECK (parallel run, compare outputs) can pass
   even when the two implementations diverge on edge-case CFG shapes,
   because each consumes the same input CFG independently and reaches
   the same final block-set with **different orderings** that both
   consumers consider valid in isolation.
2. Sole-path divergence surfaces only when downstream consumers
   (refcount insertion, dataflow, dominator-based passes) depend on
   ordering stability across the boundary.
3. The wiring gate (per D-1776714419) is the only mechanical detector
   for this class — differential JIT_DCHECK is structurally blind to
   it.

## 3. Falsification requirement for any future C-port

Any future workstream C-porting `GetRPOTraversal(BasicBlock*)` or
`GetPostOrderTraversal(BasicBlock*)` MUST:

1. Verify the new C body produces **byte-identical block-pointer
   ordering** to the existing C++ body across:
   - Linear CFG (1 entry, no branches)
   - Diamond CFG (entry → 2 branches → join)
   - Loop CFG (back-edge to header)
   - Irreducible CFG (multiple entry edges to a loop body)
   - Critical-edge CFG (predecessor with ≥2 successors → block with ≥2
     predecessors)

   Falsification harness must enumerate at least these 5 shapes; spot
   any divergence as a hard FAIL.

2. Run the wiring gate (`scripts/gate_phoenix.sh --wiring`) which
   force-compiles ≥5 functions exercising RPO consumers
   (refcount_insertion, dataflow, dominator). PASS required, **AND
   --wiring infra must be repaired before the C-port workstream
   proceeds** per testkeeper [chat L2906] infra-ticket + W32
   deferred. Manual force_compile substitute (per testkeeper L2906
   pattern) is NOT acceptable for sole-path C-port falsification —
   header-inline batches use it as a substitute on safe-by-construction
   grounds; sole-path C-port has no such guarantee.

3. NOT skip the falsification on "differential JIT_DCHECK passes"
   grounds. Differential is necessary-but-not-sufficient for this bug
   class per W26 spec §4 amendment + bug #11 precedent.

4. NOT skip the falsification on "ZERO new bridges" grounds. Per
   pythia #87 (2026-04-23) + theologian L2934 batch-class breakdown:
   the ZERO-bridge → §4 SKIP carve-out (D-1776903189) applies to
   header-inline (safe-by-construction) and pure deletion (no
   regression class) batches. Sole-path C-port introduces codegen
   drift class that ZERO-bridge claim alone does not bound.

## 4. When this surfaces

W31 is dormant until either:

- A future workstream proposes C-porting cfg.cpp Cat-B (start)
  overloads (currently no scheduled work).
- An upstream optimization pass needs an RPO variant not exposed by
  `phx_rc_get_rpo` and someone adds a C-side body.
- A performance investigation suggests the existing C++ body is the
  hotspot.

Cross-link:
- Empirical incident: R3b bug #11 (memory `project_r3b_wiring_bugs.md`)
- Pythia hypothesis: #91 #2 (2026-04-23)
- Supervisor flag: [chat L2900]
- Falsification spec: extends W26 §4b semantic-equivalence-gates pattern
- Wiring gate: D-1776714419 **sole-path gate** (per memory
  `feedback_compile_before_commit.md` + Phase 3D wiring gate adoption).
  **NB gate-vintage:** D-1776714419 is the sole-path gate (force_compile
  ≥5 functions covering RPO consumers). It is distinct from the
  earlier D-1775424410 **frame_asm gate**. Historical "wiring gate
  verified" claims in memory entries (SSAify eb277536fc, etc.) refer
  to the older frame_asm gate, NOT the D-1776714419 sole-path gate.
  W31 falsification requires the D-1776714419 sole-path gate
  specifically — frame_asm gate does not exercise RPO divergence per
  bug #11 class. Per librarian [chat L2939] gate-vintage clarification.
