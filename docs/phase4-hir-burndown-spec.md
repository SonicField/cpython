# Phase 4 — HIR Completion: HirBuilder Final Migration to Pure C

**Authored:** 2026-04-27 (theologian) per supervisor 2026-04-27T08:18:04Z
("author docs/phase4-hir-burndown-spec.md as a standalone canonical
doc"). Originated as §8 of
`docs/tier7-phase3-hirbuilder-state-extraction-spec.md` (2026-04-27
07:31Z); split into this dedicated spec to keep the parent doc
focused on the original Tier-7 state-extraction question.

**Parent context:** `docs/post-phase3d-pure-c-roadmap.md` §3 Phase 4
(roadmap policy per supervisor 2026-04-27T07:20:19Z + Alex
D-1777270945 "back on roadmap").

**Sibling specs:**
- `docs/tier7-phase3-hirbuilder-state-extraction-spec.md` — Phase 3
  (EXECUTED 2026-04-23) state-struct foundation + Class B-kept
  validation
- `docs/tier8-class-b-cport-migrate-arm-spec.md` — Tier 8 pilots A+B
  (EXECUTED 2026-04-24/25) for `exception_table_` + `block_map_`
  migration; Phase 4.C extends with pilots 3+4

**Status:** ACTIVE. Phase 4 OFFICIALLY OPEN per supervisor
2026-04-27T07:31:48Z (all 5 §4.4 decisions accepted).

---

## 1. Scope

Convert remaining HIR-area C++ files (`Python/jit/hir/*.cpp`) to pure
C, including:
- Algorithm bodies in non-builder HIR files (hir.cpp, printer.cpp,
  inliner.cpp, preload.cpp, hir_instr_c_verify.cpp,
  builtin_load_method_elimination.cpp)
- HirBuilder class state extraction completion (Class B-kept temps_,
  static_method_stack_; Class A C++ duplicate deletion)
- builder.cpp dispatch loop conversion + shrinkage to ≤200 LOC
- hir_c_api.cpp partial dissolution (~600-1,000 LOC of HIR-internal
  bridges)

**Net target:** 14,051 LOC C++ deletion (11,319 algorithm + 2,732
bridge dissolution) per `docs/post-phase3d-pure-c-roadmap.md` §3.

**Out of scope** (deferred to later phases):
- LIR area conversion (Phase 5)
- Codegen area conversion (Phase 6)
- Runtime files (Phase 7)
- pyjit + context (Phase 8)
- Cross-area bridge consumers in hir_c_api.cpp (full dissolution waits
  for Phase 5+6 completion)

---

## 2. State of HirBuilder, post-Tier-8-A+B (measured 2026-04-27)

`PhxHirBuilderState` struct (`builder_state_c.h:330-339`) currently
holds:

| Field | Type | Origin | Status |
|---|---|---|---|
| `code` | `void*` (PyCodeObject*) | Phase 3 Batch 1 mirror | Class A — mirrored, C++ field still present |
| `preloader` | `const void*` | Phase 3 Batch 1 mirror | Class A — mirrored |
| `current_func` | `void*` (Function*) | Phase 3 Batch 1 mirror | Class A — mirrored |
| `func` | `void*` (Register*) | Phase 3 Batch 1 mirror | Class A — mirrored |
| `kwnames` | `void*` (Register*) | Phase 3 Batch 1 mirror | Class A — mirrored |
| `exception_table_phx` | `PhxExceptionTable` | Tier 8 pilot 1 Phase A/B | **MIGRATED** (C++ field deleted) |
| `block_map_phx` | `PhxBlockMap` | Tier 8 pilot 2 Phase A | **MIGRATED** (C++ field deleted) |
| `bc_block_array_phx` | `PhxBcBlockArray` | Tier 8 pilot 2 Phase B | **MIGRATED** (C++ field deleted) |

C++ HIRBuilder member fields STILL in `builder.h`:

| Field | Type | Class | Bridge today | C-side callers |
|---|---|---|---|---|
| `code_` | `PyCodeObject*` | A — mirror lives in state_ | direct via state_.code | indirect via state_ |
| `preloader_` | `const Preloader&` | A — mirror lives in state_ | direct via state_.preloader | indirect via state_ |
| `current_func_` | `Function*` | A — mirror lives in state_ | direct via state_.current_func | indirect via state_ |
| `temps_` | `TempAllocator` (C++) | **B-kept** | `hir_builder_state_temps_alloc_stack_cpp` | **73** |
| `func_` | `Register*` | A — mirror lives in state_ | direct via state_.func | indirect via state_ |
| `kwnames_` | `Register*` | A — mirror lives in state_ | direct via state_.kwnames | indirect via state_ |
| `static_method_stack_` | `OperandStack` (`jit::Stack<Register*>`) | **B-kept** | `hir_builder_state_static_method_stack_{push,pop}_cpp` | **5** |
| `state_` | `PhxHirBuilderState` | (the C struct itself) | n/a | n/a |

**Class A duplicate-state cost:** the 5 mirrored fields exist in BOTH
the C++ class AND `PhxHirBuilderState`. C++ mutator sites must update
both in lockstep — silent drift on a missed mutator is a latent bug
class. Phase 4 closes this by deleting the C++ duplicates.

**Class B-kept residue:** `temps_` (15 direct C++ accesses + 73 C-side
bridge calls) and `static_method_stack_` (2 direct + 5 bridge) are the
two outstanding migrate-arm targets. Tier 8 pilot 1+2 validated the
migrate pattern; pilots 3+4 are the natural extension.

---

## 3. builder.cpp dispatch surface (measured 2026-04-27)

| Surface | Count |
|---|---|
| `HIRBuilder::*` method bodies | 128 |
| Class A member access sites (current_func_/func_/kwnames_) | 103 |
| `state_.` access sites | 18 |
| `preloader_` access sites | 28 |
| `code_` access sites | 46 |
| Direct `temps_` access | 15 |
| Direct `static_method_stack_` access | 2 |
| File total LOC | 4,527 |

Top-level entry points (the dispatch loop and its callers):
- `buildHIR()` — `compiler.cpp:236` external caller
- `buildHIRImpl(Function*, FrameState*)` — internal entry
- `inlineHIR(Function*, FrameState*)` — inliner entry (called from
  `inliner.cpp`)
- `translate(Function&, BytecodeInstructionBlock&, TranslationContext&)`
  — the dispatch loop, ~600 LOC, dispatches over bytecode opcodes to
  the 100+ emit*/helper methods (which already delegate to C bodies in
  `builder_emit_c.c`)

`TranslationContext` is a C++ struct with template `emit<T>()` /
`emitChecked<T>()` methods. POD-mirror `PhxTranslationContext` already
exists in `builder_emit_c.c:42-47` with layout-asserted compatibility
(`builder.cpp:1021-1024`). The C++ template emit* methods exist for
ergonomics; they're not algorithmically necessary — every call site
can go through `phx_tc_emit(tc, instr)` against a pre-created
instruction.

---

## 4. Phase 4 sub-step sequencing (within HIR area)

Phase 4 burndown order (lowest-state-coupling first; avoids stalling
on the unsolved class-state-extraction problem while non-state files
land):

### 4.A — Low-state HIR files (parallel with §6 design closure)
- `hir.cpp` 1,268 LOC — HIR core; mostly free functions + factory
  bodies
- `printer.cpp` 957 LOC — text serializer; pure-formatting, no
  HirBuilder dep
- `hir_instr_c_verify.cpp` 377 LOC — verifier; thin friend-struct
  pattern (per `feedback_verifier_pattern.md`)
- `builtin_load_method_elimination.cpp` 184 LOC — pass; no class state
  **(IN FLIGHT 2026-04-27 batch 1 256984d4fd, supervisor PUSH
  AUTHORIZED 08:30:45Z)**
- **Subtotal:** 2,786 LOC, 30-50 commits estimated, 4-7 sessions

  Each file is HIRBuilder-independent and can be converted using
  established Phase 3D patterns (algorithm-body split, fixture-coverage
  + ABBA gate verification per BLME precedent, _c.c port + .cpp
  delegation). No class-state work.

### 4.B — Medium-state HIR files
- `inliner.cpp` 704 LOC — uses HIRBuilder for inlineHIR; depends on
  Phase 4.C class-state migration partially landing
- `preload.cpp` 570 LOC — Preloader class; HirBuilder uses by
  const-ref but Preloader itself is independent
- **Subtotal:** 1,274 LOC, 15-25 commits, 2-3 sessions

  preload.cpp can land in parallel with Phase 4.A (no HirBuilder
  state dep). inliner.cpp gates on at least Phase 4.C pilot 3
  (`temps_` migration) since `inlineHIR` reuses the caller's `temps_`.

### 4.C — Class B-kept migrate-arm pilots (analogues of Tier 8 pilots 1+2)
- **Pilot 3:** `temps_` (TempAllocator) → `PhxTempAllocator` in `state_`
  - TempAllocator is a value-type with `Environment* env_` +
    `std::vector<Register*> cache_`
  - Migrate cache_ to PhxRegisterArray (similar to PhxExceptionTable
    pattern: data + count + capacity, doubling)
  - 3 methods to port: AllocateStack, GetOrAllocateStack,
    AllocateNonStack
  - **15 C++ direct accesses + 73 _cpp bridges → 0 bridges + 73 direct
    state-struct accesses**
  - **Bridge delete: 1** (`hir_builder_state_temps_alloc_stack_cpp`)
  - **Estimated cost:** 5-8 commits, 1 session

- **Pilot 4:** `static_method_stack_` (OperandStack =
  `jit::Stack<Register*>`) → `PhxRegisterStack` in `state_`
  - Stack of pointers; trivial Phx container (smaller than PhxBlockMap)
  - Direct linear-array LIFO with doubling
  - **2 C++ direct + 5 _cpp bridges → 0 bridges + 7 direct accesses**
  - **Bridge delete: 2** (push_cpp + pop_cpp)
  - **Estimated cost:** 3-5 commits, 0.5 session

- **Pilot 5 (lockstep with 3+4):** delete the 5 Class A C++ duplicates
  (`code_`, `preloader_`, `current_func_`, `func_`, `kwnames_`). All
  access goes through `state_` exclusively. Removes the silent-drift
  latent bug class.
  - **Estimated cost:** 5-8 commits per field × 5 = 25-40 commits, 3-4
    sessions; each commit migrates 1 field across all C++ access sites
    and verifies via grep + JIT_DCHECK + benchmark gate
- **Subtotal Phase 4.C:** 33-53 commits, 4.5-5.5 sessions

### 4.D — Dispatch loop + builder.cpp shrinkage
- `translate()` dispatch loop conversion to C — the central method
- `buildHIR`, `buildHIRImpl`, `inlineHIR` entries: convert to C bodies
  with thin C++ shims (or direct C if `compiler.cpp:236` caller can be
  rewired)
- `allocateLocalsplus`, `addInitialYield`, `addLoadArgs`,
  `addInitializeCells`, `createBlocks`, `emitInlineExceptionMatch`,
  `emitCallExceptionHandler`, `emitTypeAnnotationGuards`,
  `advancePastYieldInstr` — algorithmic helpers
- `TranslationContext` C++ struct: delete (`PhxTranslationContext`
  replaces it; template emit*/emitChecked* methods become free-function
  C dispatch via existing factory C-API)
- `TempAllocator` C++ class: delete (after Pilot 3)
- `BlockCanonicalizer` C++ class: convert (uses `temps_` +
  `PhxPtrArray`; scope dependent on Pilot 3)
- **Subtotal:** 4,527 → ≤200 LOC residual, 30-50 commits, 4-6 sessions

### 4.E — hir_c_api.cpp bridge dissolution (HIR-internal subset)
- `hir_c_api.cpp` 2,732 LOC is the C↔C++ bridge for HIR data types.
  Each bridge function exists ONLY because at least one C caller
  needed it while the C++ side held canonical state.
- After Phase 4.A-D land, every HIR-side C++ caller is gone.
  Remaining callers are: `lir/generator.cpp` + `lir/lir_c_api.cpp` +
  `codegen/{gen_asm,annotations}.cpp` + `compiler.cpp` +
  `pyjit.cpp` + `inliner.cpp` (the latter migrates with Phase 4.B).
- These are CONSUMERS of HIR APIs from other Phase-5+ areas. The
  bridge cannot fully dissolve until Phase 5 (LIR) and Phase 6
  (codegen) consume HIR via the C-only seam.
- **Phase 4 partial dissolution target:** delete the bridge functions
  whose ONLY callers were HIR-internal (now-gone). Per-batch grep:
  `grep -lE "hir_<bridge_name>" $(find Python/jit -name "*.cpp")`.
  If empty, delete bridge.
- **Estimated dissolution in Phase 4 alone:** ~600-1,000 LOC of
  hir_c_api.cpp (the HIR-internal-only bridges).
- **Full dissolution** waits for Phase 5+ completion, per the roadmap
  §3 dependency ordering.
- **Subtotal Phase 4.E:** 5-10 commits, 1 session

---

## 5. Intermediate checkpoint gates (per pythia #186 2026-04-27)

The Phase-4-close LOC-delete falsifier (§7 falsifier #1, ≥80% LOC
delete) is a single trigger at the END of Phase 4. Pythia #186 flagged
that this leaves no mid-flight signal: BLME alone (the easiest pilot)
won't validate Phase 4.C class-state path or the ≥80% target.

**Mid-Phase-4 checkpoint gates** (added per pythia #186):

### 5.A — Phase 4.A close gate
**Trigger:** when 3 of 4 Phase 4.A files have landed (3,786 - 957/377/184
= 2,602 LOC C++ deleted target before printer.cpp/hir.cpp).
**Pass criteria:**
- Cumulative C++ delete ≥ 2,200 LOC (85% of 2,602 target)
- Per-commit ABBA: same-session geo-mean ≥ -5% across the 3 files
- Zero functional regressions on the 7-test Phoenix JIT suite
- ARM64 dual-arch tree-match clean for every commit
**Fail action:** PAUSE Phase 4.B start; theologian re-spec the §4.A
remaining file (likely hir.cpp 1,268 LOC if it's the holdout).

### 5.B — Phase 4.B close gate
**Trigger:** preload.cpp + inliner.cpp both landed.
**Pass criteria:**
- Cumulative Phase 4.A+B C++ delete ≥ 3,400 LOC (84% of 4,060 target)
- Inliner.cpp post-Pilot-3 verification: temps_ usage in inlineHIR
  goes through PhxTempAllocator with zero _cpp bridge calls
- Same per-commit ABBA + ARM64 + functional gates
**Fail action:** PAUSE Phase 4.C scope expansion; theologian re-spec
the inliner-temps_ interaction.

### 5.C — Phase 4.C Pilot 3 entry gate (per §7 falsifier #1)
**Trigger:** before Pilot 3 (temps_ migration) commits begin.
**Pass criteria:**
- PhxTempAllocator design doc landed (analogous to PhxExceptionTable
  spec) — theologian-authored, supervisor-approved
- Reference benchmark: same-session ABBA on TempAllocator-heavy
  workload (e.g. simplify_c benchmark suite, or fibonacci/nqueens
  which exercise register allocation)
- Pre-Pilot-3 baseline geo-mean recorded
**Fail action:** STAND DOWN Pilot 3; investigate whether PhxRegisterArray
realloc pattern matches std::vector growth.

### 5.D — Phase 4.C Pilot 5 (Class A delete) per-field gate
**Trigger:** before each of the 5 Class A field deletions.
**Pass criteria:**
- Pre-deletion grep: zero remaining C++ direct accesses to the field
  outside of `state_.<field>` reads
- Mutator-site enumeration matches inventory in §2
- JIT_DCHECK on equality of `state_.<field>` vs C++ mirror passes
  during full test suite
**Fail action:** keep the C++ mirror until grep returns clean; do not
delete on incomplete migration.

### 5.E — Phase 4.D dispatch loop entry gate
**Trigger:** before translate() conversion commits begin.
**Pass criteria:**
- Phase 4.A-C all landed (no C++ HirBuilder state remains except
  TranslationContext + dispatch)
- Cumulative C++ delete ≥ 8,500 LOC (75% of pre-Phase-4.D target of
  11,319 LOC excluding bridge)
- Golden-output capture of representative HIR for 5+ functions BEFORE
  conversion (per §6 decision #5 acceptance criteria)
**Fail action:** if cumulative <8,500 LOC, the bottom-up sequencing
assumption is failing; theologian re-spec.

---

## 6. Decision points (Phase 4 prerequisite — supervisor input)

**Status:** all 5 below were ACCEPTED per supervisor 2026-04-27T07:31:48Z
in the original §8.4 of the parent tier7 spec. Recorded here as
canonical reference; no further input needed unless re-opened.

1. **Phase 4 entry trigger:** PARALLEL — Phase 4.A starts immediately
   (zero class-state dep) per supervisor 07:31:48Z.

2. **Pilot ordering:** 3 → 4 → 5 (bridges first, then class-state
   cleanup, then dispatch). Out-of-order Pilot 5 first leaves more
   surface area for silent-drift bugs during Pilot 3 development.

3. **TranslationContext disposition:** ELIMINATE template emit* (ZERO
   C++ admits no template residue per terminal goal); replace with
   PhxTranslationContext + factory pattern + free-function emit
   dispatch.

4. **compiler.cpp caller rewire:** DEFER to Phase 4.D (single 1-line
   change at compiler.cpp:236; low cost when builder.cpp shim is being
   shrunk anyway).

5. **Phase 4 acceptance criteria:** ALL of:
   - PhxHirBuilderState contains 100% of HIRBuilder state (zero C++
     duplicates)
   - builder.cpp ≤ 200 LOC (thin entry-point wrapper or zero per
     decision #4)
   - hir.cpp + printer.cpp + inliner.cpp + preload.cpp + blme +
     hir_instr_c_verify all converted
   - hir_c_api.cpp HIR-internal-only bridges deleted (~600-1,000 LOC
     reduction; full dissolution gated on Phase 5+)
   - Per-commit benchmark gate ≥1.0x parity (Alex hard-floor)
   - Same-session ABBA ≤5% drop (CLAUDE.md threshold)
   - Per-bench floor: no single bench drops >20% same-session
   - Wiring gate verifies sole-path equivalence per
     `gate_phoenix.sh --wiring`
   - ARM64 dual-arch tree-match clean
   - **Net LOC delta:** ≥10,000 LOC C++ deleted minus _c.c growth
     (per `docs/post-phase3d-pure-c-roadmap.md` Phase 4 target =
     14,051 LOC C++ delete ≥ corresponding _c.c growth; net should be
     strongly negative)

---

## 7. Risk register (Phase 4-specific)

| Risk | Severity | Falsifier / Mitigation |
|---|---|---|
| Class A duplicate-state silent drift during Pilot 5 | MEDIUM | Per-field migration commit; grep before+after for residual `code_`/`preloader_`/etc. C++-side reads; JIT_DCHECK on equality of state_ vs C++ mirror until C++ field deleted |
| translate() dispatch loop > 600 LOC, single largest method to convert | HIGH | Sub-divide by opcode-handler group: load*/store* / arith / control-flow / call / return / except / yield. Per-group commit. |
| Perf regression in inline_cache / deopt | MEDIUM | Per-commit ABBA discipline (existing Phase 3D protocol) + golden-output coverage extension to runtime |
| TempAllocator.cache_ vector growth pattern: ABBA-sensitive (allocation count) | MEDIUM | Pilot 3 ABBA before-after MUST run on workload with 1000+ register allocations (e.g. simplify_c benchmark suite); reject if `phx_register_array` realloc count > std::vector growth count by >2× |
| TranslationContext template emit* elimination breaks existing C body callers | LOW | All existing C bodies already use `phx_tc_emit` + factory pattern; template emit* is C++-ergonomics-only; deletion is mechanical |
| compiler.cpp caller-rewire requires compiler.cpp itself to be touched (Phase 8 territory) | LOW | 1-line change; defer to compiler.cpp's own conversion in Phase 7+ if needed; keep thin C++ buildHIR shim until then |
| hir_c_api.cpp partial-dissolution accidentally deletes bridge with cross-area caller | MEDIUM | Grep ALL Python/jit `*.cpp` (not just hir/) before each bridge delete; per-bridge per-commit pattern |
| Inliner.cpp depends on caller's HirBuilder.temps_ — Phase 4.B blocks on Phase 4.C Pilot 3 | LOW | Phase 4.B (inliner.cpp) sequence after Pilot 3 lands; spec'd in §4.3 |
| **BLME-too-easy: Phase 4.A pilot won't validate Class B path or ≥80% LOC target** (pythia #186) | MEDIUM | §5 intermediate checkpoint gates close this gap; 5.A/5.B catch under-delivery; 5.C catches Pilot 3 perf; 5.E catches dispatch sequencing failure |

---

## 8. Falsifiers (re-spec triggers)

If any of the following occur, the spec is wrong and re-spec is
required:

1. **Pilot 3 (temps_) ABBA shows >5% same-session geo-mean drop OR
   >20% single-bench drop.** Indicates `std::vector<Register*>` growth
   has perf-load-bearing properties (likely amortized doubling vs Phx
   linear-then-double); migrate-arm needs different container choice.
   **Checkpoint:** §5.C entry gate.

2. **Phase 4.A first-file (hir.cpp) takes >100 commits OR >1 week.**
   Indicates the "low-state" categorization was wrong; HIRBuilder
   coupling is deeper than expected. Re-scope.
   **Checkpoint:** §5.A close gate.

3. **Bridge dissolution in Phase 4.E deletes <200 LOC.** Indicates
   most bridges are cross-area (LIR/codegen-consumer-only), and the
   dissolution-bonus framing in
   `docs/post-phase3d-pure-c-roadmap.md` §3 Phase 4 was over-counted.
   Re-estimate Phase 4 LOC target downward.
   **Checkpoint:** §5.E entry gate (gates on cumulative ≥8,500 LOC).

4. **translate() conversion produces a regression that ABBA cannot
   detect (functional-only).** Indicates wiring-gate coverage is
   insufficient for dispatch-loop sole-path verification. Extend
   wiring gate before continuing.
   **Checkpoint:** §5.E entry gate (golden-output capture required).

5. **TempAllocator + OperandStack containers' Phx replacements show
   measurable allocation overhead in profile.** Indicates the Phx
   pattern (data + count + capacity + realloc) imposes a genuine cost
   vs std::vector / std::stack. Pythia #103 cathedral-scaffold class
   (validate keep-arm vs migrate-arm post-hoc per pilot).
   **Checkpoint:** §5.C entry gate.

---

## 9. Cross-link

- Roadmap parent: `docs/post-phase3d-pure-c-roadmap.md` §3 Phase 4
- Tier 8 precedent: `docs/tier8-class-b-cport-migrate-arm-spec.md`
  (pilots 1+2 EXECUTED; pilots 3+4 are §4.C of this spec)
- Tier 7 sibling: `docs/tier7-phase3-hirbuilder-state-extraction-spec.md`
  (Phase 3 EXECUTED 2026-04-23; foundation this spec extends)
- Class A duplicate-state delete: `feedback_grep_before_counts.md`
  (count-discipline before each Pilot 5 mutator-site enumeration)
- Per-bench gate: `feedback_benchmark_protocol.md` +
  `feedback_benchmark_save.md` (every Pilot N ABBA must save full
  per-benchmark breakdown to docs/benchmarks/)
- Wiring gate: `feedback_gate_phoenix_wiring_bug.md` (use manual
  force_compile suite for cfg/RPO sole-path verification given known
  --wiring step never runs bug)
- Bridge-dissolution gate: existing precedent from instr_effects.cpp
  Phase A deletion (deletion gate G1-G5 per
  `project_cpp_deletion_gate.md`)
- Falsifier discipline: `feedback_class_of_bug_audit.md` (per-field
  consumer-semantic enumeration during Pilot 5 Class A delete)
- BLME pilot precedent (Phase 4.A first file, in flight 2026-04-27):
  commit 256984d4fd (header-only batch 1 PUSH AUTHORIZED 08:30:45Z),
  body port + sole-path swap (Commit 2) staged
