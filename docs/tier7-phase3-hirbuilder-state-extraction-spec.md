# Tier 7 Phase 3 — HirBuilder Class State Extraction Spec

**Status:** Phase 3 EXECUTED + Phase 4 DRAFT (Phase 4 prerequisite per
post-phase3d-pure-c-roadmap.md). Originally authored 2026-04-23
per pythia #89 + supervisor 2026-04-23T21:17:25Z. Phase 4 section
(§8 onward) appended 2026-04-27 per supervisor 2026-04-27T07:24:17Z
("HirBuilder Tier-7 state-extraction spec → ... (Phase 4
prerequisite)") following Alex direction D-1777270945 (bug-(ii)
parked, team back on roadmap to ZERO C++).

**Owner:** theologian (this spec) + generalist (implementation).

**Estimated cost:**
- Phase 3: 5-10 sessions per §2.2 estimate. EXECUTED — Phase 3 closed
  2026-04-23 with §5 keep-arm validated; Tier 8 pilots A+B EXECUTED
  2026-04-24/25 with §5 migrate-arm validated for 2 of 4 Class B
  containers (`exception_table_`, `block_map_`).
- Phase 4: 80-120 commits, 2.5-3.5 weeks per
  `docs/post-phase3d-pure-c-roadmap.md` Phase 4 estimate.

**Pre-Phase-3 prerequisite:** Phase 1 ORIGINAL SCOPE 100% COMPLETE
(23 emit-methods/helpers eliminated, builder.cpp 4900→4732, -257L
cumulative, 16 pushes today). Pythia #89 spawn-vs-drain-on-Phase-3
concern addressed by this spec.

**Pre-Phase-4 prerequisite (NEW):** This spec (§8 onward) — Phase 4
launch GATED on §8 review + theologian-authored extraction
recommendation per supervisor 2026-04-27T07:24:17Z. Phase 4 begins
Phase A (low-state HIR files: hir.cpp / printer.cpp / blme / verify)
in parallel with §8 review per `docs/post-phase3d-pure-c-roadmap.md`
risk-mitigation #1 ("start with low-state files while spec is
drafted").

---

## 1. Scope

### 1.1 In-scope

Convert HirBuilder C++ class state + algorithmic methods to pure C
via state-struct extraction. Preserve external interface (callers in
compiler.cpp + tests) via thin C++ shim that delegates to
`hir_builder_state_c` struct + C functions.

### 1.2 Out-of-scope

- Emit-methods (Phase 1 burndown — 100% COMPLETE)
- Inner Function classes (PreloaderManager, etc.) — separate workstream
- TranslationContext (callee class, separate scope)
- compiler.cpp call sites — minimum-touch interface preservation

---

## 2. Architectural state inventory

### 2.1 HirBuilder class data members (builder.h:603-683)

| Member | Type | Purpose | Extraction class |
|---|---|---|---|
| `code_` | `PyCodeObject*` | input code object | A — opaque pointer |
| `block_map_` | `BlockMap` | block lookup map | B — heavy C++ container |
| `exception_table_` | `std::vector<ExceptionTableEntry>` | parsed handlers | B — STL container (CLOSED via Batch 2 1343895045 Class B-kept) |
| `pending_b2_blocks_` | `std::vector<PendingBlock>` | (DEAD per Batch 3 grep — W26-era residual) | DELETED Batch 3 |
| `preloader_` | `const Preloader&` | preload info | A — opaque ref |
| `current_func_` | `Function*` | current HIR fn | A — opaque pointer |
| `temps_` | `TempAllocator` | register allocator | B — value-type C++ class |
| `func_` | `Register*` | function reg | A — opaque pointer |
| `kwnames_` | `Register*` | KW_NAMES constant | A — opaque pointer |
| `static_method_stack_` | `OperandStack` | static method stack | B — value-type C++ class |

**Class A (5 members):** opaque pointers/refs. Direct void* equivalent
in C struct. ZERO extra bridges.

**Class B (5 members nominally, 4 live post Batch 3 dead-state delete):**
heavy C++ containers/value-types. Need either (1) keep on C++ side via
opaque pointer, or (2) full C-port of container to PhxArray/equivalent.

**AMENDMENT per pythia #93 2026-04-23 (§2.1 errata + grep-writers
discipline):** Batch 3 Step A discovered pending_b2_blocks_ was DEAD
STATE (zero writers; W26-era residual from emitInlineExceptionMatch
refactor). The original §2.1 inventory categorized by TYPE
(`std::vector<PendingBlock>`) without grepping for writers. Standing
discipline: any Class B candidate MUST grep writers
(`push_back\|emplace_back\|insert\|operator\[\]` mutation) BEFORE
Step A scoping. Type-only categorization is insufficient — dead state
masquerades as Class B until grep-verified. Apply to remaining Class B
candidates: block_map_, temps_, static_method_stack_.

### 2.2 HirBuilder internal struct types (builder.h:608-636)

- `ExceptionTableEntry` (5 fields, POD: BCOffset×3 + int + bool)
  → C struct trivial
- `PendingBlock` (2 fields: BasicBlock* + FrameState)
  → C struct + opaque pointers
- `SimpleExceptInfo` (3 fields: int + PyObject* + BCOffset)
  → C struct trivial

All 3 internal structs are POD-equivalent and can be moved to C as
plain structs.

### 2.3 HirBuilder algorithmic methods

Multi-method (at least): translate(), checkTranslate(),
parseExceptionTable(), findExceptionHandler(),
getSimpleExceptInfo(), emitInlineExceptionMatch(),
emitCallExceptionHandler(), advancePastYieldInstr().

Plus public Run() / Compile() entry points (callers in compiler.cpp).

Total: ~10-15 algorithmic methods (post Phase 1 emit-method removal).

---

## 3. Extraction strategy

### 3.1 State struct definition (proposed)

`Python/jit/hir/builder_state_c.h`:

```c
typedef struct PhxHirBuilderState {
    void *code;                /* PyCodeObject* */
    void *block_map;           /* opaque BlockMap* (Class B kept C++) */
    void *exception_table;     /* opaque std::vector<...>* */
    void *pending_b2_blocks;   /* opaque std::vector<...>* */
    const void *preloader;     /* const Preloader& */
    void *current_func;        /* Function* */
    void *temps;               /* opaque TempAllocator* */
    void *func;                /* Register* */
    void *kwnames;             /* Register* */
    void *static_method_stack; /* opaque OperandStack* */
} PhxHirBuilderState;
```

ZERO new C-API surface vs existing builder_emit_c.c bridges. Class A
members are direct void* fields; Class B members are opaque pointers
to C++-owned objects (lifetime via C++ ctor/dtor through bridge
functions).

### 3.2 C++ shim pattern

`builder.h` keeps thin C++ shim:

```cpp
class HIRBuilder {
public:
    HIRBuilder(PyCodeObject* code, ...);
    std::unique_ptr<Function> Run();
private:
    PhxHirBuilderState state_;  // owned, opaque to C++ except via bridges
};
```

Method bodies in builder.cpp become 1-line delegations to
`hir_builder_state_*_c` functions taking `&state_`.

### 3.3 Bridge inventory (estimate)

- `hir_builder_state_create(code, preloader)` — ctor
- `hir_builder_state_destroy(state)` — dtor
- `hir_builder_state_run(state) -> Function*` — main entry
- `hir_builder_state_check_translate(state) -> bool` — checkTranslate
- `hir_builder_state_parse_exception_table(state)` — parseExceptionTable
- `hir_builder_state_find_exception_handler(state, off) -> entry*` — findExceptionHandler
- `hir_builder_state_get_simple_except_info(state, handler, &info) -> bool`
- `hir_builder_state_emit_inline_exception_match(state, ...)`
- `hir_builder_state_emit_call_exception_handler(state, ...)`
- `hir_builder_state_advance_past_yield_instr(state, tc)`
- Plus container ops for Class B members (~5-10 bridges):
  block_map_lookup, exception_table_push, pending_b2_blocks_push, etc.

Total estimate: ~15-20 new bridges. Within ≤5/batch W25b discipline
if batched as 4-5 batches.

---

## 4. Test plan / falsifier strategy

### 4.1 Per-batch differential JIT_DCHECK

Same pattern as Phase A C-ports (per memory
`project_wiring_methodology.md`): for each method ported, embed
JIT_DCHECK comparing C-port output vs C++ baseline output on
parallel run. Detect divergence at compile time.

### 4.2 W45 fixtures for new bridges

Per W45 §2.2: add fixture for each new state-struct bridge to
`scripts/w45_bridge_drift_falsifier.sh`. Insert-sentinel-param
mutation must trigger build failure at C++ shim call site.

### 4.3 Phase-3-specific falsifier (NEW)

Add `scripts/phase3_state_extraction_falsifier.sh` (analogous to
W45):
- Mutate one PhxHirBuilderState field (reorder, type-change)
- Verify build OR runtime test detects divergence
- Restore field; build PASS

### 4.4 Existing gates (preserved)

- W21 golden trip-wire (codegen output comparison)
- W44 DO-NOT-USE caller gate
- 7-test Phoenix JIT suite (controlflow + comparisons + autocompile +
  await + repro_min3 + partial_conversions + W44)
- ARM64 dual-arch tree-match per CLAUDE.md
- pydebug refleak gate (per memory `feedback_no_workarounds.md`)

---

## 5. Open questions / decision points

1. **Class B members C++-kept-vs-C-port choice.** State spec proposes
   keeping (block_map_, exception_table_, pending_b2_blocks_, temps_,
   static_method_stack_) as opaque-C++-owned. Alternative: full
   C-port of containers via PhxArray. Cost vs benefit per per-member
   complexity. Defer per-member call to per-batch Step A.

   **AMENDMENT per pythia #92 2026-04-23:** Batch 2 Class B disposition
   decision is FORCED, not deferred — Batch 1's PhxHirBuilderState
   shape (Class A subset only) becomes precedent. To avoid
   Class-A-only-shape calcification (architecture-by-accretion risk),
   Batch 2 MUST re-shape struct if needed before Batch 5 lands.
   Otherwise per-method Class B disposition (Batches 3-5) becomes
   constrained by an unintended Batch 1 design choice.

   **CLOSURE AMENDMENT per pythia #103 2026-04-24 + supervisor 2026-
   04-24T00:59:39Z hybrid disposition:** Phase 3 closed with all 5
   Class B members disposed via keep-bias (4 closed Class-B-kept +
   1 deleted dead-state). The `_cpp` bridge surface (~6 _cpp + 2 _c
   algorithmic + 1 init = 9 extern surfaces) is the substantive Phase
   3 product. Class B-kept is **TRANSITIONAL DESIGN CHOICE**, NOT
   FINAL — the keep-arm of §5 forcing-decision was validated; the
   migrate-arm (port to pure C) remains untested by Phase 3 alone.

   Earlier draft framing 'Class B-kept is FINAL disposition for
   HirBuilder state' (theologian 23:35:58Z + 23:49:00Z, supervisor
   00:01:35Z + 00:14:45Z) is WITHDRAWN per pythia #103 cathedral-
   scaffold critique. The 'FINAL' framing would calcify the
   transitional state and contradict MEMORY.md ZERO-C++ terminal
   goal.

   Replacement framing: keep-bias is design choice for Phase 3
   tractability (PhxArray-equivalent C-port for std::vector /
   unordered_map / Stack / TempAllocator is multi-session-per-
   container scope); Tier 8 §5 migrate-arm pilot
   (`docs/tier8-class-b-cport-migrate-arm-spec.md`, theologian
   01:01:50Z, supervisor 01:02:46Z ADOPTED) validates the migrate-
   arm by porting `exception_table_` (POD-equivalent per §2.2) to
   PhxArray. Tier 8 acceptance #10 requires net-subtract ≥ Phase 3
   foundation cost (+257L) across all Class B containers (full
   migration projects net-negative).

   Phase 3 outcome: foundation laid + bridge surface enumerated +
   §5 keep-arm fully validated. Phase 3 is not end-state for
   HirBuilder; Tier 8 pilot is the substantive next-step toward
   ZERO-C++.

2. **builder.cpp shim file size.** Phase 3 may leave a thin
   builder.cpp shim (~100-200 lines of C++ delegations). Acceptable
   per Tier 7 spec §1.2 intentional residue framing? Or aim for full
   C migration with compiler.cpp caller-rewrite?

3. **HirBuilder ownership semantics.** `std::unique_ptr<Function>
   Run()` returns owned Function. C-side equivalent needs ownership
   discipline (PhxArray-style? caller responsibility?). Defer per
   per-batch.

---

## 6. Acceptance criteria

Phase 3 closure requires ALL:

1. PhxHirBuilderState struct landed + C++ HIRBuilder class uses it
   exclusively for state.
2. All ~10-15 HirBuilder methods converted (or honestly framed as
   intentional residue per Tier 7 §1.2).
3. ZERO new bridges beyond the ~15-20 estimate (per W25b ≤5/batch
   budget across 4-5 batches).
4. W45 fixtures cover every new bridge.
5. Differential JIT_DCHECK passes for all C-ports.
6. Full 7-test Phoenix JIT suite + W44 gate + W22 cluster + dual-arch
   tree-match clean.
7. compiler.cpp callers unaffected (interface preservation
   verified).
8. builder.cpp final size ≤500L (thin shim) OR ≤200L (deleted)
   depending on Class B disposition.
9. **Per-commit benchmark gate** (per CLAUDE.md + supervisor
   2026-04-23T22:59:54Z addendum): every Phase 3 commit runs
   4-bench (`--compile=auto --reps=3`) against pinned baseline.
   Geo-mean must stay ≥1.0x parity floor (per Alex hard-floor
   directive 2026-04-14). Phase 3 touches HirBuilder hot path;
   per-commit benchmark gate is non-optional. Block push if
   geo-mean drop >5% (per CLAUDE.md ≥5% BLOCK threshold).

   **AMENDMENT per pythia #92 2026-04-23:** Cross-session ABBA
   variance is ±7% per memory `feedback_abba_cross_session.md`. Phase
   3 spans many sessions ('multi-week unknown'). Same-session ABBA
   per Phase 3 batch is reliable; cross-session comparison is not.
   At session-start, re-baseline 4-bench against pinned baseline
   before any Phase 3 batch lands; within-session compare commit-to-
   commit. Cross-session geo-mean drift up to 7% is noise, not
   regression — block only if same-session drift exceeds 5%.

   **AMENDMENT per pythia #93 2026-04-23 (per-bench floor):**
   Geo-mean alone can mask single-bench regressions when
   high-multiple benches dominate (Batch 2 fibonacci 2.29x +
   nqueens 1.63x absorb gen_simple-class drops silently). Per-commit
   gate now requires ALL of:
   - Geo-mean ≥1.0x absolute parity floor (per Alex hard-floor)
   - Geo-mean drop ≤5% same-session vs prior commit (CLAUDE.md)
   - **No single bench drops >20% from prior commit (same-session)**
   - **No single bench drops below 0.5x absolute parity** (catches
     workload-class degradation that geo-mean smooths over)
   Per-bench thresholds are ABSOLUTE in same-session ABBA only;
   cross-session per-bench drift is noise per pythia #92 amendment.

---

## 7. Cross-link

- Pythia framing: pythia #89 2026-04-23 (Phase 3 launch gating)
- Supervisor commitment: 2026-04-23T21:17:25Z + 2026-04-23T22:57:16Z
- Pattern precedent: W25b/W26 spec docs (state-struct extraction)
- Methodology: `project_wiring_methodology.md` (differential
  JIT_DCHECK)
- Sibling falsifiers: W42 (refcount-correctness verifier), W44
  (DO-NOT-USE caller gate), W45 (bridge-signature drift)
- Memory: `feedback_grep_before_counts.md` (count-discipline at
  spec write-time + cite time)
- Phase 1 close: 16 pushes today, builder.cpp 4900→4732, -257L
  cumulative, all 23 emit-methods/helpers eliminated
- Tier 7 spec §1: docs/tier7-runtime-burndown-spec.md (in-scope file
  list + 3-class load-bearing-infrastructure framing)
- Phase 4 successor framing: `docs/post-phase3d-pure-c-roadmap.md`
  §3 Phase 4 (HIR completion, 14,051 LOC delete target inc. bridge)

---

## 8. Phase 4 — HirBuilder Final Migration to Pure C

**Authored:** 2026-04-27 (theologian) per supervisor 07:24:17Z
("Phase 4 prerequisite") + Alex D-1777270945 (bug-(ii) parked, team
back on roadmap).

### 8.1 State of HirBuilder, post-Tier-8-A+B (measured 2026-04-27)

PhxHirBuilderState struct (`builder_state_c.h:330-339`) currently holds:

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
| `temps_` | `TempAllocator` (C++) | **B-kept** | `hir_builder_state_temps_alloc_stack_cpp` | **73** (2026-04-28 re-grep: 73 in builder_emit_c.c, +1 site in builder.cpp) |
| `func_` | `Register*` | A — mirror lives in state_ | direct via state_.func | indirect via state_ |
| `kwnames_` | `Register*` | A — mirror lives in state_ | direct via state_.kwnames | indirect via state_ |
| `static_method_stack_` | `OperandStack` (`jit::Stack<Register*>`) | **B-kept** | `hir_builder_state_static_method_stack_{push,pop}_cpp` | **5** (2026-04-28 re-grep: 5 in builder_emit_c.c, +3 sites in builder.cpp) |
| `state_` | `PhxHirBuilderState` | (the C struct itself) | n/a | n/a |

**Class A duplicate-state cost:** the 5 mirrored fields exist in BOTH
the C++ class AND PhxHirBuilderState. C++ mutator sites must update
both in lockstep — silent drift on a missed mutator is a latent bug
class. Phase 4 closes this by deleting the C++ duplicates.

**Class B-kept residue:** `temps_` (15 direct C++ accesses + 73 C-side
bridge calls) and `static_method_stack_` (2 direct + 5 bridge) are the
two outstanding migrate-arm targets. Tier 8 pilot 1+2 validated the
migrate pattern; pilots 3+4 are the natural extension.

### 8.2 builder.cpp dispatch surface (measured 2026-04-27)

| Surface | Count |
|---|---|
| `HIRBuilder::*` method bodies | 128 |
| Class A member access sites (current_func_/func_/kwnames_) | 103 |
| `state_.` access sites | 18 |
| `preloader_` access sites | 28 |
| `code_` access sites | 46 |
| Direct `temps_` access | 17 (was 15 at 2026-04-27 spec; +2 drift) |
| Direct `static_method_stack_` access | 3 (was 2 at 2026-04-27 spec; +1 drift) |
| File total LOC | 4,527 (unchanged 2026-04-28; today's bundle 6157a2192b + f23e281158 did not touch builder.cpp) |

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
ergonomics; they're not algorithmically necessary — every call site can
go through `phx_tc_emit(tc, instr)` against a pre-created instruction.

### 8.3 Phase 4 sub-step sequencing (within HIR area)

Phase 4 burndown order (lowest-state-coupling first; avoids stalling on
the unsolved class-state-extraction problem while non-state files
land):

**Phase 4.A — Low-state HIR files (parallel with §8.4 design closure):**
- `hir.cpp` 1,268 LOC — HIR core; mostly free functions + factory bodies
- `printer.cpp` 514 LOC (was 957 in 2026-04-27 spec; W-PRINTER-IMMEDIATES-PORT
  P-2 → P-5c landed -443 LOC by 2026-04-28 push aa6c5703f1) — text
  serializer; pure-formatting, no HirBuilder dep
- `hir_instr_c_verify.cpp` 386 LOC — verifier; thin friend-struct pattern
  (per `feedback_verifier_pattern.md`)
- `builtin_load_method_elimination.cpp` 21 LOC (was 184 in 2026-04-27 spec;
  body essentially eliminated) — pass; no class state
- **Subtotal:** 2,189 LOC remaining (was 2,786 at spec authoring; 597 LOC
  already eliminated via printer + blme work this 2026-04-27/28 window).
  Estimated 20-35 commits remaining (down from 30-50), 3-5 sessions.

  Each file is HIRBuilder-independent and can be converted using
  established Phase 3D patterns (algorithm-body split, JIT_DCHECK
  differential, _c.c port + .cpp delegation). No class-state work.

**Phase 4.B — Medium-state HIR files:**
- `inliner.cpp` 704 LOC — uses HIRBuilder for inlineHIR; depends on
  Phase 4.D class-state migration partially landing
- `preload.cpp` 570 LOC — Preloader class; HirBuilder uses by const-ref
  but Preloader itself is independent
- **Subtotal:** 1,274 LOC, 15-25 commits, 2-3 sessions

  Preload.cpp can land in parallel with Phase 4.A (no HirBuilder
  state dep). Inliner.cpp gates on at least Phase 4.D pilot 3 (temps_
  migration) since `inlineHIR` reuses the caller's temps_.

**Phase 4.C — Class B-kept migrate-arm pilots (analogues of Tier 8 pilots 1+2):**
- **Pilot 3:** `temps_` (TempAllocator) → `PhxTempAllocator` in state_
  - TempAllocator is a value-type with `Environment* env_` + `std::vector<Register*> cache_`
  - Migrate cache_ to PhxRegisterArray (similar to PhxExceptionTable
    pattern: data + count + capacity, doubling)
  - 3 methods to port: AllocateStack, GetOrAllocateStack, AllocateNonStack
  - **15 C++ direct accesses + 73 _cpp bridges → 0 bridges + 73 direct
    state-struct accesses**
  - **Bridge delete: 1** (`hir_builder_state_temps_alloc_stack_cpp`)
  - **Estimated cost:** 5-8 commits, 1 session

- **Pilot 4:** `static_method_stack_` (OperandStack = `jit::Stack<Register*>`)
  → `PhxRegisterStack` in state_
  - Stack of pointers; trivial Phx container (smaller than PhxBlockMap)
  - Direct linear-array LIFO with doubling
  - **2 C++ direct + 5 _cpp bridges → 0 bridges + 7 direct accesses**
  - **Bridge delete: 2** (push_cpp + pop_cpp)
  - **Estimated cost:** 3-5 commits, 0.5 session

- **Pilot 5 (lockstep with 3+4):** delete the 5 Class A C++ duplicates
  (code_, preloader_, current_func_, func_, kwnames_). All access goes
  through state_ exclusively. Removes the silent-drift latent bug
  class.
  - **Estimated cost:** 5-8 commits per field × 5 = 25-40 commits, 3-4
    sessions; each commit migrates 1 field across all C++ access sites
    and verifies via grep + JIT_DCHECK + benchmark gate
  - **Subtotal Phase 4.C:** 33-53 commits, 4.5-5.5 sessions

**Phase 4.D — Dispatch loop + builder.cpp shrinkage:**
- `translate()` dispatch loop conversion to C — the central method
- `buildHIR`, `buildHIRImpl`, `inlineHIR` entries: convert to C bodies
  with thin C++ shims (or direct C if `compiler.cpp:236` caller can be
  rewired)
- `allocateLocalsplus`, `addInitialYield`, `addLoadArgs`,
  `addInitializeCells`, `createBlocks`, `emitInlineExceptionMatch`,
  `emitCallExceptionHandler`, `emitTypeAnnotationGuards`,
  `advancePastYieldInstr` — algorithmic helpers
- TranslationContext C++ struct: delete (PhxTranslationContext
  replaces it; template emit*/emitChecked* methods become free-function
  C dispatch via existing factory C-API)
- TempAllocator C++ class: delete (after Pilot 3)
- BlockCanonicalizer C++ class: convert (uses `temps_` + `PhxPtrArray`;
  scope dependent on Pilot 3)
- **Subtotal:** 4,527 → ≤200 LOC residual, 30-50 commits, 4-6 sessions

**Phase 4.E — hir_c_api.cpp bridge dissolution:**
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
- **Full dissolution** waits for Phase 5+ completion, per the
  roadmap §3 dependency ordering.
- **Subtotal Phase 4.E:** 5-10 commits, 1 session

### 8.4 Decision points (Phase 4 prerequisite — supervisor input needed)

1. **Phase 4 entry trigger:** start Phase 4.A immediately (low-state
   files), in parallel with §8 review? OR gate on §8 acceptance
   first? Recommendation: **parallel** — Phase 4.A files have ZERO
   dep on class-state design and the gate-first option burns calendar
   on review-only work.

2. **Pilot 3 vs Pilot 5 ordering:** does temps_ migrate (Pilot 3)
   land BEFORE the Class A duplicate-deletion (Pilot 5)? Pilot 3
   eliminates 73 _cpp bridge calls; Pilot 5 eliminates 5 silent-drift
   risks. Recommendation: **Pilot 3 → 4 → 5** in that order; bridges
   first, then class-state cleanup, then dispatch loop. Each pilot is
   independently verifiable; out-of-order Pilot 5 first leaves more
   surface area for silent-drift bugs during Pilot 3 development.

3. **TranslationContext disposition:** template emit* methods
   eliminate (replace with C free-function calls + factory) OR keep
   as pure-C++ ergonomics layer? Recommendation: **eliminate**
   because (a) PhxTranslationContext + factory pattern is already
   the C-side equivalent, (b) keeping them creates a dual API that
   resists Phase 4.D builder.cpp shrinkage, (c) ZERO C++ terminal
   goal admits no template residue.

4. **Compiler.cpp caller rewire:** `compiler.cpp:236` calls
   `hir::buildHIR(preloader)`. To delete the C++ entry point fully,
   this caller must rewire to `phx_hir_build(state)`. Requires
   `compiler.cpp` to consume a C entry. Defer to Phase 4.D scoping
   OR pre-stage as Phase 4.0? Recommendation: **defer to Phase 4.D**
   — single 1-line change, low cost when builder.cpp shim is being
   shrunk anyway.

5. **Phase 4 acceptance criteria:** propose ALL of:
   - PhxHirBuilderState contains 100% of HIRBuilder state (zero C++
     duplicates)
   - builder.cpp ≤ 200 LOC (thin entry-point wrapper or zero, per
     decision #4)
   - hir.cpp + printer.cpp + inliner.cpp + preload.cpp + blme +
     hir_instr_c_verify all converted (algorithm bodies in _c.c, .cpp
     ≤ delegation shim or deleted)
   - hir_c_api.cpp HIR-internal-only bridges deleted (~600-1,000 LOC
     reduction; full dissolution gated on Phase 5+)
   - Per-commit benchmark gate ≥1.0x parity (Alex hard-floor)
   - Same-session ABBA ≤5% drop (CLAUDE.md threshold)
   - Per-bench floor: no single bench drops >20% same-session
   - Wiring gate verifies sole-path equivalence per
     `gate_phoenix.sh --wiring`
   - ARM64 dual-arch tree-match clean
   - **Net LOC delta:** ≥10,000 LOC C++ deleted minus _c.c growth (per
     `docs/post-phase3d-pure-c-roadmap.md` Phase 4 target = 14,051 LOC
     C++ delete ≥ corresponding _c.c growth; net should be strongly
     negative)

### 8.5 Risk register (Phase 4-specific)

| Risk | Severity | Falsifier / Mitigation |
|---|---|---|
| Class A duplicate-state silent drift during Pilot 5 | MEDIUM | Per-field migration commit; grep before+after for residual `code_`/`preloader_`/etc. C++-side reads; JIT_DCHECK on equality of state_ vs C++ mirror until C++ field deleted |
| translate() dispatch loop > 600 LOC, single largest method to convert | HIGH | Sub-divide by opcode-handler group: load*/store* / arith / control-flow / call / return / except / yield. Per-group commit. |
| TempAllocator.cache_ vector growth pattern: ABBA-sensitive (allocation count) | MEDIUM | Pilot 3 ABBA before-after MUST run on workload with 1000+ register allocations (e.g. simplify_c benchmark suite); reject if `phx_register_array` realloc count > std::vector growth count by >2× |
| TranslationContext template emit* elimination breaks existing C body callers | LOW | All existing C bodies already use `phx_tc_emit` + factory pattern; template emit* is C++-ergonomics-only; deletion is mechanical |
| compiler.cpp caller-rewire requires compiler.cpp itself to be touched (Phase 8 territory) | LOW | 1-line change; defer to compiler.cpp's own conversion in Phase 7+ if needed; keep thin C++ buildHIR shim until then |
| hir_c_api.cpp partial-dissolution accidentally deletes bridge with cross-area caller | MEDIUM | Grep ALL Python/jit `*.cpp` (not just hir/) before each bridge delete; per-bridge per-commit pattern |
| Inliner.cpp depends on caller's HirBuilder.temps_ — Phase 4.B blocks on Phase 4.C Pilot 3 | LOW | Phase 4.B (inliner.cpp) sequence after Pilot 3 lands; spec'd in §8.3 |

### 8.6 Falsifier of this Phase 4 spec

If any of the following occur, the spec is wrong and re-spec is
required:

1. **Pilot 3 (temps_) ABBA shows >5% same-session geo-mean drop OR
   >20% single-bench drop.** Indicates `std::vector<Register*>` growth
   has perf-load-bearing properties (likely amortized doubling vs Phx
   linear-then-double); migrate-arm needs different container choice.

2. **Phase 4.A first-file (hir.cpp) takes >100 commits OR >1 week.**
   Indicates the "low-state" categorization was wrong; HIRBuilder
   coupling is deeper than expected. Re-scope.

3. **Bridge dissolution in Phase 4.E deletes <200 LOC.** Indicates
   most bridges are cross-area (LIR/codegen-consumer-only), and the
   dissolution-bonus framing in `docs/post-phase3d-pure-c-roadmap.md`
   §3 Phase 4 was over-counted. Re-estimate Phase 4 LOC target
   downward.

4. **translate() conversion produces a regression that ABBA cannot
   detect (functional-only).** Indicates wiring-gate coverage is
   insufficient for dispatch-loop sole-path verification. Extend
   wiring gate before continuing.

5. **TempAllocator + OperandStack containers' Phx replacements show
   measurable allocation overhead in profile.** Indicates the Phx
   pattern (data + count + capacity + realloc) imposes a genuine
   cost vs std::vector / std::stack. Pythia #103 cathedral-scaffold
   class (validate keep-arm vs migrate-arm post-hoc per pilot).

### 8.7 Cross-link (Phase 4-specific)

- Roadmap parent: `docs/post-phase3d-pure-c-roadmap.md` §3 Phase 4
- Tier 8 precedent: `docs/tier8-class-b-cport-migrate-arm-spec.md`
  (pilots 1+2 EXECUTED; pilots 3+4 are §8.3 Phase 4.C)
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

---

## 8.8 2026-04-28 measurement refresh + Phase 4 readiness note

**Refresh trigger:** supervisor 2026-04-28T21:06:09Z dispatched theologian
to advance this spec post-bundle 6157a2192b + f23e281158
(W-TSTATE-INTROSPECTION-FRAGILE + W-LIRORIGIN-MULTIBENCH). Refresh
captures HIR-area drift since 2026-04-27 spec authoring.

**Material changes since 2026-04-27 (2026-04-27 → 2026-04-28):**
1. `printer.cpp` 957 → 514 LOC (-443) via W-PRINTER-IMMEDIATES-PORT P-2
   through P-5c (commits a76435ced1 through aa6c5703f1). Substantively
   covers Phase 4.A printer.cpp target.
2. `builtin_load_method_elimination.cpp` 184 → 21 LOC (-163), body
   essentially eliminated. Substantively covers Phase 4.A blme target.
3. `builder.cpp` 4,527 LOC UNCHANGED. Today's bundle did not touch HIR
   area architecture (only frame_asm_c.c + phoenix_asm/x86_64.c +
   benchmark_phoenix.py).
4. `temps_` direct accesses 15 → 17 (+2 drift); `static_method_stack_`
   direct 2 → 3 (+1 drift). Within tolerance; Pilot 3/4 scope estimates
   in §8.3 unchanged.
5. `temps_alloc_stack_cpp` and `static_method_stack_{push,pop}_cpp`
   bridge call counts UNCHANGED (73 + 5 in builder_emit_c.c).

**Implication for Phase 4 sequencing (§8.3):**
- Phase 4.A is MID-EXECUTION not pre-launch (printer.cpp + blme already
  landed via W-PRINTER-IMMEDIATES-PORT workstream which substantively
  matches Phase 4.A scope, even if not formally classified as such).
  Remaining Phase 4.A work: hir.cpp (1,268) + hir_instr_c_verify.cpp
  (386) = 1,654 LOC, ~15-25 commits, 2-4 sessions.
- Phase 4.B/C/D/E unchanged in scope.
- §8.4 decision #1 ("Phase 4 entry trigger: parallel with §8 review?")
  is RESOLVED EMPIRICALLY by W-PRINTER-IMMEDIATES-PORT having already
  landed Phase 4.A targets without explicit Phase 4 entry-gate
  ratification. Recommendation: ratify retroactively as Phase 4.A
  partial-complete + continue with hir.cpp / hir_instr_c_verify.cpp as
  Phase 4.A finishing batch.

**Implication for Phase 4 readiness:**
- Phase 4.A finishing batch (hir.cpp + verify) is the natural next-up
  workstream for ZERO C++ progress per terminal goal. Phase 4.C Pilot 3
  (temps_ → PhxTempAllocator) is the highest-leverage state-extraction
  step (73 _cpp bridge call eliminations + transitional-design closure
  of last Class B-kept residue per §5 closure amendment).
- Either workstream is independently launchable; sequencing-only
  consideration is whether to clear Phase 4.A LOC backlog first
  (continuity with W-PRINTER-IMMEDIATES-PORT cadence) or pivot to
  Phase 4.C Pilot 3 (higher per-commit leverage).

**Decision points still open from §8.4:**
- #1: RESOLVED EMPIRICALLY (above).
- #2: Pilot 3 vs Pilot 5 ordering — recommendation Pilot 3 → 4 → 5
  STANDS (no new evidence to invalidate).
- #3: TranslationContext disposition — recommendation ELIMINATE STANDS.
- #4: compiler.cpp caller rewire — recommendation defer to Phase 4.D
  STANDS.
- #5: Phase 4 acceptance criteria — STANDS.

**No spec amendment beyond this refresh.** Phase 4 architecture is
sound; today's bundle did not surface any HIR-area architectural
constraint requiring re-design. Spec is ready for supervisor review +
Alex disposition on §8.4 decisions #2-5.

**Cross-link 2026-04-28 refresh:**
- Bundle that did not affect Phase 4: 6157a2192b (frame_asm_c.c,
  phoenix_asm/x86_64.c) + f23e281158 (Tools/benchmark_phoenix.py)
- W-PRINTER-IMMEDIATES-PORT execution evidence: `git log Python/jit/hir/printer.cpp --since=2026-04-27`
