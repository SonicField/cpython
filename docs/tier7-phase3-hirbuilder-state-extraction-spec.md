# Tier 7 Phase 3 — HirBuilder Class State Extraction Spec

**Status:** ACTIVE (pre-launch). Authored 2026-04-23 per pythia #89
2026-04-23 + supervisor 2026-04-23T21:17:25Z commitment ("Phase 3
launch GATED on theologian-authored spec doc landing first") +
supervisor 2026-04-23T22:57:16Z disposition.

**Owner:** theologian (this spec) + generalist (implementation).

**Estimated cost:** 5-10 sessions per Tier 7 spec §2.2 builder.cpp
estimate. Multi-week unknown.

**Pre-Phase-3 prerequisite:** Phase 1 ORIGINAL SCOPE 100% COMPLETE
(23 emit-methods/helpers eliminated, builder.cpp 4900→4732, -257L
cumulative, 16 pushes today). Pythia #89 spawn-vs-drain-on-Phase-3
concern addressed by this spec.

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
