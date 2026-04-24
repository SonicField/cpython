# Tier 8 — Class B C-port Migrate-Arm Spec

**Status:** PRE-FILED ahead of Phase 3 closure summary per pythia #103
2026-04-24 + supervisor 2026-04-24T00:59:39Z hybrid disposition. Phase
3 closure GATED on this spec landing.

**Owner:** theologian (spec) + generalist (impl when scheduled).

**Estimated cost:** 5-10 sessions per pilot (analogous to Tier 7
builder.cpp baseline). Single-pilot validation, not full migration of
all Class B.

**Pre-Tier-8 prerequisite:** Phase 3 §5 forcing-decision keep-arm
fully validated (5/5 Class B members closed via Class-B-kept or
deleted-dead).

---

## 1. Scope

### 1.1 In-scope

Validate §5 forcing-decision migrate-arm by porting at least ONE
HirBuilder Class B field from C++ container (kept-behind-opaque-pointer
in Phase 3) to pure C container (PhxArray/PhxHashTable/equivalent).
Substantive product: corresponding `_cpp` bridges + opaque-pointer wrap
ELIMINATED. Honest test of the keep-vs-migrate forcing-decision
mechanism vs satisfying §5 by emptying the test set (per pythia #103
critique).

### 1.2 Out-of-scope

- Porting ALL 4 live Class B containers (multi-Tier-8 workstream;
  pilot first, then per-container case-by-case)
- Non-HirBuilder C++ containers (Tier 9+)
- TranslationContext (callee class, separate)
- Re-architecting PhxArray API beyond what pilot needs

---

## 2. Pilot candidate selection

### 2.1 Candidate framework

| Field | Type | Pilot suitability | Reasoning |
|---|---|---|---|
| `exception_table_` | `std::vector<ExceptionTableEntry>` | **RECOMMENDED** | POD-equiv per Phase 3 spec §2.2; 5 POD fields (BCOffset×3 + int + bool); PhxArray<ExceptionTableEntry> port trivial; 4 existing `_cpp` bridges → 0 |
| `static_method_stack_` | `jit::Stack<Register*>` | Deferred | Stack template needs LIFO C-port; Register* opaque pointers safe; only 1 _cpp bridge to eliminate (lower validation density) |
| `block_map_` | `unordered_map<BCOffset, BasicBlock*>` + secondary | Deferred | Hash table C-port is its own multi-session standalone workstream |
| `temps_` | `TempAllocator` class | Deferred | Allocator state machine port; touches 73 C-side caller sites (would re-touch Phase 3 Batch 6 work) |
| `pending_b2_blocks_` | DELETED Phase 3 Batch 3 | N/A | dead-state; no port needed |

### 2.2 Pilot recommendation: `exception_table_`

Concrete migration shape:
- Port `std::vector<ExceptionTableEntry>` → `PhxArray<ExceptionTableEntry>`
  (or equivalent C array struct)
- Port `ExceptionTableEntry` C++ struct → C struct in
  `builder_state_c.h` (already POD-equiv per Phase 3 spec §2.2)
- Eliminate 4 `_cpp` bridges added Phase 3 Batch 2:
  `hir_builder_state_exception_table_push_cpp` +
  `_size_cpp` + `_entry_cpp` + `_find_exception_handler_c`
- Replace bridge access with PhxArray direct field-access via
  `PhxHirBuilderState.exception_table_phx`
- C-side callers (`parseExceptionTable` C body +
  `findExceptionHandler` C body) directly manipulate PhxArray
- C++ side accesses via opaque pointer for any remaining shim sites
  during the port (zero post-port)

---

## 3. Migration strategy

### 3.1 Step-by-step

1. Confirm PhxArray<T> API supports needed ops (push, size, at,
   linear iterate). Extend if gaps per existing PhxArray usage in
   refcount_pass C-port.
2. Add `ExceptionTableEntry` as C struct in `builder_state_c.h`
   (mirror POD layout from `builder.h:625-631`).
3. Add `PhxHirBuilderState.exception_table_phx` field
   (`PhxArray<ExceptionTableEntry>*` or equivalent).
4. Initialize PhxArray field at `PhxHirBuilderState.create`; cleanup
   at destroy.
5. Port `parseExceptionTable` C body to populate PhxArray via
   `PhxArray_push` instead of bridging to `push_cpp`.
6. Port `findExceptionHandler` C body to read PhxArray via
   `PhxArray_size` + `PhxArray_at` linear scan instead of bridging
   to `entry_cpp` / `find_c`.
7. Delete 4 `_cpp` bridges (Phase 3 Batch 2 surface).
8. Delete `std::vector<ExceptionTableEntry> exception_table_` field
   from C++ `HIRBuilder` (builder.h:632).
9. Delete C++ `HIRBuilder::parseExceptionTable` +
   `findExceptionHandler` shims; HIRBuilder no longer touches
   exception_table_ at all (PhxHirBuilderState owns it pure-C).

### 3.2 PhxArray prerequisite check

Existing PhxArray usage (refcount_pass C-port) covers:
- `PhxArray_push(arr, &elem)` — single-element push
- `PhxArray_size(arr) → size_t`
- `PhxArray_at(arr, i) → T*` (or equivalent index access)

If `PhxArray<T>` template is generic enough for `ExceptionTableEntry`
(POD struct), pilot uses existing API as-is. If gaps surface, file
W47 PhxArray-extension spec before pilot Step 1.

---

## 4. Test plan / falsifier strategy

### 4.1 Differential JIT_DCHECK during port

Per `project_wiring_methodology.md`: each port step embeds
JIT_DCHECK comparing C-port output (via PhxArray) vs C++ baseline (via
std::vector). Detects divergence at compile time during the port.
Standard pattern from Phase A C-ports.

### 4.2 W45 §3.5 fixture coverage carry-over

Existing W45 §3.5 Fixture 3 (ExceptionTableEntry depth field rename)
already protects against field-layout drift. Post-port: amend Fixture
3 to mutate the C-side `ExceptionTableEntry` struct in
`builder_state_c.h` (not the deleted C++ struct in builder.h).
Protection class preserved.

### 4.3 Bridge-deletion verification

Post-port, add `scripts/verify_no_exception_table_bridges.sh` (or
extend W33 zero-bridge verifier): assert ZERO matches for grep
`hir_builder_state_exception_table_*_cpp` in production paths. Fails
gate if any of the 4 deleted bridges has a remaining caller.

### 4.4 Existing gates carry-over

- W22 golden trip-wire (codegen output comparison)
- 7-test Phoenix JIT suite + per-bench floor (spec §6 #9 amendment)
- ARM64 dual-arch tree-match per CLAUDE.md
- §3.5 BUILD MODE per push (touched-files heuristic)
- pydebug refleak gate

---

## 5. Acceptance criteria

Tier 8 pilot closure requires ALL:

1. `exception_table_` migrated to PhxArray<ExceptionTableEntry>;
   `std::vector` field deleted from `HIRBuilder` (builder.h:632).
2. `ExceptionTableEntry` C struct lives in `builder_state_c.h`;
   C++ struct deleted (builder.h:625-631).
3. 4 `_cpp` bridges deleted (push_cpp + size_cpp + entry_cpp +
   find_c). Bridge-deletion verifier (§4.3) passes.
4. `parseExceptionTable` + `findExceptionHandler` C bodies use
   PhxArray direct (no bridges).
5. C++ `HIRBuilder` shims for these two methods deleted
   (HIRBuilder no longer has exception_table_ field or accessors).
6. Differential JIT_DCHECK passes during port + post-port
   self-consistency.
7. 7-test Phoenix JIT suite + W22 cluster + ARM64 dual-arch
   tree-match clean.
8. Per-commit benchmark gate per Phase 3 spec §6 #9 amendment
   (geo-mean ≥1.0x + ≤5% drop + no single bench >20% drop + no
   single bench <0.5x absolute).
9. W45 §3.5 Fixture 3 amended to target C-side struct; still
   triggers build-fail.
10. **Substantive C++-line burndown:** Phase 3 added +257L (B1-B6).
    Tier 8 pilot must net-subtract enough that Phase 3+Tier-8
    cumulative ≤ +0L (validates 'transitional foundation cost paid
    back' framing, not 'permanent +257L scaffold').

Estimated subtraction from pilot:
- 4 bridge impl bodies (~80L total, builder.cpp/builder_state_c.c)
- 4 bridge decls (~20L, builder_state_c.h)
- C++ struct + field + 2 method shims (~50L, builder.h + builder.cpp)
- Total estimate: ~-150L (could bring B1-B6 +257L closer to ~+107L
  cumulative; full Tier-8 across all 4 Class B containers projects
  net-negative)

---

## 6. Open questions

1. **PhxArray API sufficiency.** Spot-checked against refcount_pass
   usage; full audit deferred to pilot Step 1.
2. **Compatibility shim window.** During port (steps 5-8), C++ may
   still need read access to PhxArray for unmigrated callers. Spec
   defers shim mechanism to pilot Step 4 implementation; default:
   transient `PhxArray_at` bridge for C++-side that gets deleted at
   step 8.
3. **Pilot vs full migration.** Spec scopes ONE pilot only. Post-pilot
   evaluation: does the keep-vs-migrate cost-benefit justify porting
   all 4 Class B? Or does pilot-only suffice as §5 migrate-arm
   validation? Defer to post-pilot supervisor disposition.

---

## 7. Cross-link

- Pythia framing: pythia #94 (3) 2026-04-23 + pythia #103
  2026-04-24 ('scaffold becomes city')
- Supervisor authorization: 2026-04-24T00:59:39Z hybrid disposition
- Pattern precedent: refcount_pass C-port (PhxArray usage), W26
  fold-into-C
- Methodology: `project_wiring_methodology.md` (differential
  JIT_DCHECK)
- Sibling falsifiers: W45 §3.5 (Fixture 3 ExceptionTableEntry
  protection), W33 zero-bridge verifier, W42 refcount-correctness
- Phase 3 spec: `docs/tier7-phase3-hirbuilder-state-extraction-spec.md`
- Tier 7 spec: `docs/tier7-runtime-burndown-spec.md`
- MEMORY.md terminal goal: 'ZERO C++ (no C++ compiler needed)' — Tier
  8 is the first substantive step toward this goal beyond opaque-
  pointer wrapping.
