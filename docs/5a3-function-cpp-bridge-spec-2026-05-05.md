## Phase 5.A3 — function.cpp Bridge Spec

**Author:** theologian (W-PERF owner; spec-template per project_bridge_spec_template.md)
**Date:** 2026-05-05 (post-Cfinal-push d114224a4d)
**Source:** Python/jit/lir/function.cpp (220 LOC) + Python/jit/lir/function.h (73 LOC)
**Target:** Python/jit/lir/function_impl.c (existing 161 LOC, extend to ~350 LOC) + 2 new substrate headers
**Class:** STRUCT-tier (per D-1777621209 5.A re-scope); non-M3 wave per W-PERF §3b literal spec (~6 caller sites for Function methods)
**Falsifier class:** F3b precedent applies (non-M3); per-commit gate + 24-bench cap-check at chain close
**Estimated commits:** 5-7 (3 substrate-add + 4 helper-port + 1 copyFrom-port + 1 wire-up + 1 cleanup)

---

## §0 — Scope + non-goals

**In scope:**
- Port the 4 anonymous-namespace helpers (copyIndirect, copyOperand, copyInput, connectLinkedOperands) + deepCopyBasicBlocks to C as `static` functions in function_impl.c.
- Port Function::copyFrom body to C as `lir_function_copy_from_impl` (extern-C surface stays at lir_function_copy_from).
- Add 2 new substrate headers: `Python/jit/hir/phx_int_ptr_map.h` (int→void*) + `Python/jit/hir/phx_ptr_int_map.h` (void*→int).
- Delete C++ residue from function.cpp (Function::~Function delegation, copyFrom body, allocateBasicBlock + allocateBasicBlockAfter + sortBasicBlocks delegations) once C path is sole-path verified.

**Out of scope:**
- function.h struct layout changes (already C-shape since Phase B1; no reshape needed).
- block.cpp 9 thin-wrapper shims (deferred to 5.E per D-1777621209).
- generator.cpp (3718 LOC, deferred to 5.D).
- Inliner consumer rewrites beyond the lir_function_copy_from extern-C call (already in inliner_c.c:417).

**Move ≠ rewrite (per project_bridge_spec_template):** the helpers + copyFrom are STRUCTURAL equivalents only. No algorithmic changes. UnorderedMap → PhxIntPtrMap/PhxPtrIntMap is a substrate substitution (open-addressed hash map for std::unordered_map / phmap::flat_hash_map); no semantic divergence permitted.

---

## §1 — Substrate inventory + decisions

### §1.1 — UnorderedMap usages (3)

Three site-specific usages in function.cpp:

| Site | Type | C substrate (per supervisor 13:16:01Z option A) |
|------|------|--------------------------------------------------|
| `block_index_map` (copyFrom local; threaded to copyOperand/copyInput/deepCopyBasicBlocks) | `UnorderedMap<int, BasicBlock*>` | `PhxIntPtrMap` (new) |
| `instr_refs` (deepCopyBasicBlocks local; threaded to copyOperand/copyInput/copyIndirect/connectLinkedOperands) | `UnorderedMap<LinkedOperand*, int>` | `PhxPtrIntMap` (new) |
| `output_index_map` (deepCopyBasicBlocks local; passed to connectLinkedOperands) | `UnorderedMap<int, Instruction*>` | `PhxIntPtrMap` (new) |

**Decision per supervisor 13:16:01Z Q1 disposition:** option A (substrate-add). +1-offset cast workaround rejected per feedback_no_workarounds + W-PERF §3b spirit. New substrate matches PhxPtrMap pattern (open-addressed, power-of-2 capacity, Knuth multiplicative hash, NULL-key sentinel).

### §1.2 — std::variant usage (1) — RESOLVED via 3-bridge dispatch

`copyIndirect` uses `std::variant<Instruction*, PhyLocation>` for `dest_base` + `dest_index` passed to `setMemoryIndirect`.

**Decision (per testkeeper 13:19:29Z Q-A):** existing C bridge surface is THREE discrete bridges (NOT a tagged-union):

```c
/* lir_c_api.h:150-156 */
void lir_operand_set_memory_indirect_instr(LirOperand *op, LirInstruction *base, int32_t offset);
void lir_operand_set_memory_indirect_phy(LirOperand *op, LirPhyLocation base, int32_t offset);
void lir_operand_set_memory_indirect_phy3(LirOperand *op, LirPhyLocation base, LirPhyLocation index_reg, uint8_t multiplier);
```

C `copy_indirect` helper must dispatch on the std::variant tag (enumerated as `linked` vs `phy` per LinkedOperand::isLinked() at source):
- (D-1) base linked + index absent: `_set_memory_indirect_instr(op, instr_base, offset)`
- (D-2) base phy + index absent: `_set_memory_indirect_phy(op, phy_base, offset)`
- (D-3) base phy + index phy + multiplier: `_set_memory_indirect_phy3(op, phy_base, phy_index, multiplier)`
- (D-4) base linked + index linked: NOT representable in current 3-bridge surface — needs spec verification at draft-implementation (likely not exercised by inliner, but VERIFY pre-commit-3 OR add 4th bridge `_set_memory_indirect_instr3` if exercised).

The dispatch fan-out at copyIndirect is enumerated in §3.1 invariants I-1a/I-1b/I-1c/I-1d.

### §1.3 — Heap allocators (2) — Q-B RESOLVED, NEW BRIDGE REQUIRED

- `new BasicBlock(this)` — already wrapped by `lir_block_new(func, id)` (C path).
- `new Instruction(bb_copy, &instr, origin)` — **NEW BRIDGE REQUIRED** (per testkeeper 13:19:29Z Q-B verify: not present in lir_c_api.h; only adjacent `lir_operand_new_copy` exists for operand-class).

**Decision:** add `lir_instruction_new_copy(LirBasicBlock *bb, const LirInstruction *src, const void *origin)` as a new bridge. Pattern mirrors lir_operand_new_copy. Implementation:
1. allocate via existing `lir_instruction_new` or equivalent
2. set `parent_block_ = bb`
3. shallow-copy opcode + operation type + flags from src
4. set origin

This is a substrate-prep commit BEFORE the helper-port commits (see §4 commit 0).

---

## §2 — Substrate header specs (PhxIntPtrMap + PhxPtrIntMap)

### §2.1 — PhxIntPtrMap (int→void*)

**File:** Python/jit/hir/phx_int_ptr_map.h
**Pattern:** mirror PhxPtrMap (X3a substrate at hir/phx_ptr_map.h) with int-key + void*-value semantics.

```c
typedef struct PhxIntPtrMapEntry {
    int   key;       /* INT_MIN = empty slot sentinel (NOT 0; collides with valid id 0) */
    char  occupied;  /* explicit occupancy flag (avoids INT_MIN-as-valid edge case) */
    void *value;
} PhxIntPtrMapEntry;
```

**Sentinel choice:** since BasicBlock id_ + Instruction id_ both start at 0 and are dense ints, neither 0 nor any specific int value is a safe sentinel. Use explicit `char occupied` flag (1 byte overhead per slot vs PhxPtrMap's NULL-key sentinel; avoids off-by-one risk per Q1 supervisor disposition).

**API (mirrors PhxPtrMap):**
- `phx_int_ptr_map_init / destroy / clear`
- `phx_int_ptr_map_lookup` (returns void* or NULL on absent)
- `phx_int_ptr_map_contains` (1/0; disambiguates absent-vs-NULL-value)
- `phx_int_ptr_map_insert` (returns 1 = newly-inserted, 0 = updated)
- `phx_int_ptr_map_resize` (internal; load factor 0.7 = PHX_PTR_MAP_LOAD_NUM/DEN reused)
- `phx_int_ptr_map_size / capacity`
- `phx_int_ptr_map_at_key / at_value` (raw slot accessors for iteration)

**map_get_strict equivalent:** `phx_int_ptr_map_get_strict(m, key)` — wraps lookup with JIT_CHECK_C(value != NULL || contains(key)) loud-fail on absent. Mirrors std::unordered_map[]+at() semantics used in function.cpp via `map_get_strict` from containers.h.

### §2.2 — PhxPtrIntMap (void*→int)

**File:** Python/jit/hir/phx_ptr_int_map.h
**Pattern:** mirror PhxPtrMap with int-value semantics.

```c
typedef struct PhxPtrIntMapEntry {
    void *key;   /* NULL = empty slot sentinel (same as PhxPtrMap) */
    int   value; /* INT_MIN reserved? — NO; int values can be any int */
} PhxPtrIntMapEntry;
```

**Sentinel choice:** key=NULL same as PhxPtrMap; int-value carries no sentinel constraint (any int permitted).

**API (mirrors PhxPtrMap):** same surface as PhxIntPtrMap, with int-value variant. `phx_ptr_int_map_lookup` returns int directly + out-param for present/absent disambiguation:

```c
static inline int phx_ptr_int_map_lookup(
    const PhxPtrIntMap *m, const void *key, int *out_value, int *out_present);
```

OR alternative simpler API: lookup returns `int` and `_contains` separately. (Decision deferred to draft-implementation; both viable.)

### §2.3 — Substrate test obligations

Each new substrate header MUST land with C unit tests in Python/jit/lir/test_phx_int_ptr_map.c + test_phx_ptr_int_map.c (mirror existing PhxPtrMap test pattern). Coverage per testkeeper standard: insert/lookup/contains/resize/clear/destroy + corner cases (initial capacity, 2x resize trigger, absent-vs-NULL-value disambig).

---

## §3 — Per-function port mapping

### §3.1 — copyIndirect (function.cpp:17-55)

C signature:
```c
static void copy_indirect(
    PhxPtrIntMap *instr_refs,
    LirOperand *dest_op,
    LirMemoryIndirect *source_op);
```

**Invariants:**
- (I-1) base + index isLinked semantics: if base is LinkedOperand, dest_base = dest_op->instr() (Instruction*-tag); else dest_base = base->getPhyRegister() (PhyLocation-tag).
- (I-1a) base linked + index absent: call `lir_operand_set_memory_indirect_instr(dest_op, dest_op_instr, source_op->getOffset())`.
- (I-1b) base phy + index absent: call `lir_operand_set_memory_indirect_phy(dest_op, base->getPhyRegister(), source_op->getOffset())`.
- (I-1c) base phy + index phy: call `lir_operand_set_memory_indirect_phy3(dest_op, base->getPhyRegister(), index->getPhyRegister(), source_op->getMultipiler())`.
- (I-1d) base linked + index linked: VERIFY at draft-implementation whether this case is exercised by any inliner test; if exercised but unrepresentable in current 3-bridge surface, add 4th bridge `lir_operand_set_memory_indirect_instr3` as substrate-prep (commit 0b) before commit 3.
- (I-2) index nullability: if index is NULL (no index_reg in source), dispatch I-1a or I-1b only; index_reg + multiplier branches (I-1c) NOT taken.
- (I-3) instr_refs population: for each linked side, insert (dest_op_member_LinkedOperand*, source_LinkedOperand->getLinkedOperand()->instr()->id()) into instr_refs. Population happens AFTER setMemoryIndirect-equivalent dispatch (per source order at function.cpp:42-54).
- (I-4) memory-indirect assembly completeness: dispatch path I-1a/I-1b/I-1c MUST cover the (base-tag × index-tag) cross-product; no silent skip on unrepresentable case (per feedback_no_silent_bailout_in_helpers).

### §3.2 — copyOperand (function.cpp:57-88)

C signature:
```c
static void copy_operand(
    PhxIntPtrMap *block_index_map,
    PhxPtrIntMap *instr_refs,
    LirOperand *operand,
    LirOperand *operand_copy);
```

**Invariants:**
- (I-5) operand type discriminant: 8-way switch over OperandBase::kReg/kStack/kMem/kImm/kLabel/kInd/kNone/kVreg.
- (I-6) per-type field copy: kReg copies phy_register + data_type; kStack copies stack_slot + data_type; kMem copies memory_address; kImm copies constant + data_type; kLabel resolves block via block_index_map; kInd recurses into copy_indirect; kNone+kVreg are no-ops.
- (I-7) data_type preservation: explicit setDataType call after each non-Imm/Mem/Ind type (matches C++ source order).

### §3.3 — copyInput (function.cpp:90-106)

C signature:
```c
static void copy_input(
    PhxIntPtrMap *block_index_map,
    PhxPtrIntMap *instr_refs,
    LirOperandBase *input,
    LirInstruction *instr_copy);
```

**Invariants:**
- (I-8) linked-vs-immediate dispatch: isLinked → allocateLinkedInput(NULL) + instr_refs.emplace; else → allocateImmediateInput(0) + copy_operand.
- (I-9) data_type propagation: input_copy->setDataType(input->dataType()) ALWAYS called for immediate path (not for linked path — matches C++ source).

### §3.4 — connectLinkedOperands (function.cpp:108-114)

C signature:
```c
static void connect_linked_operands(
    PhxIntPtrMap *output_index_map,
    PhxPtrIntMap *instr_refs);
```

**Invariants:**
- (I-10) iteration: traverse instr_refs by raw-slot iteration (per phx_ptr_int_map_at_key/at_value pattern); skip absent slots.
- (I-11) per-pair effect: operand->setLinkedInstr(map_get_strict(output_index_map, instr_index)) — loud-fail if instr_index absent in output_index_map.

### §3.5 — deepCopyBasicBlocks (function.cpp:116-139)

C signature:
```c
static void deep_copy_basic_blocks(
    LirBasicBlock *const *src_blocks, size_t src_count,
    PhxIntPtrMap *block_index_map,
    const void *origin);
```

**Invariants:**
- (I-12) two local maps (output_index_map + instr_refs) lifecycle-bound to this function's stack frame; both init/destroy must pair.
- (I-13) per-bb processing: lookup bb_copy via block_index_map; addSuccessor for each src successor (resolved via block_index_map); for each instruction: new Instruction (via lir_instruction_new_copy or equivalent), appendInstr to bb_copy, emplace in output_index_map by id_, copy_operand for output, copy_input for each input.
- (I-14) link resolution: connect_linked_operands called LAST (after all output instructions are in output_index_map).

### §3.6 — Function::copyFrom body (function.cpp:152-183)

C signature:
```c
int lir_function_copy_from_impl(
    LirFunction *caller, const LirFunction *callee,
    LirBasicBlock *prev_bb, LirBasicBlock *next_bb,
    const void *origin,
    int *out_begin, int *out_end);
```

**Invariants:**
- (I-15) precondition check: prev_bb has exactly 1 successor == next_bb (JIT_CHECK_C loud-fail).
- (I-16) per-src-bb new-bb allocation + block_index_map population + ensureBlockCapacity per-iter + blocks_ tail-shift insertion (preserves last-block-as-exit-block invariant).
- (I-17) deep_copy_basic_blocks called AFTER all bb_copy allocations (output_index_map within deep_copy is local; block_index_map is shared).
- (I-18) post-copy stitching: prev_bb->setSuccessor(0, blocks_[start]) + last-inserted-bb (blocks_[end-1])->addSuccessor(next_bb); JIT_CHECK_C that blocks_[end-1] had no successors before stitching.
- (I-19) return CopyResult{start, end} via out-params.

---

## §4 — Wire-up sequence (commit-by-commit)

Estimated 6-8 commits (added commit 0 per Q-B resolution) per Tier-1 + per-commit gate per CLAUDE.md:

| # | Commit | Files | LOC | Tier |
|---|--------|-------|-----|------|
| 0 | Add lir_instruction_new_copy bridge (Q-B substrate-prep) | lir_c_api.h + instruction_impl.c (+30 LOC) | +30 | 1 (substrate-only) |
| 0b | (CONDITIONAL) Add lir_operand_set_memory_indirect_instr3 if I-1d case exercised | lir_c_api.h + operand_impl.c (+20 LOC) | +20 | 1 (substrate-only, only if Q-A I-1d verification flags it) |
| 1 | Add PhxIntPtrMap substrate + tests | hir/phx_int_ptr_map.h (new ~150 LOC) + lir/test_phx_int_ptr_map.c (new ~120 LOC) | +270 | 1 (substrate-only) |
| 2 | Add PhxPtrIntMap substrate + tests | hir/phx_ptr_int_map.h (new ~150 LOC) + lir/test_phx_ptr_int_map.c (new ~120 LOC) | +270 | 1 |
| 3 | Port copy_indirect + copy_operand to function_impl.c (NOT yet sole-path; C++ retained as parallel) | function_impl.c (+90 LOC) + lir_c_api.h (extern decls) | +95 | 1 (parallel) |
| 4 | Port copy_input + connect_linked_operands + deep_copy_basic_blocks (parallel) | function_impl.c (+80 LOC) | +80 | 1 (parallel) |
| 5 | Port lir_function_copy_from_impl C body (parallel; C extern wrapper at function.cpp wires to C body, NOT C++ Function::copyFrom) | function_impl.c (+60 LOC), function.cpp (rewire extern wrapper) | +65 / -5 | 2 (sole-path candidate) |
| 6 | Wiring gate verification (force_compile lir_function_copy_from + diff vs C++ baseline) | scripts/gate_phoenix.sh wiring suite extension | +10 | 2 |
| 7 | Delete C++ residue from function.cpp + per-commit gate at chain close | function.cpp -160 LOC (helpers + copyFrom body + delegations) | -160 | 2 (sole-path) + chain-close 24-bench cap-check |

**Per-commit gate per CLAUDE.md:** 4-bench --reps=3 ABBA after each commit; BLOCK at >5% geomean drop. Chain close = full 24-bench --reps=5 cap-check + ARM64 gate items v+vi.

**Commit 0 verification at draft-implementation:** confirm I-1d (base linked + index linked) case at copyIndirect call sites by greping all copyIndirect-reachable code paths; if absent in inliner test corpus, defer 0b commit and document as YAGNI-bounded scope.

---

## §5 — Falsifier spec

**Wave-class disposition (theologian as W-PERF owner per supervisor 13:16:01Z Q2):** 5.A3 is NON-M3 per W-PERF §3b literal spec — 6 external Function-method callsites < 20 threshold. F3b precedent applies (single re-run match against earlier same-session anchor sufficient to declare contamination if cap-check ABBA shows drift).

**Rationale:** §3b mechanism class (bit-identical-source ≠ bit-identical-machine-code via inline-relocation) does not apply to substrate-add (PhxIntPtrMap + PhxPtrIntMap are NEW C code, not relocated existing code). The perturbation class differs: new code introduces new flame-graph nodes; relocation perturbs existing flame-graph nodes via compiler-side decisions.

**However, deepCopyBasicBlocks port itself is structural-equivalent** (helper graph re-housed with no algorithmic change). For that subset, F3b precedent applies cleanly — same source intent, different execution language (C vs C++).

**Falsifier on PASS:** per-commit gate ABBA + chain-close 24-bench cap-check; no drift exceeding ±2pp same-session OR ±7pp cross-session bands.

**Falsifier on FAIL:** if chain-close ABBA shows drift, bisect within 5.A3 chain (commits 1-7) per standard cap-ceiling protocol (3 ABBAs max recursive); attribute to substrate-add vs helper-port vs copyFrom-port slice.

**Falsifier on AMBIGUOUS:** 2-5pp delta in either band → run 1 additional same-session ABBA at chain-close for variance estimate before disposition.

---

## §6 — Risks + mitigations

| Risk | Mitigation |
|------|------------|
| Substrate-add introduces hash-collision pattern that degrades copyFrom hot path | C unit tests cover insert/lookup/resize; per-commit gate catches >5% geomean drop; if isolated to commit 1-2, revert + re-spec |
| std::variant↔C-bridge dispatch (3-bridge fan-out per Q-A): I-1d (linked+linked) case unrepresentable | Verify I-1d exercised at draft-implementation grep; if exercised, add bridge as commit 0b; if absent, document YAGNI-bounded scope |
| `new Instruction(bb_copy, &instr, origin)` lacks C bridge (Q-B confirmed absent) | Add lir_instruction_new_copy as commit 0 (substrate-prep); pattern mirrors lir_operand_new_copy |
| map_get_strict loud-fail semantics differ between PhxIntPtrMap and std::unordered_map | Explicit JIT_CHECK_C in phx_int_ptr_map_get_strict; matches containers.h map_get_strict pattern |
| Sole-path flip (commit 5-7) introduces regression invisible in parallel commits 3-4 | Wiring gate per gate_phoenix.sh --wiring extension at commit 6 (per W25 wiring-gate precedent); force_compile lir_function_copy_from runs both paths for diff |
| 5.A3 chain interleaves with 5.B/5.C/5.D parallel work → cross-attribution failure on regression | Sequence: 5.A3 commits land before 5.B/5.C/5.D begin (per standard batch protocol); same-session ABBA isolates 5.A3 surface |

---

## §7 — Open questions

**Resolved at spec time (testkeeper 13:19:29Z):**
- Q-A: setMemoryIndirect = 3 discrete bridges (NOT tagged-union); see §1.2 + I-1a/b/c dispatch.
- Q-B: lir_instruction_new_copy NOT present; ADD as commit 0 substrate-prep; see §1.3.

**Unresolved (deferred to draft-implementation):**
- Q-C: PhxPtrIntMap lookup API shape — out-param + present-flag vs separate _contains call. Defer to draft-implementation; both viable.
- Q-D: I-1d (linked+linked memory-indirect) case exercised? Verify via grep on inliner_c.c + LIR test corpus pre-commit-3; if exercised, add commit 0b for `lir_operand_set_memory_indirect_instr3` bridge; if not, document YAGNI-bounded scope.

---

## §8 — References

**Spec template + methodology:**
- project_bridge_spec_template.md — query scribe BEFORE drafting; enumerate every invariant; "move" ≠ "rewrite"
- feedback_no_workarounds.md — substrate-add over cast-workaround
- W-PERF scope doc §3b (cpython/docs/w-perf-pre-w27c-bisect-scope-2026-05-01.md, d114224a4d) — M3-class wave methodology

**Substrate precedent:**
- Python/jit/hir/phx_ptr_map.h — X3a substrate (void*→void*); pattern source for §2

**Decision history:**
- D-1777621209 (theologian 2026-04-XX) — 5.A re-scope: 5.A3 = function.cpp ~220 LOC + STRUCT-tier substrate
- D-1777654758 — bit-identical-not-innocence-proof memory entry
- supervisor 13:05:12Z (2026-05-05) — 5.A3 dispatch
- supervisor 13:16:01Z (2026-05-05) — Q1/Q2/Q3 dispositions (option A substrate-add, owner-decides M3-class, 5 separate static C functions)

**Related files (not modified by this spec):**
- Python/jit/lir/inliner_c.c:417 — sole external consumer of lir_function_copy_from
- Python/jit/lir/lir_c_api.h — extern surface (extend with phx_*_map.h includes + new helper decls)

---

Spec complete. Ready for testkeeper Q-A/Q-B verification + generalist wire-up start.
