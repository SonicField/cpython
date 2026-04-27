# W-I3-RUNTIME-ASSERT — BasicBlock::id Immutability Enforcement

**Status:** Spec draft (theologian, 2026-04-24, supervisor 21:52:06Z authorized)
**Owner:** theologian (design), generalist (implementation), testkeeper (perf measurement if needed)
**Trigger:** pythia #128 + #131 #1 named risk: I3 invariant (BasicBlock::id never mutated post-allocation) is grep-only, not runtime-asserted.

---

## 1. Problem

Phase B B-gamma (Tier 8 SECOND-PILOT, push 40 4145fe3fb0) introduced
`PhxBcBlockArray` indexed by `BasicBlock::id`. Correctness depends on
**I3**: `BasicBlock::id` is allocation-monotonic AND never mutated
post-allocation.

I3 is currently **grep-verified**, not runtime-asserted. Original
rationale (theologian 2026-04-22 D-1777029604, builder_state_c.h:230-235):
"runtime sentinel would defeat dense-array O(1) win".

Risk (pythia #128 D-1777052803, #131 D-1777061189): future HIR pass
that renumbers ids (SSA-destruction-style cache-locality renumbering,
peephole block-splitter, speculative-inlining clone) silently corrupts
the dense-array lookup. The I2 read-site check catches out-of-range,
not stale-mapping-correct-range.

## 2. Enforcement options

### (I) Per-read sentinel byte check
Add a sentinel field to `PhxBcBlockEntry` (e.g., `uint8_t valid`); set
on insert; check at `phx_bc_block_array_at` lookup. Cost: 1 branch
per emit-method lookup. Hot-path (every JIT compile reads bc_block
many times). Always-on detection in both release + pydebug.

### (II) Generation counter
`PhxBcBlockArray.generation` bumped each createBlocks call; entries
store generation; lookup checks generation matches. Cost: 1 memory
read + 1 compare per lookup. Slightly more than (I). Always-on.

### (III) Debug-only sentinel (JIT_DCHECK)
Same as (I) but `#ifdef Py_DEBUG`. Cost: zero in release; catches in
pydebug only. Pydebug gate is the canonical falsifier.

### (IV) CI-time grep gate
Add a pre-commit / CI script that fails any commit introducing
`block->id =` / `block->set_id()` / equivalent id-mutation patterns
in `Python/jit/hir/`. Catches at commit-time before merge. Zero
runtime cost.

## 3. Recommendation

**(IV) + (III) combined:**
- (IV) primary: zero runtime cost, catches violations at commit-time
  (most failure modes prevented before they ship).
- (III) secondary: pydebug guard against (a) grep-pattern miss
  (e.g., id mutated via member function, indirect pointer write), (b)
  CI bypass.

(I) and (II) reverse the original perf trade without strong evidence
the always-on detection is needed. If (IV)+(III) prove insufficient
(a violation slips both), escalate to (II).

## 4. Cost

- **(IV):** ~30min (script + CI integration). Pure tooling, no
  perf measurement.
- **(III):** ~15min (sentinel field + insert/lookup edits, JIT_DCHECK
  guard). Pydebug perf trivially affected; release perf zero.
- Combined: ~45min implementation, no perf measurement needed.

## 5. Falsification test

Before landing (IV):
- Write a synthetic test patch that adds `block->id = new_value;` to
  a fake HIR pass file. CI script must FAIL on it.
- Confirm CI script PASSES on current HEAD (no false positives in
  current code).

Before landing (III):
- Same synthetic test patch + JIT_DCHECK should fire under pydebug.

## 6. Sequencing

Post-push-51 land (correction commit) → land W-I3-RUNTIME-ASSERT.
Implementation does NOT need devgpu004 (x86 alone covers CI script
+ pydebug build).

## 7. CI integration (resolved per librarian 22:05:19Z)

Phoenix has the framework: `scripts/gate_phoenix.sh` Step 1c (preserved-
symbols, inline), Step 1d (W25 lint, inline grep + extern-pattern),
Step 1e (W44 DO-NOT-USE, delegates to `scripts/check_do_not_use_callers.sh
--strict`), Step 1g (W45 §3.5 derivation-drift, delegates to
`scripts/w45_section_3_5_derivation_drift.sh --strict`).

Pattern for I3: write `scripts/check_i3_invariant.sh --strict` (exit
nonzero on violation) + wire as Step 1h in gate_phoenix.sh ~line 220
mirroring 1e/1g shape (capture OUTPUT, tee to RESULTS_FILE, GATE_PASS=0
+ FAILURES on nonzero).

Precedents: D-1776476425 (forward-looking gate rule) + D-1776477460
(retroactive wiring audit).
