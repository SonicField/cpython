# W25 §5.3 Mutation Test — Step B Falsification Baseline

**Status:** Pre-Step-B baseline procedure. Run by testkeeper at HEAD
3404a81192 (post-Step-A, pre-Step-B) to capture the empirical "drift
silently links" baseline. Re-run after Step B lands to confirm the
"drift caught at compile time" post-condition.

**Purpose:** Per W25 spec docs/w25-hbb-canonicalization.md §5.3, validate
that Step B's local-extern cleanup actually closes the signature-drift
surface. Without this empirical test, the "lint gate catches drift"
claim is theoretical.

---

## 1. Mutation choice

**CRITICAL methodology fix per theologian [chat 2026-04-22 19:26Z]:**
mutate BOTH `hir_c_api.h` declaration AND `hir_c_api.cpp` impl
together. Mutating only the header would fail in `hir_c_api.cpp` first
(header-vs-impl mismatch) — would prove Step A header-impl consistency
is tight, NOT the §1b drift surface this test should expose.

With dual-mutation (.h + .cpp consistent on new sig):
- `hir_c_api.cpp` compiles fine (.h + .cpp agree)
- §1a TUs that include `hir_c_api.h` see new sig → callers with
  wrong arg count fail to compile (proves §1a path is protected)
- §1b TUs with local extern decls keep their OLD signature → callers
  in those TUs compile fine against the local extern, then linker
  resolves to the new-signature impl → silent runtime UB (PROVES §1b
  is the unprotected drift surface — exactly what Step B closes)

**Function-target correction per theologian [chat 2026-04-22 19:29Z]:**
the original `hir_block_id` choice has NO §1b TU callers (only
`func_type_checks_c.c` which is §1a). Mutating it would catch the
§1a-path drift (proves Step A is tight) but NOT the §1b drift surface
(which is what Step B closes). To test the §1b surface, choose a
function called from §1b TUs via local extern.

Mutation function choice: `hir_c_insert_before` (§1b callers verified
via grep — `licm_c.c:17` extern + `:246` call, `pass_output_type_c.c
:940` extern + `:998/1006/1071` calls; both are §1b TUs per
`scripts/count_w25_b1b_tus.sh` inventory).

**Original signature** (in hir_c_api.h:726 + hir_c_api.cpp:2349):

```c
/* hir_c_api.h */
void hir_c_insert_before(HirInstr new_instr, HirInstr before);
/* hir_c_api.cpp */
void hir_c_insert_before(HirInstr new_instr, HirInstr before) { ... }
```

**Mutated signature** (W25 §5.3 baseline mutation, BOTH files):

```c
/* hir_c_api.h */
void hir_c_insert_before(HirInstr new_instr, HirInstr before, void *unused_drift_param);
/* hir_c_api.cpp */
void hir_c_insert_before(HirInstr new_instr, HirInstr before, void *unused_drift_param) { ... }
```

Why `hir_c_insert_before`:
- §1b callers exist (licm_c.c + pass_output_type_c.c), so the §1b
  drift surface is exercised.
- Simple void* parameter — no implicit-conversion masking, no struct
  layout dependency.
- ABI-incompatible mutation (extra arg) — §1b callers' local externs
  stay 2-arg, callers pass 2 args, linker resolves to 3-arg impl →
  silent runtime UB.
- Predictable: §1a path (none for this function) and §1b path
  (licm_c.c + pass_output_type_c.c local externs) are visibly
  identifiable in the build output.

## 2. Baseline procedure (pre-Step-B, run at HEAD e8b8c149fe or later)

```bash
# Step 1: capture baseline state
git rev-parse HEAD  # expect: e8b8c149fe... (this commit) or its descendant

# Step 2: apply DUAL mutation (.h + .cpp must stay consistent)
cd /data/users/alexturner/phoenix/cpython
git stash --include-untracked  # stash any working-tree changes
# Mutate header decl
sed -i 's|void hir_c_insert_before(HirInstr new_instr, HirInstr before);|void hir_c_insert_before(HirInstr new_instr, HirInstr before, void *unused_drift_param);|' Python/jit/hir/hir_c_api.h
# Mutate cpp impl signature line (preserves brace-on-same-line)
sed -i 's|void hir_c_insert_before(HirInstr new_instr, HirInstr before) {|void hir_c_insert_before(HirInstr new_instr, HirInstr before, void *unused_drift_param) {|' Python/jit/hir/hir_c_api.cpp

# Verify both mutations applied
grep -A1 "hir_c_insert_before" Python/jit/hir/hir_c_api.h | head -5
grep -A1 "^void hir_c_insert_before" Python/jit/hir/hir_c_api.cpp | head -3

# Step 3: try to build
scripts/build_phoenix.sh > /tmp/w25-mutation-baseline-stdout.log 2>&1
echo "BUILD_EXIT=$?" >> /tmp/w25-mutation-baseline-stdout.log

# Step 4: capture findings
echo "=== BASELINE (pre-Step-B, dual-mutation .h+.cpp) ===" > docs/w25-step-b-mutation-baseline.txt
echo "HEAD: $(git rev-parse HEAD)" >> docs/w25-step-b-mutation-baseline.txt
grep -E "error:|warning:|BUILD_EXIT" /tmp/w25-mutation-baseline-stdout.log | tail -30 >> docs/w25-step-b-mutation-baseline.txt

# Step 5: revert BOTH files
git checkout -- Python/jit/hir/hir_c_api.h Python/jit/hir/hir_c_api.cpp
git stash pop || true  # restore stashed working tree

# Step 6: verify revert succeeded
git diff Python/jit/hir/hir_c_api.h Python/jit/hir/hir_c_api.cpp  # expect: empty
```

## 3. Expected baseline outcome

**Hypothesis (theoretical):** with corrected target (hir_c_insert_before
which has §1b callers) AND atomic .h+.cpp mutation, the build will
succeed (BUILD_EXIT=0) because:
- hir_c_api.cpp compiles fine (header + impl agree on new sig)
- §1a TUs that include hir_c_api.h see new sig — but hir_c_insert_before
  has no §1a callers, so no §1a-side compile error
- §1b TUs (licm_c.c + pass_output_type_c.c) use local extern (2-arg)
  for hir_c_insert_before; their callers compile fine against local
  extern; linker resolves to 3-arg impl by name → silent runtime UB

**Acceptance flip per theologian [chat 2026-04-22 19:29Z]:** PRE-STEP-B
baseline PASSES when BUILD_EXIT=0 (drift undetected at compile time).
This proves the §1b drift surface exists.

**Surprising outcomes to investigate:**
- BUILD_EXIT≠0 with errors in licm_c.c / pass_output_type_c.c: the §1b
  protection might already be in place via some other mechanism
  (unexpected — would re-frame Step B's value).
- BUILD_EXIT≠0 with errors elsewhere: a different §1a or non-§1b TU
  has a hir_c_insert_before caller my grep missed. Document the
  unexpected callsite + adjust scope of finding.

## 4. Post-Step-B procedure

After Step B lands (local extern decls deleted, all §1b TUs include
hir_c_api.h):

```bash
# At post-Step-B HEAD
git rev-parse HEAD  # expect: <Step B's last commit hash>
# Same DUAL mutation as §2 — keep .h + .cpp consistent
sed -i 's|void hir_c_insert_before(HirInstr new_instr, HirInstr before);|void hir_c_insert_before(HirInstr new_instr, HirInstr before, void *unused_drift_param);|' Python/jit/hir/hir_c_api.h
sed -i 's|void hir_c_insert_before(HirInstr new_instr, HirInstr before) {|void hir_c_insert_before(HirInstr new_instr, HirInstr before, void *unused_drift_param) {|' Python/jit/hir/hir_c_api.cpp
scripts/build_phoenix.sh > /tmp/w25-mutation-poststepb-stdout.log 2>&1
echo "BUILD_EXIT=$?" >> /tmp/w25-mutation-poststepb-stdout.log
git checkout -- Python/jit/hir/hir_c_api.h Python/jit/hir/hir_c_api.cpp
```

**Expected post-Step-B outcome:** compile FAILS in licm_c.c +
pass_output_type_c.c (the §1b TUs that called `hir_c_insert_before`
via local extern). Step B deleted those externs and added
`#include hir_c_api.h`, so post-Step-B these TUs see the new 3-arg
signature directly — their 2-arg callers fail to compile with clear
"too few arguments" errors.

## 5. Acceptance criterion

§5.3 falsification PASSES when:
- PRE-STEP-B baseline: BUILD_EXIT=0 (drift undetected at compile time
  — §1b drift surface exists per hypothesis)
- POST-STEP-B re-run: BUILD_EXIT≠0 with compile errors in §1b TU
  callers (§1b drift surface closed by Step B)

If PRE-STEP-B baseline shows BUILD_EXIT≠0 (drift caught somewhere),
investigate: the protection might already be in place via a different
mechanism, or my mutation choice doesn't actually exercise the §1b
drift surface as predicted. Either way, the empirical finding informs
Step B's framing.

---

## 6. Sequencing

1. **Pre-Step-B baseline run** (NOW): testkeeper executes §2 procedure
   at HEAD 3404a81192. Outcome captured in
   docs/w25-step-b-mutation-baseline.txt (NEW file).
2. **Step B implementation** (NEXT): generalist deletes local externs +
   adds canonical includes in 7 cleanup-target TUs per
   count_w25_b1b_tus.sh inventory.
3. **Post-Step-B re-run** (POST-STEP-B): testkeeper repeats §4 procedure.
   Outcome appended to docs/w25-step-b-mutation-baseline.txt as POST-STEP-B
   section.
4. **§5.3 closure**: if post-Step-B drift is caught at compile time per §5
   acceptance, mutation test confirms Step B closed the drift surface.

---

## 7. Cross-references

- W25 spec: docs/w25-hbb-canonicalization.md §5.3
- §1b TU inventory: scripts/count_w25_b1b_tus.sh (17 total / 7 cleanup
  targets / 10 type-only at HEAD e6a8a2d0fb)
- Step A: e6a8a2d0fb (canonical struct-pointer typedef landed)
- §5.1 dual-include compile check: 3404a81192 (canonicalization
  validated structurally)
- §5.2 lint gate: deferred to Step C (post-Step-B)
