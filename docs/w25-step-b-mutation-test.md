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

Mutation function choice: `hir_block_id`.

**Original signature** (post-Step-A at 3404a81192):

```c
/* hir_c_api.h */
int hir_block_id(struct HirBasicBlock *block);
/* hir_c_api.cpp */
int hir_block_id(struct HirBasicBlock *block) { ... }
```

**Mutated signature** (W25 §5.3 baseline mutation, BOTH files):

```c
/* hir_c_api.h */
int hir_block_id(struct HirBasicBlock *block, int unused_drift_param);
/* hir_c_api.cpp */
int hir_block_id(struct HirBasicBlock *block, int unused_drift_param) { ... }
```

Why `hir_block_id`:
- Called from at least 1 §1b TU (per `count_w25_b1b_tus.sh` survey).
- Simple int parameter — no implicit-conversion masking.
- ABI-incompatible mutation (extra arg) — caller pushes wrong arg
  count vs callee expects new arg → silent UB at runtime.
- In C, calling a function with a different signature than its
  declaration is undefined behavior, NOT a hard link error. §5.3
  answers empirically what this UB looks like in our build.

## 2. Baseline procedure (pre-Step-B, run at HEAD e8b8c149fe or later)

```bash
# Step 1: capture baseline state
git rev-parse HEAD  # expect: e8b8c149fe... (this commit) or its descendant

# Step 2: apply DUAL mutation (.h + .cpp must stay consistent)
cd /data/users/alexturner/phoenix/cpython
git stash --include-untracked  # stash any working-tree changes
# Mutate header decl
sed -i 's|int hir_block_id(struct HirBasicBlock \*block);|int hir_block_id(struct HirBasicBlock *block, int unused_drift_param);|' Python/jit/hir/hir_c_api.h
# Mutate cpp impl signature line (preserves brace-on-same-line)
sed -i 's|int hir_block_id(struct HirBasicBlock \*block) {|int hir_block_id(struct HirBasicBlock *block, int unused_drift_param) {|' Python/jit/hir/hir_c_api.cpp

# Verify both mutations applied
grep -A1 "hir_block_id" Python/jit/hir/hir_c_api.h | head -5
grep -A1 "^int hir_block_id" Python/jit/hir/hir_c_api.cpp | head -3

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

**Hypothesis (theoretical):** the mutation will be silently linkable
because §1b TUs use local extern decls (their stale signature matches
itself; linker sees only function name).

**To-verify:** the build either:
- (a) Succeeds at link time — confirms drift surface exists. Record
  the finding as PRE-STEP-B BASELINE.
- (b) Fails at compile time — surprising; hir_c_api.h consumers (the
  §1a TUs) would catch the mismatch. Record what those compile errors
  look like.
- (c) Succeeds at compile but fails at runtime — also possible if some
  caller ends up with corrupted-stack behavior.

The actual outcome is the empirical baseline. Document it.

## 4. Post-Step-B procedure

After Step B lands (local extern decls deleted, all §1b TUs include
hir_c_api.h):

```bash
# At post-Step-B HEAD
git rev-parse HEAD  # expect: <Step B's last commit hash>
# Same DUAL mutation as §2 — keep .h + .cpp consistent
sed -i 's|int hir_block_id(struct HirBasicBlock \*block);|int hir_block_id(struct HirBasicBlock *block, int unused_drift_param);|' Python/jit/hir/hir_c_api.h
sed -i 's|int hir_block_id(struct HirBasicBlock \*block) {|int hir_block_id(struct HirBasicBlock *block, int unused_drift_param) {|' Python/jit/hir/hir_c_api.cpp
scripts/build_phoenix.sh > /tmp/w25-mutation-poststepb-stdout.log 2>&1
echo "BUILD_EXIT=$?" >> /tmp/w25-mutation-poststepb-stdout.log
git checkout -- Python/jit/hir/hir_c_api.h Python/jit/hir/hir_c_api.cpp
```

**Expected post-Step-B outcome:** compile FAILS in every consuming TU
that calls `hir_block_id` (the linker doesn't get a chance — compiler
catches the mismatch at parse time because all consumers see the
canonical signature from hir_c_api.h).

## 5. Acceptance criterion

§5.3 falsification PASSES when the pre-Step-B baseline shows DRIFT GOES
UNDETECTED (or weakly detected) AND the post-Step-B run shows DRIFT
CAUGHT AT COMPILE TIME with clear errors at every call site.

If pre-Step-B baseline shows the drift is ALREADY caught at compile time
(unexpected outcome (b) above), then either:
- The drift surface was always smaller than Step B's framing assumed
  (and Step B's value is reduced — not invalidated, just reframed).
- Or my mutation choice doesn't actually exercise the §1b drift surface
  (need a different mutation).

Either way, the empirical finding informs Step B's framing.

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
