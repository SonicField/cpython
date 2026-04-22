# W25 — `HirBasicBlock` / `HirCFG` Canonicalization

**Status:** ACTIVE (Tier 6 Phase 1 prerequisite). Authorized 2026-04-22 per Alex
Option B directive ("clear the debt now. Then we stop building new code on top
of debt").

**Scope:** Resolve dual-semantic conflicts on `HirBasicBlock` and `HirCFG` so
that any TU can include both `hir_c_api.h` and `hir_basic_block_c.h` without
typedef collision. Then complete W25 Layer B (header discipline + lint gate)
per `docs/w25-typed-bridges.md` §4.2.

**Owner:** theologian (this spec) + generalist (implementation).

**Estimated cost:** ~90-120min focused refactor + dual-arch verify cycle.
Tighter than the 3-4hr supervisor estimate because the migration is mechanical
and runtime behavior is unchanged.

---

## 1. The architectural blocker

`HirBasicBlock` and `HirCFG` each have **two incompatible typedef definitions**
in the codebase:

| Type            | `hir_c_api.h`            | `hir_basic_block_c.h`               |
|-----------------|--------------------------|-------------------------------------|
| `HirBasicBlock` | `typedef void* ...` (8B) | `typedef struct {...} ...` (96B)    |
| `HirCFG`        | `typedef void* ...` (8B) | `typedef struct {...} ...` (~40B)   |

A single TU cannot include both headers — the typedef is redefined to a
different type. Today this is worked around in **two distinct ways**:

### 1a. Eight `.c` TUs use `hir_c_api.h` only (void* world)

Empirical enumeration (generalist grep at HEAD `b808cdada4`,
confirmed Step A iter 1):

`clean_cfg.c`, `copy_propagation.c`, `dead_code_elimination.c`,
`dynamic_comparison_elimination.c`, `func_type_checks_c.c`,
`guard_removal.c`, `hir_stats_c.c`, `phi_elimination.c`. They use
`HirBasicBlock`/`HirCFG` as value types that are actually void pointers:

```c
HirBasicBlock block = hir_cfg_blocks_first(cfg);   /* 8-byte handle */
HirBasicBlock *rpo_blocks = malloc(rpo_cap * sizeof(HirBasicBlock));  /* clean_cfg.c L75 */
```

**Risk:** signatures of `hir_cfg_blocks_first` and friends are pinned to
`hir_c_api.h`. Any signature change in `hir_c_api.h` propagates here only via
recompilation — there is no second declaration to cross-check.

### 1b. 17 `.c` and `.cpp` TUs use `hir_basic_block_c.h`; 9 have local forward decls

Empirical enumeration at HEAD `b808cdada4` (testkeeper grep, 2026-04-22):

**All 17 TUs that include `hir_basic_block_c.h` (Step B candidates):**

| TU                              | extern hir_* decls |
|---------------------------------|-------------------:|
| `builder_emit_c.c`              | 167                |
| `pass_output_type_c.c`          |  34                |
| `resolve_kwargs_c.c`            |   6                |
| `refcount_pass_c.c`             |   5                |
| `refcount_env_c.c`              |   4                |
| `licm_c.c`                      |   4                |
| `ssaify_c.c`                    |   3                |
| `hir_basic_block_c.c`           |   1                |
| `cfg.cpp`                       |   1                |
| `assignment_c.c`                |   0                |
| `dominator_c.c`                 |   0                |
| `hir_cfg_rpo_c.c`               |   0                |
| `hir.cpp`                       |   0                |
| `hir_instr_c_verify.cpp`        |   0                |
| `insert_update_prev_instr_c.c`  |   0                |
| `liveness_c.c`                  |   0                |
| `ssa_check_c.c`                 |   0                |
| **TOTAL**                       | **225 across 9 TUs**|

The 9 with externs above use a **broad grep** (`extern.*hir_*`) — counts any
local extern of `hir_*` symbols. **The actual Step B cleanup target is 7
TUs**, narrower: only TUs whose local externs match the §3 Step C lint
pattern (`hir_(c_|cfg_|block_|bb_|edge_|func_|instr_)`). Two TUs
(`refcount_env_c.c`, `refcount_pass_c.c`) have local externs of
`hir_liveness_*` / `hir_refcount_*` (file-pair-scoped helpers, not
hir_c_api.h API surface) that the broad grep counts but the lint gate
correctly excludes.

**Authoritative Step B target count**: run `scripts/count_w25_b1b_tus.sh`
(generalist landed 40dd82e658). Output uses the lint pattern; matches what
Step C will enforce. Post-Step-A baseline at HEAD `e6a8a2d0fb`:
- 17 §1b TUs total
- **7 with extern decls** (Step B cleanup targets, lint-pattern match)
- 10 type-only (no-op for Step B)

Step B target list (per script output, sorted by extern count desc):
| TU                              | extern count |
|---------------------------------|-------------:|
| `builder_emit_c.c`              | 130          |
| `pass_output_type_c.c`          |  25          |
| `resolve_kwargs_c.c`            |   5          |
| `licm_c.c`                      |   3          |
| `ssaify_c.c`                    |   1          |
| `hir_basic_block_c.c`           |   1          |
| `cfg.cpp`                       |   1          |

(Broad-grep table above retained for context. Numerical authority: script
output, not table.)

`builder_emit_c.c` dominates with 167 externs (74% of total). May warrant
per-extern-batch within builder_emit_c.c if a single commit becomes
unwieldy. Generalist's call at implementation.

The workaround pattern (in TUs with externs):

```c
/* refcount_env_c.c L14 */
/* Forward declarations (avoid hir_c_api.h typedef conflict) */
extern HirInstr hir_block_first(HirBasicBlock block);
/* ... */
```

**Risk: this is a signature-drift timebomb.** If `hir_c_api.h` changes a
signature, the 225 local extern decls (across 9 TUs, see §1b table) go
stale. The compiler doesn't catch the
drift because each TU's local extern matches itself; the linker may still
resolve the symbol (calling convention, void* permissiveness). Push 51
(2026-04-22) was exactly this class of bug.

W25 Layer B's lint gate is designed to catch this — but Layer B cannot land
until the dual-semantic typedef conflict resolves.

---

## 2. Canonical resolution

**Make `HirBasicBlock` and `HirCFG` always mean the canonical struct.** Both
become opaque-pointer-to-incomplete-struct in `hir_c_api.h` (forward decl) and
fully-defined struct in `hir_basic_block_c.h`.

### Why struct-pointer over void*

- Struct is the **richer** type. Void* erases information.
- Struct already has a working canonical implementation
  (`hir_basic_block_c.h`). Choosing void* would require renaming the existing
  struct (large blast radius in 17 TUs).
- C allows forward declaration of incomplete structs: any TU that only calls
  API functions doesn't need the full layout, just the forward decl. This
  preserves the "opaque handle" property for API consumers.
- Pointer-to-struct gives the compiler stronger type-checking than void*
  (catches `HirBasicBlock *` vs `HirCFG *` confusion at the call site).

### Concrete shape post-fix

`hir_c_api.h`:
```c
/* Forward declarations — full layouts in hir_basic_block_c.h */
struct HirBasicBlock;
struct HirCFG;

/* API signatures use struct-pointer */
struct HirBasicBlock *hir_cfg_blocks_first(struct HirCFG *cfg);
struct HirBasicBlock *hir_cfg_blocks_next(struct HirCFG *cfg, struct HirBasicBlock *block);
size_t hir_cfg_get_rpo(struct HirCFG *cfg, struct HirBasicBlock **out, size_t cap);
/* ... */
```

`hir_basic_block_c.h` (unchanged): keeps the canonical struct definitions.

A TU that needs only API access: `#include "hir_c_api.h"`. Gets forward
decls + signatures. Cannot dereference fields.

A TU that needs field access: `#include "hir_basic_block_c.h"` (and
`hir_c_api.h` if it also calls API functions — no conflict because forward
decl is satisfied by full def).

---

## 3. Migration sequence

### Step A (atomic, single commit) — type renames in API surface

Files: `hir_c_api.h` + `hir_c_api.cpp` + 8 `.c` TUs from §1a.

Changes:
- `hir_c_api.h`: replace `typedef void* HirBasicBlock` and
  `typedef void* HirCFG` with forward struct decls. Update ~10 API signatures
  (`HirBasicBlock` → `struct HirBasicBlock *`, `HirCFG` → `struct HirCFG *`,
  `HirBasicBlock *out` → `struct HirBasicBlock **out`).
- `hir_c_api.cpp`: update implementation signatures to match. Update
  `as_block(HirBasicBlock b)` to `as_block(struct HirBasicBlock *b)`. The
  `reinterpret_cast<BasicBlock*>` body stays.
- 8 `.c` TUs from §1a: insert `*` in local declarations.
  `HirBasicBlock block` → `struct HirBasicBlock *block`. ~50 lines total
  across 8 files. `clean_cfg.c` L75 `sizeof(HirBasicBlock)` becomes
  `sizeof(struct HirBasicBlock *)` (semantic preserved — pointer-array sizing).

This commit is atomic: the 8 TUs and the API header must change together or
nothing builds.

Verification: full build + 4-benchmark gate. Runtime behavior unchanged
(types renamed, calling convention preserved).

### Step B (per-TU cleanup, 5-10 commits)

**B-0 empirical finding (cfg.cpp, 2026-04-22):** Step B is NOT pure
"delete extern + add `hir_c_api.h` include" mechanical work. Per-extern
classification is required. Four classes:

| Class | Description                                         | Cleanup pattern |
|-------|-----------------------------------------------------|-----------------|
| **A** | Local extern name matches a decl in `hir_c_api.h`  | Delete extern → `#include "cinderx/Jit/hir/hir_c_api.h"`. No caller change. |
| **B** | Name-mismatch with canonical (e.g., local `hir_c_create_primitive_box_bool` vs canonical `..._reg`) | Smell — investigate: linker resolves by exact name, so a mismatch means either (i) local extern names a different real symbol than canonical decl, (ii) canonical decl is stale, or (iii) function is dead. Resolve THEN clean up via Class A pattern. |
| **C1**| Name not in `hir_c_api.h`, conceptually API surface | Add to `hir_c_api.h` first (canonical expansion), THEN apply Class A. |
| **C2**| Name not in `hir_c_api.h`, file-pair-scoped helper with its own header (e.g., `hir_cfg_get_rpo_c` lives in `hir_cfg_rpo_c.h`/`.c`) | Delete extern → `#include` the helper's OWN header. Don't pull in `hir_c_api.h` unless other Class A externs in same TU also need it. |

Class C2 is more common than initially scoped: `hir_cfg_get_rpo_c` alone
has 5+ §1b/non-§1b consumers via local extern. Replacing externs with
helper-own-header includes preserves modularity (TUs don't pull in the
full `hir_c_api.h` surface for one helper call).

**Per-TU procedure:**

1. Run `scripts/count_w25_b1b_tus.sh` to confirm TU is still in the
   target list.
2. For each local extern in TU: classify A/B/C1/C2 by grep of
   `hir_c_api.h` + the helper's own header (if any).
3. Resolve Class B smells before cleanup (separate investigation).
4. Apply per-class cleanup pattern.
5. Compile-only verify before commit.
6. Commit as small per-TU bundle. Recommended bundle granularity:
   - Standalone commit if TU has >25 externs (e.g., `builder_emit_c.c`)
   - Bundle 3-4 small TUs per commit otherwise.

**Sequencing recommendation (smallest first to validate pattern):**

| Phase | TUs                                                  | Rationale |
|-------|------------------------------------------------------|-----------|
| B-0   | `cfg.cpp` (1 extern)                                 | Validate pattern + classification on smallest C++ TU. |
| B-1   | `hir_basic_block_c.c`, `ssaify_c.c` (1 each), `licm_c.c` (3) | Small bundle to scale up. |
| B-2   | `resolve_kwargs_c.c` (5), `pass_output_type_c.c` (25) | Medium bundle. |
| B-3   | `builder_emit_c.c` (130)                             | Heavy hitter own commit. Likely contains the most Class B smells given size. |

Each commit is independently reversible. If one TU resists (signature
drift already accumulated, Class B smell complex), revert that commit
only and investigate.

### Step C (lint gate, single commit)

Add the W25 §3 grep check to `scripts/gate_phoenix.sh`:

```bash
# Detect local extern decls of API functions outside hir_c_api.h
violators=$(grep -rE '^extern[[:space:]]+.*hir_(c_|cfg_|block_|bb_|edge_|func_|instr_)' \
    Python/jit/hir/ \
    --include='*.c' --include='*.cpp' \
    --exclude='hir_c_api.h' --exclude='hir_basic_block_c.h' --exclude='builder.cpp')
if [ -n "$violators" ]; then
    echo "W25 LINT FAIL: extern decls of API functions outside canonical headers:"
    echo "$violators"
    exit 1
fi
```

`builder.cpp` exception per `docs/w25-typed-bridges.md` §3 (legitimate
`extern "C"` bridge implementations).

---

## 4. Bridge inventory

### API functions whose signatures change in Step A (~10)

`hir_c_api.h` — `HirBasicBlock` and `HirCFG` parameters/returns:
- `hir_cfg_get_rpo`
- `hir_cfg_blocks_first`
- `hir_cfg_blocks_next`
- `hir_block_empty`
- `hir_block_id`
- `hir_block_first`
- `hir_block_next`
- `hir_block_terminator`
- `hir_block_append`
- `hir_block_pop_front`
- `hir_block_in_edges_count`
- `hir_block_fixup_phis`
- `hir_block_back`
- `hir_instr_successor`
- `hir_branch_target`

(Final list authoritative from `hir_c_api.h` post-Step-A grep.)

### TUs cleaned up in Step B (7 with lint-pattern externs, 10 type-only)

Authoritative count and list: run `scripts/count_w25_b1b_tus.sh` at the
HEAD where Step B will land. Lint-pattern target = 7 TUs (post-Step-A
baseline at `e6a8a2d0fb`). See §1b for the per-TU table and methodology
note (broad-grep 9 vs lint-pattern 7 distinction).

Step B work is concentrated in `builder_emit_c.c` (130 of ~166 lint-pattern
externs ≈ 78%). The 10 type-only TUs benefit from Step A's type-rename
automatically; Step B may be a no-op for them.

### Out-of-scope dual semantics

`HirInstr`, `HirRegister`, `HirFunction` are also `typedef void*` in
`hir_c_api.h` AND in 4 other headers (`assignment_c.h`, `dominator_c.h`,
`refcount_env_c.h`, `ssa_check_c.h`, `ssaify_c.h`). These typedefs are all
**identical**, so they don't conflict — but they're redundant and they keep
the type as void* (no struct upgrade path). Three options:

1. Leave alone (acceptable — no current bug, no current blocker).
2. Strip the redundant typedefs from the 5 secondary headers, leave
   `hir_c_api.h` as the single source of truth (small cleanup, ~10min).
3. Upgrade to opaque struct pointers like HBB (needs corresponding struct
   definitions, which don't currently exist for these — bigger work).

Recommendation: file as future "W25b" cleanup. Not blocking Tier 6 INVOKE_*.

---

## 5. Falsification

Three orthogonal checks, all must pass post-migration:

### 5.1 Dual-include compile test

**Goal:** structural assertion that the typedef collision is gone — any TU
can include both canonical headers without conflict. Compile-only test (no
runtime behavior).

**File:** `Python/jit/hir/w25_dual_include_check.c` (note: NOT `test_*`
prefix — `jit_build/CMakeLists.txt:131` excludes `test_*` from JIT_SOURCES,
so a `test_` name would silently not be compiled. Implemented as
`w25_dual_include_check.c` so it compiles automatically with every JIT
build.)

**Content:**
```c
/*
 * test_w25_dual_include.c — W25 §5.1 falsification artifact.
 *
 * Compile-only test that hir_c_api.h and hir_basic_block_c.h coexist in
 * the same TU without typedef collision. Pre-W25, this file would not
 * compile because HirBasicBlock + HirCFG had two incompatible typedefs.
 * Post-W25 Step A, both headers can be included; this TU validates that
 * structurally on every build.
 *
 * If this file ever fails to compile, W25 canonicalization has regressed.
 */

#include "cinderx/Jit/hir/hir_c_api.h"
#include "cinderx/Jit/hir/hir_basic_block_c.h"

/* Dummy function exercising both API surfaces with the canonical types. */
int test_w25_dual_include_id(struct HirBasicBlock *bb) {
    return hir_bb_id(bb);  /* hir_basic_block_c.h API */
}

/* Dummy function exercising hir_c_api.h API with canonical struct ptr. */
struct HirBasicBlock *test_w25_dual_include_first(struct HirCFG *cfg) {
    return hir_cfg_blocks_first(cfg);  /* hir_c_api.h API */
}
```

**Build integration:**
- File lives in `Python/jit/hir/` and is auto-picked-up by the JIT_SOURCES
  glob in `jit_build/CMakeLists.txt`. **No CMakeLists amendment required**
  given the non-`test_*` naming.
- The functions above are never called at runtime — they exist to force
  the compiler to type-check both API surfaces with the canonical
  struct-pointer types. Linker may DCE them; that's fine.

**Pass criterion:** the file compiles cleanly (zero warnings, zero
errors) as part of `scripts/build_phoenix.sh`. Verified by build success
plus a one-line check in `gate_phoenix.sh` confirming the .o is produced.

**Fail mode + diagnostic:** if Step A regresses (someone re-introduces
`typedef void* HirBasicBlock` in `hir_c_api.h`), this file fails to
compile with "conflicting types for HirBasicBlock". The build halts; the
gate flags W25 regression by file path.

**Why a TU and not a CPython unittest:** the assertion is at the C
preprocessor + type-system level, not runtime behavior. CPython unittest
infrastructure runs after build; this needs to fail at compile time. A TU
compiled into the JIT module is the right granularity.

**Implemented at:** commit `3404a81192` (push 66 bundle).

**Scope (~10min for generalist, complete):**
1. Create `Python/jit/hir/w25_dual_include_check.c` with content above.
2. Auto-picked-up by JIT_SOURCES glob (no CMakeLists change).
3. Verify compile is part of normal build.
4. Commit per supervisor [chat L2113] option (A): bundled with
   `count_w25_b1b_tus.sh` script as 2-commit Step A follow-up bundle.

### 5.2 Lint gate active

Run `gate_phoenix.sh` and confirm Step C lint is exercised. Manually inject a
spurious `extern` decl in a test branch; verify gate fails. Revert.

### 5.3 Signature-drift reproducer (regression test)

Mutation test: in a throwaway branch, change one parameter type in
`hir_c_api.h` (e.g., `HirInstr` → `HirInstrLayout *` in
`hir_block_append`). Build. Confirm the drift is now caught at compile time
in every consuming TU. (Pre-W25, this would silently link.) Revert.

This validates that Step B's cleanup (deleted forward decls) actually closes
the drift surface.

**Empirical result (2026-04-22, post-Step-B HEAD `ab0764195e`):** PASS
end-to-end.

- PRE-STEP-B baseline (commit `911e17359b`): mutation `hir_c_insert_before`
  +3rd arg in both `.h`+`.cpp`, `BUILD_EXIT=0` — drift undetected, §1b
  surface real (per `docs/w25-step-b-mutation-baseline.txt`).
- POST-STEP-B re-run (commit `ab0764195e`): same mutation,
  `BUILD_EXIT=2` with 6 `too few arguments` compile errors at:
  - `licm_c.c:244:50`
  - `pass_output_type_c.c:981:53`, `989:48`, `1054:59`
  - `simplify_c.c:40:57`, `2829:54`
- Acceptance per §5: PASS (PRE BUILD_EXIT=0 + POST BUILD_EXIT≠0 with errors
  at expected §1b TUs). §1b drift surface CLOSED for `HirBasicBlock*` /
  `HirCFG*` typed args.

**Caveat per §7:** PASS is for struct-typed args ONLY. void*-aliased
handles (`HirInstr`/`HirRegister`/`HirFunction`) remain drift-prone — a
mutation on those args would compile silently both pre- AND post-Step-B
because the type-checker can't distinguish void* from void*. Honest framing
of the §5.3 result: "Step B closes drift surface for struct-typed handle
args" — not "all drift surface" until W25b lands.

---

## 6. Rollback

Each step is independently revertable:
- Step A: `git revert` restores `void*` typedefs and old signatures. 8 TUs
  build again as before.
- Step B: per-TU. Revert any single TU restores the local forward decls.
- Step C: revert removes the lint check.

Risk is concentrated in Step A's atomicity. If Step A's commit causes
unexpected test failures, revert is clean (single commit). Likelihood low —
the change is mechanical and runtime behavior is unchanged.

---

## 7. What this does NOT solve

- **`HirInstr`/`HirRegister`/`HirFunction` void* aliases — DRIFT REMAINS POSSIBLE
  for these handle types** (per pythia #83 2026-04-22). The type-checker catches
  signature drift only when the changed parameter type is structurally distinct
  from the old type. After W25, `HirBasicBlock`/`HirCFG` are struct pointers —
  any drift on these args fails compile (proven empirically by §5.3 POST-STEP-B
  re-run). But `HirInstr`/`HirRegister`/`HirFunction` are STILL `typedef void*`
  in `hir_c_api.h`. A signature change on these args (e.g. swapping two
  `HirInstr` parameters) compiles silently because both sides are void*. The
  empirical "Step B closes drift surface" claim holds for struct-typed handles
  ONLY; void*-aliased handles remain at push-51-class risk. Closing this
  requires W25b: promote `HirInstr`/`HirRegister`/`HirFunction` to canonical
  struct pointers (same pattern as W25 Step A for `HirBasicBlock`/`HirCFG`).
  W25b is NOT scheduled — flagged as Tier 6+ scope contingent on next push-51-
  class incident or planned hardening sprint.
- `PhxTranslationContext` vs `TranslationContext` ABI compatibility (still
  per-struct `static_assert` in `builder.cpp:1020-1022`).
- C++ stub miscompile class (push 51 root cause was in the C++ stub, not the
  C body). Addressed by per-bridge LITE SPEC discipline + W26 gate hardening.
- New bridge introductions during Tier 6 INVOKE_* — they get the canonical
  type system from day 1 once W25 lands, FOR struct-typed args. void*-typed
  args (HirInstr/HirRegister/HirFunction params) remain drift-prone until W25b.
- Function-scoped (intra-function) extern decls — the `^extern` lint pattern
  in §3 Step C is line-anchored and misses indented function-scoped externs.
  Caught empirically in B-5 (2 hir_bb_* externs surfaced via compile error
  not lint). Step C lint pattern revision proposed: drop `^` anchor or use
  `(^|[[:space:]]+)extern` matcher. Fix scheduled in Step C lint commit.

---

## 8. Connection to Tier 6

Per pythia #78 #3 projection, Tier 6 INVOKE_* (3-4 method conversions) needs
~15-20 new bridges. With W25 protection in place:
- Each new bridge declared once in `hir_c_api.h` with canonical types.
- No local extern workarounds (lint gate blocks them).
- Signature drift impossible (single declaration site, struct types).

Without W25, those ~15-20 bridges would each follow the §1b pattern
(forward-decl workaround), each adding to the signature-drift surface.
Push-51 risk class scales linearly with new-bridge count.

This is the load-bearing reason Alex picked Option B.

---

## 9. Sequencing within this workstream

1. **Spec review** (now): supervisor + medic + gatekeeper review this doc.
   Substantive feedback before generalist starts implementation.
2. **Step A** (~30min implementation + dual-arch verify cycle).
3. **Step B** (per-TU, ~30min implementation + dual-arch verify cycle).
4. **Step C** (~10min implementation + dual-arch verify cycle).
5. **Falsification §5.1-5.3** (~20min — mutation test is real-time work).
6. **Workstream close**: update `docs/w25-typed-bridges.md` §7 from PARKED
   to RESOLVED with pointer to this doc.

Total: ~90-120min focused work, 3 push gates.

Tier 6 INVOKE_* unblocked at workstream close.
