# W25b — `HirInstr` / `HirRegister` / `HirFunction` Typedef Promotion

**Status:** ACTIVE (post-W25, pre-Tier 6 INVOKE_* Phase 2). Authorized 2026-04-22
per supervisor [chat L2389] post-empirical confirmation that void*-typedef drift
class is real (push 78 §5.3 type-only mutation BUILD_EXIT=0).

**Scope:** Promote three `typedef void*` aliases (`HirInstr`, `HirRegister`,
`HirFunction`) to distinct opaque struct-pointer types so that the C type
system catches signature drift on these handle args at compile time. Mirror
of W25 Step A pattern for `HirBasicBlock`/`HirCFG`, but minimal-cost variant:
forward-declare-only (no real struct definition required).

**Owner:** theologian (this spec) + generalist (implementation).

**Estimated cost:** ~50min (typedef change + ~10 file updates + dual-arch verify).
Tighter than W25 Step A (~30min for 2 types + 10 files) because no caller-side
`*` insertion needed — handle-as-value semantics preserved via opaque-pointer
typedef.

---

## 1. The empirically confirmed drift class

W25 closed arity-mismatch drift on canonical-included paths
(`docs/w25-hbb-canonicalization.md` §5.3 PASS, push 76 baseline + push 76
post-Step-B). But W25 left a NARROWER drift class open: **type-only drift on
`typedef void*` handles**.

Empirical proof — push 78 §5.3 type-only mutation
(`docs/w25-step-b-mutation-baseline.txt` POST-STEP-B-FINAL section):

- Mutation: `hir_c_insert_before` 2nd arg `HirInstr` → `struct HirBasicBlock *`
  in BOTH `.h`+`.cpp` (no arity change, pure type swap)
- Pre-mutation callers pass `HirInstr` (= `void*`)
- Post-mutation canonical decl expects `struct HirBasicBlock *`
- Build result: **BUILD_EXIT=0 — compile-clean**, drift undetected
- Root cause: C99 §6.3.2.3 implicit `void*` ↔ object-pointer conversion silently
  masks the type swap

Linker resolves; runtime would mis-interpret bits if the bits diverged
(here pointers happen to be 8 bytes either way, so no immediate UB — but if a
future signature change swaps two void*-handle args of different types, the
silent bit-misinterpretation would surface as a runtime miscompile).

**Risk class:** push-51 ABI-mismatch class scales with each new bridge that
takes any `HirInstr`/`HirRegister`/`HirFunction` arg. Tier 6 INVOKE_* Phase 2
introduces ~15-20 new bridges; without W25b, those bridges accept `void*`
handles and join the open drift surface.

---

## 2. Canonical resolution

**Make `HirInstr`, `HirRegister`, `HirFunction` distinct opaque struct-pointer
types via forward-decl-only.**

`hir_c_api.h`:
```c
/* Forward declarations — opaque struct tags, no struct definition needed. */
struct HirInstrOpaque;
struct HirRegisterOpaque;
struct HirFunctionOpaque;

/* W25b canonical: handles are typed pointers, not void*.
 * Compiler treats them as distinct types — drift between them caught at compile. */
typedef struct HirInstrOpaque *HirInstr;
typedef struct HirRegisterOpaque *HirRegister;
typedef struct HirFunctionOpaque *HirFunction;
```

### Why opaque-struct-pointer (forward-decl-only)

- **Distinct types**: each `*Opaque` tag is a fake forward-declared struct that
  is never defined. The compiler treats `HirInstr` and `HirRegister` as
  unrelated pointer types — assigning one to the other or mixing in a function
  call is a compile error.
- **Opaque**: the struct is forward-declared but never defined, so consumers
  can't accidentally dereference fields. The handle remains opaque.
- **No real struct needed**: unlike W25 Step A for `HirBasicBlock`/`HirCFG`
  (which already had real struct definitions in `hir_basic_block_c.h`),
  W25b doesn't need `hir_instr_c.h`'s `HirInstrLayout`/`HirRegisterLayout`/
  `HirFunctionLayout` to be renamed. Those stay as direct-access struct types
  for the .c TUs that need field access.
- **Caller-side compatibility preserved**: code that uses `HirInstr` as an
  opaque handle (passing around, storing in arrays) continues to compile
  unchanged. The typedef's ABI is still 8 bytes, same as `void*`. Only
  cross-type assignments and call-site mismatches now error.

### What it does NOT change

- `HirInstrLayout` / `HirRegisterLayout` / `HirFunctionLayout` definitions in
  `hir_instr_c.h` — these are direct-access structs for field access, separate
  concern from the canonical opaque handle.
- ABI / runtime behavior — pointer size and calling convention identical to
  pre-W25b.
- Existing per-TU `static_cast<X>(handle)` patterns in `hir_c_api.cpp` — those
  continue to work, since `HirInstr` post-W25b is `struct HirInstrOpaque *`
  and `static_cast<Instr*>(struct HirInstrOpaque *)` is legal C++.

---

## 3. Migration sequence

### Step A' (atomic, single commit) — typedef promotion + redundancy cleanup

Files: `hir_c_api.h` + 5 secondary headers with redundant typedefs +
`hir_c_api.cpp` if any cast surface needs adjustment.

Changes:
- `hir_c_api.h`: replace 3 `typedef void* X` with forward-decl + struct-ptr
  typedef (per §2 sketch).
- 5 secondary headers (`assignment_c.h`, `dominator_c.h`, `refcount_env_c.h`,
  `ssa_check_c.h`, `ssaify_c.h`): DELETE redundant `typedef void* HirFunction`
  / `typedef void* HirRegister` lines. They were duplicated decls. Now that
  `hir_c_api.h` has the canonical typedef, secondary headers can remove their
  duplicates and (if needed) `#include "cinderx/Jit/hir/hir_c_api.h"` for the
  typedef.
- `hir_c_api.cpp`: any `static_cast<X*>(handle)` may need adjustment if the
  destination type assumed `void*` source. Likely no change needed — C++ allows
  `static_cast` between `struct X *` and other pointer types via `void*`
  intermediate.

This commit is atomic: typedef change + redundant decl removal + impl
adjustments must land together or duplicated typedefs would conflict.

Verification: full build + dual-arch + 4-benchmark gate. Runtime behavior
unchanged.

### Step B' (per-TU caller cleanup, expected zero work)

Most callers use `HirInstr` etc. as opaque handles (pass-around, store, return).
These compile unchanged after Step A'. No per-TU edits expected.

If Step A' build surfaces caller-side compile errors (e.g., `HirInstr` assigned
directly to `HirRegister` somewhere — drift was real and compiled previously),
Step B' fixes those sites:
- Genuine drift bug: fix the call site to pass the right type
- Intentional type punning: add explicit cast

Likely scope: ~0-5 sites (W25b is meant to expose drift, not introduce new
cleanup work). Generalist surveys post-Step-A' build output.

### Step C' (no new lint, existing W25 lint applies)

Step C from W25 (`scripts/gate_phoenix.sh` lint pattern) catches local extern
decls. W25b doesn't add new extern surface — it changes typedef shape. Existing
lint suffices.

---

## 4. Bridge inventory

### Typedef sites changed in Step A'

`hir_c_api.h` — 3 typedefs replaced + 3 forward decls added.

5 secondary headers — 1-2 redundant typedefs removed each:
- `assignment_c.h`: -2 (HirFunction + HirRegister)
- `dominator_c.h`: -1 (HirFunction)
- `refcount_env_c.h`: -1 (HirFunction)
- `ssa_check_c.h`: -1 (HirFunction)
- `ssaify_c.h`: -1 (HirFunction)

Total: 7 files modified.

### Caller surface analysis

`HirInstr` / `HirRegister` / `HirFunction` referenced in ~30 .c/.cpp/.h files
across `Python/jit/hir/`. Most uses are opaque handle pass-around (function
args, return values, local storage). These compile unchanged post-Step-A'.

Generalist runs `scripts/build_phoenix.sh` post-Step-A' edit and surfaces any
caller-side compile errors. Each error is either a genuine drift bug (fix the
type) or an intentional cast (add `static_cast` / `reinterpret_cast`).

### Out-of-scope

- `HirCFG` / `HirBasicBlock` — already canonical struct pointers per W25 Step A
- `HirInstrLayout` / `HirRegisterLayout` / `HirFunctionLayout` — direct-access
  structs for .c TUs; remain unchanged
- Other typedefs in the codebase (`PyObject*` handle types etc.) — out of W25b
  scope

---

## 5. Falsification

### 5.1 §5.3 type-only re-run post-W25b (PRIMARY acceptance)

**Goal:** confirm that the same TYPE-ONLY mutation that BUILD_EXIT=0'd at
HEAD `dac46b963e` (push 78) now BUILD_EXIT=2 post-W25b.

**Procedure:** repeat `docs/w25-step-b-mutation-test.md` §4 mutation
(`hir_c_insert_before` 2nd arg `HirInstr` → `struct HirBasicBlock *` in both
`.h`+`.cpp`) at post-W25b HEAD. Build. Capture `BUILD_EXIT`.

**Acceptance:**
- BUILD_EXIT=2 with compile errors at the 6 §1b call sites that the W25 §5.3
  arity-mutation already exposed (licm_c.c:244, pass_output_type_c.c:981/989/
  1054, simplify_c.c:40/2829) → W25b CLOSES the void*-typedef drift class
- BUILD_EXIT=0 (UNEXPECTED) → W25b implementation incomplete; investigate

**Why this works:** post-W25b, `HirInstr` is `struct HirInstrOpaque *`, distinct
from `struct HirBasicBlock *`. C99 implicit conversion no longer applies between
the two pointer types — compiler must error.

### 5.2 Distinct-handle cross-assignment test (regression test)

Add `Python/jit/hir/w25b_distinct_handles_check.c`:

```c
/*
 * w25b_distinct_handles_check.c — W25b §5.2 falsification artifact.
 *
 * Compile-only test that HirInstr / HirRegister / HirFunction are distinct
 * pointer types post-W25b. Pre-W25b, all three were void* — assignments
 * between them compiled silently. Post-W25b, cross-assignment must fail.
 *
 * If this file ever fails to compile (cross-assignments are ERRORS), W25b
 * is intact. The negative-space pattern: we WANT the marked sections to
 * fail compile if uncommented; their commented-out state is the test.
 */

#include "cinderx/Jit/hir/hir_c_api.h"

void w25b_distinct_handles_check(HirInstr i, HirRegister r, HirFunction f) {
    /* These should ALL be compile errors post-W25b. Kept commented-out as
     * documentation of what W25b protects against. Uncommenting any line
     * MUST produce a compile error; if it doesn't, W25b regressed. */

    // HirInstr i_from_r = r;     /* W25b BLOCK: HirRegister → HirInstr */
    // HirInstr i_from_f = f;     /* W25b BLOCK: HirFunction → HirInstr */
    // HirRegister r_from_i = i;  /* W25b BLOCK: HirInstr → HirRegister */
    // HirRegister r_from_f = f;  /* W25b BLOCK: HirFunction → HirRegister */
    // HirFunction f_from_i = i;  /* W25b BLOCK: HirInstr → HirFunction */
    // HirFunction f_from_r = r;  /* W25b BLOCK: HirRegister → HirFunction */

    (void)i; (void)r; (void)f;  /* silence unused-param */
}
```

**Build integration:** auto-picked-up by JIT_SOURCES glob like
`w25_dual_include_check.c` (non-`test_*` naming dodges cmake exclude).

**Pass criterion:** the file compiles cleanly (commented-out lines silent).

**Negative-pass (manual periodic check):** uncomment one line, build, expect
compile error, revert. Documented procedure; not in CI.

---

## 6. Rollback

Step A' is a single commit; `git revert` restores `typedef void*` and removes
the 3 forward decls. Secondary header redundant typedefs come back (slightly
inconvenient but harmless). All callers compile again.

Risk concentration in Step A' atomicity. If unexpected caller-side compile
errors are extensive (>10 sites), consider whether they represent real drift
that should be fixed (proceed with Step B') vs intentional patterns we don't
want to disturb (revert Step A' and reassess).

---

## 7. What this does NOT solve

- Drift on direct-access struct pointers (`HirInstrLayout *` etc.) — these are
  used by .c TUs that read fields directly. Not a typical drift surface; the
  Layout structs are layout-validated via `static_assert` in
  `hir_instr_c_verify.cpp`.
- C++ stub miscompile class (push 51 root cause). Addressed by per-bridge LITE
  SPEC discipline.
- Drift between handle types and `void*` at C-bridge boundaries. C99 implicit
  conversion rules persist; W25b only protects intra-handle-type relationships.

---

## 8. Connection to Tier 6 INVOKE_* Phase 2

Per pythia L1969 + supervisor [chat L2389]: Tier 6 INVOKE_* introduces ~15-20
new bridges. With W25b in place:
- Each new bridge declares typed `HirInstr`/`HirRegister`/`HirFunction` args
- Cross-handle drift caught at compile time (C99 implicit conversion no longer
  silently masks)
- Combined with W25 (struct-typed `HirCFG`/`HirBasicBlock` args) and Step C
  lint (no local extern reintroductions), the API surface is comprehensively
  drift-protected for the new bridge class.

Without W25b, those 15-20 new bridges would inherit the open void*-typedef
drift surface — push-51 risk class scales linearly.

---

## 9. Sequencing

1. **Spec review** (now): supervisor + gatekeeper review this doc.
2. **Step A'** (~30min): typedef promotion + redundant decl cleanup +
   impl adjustments. Single atomic commit.
3. **Step B'** (~10-15min): per-TU caller cleanup if any compile errors
   surface. Expected near-zero scope.
4. **§5.2 distinct-handles check** (~10min): add
   `w25b_distinct_handles_check.c` to the build (parallel to `w25_dual_include_check.c`).
5. **§5.1 type-only re-run** (~5min): testkeeper repeats the push 78 mutation
   at post-W25b HEAD; BUILD_EXIT=2 expected.
6. **Workstream close**: update this doc's §5.1 with empirical result.

Total: ~60-70min focused work, 2-3 push gates.

Tier 6 INVOKE_* Phase 2 unblocks at workstream close.
