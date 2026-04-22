# W25 — Typed Bridge Declarations + Header Discipline

**Status:** Design pre-draft (theologian, 2026-04-22 11:56Z, supervisor-authorized)
**Owner:** theologian (design), generalist (implementation)
**Schedule:** Implementation post-Tier-5-close; design ready on backlog now
**Origin:** pythia #68 + push 51 SEGV (first observed instance of the predicted class)

---

## 1. Problem statement

Bridge functions between C and C++ in the JIT (e.g. `hir_c_create_check_exc_reg`) are declared canonically in `Python/jit/hir/hir_c_api.h` using the typedefs:

```c
typedef void* HirInstr;
typedef void* HirRegister;
```

The header has 252 such canonical declarations. However, several `_c.c` translation units add **local `extern` declarations** for the same functions instead of including `hir_c_api.h`:

| TU                       | Local extern count |
|--------------------------|--------------------|
| `builder_emit_c.c`       | 93                 |
| `simplify_c.c`           | 8                  |
| `pass_output_type_c.c`   | 3                  |
| `resolve_kwargs_c.c`     | 2                  |
| **Total**                | **106**            |

Each local extern is a potential **signature drift** point. The C linker matches by symbol name only — it does **not** check that local extern signatures agree with the canonical definition. A local extern with the wrong argument count, wrong types, or wrong return type compiles silently and links silently. The runtime result is undefined behavior.

### 1.1 Push 51 as the first observed instance

In commit `05e2b8821a` (push 51 candidate, emitYieldValue → C), generalist added 4 new local extern decls in `builder_emit_c.c`:

```c
extern void *hir_c_create_call_cfunc_reg(size_t n_operands, void *dst,
                                          int32_t func_enum, void **operands);
extern void *hir_c_create_check_exc_reg(void *dst, void *src);
extern void *hir_c_create_yield_value_reg(void *dst, void *src, void *frame_state);
extern void *hir_c_create_yield_from_reg(void *dst, void *send_value,
                                          void *iter, void *frame_state);
```

These signatures **happened to match** the canonical decls in `hir_c_api.h` (verified by inspection — see theologian 11:43:50Z chat). The push 51 SEGV is therefore **not** a signature-drift instance; bisection eventually localized to the C++ stub (testkeeper 11:49:45Z), and the C body was found innocent.

But the **architectural risk** is unchanged: 106 local extern decls means 106 places where a future commit could introduce signature drift that compiles cleanly and crashes at runtime. Push 51 demonstrated the pattern's existence; the next instance may not be so lucky.

## 2. Two-layer protection

### Layer A — Opaque struct typing (cosmetic)

Replace:
```c
typedef void* HirInstr;
typedef void* HirRegister;
```

with:
```c
typedef struct HirInstrOpaque *HirInstr;
typedef struct HirRegisterOpaque *HirRegister;
```

The C++ implementation does not need to define these structs (they remain incomplete types) — only declared. C++ implementation files cast to/from the real types (`Instr*`, `Register*`) internally.

**What this catches:** accidental passing of a `void*` (or wrong-type pointer) where `HirInstr` is expected. The C compiler will warn or error on incompatible pointer types.

**What this does NOT catch:** wrong argument count, wrong scalar types, signature drift in local extern decls. Linker still matches by name only.

**Cost:** ~30min mechanical replacement in `hir_c_api.h`; potentially additional cast-cleanups in 4-6 implementation files. Low-risk.

### Layer B — Header discipline (the real protection)

**Rule:** Any `.c` TU that calls `hir_c_*` functions MUST `#include "cinderx/Jit/hir/hir_c_api.h"` and MUST NOT add local `extern` declarations for those functions.

**Why this catches signature drift:** Once the canonical decl is in scope via header inclusion, any local `extern` with a mismatched signature triggers a redeclaration error at compile time. The compiler enforces signature agreement.

**Cost per TU:**
- `builder_emit_c.c`: add `#include`, delete 93 local extern lines. ~10min mechanical, 0 logic changes.
- `simplify_c.c`: add `#include`, delete 8 local extern lines. ~5min.
- `pass_output_type_c.c`: add `#include`, delete 3 local extern lines. ~5min.
- `resolve_kwargs_c.c`: add `#include`, delete 2 local extern lines. ~5min.
- **Total:** ~25min implementation, 1 commit per TU (4 commits) for clean bisection if any breaks.

**Risk:** including `hir_c_api.h` may pull in headers that conflict with TU-local includes. **EMPIRICALLY VALIDATED 2026-04-22 13:19Z**: this risk is REAL and load-bearing. See §4.1 below.

## 3. Lint enforcement (post-migration)

Add to the gate-before-push checklist (gatekeeper) and `gate_phoenix.sh`:

```bash
# Flag local extern decls for canonical bridge functions outside hir_c_api.h
violators=$(grep -rln '^extern.*hir_c_create\|^extern.*hir_builder_emit' \
    Python/jit/hir/ \
    --include='*.c' --include='*.cpp' \
    | grep -v 'hir_c_api\|builder.cpp')
if [ -n "$violators" ]; then
    echo "BLOCK: local extern decls found in: $violators"
    exit 1
fi
```

**Exception:** `builder.cpp` legitimately uses `extern "C"` decls for the C-bound bridges it implements (e.g., `hir_builder_temps_alloc_stack`). The grep filter excludes it.

## 4. Migration order — REVISED 2026-04-22 13:21Z

**Original plan FAILED in implementation attempt.** See §4.1 for what we learned. Revised plan below.

### 4.1 Implementation lessons (2026-04-22 attempt)

**Layer A is NOT source-compatible in C++** (corrected from §1.1 original claim):
- `typedef struct HirInstrOpaque *HirInstr` requires explicit `static_cast` at every C++ call site that passes `Instr*` directly
- `builder.cpp` alone has 137 such call sites
- Total Layer A C++ blast radius: ~150+ static_cast additions across 5+ TUs
- Risk of cast-typo silently compiling is non-trivial (`static_cast<HirInstr>(wrong-type-pointer)` round-trips via `void*` at C boundary)
- **Decision:** Layer A skipped, deferred to W25b future work

**Layer B prelude (header reconciliation) FAILED:**
- Attempt: change `hir_c_api.h` `HirCFG`/`HirBasicBlock` typedefs from `void*` to opaque struct (matching `hir_basic_block_c.h`'s canonical form), remove conflict
- Failure: 5+ unrelated `.c` TUs use `HirCFG`/`HirBasicBlock` as `void*` (pointer arithmetic, comparisons, casts) — they break under struct typedef
- Plus: `hir_c_api.h` has function `hir_c_unlink` whose decl conflicts with `hir_basic_block_c.h`'s version once both are in scope
- Net: prelude is a 60+ min multi-TU refactor, not a small fix
- **Decision:** prelude deferred; full architectural fix needed

**The real cost of Layer B is much higher than originally estimated** because the canonical-header conflict between `hir_c_api.h` and `hir_basic_block_c.h` blocks naive `#include` adoption.

### 4.2 Revised execution plan (post-Tier-5-close)

W25 is a **single architectural cycle** (~75-90min total), not a series of mechanical TU migrations:

1. **Header split** (~30min): split `hir_c_api.h` into:
   - `hir_c_api_funcs.h` — function declarations only (no typedefs)
   - `hir_c_api_types.h` — typedefs only (the `void*` aliases for `HirInstr`, `HirRegister`, `HirFunction`, `HirRegUses`, etc.; NOT `HirCFG`/`HirBasicBlock` since those are canonical in `hir_basic_block_c.h`)
2. **Reconcile function-decl conflicts** (~15min): identify and resolve all double-declarations between `hir_c_api.h` and `hir_basic_block_c.h` (e.g., `hir_c_unlink`).
3. **Layer B per TU** (~30min, 4 commits): each TU includes `hir_c_api_funcs.h` + its preferred typedef header. No conflict.
4. **Lint gate** (~10min, 1 commit): grep BLOCK in `gate_phoenix.sh`.

Total: 6+ commits, ~75-90min. ACK-before-each.

### 4.3 Layer A status

Deferred indefinitely as W25b. Future option if defense-in-depth via conversion operator OR systematic `static_cast` review is wanted. Cost: 60-90min on top of W25 base. Not currently scheduled.
3. **Lint gate** (single commit): add the grep check to `gate_phoenix.sh`. Future violations BLOCK at gate time.

**Total cost:** ~1hr active work + 6 commits. Fits in a single Tier-5-close session.

## 5. Open question for review

**Should we ship Layer A + B together, or just Layer B?**

Layer A alone catches few real bugs (most signature drift is in argument count, not pointer-type confusion). Layer B catches all signature drift via redeclaration error.

Layer A's value is **defense in depth** + **future readability** (signatures self-document handle types). Cost is low (~30min).

Recommendation: ship A+B together. The marginal cost of A is small, and once the convention exists, future bridge additions get the strong typing for free.

## 6. What W25 does NOT solve

- **C++ stub miscompile** (push 51 root cause is in the C++ stub, not the C body). W25 hardens the C-side bridge contract; it does not address C++-side stub correctness. A separate concern (W26?) would be: "C++ stub design template" — pattern for the ~50 remaining emit-method ports, with invariants enumerated up-front (similar to project_bridge_spec_template.md but for the stub side).
- **ABI compatibility across C and C++ struct layouts** (e.g., `PhxTranslationContext` vs `TranslationContext`). Currently enforced via `static_assert` in `builder.cpp:1020-1022`. W25 does not extend this discipline; it remains per-struct manual.
- **Header-included-but-still-local-extern violations** (defense bypass). Caught by the lint in §3.

## 7. PARKED — 2026-04-22 13:48Z

**Status:** PARKED. W25 implementation attempt 2026-04-22 surfaced a deeper architectural blocker that W25 alone cannot solve. Supervisor decision per Alex autonomy directive (intermediate scope, team-internal, not Alex-escalation).

### 7.0 Workstream metadata (per filed-workstream discipline, pythia #76 #3)

- **Owner:** theologian (design) + generalist (implementation)
- **Schedule:** entry-trigger = Tier 5 ZERO C++ complete + HirBasicBlock dual-semantics canonicalization workstream begun. Exit-trigger = W25 Layer B per-TU migrations land + lint gate active.
- **Falsification:** Layer B incomplete OR signature-drift recurs in C TUs after declared-complete = workstream re-opens. Concrete reproducer: any TU with `^extern.*hir_c_create` outside `hir_c_api.h` (or its split-headers) post-completion proves W25 unfinished.
- **Closure criteria:** all 4 W25-target TUs (`builder_emit_c.c`, `simplify_c.c`, `pass_output_type_c.c`, `resolve_kwargs_c.c`) include canonical header + zero local extern decls + lint gate in `gate_phoenix.sh` BLOCKs violations.
- **Scope-honest acknowledgment:** parking ≠ done. Carrying cost noted in §7.4 (12-18 new signature-drift surfaces accepted during Tier 5 close).

### 7.1 What blocked W25

`HirBasicBlock` has **two incompatible semantics in the codebase**:
- `hir_c_api.h`: `typedef void* HirBasicBlock` — opaque-handle, 8 bytes, used as pointer
- `hir_basic_block_c.h`: `typedef struct HirBasicBlock {...} HirBasicBlock` — value-struct, 96 bytes

These cannot coexist when both headers are pulled into the same TU. W25 Step 1.5 (dead-code cleanup of duplicated `hir_c_unlink`) eliminated the function-decl conflict, but the typedef conflict on `HirBasicBlock` REMAINS at the architectural level.

`clean_cfg.c` is the explicit dependency: `malloc(rpo_cap * sizeof(HirBasicBlock))` allocates an array sized for opaque-handle pointers, not value-structs. Other 4 .c TUs (`dead_code_elimination.c`, `dynamic_comparison_elimination.c`, `func_type_checks_c.c`, `guard_removal.c`) have mechanical-only void* usage that COULD be migrated, but `clean_cfg.c` requires logic redesign.

### 7.2 Pre-Step-1 audit was insufficient

The W25 §4 "header split" plan was based on an incomplete model of cross-TU header dependencies. Specifically:
- §4 Q(d) answer (theologian 13:39:37Z) claimed the 5 .c TUs "continue to work because hir_c_api.h becomes a thin facade, canonical typedefs come from hir_basic_block_c.h"
- This conflated **Layer B scope** (TUs with local extern decls — 4 files) with **facade-cascade scope** (ALL TUs that #include hir_c_api.h — many more files, including the 5 .c TUs that USE HirBasicBlock as void*)
- Step 1's facade #include of _funcs.h pulls hir_basic_block_c.h into ALL transitive includers, breaking their void* usage

**Lesson for future architectural-debt fixes:** scope estimates need a "discoverable-debt scan" as an explicit audit step BEFORE committing to scope. Pre-Step-1 audit should have:
1. Listed all TUs that include the modified header (transitively)
2. Categorized each TU's usage of the changed types (mechanical vs logic-class)
3. Surfaced the dual-semantics ambiguity as a blocker BEFORE Step 1 began

### 7.3 Tier 6 architectural cleanup workstream

W25 (typed bridges) is REQUEUED for Tier 6 cleanup phase, contingent on prior resolution of the `HirBasicBlock` dual-semantics issue. That resolution is itself a separate architectural workstream (~3-4hr scope per supervisor 13:48:43Z option B):
- Choose ONE canonical semantic for `HirBasicBlock`: either pointer-typedef (matches current opaque-handle usage in 5+ TUs + bridge functions) OR value-struct (matches `hir_basic_block_c.h`)
- Migrate all dissenting TUs to the canonical form
- Then W25 Layer B becomes naturally implementable

### 7.4 Carrying cost during defer

Tier 5 stream A continues without W25 protection. Estimated 12-18 new signature-drift surfaces will be introduced across the remaining ~6 emit-method conversions, all under `void*` extern-decl convention. Partial mitigations in effect:
- W26 gate-hardening (`scripts/build_phoenix.sh` force --clean after recent compile-fail)
- Dual-arch --clean discipline at every push gate
- Per-instance W23 wrapper verification (Phase B)
- Per-bridge LITE SPEC TEMPLATE (D-1776823737) for all new bridge introductions

If a push-51-class signature-drift bug surfaces in remaining 6 methods, the architectural fix (option B) gets escalated to top of backlog immediately.

### 7.5 What stays from this attempt

- `c9c0318b78` (W25 Step 1.5): dead-code removal of unused 1-arg `hir_c_unlink` + dedupe of redundant `hir_instr_unlink` in `hir_c_api.h`. Independent semantic win, NOT reverted.
- This document (`docs/w25-typed-bridges.md`): preserved as the canonical W25 design + lessons-learned record. §1.1, §4.1, §4.2 capture the iterative learning; §7 captures the parking decision.
