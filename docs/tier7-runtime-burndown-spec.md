# Tier 7 — Runtime Burndown Spec

**Status:** DRAFT (theologian, 2026-04-23, post-Batch-2-H 18→8 turning
point per supervisor [chat L2966] + theologian L2970 turning-point
framing). Concur per supervisor [chat L2971] (Z)+(Y) sequence.

**Owner:** theologian (spec) → generalist (impl) → testkeeper (gate)

**Estimated cost:** 5-10 sessions across 6 runtime files (~8284L
total, 318 methods).

---

## 1. Scope

### 1.1 In-scope (5 runtime files = Tier 7 burndown targets)

| File | Lines | Methods | Class shape |
|------|------:|--------:|-------------|
| builder.cpp | 4935 | 168 | HIRBuilder write-path (largest) |
| hir.cpp | 1437 | 85 | HIR core data structures |
| inliner.cpp | 705 | 10 | HIRInliner Pass class |
| preload.cpp | 569 | 29 | Preloader + PreloaderManager |
| builtin_load_method_elimination.cpp | 280 | 6 | BuiltinLoadMethodElimination Pass |
| **TOTAL** | **7327** | **244** | |

### 1.2 Out-of-scope — 3-class load-bearing infrastructure (per L2976 framing, terminology revised per pythia #88 verifier-reclassification critique)

| File | Lines | Class | Reason |
|------|------:|-------|--------|
| hir_c_api.cpp | 2722 | Boundary | C-bridge layer per L2837 cap (~17/200L used). It IS the C-bridge; eliminating defeats architectural purpose. |
| hir_instr_c_verify.cpp | 368 | Safety-net | Compile-time sizeof+offsetof static_asserts via friend struct per memory `feedback_verifier_pattern.md`. Caught Edge/BasicBlock/CFG/Register layout bugs historically. |
| printer.cpp | 957 | Diagnostic | HIRPrinter debug/log infrastructure (Print Function/CFG/BasicBlock/Instr/FrameState + 70+ per-instruction printers). Used for JIT_CHECK crash-dumps + JIT log_debug + compiled artifact dumping. Organizationally separable from runtime code paths. |

**Honest framing per W27e + Cat A/B precedent:** "ZERO C++ for active
runtime code paths; load-bearing infrastructure for safety-net (compile-time)
+ diagnostic (debug log) + boundary (C-bridge layer). All 3 classes
organizationally separable, none affects production correctness."

**Terminology note (per pythia #88 critique 2026-04-23):** original
framing used "intentional residue" — pythia correctly objected that
"residue" connotes "leftover, possibly safe to remove" which would
mislead future maintainers. Replaced with "load-bearing
infrastructure" to signal NOT-targets-for-removal, actively
load-bearing function. hir_instr_c_verify.cpp specifically has
historically caught Edge/BB/CFG/Register layout bugs per memory
`feedback_verifier_pattern.md` — describing it as anything that
suggests it's expendable would invite a future ARM64 LSE atomics or
3.14 frame layout change to silently bypass the safety net.

## 2. Tier 7 strategy — chunked C-port pattern

Header-inline pattern (Batches 2-E/2-F/2-G/2-H) hit its scaling limit
at ssa.cpp 227L per session experience. The 6 Tier-7 files (avg
1380L each, max 4935L) cannot header-inline cleanly without massive
header bloat + recompile-cost explosion across includers.

### 2.1 Pattern: per-method or per-method-cluster C-port

Borrow W27 emit-method burndown precedent (Tier 5 closure 2026-04-22
at a642405a5c, 100/123 = 81.3% via 4 atomic 5-commit batches W27a/b/c/d):

1. **Per-method conversion**: convert one method at a time from C++
   to C (or to an existing C primitive if available). Wire the C
   body via an `extern "C"` bridge from the original .cpp.
2. **Cluster within file**: methods that share data (same Pass
   class, same instruction visitor, etc.) move together to preserve
   shared-state coherence.
3. **Bridge per cluster, not per method**: minimize bridge surface
   (W25b/W26 minimal-bridge discipline).
4. **Falsification per cluster**: W26 §4 spec mutation + W26 §4b
   semantic-equivalence-gates (W21 golden + force_compile sole-path)
   apply per cluster, NOT per method. Skip per L2518 ZERO-bridge
   carve-out where applicable.

### 2.2 Per-file pattern variant

Each Tier-7 file has a distinct shape; pattern variant per file:

#### builder.cpp (4935L, 168 methods, HIRBuilder write-path)

Largest file. Pattern: extend W27 emit-method burndown to
non-emit HIRBuilder methods. Per-method C-port (similar to W27 Tier
5 closure scope = 100 methods done, 23 PARTIAL accepted-residual).

Estimated chunking: 10-15 method-clusters by data dependency
(framestate, instruction insertion, basic block management,
preloader queries, opcode dispatch, etc.). 5-10 sessions.

Bridges: continue using existing C-side primitives
(builder_emit_c.c is already extensive). New bridge surface
projected ~20-40 (modest, similar to W27 scope).

#### hir.cpp (1437L, 85 methods, HIR core)

HIR core data-structure methods (BasicBlock, Function, Register
operations). Many are likely Cat-A (delegations to existing
hir_basic_block_c.h, hir_c_api.h primitives).

Estimated 4-8 method-clusters. 2-3 sessions.

Bridges: most needed primitives already exist. New surface ~5-10.

#### printer.cpp (957L, 74 methods, HIRPrinter) — KEEP per L2976

KEEP per 3-class load-bearing infrastructure framing (§1.2 Diagnostic class).
Resolved per supervisor L2975 + theologian L2976 honest framing:
debug/log infrastructure organizationally separable from runtime
code paths. C++ compiler still needed for build (acceptable) but
runtime dependency on Python/jit/hir runs through C primitives only.

Decision: NO conversion required. Tier 7 in-scope reduces from 6 to
5 files, 8284→7327L. Updated order: blme → preload → inliner → hir
→ builder.

#### inliner.cpp (705L, 10 methods, HIRInliner Pass)

HIRInliner with 10 methods (large per-method avg ~70L). Real
algorithm + Pass class. Pattern: chunked C-port per method, similar
to refcount_insertion R3b precedent.

Estimated 2-4 method-clusters. 2-3 sessions.

Bridges: needs hir_inliner_* C-side primitives (none exist yet).
New bridge surface ~10-15.

#### preload.cpp (569L, 29 methods, Preloader + PreloaderManager)

Preloader class with many methods touching CPython internals
(_PyClassLoader_*, vtable_builder, strict module objects). Heavy
CPython-API surface.

**Decision point**: many Preloader methods are 1-line lookups (Type
type(), int primitiveTypecode(), etc.) that likely already have
C-side equivalents from W27 emit-method work. Inventory needed
before scoping.

Estimated 3-5 method-clusters. 2-3 sessions.

Bridges: ~10-15.

#### builtin_load_method_elimination.cpp (280L, 6 methods, BLME Pass)

Smallest Tier-7 file. Single Pass + helpers + tryEliminateLoadMethod
algorithm. while-loop + UnorderedMap + per-block iteration =
Cat-B real algorithm. Header-inline borderline (per L2970 280L is
upper bound).

Pattern: PARTIAL stub pattern per W27e — Pass::Run wraps a C-side
hir_blme_run_c (~280L pure C). Stub stays as 1-line delegate.

Estimated 1 session. Bridge surface ~5.

### 2.3 Order of attack

**Smallest-first** for confidence-building + tooling iteration:

1. builtin_load_method_elimination.cpp (280L, 1 session) — proves
   Pass-class chunked C-port pattern works for this file class.
2. preload.cpp (569L, 2-3 sessions) — proves Preloader-class
   pattern; many methods may be Cat-A delegations (cheap wins).
3. inliner.cpp (705L, 2-3 sessions) — Pass-class with real
   algorithm; tests pattern at scale.
4. hir.cpp (1437L, 2-3 sessions) — HIR core; should be many Cat-A
   delegations + few real Cat-B.
5. builder.cpp (4935L, 5-10 sessions) — largest, last; extends W27
   emit-method burndown precedent to non-emit methods.

(printer.cpp REMOVED from order per L2976 KEEP; load-bearing infrastructure.)

**Alternative: largest-first for risk-front-loading**: builder.cpp
first since it's the dominant unknown. Counter: smallest-first
builds tooling that compounds.

LEAN: smallest-first (1→6 above).

## 3. Bridge surface budget

Cumulative new bridge surface across Tier 7 (preliminary):

| File | New bridges (est) |
|------|------------------:|
| builtin_load_method_elimination | 5 |
| preload | 10-15 |
| inliner | 10-15 |
| printer | 0 (KEEP) or 10 |
| hir | 5-10 |
| builder | 20-40 |
| **TOTAL** | **50-95** |

Per W25b/W26 minimal-bridge discipline + L2837 cap principle
(hir_c_api.cpp ≤200L growth budget): cumulative new bridges + cap
budget = ~200L across Tier 7 burndown. Feasible if discipline holds.

## 4. Falsification gates per cluster

Same gates as W27 Tier 5 closure:

1. **Differential JIT_DCHECK** (parallel run, compare outputs) —
   per memory `project_wiring_methodology.md`.
2. **W21 golden trip-wire** (codegen output comparison) — per
   project_golden_canonicalization.md.
3. **Force_compile sole-path coverage** (manual smoke OR repaired
   `--wiring` per W32 verification triplet).
4. **W26 §4 spec mutation** (ZERO-bridge SKIP per L2518 carve-out
   when applicable; required when new bridges are added).
5. **W26 §4b semantic-equivalence-gates** (W21 golden + falsifier +
   force_compile sole-path).
6. **W34 __static__ workload pass** (post-burndown audit per
   supervisor L2935).

W31/W32 wiring-gate-repair-precondition applies for ANY sole-path
C-port (which Tier 7 IS). W32 verification triplet must complete
BEFORE Tier 7 burndown begins per W31 §3.2.

## 5. Open questions / decision points

1. **printer.cpp KEEP vs CONVERT** — Alex call. Debug-only code:
   strict ZERO C++ reading = convert; honest residue framing =
   keep. Recommend escalate at Tier 7 milestone.
2. **Pass<T> framework rewire** — separate strategic workstream
   (per L2949 path-forward note). Tier 7 leaves Pass framework
   alive; framework rewire is post-Tier-7 (or never if Pass
   framework is acceptable scaffolding).
3. **builder.cpp chunking strategy** — how to cluster 168 methods
   by data dependency? Pre-Tier-7 inventory session needed.
4. **W34 __static__ pass timing** — supervisor L2935 said "post-
   burndown". Burndown has reached natural turning point at 18→8;
   "post-burndown" could mean post-Batch-2-H or post-Tier-7.
   Recommend: run W34 NOW (post-Batch-2-H) as baseline; re-run
   post-Tier-7-completion as final audit.

## 6. Cross-link

- Pythia framing: #87 (2026-04-23) re-issue from L2933 + #91
  2026-04-23
- Theologian batch-class breakdown: [chat L2934]
- Turning-point framing: [chat L2970] + supervisor concur L2971
- W27 emit-method burndown precedent: Tier 5 closure a642405a5c
- Memory: feedback_dispatch_glue_categorization.md + project_r3b_wiring_bugs.md
  + project_wiring_methodology.md + project_golden_canonicalization.md
- W31 sole-path C-port falsification: docs/w31-getrpotraversal-cport-falsification.md
- W32 wiring infra repair: docs/w32-gate-phoenix-wiring-repair.md
- W33 ZERO-bridge verifier script: docs/w33-zero-bridge-verifier-script.md
- W34 __static__ retroactive test pass: docs/w34-static-retroactive-test-pass.md
