# Post-Phase-3D Pure-C Roadmap (Phoenix JIT)

**Author:** theologian
**Date:** 2026-04-27
**Authoring trigger:** D-1777270945 (Alex DROPPED bug-(ii) heavy-tier; team back on roadmap to ZERO C++)
**Status:** spec — not policy until supervisor reviews

---

## 1. Ground truth (LOC, measured 2026-04-27)

| Area | C bodies (LOC) | C++ remaining (LOC) | C-share | Notes |
|------|----------------|---------------------|---------|-------|
| HIR (Python/jit/hir/) | 15,441 | 11,319 | 58% | Phase 3D landed algorithm bodies; bridge + dispatch + class state remain |
| LIR (Python/jit/lir/) | 5,117 | 7,148 | 42% | Core helpers C; generator/regalloc/parser still C++ |
| Codegen (Python/jit/codegen/) | 4,242 | 5,220 | 45% | autogen partly translated; gen_asm template-heavy |
| Root (runtime + module + misc) | 3,692 | 15,616 | 19% | Largest greenfield; pyjit.cpp 4357 + jit_rt 2592 + inline_cache 1754 |
| **Total** | **28,492** | **39,303** | **42%** | Excludes asmjit (replaced by phoenix-asm) |

Per-file C++ remaining (top 12, LOC):
1. `hir/builder.cpp` — 4,527 (Phase 3D residual: dispatch loop + class state + type-marshaling shells)
2. `pyjit.cpp` — 4,357 (module init + config + watcher logic)
3. `lir/generator.cpp` — 3,718 (HIR → LIR translator)
4. `codegen/gen_asm.cpp` — 3,693 (LIR → x86/ARM emitter)
5. `hir/hir_c_api.cpp` — 2,732 (C↔C++ bridge; deletion-bonus when consumers go pure-C)
6. `jit_rt.cpp` — 2,592 (runtime helpers)
7. `inline_cache.cpp` — 1,754 (IC allocation/lookup)
8. `lir/regalloc.cpp` — 1,684 (linear-scan regalloc)
9. `codegen/autogen.cpp` — 1,404 (instruction-table dispatch)
10. `hir/hir.cpp` — 1,268 (HIR core)
11. `generators_rt.cpp` — 975 (generator runtime)
12. `hir/printer.cpp` — 957 (text serializer)

## 2. Dependency graph

Direction: A → B means "A includes / calls into B."

```
              ┌──────────────────────────┐
              │ pyjit.cpp (module init)  │
              └────┬─────────────────────┘
                   │ (everything)
        ┌──────────┴───────────┐
        ▼                      ▼
┌──────────────┐    ┌─────────────────────┐
│ codegen/*    │───▶│ runtime (jit_rt,    │
│ gen_asm,     │    │ deopt, inline_cache,│
│ autogen      │    │ frame_shadow, etc.) │
└──┬───────────┘    └─────────────────────┘
   │              ▲
   ▼              │
┌──────────────┐  │
│ lir/*        │──┘
│ generator,   │  (runtime headers used
│ regalloc,    │   by lir/codegen for
│ parser       │   IC + deopt structs)
└──┬───────────┘
   │
   ▼
┌──────────────────────────┐
│ hir/* (builder, hir,     │
│ printer, inliner,        │
│ preload, blme, verify)   │
└──────────────────────────┘
```

Bridge files (deletion-bonus when consumers complete):
- `hir/hir_c_api.cpp` (2,732 LOC) — dissolves when `builder + hir + printer + inliner + preload + blme` are pure-C.
- `lir/lir_c_api.cpp` (580 LOC) — dissolves when `generator + regalloc + parser + instruction + function + block` are pure-C.
- `jit_config_c_bridge.cpp` (126 LOC), `threaded_compile_c_bridge.cpp` (46 LOC) — leaf bridges, dissolve when pyjit.cpp goes.

## 3. Sequencing recommendation

### Phase 4 — HIR completion (HIR-misc + builder.cpp residual)

**Scope (LOC C++ to delete):**
- `hir/builder.cpp` 4,527 — dispatch loop + class state extraction (already in flight per pythia #89 spec)
- `hir/hir.cpp` 1,268
- `hir/printer.cpp` 957
- `hir/inliner.cpp` 704
- `hir/preload.cpp` 570
- `hir/hir_instr_c_verify.cpp` 377 (verifier; thin)
- `hir/builtin_load_method_elimination.cpp` 184
- **Bridge bonus:** `hir/hir_c_api.cpp` 2,732 dissolves on completion
- **Total deletion:** 11,319 algorithm + 2,732 bridge = **14,051 LOC**

**Rationale:** highest dollar-per-commit. HIR is the most-converted area; 58% already in C, and the bridge dissolution on completion is the largest single delete event in the project. Doing this first establishes the pattern: convert, then dissolve.

**Risk:** class-state extraction (HirBuilder member fields) is the unsolved sub-problem from Phase 3D close. Pythia #89 + theologian Tier-7 spec are pre-existing artifacts.

### Phase 5 — LIR completion

**Scope (LOC C++ to delete):**
- `lir/generator.cpp` 3,718 (HIR → LIR translator)
- `lir/regalloc.cpp` 1,684 (linear-scan)
- `lir/parser.cpp` 600 (text parser; testing-only?)
- `lir/instruction.cpp` 273
- `lir/function.cpp` 220
- `lir/block.cpp` 73
- **Bridge bonus:** `lir/lir_c_api.cpp` 580 dissolves
- **Total deletion:** 6,568 algorithm + 580 bridge = **7,148 LOC**

**Rationale:** LIR is consumer of HIR; freezing LIR-as-C exposes a clean C-only seam to codegen. generator.cpp is translator-shaped (similar to builder.cpp emit methods) so Phase 3D conversion patterns transfer.

**Risk:** regalloc is performance-critical; perf-sensitive. Needs ABBA discipline matching Phase 3D. Parser may be eligible for outright deletion if only used in tests.

### Phase 6 — Codegen completion

**Scope (LOC C++ to delete):**
- `codegen/gen_asm.cpp` 3,693 (LIR → machine emit)
- `codegen/autogen.cpp` 1,404 (instruction-table dispatch; partially in `autogen_translate_c.c` 2,173)
- `codegen/annotations.cpp` 123
- **Total:** **5,220 LOC**

**Rationale:** depends on LIR (now C) and HIR (now C). autogen is partly-converted already (autogen_translate_c.c at 2173 LOC). Sequencing AFTER LIR avoids consuming a moving target.

**Risk:** templates concentrated here (17 vs 0-1 elsewhere). gen_asm uses `template <int N>` and `template <typename T>` patterns that map awkwardly to C — likely require manual specialization or codegen tool. Expect lower commits/LOC than Phase 4/5.

### Phase 7 — Runtime burndown

**Scope (LOC C++ to delete):**
- `jit_rt.cpp` 2,592 (runtime helpers, JIT-emitted-code callees)
- `inline_cache.cpp` 1,754 (IC alloc/lookup; perf-critical)
- `generators_rt.cpp` 975
- `frame_shadow.cpp` 725
- `frame.cpp` 668
- `deopt.cpp` 559
- `global_cache.cpp` 373
- `jit_list.cpp` 333
- `type_deopt_patchers.cpp` 62
- `compiled_function.cpp` 66
- **Total:** **8,107 LOC**

**Rationale:** mostly self-contained; can run in PARALLEL with Phase 5/6 once Phase 4 bridge is gone. Lower coupling means lower velocity-coupling between agents.

**Risk:** inline_cache + deopt are perf-critical and called from JIT-emitted code (ABI-sensitive). Need same-shape codegen for hotpaths or perf regresses. Golden-output coverage is limited here vs HIR.

### Phase 8 — Module + context (pyjit + context)

**Scope (LOC C++ to delete):**
- `pyjit.cpp` 4,357 (module init + Python C-API + config + watchers + atomic + chrono + std::filesystem)
- `context.cpp` 858
- **Total:** **5,215 LOC**

**Rationale:** depends on everything; convert last when all callees are C-friendly. Module bootstrap is Python-C-API heavy.

**Risk (highest):** `pyjit.cpp` uses `std::atomic`, `std::chrono`, `std::filesystem`, `fmt::std`, `dlfcn` — these are not 1:1 portable to C. **Recommend NOT 100% conversion target** for this file; leave as a thin C++ shell calling pure-C internals if cost exceeds value. **This is a decision point for Alex.**

### Phase X — Leaf-file opportunism (parallel)

**Scope (LOC C++ to delete):**
- `perf_jitdump.cpp` 536, `code_allocator.cpp` 347, `compiler.cpp` 319, `symbolizer.cpp` 254, `code_patcher.cpp` 233, `bytecode.cpp` 140, `jit_time_log.cpp` 137, `debug_info.cpp` 128, `jit_config_c_bridge.cpp` 126, `threaded_compile_c_bridge.cpp` 46, `test_headers.cpp` 28
- **Total:** **2,294 LOC**

**Rationale:** small, mostly leaf, low risk. Workers pick up opportunistically between heavy-area work.

## 4. Why bottom-up (HIR → LIR → codegen → runtime → pyjit) and not other orders

| Order | Argument for | Argument against | Verdict |
|-------|--------------|------------------|---------|
| **Bottom-up (recommended)** | Each level freezes a C-only seam for the level above. Bridge dissolution is gated on level completion. | Phase 4 is the longest single phase; momentum-tax if it stalls. | **Take.** Bridge-dissolution dollar-per-commit dominates. |
| Top-down (pyjit → ... → HIR) | Removes Python-C-API friction first | pyjit depends on every C++ callee still being C++; nothing dissolves until end | Reject. Maximises rework. |
| Codegen-first (gen_asm to phoenix-asm-direct) | Most "JIT-ness" is here | Templates resist conversion; depends on LIR + HIR still being C++ during the work | Reject. High-friction-first violates leverage rule. |
| Independent leaves first | Build momentum + skill | LOC count is small; doesn't move the needle on bridges | Run as Phase X background, not primary order. |
| Parallel-everywhere | Maximises agent utilisation | Cross-area churn during partial conversions creates rebase + bridge-API thrash | Reject for HIR/LIR/codegen; **OK for runtime + Phase X**. |

**Falsifier of recommended order:** if Phase 4 class-state extraction fails (HirBuilder member fields cannot be extracted to a C struct without breaking Phase 3D dispatch loop), then bottom-up stalls and we should pivot to runtime + Phase X parallel work while a different solution is found. Spec for HirBuilder state extraction (pythia #89 + theologian Tier-7) must land before Phase 4 commits in earnest.

## 5. Commit / session estimates (Phase 3D scaling reference)

Phase 3D delivered ~250-300 commits over ~3-4 weeks (Phase A 52, R3b 33, builder Tier 1-7 100+, simplify/ssaify burndown ~50). Density varied 30-100 LOC/commit (algorithm) to 5-20 LOC/commit (state/dispatch).

| Phase | LOC C++ to delete | Density estimate | Commit estimate | Calendar (1 worker) |
|-------|-------------------|------------------|------------------|---------------------|
| 4 — HIR completion | 14,051 (incl bridge) | 60 LOC/commit avg; lower for class-state | 200-280 | 2.5-3.5 weeks |
| 5 — LIR completion | 7,148 (incl bridge) | 70 LOC/commit (translator-shaped) | 90-130 | 1.5-2 weeks |
| 6 — Codegen | 5,220 | 40 LOC/commit (templates slow) | 110-150 | 2 weeks |
| 7 — Runtime | 8,107 | 80 LOC/commit (self-contained) | 90-130 | 1.5-2 weeks |
| 8 — pyjit + context | 5,215 | 30 LOC/commit if full; N/A if shell | 150-200 (full); 50-80 (shell) | 2-3 weeks (full); 1 week (shell) |
| X — Leaves (parallel) | 2,294 | 100 LOC/commit (small files) | 25-40 | opportunistic |

**Total: ~665-930 commits, ~10-13 weeks** with single-worker serial. With 2-3 workers in parallel for Phase 7 + X alongside Phase 4-6, calendar can compress to **6-8 weeks**.

## 6. Structural risks (per area)

| Risk | Area | Severity | Falsifier / mitigation |
|------|------|----------|------------------------|
| HirBuilder class-state extraction unsolved | Phase 4 | HIGH | Pythia #89 + theologian Tier-7 spec must land + falsify before commits. Mitigation: start with hir.cpp / printer.cpp (no class state) while spec is drafted. |
| Templates in gen_asm/autogen | Phase 6 | MEDIUM | Manual specialization or generator script. Falsifier: prototype 1 template conversion before authorising the area. |
| Perf regression in inline_cache / deopt | Phase 7 | MEDIUM | Per-commit ABBA discipline (existing Phase 3D protocol) + golden-output coverage extension to runtime. |
| std:: features in pyjit (atomic/chrono/filesystem) | Phase 8 | HIGH (cost) | Decision point: full conversion vs thin C++ shell. Recommend defer call to Alex after Phase 4-7 land. |
| Bridge file dissolution depends on ALL consumers | Phases 4 + 5 | MEDIUM | Strict gate: no bridge delete until 0 C++ callers grep-confirmed (precedent: instr_effects.cpp deletion in Phase 3D). |
| Cross-phase rebase churn | Phases 5-7 if parallel | LOW-MEDIUM | Serial Phase 4-6; Phase 7 + X opportunistic-parallel only. |
| Exception-throwing C++ (gen_asm, parser, builder, jit_list, frame, pyjit) | Phases 4/5/6/7/8 | LOW | C error returns + JIT_CHECK; pattern established in Phase 3D. |
| pyjit may not be deletable to 0 C++ at all | Phase 8 | DECISION | Alex direction needed. Terminal goal "pure C" may admit a thin Python-module C++ shell as pragmatic. |

## 7. Decision points for Alex (in order)

1. **Phase ordering accept/reject?** Spec proposes HIR → LIR → codegen → (runtime ‖ leaves) → pyjit.
2. **Phase 4 prerequisite:** HirBuilder class-state extraction spec (pythia #89 + theologian Tier-7) — should it land before any Phase 4 commits, or in parallel with low-state hir.cpp/printer.cpp work?
3. **pyjit.cpp end-state:** 100% pure C (target ~5,215 LOC convert, 2-3 weeks, std:: ports tedious) OR thin C++ module shell calling pure-C internals (~50-80 commits, 1 week, 4,357 LOC C++ remains as Python-C-API glue)?
4. **Parallelism:** authorise Phase 7 (runtime) + Phase X (leaves) as background workstreams parallel with Phase 4-6, or strict serial?
5. **bug (ii) test+doc** (D-1777270945) is sequenced FIRST; this roadmap follows immediately after that lands.

## 8. Open questions for the team

- Generalist: confirm `lir/parser.cpp` (600 LOC) usage scope — production path or test-only? If test-only, candidate for delete vs convert.
- Testkeeper: golden-output coverage for runtime files (Phase 7) — does current corpus exercise inline_cache + deopt enough for ABBA-style validation? If not, extension needed before Phase 7.
- Scribe: search for prior "pyjit.cpp full-conversion" decisions or theologian/Alex notes — has the shell-vs-full question been decided historically?

---

**Falsifier of this entire spec:** if Phase 3D-style algorithm-translation plus bridge dissolution does not yield ≥80% LOC reduction by end of Phase 4 (target: 14,051 LOC delete, ~3.5 weeks), the bottom-up sequencing assumption is wrong and the team should reconsider — likely indicating that class-state extraction is structurally infeasible or that hir_c_api.cpp doesn't dissolve cleanly. Re-spec at that gate.
