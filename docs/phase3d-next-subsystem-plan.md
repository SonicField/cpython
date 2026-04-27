# Phase 3D Next Subsystem Plan: 54 C++ files → 0

> **SUPERSEDED BY** `docs/post-phase3d-pure-c-roadmap.md` (canonical
> Phase 4+ roadmap, supervisor-promoted 2026-04-27 07:20:19Z) and
> `docs/phase4-hir-burndown-spec.md` (Phase 4 HIR-area deep dive).
> Retained for the 2026-04-17-vintage 5-phase scoping + per-file LOC
> table, which informed the bottom-up area-prioritization in the
> canonical roadmap. Plan estimates and ordering reflect pre-Phase-3D
> state; refer to the canonical roadmap for current sequencing.

*Theologian analysis, 2026-04-17*

## Current State

- 54 .cpp files remain (50,474 LOC)
- hir.h data type porting COMPLETE (7/7 phases)
- All remaining files are ALGORITHM files, not data files
- C data types (Instr, BasicBlock, Function, LIR types) already exist

## The 5 Blocking C++ Abstractions

In dependency order:

1. **BorrowedRef/Ref\<T\>** (ref.h) — smart pointer wrappers around PyObject*. 25 files.
2. **std::string + fmt::format + JIT_LOG/JIT_DLOG** — logging/formatting. 27 files.
3. **std::vector/unordered_map/containers.h** — algorithmic containers. 35 files.
4. **std::unique_ptr** — ownership semantics. 18 files.
5. **Class hierarchies (Pass, NativeGenerator, OperandBase)** — virtual dispatch. 8 files.

## Subsystem Inventory

### Group A: LIR (7 files, 8,378 LOC)

| File | LOC | C++ Density | Notes |
|------|-----|-------------|-------|
| block.cpp | 73 | 0 | Pure bridge, dies when callers convert |
| instruction.cpp | 273 | 1 | Bridge methods |
| function.cpp | 220 | 3 | copyFrom with std::variant |
| lir_c_api.cpp | 580 | 2 | Extern C wrapper, last to convert |
| parser.cpp | 600 | 57 | Testing-only, std::regex |
| regalloc.cpp | 1,684 | 63 | Hardest LIR file, linear scan allocator |
| generator.cpp | 3,719 | 21 | HIR-to-LIR lowering, CRITICAL |

### Group B: HIR (20 files, 19,983 LOC)

| File | LOC | Status | Notes |
|------|-----|--------|-------|
| builtin_load_method_elimination.cpp | 279 | Has _c.h | Category B wrapper |
| licm.cpp | 215 | Has _c.h | Category B wrapper |
| insert_update_prev_instr.cpp | 194 | Has _c.h | Category B wrapper |
| resolve_kwargs.cpp | 324 | Has _c.h | Category B wrapper |
| liveness_c.cpp | 45 | Has _c.h | Bridge |
| cfg.cpp | 186 | — | Small, std::vector |
| hir_instr_c_verify.cpp | 321 | — | No C++ STL |
| instr_effects.cpp | 560 | — | No C++ STL, giant switch |
| type.cpp | 166 | — | std::string only |
| ssa.cpp | 484 | — | std::vector + std::deque |
| analysis.cpp | 605 | — | Liveness, std::vector |
| pass.cpp | 884 | — | Pass base class, std::function |
| printer.cpp | 957 | — | Pure fmt::format, informational |
| preload.cpp | 569 | — | BorrowedRef heavy |
| inliner.cpp | 703 | — | BorrowedRef + unique_ptr |
| hir_c_api.cpp | 2,267 | — | Extern C bridge |
| hir.cpp | 1,433 | — | Instr methods, std::function |
| refcount_insertion.cpp | 1,376 | — | Private classes |
| simplify.cpp | 2,957 | — | Peephole optimizer, VERY HIGH |
| builder.cpp | 6,719 | — | LARGEST file, CRITICAL |
| parser.cpp | 1,362 | — | Testing-only |

### Group C: Codegen (3 files, 5,209 LOC)

| File | LOC | Notes |
|------|-----|-------|
| annotations.cpp | 111 | std::string, small |
| autogen.cpp | 1,404 | Translation dispatch, has _c.c |
| gen_asm.cpp | 3,694 | NativeGenerator class, CRITICAL |

### Group D: Top-level (24 files, 16,904 LOC)

| File | LOC | Notes |
|------|-----|-------|
| pyjit.cpp | 4,358 | Entry point, MUST BE LAST |
| jit_rt.cpp | 2,592 | Runtime helpers |
| inline_cache.cpp | 1,710 | Templates |
| generators_rt.cpp | 975 | Generator runtime |
| frame_shadow.cpp | 726 | std::function callbacks |
| frame.cpp | 668 | BorrowedRef |
| deopt.cpp | 559 | BorrowedRef + shared_mutex |
| context.cpp | 855 | Context singleton, second-to-last |
| compiler.cpp | 282 | Pass pipeline driver |
| perf_jitdump.cpp | 536 | Optional profiling |
| global_cache.cpp | 349 | Dict watchers |
| code_allocator.cpp | 347 | mmap wrapper |
| jit_list.cpp | 333 | File I/O |
| symbolizer.cpp | 254 | dladdr + demangling |
| code_patcher.cpp | 233 | Virtual apply() |
| bytecode.cpp | 140 | Near-C already |
| jit_time_log.cpp | 137 | Logging |
| debug_info.cpp | 128 | std::deque |
| jit_config_c_bridge.cpp | 126 | Bridge, may die |
| compiled_function.cpp | 66 | Minimal |
| type_deopt_patchers.cpp | 62 | Near-C |
| threaded_compile_c_bridge.cpp | 46 | Bridge |
| test_headers.cpp | 28 | Compilation test |

## Conversion Phases

### Phase I: Cross-cutting Infrastructure (unblocks everything)

**I-1: BorrowedRef/Ref\<T\> elimination**
- BorrowedRef\<T\> → T* (no refcount change, it's borrowed)
- Ref\<T\> → T* + explicit Py_DECREF at scope exit
- Mechanical but touches 25 files
- Do as a sweep, not per-file

**I-2: Logging C API**
- Create jit_log_c.h with C-callable macros
- Calls through to existing C++ logger via extern C bridge
- ~100 lines, unblocks 27 files

**I-3: Container C replacements**
- PhxVector (typed arrays with push/pop/grow)
- PhxHashMap (open-addressing, void* keys)
- Extend existing PhxPtrArray

Estimated: 2-3 sessions.

### Phase II: Leaf Algorithm Files

Files that use ONLY C data types + Phase I infrastructure:

- instr_effects.cpp (560) — no STL, giant switch
- type.cpp (166) — std::string only
- annotations.cpp (111) — std::string + std::vector
- hir_instr_c_verify.cpp (321) — no STL
- compiled_function.cpp (66) — minimal
- jit_config_c_bridge.cpp (126) / threaded_compile_c_bridge.cpp (46) — bridges
- test_headers.cpp (28) — compilation test

Estimated: 1-2 sessions, ~1,424 LOC.

### Phase III: LIR Subsystem (7 files, 8,378 LOC)

Order:
1. block.cpp (73) — bridge, redirect callers → delete
2. instruction.cpp (273) — bridge, same pattern
3. function.cpp (220) — rewrite copyFrom
4. regalloc.cpp (1,684) — LiveInterval to flat arrays
5. parser.cpp (600) — testing-only, may defer
6. generator.cpp (3,719) — HIR-to-LIR lowering, CRITICAL
7. lir_c_api.cpp (580) — absorb remaining API, last

Estimated: 4-6 sessions.

### Phase IV: HIR Passes (20 files, 19,983 LOC)

**IV-A (simple passes, ~2,500 LOC):**
cfg, ssa, licm, insert_update_prev_instr, builtin_load_method_elimination, resolve_kwargs, liveness_c

**IV-B (medium passes, ~4,700 LOC):**
analysis, pass (Pass→function pointer table), preload, inliner

**IV-C (hard passes, ~4,500 LOC):**
refcount_insertion, hir.cpp, printer

**IV-D (critical passes, ~9,700 LOC):**
simplify (2,957), builder (6,719 — THE hardest single file, LAST in HIR)

Estimated: 8-12 sessions.

### Phase V: Codegen + Top-level (27 files, 22,113 LOC)

**V-A (codegen):** annotations, autogen, gen_asm (NativeGenerator → C struct)
**V-B (runtime):** jit_rt, generators_rt, deopt, frame, frame_shadow
**V-C (infrastructure):** inline_cache, global_cache, code_allocator, debug_info, etc.
**V-D (orchestration, LAST 3):** compiler → context → pyjit

Estimated: 10-15 sessions.

## Summary

| Phase | Files | LOC | Sessions |
|-------|-------|-----|----------|
| I (infrastructure) | cross-cutting | ~500 new | 2-3 |
| II (leaf algorithms) | 7 | 1,424 | 1-2 |
| III (LIR) | 7 | 8,378 | 4-6 |
| IV (HIR passes) | 20 | 19,983 | 8-12 |
| V (codegen + top) | 27 | 22,113 | 10-15 |
| **Total** | **54** | **50,474** | **25-38** |

## Risk Assessment

**Risk 1: BorrowedRef elimination introduces refcount bugs.**
Falsifier: Any test_phoenix crash or leak after I-1.
Mitigation: ASAN + valgrind gate after each batch.

**Risk 2: Container replacement changes algorithmic complexity.**
Falsifier: Benchmark regression on regalloc-heavy code.
Mitigation: ABBA benchmark gate after Phase III.

**Risk 3: std::function callbacks non-trivial to convert.**
Falsifier: Callback that captures mutable state incompatible with void* context.
Assessment: LOW — checked all 7 uses, all compatible.

**Risk 4: generator.cpp + builder.cpp too large for single-file conversion.**
Falsifier: Either takes >3 sessions.
Mitigation: Split into logical sub-conversions.

**Risk 5: ARM64 gate debt compounds.**
Not a scoping risk but a process risk. 40+ commits without ARM64 gate.
Mitigation: Resolve devgpu004 2FA before starting Phase I.

## Opaque Blob Convertibility

All three remaining opaque blobs ARE convertible:

- **code_patchers**: Devirtualize CodePatcher (T2-C pattern). Phase V-C.
- **InlineFunctionStats**: PhxHashMap after I-3. Phase IV-D.
- **compilation_phase_timer**: C struct + clock_gettime. Low priority.

None should be converted until their owning subsystems convert.

## Critical Ordering Constraints

1. Phase I before everything (unblocks 25/54 files)
2. LIR before HIR (smaller, proves the pattern)
3. builder.cpp last in HIR (depends on everything)
4. pyjit.cpp ABSOLUTE LAST (depends on everything)
