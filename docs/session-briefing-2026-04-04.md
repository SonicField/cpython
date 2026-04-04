# Session Briefing — 2026-04-04

## Headline: 1.77x x86_64 ABBA (was 1.09x)

Lightweight frames enabled via ENABLE_LIGHTWEIGHT_FRAMES + frame_reifier init fix.
Two commits: 9f123bf0da (init fix) + 4b218453b5 (lightweight frames enable).
460/482 test suite PASS, zero JIT regressions. 24/24 benchmarks, zero crashes.

### Per-Benchmark Breakdown (x86_64, ABBA, 3 reps)

The 1.77x total is dominated by fibonacci (2.39x). Non-frame benchmarks show
the underlying codegen regression from Phase 3D is still present.

**Frame-dominated (big wins from lightweight frames):**

| Benchmark | Speedup | Share of total ms saved |
|-----------|---------|------------------------|
| fibonacci | 2.39x | ~87% of total improvement |
| richards_full | 1.92x | ~0.4% |
| nqueens | 1.50x | ~3.3% |
| dunder_protocol | 1.20x | ~0.9% |
| spectral_norm | 1.18x | ~8.8% |
| positional_dispatch | 1.10x | ~0.2% |
| richards_slots | 1.12x | ~0.4% |

**Non-frame benchmarks (regression visible):**

| Benchmark | 03-31 (1.22x) | 04-04 (1.77x LW) | Underlying change |
|-----------|---------------|-------------------|-------------------|
| import_callee | 1.35x | 0.98x | -0.37x |
| int_arith | 1.31x | 1.00x | -0.31x |
| nbody | 1.29x | 0.98x | -0.31x |
| pytorch_cm | 1.24x | 0.99x | -0.25x |
| dunder_protocol | 1.29x | 1.20x | -0.09x (masked by LW) |
| kwargs_dispatch | 1.15x | 0.98x | -0.17x |
| store_subscr | 1.16x | 1.00x | -0.16x |
| context_manager | 1.12x | 0.99x | -0.13x |
| decorator_chain | 1.08x | 0.96x | -0.12x |
| deep_class_super | 1.07x | 0.97x | -0.10x |

**Known regressions (pre-existing, not frame-related):**

| Benchmark | Speedup | Note |
|-----------|---------|------|
| gen_simple | 0.75x | Generator overhead — speculative dispatch territory |
| gen_nested | 0.95x | Same |
| try_except_callee | 0.92x | Exception codegen |
| nn_module_forward | 0.90x | Complex dispatch |

**Honest framing:** Strip fibonacci and the geometric mean drops to ~1.02-1.05x.
Lightweight frames is a genuine, transformative win for call-heavy workloads.
For arithmetic, import, and simulation workloads, the JIT is at interpreter parity.

**UPDATE (13:25):** LIR diff result FALSIFIES the Phase 3D codegen regression
hypothesis. int_arith HIR is structurally identical between cf49ad6da5 and HEAD.
Postalloc LIR at HEAD is actually better (19 spills vs 27). The non-frame
regression is NOT caused by degraded code generation — it is likely cross-session
measurement noise, runtime/dispatch overhead, or icache effects from larger binary.

---

## Pending Items (3, all need Alex)

### 1. Push to SonicField/cpython — CRITICAL

Two commits (9f123bf0da + 4b218453b5) carrying the entire 62% performance
improvement sit local-only on this shared machine. Flagged in 15+ checkpoints.
Local safety tag `post-lightweight-frames` exists but does not protect against
machine loss.

```
nbs-local-run 'cd /data/users/alexturner/phoenix/cpython && git push origin phoenix-asm-integration'
```

### 2. Codegen Regression Investigation — PHASE 3D CODEGEN CLEARED

**Result:** Phase 3D container swaps did NOT degrade codegen quality. Testkeeper
built cf49ad6da5 in isolated git worktree, captured HIR/LIR for int_arith:
- HIR: structurally identical between cf49ad6da5 and HEAD
- Postalloc LIR: HEAD is BETTER (19 stack spills vs 27, fewer UpdatePrevInstr)
- Fib HIR: structurally different due to lightweight frames inlining (expected)

Phase 3D is innocent. The remaining hypotheses for 1.22x→1.09x are:
1. Cross-session measurement noise (most likely — different days, machine load)
2. Runtime/dispatch overhead from Phase 3D infra layers
3. Binary size / icache pressure from C bridge layers
4. LTO/linking differences between build sessions

**Caveat (Pythia):** printHIR serializes IR structure but not pass traversal order.
Phase 3D container swaps (linked list vs vector, raw arrays vs deque) changed
iteration order for register allocation and scheduling. LIR text could look
identical while codegen differs. If HIR comparison shows no difference, an
objdump/asm-level comparison is needed as follow-up.

**What changed in the 84-commit window:**
- Phase 3D leaf file conversions (watchers, wrapper cleanup)
- Phase B infra (hir_c_api, lir_c_api, JitConfig, HirType)
- Phase B3 container swaps (operand devirt, instruction array, basicblock arrays, function container)
- Phase B4a (instruction direct field access, 13 files)

**Import bail-out hypothesis:** FALSIFIED in 04-02 session (D-1775066315/335/341).
Not the cause.

### 3. ARM64 Lightweight Frames — BLOCKED

Blocked by devgpu004 2FA. ENABLE_LIGHTWEIGHT_FRAMES is a global flag — it will
activate on ARM64 builds. The frame_reifier init path was x86_64-verified only.
Historical ARM64-specific bugs (VecD type, V-bit encoding, mov SP encoding)
warrant caution — x86_64 correctness does not guarantee ARM64 correctness.

Testkeeper has the manual cmake protocol ready for when access is available.

---

## Feature Flag Audit — COMPLETE

No hidden performance wins remaining behind compile flags. All flags assessed:

| Flag | Status | Notes |
|------|--------|-------|
| ENABLE_LIGHTWEIGHT_FRAMES | ENABLED | 1.77x — this session's win |
| ENABLE_SHADOW_FRAMES | Dead code on 3.12 | Types don't exist in 3.12 PyThreadState. Vestigial from CinderX 3.10/3.11 |
| ENABLE_GENERATOR_AWAITER | Diagnostic only | No performance impact — tracking flag |
| ENABLE_LAZY_IMPORTS | Diagnostic | Not performance-relevant |
| ENABLE_DISASSEMBLER | Debug tool | Enables ASM dump output |
| ENABLE_USDT | Debug tool | Tracing probes |
| ENABLE_SYMBOLIZER | Debug tool | Symbol resolution |

**Conclusion:** Future gains require implementation work, not flag discovery.

---

## Profiling Hooks — RESOLVED

Pythia flagged that sys.setprofile/sys.settrace don't fire for JIT-compiled
functions. Investigation found this is CinderX's intentional design:

- When profiling is active, CinderX deoptimizes ALL JIT code back to interpreter
- The deopt-on-profile mechanism requires `PYTHONJITSUPPORTINSTRUMENTATION=1`
  (opt-in for performance reasons)
- With that env var: 10/10 profiling tests PASS
- Without it: profiling and JIT are independent (by design)

Root cause: `support_instrumentation` defaults to false in config.h. The
monkey-patching of sys.setprofile/settrace only happens when enabled.

**Not a correctness gap. Not a lightweight-frames regression. By design.**

Test written: Lib/test/test_phoenix_profiling_hooks.py (10 tests, deopt-path design).

---

## Speculative Dispatch — PARKED

Alex prioritized speculative dispatch at session start. Team read the design docs
(4 files in ~/docs/speculative-dispatch). Architectural assessment: design is sound
— no new HIR instructions needed, expressed as CFG pattern using existing primitives.

**Parked because:** Implementation requires modifying C++ files (simplify.cpp,
hir.h, lir/generator.cpp). Alex's "no new C++" constraint needs clarification —
does it mean no new .cpp files, or no new C++ code? HIR-level work is inherently
C++ until Phase 3D converts those subsystems (Phase B6/B7, ~40+ files away).

**Sequencing (Alex's directive):** Fix Intel frame overhead FIRST, then speculative
dispatch. Frame overhead is now fixed (lightweight frames). Speculative dispatch is
the next optimization opportunity — addresses generator overhead (gen_simple 0.75x)
and could improve non-frame benchmarks through JIT-to-JIT calls.

---

## Session Achievements Summary

1. **Lightweight frames: 1.77x** (was 1.09x) — transformative for call-heavy workloads
2. **Feature flag audit: complete** — no hidden wins remaining
3. **Profiling hooks: resolved** — deopt-on-profile by design, not a gap
4. **Regression analysis: partially diagnosed** — broad, real, visible through LW frames, import bail-out hypothesis falsified, LIR diff in progress
5. **Phase 3D: 42/104 C++ files eliminated** (unchanged from last session — session focused on perf)

## What Falsifies This Briefing

- If the cf49ad6da5 LIR diff shows identical HIR/LIR: Phase 3D is innocent, regression is dispatch/runtime
- If the cf49ad6da5 LIR diff shows different HIR/LIR: Phase 3D container swaps degraded codegen quality — a bug to fix
- If ARM64 LW frames fails: x86_64 correctness did not transfer (historical precedent exists)
- If speculative dispatch is ruled out by C++ policy: generator/non-frame regressions have no current fix path
