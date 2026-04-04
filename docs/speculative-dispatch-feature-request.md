# Feature Request: Speculative Dispatch

## Context

All C++ has been removed from Phoenix JIT. The HIR, LIR, and codegen subsystems
are now pure C. This document describes the speculative dispatch optimization,
ready for implementation in C.

## Problem

The Phoenix JIT compiles functions based on type profiles observed during
interpretation. When CPython's adaptive interpreter sees `BINARY_OP_ADD_INT`,
the JIT emits a `GuardType` for `PyLong_Type` and a specialized `LongBinaryOp`.
If the function is later called with `float` arguments, the guard fails and the
entire compiled function deoptimizes — reifying the JIT frame, spilling registers,
and re-entering the interpreter.

This is the binary guard problem: the JIT has exactly one fast path and one
catastrophic fallback. There is no middle ground.

**Measured impact:**
- `gen_simple`: 0.75x (generator overhead forces deopt on type transitions)
- `try_except_callee`: 0.92x (exception path triggers guard failure)
- Any function called with 2+ type combinations deopts on the less-common type
- Deopt cost: ~hundreds of instructions (register spill + frame reification +
  interpreter re-entry) vs ~2 instructions for a type check

## Solution: Multi-Path Type Speculation

Replace the binary guard (fast path XOR deopt) with a chain of up to 4
type-specialized paths, followed by a generic CPython C-API fallback that
does NOT deopt.

```
bb_check_int:
  if (left->ob_type == &PyLong_Type && right->ob_type == &PyLong_Type)
    result = LongBinaryOp(kAdd, left, right)     // direct slot call
    goto merge

bb_check_float:
  if (left->ob_type == &PyFloat_Type && right->ob_type == &PyFloat_Type)
    result = FloatBinaryOp(kAdd, left, right)    // direct slot call
    goto merge

bb_slowpath:
  result = PyNumber_Add(left, right)             // generic C-API, any type
  goto merge

bb_merge:
  // continue with result (Phi node)
```

**Cost per speculation path:** 2 machine instructions (cmp + jne) on the
non-matching path. On the matching path: 1 cmp + fall-through. The first
speculation is identical to today's single guard — zero overhead when the
common type matches.

**Slow path cost:** N type checks + 1 C-API function call. Compare to today's
deopt: register spill + frame reification + interpreter dispatch + re-entry.
The slow path is 10-100x cheaper than deopt.

## Architecture

### Insertion Point: HIR Pass

The implementation is a single new HIR pass — `speculative_expansion` — that
runs in the optimization pipeline:

```
SSAify -> Simplify -> SpeculativeExpansion -> DynamicComparisonElim ->
GuardTypeRemoval -> PhiElim -> ... -> RefcountInsertion
```

**No new HIR instructions are needed.** The speculative dispatch is expressed
entirely using existing HIR primitives:

| Primitive | Role | Already exists |
|-----------|------|---------------|
| `LoadField` | Load `ob_type` from object | Yes |
| `Compare Is` | Compare type pointer against expected | Yes |
| `CondBranch` | Branch on type match/mismatch | Yes |
| `LongBinaryOp` / `FloatBinaryOp` | Type-specialized operations | Yes |
| `BinaryOp` | Generic C-API fallback | Yes |
| `Phi` | Merge results from all paths | Yes |

The pass constructs a CFG diamond pattern — the same pattern that `Simplify`
already creates with `emitCond()` and `emitCondSlowPath()`. All downstream
passes (SSAify, DCE, RefcountInsertion, PhiElim) handle diamond CFGs correctly.

### Key Design Properties

1. **No deopt on type mismatch.** Speculative checks use `Compare Is` +
   `CondBranch`, NOT `GuardType`. The failure branch goes to the next
   speculation or the slow path — never to the deopt trampoline.

2. **Slow path calls CPython C-API.** `PyNumber_Add`, `PyObject_GetItem`,
   `PyObject_GetAttr` etc. These handle ANY Python type correctly. No frame
   reification, no interpreter re-entry.

3. **Profile-guided ordering.** The check order follows `HintType` frequency
   data (already stored at the HIR level). The most common type is checked
   first — shortest path for the hot case.

4. **Up to 4 speculative paths.** Configurable. Each additional path adds
   ~20-40 bytes of machine code per dispatch site.

5. **RefcountInsertion handles all paths.** Because speculative paths are
   expressed as HIR basic blocks, the mandatory last pass (RefcountInsertion)
   inserts correct Py_INCREF/Py_DECREF on all paths automatically.

### What Downstream Passes See

After `SpeculativeExpansion`, the HIR contains a standard diamond CFG:

```
             bb_entry
            /        \
    bb_check_int   bb_check_float
         |              |
    bb_int_op      bb_float_op     bb_slowpath
         \              |            /
              bb_merge (Phi)
```

Every pass that handles `CondBranch` and `Phi` already handles this pattern.
Verified against: `ssa.c`, `simplify.c`, `refcount_insertion.c`,
`dead_code_elimination.c`, `clean_cfg.c`.

## Implementation Plan

### Files to Modify

| File | Change | Lines |
|------|--------|-------|
| `speculative_expansion.c` | **New file** — the pass itself | ~250 |
| `compiler.h` | Add `SpeculativeExpansion` to `PassConfig` enum | ~3 |
| `compiler.c` | Register pass in pipeline after Simplify | ~5 |

No changes needed to: `hir.h`, `lir/generator.c`, codegen, or any architecture-
specific code.

### Pass Algorithm

```
FOR each basic block B in function:
  FOR each instruction I in B:
    IF I is a GuardType that protects a type-specialized operation:
      profiles = find_HintType_for(I.operand)
      IF profiles has >= 2 distinct type combinations:
        expansion_types = profiles[0..min(3, len(profiles)-1)]
        ReplaceWithSpeculativeChain(I, expansion_types)
```

### Candidate Selection Criteria

Not all `GuardType` sites benefit. A site is a candidate when:

1. `HintType` data shows >= 2 distinct type profiles at this site
2. The guarded operation has a type-specialized form (no benefit if all paths
   call the same C-API function)
3. The operation is on the hot path (speculation adds code size)

### Phase 1: Hardcoded PoC (Binary Add)

Target: `BinaryOp(kAdd)` guarded by `GuardType`.
Expand to: check int -> `LongBinaryOp` -> check float -> `FloatBinaryOp` ->
generic `PyNumber_Add` fallback.

**Falsification test:** Compile a function that adds mixed int+float values.
Verify: (1) no deopt occurs, (2) results are correct, (3) the compiled code
contains the expected cmp+jne chain via `PYTHONJITDUMPASM=1`.

**Expected benchmark impact:**
- Functions with mixed numeric types: eliminate deopt entirely
- `gen_simple` / `gen_nested`: significant improvement if generator type
  transitions no longer trigger deopt
- Sort/search with mixed comparisons: O(N log N) operations stay in JIT

### Phase 2: Generalized Pass

Extend to all `GuardType`-protected operations:

| Python Pattern | Types Dispatched | Slow Path |
|---------------|------------------|-----------|
| `a + b` | int, float | `PyNumber_Add` |
| `container[key]` | list, dict, tuple | `PyObject_GetItem` |
| `obj.attr` | multiple classes | `PyObject_GetAttr` |
| `a < b` | int, float, str | `PyObject_RichCompare` |
| `acc += item` | int+int, int+float, float+float | `PyNumber_InPlaceAdd` |

### Phase 3: Profile-Guided Ordering

Wire up `HintType` profiled type frequencies to determine check order.
Most-frequent type first, least-frequent last, generic fallback at the end.

## Cost Model

### Hot Path (First Speculation Matches)

| Component | Cost |
|-----------|------|
| Load ob_type | 1 memory load |
| Compare against expected type | 1 cmp |
| Conditional branch (not taken) | ~0 (predicted) |
| Specialized operation | Same as today |

**Total overhead vs current single guard: ZERO.** The first speculation IS the
current guard check.

### Second Speculation Matches

| Component | Cost |
|-----------|------|
| First type check (miss) | 1 cmp + 1 branch (predicted after warmup) |
| Second type check (hit) | 1 cmp + fall-through |
| Specialized operation | Direct slot call |

**Total overhead vs full deopt: ~4 instructions instead of ~hundreds.**

### Slow Path (No Speculation Matches)

| Component | Cost |
|-----------|------|
| N type checks (all miss) | N * (1 cmp + 1 branch) |
| Generic C-API call | Same as interpreter |

**Total overhead vs full deopt: N type checks + 1 function call instead of
register spill + frame reification + interpreter re-entry.**

### Code Size

Each speculative path adds ~20-40 bytes of machine code per dispatch site.
With 2 paths: 40-80 bytes. With 4 paths: 80-160 bytes. This is small compared
to the function prologue/epilogue and the existing specialized operation code.

## What Would Falsify This

1. **Branch misprediction dominates.** If the type distribution is uniform
   across all 4 paths (no clear hot path), the branch predictor cannot learn
   a stable pattern. The misprediction penalty (~15 cycles on modern x86)
   erases the benefit. Test: benchmark with uniform vs skewed distributions.

2. **Code size pressure.** If speculative expansion increases L1 icache misses
   enough to slow down other hot functions. Test: measure icache miss rate
   before and after enabling speculation.

3. **HintType profiles are inaccurate.** If profiled types don't match runtime
   types, the speculative paths are never taken and only add overhead. Test:
   compare HintType profiles against actual runtime type distributions.

4. **Deopt is rare enough to be acceptable.** If the workload is monomorphic
   (single type at each site), speculation adds code size for unused paths.
   Test: count deopt events per guard site — if zero, speculation was unnecessary.

## Relationship to Other Optimizations

- **Lightweight frames** eliminated frame overhead for call-heavy workloads
  (fibonacci 2.39x). Speculative dispatch targets a different axis: type
  polymorphism within compiled code. They are complementary.

- **Exit-path inlining** reduces the cost of the REMAINING frame overhead
  (~7-12% additional gain). Speculative dispatch reduces deopt frequency.
  Again, complementary.

- **Phase 3D** converted the JIT infrastructure to C without degrading codegen
  quality (verified: HIR identical, LIR improved). Speculative dispatch builds
  on the now-pure-C substrate.

## Summary

Speculative dispatch replaces the JIT's binary guard (fast XOR catastrophe) with
a graduated response (try type A, try type B, try type C, call generic C-API).
The implementation is one new C file (~250 lines), one enum addition, and one
pipeline registration. No new HIR instructions, no codegen changes, no
architecture-specific code. The design is verified against all existing HIR
passes. The first speculation path has zero overhead compared to today's guard.
