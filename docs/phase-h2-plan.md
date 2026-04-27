# Phase H2: hir.h Instr Classes → C Structs

> **SUPERSEDED BY COMPLETION** (April 2026). The H2-A/B C-struct
> migration landed: see `Python/jit/hir/hir_instr_c.h` for the 46+
> `HirInstrLayout` / `HirDeoptLayout` / per-opcode struct definitions
> + the `HIR_INSTR_FIELDS` / `HIR_DEOPT_FIELDS` macro infrastructure,
> and `Python/jit/hir/hir_instr_c_verify.cpp` for the cross-arch
> `offsetof` / `sizeof` `static_assert` gate. Plan retained for the
> 2026-04-15 progress diary and ordering rationale that informed the
> H2-E container-conversion follow-up.

## Prerequisite (COMPLETED)
- builder.cpp 412/412 emit<> sites converted to C factories
- simplify.cpp 241/241 emit<> sites converted to C factories
- 653 total C factory callsites established (stable API surface)

## Prerequisite (OPEN)
- x86_64 pydebug JIT execution — auto-compilation works, force_compile SEGFAULTs, build env blocks python -m test

## Session 2026-04-15 Progress
- H2-A COMPLETE: 46 C structs + 46 sizeof + 26 offsetof static_asserts. 12 base class mismatches caught by asserts
- H2-B1 COMPLETE: 16 non-DeoptBase pure C factories
- H2-B2 AT CEILING: ~75 total pure C factories (from 137 create() calls). 8 remain as C++ bridges (3 HIGH RISK generator, 2 Type, 3 containers)
- Placement new pattern validated for DeoptBase container init (hir_c_init_deopt)
- hir_c_free_instr DELETED (unused trap function)
- 6 pydebug bugs fixed: IntrusiveListNode, reinterpret_cast (42 sites), PrimitiveBox, is_add_sub_imm, Decref/XDecref, resumeInInterpreter throwflag
- ARM64 pydebug 13/13 FIRST EVER with real JIT assertions

## Current State
- 93 instruction types in hir.h (88 INSTR_CLASS + 83 DEFINE_SIMPLE_INSTR, overlaps)
- 50 already have C struct equivalents in hir_instr_c.h
- 43 still need C structs (almost all DEFINE_SIMPLE_INSTR — no custom fields)

## Phases

### H2-A: Add Missing C Structs (batch, mechanical)
Add 43 DEFINE_SIMPLE_INSTR C structs to hir_instr_c.h:
- Instr-base types: `typedef struct { HIR_INSTR_FIELDS; } HirAssign;`
- DeoptBase types: `typedef struct { HIR_DEOPT_FIELDS; } HirDeopt;`
- No custom fields — trivial batch generation

### H2-B1: Rewrite Non-DeoptBase Factory Implementations (SAFE NOW)
Convert Instr-base (non-DeoptBase) factories from C++ `create()` to pure C.
These have NO C++ containers — calloc + field assignment is safe.
Types: Assign, Unreachable, CIntToCBool, Return, BitCast, RefineType, PrimitiveCompare,
IntBinaryOp, DoubleBinaryOp, PrimitiveUnaryOp, LoadConst, etc.

### H2-B2: Rewrite DeoptBase Factory Implementations (BLOCKED)
DeoptBase has C++ containers (std::vector<RegState> live_regs_, std::string descr_)
that CANNOT be calloc-initialized. BLOCKED until containers are converted to C arrays.
Prerequisite: convert live_regs_ to PhxArray and descr_ to char[]/malloc'd string.

### H2-B (original plan, revised):
Convert each C factory from calling C++ `create()` to pure C `hir_c_alloc_instr()` + `hir_c_init_instr()`:
```c
// Before (C++ bridge):
return YieldValue::create(as_reg(dst), as_reg(src), *fs);

// After (pure C):
HirYieldValue* yv = (HirYieldValue*)hir_c_alloc_instr(sizeof(HirYieldValue), 1);
hir_c_init_deopt(yv, HIR_OP_YieldValue);
hir_c_set_operand(yv, 0, src);
yv->output = dst;
return yv;
```
- Factory SIGNATURES don't change — builder.cpp/simplify.cpp unaffected
- Only factory IMPLEMENTATIONS change

### H2-C: Convert Instr Base Class Methods to C Functions
- NumOperands → C function (opcode-based dispatch, T2-C3 table exists)
- GetOperand/SetOperand → C functions (operand array access)
- GetOperandType → C function (T2-C3 dispatch table exists)
- visitUses → C function
- Destroy → C function (T2-C6 opcode dispatch exists)
- Partially done by T2-C devirtualization

### H2-D: Convert DeoptBase Methods to C Functions
- setFrameState, frameState, live_regs access
- Depends on H2-C completion

### H2-E: Eliminate InstrT<> Template Hierarchy (rate-limiting step)
- InstrT<T, opc, Base, Tys...> generates: constructors, operand storage, HasOutput
- Replace with C macros (HIR_INSTR_FIELDS already covers layout)
- Each instruction type needs explicit constructor replacement
- This is the most complex phase — 4 template specializations to unwind

### H2-F: Split hir.h into C-Compatible Header
- Forward declarations + Edge struct → hir_forward.h
- Instr/DeoptBase C structs → hir_instr.h (already hir_instr_c.h)
- BasicBlock → hir_block.h
- Environment → hir_env.h
- Enums → already C-compatible

## Parallelization
- H2-A + H2-B: interleaved per-type batch (add struct → rewrite factory → next)
- H2-C: can START during H2-B (base methods independent of specific types)
- H2-D: after H2-C
- H2-E: after H2-C + H2-D (InstrT depends on base class)
- H2-F: after H2-E

## Estimate
- H2-A+B: 2-3 sessions
- H2-C+D: 1-2 sessions
- H2-E+F: 3-5 sessions
- Total: 6-10 sessions

## Key Invariants
- Factory signatures are the stable API — don't change during H2
- hir_c_init_instr/hir_c_init_deopt centralizes post-alloc init
- No reinterpret_cast between HirType and Type (standing rule)
- All builds through scripts/build_phoenix.sh
- ARM64 pydebug 13/13 gate on every commit
