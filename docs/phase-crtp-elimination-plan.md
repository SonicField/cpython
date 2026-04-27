# Phase CRTP: InstrT Template Elimination

> **SUPERSEDED BY** `docs/post-phase3d-pure-c-roadmap.md` (canonical
> Phase 4+ roadmap). Template-elimination work belongs to the HIR
> convergence sweep (Phase 4 + the CRTP/template residue cleanup
> deferred until consumers go pure-C). Retained for historical
> reference to the Pre-Phase-CRTP H2-B/H2-E/H2-C state and the original
> 4-phase clone()/InstrT-shell/struct-defs/class-delete sequencing
> intent.

## Current State (Post H2-B/H2-E/H2-C)
InstrT<T, opc, Base, Tys...> provides 4 things, 3 already replaced:

1. **CONSTRUCTOR** — REPLACED by hir_c_init_instr/hir_c_init_deopt (C factories)
2. **CREATE** — REPLACED by hir_c_alloc_instr + C factories (~195/200 pure C)
3. **GetOperandType** — REPLACED by T2-C3 opcode dispatch table
4. **CLONE** — NOT YET REPLACED (~15 call sites in optimization passes)

Operands<N> (create with arity) and HasOutput (output register) — both replaced by C factories.

## Phases

### Phase 1: Convert clone() to C (1 session)
- Opcode-dispatched copy function: hir_c_clone_instr(void* instr) → void*
- Pattern: allocate(sizeof(T), num_operands) + memcpy fields + copy operands
- DeoptBase types: also copy live_regs, descr (strdup), frame_state (new+copy)
- Edge-bearing types: use C++ bridge for Edge copy (same as Destroy pattern)
- ~15 call sites to update
- Gate: 980/980 + ARM64 pydebug 13/13

### Phase 2: Make InstrT empty shell (1 session)
- Remove: constructors, create(), clone() — all replaced by C equivalents
- Keep: class definition (type tag for static_cast), GetOperandTypeImpl
- DEFINE_SIMPLE_INSTR becomes: class T final : public Base { using Base::Base; };
- Gate: compile + test — no behavioral change

### Phase 3: Replace macros with C struct definitions (1 session)
- DEFINE_SIMPLE_INSTR → typedef struct { HIR_BASE_FIELDS; } HirT; (already exist in hir_instr_c.h)
- INSTR_CLASS → direct struct definition in hir_instr_c.h
- C++ class definitions become thin wrappers around C structs
- Gate: compile + test

### Phase 4: Delete class definitions (1-2 sessions)
- Replace remaining static_cast<T*> with opcode-based void* dispatch
- T2-C already converted most dispatch — audit remaining static_casts
- Delete class definitions from hir.h
- hir.h becomes C-compatible header
- Gate: compile + test + full ARM64

## Dependencies
- Phase 1 gates Phase 2 (clone must be C before constructors are removed)
- Phase 2 gates Phase 3 (classes must be empty before macros change)
- Phase 3 gates Phase 4 (structs must exist before classes are deleted)
- Strictly sequential — no parallelization possible

## Estimate
- Total: 3-4 sessions
- Phase 1 is the critical path — proven pattern (same as Destroy() C conversion)

## Risks
- clone() for Edge-bearing types needs C++ bridge (same as Destroy)
- static_cast audit for Phase 4 may reveal more sites than expected
- HintType stays as C++ exception until its ProfiledTypes vector is converted
