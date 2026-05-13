# Pre-Port Audit Checklist (Layout-Pinned C++ Struct Ports)

**Authored:** theologian, 2026-05-13 (per supervisor 00:51:18Z (B), pythia #368 audit-completeness gap)

**Scope:** Mandatory checklist for any C++→C port commit that touches a C++ struct or class with C-side layout-pinned representation (HirInstrLayout, HirBeginInlinedFunction, HirFrameStateLayout, LirOperand, LirBasicBlock, LirFunction, LirInstruction, etc.). Supplements `synthetic_falsifier_at_gate.md` (1546c4330e) with a structural pre-commit audit specifically targeting the cascade-audit-3-category failure class (`feedback_cascade_audit_3category`).

## Motivation

c14 (a6500318e0, 2026-05-13 00:06:50Z) shipped after a "single-pass audit" that cited only category (a) — `sizeof(HirBeginInlinedFunction) == sizeof(BeginInlinedFunction)`. The audit missed:

- (b) field-offset cross-validation for HirBeginInlinedFunction `func` and `caller_state_ptr` (only `fullname` was offsetof-validated)
- (c) hardcoded literal-offset assumption: the port called `frame_asm_c_frame_header_size` (the asm-helper variant that adds `+sizeof(void*)` when `ENABLE_SHADOW_FRAMES` undef) instead of the LIR-generation-equivalent `jit_frame_header_size` direct call. The asm-helper variant produces +8-byte offsets at runtime, crashing JIT in `test_phoenix_benchmark_correctness::test_fibonacci`.

The audit metric measured velocity (~20s dispatch→auth) and missed completeness. This checklist makes completeness auditable structurally, not by author judgment.

## The 3 Categories (per `feedback_cascade_audit_3category`)

Every port commit touching layout-pinned C++ structs MUST cite all three categories explicitly in the audit (chat or commit body), with specific C-side equivalents named:

### (a) Sizeof equivalence

For each C++ struct/class touched by the port (including via `reinterpret_cast`, field access, or method dispatch):

- Locate the `static_assert(sizeof(HirX) == sizeof(CppY))` line in the appropriate verifier file (e.g., `hir_instr_c_verify.cpp`, `lir_instr_c_verify.cpp`).
- Cite the file:line.
- If MISSING, ADD the static_assert as part of the port commit OR a precursor commit (do NOT proceed without one — the c14 SIGSEGV class).

### (b) Offsetof equivalence per field

For each C++ field accessed by the port via `instr->field` or equivalent C-side cast:

- Locate the `static_assert(offsetof(HirX, c_field) == offsetof(CppY, cpp_field_))` line in the verifier file.
- Cite the file:line per field.
- If MISSING for any accessed field, ADD the offsetof asserts BEFORE the port commit (precursor) OR in the same commit as part of the verifier-extension scope.
- NOTE: sizeof equivalence (a) is NECESSARY but NOT SUFFICIENT; an empty base class subobject (e.g., `InlineBase`) can shift derived-class field offsets without changing total sizeof if the shift is symmetric on both sides — an offsetof per-field check catches asymmetric layouts.

### (c) Hardcoded literal-offset equivalents

For each C-side function or constant the port calls/uses as a substitute for a C++ method or constant:

- Verify the SUBSTITUTE has the SAME semantic behavior as the C++ original. Common asymmetries:
  - asm-helper variants vs LIR-generation variants (c14's `frame_asm_c_frame_header_size` vs `jit_frame_header_size` direct call)
  - Constants with version-conditional adjustments (`ENABLE_SHADOW_FRAMES`, `PY_VERSION_HEX`)
  - C-side fields that look identical-named but encode different shifts (e.g., `caller_state_ptr` vs `caller_state_` — same field, but the question is what the *value* means; if C++ wraps it via accessor with extra logic, the raw field read may differ)
- If the substitute's semantics differ in any way, EITHER:
  - Construct the C++-equivalent expression manually in the C port (matching the args the C++ wrapper passes — c14's amend used `jit_frame_header_size(code, lightweight, sizeof(void*), sizeof(PyObject*))` to match `frameHeaderSize`'s exact behavior)
  - Add a static_assert proving the substitute produces the same value for the relevant inputs (e.g., for c14: `static_assert(JIT_FRAME_HEADER_SIZE_AS_USED_IN_C_HELPER == sizeof(jit::FrameHeader))`)
  - Document the asymmetry in the port comment AND the commit body, with a falsifier test if runtime behavior differs.

## Pre-Audit Procedure

Before posting the c-series scope-confirmation request to chat:

1. **Enumerate (a):** list every C++ struct/class the port touches; for each, file:line of the sizeof static_assert. If any are missing, decide whether to (i) add the assert as part of port commit OR (ii) add as precursor commit. Document the decision in the audit post.

2. **Enumerate (b):** list every C++ field the port accesses (directly or via accessor); for each, file:line of the offsetof static_assert. If missing, same decision tree as (a). Specifically: when the C++ class uses MULTIPLE INHERITANCE (even from an empty base), per-field offsetof asserts are NECESSARY because empty-base-optimization is implementation-defined.

3. **Enumerate (c):** list every C-side function and constant the port uses as a substitute for C++. For each substitute, locate the C-side definition (file:line) AND the C++ original (file:line); compare them line-by-line for asymmetries (extra args, version conditionals, extra arithmetic). If any asymmetry, document the resolution.

4. **Compose audit post:** include all three category enumerations explicitly. Failure to cite all three categories is a `feedback_cascade_audit_3category` violation (medic-flaggable).

## Pre-Audit Procedure Example (c14 retrospective)

What c14 audit SHOULD have said:

> (a) Sizeof equivalence:
> - HirBeginInlinedFunction: sizeof matched at hir_instr_c_verify.cpp:281 ✓
> - HirFrameStateLayout: sizeof matched at hir_instr_c_verify.cpp:106 ✓
>
> (b) Offsetof equivalence per field:
> - HirBeginInlinedFunction.func — UNVERIFIED ⚠ (only .fullname offset asserted at line 268). Multiple inheritance from InlineBase (empty class) may cause subtle layout shift. ACTION: ADD offsetof(HirBeginInlinedFunction, func) == offsetof(BeginInlinedFunction, func_) static_assert as part of c14 commit.
> - HirBeginInlinedFunction.caller_state_ptr — UNVERIFIED ⚠. Same. ACTION: ADD.
> - HirFrameStateLayout.parent — VERIFIED at hir_instr_c_verify.cpp:120 ✓
> - HirFrameStateLayout.code — VERIFIED at hir_instr_c_verify.cpp:113 ✓
>
> (c) Hardcoded literal-offset equivalents:
> - C++ `frameHeaderSize(PyCodeObject*)` (frame_header.h:67) calls `jit_frame_header_size(code, lightweight, sizeof(FrameHeader), sizeof(PyObject*))` with sizeof(FrameHeader)=sizeof(void*) on 3.12+.
> - C-side candidate `frame_asm_c_frame_header_size(PyCodeObject*)` (frame_asm_c.c:351) calls jit_frame_header_size with hardcoded JIT_FRAME_HEADER_SIZE=sizeof(void*) BUT THEN ADDS `+sizeof(void*)` when ENABLE_SHADOW_FRAMES is undef. ⚠ ASYMMETRIC. ACTION: do NOT use frame_asm_c_frame_header_size; call jit_frame_header_size directly with C++-equivalent args.

If c14 audit had cited (b) and (c) at this level of detail, the SIGSEGV would have been caught at audit time, not at gate.

## Enforcement

- **Author-side:** every port commit's audit post (chat or commit body) MUST include all three category enumerations.
- **Gatekeeper:** APPROVE checklist for layout-pinned-port-class commits MUST verify all three category citations are present and concrete (file:line, not "verified").
- **Medic:** [MEDIC-WARNING] on any port commit landing without 3-category audit citation.
- **Supervisor:** disposes flags from medic.

## Out-of-scope

- Routine code commits that don't touch layout-pinned C++ structs (no audit needed).
- Documentation-only commits.
- Test-only commits.
- Refactors within a single language (e.g., C++ method extraction without C interop).

## Non-coverage caveat

This checklist catches structural layout asymmetry. It does NOT catch:

- Algorithmic differences between C and C++ implementations (the C-port body could compute the wrong thing even if all layout is correct).
- Side effects (e.g., reference counting, exception state) that C++ handles implicitly via destructors but C must handle explicitly.
- Threading/synchronization assumptions.

These remain the author's responsibility to verify via runtime tests (gate Tier 1, force_compile, ABBA). The 3-category audit is a NECESSARY but NOT SUFFICIENT condition.

## Self-falsifying disclaimer

If a future port commit cites the 3 categories at this level of detail and STILL produces a runtime bug at gate, this checklist is incomplete and needs extension. The c14 retrospective example above is the calibration anchor: any pre-port audit that would NOT have caught c14's frame_asm_c asymmetry given this checklist is a checklist failure, not an author failure.
