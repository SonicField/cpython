# W26 — `emitAnyCall` Full Conversion + 149b7e2d40 PartialConversion Reabsorb

**Status:** ACTIVE (post-INVOKE_* Phase 2 substantive close, pre-emitAnyCall
full conversion). Authorized 2026-04-22 per supervisor [chat L2454] (c)-defer
decision + REABSORB-WHEN trigger fired.

**Scope:** Convert `HIRBuilder::emitAnyCall` from C++ to C (~140 lines, 5 new
bridges per generalist [chat L2447] empirical scope estimate). Reabsorb the
`hir_builder_emit_awaited_call_tail_c` PartialConversion bridge introduced
at `149b7e2d40` back into the full C body. Mirror of Tier 5 emit-method
conversion pattern + INVOKE_* Phase 2 precedent (push 81).

**Owner:** theologian (this spec) + generalist (implementation).

**Estimated cost:** ~90-120min focused work (per generalist L2447 empirical
scope). 5 new bridges + emitAnyCall C body + reabsorb. Two-phase work
(bridge inventory first + conversion second).

---

## 1. The PartialConversion artifact + REABSORB-WHEN trigger

`149b7e2d40` (Tier 5 close) extracted the await-tail dispatch from
`emitAnyCall` into a standalone bridge `hir_builder_emit_awaited_call_tail_c`
because Tier 5's INVOKE_* sub-methods (emitInvokeFunction/Native/Method)
remained C++ at that time. The bridge call lives in `emitAnyCall`'s C++
body at `builder.cpp:3004-3012`.

The REABSORB-WHEN comment on the bridge (`builder_emit_c.c:3367`) reads:

> PARTIAL CONVERSION ARTIFACT — emitAnyCall await-tail extracted while
> emitAnyCall opcode-switch + 3 INVOKE_* sub-methods (emitInvokeFunction,
> emitInvokeNative, emitInvokeMethod) remain C++. REABSORB WHEN: Tier 6
> INVOKE_* family fully converts to C; then emitAnyCall fully converts and
> this bridge can inline back into the full C body.

**REABSORB-WHEN trigger FIRED at push 81 (2026-04-22):** all 3 INVOKE_*
methods are now C-converted (emitInvokeNative, emitInvokeMethod,
emitInvokeFunction). emitAnyCall full conversion can now happen, allowing
the await-tail bridge to inline back into the C body.

W26 closes the REABSORB-WHEN by converting emitAnyCall to C and inlining
the bridge.

---

## 2. emitAnyCall structure (per builder.cpp L2875-3014)

5 opcode paths in the switch statement:

| Opcode                                       | Helper called                       | C-status |
|----------------------------------------------|-------------------------------------|----------|
| `CALL_FUNCTION`/`CALL_FUNCTION_KW`           | `tc.emitVariadic<VectorCall>`       | C++ template — needs C bridge |
| `CALL_FUNCTION_EX`                           | `emitCallEx`                        | already C-bridged |
| `CALL`/`CALL_KW`/`CALL_METHOD`               | `tc.emitCallMethod` + `findExceptionHandler` + `getSimpleExceptInfo` + `emitCallExceptionHandler` | mixed: emitCallExceptionHandler already C; others not |
| `INVOKE_FUNCTION`                            | `emitInvokeFunction`                | already C (push 81) |
| `INVOKE_NATIVE`                              | `emitInvokeNative`                  | already C (push 81) |
| `INVOKE_METHOD`                              | `emitInvokeMethod`                  | already C (push 81) |

Plus pre-switch:
- `is_awaited` computation: `code_->co_flags & CO_COROUTINE`,
  `bc_it.remainingIndices()`, `bc_instr.nextInstr().opcode()`, plus
  `PY_VERSION_HEX >= 0x030C0000` conditional
- `kwnames_` member field access

Plus post-switch awaited-tail block (lines 2981-3013):
- `bc_it++` advancement (3 times) + opcode assertions
- `checkAsyncWithError(bc_instrs, get_awaitable_bc)` helper call
- `hir_builder_emit_awaited_call_tail_c` bridge call (THE artifact to reabsorb)

---

## 3. Bridge inventory

### NEW bridges (~5 per generalist L2447):

1. **emitVariadic<VectorCall>** — C++ template; needs C-side equivalent
   bridge taking opcode-class + temps_ + num_operands + flags. Could be
   a single `hir_builder_emit_variadic_vector_call_c` or split per
   instantiation.
2. **emitCallMethod** — C++ method on TranslationContext returning
   instruction pointer; bridge `hir_builder_emit_call_method_c` takes
   tc + num_operands + out_reg + flags, returns instr pointer (HirInstr).
3. **findExceptionHandler(BCOffset)** — searches HIRBuilder's
   exception-handler map; bridge `hir_builder_find_exception_handler_c`
   takes builder + cur_offset, returns handler pointer (or null).
4. **getSimpleExceptInfo** — extracts SimpleExceptInfo struct from
   handler; bridge `hir_builder_get_simple_except_info_c` takes builder +
   handler + out_info_struct.
5. **kwnames_ get/set** — class member field access; bridge pair
   `hir_builder_kwnames_get_c` / `hir_builder_kwnames_set_c` (or fold
   into existing builder-context bridge).

Plus optional bridge for **checkAsyncWithError** (returns 2-int tuple) —
small inline; could fold into the awaited-tail block bridge or keep as
standalone.

### REUSED bridges (no new work):
- `emitCallEx` (CALL_FUNCTION_EX path)
- `emitCallExceptionHandler` (CALL/CALL_KW/CALL_METHOD path post-handler)
- `emitInvokeFunction`/`Native`/`Method` (INVOKE_* paths, push 81)
- `phx_ptr_arr_pop`/`push` (stack manipulation)
- `temps_.AllocateStack` (already C)

### REABSORBED (deletion):
- `hir_builder_emit_awaited_call_tail_c` (149b7e2d40 PartialConversion bridge)
  inlined back into the C body of emitAnyCall — bridge declaration removed
  from `builder.cpp` + `builder_emit_c.c` + the `REABSORB-WHEN` comment block.

### W25c-mini compliance per supervisor L2441 refined invariant:
- All NEW bridges declare typed handle args (HirInstr/HirRegister/HirFunction)
  per W25b discipline at boundaries-with-typed-destinations.
- void*-intermediates allowed within all-void* call chains (e.g., bytecode
  iterator pointer pass-through).

---

## 4. Migration sequence

### Step A — bridge inventory + per-bridge LITE SPEC (~30min)

Generalist authors per-bridge LITE SPEC (D-1776823737 discipline) for
each of the 4 new bridges (5→4 after empirical scope reduction per
generalist [chat L2461] + theologian [chat L2466] (B) decision):
- Function signature (typed handles per W25b)
- Calling convention + ownership semantics
- C++ side delegation pattern
- C side body sketch

Theologian pre-audits each LITE SPEC before implementation begins.

### Step A falsification — bridge-inventory mutation test (~10min, MANDATORY)

Per pythia #86 + supervisor [chat L2469]: bridge inventory is a
COMPLETENESS ASSERTION (this set of N bridges is sufficient). Spec §3
empirical compile-clean enumeration is necessary but not sufficient —
W25 §5.3 lesson was that compile-clean inventory can have hidden gaps
(another bridge silently absorbs the drift). Step A falsification
catches inventory gaps BEFORE Step B implementation.

**Procedure (testkeeper):** at the post-Step-A HEAD (LITE SPECs
finalized, no Step B implementation yet), mutate one bridge signature
(add unused param to e.g.
`hir_builder_emit_call_method_exception_handler_inline_c`). The
mutation should be visible at expected call sites in any draft Step B C
body OR at the LITE SPEC's stated callers. Verify:
- BUILD_EXIT=2 with compile error at expected call site → bridge
  inventory is correct (mutation caught at the right location)
- BUILD_EXIT=0 → bridge inventory has GAP (mutation silently absorbed
  somewhere) → flag + reassess inventory before Step B

**Acceptance:** Step B begins ONLY if Step A falsification PASSES (drift
caught at expected location). Adds ~10min to W26 cycle but catches
inventory gaps at the boundary where they're cheapest to fix.

**General principle (per supervisor L2469 + theologian future-spec
amendment):** all future workstream specs require Step A falsification
test in their gate-completion criteria. Bridge inventory completeness
must be empirically falsified, not just enumerated.

### Step B — single-commit conversion (~60-90min)

Single atomic commit:
- C body: `hir_builder_emit_any_call_c` in `builder_emit_c.c` (~150 lines)
- 5 NEW bridge declarations + implementations (mirror Tier 5 + push 81 patterns)
- C++ `HIRBuilder::emitAnyCall` reduced to ~7-line delegation stub
- 149b7e2d40 PartialConversion REABSORBED: `hir_builder_emit_awaited_call_tail_c`
  declaration + impl + REABSORB-WHEN comment block REMOVED from
  `builder.cpp` + `builder_emit_c.c` (the bridge body inlines into the C body)

Verification: full build + dual-arch + 7-test gate (6 falsifier + 1 W21 golden).

### Step C — falsification (no new spec test, existing infra catches)

emitAnyCall is exercised by virtually every benchmark + test. Existing
test surface (falsifier 6/6, W21 golden, full test suite) IS the
falsification. No §5.3-style mutation test needed — drift surface for
emitAnyCall args was already validated in W25 + W25b cumulative work.

---

## 5. Pre-flight risk assessment

### W25b lesson application

W25b expanded mid-flight when 25+ caller errors surfaced. Avoiding
recurrence:
- Step A bridge inventory FIRST surfaces real scope before implementation
- Per-bridge LITE SPEC lets theologian pre-audit catch surprises early
- If Step B compile reveals additional bridges needed beyond the 5 in §3,
  HALT + reassess scope before continuing (don't expand mid-implementation)

### Templates + macros risk

`emitVariadic<VectorCall>` is a C++ template. C side cannot directly
instantiate; bridge must be type-erased. Likely pattern: bridge takes an
"opcode class" enum and dispatches internally to the right C++ template
instantiation. OR: bridge inlines the relevant fields + makes the
underlying call directly.

`PY_VERSION_HEX >= 0x030C0000` conditional in is_awaited computation: same
pattern as INVOKE_* #3 — fold into bridge OR use #if in C body.

### `kwnames_` member access

Class member field. Two strategies:
- Get/set bridge pair (clean but 2 bridges per access)
- Direct field access via offsetof + cast (faster, no bridge, same as Layout structs)

Recommend direct field access if field offset is stable (per
`hir_instr_c_verify.cpp` static_assert pattern). One static_assert, then C
body reads/writes directly.

---

## 6. Rollback

Single Step B commit. `git revert` restores emitAnyCall C++ + the
PartialConversion bridge. Working tree matches pre-W26 state. All 5 new
bridges removed, REABSORB-WHEN comment restored.

Risk concentration in Step B atomicity. If unexpected scope explosion (>5
new bridges) surfaces during implementation: HALT + revert + reassess.

---

## 7. What this does NOT solve

- emitVariadic-related call sites OUTSIDE emitAnyCall (e.g., other emit-
  methods that use emitVariadic for different opcodes). W26 only covers
  emitAnyCall's CALL_FUNCTION/CALL_FUNCTION_KW path. Any other
  emitVariadic users get the same C bridge as a side-benefit (one bridge
  serves all callers).
- Other PartialConversion artifacts in the codebase. None currently exist
  per `nbs-ts-grep PARTIAL CONVERSION ARTIFACT` survey (only 149b7e2d40).
- emitAnyCall's POST-CONVERSION optimization (e.g., switch → table dispatch).
  Out of scope; mechanical conversion preserves existing structure.

---

## 8. Connection to Tier 6 ZERO C++ goal

After W26 lands:
- Pure-C++ emit-method count: 97/123 → 96/123 + emitAnyCall removed from
  the partial-conversion-pending list = 96/123 - 1 = **95/123** unique
  emit-methods C-converted (Phase 2 substantive completion + emitAnyCall
  full).
- 19 pure-C++ emit-methods remain (per push 81 close framing).
- 149b7e2d40 PartialConversion artifact removed from codebase.
- REABSORB-WHEN trigger fully resolved.

Path to ZERO C++: 19 remaining emit-methods × ~30min average = ~9-10hr
additional sessions. Future sessions / W27+ workstreams.

---

## 9. Sequencing

1. **Spec review** (now): supervisor + gatekeeper review this doc.
2. **Step A** (~30min): generalist authors per-bridge LITE SPEC for 5 new
   bridges. Theologian pre-audits.
3. **Step B** (~60-90min): single-commit C body + 5 new bridges + reabsorb
   149b7e2d40 PartialConversion.
4. **Step C** (~10min): testkeeper STRICT verify + 7-test gate (existing
   falsifier + golden + partial-conv).
5. **Workstream close**: update this doc's §5 with empirical result.
6. **149b7e2d40 reabsorb verified**: `nbs-ts-grep "PARTIAL CONVERSION
   ARTIFACT"` returns 0 hits in `Python/jit/hir/`.

Total: ~100-130min focused work (matches generalist L2447 90-120min range +
spec/review overhead), 1-2 push gates.

After W26 close: 19 remaining pure-C++ emit-methods queued for future
workstreams.
