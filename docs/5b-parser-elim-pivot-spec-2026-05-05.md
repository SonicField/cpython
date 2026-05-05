## Phase 5.B — parser.cpp ELIMINATE via Inliner-Consumer PIVOT

**Author:** theologian (per project_bridge_spec_template.md, light-scope variant)
**Date:** 2026-05-05 (post-5.A3 chain-close, gated on cap-check + §3a results)
**Source:** Python/jit/lir/parser.cpp (~600 LOC) + Python/jit/lir/parser.h (~50 LOC) + lir_parser_parse extern in lir_impl_internal.h:121
**Target:** ELIMINATE both files. Pivot single production consumer (inliner_c.c:86) to use pre-built LirFunction* instead of parsed LIR text.
**Class:** ELIMINATE-class (Q-5-2 path); 3 commits estimated; non-M3 wave (single consumer); F3b precedent applies for performance gating.
**Falsifier:** structural diff (printer_c.c byte-match serialization) of pivoted-construction LirFunction* vs parser-output LirFunction* reference at commit 1.

---

## §0 — Scope + non-goals

**In scope:**
- c_helper_translations.c: replace `init_cast_lir()` snprintf-builds-text with `init_cast_lir_function()` that constructs JITRT_Cast LirFunction* programmatically via existing C bridges.
- c_helper_translations.h: change `jit_lir_map_c_helper_to_lir(uint64_t addr)` return type from `const char*` to `LirFunction*` (or equivalent opaque pointer per existing extern convention).
- inliner_c.c:76-86: skip parse step; pass pre-built LirFunction* directly to lir_function_copy_from.
- DELETE Python/jit/lir/parser.cpp (~600 LOC).
- DELETE Python/jit/lir/parser.h (~50 LOC).
- DELETE `int lir_parser_parse(const char*, void**)` extern from lir_impl_internal.h:121.
- DELETE any orphaned build/cmake/Makefile.jit refs to parser.cpp/.h (per generalist 15:09:25Z parallel grep).
- DELETE any test fixtures that exercise parser-only code paths (test fixtures don't count per Q-5-2 precondition).

**Out of scope:**
- jit_lir_map_c_helper_to_lir surface beyond the JITRT_Cast helper (only one helper currently mapped; future helpers can extend the pivoted API uniformly).
- printer_c.c (LIR text serializer) — retained as diagnostic / dump tool; no functional change.
- Any other LIR text producer/consumer; grep confirmed parser is the only LIR text consumer.

**Move ≠ rewrite (project_bridge_spec_template):** the JITRT_Cast LIR is structurally equivalent across both paths. Pivot constructs the SAME LirFunction* shape — just programmatically instead of via text-then-parse. No semantic change to the inliner's behavior on JITRT_Cast.

---

## §1 — JITRT_Cast LirFunction* construction sequence

The reference shape is the current snprintf'd text in c_helper_translations.c:26-56. Pivot constructs an equivalent LirFunction* via existing C bridges. Sequence (mirrors snprintf order):

```c
static LirFunction *cast_lir_func = NULL;
static int cast_lir_function_initialized = 0;

static void init_cast_lir_function(void) {
    LirFunction *func = lir_function_new(NULL);  /* hir_func = NULL for c-helper */
    LirBasicBlock *bb0 = lir_function_alloc_block(func);
    LirBasicBlock *bb1 = lir_function_alloc_block(func);
    LirBasicBlock *bb2 = lir_function_alloc_block(func);
    LirBasicBlock *bb3 = lir_function_alloc_block(func);
    LirBasicBlock *bb4 = lir_function_alloc_block(func);

    /* BB 0: 2 LoadArg + Move(ob_type) + Equal + CondBranch */
    /* %5 = LoadArg 0 */
    /* %6 = LoadArg 1 */
    /* %7 = Move [%5 + offsetof(PyObject, ob_type)] */
    /* %8 = Equal %7, %6 */
    /* CondBranch %8 */
    /* successors: bb2, bb1 */

    /* BB 1: Call PyType_IsSubtype + CondBranch */
    /* %10 = Call PyType_IsSubtype, %7, %6 */
    /* CondBranch %10 */
    /* successors: bb2, bb3 */

    /* BB 2: Return success (%5) */
    /* successors: bb4 */

    /* BB 3: Move(tp_name) x2 + Call PyErr_Format + Move(0) + Return */
    /* %13 = Move [%7 + offsetof(PyTypeObject, tp_name)] */
    /* %14 = Move [%6 + offsetof(PyTypeObject, tp_name)] */
    /* Call PyErr_Format, PyExc_TypeError, "expected '%s', got '%s'", %14, %13 */
    /* %16 = Move 0 */
    /* Return %16 */
    /* successors: bb4 */

    /* BB 4: empty exit */

    cast_lir_func = func;
    cast_lir_function_initialized = 1;
}
```

**Construction invariants:**
- (I-1) BB allocation order: bb0-bb4 in lir_function_alloc_block call order. Matches snprintf BB id order (%0, %1, %2, %3, %4).
- (I-2) Successor wiring: bb0→{bb2, bb1}, bb1→{bb2, bb3}, bb2→{bb4}, bb3→{bb4}. Matches snprintf successor lists exactly.
- (I-3) Per-instruction operand types: LoadArg (immediate constant input), Move (memory-indirect: base + offset), Equal (two register inputs), Call (function pointer + arg list), CondBranch (single input), Return (single input).
- (I-4) Offset substitutions: 3 offsetof values must match snprintf substitutions exactly:
  - bb0 %7 Move base offset = `offsetof(PyObject, ob_type)`
  - bb3 %13 Move base offset = `offsetof(PyTypeObject, tp_name)`
  - bb3 %14 Move base offset = `offsetof(PyTypeObject, tp_name)` (same as %13)
- (I-5) Function pointer constants: PyType_IsSubtype (bb1 Call), PyErr_Format (bb3 Call), PyExc_TypeError (bb3 Call arg). Use `lir_operand_set_constant` with raw addresses.
- (I-6) String literal: bb3 PyErr_Format format string `"expected '%s', got '%s'"`. Constant operand pointing to .rodata literal.
- (I-7) Instruction ids: must be SSA-unique across function (0-16 in source ordering); use `lir_function_allocate_id` per instruction in construction order.

---

## §2 — Equivalence falsifier (commit 1)

**Canonical falsifier per supervisor 15:09:25Z:** byte-match serialization via existing `printer_c.c` LIR dumper.

**Test:**
1. Build snprintf-text-then-parse path (current) → reference LirFunction* `ref_func` at JITRT_Cast addr.
2. Build pivoted-construction path → test LirFunction* `pivot_func`.
3. Serialize both via `printer_c.c` LIR dumper to char buffers.
4. Assert byte-for-byte match: `memcmp(ref_buf, pivot_buf, len) == 0` AND lengths match.

**Pass criteria:** byte-match. Any divergence = bug in pivot construction; BLOCK c2 (delete) until resolved.

**Commit 1 scope:** PARALLEL — both paths exist. Pivot LirFunction* available via new `jit_lir_map_c_helper_to_lir_func` (or signature-changed existing function depending on rollout strategy); old `jit_lir_map_c_helper_to_lir` retained until equivalence verified. Post-verify, commit 1 wires inliner_c.c to use pivoted path (sole-path flip).

**Note on rollout strategy:** since signature change (const char* → LirFunction*) breaks ABI of jit_lir_map_c_helper_to_lir, two options:
- (a) atomic signature change in commit 1 (cleaner, but requires both paths in same commit + falsifier inline)
- (b) add new function jit_lir_map_c_helper_to_lir_func, retain old, switch consumer in commit 1, delete old in commit 2 (cleaner separation, slightly more LOC churn)

Recommend (b) for falsifier-test cleanliness; (a) viable if generalist prefers atomic.

---

## §3 — Per-commit gate

Standard per CLAUDE.md:
- 4-bench --reps=3 ABBA after each commit; BLOCK at >5% geomean drop.
- Wiring-gate via `gate_phoenix.sh --wiring`: post-5.A3, the `inline_caller_two_calls` entry already exercises lir_function_copy_from path including jit_lir_map_c_helper_to_lir consumption. JITRT_Cast inlining is the canonical test surface; if c_helper translation pivot breaks anything, wiring-gate catches it.

**Chain-close (post-c3):** standard 24-bench cap-check ABBA + ARM64 v+vi gate per CLAUDE.md Push Authorization protocol.

---

## §4 — Commit table

| # | Commit | Files | LOC est | Tier |
|---|--------|-------|---------|------|
| 1 | Pivot init_cast_lir → init_cast_lir_function + add jit_lir_map_c_helper_to_lir_func + switch inliner_c.c consumer + equivalence falsifier test | c_helper_translations.{c,h} (+90/-60) + inliner_c.c (+5/-15) + tests (+30) | +95/-75 | 1 (parallel during verification, sole-path post-test PASS) |
| 2 | DELETE parser.cpp + parser.h + lir_parser_parse extern + delete old jit_lir_map_c_helper_to_lir | parser.{cpp,h} (-650) + lir_impl_internal.h (-2) + c_helper_translations.{c,h} (-30) | -682 | 2 (sole-path; depends on c1 falsifier PASS) |
| 3 | Cleanup: build/cmake/Makefile.jit refs to parser.cpp/.h per generalist 15:09:25Z grep + delete any orphaned parser-test fixtures | per generalist enumeration | TBD (~10-30 LOC) | 1 (cleanup) |

**Net LOC (estimated):** +95 - 682 + ~20 = roughly -570 LOC.

---

## §5 — Risks + mitigations

| Risk | Mitigation |
|------|------------|
| JITRT_Cast LirFunction* shape divergence between text-parse and pivot-construct paths | §2 byte-match printer_c.c serialization falsifier at commit 1; BLOCK c2 if divergence |
| Future C helper additions need a new programmatic builder per helper (vs current generic snprintf+parse pattern) | Currently 1 helper; pivot is small. Future helpers can introduce per-helper builders OR re-introduce a parser if scale demands. YAGNI bound. |
| Build/cmake refs to parser.cpp not enumerated → c2 link failure | Generalist 15:09:25Z parallel grep enumerates pre-c1; c3 sweeps any residue |
| Loss of human-debuggable LIR text for c_helpers | printer_c.c retained; can dump LirFunction* on demand for debug |
| inliner_c.c return-NULL semantics change with signature swap (NULL char* vs NULL LirFunction*) | Both are pointer-NULL-checks; semantic preserved |

---

## §6 — References

- D-1777614132 — Phase 5 LIR pre-analysis: 5.B = parser.cpp ELIMINATE precondition
- D-1777614196 — supervisor Q-5-2 disposition: ELIMINATE preferred over C-port pending grep
- generalist 15:06:17Z — full-repo grep surfaced 1 production consumer (inliner_c.c:86)
- theologian 15:07:41Z — option (2) PIVOT recommended over option (1) C-port
- supervisor 15:07:57Z — PIVOT confirmed
- generalist 15:08:43Z — verification (1) golden-fixture grep clean; pivot-feasibility HIGH
- supervisor 15:09:25Z — light-spec authorization
- project_bridge_spec_template.md — spec template (light-scope variant for ELIMINATE-class)
- feedback_no_workarounds.md — substrate-elimination preferred over keep-and-port

---

Spec complete. Lightweight per ELIMINATE-class scope. Ready for generalist commit 1 implementation.
