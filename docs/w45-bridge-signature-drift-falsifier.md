# W45 — Bridge-Signature Drift Falsifier

**Status:** FILED background workstream candidate per supervisor
2026-04-23T21:55:36Z + theologian 2026-04-23T21:55:24Z + pythia
#90 substantive concern.

**Owner:** TBD (theologian sketch + generalist impl).

**Estimated cost:** ~1-2 hours infra (similar pattern to W44
scripts/check_do_not_use_callers.sh).

**Activates when:** Phase 1 burndown reaches Cat-B fold-into-C
territory (emitBeforeWith / emitSetupWith) — preventive, not
blocking.

---

## 1. Problem statement

Phase 1 'delegation-stub delete' pattern (Phase 1 #1-#5, today)
replaces typed C++ overload-resolved calls:

```cpp
emitCopyFreeVars(tc, nfreevars);  /* TranslationContext& tc, int */
```

with direct C-bridge calls where the signature uses void* params:

```cpp
hir_builder_emit_copy_free_vars_c(&tc, current_func_, this, code_, nfreevars);
/* C signature: void *tc, void *func, void *builder, void *code, int */
```

builder.cpp now has 252+ such dispatch-switch call sites (post Phase
1 #4). The PC1 C++-side carve-out (D-1776896740 'C++ class hierarchy
already type-safe') was justified one-time, but Phase 1 multiplies
the exposed surface from ~20 INVOKE_* bridges (pythia #88 scope) to
252+ dispatch sites at ZERO-C++ landing.

### 1.1 Drift surface

If a future C-side refactor reorders or inserts void* params in a
bridge signature (e.g., swap `func`/`builder`/`code` order), the
C-side will pass `-Werror=incompatible-pointer-types` because C
typed handles (W25b) catch C-side miswire. But the C++ caller side
silently passes void* args in the OLD order because:
- All caller args (`&tc`, `current_func_`, `this`, `code_`) are
  silently erased to `void*` at the bridge crossing
- C++ overload resolution doesn't fire (the call is to a free C
  function, not an overloaded class method)
- The compiler accepts any void* in any position

Material risk: silent miswire surfaces only at runtime, far from the
refactor commit. Bisection lands on a refactor that "passes all gates"
but actually broke 2 of 252 dispatch sites.

## 2. Resolution

### 2.1 Mechanism

`scripts/w45_bridge_drift_falsifier.sh` — falsification test that
mechanically mutates a bridge signature and verifies the build fails
at expected dispatch-switch call sites.

Pattern:

```bash
# 1. Pick a sample bridge function (e.g., hir_builder_emit_copy_free_vars_c)
# 2. Mutate its signature (e.g., insert a fake void* param at position 0)
#    in builder_emit_c.h declaration
# 3. Run build
# 4. Verify build fails at dispatch-switch call site in builder.cpp
# 5. Restore signature; run build; verify build passes
```

If step 4 fails (build passes despite signature mismatch) → drift
class is real and W45 verifier surfaces a structural gap.

### 2.2 Sample bridges

Initial fixture set (covers spectrum of arg counts):
- `hir_builder_emit_copy_free_vars_c` (5 args: tc, func, builder,
  code, int)
- `hir_builder_emit_format_simple_c` (3 args: tc, func, builder)
- `hir_builder_emit_get_yield_from_iter_c` (5 args: tc, func,
  builder, int, void*)
- `hir_builder_emit_primitive_load_const_c` (5 args: tc, func,
  builder, code, oparg)
- One bridge per Phase 1 batch (#1-#5)

### 2.3 Acceptance criteria

1. Script mutates ONE bridge sig at a time + restores after each test
2. Each mutation triggers build failure at expected dispatch-switch
   site (test PASS)
3. If any mutation lets build pass silently → STRUCTURAL DRIFT
   surface confirmed; surface as W45-class blocker for future Phase
   1 batches
4. Integration into `gate_phoenix.sh` as Step 1f static analysis
   (~5min cost per push) — sibling of W44 Step 1e gate

### 2.4 What W45 doesn't catch

- Runtime semantic drift (the bridge does the wrong thing internally)
  — that's W42 (refcount-correctness) territory
- Bridge growth beyond ≤5 budget per W25b — that's W33 ZERO-bridge
  verifier territory
- Caller-side use of `_DO_NOT_USE_*` helpers — that's W44 caller-gate
  territory

W45 is structural-type-discipline-only.

### 2.5 Section 3.5 — fold-into-C derivation falsifier (pythia #91 2026-04-23)

When a fold-into-C refactor (Phase 1 #6 emitBeforeWith pattern) moves
algorithmic derivation logic INTO the C body (e.g., PY_VERSION_HEX +
opcode → enter_id/exit_id/is_async picking), the C body becomes
load-bearing for the derived constants. Failure mode (per W26
cinder_opcode.h Py_OPCODE_H header-shadow class precedent at
D-1776902133): a future header reorganization silently shadows the
opcode constants used in the derivation, the C body picks the wrong
flavor, and the only catch is W21 golden trip wire after-the-fact.

W45 §3.5 adds a falsification fixture for fold-into-C derived
constants: mutate the opcode constant the derivation depends on
(e.g., #define BEFORE_WITH 999 in a test header), verify the build
fails OR the W21 golden detects divergence. Specific to fold-into-C
batches; not blanket coverage.

**§3.5 trigger (theologian + supervisor 2026-04-23T22:47:15Z compound):**
§3.5 implementation activates on whichever fires FIRST:
- **(A) W21 golden integration landmark**: §3.5 depends on W21 golden
  trip-wire infrastructure (the only structural detector for header-
  shadow class per memory L92). Implements when W21 integration is
  convenient.
- **(B) 5-batch cap (forced backstop)**: 5 batches without §3.5 =
  mandatory revisit. Counted batches: fold-into-C (Phase 1 #6+#7) +
  Phase 3 state-extraction batches (per pythia #93 amendment 2026-
  04-23). Phase 1 fold-into-C count = 2 (#6 emitBeforeWith + #7
  emitSetupWith). Phase 3 batches as of 2026-04-23 = 2 (Batch 1
  c905827742 + Batch 2 1343895045). Combined count = 4; 1 more
  qualifying batch reaches cap. Avoids indefinite spawn-vs-drain
  per pythia #91+#93.

**AMENDMENT per pythia #93 2026-04-23 (Phase 3 inclusion):** The
original (B) trigger only counted fold-into-C batches; Phase 3
state-extraction batches accrued indefinitely without contributing
to the §3.5 backstop count. Phase 3 batches now COUNT toward the
5-batch cap because they introduce derived-state access patterns
(bridge-derived field-reads, see Batch 2 entry_cpp / size_cpp) that
are structurally analogous to fold-into-C derived constants — both
classes are subject to header-reorganization shadowing and silent
divergence.

Compound trigger ensures §3.5 lands when most useful AND time-bounded.

## 2.6 Retro regression-tests for Phase 1 #6 + #7 (pythia #91 2026-04-23)

Per pythia #91 spawn-vs-drain framing: W45 was filed AFTER Phase 1 #6
(emitBeforeWith fold-into-C, a338a57d55, 2026-04-23T22:15:59Z)
landed; Phase 1 #7 (emitSetupWith fold-into-C, 37e80859e7, 2026-04-23
T22:32:48Z) followed before W45 implementation.

W45 first acceptance criteria (per supervisor 2026-04-23T22:33:17Z):
include RETRO regression-test fixtures for both #6 and #7 sig
mutations:

1. **emitBeforeWith fixture**: mutate `hir_builder_emit_before_with_c`
   signature in `builder_emit_c.h` (e.g., drop the opcode arg). Build
   must fail at builder.cpp dispatch site (line 2404 area). Verifies
   #6 mutation is now protected.

2. **emitSetupWith fixture**: mutate `hir_builder_emit_setup_with_c`
   signature (e.g., drop oparg). Build must fail at dispatch site
   (line 2413 area). Verifies #7 mutation is now protected.

If either mutation lets build pass silently → STRUCTURAL DRIFT
confirmed; W45 implementation incomplete; revisit.

## 2.7 §3.5 Implementation sketch — derivation-drift falsifier (5/5 trigger fired 2026-04-24)

**Status:** TRIGGER FIRED post-Phase 3 Batch 4 land (b44a5143cc, 2026-04-24
~00:11Z). 5-batch backstop reached: W26 fold-into-C #6 a338a57d55 + #7
37e80859e7 + Phase 3 Batch 1 c905827742 + Batch 2 1343895045 + Batch 4
b44a5143cc = 5/5. §3.5 implementation now ACTIVE workstream.

**Owner:** TBD per supervisor disposition (theologian sketch + generalist
impl per W45 §1-§2 + W44 caller_grep.sh template precedent).

**Estimated cost:** ~2-3 hours infra one-time + ~15min per future fixture
addition.

### 2.7.1 Mechanism

`scripts/w45_section_3_5_derivation_drift.sh` — Mutate-Build-Verify-Restore
loop (analogous to W45 §1-§2), but on DERIVED CONSTANTS / DERIVED
FIELD-LAYOUTS instead of bridge signatures. Two operating modes per
compound trigger (A) W21 / (B) source-mutation:

(A) **W21 golden-driven** (when W21 lands): mutate derived constant or
field offset; build SUCCEEDS (silent shadow); run translate-test; compare
HIR output to W21 golden. PASS criterion: golden divergence detected.
FAIL criterion: golden output unchanged → silent shadow confirmed,
implementation incomplete.

(B) **Source-level fixture-driven** (no W21 dep, ships first): scripted
sed-style mutation; run `scripts/build_phoenix.sh`; expect `BUILD_EXIT
!= 0`; restore via `git checkout`. PASS criterion: build failed at
expected C-body site. Cost: ~80s per fixture serial; ~2min for 4-fixture
set parallel.

### 2.7.2 Fixture set (initial — 4 fixtures spanning both classes)

**Class A — fold-into-C derived constants (Phase 1):**

1. **emitBeforeWith opcode-derivation fixture** (W26 a338a57d55): C body
   in `builder_emit_c.c` picks `enter_id`/`exit_id`/`is_async` based on
   opcode comparison `BEFORE_WITH` vs `BEFORE_ASYNC_WITH`. Mutation:
   inject `#define BEFORE_WITH 999` in test header that shadows real
   opcode constant. Verify build FAILS at C-body comparison site OR (B-mode)
   W21 golden detects translate-time divergence.

2. **emitSetupWith identifier-derivation fixture** (W26 37e80859e7):
   Spec sketch suggested `SETUP_WITH` opcode mutation, but the actual
   load-bearing surface in `emitSetupWith` C body is `_Py_ID(...)` macro
   for dunder-method lookup (`__aenter__`/`__aexit__`/`__enter__`/
   `__exit__`). Implementation (bfc6321b77) targets `_Py_ID` rename in
   line range covering the emitSetupWith body region. Substantively
   equivalent — both classes are derivation-drift of constants the C
   body load-bears on.

**Class B — Phase 3 bridge-derived field-reads:**

3. **Phase 3 Batch 2 entry_cpp/size_cpp fixture** (1343895045): C body
   iterates `exception_table_` via `_entry_cpp` bridge return-type
   `ExceptionTableEntry*`. The C body assumes specific struct field
   layout (start/end/handler/depth/lasti). Mutation: reorder
   `ExceptionTableEntry` field declaration in C-side mirror struct (or
   mutate bridge return-type signature to return adjacent field). Verify
   build FAILS at C-body field access site (`find_exception_handler_c`).

4. **Phase 3 Batch 4 block_map_blocks_lookup_cpp fixture**
   (b44a5143cc): C-side bridge return value `BasicBlock*` consumed by
   `hir_builder_get_block_at_off` in `hir_c_api.cpp:2629`. Mutation:
   change bridge return-type from `void*` to `int` in
   `builder_state_c.h`. Verify build FAILS at hir_c_api.cpp caller site.

### 2.7.3 Acceptance criteria

1. Each of 4 initial fixtures triggers BUILD FAIL (B-mode) OR W21 GOLDEN
   DIVERGENCE (A-mode when W21 lands).
2. Restore via `git checkout HEAD -- <file>` after each fixture
   (deterministic cleanup; idempotent).
3. Integration into `scripts/gate_phoenix.sh` as Step 1g (after W45 §1-§2
   bridge-sig + W44 DO-NOT-USE-callers). Cost ≤5 min per push.
4. False-positive rate: zero (fixtures designed to trigger only on
   intended drift class; whitelist mechanism per W45 §1-§2 precedent if
   needed).
5. Future fold-into-C / Phase 3 bridge-derived-read addition: NEW fixture
   appended to fixture set in SAME COMMIT as the burndown change (per
   shepard atomic-commit + same-commit-fixture discipline). Prevents
   spawn-vs-drain regression.

### 2.7.4 Initial fixture coverage gap (acknowledged)

The 4 initial fixtures cover: 2 fold-into-C (Phase 1 #6+#7) + 2 Phase 3
bridge-derived (Batch 2 + Batch 4). Phase 3 Batch 1 foundation
(c905827742) introduced PhxHirBuilderState struct + parseExceptionTable
port — these are NOT bridge-derived field-read shape (parseExceptionTable
WRITES exception_table_ via push_cpp; doesn't READ via field-derived).
Fixture set deliberately scopes to READ-side derivation drift; write-side
bridges are covered by W45 §1-§2 (signature-mutation).

### 2.7.5 Cross-link to W21

When W21 golden trip-wire integration lands, §2.7.1 mode (A) becomes
preferred: golden-divergence is structurally stronger than build-failure
because it catches type-compatible silent shadows that compile clean
(e.g., field reorder where both old and new fields are `int`, builds
fine, produces wrong HIR output). §2.7.1 mode (B) source-mutation is the
ships-first stopgap pending W21 landing.

## 3. Cross-link

- Pythia identification: pythia #90 2026-04-23T21:54:37Z
- Theologian filing: 2026-04-23T21:55:24Z
- Supervisor authorization: 2026-04-23T21:55:36Z
- Pattern precedent: `scripts/caller_grep.sh` (push 92), W44
  `scripts/check_do_not_use_callers.sh`
- Memory: `feedback_grep_before_counts.md` (count-discipline) +
  `feedback_no_workarounds.md` (no symptom-fix without root-cause)
- Sibling workstreams: W33 (zero-bridge verifier), W42 (refcount-
  correctness verifier), W44 (DO-NOT-USE caller gate)
- Empirical surface: 252+ hir_builder_emit_*_c call sites in
  builder.cpp post Phase 1 #4 (35fb5f9777..514df3c4d1)
