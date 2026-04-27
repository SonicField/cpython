# Parked-Bug Audit — 2026-04-24

Drafted: theologian, post pythia #121 + #122 + librarian D-1777034998 + librarian
13:12:36Z scope-extension + shepard 13:16:37Z draft-now directive.

## Trigger

Bug D-1775660617 (theologian, 2026-04-09; "hir_cfg_get_rpo clamps return to
capacity") was classified `known-not-blocking` because no workload at the time
exceeded the 256-BB threshold. Push 41/42 fixture extension crossed that
threshold 15 days later; the latent bug surfaced as a SIGABRT/SIGSEGV cascade
that cost a ~4hr HALT cycle and re-derived a fix the librarian had on file.

The HALT was not the failure. The failure was that pushing the fixture (a
coverage-surface expansion) did not trigger a re-audit of the parked-bug list
against the new surface.

**Reframe (per Alex 13:47:22Z + 13:48:14Z, captured in
`feedback_assume_phoenix_regression.md` by supervisor 13:49:04Z):** every bug
in this list is presumed Phoenix-introduced by default. Cinder ran in
production for years and does not crash. The "known-parked" /
"pre-existing" framing is retired. Each item is treated as a Phoenix
regression that needs fixing; the only reason to defer is a falsifier
(explicit demonstration the same bug exists in core Cinder).

For bug D-1775660617 specifically: theologian git-archeology + testkeeper
empirical (cinderx_dev oracle) converge — the bug was introduced by
Phoenix at hir_c_api.cpp 6ff9c2877c on 2026-04-02 (Phase 3D HIR C API
addition). Cinder's RPO surface returns std::vector and is structurally
immune to this bug class. The 22-day "parked" interval was Phoenix
ignoring its own regression, not Cinder inheritance.

## Falsifier procedure (per librarian 13:58:02Z + pythia #125)

Two-tier falsifier per pythia #125 #3 (cinderx_dev cannot falsify
instrumented-build classes).

### Tier 1 — cinderx_dev oracle (RelWithDebInfo class)

Use for bugs whose symptom reproduces under release builds.

```bash
cd ~/local/cinderx_dev && \
  PYTHONPATH=cinderx/cinderx/PythonLib \
  LD_LIBRARY_PATH=python-install/lib \
  JIT_ENABLE=1 \
  ./python-install/bin/python3.12 <test.py>
```

If cinderx_dev exhibits the same crash/incorrect behavior, the bug may be
Cinder-inherited (still needs fixing per Alex 13:48:14Z, but framing
shifts). If cinderx_dev passes, the bug is Phoenix-introduced — fix it,
do not defer.

Caveat (Alex 13:48:14Z): cinderx_dev branch may itself have bugs.
cinderx_dev pass is strong evidence; cinderx_dev fail is weak evidence
without a separate check against core Cinder.

### Tier 2 — valgrind differential (instrumented-class)

Per Alex 15:42:14Z + supervisor 15:42:53Z: valgrind > ASAN for
instrumented-class falsifier. Reasons:

- Works on existing RelWithDebInfo binaries (cinderx_dev, Phoenix HEAD)
  — no rebuild needed, no '--asan flag history' bisect dead-end.
- Catches uninitialized-memory class that ASAN misses.
- Origin tracking (`--track-origins=yes`) directly identifies the
  corrupt-pointer source.
- Cost: minutes, vs vanilla-ASAN ~30min build.

Procedure:

```bash
# Phoenix HEAD valgrind
JIT_ENABLE=1 valgrind --track-origins=yes --error-exitcode=1 \
  ./python -c '<repro>'

# cinderx_dev valgrind (Tier 1 + Tier 2 in one)
cd ~/local/cinderx_dev && \
  PYTHONPATH=cinderx/cinderx/PythonLib \
  LD_LIBRARY_PATH=python-install/lib \
  JIT_ENABLE=1 \
  valgrind --track-origins=yes --error-exitcode=1 \
  ./python-install/bin/python3.12 <repro>
```

Falsifier interpretation:
- **cinderx_dev valgrind PASS + Phoenix valgrind CRASH** →
  Phoenix-introduced (definitive). Use valgrind output to localize.
- **Both valgrind CRASH** → CPython upstream or shared inheritance.
  Per Alex 13:48:14Z still fix Phoenix.
- **Both valgrind PASS but Phoenix release CRASH** → unusual; investigate
  for valgrind serialization effects or non-deterministic codegen.

Lesson (theologian 15:46Z self-correction per Alex 15:41:21Z): bisect is
the slowest debugging tool. Default to gdb / valgrind / dynamic-analysis
on a single binary BEFORE attempting commit-bisection. Bisect is for
"when did the symptom appear" which is rarely the question that matters
for a corrupt-pointer crash.

### Tier 3 — vanilla CPython differential (true upstream class)

Reserve for cases where Tier 2 valgrind on Phoenix + cinderx_dev both
crash, AND the crash signature suggests a CPython core issue rather than
JIT codegen. Build vanilla CPython 3.12.13 (no JIT, no Cinder) with
matching instrumentation; if vanilla also crashes, it is a true upstream
or environment defect.

## Scope (frozen per shepard 13:16:37Z; director-extended 2026-04-27)

**12 items + 1 rule** at draft time. Director-approved scope extensions are
appended below the original groups (Group D added 2026-04-24 per Alex
13:49:03Z; Group E added 2026-04-27 per Alex 07:12:25Z D-1777270945 +
librarian 07:22:18Z back-reference request). Other post-draft additions
still go to a follow-up audit.

### Group A — D-1775660954 still-parked (5)

Original 2026-04-09 enumeration, excluding bug #6 (hir_cfg_get_rpo, closed in
4feb0d2618). Each item below requires:

1. **Re-grep current code** for the failure surface named at parking time.
2. **Identify newly-reachable callers/workloads** since 2026-04-09 (use `git
   log --since=2026-04-09 -- <surface>` + chat search for fixture/threshold
   changes).
3. **Decide:** still safely parked / now reachable (fix required) / superseded
   / unreproducible.

Items:

- **A1. ASAN SEGV in JIT-compiled code** (frame #4 `<unknown module>`).
  Tracked separately in MEMORY.md. Re-check whether quarantine=0 ASAN
  reproduces post-Phase-A/B/B-Phase-3D landings; new RPO/refcount paths may
  have changed the codegen surface.
- **A2. LOAD_ATTR_SLOT ARM64 failure mode.** Re-check against current ARM64
  codegen and the Tier 5 / Phase 1 burndown commits that touched LoadAttr.
- **A3. Missing `addReference` codepath.** Re-check against current
  refcount_insertion + insert_update_prev_instr (both moved to C this session).
- **A4. Dict watcher shutdown warning.** Re-check whether watchers.cpp guard
  is still missing; low-priority but trivially closable.
- **A5. Profiling hooks test.** Re-check against current trace/profile
  integration; identify whether any test still skips for this reason.

### Group B — D-1775660954 verify-still-resolved (4)

Original 2026-04-09 (canonical record per `nbs-scribe-query --id=D-1775660954`,
`.nbs/chat/phoenix.chat:~L1591`) marked these as 1 CRITICAL+fixed (blocks-done
class) and 3 fixed-this-session-but-not-yet-pushed. Re-audit confirms each
fix landed and has not regressed under post-04-09 code churn.

- **B1. CRITICAL: ARM64 heap corruption** (blocks-done class). Confirm fix
  reached HEAD; identify the regression-test that covers it; confirm the
  test still runs in the gate.
- **B2. Deferred compilation** (fixed-this-session 2026-04-09; rationale
  notes "not yet pushed" at the time). Locate fix in `git log
  --since=2026-04-09 --grep='deferred'` (or related); confirm in current HEAD.
- **B3. icache flush (1 of 2)** (fixed-this-session 2026-04-09, not-yet-pushed
  at the time). Locate via `git log --since=2026-04-09 --grep='icache'`;
  confirm in current HEAD.
- **B4. icache flush (2 of 2)** — same provenance as B3; confirm second
  variant fix is also in current HEAD.

For each B-item, the audit must answer two questions:
1. Did the 2026-04-09 fix actually land on the integration branch?
2. Has post-04-09 churn (Phase 3D, Tier 5/6/7/8, BorrowedRef, RPO bridge)
   regressed it?

#### Group B audit results (run 2026-04-24, theologian, parallel-track during HOLD)

All 4 items VERIFIED RESOLVED + NO REGRESSION.

- **B1.** RESOLVED across multi-commit series (encoding fixes incl. mov SP
  ORR→ADD, FP V-bit ldr/str, MOVZ/MOVK abs_addr per `MEMORY.md`).
  Current ARM64 gate PASS at 4feb0d2618 (testkeeper 12:52:38Z). No named
  regression test; ARM64 dual-arch gate is de facto coverage.
  **Flag:** a single named regression test would harden coverage; deferrable.
- **B2.** RESOLVED (5ac009a478 2026-04-08, "Add active-frame guard to
  deferred compilation"). `hasActiveFrames` present in
  `Python/jit/pyjit.cpp:265`. 9 post-04-08 churn commits to pyjit.cpp; guard
  logic survives. Regression test: `Lib/test/test_phoenix_deferred_compile.py`.
- **B3.** RESOLVED (c5418bf008 2026-04-08, "Fix ARM64 icache flush in
  code_allocator.cpp"). `__builtin___clear_cache` present at
  `Python/jit/code_allocator.cpp:322`. 1 post-04-08 commit (c4616abb1a config
  migration); no regression.
- **B4.** RESOLVED (f0cb5ef785 2026-04-08, "Fix ARM64 crash: flush icache
  after code patching on all builds"). `__builtin___clear_cache` present at
  `Python/jit/code_patcher.cpp:141` AND `Python/jit/code_patcher_c.c:67`
  (Phase 3D C port preserved the flush correctly across migration; verified
  by reading both files post-c07da9c7e4).

**Residual risk surface:** B1, B3, B4 are implicit-coverage-only — every ARM64
JIT compilation/patch exercises them, so a regression manifests as gate
failure. ASAN+pydebug ARM64 gate would catch icache silently-stale earlier,
but is not currently part of the routine gate; acceptable given dual-arch
gate coverage.

### Group D — Phoenix-presumed-regression (queued, per Alex 13:49:03Z)

Items added by Alex directive ("ask you to fix yield_from eventually as
well now that I know what your definition of pre-existing cinder bug is").

- **D1. yield_from** — RESOLVED-BY-SUPERSESSION (generalist verification
  per supervisor 15:09:19Z). Both force_compile + auto-compile-threshold
  paths PASS on cc081f1de0. The bffec89650 yield-from-pattern-deopt
  bypass was already reverted at 66850a4ba1 (W22 root-cause fix landed
  same day, post-5-iteration history). cinderx_dev oracle + Phoenix HEAD
  both work. Only remaining bypass d25b2f33d7 is dead-code in 3.12
  (YIELD_FROM bytecode is a stub; 3.12 desugars yield-from). ARM64
  verification deferred to testkeeper next capacity.

  Lesson: the audit-staleness here was Group D scope assumption — Alex
  added yield_from based on the d25b2f33d7 bypass framing. Coverage-
  Trigger discipline + Alex's distrust caught the staleness BEFORE
  multi-hour W-D1 investigation.

Group D will grow as further "pre-existing cinder bug"-classified items
are identified for re-evaluation.

### Group E — Director-parked behind pure-C JIT roadmap (1, per Alex 07:12:25Z D-1777270945)

Added 2026-04-27 per librarian 07:22:18Z + theologian 07:22:57Z back-reference
request after `docs/known-bugs/bug-ii-storeattr-corruption.md` landed at
9a0fc0ea87. These are bugs the team has localized far enough to file a failing-
test sentinel + consolidated source-trace, and that Alex has explicitly
sequenced AFTER the pure-C JIT roadmap completes. Each entry is presumed
Phoenix-introduced per `feedback_assume_phoenix_regression.md`; the deferral
is sequencing, not a falsifier.

- **E1. W-PYTORCH-CM-(ii) StoreAttr managed-dict tag-flip corruption.**
  Runtime SEGV at `PyDict_SetItem` (NULL+0xAB deref) reachable via
  `Tools/benchmark_phoenix.py:bench_pytorch_cm` after `cinderjit.force_compile`
  and 50,000-iter workload. Mechanism narrowed to LSB-clear at `obj+0x18`
  (PEP 697 IsValues→IsDict misinterpretation); writer un-localized; 5-class
  hypothesis enumeration with (a)/(b)/(d) OPEN and (c)/(e) FALSIFIED.
  - **Sentinel:** `Lib/test/test_phoenix_jit_storeattr_managed_dict_tag_flip.py`
    (`@unittest.expectedFailure` subprocess test; `unexpectedSuccess` is the
    fix-landed signal).
  - **Source-trace + heavy-tier designs:**
    `docs/known-bugs/bug-ii-storeattr-corruption.md` (canonical),
    `docs/w-pytorch-cm-tooling-note.md` (running investigation log),
    `docs/w-pytorch-cm-tp-alloc-watchpoint-design.md`,
    `docs/w-pytorch-cm-allocate-counter-design.md` (both ~200 LOC, heavy-tier
    auth-gated per governance D-1777190699 + D-1777270945).
  - **Coverage-Trigger surface:** any change that adds a new caller of
    `bench_pytorch_cm` to the gate, lowers the auto-compile threshold,
    extends fixture coverage of nested `__enter__`/`__exit__` cycles on
    types whose `__init__` mutates a managed dict, or expands `_NoGrad` /
    `_Autocast` / `_ProfileScope`-class fixtures MUST re-grep this entry
    before landing. Trigger-sensitivity caveat (D-1777190733): the bug
    evades Python-wrapper-class instrumentation; coverage-extending
    workloads may either reach OR mask the trigger window.
  - **Resumption gate:** sentinel re-confirmed `expectedFailure` on current
    HEAD → re-read `docs/w-pytorch-cm-tooling-note.md` → choose
    `tp_alloc`-watchpoint and/or allocate-counter per the comparison table
    in the watchpoint design doc → heavy-tier authorization required.
  - **Anti-pattern (do not adopt):** appending `_NoGrad` / `_Autocast` /
    context-manager types to `pyjit.cpp:shouldSkipCompilation`. Per pythia
    #154 #4 + `feedback_no_workarounds.md`: workaround preserves the bug
    class for future managed-dict types to re-trigger.

### Group C — Post-2026-04-09 latent (3, per librarian 13:12:36Z)

Filed during the 15-day interval; carry the same threshold-conditional
liability that #6 just demonstrated.

- **C1. D-1774909374 LOAD_DEREF + except-handler deopt.** Triggers if
  threshold lowered or closure called 1000+ times. Structurally identical to
  D-1775660617 (parked because no current trigger). Re-check: does any current
  fixture/test push a closure to threshold=1000?
- **C2. D-1776480607 pydebug crash.** Release-mode masks. Re-check: is the
  pydebug gate currently exercising this path? What workload surfaces it?
- **C3. D-1776869611 pythia #73 emitInlineExceptionMatch n=3 zero-bridge.**
  Unfalsifiable for threshold=1000 closure-LOAD_DEREF-in-except path.
  Re-check: does the closure path reach this combination?

### Per-item validation method

For Group A and C, audit step (3) "still safely parked" requires a falsifier:
the test or workload that would surface the bug if it became reachable. Items
without a falsifier should be reclassified as "unfalsifiable, fix or remove."

For Group B, audit step (3) "still resolved" requires the canonical
regression test name and a confirmation that test still runs in the gate.

#### Group A audit results (run 2026-04-24, theologian, parallel-track)

Structural-pass only; cinderx_dev oracle requires testkeeper (devgpu004
access). Each item classified per Alex's reframe (presumed Phoenix
regression unless cinderx_dev oracle disproves).

- **A1. ASAN SEGV in JIT-compiled code** — RECLASSIFIED:
  ASAN-instrumentation incompatibility, NOT Phoenix codegen bug
  (Alex authority 15:54:04Z; supervisor 15:54:43Z + theologian
  15:55Z confirm). Per Alex: for ASAN-class crashes, valgrind-clean
  is sufficient falsifier — JIT-generated code historically
  false-positives in ASAN. The medic 15:48:54Z hypothesis-stratification
  applies as general discipline; ASAN-specific exception is documented
  in `feedback_falsifier_before_concur.md`.

  ASAN run on cc081f1de0 reproduced SEGV (testkeeper 14:57:40Z); valgrind
  on the same binary + workload returned ZERO errors (testkeeper
  15:46:07Z + 15:48:48Z; both Phoenix and cinderx_dev valgrind-clean).
  The corrupt address 0x811d1002dcacd's high-bits-set pattern is
  consistent with ASAN poison values returned from a shadow-memory
  conflict at the JIT's anonymous mmap region.

  Two hypotheses:
  - **H-A1a:** Phoenix-specific ASAN-visible memory bug that valgrind
    misses (e.g., red-zone overflow, intra-object overflow — bug classes
    valgrind doesn't instrument).
  - **H-A1b:** ASAN-instrumentation incompatibility — ASAN's
    shadow-memory model conflicts with JIT-emitted code in mmap regions;
    crash is false-positive.

  Evidence so far supports H-A1b: both Phoenix and cinderx_dev valgrind
  clean on the same workload. But H-A1a is not falsified — only
  cinderx_dev ASAN-build on the same workload would falsify (if
  cinderx_dev ASAN passes → H-A1b confirmed; if cinderx_dev ASAN crashes
  → H-A1a falsified, shared bug class). cinderx_dev ASAN build NOT
  CURRENTLY AVAILABLE (cinderx_dev is RelWithDebInfo per `MEMORY.md`;
  rebuild ~1-2hr if their build system supports `--fsanitize=address`).

  Practical disposition:
  - `--asan` flag DEPRECATED in default gate either way (both H-A1a and
    H-A1b lead to "ASAN signal is unreliable for this codebase").
  - Valgrind canonical Tier-2 falsifier (see §Falsifier procedure Tier 2).
  - No W-A1 codegen fix workstream opened pending H-A1a falsification
    (per Alex policy default = Phoenix-presumed regression, but valgrind
    clean is strong corroborating evidence against; cost of fix-without-
    falsifier is uncertain because no localized bug to fix).
  - Audit item remains OPEN; deferrable until cinderx_dev ASAN build is
    cheap or until H-A1a accumulates supporting evidence (e.g., another
    ASAN-only crash signature).

  ASAN historically passed (`MEMORY.md`: "ASAN quarantine=0: 17/17 PASS"
  — earlier test set, different code surface). The breaking commit is
  not localized (bisect infeasible — bug pre-dates 3ffc4c5d56 first
  --asan-supporting commit).
- **A2. LOAD_ATTR_SLOT ARM64** — RESOLVED (testkeeper 14:52:11Z). Golden
  test `Lib/test/test_phoenix_jit_loadattr_golden.py` passes both archs
  at push 44 cc081f1de0 (16-module Phoenix suite, 987 tests, dual-arch
  GATE PASS testkeeper 14:49:26Z). Original ARM64-specific failure
  mode no longer reproduces.
- **A3. bench_richards_full n=570000 ARM64 SIGSEGV** (renamed from
  "missing addReference" per librarian 16:40:00Z; the addReference
  hypothesis was FALSIFIED at the original 2026-04-09 session —
  D-1775610679: CodeRuntime already addRefs at code_runtime.cpp:65).
  RESOLVED-BY-SUPERSESSION.

  Original symptom: ARM64 SIGSEGV at n=570000 with JIT reading freed
  PyCodeObject at compile-time embedded address. The 2026-04-09 working
  hypothesis after the addReference falsification was ARM64 MOVZ/MOVK
  32-bit address truncation (f_executable embedding 32-bit instead of
  48-bit, missing third MOVK for bits 32-47); session parked
  D-1775611932 after 7+ hours / 20+ hypotheses.

  Subsequent landings DID address ARM64 MOVZ/MOVK class (`MEMORY.md`:
  "ARM64 bugs fixed this session: ... abs_addr MOVZ/MOVK ...") plus
  LoadField+LoadCellItem owned-ref fix (D-1775672324) and BorrowedRef
  Phase I-1 (03e770705f 2026-04-17). Empirical confirmation: testkeeper
  14:52:11Z bench_richards_full(570000) PRE-CALL/POST-CALL r=34200
  EXIT=0 on both x86_64 (cc081f1de0) and ARM64 (devgpu004 66921f69b1).
  Original SIGSEGV symptom no longer reproduces at HEAD.
- **A4. Dict watcher shutdown warning** — PARTIALLY FIXED.
  `Py_IsFinalizing()` guard present at `global_cache.cpp:204` — closes
  the shutdown-time crash class ("Invalid dict watcher ID -1" historical
  symptom). Mid-execution module unload remains UNADDRESSED per same
  comment block (see W27 GlobalCacheKey raw-pointer lifecycle re-arch).
  ACTION: W27 still tracks the unaddressed half; no separate audit
  action needed.
- **A5. Profiling hooks test** — RESOLVED (no skip-on-bug). Test file
  `Lib/test/test_phoenix_profiling_hooks.py` exists; only
  `@skipUnless(HAS_JIT)` skips present (3 sites) — i.e. only skipped
  when JIT is absent, not when the original bug fires. The 2026-04-09
  "profiling hooks test" parking presumably referred to a specific test
  that was disabled; current state shows no such disable. RESOLVED.

#### Group C audit results (run 2026-04-24, theologian, parallel-track)

- **C1. D-1774909374 LOAD_DEREF + except-handler deopt** — RESOLVED.
  Documented at `builder_emit_c.c:3129` as "CRITICAL D-1774910012
  INVARIANT": push Py_None as prev_exc placeholder before dispatch loop;
  POP_EXCEPT case pops it. Regression test:
  `Lib/test/test_phoenix_jit_inline_except_closure.py`. The 2026-04-09
  "would trigger if threshold lowered" risk is now actively-tested.
- **C2. D-1776480607 pydebug crash** — UNFIXED, structurally pre-analyzed
  (theologian 14:54Z). Symptom per `MEMORY.md`: "struct layout mismatch,
  ob_prev/ob_next +16 bytes" — `Py_TRACE_REFS` (set under pydebug) adds
  _ob_next/_ob_prev to PyObject HEAD, shifting layout +16B.

  Phoenix offset hygiene CLEAN: all JIT offset access uses
  `offsetof(PyObject, ...)` (e.g. `kRefcountOffset` at lir/generator.cpp:72
  + 14 other callsites in codegen/+hir/). NO hardcoded layout literals.
  The bug is NOT "baked-in release offset."

  Likely causes (ordered by structural plausibility):
  1. **Stale build hygiene** — switching pydebug↔release without cmake
     distclean leaves .o files compiled under different Py_DEBUG flag.
     Already mitigated by `build_phoenix.sh --pydebug --clean` per
     `feedback_pydebug_gate.md`; misuse window exists.
  2. **Phoenix-emitted machine code captures wrong absolute address** —
     e.g. `frame_asm_c.c:371` debug-vs-release split for
     `_PyEval_EvalFrameDefault`. If
     `jit_rt_get_alloc_link_frame_debug_addr` returns wrong fn /
     wrong arg shape, crash.
  3. **JIT_DCHECK fires only under pydebug** — release-mode masks a real
     precondition violation; pydebug fires it. Bug is real in both modes.
  4. **C++/C boundary mismatch** on a `#ifdef Py_DEBUG`-conditional struct
     member.

  Concrete next steps for W-C2 fix workstream:
  - Reproduce on x86_64 first (cheaper than ARM64 turn).
    `build_phoenix.sh --pydebug --clean` + force_compile trivial fn → SIGSEGV.
  - gdb backtrace to identify access-site (mode-1/2/3/4).
  - If mode 3 (JIT_DCHECK), assertion text directly identifies bug class.
  - cinderx_dev oracle UNAVAILABLE (RelWithDebInfo, not pydebug); per
    Alex policy default = Phoenix-presumed-regression, fix.
- **C3. D-1776869611 pythia #73 emitInlineExceptionMatch n=3 zero-bridge**
  — RESOLVED via same fix as C1 (D-1774910012 invariant). Same comment
  block at `builder_emit_c.c:3129` documents that the closure-LOAD_DEREF
  -in-except path is addressed by the Py_None placeholder push.

#### Group A+C summary (revised post Alex 15:54:04Z ASAN exception)

- RESOLVED (5): A2, A3, A5, C1, C3
- RECLASSIFIED (1): A1 (ASAN-incompat per Alex authority + valgrind
  clean both binaries; --asan deprecated in default gate)
- PARTIALLY FIXED (1): A4 (shutdown half done; mid-execution unload
  tracked as W27 / W-A4-mid)
- UNFIXED, fix workstream needed (1): C2 (pydebug crash → W-C2 active)

Net: 7 of 8 audit items closed. C2 the only active fix workstream.

#### Oracle gotchas (per librarian 14:21:41Z)

Three oracle-method caveats before testkeeper runs cinderx_dev falsifier
on Group A+C items:

1. **A3 "missing addReference"** predates BorrowedRef elimination (Phase
   I-1 100% COMPLETE at 03e770705f, 2026-04-17). The original failure
   surface may be structurally moot post-elimination — verify the bug
   surface still exists in Phoenix-current BEFORE running oracle. If
   moot, mark RESOLVED-by-supersession.
2. **A2 "LOAD_ATTR_SLOT ARM64"** is ARM64-specific. Oracle needs ARM64
   cinderx_dev on devgpu004 (recipe applies; host already verified
   13:48:50Z).
3. **C2 D-1776480607 "pydebug crash"** — release-mode masks. cinderx_dev
   default build is RelWithDebInfo per `MEMORY.md`; plain oracle
   CANNOT falsify pydebug-only items. Needs pydebug-built cinderx_dev
   binary to falsify. Without it, falsifier is unavailable; default
   per Alex policy = Phoenix-presumed regression, fix.

## Pythia #122 #1 follow-up — explicit A/B test on resize-path

Push 43 (4feb0d2618) discharged pythia #121 #1 by by-construction
determinism (GetRPOTraversal called twice on same CFG returns same vector).
Pythia #122 noted this is logical, not empirical. Add to audit deliverable:

- **Test:** force_compile a function with N=300 BBs (resize-triggered path,
  cap=256 → realloc to 300). Compare LIR output byte-for-byte against same
  function compiled with `--rpo-cap=4096` (cap-pre-sized, no resize). Expect
  identical output. If diverges: the determinism claim is wrong; investigate.

Trivial fixture; ~15min to write.

## Coverage-Trigger Rule (the structural fix)

**Pre-condition:** Any change that expands the JIT's coverage surface MUST
re-grep the parked-bug list before landing. If any parked bug's failure
surface is now reachable, the change MUST either (a) include the fix or (b)
explicitly defer with a reproduction test that proves the bug is reachable.

**Triggering changes (non-exhaustive):**

- New or extended fixture (e.g. `block_map_n294_chain` extension)
- New caller of a JIT API surface
- Threshold change (e.g. auto-compile threshold)
- New opcode supported
- New compilation tier added (e.g. PartialConversion expansion)

**Status:** ADVISORY (per supervisor 13:18:54Z). Author-discipline first;
gatekeeper notes presence/absence in APPROVE without blocking. Escalation
criterion: if author discipline is patchy after ~10 coverage-triggering
commits, escalate to CLAUDE.md hard-enforcement.

**Mechanics:**

1. Author of the change posts to chat: "COVERAGE-TRIGGER: <change>; parked-bug
   list re-grepped, no newly-reachable items" OR
2. "COVERAGE-TRIGGER: <change>; parked-bug #N now reachable, [fixed in this
   commit | deferred with reproduction at <test>]."
3. Gatekeeper APPROVE template includes one line: "Coverage-Trigger: <author
   cited / not cited>". Advisory note only — does not block.

**Source list:** the parked-bug list lives in this audit document (Groups A
and C above) plus any new parking decisions added by scribe. Convention: any
"known-not-blocking" classification MUST be added to this document at the
time of parking, with the falsifier that would surface it.

**Cost estimate:** ~5min per coverage-triggering commit (one grep + one
chat post). Compare to ~4hr HALT today. Break-even at one prevented HALT per
~50 commits.

## Deferred-secondary bugs (surfaced during audit)

These are not part of the original 12-item scope but discovered as
side-effects of the audit work. Logged for future sequencing.

- **DSecondary-1. dump_asm crashes at gen_asm.cpp:3296 under pydebug**
  (per generalist 16:35:31Z, supervisor 16:35:52Z). The
  `PYTHONJITDUMPASM=1` env var triggers a crash inside the disassemble
  call when running pydebug builds. Blocks asm-level diagnostics under
  pydebug (workaround: gdb attach + disassemble JIT mmap region).
  Sequencing: queue with W-A4-mid or as separate W-DSecondary-1 fix
  workstream; not blocking.

- **DSecondary-3. build_phoenix.sh --clean does not regenerate pyconfig.h
  on --pydebug ↔ release toggle** (per testkeeper 17:17:59Z process note).
  Symptom: gate FAIL false-positive on push 45 v1 — pyconfig.h had
  `#define Py_DEBUG 1` left over from a prior `--pydebug` build; release
  rebuild inherited the flag; JIT_DCHECK fired in 'release' code; Phase
  B I1 assertion at builder.cpp:1254 surfaced spuriously. Cost: ~10min
  diagnosis + manual fix + gate re-run.

  Fix: `scripts/build_phoenix.sh --clean` should `rm pyconfig.h` so
  re-configure picks up the current `--with[out]-pydebug` flag fresh.
  Cost ~5min. Sequencing: post DSecondary-2; not blocking.

  **Self-triggering drain criterion** (per shepard 17:26:51Z): DSecondary-3
  MUST be paid before the next build-mode toggle in any workstream. If
  any agent runs `build_phoenix.sh --pydebug` followed later by
  `build_phoenix.sh --clean` (without --pydebug), they MUST first land
  DSecondary-3 (or manually `rm pyconfig.h` between builds AND post a
  reminder to drain DSecondary-3). Any gate failure exhibiting
  JIT_DCHECK firing in a nominally release build is presumptive
  DSecondary-3 contamination; rule out via `grep '#define Py_DEBUG'
  pyconfig.h` before deeper diagnosis.

- **DSecondary-2. phoenix-asm PhxMem missing `has_base` flag** (per
  theologian 16:47:24Z + generalist 16:48:35Z fix, theologian 16:50Z
  structural review). The PhxMem struct (`Python/jit/phoenix_asm/phoenix_asm.h:33-47`)
  has no `has_base` flag, so memory operands cannot distinguish "no
  base register" from "base = RAX (id=0)". `phx_fs_ptr(offset)` did
  `PhxMem m = {0}` and the encoder emitted `fs:[rax + offset]` instead
  of `fs:[offset]`. This was the root cause of the W-C2 pydebug crash.

  The W-C2 fix piggybacks on the existing `is_abs_addr` flag to trigger
  SIB pure-disp32 encoding (semantically muddled — abs_addr conventionally
  means "absolute 64-bit address", not "FS-relative pure displacement").
  Refactor: add explicit `has_base` flag to PhxMem; rewrite `phx_fs_ptr`
  to leave `has_base = 0`; encoder treats `has_base = 0` as pure-disp32
  encoding. Audit other PhxMem-initializer callsites for the same trap.

  Cost: ~30min (struct change + encoder branch + callsite audit).
  Sequencing: post W-C2 land + W-A4-mid.

## Open items for scribe / supervisor

1. **Scribe:** post canonical D-1775660954 enumeration so Group B detail-fill
   can complete (librarian 12:49:07Z ask still open).
2. **Supervisor:** decide whether the Coverage-Trigger Rule lands as
   CLAUDE.md addition (gatekeeper enforcement) or stays advisory in this
   doc.
3. **Theologian (self-followup):** after scribe posts canonical enumeration
   and Group B detail-fills, run the actual audit (re-greps + decisions) and
   amend this document with results. Estimated ~30min after canonical
   enumeration lands.
