# W-PYTORCH-CM-PHOENIX-CRASH Tooling Note for Resumption

Per pythia #153 #2 + shepard 2026-04-26T05:38:02Z directive: tonight's
investigation hit pivot-rate concern (6 hypotheses ruled out via printf-
granularity instrumentation, mechanism still unidentified). The binding
constraint is OBSERVABILITY PRIMITIVE not agent freshness. Fresh-agent
resumption with same printf+lldb-single-shot toolset risks repeating the
6-pivot pattern.

This note documents tooling changes recommended BEFORE D9+ falsifiers.

## SPLIT 2026-04-26T11:37Z — W-PYTORCH-CM is 2 INDEPENDENT BUGS

Valgrind run (testkeeper 11:19:29Z + 11:37:15Z, authorized supervisor
11:07:34Z + 11:14:44Z) discriminated bug class via (c) tooling
(librarian D-1777200177 prior-art):

- **W-PYTORCH-CM-(i) compile-time type-confusion: FIXED (push 63).**
  hir_c_compare_op casts to HirCompare* (HIR_DEOPT_FIELDS, ~108B);
  was called from simplify_primitive_compare_c (simplify_c.c:718) on
  HirPrimitiveCompare (HIR_INSTR_FIELDS, ~52B), reading +56 past
  struct end. Fix: added hir_c_primitive_compare_op accessor casting
  to HirPrimitiveCompare* (generalist 11:32:50Z, theologian structural
  review 11:33:36Z, valgrind discriminator 26→1 errors testkeeper
  11:37:15Z). Class-of-bug audit clean (1 caller bug; no parallel
  type-confusion in DEOPT-vs-INSTR Hir*Compare/Hir*UnaryOp/etc
  accessors).

- **W-PYTORCH-CM-(ii) runtime StoreAttr slot corruption: UNFIXED.**
  D2 still captures LSB transition (0x97→0x96 at obj-0x18) post-(i)-
  fix; D2 confirmed structurally INDEPENDENT from (i). Mechanism
  remains per SoK 4-class enumeration below. Pythia #160 #4
  vindicated: (c) tooling found a real bug AND clarified the
  remaining bug is independent (not downstream of compile-time UAF).

## State of Knowledge (per scribe D-1777180692)

- W-PYTORCH-CM-PHOENIX-CRASH: real Phoenix-introduced bug per cinderx_dev
  oracle PASS + 11d Alex prior-art D-1775658159
- Crash: instance dict slot at obj-0x18 contains unaligned garbage
  (0x...bfd6) at second StoreAttrCache invoke on _NoGrad
- Pre-W27c (HIR identical p53↔p60)
- Falsified hypotheses:
  - C5 TypeAttrCache invalidation race (W-RE-PARSER discriminator showed
    different crash site)
  - C6 tp_alloc pre-header init (D4 always shows LSB=1 post-tp_alloc)
- D8 + T2.5 RESOLVED both hypothesis classes empirically:
  - Hyp B (PyMem recycle): CONFIRMED via T2.5 (testkeeper 06:53Z).
    init_inline_values lifecycle log shows PyMem reuses values addresses
    across consecutive instances at SAME obj address (#197/#198,
    #195/#196 patterns). Allocator pattern is real.
  - Hyp A (slot corruption): CONFIRMED via D2[1] slot value 0x...ffd6
    being INVALID for any PEP 697 state (not 8-aligned, can't be
    IsValues with LSB=0, can't be valid IsDict pointer). Real corruption
    observed.
  - COMBINED: corruption is on real recycled values chunk, not
    coincidence alone.
- MECHANISM NARROWED (per generalist 06:59:17Z encoding correction):
  - Encoding: _PyDictOrValues_SetValues stores `(char*)values - 1`;
    GetValues recovers via `+1`. 8-aligned addrs end in 0/8 → -1 ends
    in 7/F (low 3 bits 0b0111).
  - D2[0] 0x97 IS CORRECT IsValues encoding for values_ptr 0x98 (T2.5
    confirmed 0x98 is heavily-recycled values chunk).
  - D2[1] 0x96 = ONE BIT difference (bit 0 cleared). Pattern cannot
    result from vanilla CPython slot-write (writes 0x97 IsValues, 0x00
    NULL, or 8-aligned dict_ptr ending 0/8).
  - 1-byte write 0x97 → 0x96 (AND-with-~1, OR subtract-1, OR direct
    write of 0x96).
  - Phoenix source-level slot-writer audit (grep Python/cinderx/Jit +
    Python/jit): NO direct writes to obj-0x18. Phoenix source only READS
    via _PyObject_DictOrValuesPointer (SplitMutator setAttr/getAttr)
    using correct macros.
  - **CAVEAT (per pythia #156 #1):** source-grep excludes ONLY
    source-level writes. JIT-EMITTED machine-code writes (Phoenix runtime
    helpers, JIT-emitted prologues, etc.) are NOT testable by source
    grep. JIT-emit candidate writer remains undischarged by the audit.
    rr-record / ptrace watchpoint (T1/T2 below) covers this gap.
  - **D2 + id() correlation RAN 2026-04-26T08:03Z** (testkeeper
    self-authorized per governance D-1777187564). Wrapper-induced
    perturbation: original LSB=0 trigger DID NOT reproduce under Python
    __enter__ wrapper. Bug is TIMING-SENSITIVE — wrapper changes JIT/
    interpreter call timing enough to avoid the trigger.
  - Useful negative result: rules out 'LSB transition is universal
    across all Phoenix StoreAttr invocations' — trigger is a specific
    JIT/interpreter sequence the wrapper perturbs.
  - Address-based identity CONFLATES recycled instances (per D3d).
    Cheap printf cannot disambiguate beyond address.
  - **CHEAP DISCRIMINATOR CLASS EXHAUSTED for this question:**
    instance-identity discrimination requires perturbation-free
    instrumentation (C-side per-instance counter at tp_alloc, OR
    rr-record/replay, OR ptrace watchpoint).
  - **CURRENT HEADLINE STATUS (per supervisor 2026-04-26T09:16:16Z +
    pythia #158 #3; extended to 5-class per supervisor 12:57:00Z +
    librarian C5 prior-art surface 12:38:56Z):** ONE corrupted slot
    snapshot observed (D2[1] = 0x...96); writer UNIDENTIFIED; FIVE
    candidate classes remain open:
    (a) narrow 1-byte writer at obj-0x18 byte-0 (AND-with-~1 / sub-1
        / direct 0x96)
    (b) wider write clipping LSB (e.g. 2/4/8-byte store whose low byte
        happens to be 0x96 due to upstream value)
    (c) wild-write / UAF that happens to land LSB-aligned at obj-0x18
        (writer is unrelated to managed-dict slot semantically)
    (d) D2 was two-instance conflation — D2[0] and D2[1] are different
        instances at recycled address; no single-instance mutation
        ever happened. Address-keyed printf cannot resolve.
    (e) cache-load-side: TypeAttrCache value slot at 0xd42018
        baked into JIT'd code at compile time (testkeeper 12:27:50Z
        lldb confirmed 0xd42018 load + INCREF dance still live at
        StoreAttr JIT site post-(i)-fix). C5 invalidation-race
        (D-1777179278 NEVER CLOSED): cache slot writer races with
        JIT'd LOAD → JIT loads torn/stale value → STORE writes
        corrupted value to obj-0x18. Reframes corruption as LOAD-side
        (cache), not STORE-side (slot).

    **POST-FALSIFICATION STATE (per supervisor 2026-04-26T13:06:46Z
    after testkeeper 13:03Z ASAN + generalist 13:06Z watchpoint):**
    - (c) wild-write/UAF: FALSIFIED via ASAN on push 63 binary —
      crash is NULL+0xAB deref via Py_TYPE(NULL)->tp_flags, not UAF.
      Mechanism: LSB-clear at obj-0x18 flips PEP 697 IsValues
      encoding (stored = values_ptr - 1, ends 0x7) → IsDict
      misinterpretation (LSB=0); 0x96 read as dict_ptr → ob_type at
      0x96+8 = NULL/junk → SEGV. Crash is DOWNSTREAM CONSEQUENCE of
      LSB-clear, not the corruption itself.
    - (e)-at-TYPE-slot 0xd33020: FALSIFIED via hw-watchpoint
      (generalist 13:06Z) — slot written 2x at JIT preload only,
      NEVER written during 50K-iter workload, SEGV still occurs.
      Writer is NOT corrupting cache-TYPE slots.
    - (e)-at-VALUE-slot 0xd42018: FALSIFIED via hw-watchpoint
      (generalist 13:39Z) — slot written 2x at JIT compile (ctor +
      fill to _Py_TrueStruct), NEVER written during 50K-iter workload,
      SEGV still fires.
    - (e)-broader (StoreAttrCache cache_ entries per-cache, inline
      values area) UNTESTED — narrowing-not-closing per librarian scope.

    **GENUINE PAUSE — DISCRIMINATOR-SATURATED 2026-04-26T14:21:09Z
    (per supervisor):** cheap-tier TRULY exhausted across BOTH
    substrate dimensions:
    - Static-slot watchpoint substrate: 3/3 enumerated cache slots
      tested (TYPE 0xd33020, VALUE 0xd42018, StoreAttrCache cache_
      0xd5b2a0) — all stable post-fill, no runtime writes during
      workload. (e) cache-load-side hypothesis CLOSED at the SEGV-frame
      StoreAttrCache::invoke. **Scope-honest correction (per pythia
      #166 #1 + supervisor 14:51:15Z):** 3/3 enumeration was
      per-frame (single JIT-disasm pass at SEGV site); broader
      objdump-grep across compile-unit cache-load immediates is
      RESIDUAL CHEAP-TIER UNRUN. Other JIT-emitted StoreAttr/LoadAttr
      sites compiled during workload not enumerated; no global
      cache-slot manifest. Pause stands per supervisor 14:21:09Z
      no-further-reversals criterion; residual cheap-tier item
      documented for resumption.
    - Per-instance D2+id native (header-byte) substrate: tested via
      ~10 LOC compile-print patch (theologian 14:17:41Z, applied
      testkeeper 14:20:44Z). D2[0] and D2[1] have IDENTICAL refcnt=2
      + same type_ptr + same first8 — header bytes cannot distinguish
      R1 (single-instance mutation) vs R2 (two recycled fresh
      _NoGrad instances at same PyMem chunk; both have refcnt=2 from
      __new__ + with-binding, same class type_ptr). Discrimination
      strategy fundamentally insufficient for fresh-instance-recycle-
      at-same-address case.

    Mechanism narrowed: corruption is per-instance write at obj-0x18,
    NOT cache-side; (a)/(b)/(d) remain open + INDISTINGUISHABLE by
    static-slot + header-byte means.

    Resumption requires HEAVY-TIER per governance D-1777190699:
    - per-instance dynamic watchpoint (instrumented tp_alloc to set
      watchpoint per-instance), OR
    - rr-record/replay deterministic execution, OR
    - allocate-counter implementation per
      docs/w-pytorch-cm-allocate-counter-design.md (~200 LOC + rebuild)
    Authorization gates: (a) Alex direction OR (b) explicit
    heavy-tier authorization. Per supervisor 14:15:58Z commit-criterion:
    NO further cheap-tier reversals — pause stands.
    - (a) narrow 1-byte writer at obj-0x18: STILL OPEN
    - (b) wider write clipping LSB: STILL OPEN
    - (d) two-instance conflation: partially open
    Remaining hypotheses concentrate on per-instance writes to
    obj-0x18 NOT via cache.

    D-1777190733 (D2+id wrapper negative result, trigger evaded) is
    consistent with REMAINING (a)/(b)/(d)/(e-broader). Earlier '1-byte 0x97→0x96 writer'
    headline was inference from 2-snapshot D2, not observation —
    successor agents should NOT scaffold falsifiers from the inferred
    writer; scaffold from the substrate (which class is being
    discriminated, what observation discriminates it).
  - **Roster-coverage gap (per pythia #159 #1, supervisor 2026-04-26T10:31:10Z;
    corrected per librarian 10:42:35Z + supervisor 10:43:12Z;
    extended for class (e) per supervisor 12:57:00Z):**
    Resumption tooling roster covers (a)+(c)+(d)+(e):
    - (a) narrow 1-byte writer: rr/ptrace 1-byte hardware-watchpoint
      (T1/T2 above) — authored, awaiting Alex authorization
    - (c) wild-write / UAF LSB-aligned coincidence:
      ASAN + valgrind --leak-check=full + Py_REF_DEBUG, ALL PROVEN ON
      PHOENIX JIT (D-1776492224 valgrind 416B/25-block baseline,
      D-1776544456 + D-1776545294 ASAN root-caused 2 prior Phoenix UAFs,
      D-1776530198 Py_REF_DEBUG +22 baseline; D-1776535585 caveat:
      GDB/LLDB fail on LTO, ASAN+valgrind work)
    - (d) two-instance conflation: allocate-counter design
      D-1777191857 (caveat #1 + #2 per pythia #157)
    - (e) cache-load-side TypeAttrCache invalidation race:
      hardware-watchpoint on 0xd42018 catches every write to cache
      value slot. Confirms: 0x96 caught at write site → C5 confirmed,
      writer site localized. Falsifies: no 0x96 caught → corruption
      elsewhere. Cheap-tier observation per D-1777190699 (~10min lldb
      attach + run on push 63 binary).
    Only (b) wider-write-clipping-LSB (needs wider watchpoint or
    store-trace) is genuinely un-authored — intentional gap until
    (a)+(c)+(d)+(e) tooling lands or (b)-class evidence emerges.
  - D8 ruled out _PyObject_StoreInstanceAttribute.
  - Writer candidates: (a) indirect Phoenix path via JIT-emitted runtime
    helper, (b) wild write/UAF from unrelated code, (c) unaudited
    helper/macro.
- Resumption gate: rr-record/ptrace watchpoint on obj-24 byte 0 (per
  T1/T2 below) localizes writer.
- D8 + T2.5 reconciliation: 3-cycle interpretation churn at 05:42-06:12
  was instructive — observation-anchored falsifiers (T2.5 init_inline_values
  log) discharged hypothesis classes that pure-reasoning could not. Per
  pythia #154 #1 lesson: instrument before structural reasoning.

## Tooling Limitations Tonight

Three observability primitives used:
1. printf instrumentation (D1/D2/D2-ext/D3d/D4/D8): produces line-at-a-
   time logs; can't capture multi-subsystem state simultaneously
2. lldb single-shot at SEGV (G1/H4-B/H6): captures registers + stack at
   crash; doesn't show evolution of state leading to crash
3. ASAN (F1/F7): flagged misleading allocator-reuse pattern; required
   F5 + R1+G1 to disentangle

What none of them captured: the EVOLUTION of obj-0x18 slot value across
the lifetime of one instance, correlated with which Phoenix code wrote
each value.

## Recommended Tooling Changes for Resumption

### Option T1 — rr (Mozilla record-and-replay)

Record the failing run via `rr record`. Replay via `rr replay`. At SEGV,
use `reverse-continue` + watchpoints on obj-0x18 slot to walk backward
to the actual write that left the slot in the bad state.

Cost: rr install + record overhead (~5x slowdown), but solves the "which
code wrote the bad value" question that printf can't.

### Option T2 — ptrace watchpoint via gdb

Set hardware watchpoint via gdb `watch *0x...bfd7` on the obj-0x18 slot
once an instance is allocated. Watchpoint fires on write. Requires
either:
- Pause execution at known-good point (e.g., __new__ return) to set
  watchpoint, then continue
- Per-instance addresses change with allocator recycling, so watchpoint
  needs to be re-set per instance OR set on one specific recycled
  address that the bug pattern hits repeatedly

**Watchpoint scope spec (per pythia #154 #2):** byte-level watchpoint
on obj-0x18 LSB byte, NOT word watchpoint (would fire on every legitimate
slot write including IsValues init). Filter pattern: tp_name == _NoGrad
+ writer is NOT _PyObject_StoreInstanceAttribute (D8 already ruled
that out).

Cost: lower than rr but requires manual gdb scripting to track instance
lifecycle.

### Option T2.5 — new_values allocator instrumentation (alongside T2)

Per pythia #154 #3: hyp-B (allocator coincidence) reversal at
2026-04-26T05:44:20Z rested on REASONING not observation. New_values
allocator recycle pattern is unmeasured. Cheap (~5min) parallel addition:
instrument `new_values()` in Objects/dictobject.c to log address +
allocation index for _NoGrad-class allocations. Catches Hyp B
falsification empirically.

This is printf-class so should be bundled WITH T1/T2 tooling change,
not run standalone (per shepard 2026-04-26T06:09:20Z 'no mechanism work
without tooling change').

### Option T3 — Phoenix-side dict_watcher trace

Phoenix already has dict_watcher infrastructure (per Python/jit/
global_cache.cpp). Add a watcher hook that logs every write to a managed-
dict slot of a heap-class instance. Filter for _NoGrad / pytorch_cm
context.

Cost: Phoenix-side instrumentation; integrates with existing
dict_watcher; provides Phoenix-code-anchored evidence.

### Option T4 — Run D8 first

Before any tooling change: run the deferred D8 (instrument
_PyObject_StoreInstanceAttribute, ~5min). If it shows the LSB transition
is allocator coincidence (different instances at addresses sharing low
bits modulo LSB), the entire mechanism question may dissolve and tooling
investment is unnecessary.

D8 may have been started by generalist at 2026-04-26T05:34:27Z (per
chat); resumption agent should check status first.

## Resumption Procedure

1. Query `nbs-scribe-query D-1777180692` for state-of-knowledge brief
2. Check D8 status (may be in flight or completed)
3. If D8 inconclusive:
   - Choose tooling option T1/T2/T3 based on availability
   - Run with refined falsifier (NOT printf-only)
4. Apply state-model checkpoint discipline per pythia #152: state model,
   what falsifies, what confirms BEFORE running falsifier

## Discipline Notes

Per pythia #150 + #151 + #152 + #153 lessons codified in
W-PROTOCOL-CODIFY:
- P10 variance characterization: not directly applicable here
- Falsifier-economics (informal): cheapest+highest-info first; D8 was
  identified as cheapest at 04:14Z but deferred 9 times in the
  hypothesis sequence. Future investigations: if a falsifier is in the
  cheap+high-info quadrant, run it first.
- Pivot-rate as model-failure-rate signal: 6 pivots in 14min indicates
  multi-subsystem mechanism; acceptable cost if peer-correction holds
  AND each pivot rules out cleanly. Tonight's 6 pivots all met that bar
  but cumulative time was high.

## Anti-Pattern Warning

Per pythia #154 #4: path of least resistance after multi-pivot
investigation is appending _NoGrad / _Autocast / context-manager types
to Phoenix's existing `shouldSkipCompilation` skip-list (pyjit.cpp).
That is a WORKAROUND not a fix — it preserves the underlying bug
class for future managed-dict types to re-trigger. Resumption agent
should NOT propose skip-list extension as the fix; root-cause the
LSB-clear writer.

## Out of Scope

- Codifying tooling-change requirement as a new W-PROTOCOL-CODIFY
  protocol (per supervisor 2026-04-26T03:45:18Z extended discipline
  pause: no new methodology spec until current ones independently-tested).
  This note documents per-workstream tooling, not a new rule.
