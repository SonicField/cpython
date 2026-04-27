# W-PYTORCH-CM-(ii) — StoreAttr managed-dict tag-flip corruption

**Status:** PARKED behind pure-C JIT roadmap completion (Alex
2026-04-27T07:12:25Z, supervisor cascade 07:13:18Z; D-1777270945).
Failing-test sentinel:
``Lib/test/test_phoenix_jit_storeattr_managed_dict_tag_flip.py``
(``@unittest.expectedFailure``).

**Workstream history:** D-1777180692 state-of-knowledge brief +
``docs/w-pytorch-cm-tooling-note.md`` running investigation log.
W-PYTORCH-CM was split 2026-04-26T11:37Z into (i) compile-time
type-confusion (FIXED at push 63 by adding
``hir_c_primitive_compare_op`` accessor) and (ii) the runtime
StoreAttr corruption documented here. (ii) is structurally
INDEPENDENT of (i) per testkeeper valgrind discriminator
2026-04-26T11:37:15Z (D2 LSB transition still captured post-(i)
fix).

## Symptom

```
$ ./python /tmp/repro_s3.py
... (50,000-iter bench_pytorch_cm post-force_compile) ...
Segmentation fault (core dumped)
```

Crash is a NULL+0xAB deref inside ``PyDict_SetItem`` reaching
``Py_TYPE(NULL)->tp_flags`` (offset 0xAB into ``PyTypeObject``).
Confirmed via ASAN on push 63 (testkeeper 2026-04-26T13:03Z): the
SEGV is a **downstream consequence** of an LSB-clear at ``obj +
0x18`` — not a wild write or UAF.

## Mechanism (narrowed; writer un-localized)

PEP 697 managed-dict encoding stores ``(char*)values_ptr - 1`` in the
slot at ``obj + 0x18``. 8-aligned addresses end in ``0x0`` / ``0x8``,
so the encoded form ends in ``0x7`` / ``0xF`` (low 3 bits ``0b111``)
when IsValues is set. ``IsDict`` is signalled by LSB == 0.

Sequence observed in the repro:

1. ``D2[0]`` snapshot: slot byte 0 = ``0x97`` (correct IsValues
   encoding for ``values_ptr = 0x98``; T2.5 confirmed ``0x98`` is the
   heavily-recycled values chunk).
2. ``D2[1]`` snapshot at the same ``obj`` address: slot byte 0 =
   ``0x96`` — exactly one bit cleared (LSB).
3. The IsDict path reads the now-LSB-zero word as a ``PyDictObject*``;
   ``ob_type`` at offset 8 of ``0x96`` is NULL/junk.
4. ``PyDict_SetItem`` is called with that NULL dict and SEGVs at
   ``Py_TYPE(NULL)->tp_flags``.

**Class-invariant pattern:** byte 0 of ``obj + 0x18`` for ``_NoGrad``
instances allocated by ``Tools/benchmark_phoenix.py:bench_pytorch_cm``
gets its low bit cleared. Pattern cannot result from any vanilla
CPython slot write (writes ``0x97`` IsValues, ``0x00`` NULL, or an
8-aligned dict pointer ending ``0x0`` / ``0x8``).

**Source-level audit (Phoenix Python/cinderx + Python/jit):** NO
direct writes to ``obj + 0x18``. The Phoenix source only READS via
``_PyObject_DictOrValuesPointer`` (e.g. ``SplitMutator::setAttr`` /
``getAttr``) using the correct macros.

**JIT-emit caveat (pythia #156 #1):** the source-grep audit covers
source-level writes only. JIT-emitted machine-code writes (Phoenix
runtime helpers, JIT-emitted prologues) are NOT testable by source
grep. The writer for the LSB-clear remains undischarged by the
audit.

## Hypothesis classes after cheap-tier exhaustion

(2026-04-26T14:21:09Z — discriminator-saturated, GENUINE PAUSE called
by supervisor). Five candidates were enumerated; three are FALSIFIED;
two and a half remain OPEN.

| Class | Description | Status |
|-------|-------------|--------|
| (a)   | Narrow 1-byte writer at ``obj+0x18`` byte 0 (AND-with-~1, sub-1, or direct ``0x96`` store) | OPEN |
| (b)   | Wider write clipping LSB (2/4/8-byte store whose low byte happens to be ``0x96``) | OPEN |
| (c)   | Wild write / UAF coincidentally LSB-aligned at ``obj+0x18`` | FALSIFIED (ASAN on push 63: crash is NULL+0xAB deref, not UAF; LSB-clear is the cause not the corruption itself) |
| (d)   | Two-instance conflation — ``D2[0]`` and ``D2[1]`` are different recycled instances at the same address; no single-instance mutation occurred | OPEN (cannot be discriminated from header bytes — refcnt + type_ptr + first8 identical for fresh ``_NoGrad`` instances; testkeeper 2026-04-26T14:20:44Z) |
| (e)   | Cache-load-side: ``TypeAttrCache`` value slot baked into JIT'd code at compile, racing with cache-slot writer → JIT loads torn value → STORE writes corrupted value to ``obj+0x18`` | FALSIFIED at the per-frame SEGV-site enumeration (3/3 cache slots tested by hardware-watchpoint: TYPE 0xd33020, VALUE 0xd42018, ``cache_`` 0xd5b2a0 all stable post-fill, no runtime writes during workload). RESIDUAL CHEAP-TIER UNRUN: broader objdump-grep across compile-unit cache-load immediates not enumerated. |

## Trigger sensitivity

Bug is **TIMING-SENSITIVE** (D-1777190733, testkeeper 2026-04-26T08:03Z):
the original LSB=0 trigger DID NOT reproduce when the workload was
wrapped in a Python ``__enter__`` context manager. Wrapper added
~100µs/iter of Python interpreter overhead, shifting JIT-call-counter
timing relative to the auto-compile threshold and thereby evading the
trigger window.

Implication for instrumentation: any printf-class observer that adds
Python-level overhead may also evade. Heavy-tier discriminators
(C-side allocate-counter, hardware watchpoint via ``tp_alloc`` hook)
are the next observability tier.

## Reproducer

``/tmp/repro_s3.py`` (228 bytes, preserved verbatim in
``Lib/test/test_phoenix_jit_storeattr_managed_dict_tag_flip.py`` as
the test's HARNESS_SOURCE):

```python
import sys; sys.path.insert(0, 'Tools')
import _cinderx, cinderjit
from benchmark_phoenix import bench_pytorch_cm
bench_pytorch_cm(5000)  # warmup
cinderjit.force_compile(bench_pytorch_cm)
bench_pytorch_cm(50000)
print("S3 OK")
```

``bench_pytorch_cm`` is a self-contained
``Tools/benchmark_phoenix.py`` benchmark exercising nested context
managers (``_NoGrad`` / ``_Autocast`` / ``_ProfileScope``) — the
PyTorch-style pattern that prompted the workstream name. No
``torch`` runtime dependency.

## Heavy-tier instrumentation designs (on disk, un-implemented)

Both ~200 LOC + rebuild; gated on heavy-tier authorization (Alex
direction OR explicit team auth) per governance D-1777190699. Both
documented by theologian under the 2026-04-26 stand-down and ready
for resumption.

### tp_alloc hardware watchpoint
``docs/w-pytorch-cm-tp-alloc-watchpoint-design.md``

Hook ``_NoGrad`` ``tp_alloc``; on each allocation set a 1-byte
hardware watchpoint (DR0-DR3) on ``obj + 0x18`` with write-only
trigger; SIGTRAP handler captures ``RIP`` + backtrace + register
dump. Discriminates (a) narrow 1-byte writer vs (b) wider clipping
write directly from the faulting instruction. (d) two-instance
conflation manifests as "watchpoint never fires on watched instance
even though ``D2`` captures the LSB transition on a different
recycled instance".

### Allocate-counter side-table
``docs/w-pytorch-cm-allocate-counter-design.md``

Add a 64-bit monotonic ``alloc_id`` per ``_NoGrad`` instance via a
hash-table side-table (keyed on ``obj`` pointer; populated at
``init_inline_values``, looked up at the ``D2`` print site).
Discriminates (d) instance conflation from (a)/(b) single-instance
mutation by comparing ``D2[0].alloc_id`` to ``D2[1].alloc_id`` at
the same ``obj`` address.

**Recommended ordering** (per
``w-pytorch-cm-tp-alloc-watchpoint-design.md`` §"Comparison"): if
only one design is authorized, run ``tp_alloc`` watchpoint first —
it directly identifies the writer when (a) or (b) holds. If the
watchpoint never fires on the watched instance during a confirmed
``D2`` transition, (d) becomes the load-bearing hypothesis and the
allocate-counter design is then run.

## Why parked (Alex 2026-04-27T07:12:25Z)

Bug only fires under the contrived ``repro_s3.py`` 50,000-iter
workload after explicit ``force_compile``. Not seen in:

- The CinderX prod codebase (``cinderx_dev`` oracle PASS;
  D-1775658159 11-day Alex prior-art).
- The regular Phoenix test suite (480-test x86_64 + 483-test ARM64
  runs).
- The 24-benchmark ABBA + per-commit 4-benchmark gate.

The fix-class falsifier (cinderx_dev oracle) shows core Cinder is
structurally immune to this bug — Phoenix introduced it. Per
``feedback_assume_phoenix_regression.md`` the bug is presumed
Phoenix-introduced and warrants a real fix, but Alex's 07:12:25Z
direction sequences it after the pure-C JIT roadmap is complete.

## Resumption gate

Before re-engaging the writer hunt:

1. Re-confirm ``Lib/test/test_phoenix_jit_storeattr_managed_dict_tag_flip.py``
   still ``expectedFailure``s on the current HEAD (subprocess SEGV
   reproducible).
2. Read ``docs/w-pytorch-cm-tooling-note.md`` for the full
   investigation log including 6 falsified hypotheses, the 3-cycle
   D8 + T2.5 reconciliation, and the
   ``shouldSkipCompilation``-skip-list anti-pattern warning (pythia
   #154 #4).
3. Choose ``tp_alloc``-watchpoint, allocate-counter, or both per
   the comparison table in
   ``w-pytorch-cm-tp-alloc-watchpoint-design.md``.
4. Heavy-tier authorization required per governance D-1777190699 +
   D-1777270945 (Alex parking decision; resumption is the trigger to
   re-engage).

## Anti-pattern (do not adopt)

Per pythia #154 #4 + ``feedback_no_workarounds.md``: the path of
least resistance after a multi-pivot investigation is appending
``_NoGrad`` / ``_Autocast`` / context-manager types to Phoenix's
``shouldSkipCompilation`` skip-list (``pyjit.cpp``). That is a
WORKAROUND — it preserves the underlying bug class for future
managed-dict types to re-trigger. Resumption agent must root-cause
the LSB-clear writer; do NOT extend the skip-list.
