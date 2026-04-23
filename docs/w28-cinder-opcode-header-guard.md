# W28 — `cinder_opcode.h` / `opcode.h` Header Guard Collision

**Status:** FILED (deferred per supervisor [chat L2626] + L2742). Address when
current burndown stabilizes.

**Scope:** Resolve the structural fragility where `cinder_opcode.h` and
`Include/opcode.h` share the `Py_OPCODE_H` header guard, causing one to
silently shadow the other depending on include order. Empirically caught
in W26 push 84 (95c9f9b891 → 1553c14ae8 PhxCallKind dispatch fix per
theologian [chat L2488] hypothesis (A) + W21 golden trip-wire).

**Owner:** TBD (theologian draft + generalist implementation when
scheduled).

**Estimated cost:** ~30-60min (rename guard + verify all transitive
includers + dual-arch test).

---

## 1. Problem statement

`cinder_opcode_ids.h` defines `Py_OPCODE_H` as its own header guard.
`Include/opcode.h` also uses `Py_OPCODE_H`. When both headers are pulled
into the same translation unit, the second include is silently SKIPPED —
whichever loaded first wins, the other's `#define INVOKE_*` /
`#define BINARY_OP_ADD_INT` etc. are missing.

Empirical proof from W26 push 84:
- W26 Step B added `cinder_opcode.h` BEFORE `opcode.h` in
  `builder_emit_c.c` (to access INVOKE_FUNCTION constants for the
  `emitAnyCall` switch).
- `Include/opcode.h` was silently skipped because `cinder_opcode_ids.h`
  set `Py_OPCODE_H` first.
- `BINARY_OP_ADD_INT` became undefined.
- The `#ifdef BINARY_OP_ADD_INT` gate at `builder_emit_c.c:1135` silently
  compiled out the `BINARY_OP` specialization path.
- `attr_probe`'s `a + b + c` lost 4 GuardType + 2 LongBinaryOp
  specializations.
- W21 golden test caught the regression at the first build attempt.
- Fix at 1553c14ae8: rewire emitAnyCall to use a `PhxCallKind` enum
  dispatch (defined in cinderx headers, no opcode.h dependency in C body),
  reverting `cinder_opcode.h` from `builder_emit_c.c`.

**The 1553c14ae8 fix RESOLVES the symptom for emitAnyCall but leaves the
underlying header-guard collision in place.** Any future change in any
TU that pulls in `cinder_opcode.h` before `Include/opcode.h` would
silently re-trigger the same class of regression.

## 2. Resolution options

### Option A — Rename `cinder_opcode_ids.h` guard

Change `cinder_opcode_ids.h` from `#define Py_OPCODE_H` to e.g.
`#define CINDERX_OPCODE_IDS_H`. This breaks the share with
`Include/opcode.h`; both can now be included in either order.

Cost: ~5min code change + ~15min find-and-update transitive includers
+ dual-arch test.

Risk: low. Renaming a header guard is semantically a no-op for content;
only the duplicate-include suppression differs. If anything in the
codebase was incorrectly relying on `cinder_opcode_ids.h`'s guard to
suppress `opcode.h` (intentional or not), that callers compile would
break — survey via `grep Py_OPCODE_H` before commit.

### Option B — Static assertion in `cinder_opcode.h`

Add at top of `cinder_opcode.h`:
```c
#ifdef Py_OPCODE_H
# error "cinder_opcode.h must be included BEFORE Include/opcode.h (or guard renamed per W28)"
#endif
```

Detects the collision at compile time but doesn't FIX it — only flags it
loudly. Forces every consumer to think about include order.

Cost: ~5min. Lower risk than Option A (no rename), but doesn't eliminate
the fragility — just hardens detection.

### Option C — Header-comment + chat-ephemera (current state)

Document in `cinder_opcode.h` header comment that include order matters.
This is the CURRENT state per W26 incident; pythia #87 #3 explicitly
flagged it as insufficient.

NOT RECOMMENDED — the chat-ephemera-only state is what caused the W26
regression to slip through code review (no in-code marker).

### Recommendation

**Option A (rename).** Lower carrying cost than B, eliminates the class
entirely. Run `grep -rn Py_OPCODE_H Include/ Python/jit/` first to verify
no surprise dependencies, then rename + recompile.

Survey scope: ~10 minutes to grep + classify. Implementation: ~5 minutes.
Test: dual-arch build per W26 protocol.

---

## 3. Related fragility — W29 candidate

Per pythia #89 #3 (re-issue, 2026-04-23): `PHX_PRIM_OP_*` /
`PHX_PRIM_UOP_*` values are hard-coded in
`Python/jit/hir/builder_emit_c.c` lines 3727-3746 with NO `static_assert`
binding to the authoritative `#define`s in
`Python/jit_stubs/classloader.h`.

If upstream renumbering changes `PRIM_OP_ADD_INT` etc., the hard-coded
values in `builder_emit_c.c` silently emit wrong opcodes. The W27a/W27b
ZERO-new-bridge falsification SKIP standing rule + W27e accept-residual
both implicitly assume this duplicated-constant table stays synchronized
forever.

W29 candidate scope: add `static_assert(PHX_PRIM_OP_ADD_INT ==
PRIM_OP_ADD_INT, "out of sync with classloader.h")` for each constant.
Same pattern as `hir.cpp:18-23` static_asserts for HirBasicBlock layout.

Cost: ~10min (add ~20 static_asserts in builder_emit_c.c next to the
hard-coded constants).

W29 to be filed as separate workstream when scheduled.

---

## 4. When to schedule

Per supervisor [chat L2626 + L2742]: defer until current burndown
stabilizes. W28 + W29 are real fragilities but not urgent — both surface
only on (a) future include-order changes (W28) or (b) upstream
classloader.h renumbering (W29). Neither is currently breaking.

Schedule: post-Batch-2-{B,C,...} burndown, before any major upstream
sync that could touch `Include/opcode.h` or `classloader.h`.

Cross-link: 
- Empirical incident: W26 push 84 95c9f9b891 → 1553c14ae8 fix
- Theologian hypothesis: [chat L2488]
- Pythia checkpoint: #87 #3 (2026-04-23) + #89 #3 (re-issue)
- Supervisor deferral: [chat L2626] + L2742
