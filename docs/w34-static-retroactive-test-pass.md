# W34 — `__static__` Retroactive Test Pass Post-Batch-2-G Burndown

**Status:** FILED (deferred per supervisor [chat L2935]). ~30-60min;
post-burndown audit of W27e accepted-residual. Schedule parallel to
24-bench full ABBA (testkeeper queued post-push 101); sequence
**after** the 24-bench run so a perf regression is bisected before
the workload regression class is added on top.

**Owner:** TBD (testkeeper most likely; theologian draft)

---

## 1. Problem statement

W27 closure (a642405a5c, Tier 5 close 2026-04-22) accepted 7 PARTIAL
emit-method stubs as residual: 3 Cat-A (PURE dispatch glue,
structural residual) + 4 Cat-B (data-conditional residue, accepted on
cost-tradeoff grounds per supervisor L2609).

The 4 Cat-B PARTIAL stubs (emitAnyCall, emitLoadAttr,
emitCallExceptionHandler, emitInlineExceptionMatch) are validated by:
- Differential JIT_DCHECK (parallel run, compares C vs C++ outputs)
- Wiring gate (force_compile sole-path) — currently INFRA-BROKEN per
  W32, manual smoke as substitute
- W21 golden trip-wire (codegen output comparison) — partial mainline
  coverage (W27a 0/5, W27b 2/5 per pythia #87 figures)

Per pythia #87 (2026-04-23): "the deferred __static__-only sole-path
coverage (D-1776904264 caveat carried forward from L2436) has no
scheduled retroactive test pass — W25c is queued, but no equivalent
W-series tracks 'run __static__ suite against post-W27 binary'."

The risk surface: a `__static__` Python customer triggering a
register-renumbering or refcount drift in a JIT-compiled body inside
the 4 Cat-B PARTIAL window would surface a regression with
no defense-in-depth between commit and customer.

## 2. Resolution — One-time __static__ test pass

### Scope
Run the `__static__` Python test suite (Lib/test/test_static_*.py
+ cinderx StaticPython tests) against the post-Batch-2-G HEAD binary
(currently cbb945324d / push 101).

### Procedure
1. Build current HEAD with `scripts/build_phoenix.sh --clean` (no
   pydebug — release binary for reference behavior).
2. Run `__static__` test files individually with JIT_ENABLE=1 +
   force_compile or auto-compile:
   ```
   ./python -m test test_static_python -v
   ./python -m test test_static_compiler -v
   ./python -m test test_static_jit -v
   (and any other test_static_* files in Lib/test/)
   ```
3. For each test that exercises emitAnyCall, emitLoadAttr,
   emitCallExceptionHandler, emitInlineExceptionMatch (the 4 Cat-B
   PARTIAL stubs), capture pass/fail + any JIT_CHECK failures or
   register-renumbering anomalies.
4. Compare per-test outcome to vanilla CPython at
   ../cpython-vanilla/python.

### Falsification criterion
- Any divergence (test failure under JIT-on but not under JIT-off, or
  JIT_CHECK fire, or refcount delta differing from vanilla) is a
  Cat-B PARTIAL accepted-residual hypothesis FAILURE.
- ANY divergence triggers W25b/W26-style root-cause investigation
  before continuing burndown.

### Pass criterion
All `__static__` tests pass under JIT-on with no JIT_CHECK fires and
refcount-delta within vanilla noise. Cat-B PARTIAL accepted-residual
empirically validated for current workload coverage.

## 3. Cadence

ONE-TIME audit post-Batch-2-G burndown stabilization. Not a
per-commit gate. Rationale: (a) burndown velocity is high (5+ batches
per session); per-commit __static__ pass is too expensive; (b) the
Cat-B PARTIAL surface is FIXED (4 methods, won't grow); a single
clean pass establishes empirical baseline; (c) any future burndown
that touches one of the 4 Cat-B PARTIAL methods (or PARTIAL stubs in
general) re-triggers the W34 test pass for that method's call paths
specifically.

## 4. When to schedule

Per supervisor [chat L2935]: post-Batch-2-G land + post 24-bench full
ABBA. Sequence:
1. 24-bench full ABBA (testkeeper queued post-push 101) — perf gate
2. W34 __static__ test pass — workload gate
3. Continue Batch 2-H burndown if both PASS

Cross-link:
- Pythia hypothesis: #87 (2026-04-23) — re-issue from L2933
- Theologian batch-class breakdown: [chat L2934]
- Supervisor concur: [chat L2935]
- W27e closure: a642405a5c + supervisor L2609 (Tier 5)
- W27 Cat-A/B classification: feedback memory
  `feedback_dispatch_glue_categorization.md`
- Sibling deferred audit: W25c void*-locals cleanup
