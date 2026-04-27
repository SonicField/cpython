# W32 — `gate_phoenix.sh --wiring` Infra Repair

**Status:** FILED ACTIVE (per supervisor [chat L2935] + librarian
[chat L2939] + shepard [chat L2942] W-numbering convention). Pre-
condition for any sole-path C-port falsification per W31 §3.2.

**Owner:** theologian (draft) → testkeeper (impl + verify).

**Estimated cost:** ~30-60min (regex fix + tolerant-pipefail + verify
end-to-end exit code reaches Step 6).

---

## 1. Problem statement

`scripts/gate_phoenix.sh --wiring` does **not** actually run the Step
6 wiring smoke test in current state. The script exits silently at
Step 3 (Phoenix Tests) via `set -euo pipefail` triggered by a grep-
no-match condition. Step 6 — the force_compile sole-path coverage
that catches C-RPO divergence (R3b bug #11 class) — has never run
end-to-end since the gate was implemented at D-1776714419.

Empirical surface: Batch 2-F (push 100 cd5a1c50a0, 2026-04-23) was
the first attempted invocation post-D-1776714419 (Batches 2-A through
2-E were structural reorgs that didn't need it per librarian L2939).
Both invocations exited cleanly (exit 0) but `docs/gates/<sha>.log`
ended at:
```
--- Step 3: Phoenix Tests ---
Phoenix: 0 tests, 0/0 modules, Result: UNKNOWN
Phoenix FAILURES:
```

No Step 4-6 output. Per testkeeper L2906 self-reflection.

## 2. Root cause

Per memory `feedback_gate_phoenix_wiring_bug.md`:

1. `PHOENIX_OUTPUT=$(... -m test $PHOENIX_MODULES 2>&1 || true)` runs
   Phoenix tests.
2. `PHOENIX_RESULT=$(echo "$PHOENIX_OUTPUT" | grep -oP 'Result: \K\w+'
   || echo "UNKNOWN")` extracts result.
3. If Phoenix output doesn't include the `Result: SUCCESS` literal
   (real output may be `OK` or empty if zero tests match the filter),
   `PHOENIX_RESULT=UNKNOWN`.
4. `if [ "$PHOENIX_RESULT" != "SUCCESS" ]` triggers
   `echo "Phoenix FAILURES:" + echo "$PHOENIX_OUTPUT" | grep -E "FAIL|ERROR|CRASH|Assertion failed"`.
5. With `set -euo pipefail`, the grep with no matches returns exit 1
   → script terminates without `|| true` guard.
6. Script exits via pipefail BEFORE Step 4 (CPython tests) or Step 6
   (wiring smoke) run.

## 3. Resolution options

### Option A — Tolerant grep (`|| true` guard) [RECOMMENDED]

Add `|| true` to the FAILURES grep at step 4:

```bash
echo "$PHOENIX_OUTPUT" | grep -E "FAIL|ERROR|CRASH|Assertion failed" || true
```

Smallest possible change, preserves existing pipeline semantics, fixes
the immediate exit. Doesn't address the stale Result regex but
unblocks Step 4-6.

Cost: ~2min code change + dual-arch verify.

### Option B — Replace regex with exit-code check

Replace `PHOENIX_RESULT` regex extraction with the actual exit code
of the test command:

```bash
PHOENIX_OUTPUT=$(... -m test $PHOENIX_MODULES 2>&1)
PHOENIX_EXIT=$?
if [ $PHOENIX_EXIT -ne 0 ]; then
    PHOENIX_RESULT="FAIL"
else
    PHOENIX_RESULT="SUCCESS"
fi
```

More semantically correct (exit code is the canonical signal). Doesn't
depend on output formatting which has drifted at least once.

Cost: ~10min code change + verify both Phoenix-test-pass and
Phoenix-test-fail paths emit correct PHOENIX_RESULT classification.

### Option C — Both A + B

Defense-in-depth. ~12min total. RECOMMENDED long-term per W26 §4b
layered-gate pattern.

## 4. Verification — Post-fix end-to-end exit

The fix is complete only when:

1. `scripts/gate_phoenix.sh --wiring` runs to completion (Step 6
   reached) on a known-PASS commit (e.g. cbb945324d push 101).
2. The gate fails when Step 6 force_compile fails (induced fault:
   manually corrupt one of the 5 force_compile fns to trigger
   JIT_CHECK).
3. Output log includes Step 6 trailer: `wiring smoke: PASS` or
   `wiring smoke: FAIL <function> <reason>`.

Without (1)+(2)+(3), the gate is not repaired — only mute.

## 5. Manual force_compile substitute (current state)

Until W32 is implemented, sole-path verification uses the manual
4-test force_compile suite per memory
`feedback_gate_phoenix_wiring_bug.md`:

```python
JIT_ENABLE=1 ./python -c "
import _cinderx, cinderjit
def straight_add(x, y): return x + y
def recursive_fib(n): return n if n < 2 else recursive_fib(n-1) + recursive_fib(n-2)
def loop_sum(n):
    s = 0
    for i in range(n): s += i
    return s
def make_function_with_defaults():
    def inner(a=1, b=2, c=3): return a + b + c
    return inner
cinderjit.force_compile(straight_add); assert straight_add(2,3) == 5
cinderjit.force_compile(recursive_fib); assert recursive_fib(10) == 55
cinderjit.force_compile(loop_sum); assert loop_sum(10) == 45
cinderjit.force_compile(make_function_with_defaults)
inner = make_function_with_defaults(); assert inner() == 6
print('wiring smoke: PASS')
"
```

This is acceptable for header-inline batches (safe-by-construction)
per testkeeper L2906 + theologian L2907 + L2934 batch-class
breakdown. Per W31 §3.2 it is **NOT acceptable** for sole-path C-port
falsification — that requires the repaired --wiring infra.

## 6. When to schedule

Per supervisor [chat L2935] + librarian [chat L2939]: file post-Batch-
2-G land. ~30min infra; W31 §3.2 names W32 repair as precondition for
any future GetRPOTraversal(start) C-port (or any other sole-path C-
port post-Phase-3D burndown). Scheduling priority is:

1. Complete current Batch 2-* burndown (header-inline pattern, doesn't
   need --wiring repair).
2. Schedule W32 repair as a docs+infra-only commit when burndown
   stabilizes.
3. Sole-path C-port workstreams (none currently scheduled) blocked on
   W32 PASS per W31 §3.2.

Cross-link:
- Memory: `feedback_gate_phoenix_wiring_bug.md` (librarian, 2026-04-23)
- Empirical incident: Batch 2-F push 100 first invocation
- Pythia hypothesis: #91 #2 (2026-04-23)
- Librarian gate-vintage clarification: [chat L2939] —
  D-1776714419 sole-path gate is distinct from D-1775424410 frame_asm
  gate; the latter has been functioning, the former has never run
  end-to-end
- Supervisor flag: [chat L2900]
- Theologian downstream dependency: W31 §3.2
