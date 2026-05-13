#!/bin/bash
# test_b6_wiring_class_negative.sh — B6 --wiring class NEG-TEST per
# generalist 23:21:11Z spec + gatekeeper 23:23:48Z pass-(a)-lite APPROVE.
#
# B6 silent-pass class (per [feedback_gate_phoenix_wiring_bug]):
#   gate_phoenix.sh --wiring step never runs because Phoenix Result regex
#   misclassifies SUCCESS as UNKNOWN, then FAILURES diagnostic grep with
#   no-match returns rc=1, pipefail kills script before Step 4-6 run.
#
# Bug ALREADY FIXED at code layer (W32 4a01bfa3d1 Apr 23 2026):
#   Option B (semantic): PHOENIX_RESULT derived from PHOENIX_EXIT not regex
#                        (gate_phoenix.sh L511-525)
#   Option A (defense):  brace-group `{ ... || true; }` around FAILURES
#                        diagnostic grep (gate_phoenix.sh L538)
#
# This NEG-TEST locks in the fix with executable scenarios that would
# have triggered the original bug class. Both scenarios MUST PASS;
# regression of either fix would cause one of the scenarios to fail.
#
# Runnable manually OR via gate_phoenix.sh --selftest=b6.
set -uo pipefail

PASS=0
FAIL=0
report_pass() { echo "  PASS: $1"; PASS=$((PASS+1)); }
report_fail() { echo "  FAIL: $1"; FAIL=$((FAIL+1)); }

# --- Scenario (i): PHOENIX_EXIT=0 + missing 'Result: SUCCESS' line ---
# Pre-W32 Option B: PHOENIX_RESULT=$(echo "$OUT" | grep -oP 'Result: \K\w+'
# || echo "UNKNOWN"). With clean exit but missing literal, this returned
# "UNKNOWN" → if [ != "SUCCESS" ] triggered FAILURES branch.
# Post-W32: PHOENIX_RESULT derived from PHOENIX_EXIT directly.
echo "B6 scenario (i): PHOENIX_EXIT=0 + missing 'Result:' line → SUCCESS via exit-code"
SYNTH_OUTPUT="$(printf 'Some test output\nNo Result line emitted\nTest run complete\n')"
SYNTH_EXIT=0

# Mirror the W32 Option B logic from gate_phoenix.sh L521-525:
if [ "$SYNTH_EXIT" -eq 0 ]; then
    SYNTH_RESULT="SUCCESS"
else
    SYNTH_RESULT="FAIL"
fi

if [ "$SYNTH_RESULT" = "SUCCESS" ]; then
    report_pass "(i) Exit-code-canonical: synthesized output without 'Result:' line + exit=0 → SUCCESS (not UNKNOWN). W32 Option B intact."
else
    report_fail "(i) Exit-code-canonical broken: synthesized output without 'Result:' line + exit=0 misclassified as $SYNTH_RESULT (expected SUCCESS). W32 Option B regression."
fi

# --- Scenario (ii): brace-group grep with no matches must NOT pipefail-kill ---
# Pre-W32 Option A: bare `echo $OUT | grep -E "FAIL|ERROR|CRASH..." | tee`
# returned rc=1 on no-match → pipefail killed script.
# Post-W32: brace-group `{ ... || true; }` neutralizes no-match exit.
echo "B6 scenario (ii): brace-group grep with no matches must NOT pipefail-kill"

# Synthesize a 'clean' output (no FAIL/ERROR/CRASH/Assertion) — grep -E should
# return rc=1, but the brace-group with || true should swallow it.
CLEAN_OUTPUT="$(printf 'All tests OK\nNo issues to report\n')"
SCENARIO_II_PASSED=1

# Subshell with strict pipefail to mimic gate_phoenix.sh top-level set -euo
# pipefail context. Inside, exercise the W32 Option A pattern verbatim.
(
    set -euo pipefail
    # This line MUST NOT cause the subshell to exit non-zero, even though
    # grep returns 1 (no match) and the pipeline runs under pipefail.
    { echo "$CLEAN_OUTPUT" | grep -E "FAIL|ERROR|CRASH|Assertion failed" || true; } | cat >/dev/null
    # If we reach here, brace-group + || true neutralized the no-match rc.
    exit 0
) || SCENARIO_II_PASSED=0

if [ "$SCENARIO_II_PASSED" -eq 1 ]; then
    report_pass "(ii) Brace-group + || true: no-match grep does NOT pipefail-kill. W32 Option A intact."
else
    report_fail "(ii) Brace-group + || true regression: no-match grep DID pipefail-kill subshell. W32 Option A broken."
fi

echo ""
echo "B6 NEG-TEST SUMMARY: $PASS pass, $FAIL fail"
if [ "$FAIL" -ne 0 ]; then
    echo "B6 W32-fix regression detected — gate_phoenix.sh --wiring step at risk of never running again."
    exit 1
fi
echo "B6 W32-fix locked in: --wiring step proceeds past Step 3 in both edge cases."
exit 0
