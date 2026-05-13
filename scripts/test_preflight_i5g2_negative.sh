#!/bin/bash
# test_preflight_i5g2_negative.sh — I5γ γ-2 ENV-READINESS NEG-TEST per
# theologian 22:14:12Z + gatekeeper 22:15:07Z spec under supervisor
# 22:17:44Z RESEQUENCE.
#
# Procedure:
#   1. Aggressive PATH restriction: PATH=/tmp only (per gatekeeper
#      22:15:07Z (d) — more aggressive than /tmp:/usr/bin which keeps
#      /usr/bin tools accessible). Removes ALL standard tool paths.
#   2. Invoke preflight_check_env_readiness in subshell; assert exit
#      non-zero AND output contains 'γ-2 ENV-READINESS' canonical message.
#   3. Restore PATH to caller's original.
#   4. POSITIVE-CONTROL: invoke with caller's normal PATH; assert
#      preflight returns 0 (no false-positive when tools are present).
#
# Runnable manually OR via gate_phoenix.sh --selftest=i5g2.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/lib_preflight.sh"

ORIG_PATH="$PATH"

PASS=0
FAIL=0
report_pass() { echo "  PASS: $1"; PASS=$((PASS+1)); }
report_fail() { echo "  FAIL: $1"; FAIL=$((FAIL+1)); }

# --- NEG: aggressive PATH restriction → preflight MUST fail ---
echo "γ-2 NEG: aggressive PATH=/tmp restriction (no cmake/clang/python3)"
NEG_OUT=$( ( PATH=/tmp preflight_check_env_readiness ) 2>&1 )
NEG_RC=$?

if [ "$NEG_RC" -eq 0 ]; then
    report_fail "NEG: preflight returned 0 with PATH=/tmp (should FAIL on missing cmake/clang)"
elif ! echo "$NEG_OUT" | grep -q "γ-2 ENV-READINESS"; then
    report_fail "NEG: preflight rejected but did not emit canonical 'γ-2 ENV-READINESS' message"
    echo "    output: $NEG_OUT"
elif ! echo "$NEG_OUT" | grep -q "missing:"; then
    report_fail "NEG: missing-tool list not surfaced in diagnostic"
    echo "    output: $NEG_OUT"
elif ! echo "$NEG_OUT" | grep -q "install hint:"; then
    report_fail "NEG: install hint not surfaced in diagnostic"
    echo "    output: $NEG_OUT"
else
    report_pass "NEG: PATH-restricted env → FAIL with canonical message + missing list + install hint"
fi

# --- POS-CONTROL: caller's normal PATH → preflight MUST succeed ---
echo "γ-2 POS-CONTROL: caller's normal PATH (tools present)"
POS_OUT=$( ( PATH="$ORIG_PATH" preflight_check_env_readiness ) 2>&1 )
POS_RC=$?

if [ "$POS_RC" -ne 0 ]; then
    report_fail "POS-CONTROL: preflight FAILED with caller's normal PATH (false-positive)"
    echo "    output: $POS_OUT"
else
    report_pass "POS-CONTROL: caller's normal PATH → preflight returns 0 (no false-positive)"
fi

echo ""
echo "I5γ γ-2 NEG-TEST SUMMARY: $PASS pass, $FAIL fail"
if [ "$FAIL" -ne 0 ]; then
    exit 1
fi
exit 0
