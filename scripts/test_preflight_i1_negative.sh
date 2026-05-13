#!/bin/bash
# test_preflight_i1_negative.sh — I1 BINARY IDENTITY negative-test per
# theologian 20:30:40Z (I1-3) + gatekeeper 20:31:18Z spec under supervisor
# 20:18:04Z RESEQUENCE (D-1778705500).
#
# Procedure:
#   1. Copy $CPYTHON_ROOT/python to a temp gate binary
#   2. Capture md5+mtime baseline via preflight_capture_binary
#   3. Deliberately mutate the temp binary (append bytes — perturbs
#      both md5 and mtime)
#   4. Invoke preflight_check_binary in subshell; assert exit non-zero
#      AND stderr/stdout contains 'I1 BINARY IDENTITY drift'
#
# Runnable manually OR via gate_phoenix.sh --selftest=i1.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CPYTHON_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
. "$SCRIPT_DIR/lib_preflight.sh"

if [ ! -x "$CPYTHON_ROOT/python" ]; then
    echo "I1 NEG-TEST FAIL — \$CPYTHON_ROOT/python missing or not executable; build first"
    exit 1
fi

TMP_GATE="$CPYTHON_ROOT/python_i1_neg_$$"
trap 'rm -f "$TMP_GATE"' EXIT

cp "$CPYTHON_ROOT/python" "$TMP_GATE"
PYTHON="$TMP_GATE"
RESULTS_FILE=""  # silence tee in lib (unset would still expand, empty = lib uses plain echo)

preflight_capture_binary

# Mutate temp gate binary: append bytes (perturbs md5; sleep 1 so mtime
# also changes — covers both halves of the I1 (md5 OR mtime) check).
echo "I1_NEG_POISON" >> "$TMP_GATE"
sleep 1
touch "$TMP_GATE"

# Run check in subshell so its exit 1 doesn't kill this script.
CHECK_OUT=$( ( preflight_check_binary "i1-neg-test" ) 2>&1 )
CHECK_RC=$?

if [ "$CHECK_RC" -eq 0 ]; then
    echo "I1 NEG-TEST FAIL — drift not detected (preflight_check_binary returned 0 after corruption)"
    echo "  output: $CHECK_OUT"
    exit 1
fi

if ! echo "$CHECK_OUT" | grep -q "I1 BINARY IDENTITY drift"; then
    echo "I1 NEG-TEST FAIL — preflight rejected drift but did not emit canonical 'I1 BINARY IDENTITY drift' message"
    echo "  output: $CHECK_OUT"
    exit 1
fi

echo "I1 NEG-TEST PASS — drift detected; canonical message emitted"
echo "  rc=$CHECK_RC"
echo "  message: $(echo "$CHECK_OUT" | grep 'I1 BINARY IDENTITY drift' | head -1)"
exit 0
