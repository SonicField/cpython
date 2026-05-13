#!/bin/bash
# test_preflight_i4_negative.sh — I4 BUILD-PRODUCT FRESHNESS negative-test
# per theologian 20:30:40Z (I4-3) + gatekeeper 20:31:18Z spec under
# supervisor 20:18:04Z RESEQUENCE (D-1778705500).
#
# Procedure:
#   1. Pre-flight baseline check: assert no tracked .c/.h is already
#      newer than $CPYTHON_ROOT/python (else SKIP — rebuild required)
#   2. Touch a tracked .c file to make it newer than the binary
#   3. Invoke preflight_check_freshness in subshell; assert exit non-zero
#      AND output contains 'I4 BUILD-PRODUCT FRESHNESS'
#   4. Cleanup: git checkout HEAD -- <file> per gatekeeper 20:31:18Z (c)
#      ensures gate state idempotent (file content + mtime restored to
#      git-pristine state)
#
# Runnable manually OR via gate_phoenix.sh --selftest=i4.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CPYTHON_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
. "$SCRIPT_DIR/lib_preflight.sh"

if [ ! -x "$CPYTHON_ROOT/python" ]; then
    echo "I4 NEG-TEST FAIL — \$CPYTHON_ROOT/python missing or not executable; build first"
    exit 1
fi

PYTHON="$CPYTHON_ROOT/python"
RESULTS_FILE=""

# Pick a small, stable, tracked .c/.cpp target. lir_block_builder_c.cpp
# is canonical (c22b-mech site) and unlikely to be missing.
TARGET="$CPYTHON_ROOT/Python/jit/lir/lir_block_builder_c.cpp"
if [ ! -f "$TARGET" ]; then
    echo "I4 NEG-TEST FAIL — neg-test target $TARGET missing"
    exit 1
fi

# Cleanup trap MUST come before any mutation. Always restore via git
# checkout (idempotent: no-op if file is already pristine).
trap 'git -C "$CPYTHON_ROOT" checkout HEAD -- "$TARGET" >/dev/null 2>&1 || true' EXIT

# Baseline freshness probe: if the build is ALREADY stale, the neg-test
# can't distinguish its mutation from baseline staleness. SKIP rather
# than false-PASS.
BASELINE_OUT=$( ( preflight_check_freshness ) 2>&1 ) || BASELINE_RC=$?
BASELINE_RC="${BASELINE_RC:-0}"
if [ "$BASELINE_RC" -ne 0 ]; then
    echo "I4 NEG-TEST SKIP — baseline already stale (preflight_check_freshness fails before mutation):"
    echo "$BASELINE_OUT" | head -10
    echo "  Rebuild via scripts/build_phoenix.sh first, then retry."
    exit 2
fi

# Mutate target: touch to make mtime > $PYTHON mtime. sleep 1 ensures
# filesystem mtime resolution doesn't collapse to ==.
sleep 1
touch "$TARGET"

# Run check in subshell so its exit 1 doesn't kill this script.
CHECK_OUT=$( ( preflight_check_freshness ) 2>&1 )
CHECK_RC=$?

# Restore via git checkout (per gatekeeper (c)). EXIT trap also runs
# this — explicit invocation here ensures restoration even if subsequent
# greps cause non-zero exits before trap.
git -C "$CPYTHON_ROOT" checkout HEAD -- "$TARGET" >/dev/null 2>&1 || true

if [ "$CHECK_RC" -eq 0 ]; then
    echo "I4 NEG-TEST FAIL — staleness not detected (preflight_check_freshness returned 0 after touch)"
    echo "  output: $CHECK_OUT"
    exit 1
fi

if ! echo "$CHECK_OUT" | grep -q "I4 BUILD-PRODUCT FRESHNESS"; then
    echo "I4 NEG-TEST FAIL — preflight rejected staleness but did not emit canonical 'I4 BUILD-PRODUCT FRESHNESS' message"
    echo "  output: $CHECK_OUT"
    exit 1
fi

echo "I4 NEG-TEST PASS — staleness detected; canonical message emitted"
echo "  rc=$CHECK_RC"
echo "  message: $(echo "$CHECK_OUT" | grep 'I4 BUILD-PRODUCT FRESHNESS' | head -1)"
exit 0
