#!/bin/bash
# test_phoenix.sh — Full Phoenix JIT test suite
# Calls verify_jit.sh first — if JIT isn't compiling, test suite doesn't run
set -euo pipefail

CPYTHON_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="$CPYTHON_ROOT/scripts"

echo "=== Phoenix JIT Test Suite ==="

# Step 1: Verify JIT is compiling (PREREQUISITE)
echo "--- Step 1: Verify JIT is compiling ---"
"$SCRIPTS/verify_jit.sh" || { echo "FAIL: JIT not compiling. Aborting test suite."; exit 1; }

# Step 2: Full test suite
echo ""
echo "--- Step 2: Full test suite ---"
ASAN_OPTIONS=detect_leaks=0 "$CPYTHON_ROOT/python" -m test -j4 2>&1 | tee /tmp/phoenix_test_output.txt

# Step 3: Report
echo ""
echo "--- Step 3: Summary ---"
echo "Full output: /tmp/phoenix_test_output.txt"
tail -10 /tmp/phoenix_test_output.txt
