#!/bin/bash
# bisect_phase1_stability.sh — WS3 Phase 1 stability bisect driver
#
# Per theologian 05:59:02Z (boundary semantics) + 06:58:47Z (wrapper grammar
# correction). Used as: git bisect run scripts/bisect_phase1_stability.sh
#
# Boundary: first commit where `import _cinderx` returns 0 within 30s.
# Range: dd70e1cf89 (BAD, init segfaults) .. 2e15d1e70d (GOOD, init OK).
# Anchor: testkeeper 03:41:37Z bd00b75500 in-tree measurement.
#
# Exit codes (git bisect run convention):
#   0   = good (JIT init succeeded)
#   1   = bad  (JIT init crashed/hung/failed)
#   125 = skip (build failure; commit cannot be tested)
#
# Runs from a worktree per testkeeper 03:41:37Z pattern. Per CLAUDE.md
# Build Lock: only testkeeper executes this driver.
set -uo pipefail

CPYTHON_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CPYTHON_ROOT"

CUR=$(git log -1 --format=%H)
echo "[bisect-phase1] commit $CUR"

# Step 1: Clean build (CLEAN avoids stale objects across bisect commits).
if ! scripts/build_phoenix.sh --clean 2>&1 | tail -5; then
    echo "[bisect-phase1] BUILD FAIL on $CUR -> skip (125)"
    exit 125
fi

# Step 2: Init-stability test per theologian 06:58:47Z authorized grammar.
TEST_OUT=$(mktemp)
PYTHONJITAUTO=1 timeout 30 ./python -c 'import _cinderx; print("OK")' \
    >"$TEST_OUT" 2>&1
TEST_EXIT=$?

if grep -q '^OK$' "$TEST_OUT"; then
    echo "[bisect-phase1] $CUR -> GOOD (init OK)"
    rm -f "$TEST_OUT"
    exit 0
fi

echo "[bisect-phase1] $CUR -> BAD (exit=$TEST_EXIT)"
echo "[bisect-phase1] last 5 lines of test output:"
tail -5 "$TEST_OUT"
rm -f "$TEST_OUT"
exit 1
