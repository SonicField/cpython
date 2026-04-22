#!/bin/bash
# rc_oracle_self_test.sh — W3 R4 oracle self-test (Step 5)
#
# Validates the oracle's diagnostic capability: synthetic refcount
# divergence injection must produce non-empty diff (oracle WORKS), and
# clean run must produce empty diff (oracle is NOT noisy).
#
# Per theologian 2026-04-22 02:30:08Z W3 spec falsifier:
#   "inject a synthetic refcount bug into the C path (e.g., remove an
#   Incref or add an extra Decref after CheckField) — scripts/
#   rc_diff_oracle.sh produces non-empty output (showing the divergence).
#   If diff is empty under injection, oracle is non-functional and W3
#   has not delivered the diagnostic capability it promised."
#
# Per supervisor 2026-04-22 02:51:14Z option (c): out-of-band link of
# python_rc_cpp via the recipe documented in scripts/build_oracle.sh.
# This script does NOT invoke the link itself — it verifies the
# pre-conditions and exercises the oracle once the operator has built
# both binaries.
#
# Usage:
#   scripts/rc_oracle_self_test.sh           # full self-test
#   scripts/rc_oracle_self_test.sh --check   # pre-conditions only
#
# Exit code: 0 = self-test PASS, non-zero = FAIL with diagnostic

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CPYTHON_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRATCH_DIR="$CPYTHON_ROOT/docs/oracle_scratch"
PROD_PYTHON="$CPYTHON_ROOT/python"
RC_PYTHON="$SCRATCH_DIR/python_rc_cpp"
DIFF_DRIVER="$SCRIPT_DIR/rc_diff_oracle.sh"
INJECT_TARGET="$CPYTHON_ROOT/Python/jit/hir/refcount_pass_c.c"

CHECK_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --check) CHECK_ONLY=1 ;;
        *)       echo "Unknown flag: $arg"; exit 4 ;;
    esac
done

# ---- Pre-condition #1: production python exists + has NO rc_oracle symbols ----

echo "=== rc_oracle_self_test ==="
echo ""
echo "--- Pre-condition #1: production python free of rc_oracle symbols ---"
if [ ! -x "$PROD_PYTHON" ]; then
    echo "FAIL: $PROD_PYTHON does not exist or is not executable" >&2
    echo "      Run scripts/build_phoenix.sh first" >&2
    exit 1
fi

PROD_NM_HITS=$(nm "$PROD_PYTHON" 2>/dev/null | grep -c rc_oracle || true)
if [ "$PROD_NM_HITS" -ne 0 ]; then
    echo "FAIL: production python contains $PROD_NM_HITS rc_oracle symbol(s)" >&2
    echo "      Falsifier triggered: build system accidentally enabled RC_ORACLE" >&2
    nm "$PROD_PYTHON" | grep rc_oracle | head -5 >&2
    exit 1
fi
echo "PASS: production python has 0 rc_oracle symbols (RC_ORACLE undefined)"

# ---- Pre-condition #2: scratch lib exists ----

echo ""
echo "--- Pre-condition #2: libphoenix_rc_oracle.a exists ---"
SCRATCH_LIB="$SCRATCH_DIR/build/libphoenix_rc_oracle.a"
if [ ! -f "$SCRATCH_LIB" ]; then
    echo "FAIL: $SCRATCH_LIB does not exist" >&2
    echo "      Run scripts/build_oracle.sh to build the scratch lib" >&2
    exit 1
fi
LIB_HITS=$(nm "$SCRATCH_LIB" 2>/dev/null | grep -c ' T rc_oracle_run' || true)
if [ "$LIB_HITS" -lt 1 ]; then
    echo "FAIL: $SCRATCH_LIB does not export rc_oracle_run T-symbol" >&2
    exit 1
fi
echo "PASS: scratch lib exports rc_oracle_run (C entry point)"

# ---- Pre-condition #3: python_rc_cpp exists + HAS rc_oracle symbols ----

echo ""
echo "--- Pre-condition #3: python_rc_cpp built with RC_ORACLE active ---"
if [ ! -x "$RC_PYTHON" ]; then
    echo "FAIL: $RC_PYTHON does not exist or is not executable" >&2
    echo "      Out-of-band link required (see build_oracle.sh tail for recipe)" >&2
    echo "      Per supervisor 02:51:14Z option (c): operator runs link manually." >&2
    exit 1
fi

RC_NM_HITS=$(nm "$RC_PYTHON" 2>/dev/null | grep -c ' T rc_oracle_run' || true)
if [ "$RC_NM_HITS" -lt 1 ]; then
    echo "FAIL: python_rc_cpp does not contain rc_oracle_run T-symbol" >&2
    echo "      Either RC_ORACLE was not defined OR libphoenix_rc_oracle.a not linked" >&2
    exit 1
fi
echo "PASS: python_rc_cpp has rc_oracle_run T-symbol (RC_ORACLE defined + lib linked)"

if [ "$CHECK_ONLY" -eq 1 ]; then
    echo ""
    echo "=== --check complete: all pre-conditions PASS ==="
    exit 0
fi

# ---- Phase A: clean run (no injection) — diff MUST be empty ----

echo ""
echo "--- Phase A: clean run, expect empty diff ---"
if "$DIFF_DRIVER"; then
    echo "PASS: clean oracle run produces empty diff (C and C++ refcount sequences match)"
else
    echo "FAIL: clean oracle run produces non-empty diff (PRE-EXISTING DIVERGENCE)" >&2
    echo "      Either C path has a real bug or pre/post pass conflation (invariant #6)" >&2
    exit 1
fi

# ---- Phase B: synthetic injection — diff MUST be non-empty ----

echo ""
echo "--- Phase B: synthetic refcount-bug injection, expect non-empty diff ---"

if [ ! -f "$INJECT_TARGET" ]; then
    echo "FAIL: injection target $INJECT_TARGET does not exist" >&2
    echo "      Self-test cannot prove the oracle catches injected divergence" >&2
    exit 1
fi

# Synthetic defect: comment out one Incref call in refcount_pass_c.c.
# This produces a real refcount divergence (under-count → leak under
# Py_REF_DEBUG OR use-after-free under PYDEBUG).
INJECT_BACKUP="$INJECT_TARGET.oracle_backup"
cp "$INJECT_TARGET" "$INJECT_BACKUP"
trap 'cp "$INJECT_BACKUP" "$INJECT_TARGET" 2>/dev/null; rm -f "$INJECT_BACKUP"' EXIT

# Find the FIRST 'phx_rc_emit_incref' call site and comment it.
# (Pattern is stable across the C port per Phase 0 audit.)
INCREF_LINE=$(grep -n 'phx_rc_emit_incref' "$INJECT_TARGET" | head -1 | cut -d: -f1 || true)
if [ -z "$INCREF_LINE" ]; then
    echo "FAIL: no phx_rc_emit_incref call site found in $INJECT_TARGET" >&2
    echo "      Refactor self-test to use a different injection target" >&2
    exit 1
fi

echo "Injecting at $INJECT_TARGET:$INCREF_LINE (commenting out first phx_rc_emit_incref)"
sed -i "${INCREF_LINE}s|^|//RC_ORACLE_INJECT://|" "$INJECT_TARGET"

# Rebuild the C path.
echo "Rebuilding C path with injected defect..."
( cd "$CPYTHON_ROOT" && cmake --build Python/jit_build/build --target phoenix_jit -- -j32 ) >/dev/null
( cd "$CPYTHON_ROOT" && make -j32 python ) >/dev/null

# Run the diff driver. Expect NON-empty (exit 1).
echo "Running diff driver under injection..."
if "$DIFF_DRIVER"; then
    echo "FAIL: oracle produced empty diff under injection" >&2
    echo "      The injected defect was NOT caught — oracle is non-functional" >&2
    exit 1
fi
echo "PASS: oracle produces non-empty diff under injection (oracle WORKS)"

# Trap will restore the file. Rebuild after restore.
cp "$INJECT_BACKUP" "$INJECT_TARGET"
rm -f "$INJECT_BACKUP"
trap - EXIT
echo "Restoring C path + rebuilding..."
( cd "$CPYTHON_ROOT" && cmake --build Python/jit_build/build --target phoenix_jit -- -j32 ) >/dev/null
( cd "$CPYTHON_ROOT" && make -j32 python ) >/dev/null

# Final verification: clean run after restore must again produce empty diff.
echo "Final verification: post-restore diff must be empty..."
if "$DIFF_DRIVER"; then
    echo "PASS: post-restore oracle produces empty diff (restore successful)"
else
    echo "FAIL: post-restore diff non-empty — RESTORE LEFT C PATH BROKEN" >&2
    echo "      git checkout $INJECT_TARGET to recover" >&2
    exit 1
fi

echo ""
echo "=== rc_oracle_self_test PASS ==="
echo ""
echo "Falsifier evidence:"
echo "  - Pre-condition #1: production python has 0 rc_oracle symbols"
echo "  - Pre-condition #3: python_rc_cpp has rc_oracle_run T-symbol"
echo "  - Phase A:          clean diff is empty"
echo "  - Phase B:          injection produces non-empty diff (oracle WORKS)"
echo "  - Post-restore:     diff is empty (restore clean)"
echo ""
echo "W3 R4 oracle is OPERATIONAL and DIAGNOSTIC."
exit 0
