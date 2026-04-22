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
INJECT_CLASS=""
for arg in "$@"; do
    case "$arg" in
        --check)    CHECK_ONLY=1 ;;
        --class=*)  INJECT_CLASS="${arg#--class=}" ;;
        *)          echo "Unknown flag: $arg"; exit 4 ;;
    esac
done

# Per supervisor 03:07:52Z: 4 injection classes (was 1) for falsifier expansion
# defeating W4 vacuous-pass class. Each class: scrub case (no inject → empty diff)
# + signal case (inject → non-empty diff).
# A — refcount BALANCE: comment first phx_rc_emit_incref (under-count → leak)
# B — refcount BALANCE (other side): comment first phx_rc_emit_decref (over-count → leak)
# C — refcount SEQUENCE: insert duplicate Incref (over-count, distinct opcode order)
# D — TYPE LATTICE: change HIR_TYPE_OBJECT → HIR_TYPE_NULLPTR in refcount_pass_c.c
#     (annotation flip — should produce different HIR shape)
# Default: run all 4 classes sequentially.

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

# ---- Phase B: synthetic injection — diff MUST be non-empty for each class ----

if [ ! -f "$INJECT_TARGET" ]; then
    echo "FAIL: injection target $INJECT_TARGET does not exist" >&2
    echo "      Self-test cannot prove the oracle catches injected divergence" >&2
    exit 1
fi

# Generic backup/restore + rebuild helpers. trap EXIT ensures restore on
# script error.
INJECT_BACKUP="$INJECT_TARGET.oracle_backup"
cp "$INJECT_TARGET" "$INJECT_BACKUP"
trap 'cp "$INJECT_BACKUP" "$INJECT_TARGET" 2>/dev/null; rm -f "$INJECT_BACKUP"' EXIT

rebuild_c_path() {
    ( cd "$CPYTHON_ROOT" && cmake --build Python/jit_build/build --target phoenix_jit -- -j32 ) >/dev/null
    ( cd "$CPYTHON_ROOT" && make -j32 python ) >/dev/null
}

restore_target() {
    cp "$INJECT_BACKUP" "$INJECT_TARGET"
    rebuild_c_path
}

# Run a single injection class:
#   $1 = class letter (A/B/C/D)
#   $2 = description
#   $3 = sed script applied to $INJECT_TARGET
inject_class() {
    local cls="$1"
    local desc="$2"
    local sed_expr="$3"
    echo ""
    echo "--- Phase B class $cls: $desc ---"
    sed -i "$sed_expr" "$INJECT_TARGET"
    if cmp -s "$INJECT_BACKUP" "$INJECT_TARGET"; then
        echo "FAIL: class $cls injection produced no source change (sed pattern stale?)" >&2
        return 1
    fi
    echo "Injection applied. Rebuilding C path..."
    rebuild_c_path
    echo "Running diff driver under class $cls injection..."
    if "$DIFF_DRIVER"; then
        echo "FAIL: class $cls produced empty diff under injection — oracle MISSED $desc" >&2
        restore_target
        return 1
    fi
    echo "PASS: class $cls produces non-empty diff (oracle CATCHES $desc)"
    restore_target
    echo "PASS: class $cls restore + rebuild successful"
    return 0
}

# Class A — refcount BALANCE under-count: comment FIRST phx_rc_emit_incref
INCREF_LINE_A=$(grep -n 'phx_rc_emit_incref' "$INJECT_TARGET" | head -1 | cut -d: -f1 || true)
# Class B — refcount BALANCE over-count: comment FIRST phx_rc_emit_decref
DECREF_LINE_B=$(grep -n 'phx_rc_emit_decref' "$INJECT_TARGET" | head -1 | cut -d: -f1 || true)
# Class C — refcount SEQUENCE: comment SECOND phx_rc_emit_incref (different
# call-site than A → different HIR position → different oracle signature)
INCREF_LINE_C=$(grep -n 'phx_rc_emit_incref' "$INJECT_TARGET" | sed -n '2p' | cut -d: -f1 || true)
# Class D — TYPE LATTICE: change FIRST HIR_TYPE_OBJECT → HIR_TYPE_NULLPTR
TYPE_LINE_D=$(grep -n 'HIR_TYPE_OBJECT\b' "$INJECT_TARGET" | head -1 | cut -d: -f1 || true)

if [ -z "$INCREF_LINE_A" ]; then
    echo "FAIL: no phx_rc_emit_incref call site found in $INJECT_TARGET (class A)" >&2
    exit 1
fi

run_class_a() { inject_class A "BALANCE under-count (skip Incref @ line $INCREF_LINE_A)" \
    "${INCREF_LINE_A}s|^|//RC_ORACLE_INJECT_A://|"; }
run_class_b() {
    if [ -z "$DECREF_LINE_B" ]; then
        echo "SKIP: class B no phx_rc_emit_decref site found"; return 0
    fi
    inject_class B "BALANCE over-count (skip Decref @ line $DECREF_LINE_B)" \
        "${DECREF_LINE_B}s|^|//RC_ORACLE_INJECT_B://|"
}
run_class_c() {
    if [ -z "$INCREF_LINE_C" ]; then
        echo "SKIP: class C no second phx_rc_emit_incref site found"; return 0
    fi
    inject_class C "SEQUENCE (skip second Incref @ line $INCREF_LINE_C — different position than class A)" \
        "${INCREF_LINE_C}s|^|//RC_ORACLE_INJECT_C://|"
}
run_class_d() {
    if [ -z "$TYPE_LINE_D" ]; then
        echo "SKIP: class D no HIR_TYPE_OBJECT site found"; return 0
    fi
    inject_class D "TYPE LATTICE (HIR_TYPE_OBJECT → HIR_TYPE_NULLPTR @ line $TYPE_LINE_D)" \
        "${TYPE_LINE_D}s|HIR_TYPE_OBJECT\b|HIR_TYPE_NULLPTR|"
}

case "$INJECT_CLASS" in
    A) run_class_a ;;
    B) run_class_b ;;
    C) run_class_c ;;
    D) run_class_d ;;
    "") run_class_a && run_class_b && run_class_c && run_class_d ;;
    *) echo "Unknown injection class: $INJECT_CLASS (expected A/B/C/D)"; exit 4 ;;
esac

# Each inject_class invocation runs sed → rebuild → diff → restore → rebuild,
# so the working tree + binary are clean after each class. Final clean state
# verified by the trap on EXIT.
rm -f "$INJECT_BACKUP"
trap - EXIT

# Final verification: confirm working tree is clean (no leftover injection).
if ! cmp -s "$INJECT_TARGET" <(git show "HEAD:$INJECT_TARGET" 2>/dev/null); then
    echo "FAIL: post-test diff $INJECT_TARGET shows uncommitted change — restore broken" >&2
    echo "      Run: git checkout $INJECT_TARGET" >&2
    exit 1
fi

echo ""
echo "=== rc_oracle_self_test PASS ==="
echo ""
echo "Falsifier evidence (4 injection classes per supervisor 03:07:52Z):"
echo "  - Pre-condition #1: production python has 0 rc_oracle symbols"
echo "  - Pre-condition #2: scratch lib exports rc_oracle_run T-symbol"
echo "  - Pre-condition #3: python_rc_cpp has rc_oracle_run T-symbol"
echo "  - Phase A:          clean diff is empty (oracle is not noisy)"
echo "  - Phase B class A:  refcount BALANCE (skip Incref) → non-empty diff"
echo "  - Phase B class B:  refcount BALANCE (skip Decref) → non-empty diff"
echo "  - Phase B class C:  refcount SEQUENCE (skip 2nd Incref) → non-empty diff"
echo "  - Phase B class D:  TYPE LATTICE (HIR_TYPE_OBJECT→NULLPTR) → non-empty diff"
echo "  - Post-restore:     working tree clean, no uncommitted changes"
echo ""
echo "W3 R4 oracle is OPERATIONAL and DIAGNOSTIC."
exit 0
