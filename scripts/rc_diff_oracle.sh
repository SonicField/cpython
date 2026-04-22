#!/bin/bash
# rc_diff_oracle.sh — W3 R4 oracle resurrection driver
#
# Reconstructed-C++ refcount_insertion (from cc4a18e7e5, the commit before R4
# deleted it) vs current-C refcount_insertion. Captures post-pass HIR from
# both implementations for a representative function corpus and diffs the
# two. Empty diff = no refcount divergence; non-empty = oracle has surfaced
# a class of bug equivalent to what RC_DIFF caught in the prior session.
#
# Driver only — the scratch C++ binary is built by Step 4 (CMakeLists) and
# requires Step 3 (_rc_oracle_adapter.h shim) which marshals between the
# cc4a18e7e5 LivenessAnalysis class and current HirLivenessState C API.
#
# Per theologian 2026-04-22 02:30:08Z W3 spec:
#   ORACLE PIN — adapter is built against d81e5806c3 ONLY. NOT against HEAD.
#   Pin enforced by CMakeLists git checkout in scratch build dir.
#
# Per supervisor 2026-04-22 02:29:15Z: oracle-pinned to push 35 HEAD d81e5806c3.
#
# Usage:
#   scripts/rc_diff_oracle.sh                        # run full corpus, exit non-zero on any diff
#   scripts/rc_diff_oracle.sh --only=fib             # single function
#   scripts/rc_diff_oracle.sh --inject=missing-incref  # synthetic-defect mode (Step 5)
#
# Exit code: 0 = oracle PASS (no divergence), 1 = oracle FAIL (divergence captured)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CPYTHON_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Pin per theologian 02:30:08Z + supervisor 02:29:15Z.
ORACLE_PIN="d81e5806c3"

# Path layout (matches expected Step 2/4 outputs).
SCRATCH_DIR="${RC_ORACLE_SCRATCH:-$CPYTHON_ROOT/docs/oracle_scratch}"
C_PYTHON="$CPYTHON_ROOT/python"
CPP_PYTHON="$SCRATCH_DIR/python_rc_cpp"

# Corpus — small functions that exercise the refcount_insertion pass on
# different HIR shapes. Kept inline to avoid external file churn.
CORPUS_FILE="$SCRATCH_DIR/rc_corpus.py"

# Default function list. Extend by editing the rc_corpus.py file post-Step-2.
DEFAULT_FUNCTIONS="fib add_chain attr_probe loop_sum tuple_make"

# Parse flags
ONLY=""
INJECT=""
for arg in "$@"; do
    case "$arg" in
        --only=*)   ONLY="${arg#--only=}" ;;
        --inject=*) INJECT="${arg#--inject=}" ;;
        *)          echo "Unknown flag: $arg" >&2; exit 1 ;;
    esac
done

echo "=== rc_diff_oracle (W3 R4 resurrection) ==="
echo "Oracle pin: $ORACLE_PIN"
echo "C path binary:   $C_PYTHON"
echo "C++ path binary: $CPP_PYTHON"
echo ""

# Pre-flight: both binaries must exist and report the pinned commit (or, for
# the C path, ANY commit at-or-after the pin — the current code is the path
# being validated).
if [ ! -x "$C_PYTHON" ]; then
    echo "FAIL: C-path Python binary not built at $C_PYTHON" >&2
    echo "      run scripts/build_phoenix.sh first" >&2
    exit 1
fi
if [ ! -x "$CPP_PYTHON" ]; then
    echo "FAIL: C++-path Python binary not built at $CPP_PYTHON" >&2
    echo "      run scripts/build_oracle_scratch.sh (Step 4 CMakeLists)" >&2
    echo "      requires _rc_oracle_adapter.h shim (Step 3)" >&2
    exit 1
fi
if [ ! -f "$CORPUS_FILE" ]; then
    echo "FAIL: corpus file not found at $CORPUS_FILE" >&2
    echo "      Step 2 places the corpus alongside reconstructed sources" >&2
    exit 1
fi

# Verify C++-path binary is pinned correctly (must contain the pin hash).
CPP_VERSION=$("$CPP_PYTHON" -c 'import sys; print(sys.version)' 2>&1)
if ! echo "$CPP_VERSION" | grep -q "$ORACLE_PIN"; then
    echo "FAIL: C++-path binary not pinned to $ORACLE_PIN" >&2
    echo "      reports: $CPP_VERSION" >&2
    echo "      W3 demands oracle is FROZEN at d81e5806c3 — rebuild scratch lib" >&2
    exit 1
fi
echo "C++-path BINARY_PIN: $ORACLE_PIN ✓"

# Determine corpus to run.
FUNCTIONS="${ONLY:-$DEFAULT_FUNCTIONS}"
FUNCTIONS=$(echo "$FUNCTIONS" | tr ',' ' ')

# Optional synthetic-defect injection (Step 5 self-test mode).
if [ -n "$INJECT" ]; then
    echo "INJECTION MODE: $INJECT"
    echo "  - apply known refcount divergence to C path before diffing"
    echo "  - expected: non-empty diff (oracle surfaces the injection)"
    case "$INJECT" in
        missing-incref|extra-decref|swap-incref-order)
            ;;
        *)
            echo "FAIL: unknown injection '$INJECT'" >&2
            echo "      supported: missing-incref, extra-decref, swap-incref-order" >&2
            exit 1
            ;;
    esac
    # Step 5 will implement the actual injection mechanism (likely a sed patch
    # to refcount_insertion_c.c followed by rebuild). Stub here marks the
    # contract.
    echo "  (injection mechanism pending Step 5 implementation)"
fi

# Run the diff for each function.
TMPDIR=$(mktemp -d -t rc_diff_oracle.XXXXXX)
trap "rm -rf $TMPDIR" EXIT
TOTAL=0
DIFF_COUNT=0
for func in $FUNCTIONS; do
    TOTAL=$((TOTAL + 1))
    C_OUT="$TMPDIR/${func}.c.hir"
    CPP_OUT="$TMPDIR/${func}.cpp.hir"

    # Capture C-path HIR (current refcount_insertion).
    PHOENIX_GOLDEN_CAPTURE=1 RC_ORACLE_FUNC="$func" "$C_PYTHON" "$CORPUS_FILE" \
        2> "$C_OUT" >/dev/null || true

    # Capture C++-path HIR (reconstructed cc4a18e7e5 refcount_insertion via
    # the scratch lib).
    PHOENIX_GOLDEN_CAPTURE=1 RC_ORACLE_FUNC="$func" "$CPP_PYTHON" "$CORPUS_FILE" \
        2> "$CPP_OUT" >/dev/null || true

    # Extract just the post-refcount-insertion HIR block. Use the existing
    # GOLDEN_HIR_FINAL marker (refcount_insertion runs late in the pass
    # pipeline, so HIR_FINAL captures the post-pass state).
    awk '/^GOLDEN_HIR_FINAL/,/^END_GOLDEN_HIR_FINAL/' "$C_OUT" > "$TMPDIR/${func}.c.block"
    awk '/^GOLDEN_HIR_FINAL/,/^END_GOLDEN_HIR_FINAL/' "$CPP_OUT" > "$TMPDIR/${func}.cpp.block"

    if [ ! -s "$TMPDIR/${func}.c.block" ] || [ ! -s "$TMPDIR/${func}.cpp.block" ]; then
        echo "FAIL [$func]: missing HIR block from one or both binaries" >&2
        echo "  C-path size:   $(wc -c < "$TMPDIR/${func}.c.block") bytes" >&2
        echo "  C++-path size: $(wc -c < "$TMPDIR/${func}.cpp.block") bytes" >&2
        DIFF_COUNT=$((DIFF_COUNT + 1))
        continue
    fi

    if diff -u "$TMPDIR/${func}.cpp.block" "$TMPDIR/${func}.c.block" > "$TMPDIR/${func}.diff"; then
        echo "PASS [$func]: refcount sequence identical (oracle clean)"
    else
        echo "FAIL [$func]: refcount sequence diverged — diff follows"
        head -50 "$TMPDIR/${func}.diff" >&2
        DIFF_COUNT=$((DIFF_COUNT + 1))
    fi
done

echo ""
echo "=== rc_diff_oracle summary ==="
echo "Functions checked: $TOTAL"
echo "Divergences:       $DIFF_COUNT"

if [ -n "$INJECT" ]; then
    if [ "$DIFF_COUNT" -gt 0 ]; then
        echo "INJECTION SELF-TEST PASS — oracle surfaced the synthetic defect"
        exit 0
    else
        echo "INJECTION SELF-TEST FAIL — oracle did NOT surface synthetic '$INJECT'"
        echo "  W3 has not delivered diagnostic capability — adapter or driver broken"
        exit 1
    fi
fi

if [ "$DIFF_COUNT" -gt 0 ]; then
    echo "ORACLE FAIL — refcount divergence detected"
    exit 1
fi
echo "ORACLE PASS — current C refcount_insertion matches cc4a18e7e5 C++ on $TOTAL functions"
