#!/bin/bash
# w45_bridge_drift_falsifier.sh — bridge-signature drift falsifier.
#
# Mutates a bridge signature (extern "C" decl in builder.cpp + function
# definition in builder_emit_c.c, in lockstep) and verifies the build
# fails at the C++ dispatch-switch call site, which has the OLD arity.
#
# If a mutation lets the build pass silently → STRUCTURAL DRIFT surface
# confirmed; void* args at the bridge crossing erase the type-safety the
# C++ side relied on (252+ dispatch sites in builder.cpp post Phase 1).
#
# Filed per W45 (D-1776981403, supervisor 21:55:36Z + theologian
# 21:55:24Z + pythia #90/#91 substantive). Spec:
# docs/w45-bridge-signature-drift-falsifier.md.
#
# Symmetric to scripts/check_do_not_use_callers.sh (W44 caller-gate)
# and scripts/caller_grep.sh (push 92 dispatch-callers audit).
#
# Usage:
#   scripts/w45_bridge_drift_falsifier.sh             # default fixtures
#   scripts/w45_bridge_drift_falsifier.sh --dry-run   # print mutations only
#   scripts/w45_bridge_drift_falsifier.sh --strict    # exit 1 on any FAIL
#   scripts/w45_bridge_drift_falsifier.sh --verbose   # show build stderr
#
# Build lock: this script invokes cmake --build (sole-arg target=jit).
# Per CLAUDE.md Phase 3D Build Lock, ONLY testkeeper or gate_phoenix.sh
# may invoke this script with builds enabled. Use --dry-run for non-
# build verification (any agent).

set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

DRY_RUN=0
STRICT=0
VERBOSE=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --strict)  STRICT=1 ;;
        --verbose) VERBOSE=1 ;;
        *) echo "ERROR: unknown arg '$arg'" >&2
           echo "Usage: $0 [--dry-run] [--strict] [--verbose]" >&2
           exit 1 ;;
    esac
done

BUILDER_CPP="Python/jit/hir/builder.cpp"
BUILDER_EMIT_C="Python/jit/hir/builder_emit_c.c"
BUILDER_STATE_C_H="Python/jit/hir/builder_state_c.h"
BUILDER_STATE_C="Python/jit/hir/builder_state_c.c"
CMAKE_BUILD_DIR="Python/jit_build/build"

for f in "$BUILDER_CPP" "$BUILDER_EMIT_C" "$BUILDER_STATE_C_H" "$BUILDER_STATE_C"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing $f — wrong CWD?" >&2
        exit 1
    fi
done

echo "=== w45_bridge_drift_falsifier.sh — W45 gate ==="
echo "HEAD: $(git rev-parse HEAD)"
echo "Mode: $([ $DRY_RUN -eq 1 ] && echo DRY-RUN || echo BUILD)"
echo ""

# Backup files atomically; restore on any exit.
BACKUP_CPP=$(mktemp /tmp/w45_builder_cpp.XXXXXX)
BACKUP_C=$(mktemp /tmp/w45_builder_emit_c.XXXXXX)
BACKUP_STATE_H=$(mktemp /tmp/w45_builder_state_h.XXXXXX)
BACKUP_STATE_C=$(mktemp /tmp/w45_builder_state_c.XXXXXX)
cp "$BUILDER_CPP" "$BACKUP_CPP"
cp "$BUILDER_EMIT_C" "$BACKUP_C"
cp "$BUILDER_STATE_C_H" "$BACKUP_STATE_H"
cp "$BUILDER_STATE_C" "$BACKUP_STATE_C"

restore_files() {
    cp "$BACKUP_CPP" "$BUILDER_CPP" 2>/dev/null || true
    cp "$BACKUP_C" "$BUILDER_EMIT_C" 2>/dev/null || true
    cp "$BACKUP_STATE_H" "$BUILDER_STATE_C_H" 2>/dev/null || true
    cp "$BACKUP_STATE_C" "$BUILDER_STATE_C" 2>/dev/null || true
    rm -f "$BACKUP_CPP" "$BACKUP_C" "$BACKUP_STATE_H" "$BACKUP_STATE_C" 2>/dev/null || true
}
trap restore_files EXIT

# Test fixtures — bridge-symbol : description.
# Retro fixtures (#6 + #7) per spec §2.6 + supervisor 22:33:17Z DISPOSITION (C).
# Sample bridges per spec §2.2 cover spectrum of arg counts (3-5).
FIXTURES=(
    "hir_builder_emit_before_with_c|Phase 1 #6 emitBeforeWith fold-into-C (3 args)"
    "hir_builder_emit_setup_with_c|Phase 1 #7 emitSetupWith fold-into-C (4 args)"
    "hir_builder_emit_format_simple_c|Phase 1 #2 emitFormatSimple delegation (3 args)"
    "hir_builder_emit_copy_free_vars_c|Phase 1 #4 emitCopyFreeVars delegation (5 args)"
    "hir_builder_emit_get_yield_from_iter_c|Phase 1 #4 emitGetYieldFromIter delegation (5 args)"
    "hir_builder_emit_primitive_load_const_c|Phase 1 #5 emitPrimitiveLoadConst delegation (5 args)"
    "hir_builder_state_init|Phase 3 Batch 1 state init (3 args)"
    "hir_builder_state_parse_exception_table_c|Phase 3 Batch 1 parseExceptionTable C body (2 args)"
    "hir_builder_state_exception_table_push_cpp|Phase 3 Batch 1 exception_table push bridge (6 args)"
    "hir_builder_state_exception_table_size_cpp|Phase 3 Batch 2 exception_table size bridge (1 arg)"
    "hir_builder_state_exception_table_entry_cpp|Phase 3 Batch 2 exception_table entry bridge (7 args)"
    "hir_builder_state_find_exception_handler_c|Phase 3 Batch 2 findExceptionHandler C body (4 args)"
    "hir_builder_state_block_map_blocks_lookup_cpp|Phase 3 Batch 4 block_map blocks lookup bridge (2 args)"
    "hir_builder_state_static_method_stack_pop_cpp|Phase 3 Batch 5 static_method_stack pop bridge (1 arg)"
)

# Mutation: append ', int phx_w45_drift' before the closing paren of the
# bridge function decl + def. Uses perl -0 (slurp) for multi-line regex
# since extern decls span 2 lines and definitions span more.
mutate_bridge() {
    local symbol="$1"
    # Match: <return-type> <symbol>( ... ) — capture inner contents, append
    # drift param. Multi-line via perl -0777. Return type is any non-paren
    # token sequence (int, void, void*, bool, etc.). Apply across all
    # source files containing bridge decls / defs.
    perl -i -0777 -pe \
        "s/(\b\w[\w\s\*]*?[\s\*]+${symbol}\s*\([^)]*?)\)/\${1}, int phx_w45_drift)/g" \
        "$BUILDER_CPP" "$BUILDER_EMIT_C" "$BUILDER_STATE_C_H" "$BUILDER_STATE_C"
}

# Verify mutation actually applied somewhere (perl substitution silently
# no-ops if pattern doesn't match — guard against false-positive PASS).
verify_mutated() {
    local symbol="$1"
    grep -l "phx_w45_drift" \
        "$BUILDER_CPP" "$BUILDER_EMIT_C" "$BUILDER_STATE_C_H" "$BUILDER_STATE_C" \
        2>/dev/null | grep -q .
}

PASS=0
FAIL=0
SKIP=0

for fixture in "${FIXTURES[@]}"; do
    SYMBOL="${fixture%%|*}"
    DESC="${fixture#*|}"

    echo "--- Fixture: $SYMBOL"
    echo "    $DESC"

    # Restore before each fixture (defensive).
    cp "$BACKUP_CPP" "$BUILDER_CPP"
    cp "$BACKUP_C" "$BUILDER_EMIT_C"
    cp "$BACKUP_STATE_H" "$BUILDER_STATE_C_H"
    cp "$BACKUP_STATE_C" "$BUILDER_STATE_C"

    # Mutate.
    mutate_bridge "$SYMBOL"

    if ! verify_mutated "$SYMBOL"; then
        echo "    [SKIP] mutation did not apply (symbol not found in either file)"
        SKIP=$((SKIP+1))
        echo ""
        continue
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "    [DRY] mutation applied; mutation-marker sites (build skipped):"
        { grep -n "phx_w45_drift" "$BUILDER_CPP" 2>/dev/null || true; } | sed 's/^/      cpp:     /'
        { grep -n "phx_w45_drift" "$BUILDER_EMIT_C" 2>/dev/null || true; } | sed 's/^/      emit_c:  /'
        { grep -n "phx_w45_drift" "$BUILDER_STATE_C_H" 2>/dev/null || true; } | sed 's/^/      state_h: /'
        { grep -n "phx_w45_drift" "$BUILDER_STATE_C" 2>/dev/null || true; } | sed 's/^/      state_c: /'
        SKIP=$((SKIP+1))
        echo ""
        continue
    fi

    # Build with mutation. Expect FAIL at dispatch site (call has OLD arity).
    BUILD_OUT=$(mktemp /tmp/w45_build.XXXXXX)
    BUILD_RC=0
    cmake --build "$CMAKE_BUILD_DIR" --target jit -j8 > "$BUILD_OUT" 2>&1 || BUILD_RC=$?

    if [ "$BUILD_RC" -eq 0 ]; then
        echo "    [FAIL] build PASSED with mutated sig — DRIFT SURFACE UNDETECTED"
        FAIL=$((FAIL+1))
        if [ "$VERBOSE" -eq 1 ]; then
            tail -20 "$BUILD_OUT" | sed 's/^/      build: /'
        fi
    else
        # Verify the failure mentions builder.cpp dispatch site
        # (i.e., the C++ call site complains about arg count, not some
        # unrelated error like a header gone missing).
        if grep -qE "builder\.cpp.*${SYMBOL}|${SYMBOL}.*builder\.cpp" "$BUILD_OUT"; then
            echo "    [PASS] build failed at builder.cpp dispatch site (drift caught)"
            PASS=$((PASS+1))
        elif grep -q "$SYMBOL" "$BUILD_OUT"; then
            # Failure mentions the symbol but not necessarily builder.cpp;
            # still counts as detection.
            echo "    [PASS] build failed (symbol cited in error)"
            PASS=$((PASS+1))
        else
            echo "    [FAIL] build failed but NOT at expected dispatch site"
            FAIL=$((FAIL+1))
            if [ "$VERBOSE" -eq 1 ]; then
                tail -20 "$BUILD_OUT" | sed 's/^/      build: /'
            fi
        fi
    fi
    rm -f "$BUILD_OUT"
    echo ""
done

# Final restore + clean rebuild verification (skipped in dry-run).
echo "--- Restore + verify clean rebuild"
cp "$BACKUP_CPP" "$BUILDER_CPP"
cp "$BACKUP_C" "$BUILDER_EMIT_C"
cp "$BACKUP_STATE_H" "$BUILDER_STATE_C_H"
cp "$BACKUP_STATE_C" "$BUILDER_STATE_C"

if [ "$DRY_RUN" -eq 0 ]; then
    BUILD_OUT=$(mktemp /tmp/w45_build_restore.XXXXXX)
    BUILD_RC=0
    cmake --build "$CMAKE_BUILD_DIR" --target jit -j8 > "$BUILD_OUT" 2>&1 || BUILD_RC=$?
    if [ "$BUILD_RC" -eq 0 ]; then
        echo "    [OK] post-restore build PASSED (working tree clean)"
    else
        echo "    [WARN] post-restore build FAILED — check working tree:"
        tail -10 "$BUILD_OUT" | sed 's/^/      build: /'
    fi
    rm -f "$BUILD_OUT"
fi
echo ""

echo "=== W45 Verdict ==="
echo "Fixtures: ${#FIXTURES[@]}  PASS: $PASS  FAIL: $FAIL  SKIP: $SKIP"

if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "GATE FAIL: $FAIL fixture(s) silently passed build despite signature"
    echo "mutation. void* arg-erasure at C-bridge crossing is NOT being"
    echo "caught by the toolchain at the affected dispatch site(s)."
    [ "$STRICT" -eq 1 ] && exit 1
    exit 0
fi

if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY-RUN COMPLETE: $SKIP fixture(s) staged. Re-run without --dry-run to verify build behavior."
    exit 0
fi

echo "GATE PASS: all $PASS fixture(s) caught at build time. Bridge-sig drift surface protected."
exit 0
