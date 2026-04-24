#!/bin/bash
# w45_section_3_5_derivation_drift.sh — derivation-drift falsifier.
#
# Sibling to scripts/w45_bridge_drift_falsifier.sh. Where W45 §1-§2
# mutates BRIDGE SIGNATURES (param insertion), this script mutates
# DERIVED CONSTANTS, struct FIELD LAYOUTS, and bridge RETURN TYPES
# that C-body implementations depend on.
#
# Trigger: §3.5 5/5 backstop reached at Phase 3 Batch 4 b44a5143cc per
# supervisor 23:49:22Z (W45 §2.5 amendment 23:49:00Z + §2.7 sketch
# 00:14:30Z).
#
# Mutate-Build-Verify-Restore loop, mode (B) source-mutation (mode (A)
# W21-golden deferred until W21 lands per spec §2.7.5).
#
# Initial 4 fixtures per spec §2.7.2:
#   Class A — fold-into-C derived constants (Phase 1):
#     1. BEFORE_ASYNC_WITH opcode-derivation in emitBeforeWith C body
#     2. _Py_ID identifier-derivation in emitSetupWith C body
#   Class B — Phase 3 bridge-derived field-reads:
#     3. ExceptionTableEntry struct field rename (Batch 2 read surface)
#     4. block_map_blocks_lookup_cpp return-type change (Batch 4)
#
# Future fixtures append per shepard 22:46:33Z atomic-commit-with-burndown
# discipline (W45 §2.5/§2.7 same-commit fixture rule).
#
# Build lock: this script invokes scripts/build_phoenix.sh. Per CLAUDE.md
# Phase 3D Build Lock, ONLY testkeeper or gate_phoenix.sh may invoke
# this script with builds enabled. Use --dry-run for non-build verification
# (any agent).
#
# Restore via `git checkout HEAD -- <file>` per spec §2.7.3 (deterministic
# cleanup; idempotent; no temp-file shuffling).
#
# Usage:
#   scripts/w45_section_3_5_derivation_drift.sh             # default fixtures
#   scripts/w45_section_3_5_derivation_drift.sh --dry-run   # print mutations
#   scripts/w45_section_3_5_derivation_drift.sh --strict    # exit 1 on any FAIL
#   scripts/w45_section_3_5_derivation_drift.sh --verbose   # show build stderr

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

BUILDER_EMIT_C="Python/jit/hir/builder_emit_c.c"
BUILDER_H="Python/jit/hir/builder.h"
BUILDER_STATE_C_H="Python/jit/hir/builder_state_c.h"
HIR_C_API_CPP="Python/jit/hir/hir_c_api.cpp"

for f in "$BUILDER_EMIT_C" "$BUILDER_H" "$BUILDER_STATE_C_H" "$HIR_C_API_CPP"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing $f — wrong CWD?" >&2
        exit 1
    fi
done

echo "=== w45_section_3_5_derivation_drift.sh — §3.5 falsifier ==="
echo "HEAD: $(git rev-parse HEAD)"
echo "Mode: $([ $DRY_RUN -eq 1 ] && echo DRY-RUN || echo BUILD)"
echo ""

# Restore tracked files to their PRE-SCRIPT-INVOCATION state. Per the
# 8-incident root-cause investigation (2026-04-24, supervisor 04:14:27Z):
# the prior `git checkout HEAD --` approach BLEW AWAY pre-existing
# unstaged modifications on touched files (e.g., agent's in-flight
# Tier 8 Phase A content), causing the recurring "external revert"
# class. Fix: snapshot per-file content to /tmp on first touch; restore
# from snapshot (not HEAD) so pre-script edits survive intact.
TOUCHED_FILES=()
declare -A FILE_SNAPSHOTS
snapshot_file_if_new() {
    local f="$1"
    if [ -z "${FILE_SNAPSHOTS[$f]:-}" ]; then
        local snap
        snap=$(mktemp /tmp/w45_35_snap.XXXXXX)
        cp "$f" "$snap"
        FILE_SNAPSHOTS[$f]="$snap"
    fi
}
restore_files() {
    for f in "${!FILE_SNAPSHOTS[@]}"; do
        cp "${FILE_SNAPSHOTS[$f]}" "$f" 2>/dev/null || true
        rm -f "${FILE_SNAPSHOTS[$f]}" 2>/dev/null || true
        unset "FILE_SNAPSHOTS[$f]"
    done
    TOUCHED_FILES=()
}
trap restore_files EXIT

# --- Fixture mutation helpers ----------------------------------------------

# Fixture 1: BEFORE_ASYNC_WITH opcode-derivation rename in emitBeforeWith C
# body (builder_emit_c.c). Renames the constant to a non-existent symbol;
# C-body comparison site fails to compile (undeclared identifier).
mutate_fixture_1() {
    snapshot_file_if_new "$BUILDER_EMIT_C"
    perl -i -pe 's/\bBEFORE_ASYNC_WITH\b/BEFORE_ASYNC_WITH_PHX_W45_DRIFT/g' \
        "$BUILDER_EMIT_C"
    TOUCHED_FILES+=("$BUILDER_EMIT_C")
}
verify_fixture_1() {
    grep -q "BEFORE_ASYNC_WITH_PHX_W45_DRIFT" "$BUILDER_EMIT_C"
}
expected_site_fixture_1="builder_emit_c.c"

# Fixture 2: _Py_ID identifier-derivation rename in emitSetupWith C body.
# C body picks _Py_ID(__aenter__) / __aexit__ / __enter__ / __exit__; rename
# the macro to a non-existent symbol so the C-body identifier site fails to
# compile.
mutate_fixture_2() {
    snapshot_file_if_new "$BUILDER_EMIT_C"
    # Only target the _Py_ID call sites in emitSetupWith C body region.
    # Rename _Py_ID identifier in lines containing the bridge function name.
    # Use a sentinel substitution scoped to lines 4140-4220 (emit_setup_with_c +
    # emit_before_with_c bodies). Coarser scope is acceptable — both are
    # opcode-derivation surfaces and both should be detectable.
    perl -i -pe '
        if ($. >= 4140 && $. <= 4220 && /_Py_ID\(/) {
            s/\b_Py_ID\b/_Py_ID_PHX_W45_DRIFT/g;
        }
    ' "$BUILDER_EMIT_C"
    TOUCHED_FILES+=("$BUILDER_EMIT_C")
}
verify_fixture_2() {
    grep -q "_Py_ID_PHX_W45_DRIFT" "$BUILDER_EMIT_C"
}
expected_site_fixture_2="builder_emit_c.c"

# Fixture 3: ExceptionTableEntry struct field rename in builder_state_c.h.
# Tier 8 pilot Phase A: ExceptionTableEntry moved from builder.h C++ struct
# to builder_state_c.h C struct. Renames the `depth` field to
# `depth_phx_w45_drift`; consumers in builder_state_c.c (parse +
# find_exception_handler_c bodies populating entry.depth) and builder.cpp
# (handler.depth in emitInlineExceptionMatch + emitCallExceptionHandler)
# all fail to compile (no member named 'depth').
mutate_fixture_3() {
    snapshot_file_if_new "$BUILDER_STATE_C_H"
    perl -i -pe '
        if (/^\s*int depth;\s*\/\*/ && !/depth_phx_w45_drift/) {
            s/\bdepth\b/depth_phx_w45_drift/;
        }
    ' "$BUILDER_STATE_C_H"
    TOUCHED_FILES+=("$BUILDER_STATE_C_H")
}
verify_fixture_3() {
    grep -q "depth_phx_w45_drift" "$BUILDER_STATE_C_H"
}
expected_site_fixture_3="builder_state_c"

# Fixture 4: phx_hir_builder_state return-type change in
# builder_state_c.h. Changes return type from `PhxHirBuilderState *` to
# `int`; the C-side callers in builder_emit_c.c dereference the result
# (`->block_map_phx`), so an int return type fails compilation. Replaces
# the prior fixture-4 (block_map_blocks_lookup_cpp) which targeted a
# bridge deleted by Tier 8 SECOND-PILOT Phase A (theologian 10:25:08Z +
# supervisor 10:29:05Z).
mutate_fixture_4() {
    snapshot_file_if_new "$BUILDER_STATE_C_H"
    perl -i -0777 -pe \
        's/PhxHirBuilderState \*phx_hir_builder_state\(/int phx_hir_builder_state(/g' \
        "$BUILDER_STATE_C_H"
    TOUCHED_FILES+=("$BUILDER_STATE_C_H")
}
verify_fixture_4() {
    grep -q "^int phx_hir_builder_state" "$BUILDER_STATE_C_H"
}
expected_site_fixture_4="builder_emit_c.c"

# --- Fixture runner --------------------------------------------------------

FIXTURES=(
    "1|BEFORE_ASYNC_WITH opcode-derivation (Phase 1 #6 emitBeforeWith)"
    "2|_Py_ID identifier-derivation (Phase 1 #7 emitSetupWith body)"
    "3|ExceptionTableEntry depth field rename (Phase 3 Batch 2)"
    "4|phx_hir_builder_state return-type PhxHirBuilderState*->int (Tier 8 SECOND-PILOT)"
)

PASS=0
FAIL=0
SKIP=0

CMAKE_BUILD_DIR="Python/jit_build/build"

for fixture in "${FIXTURES[@]}"; do
    NUM="${fixture%%|*}"
    DESC="${fixture#*|}"

    echo "--- Fixture $NUM: $DESC"

    # Defensive restore + reset touched-files between fixtures.
    restore_files
    TOUCHED_FILES=()

    "mutate_fixture_$NUM"

    if ! "verify_fixture_$NUM"; then
        echo "    [SKIP] mutation did not apply (sentinel marker not found)"
        SKIP=$((SKIP+1))
        echo ""
        continue
    fi

    EXPECTED_VAR="expected_site_fixture_$NUM"
    EXPECTED_SITE="${!EXPECTED_VAR}"

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "    [DRY] mutation applied; touched files:"
        printf '      %s\n' "${TOUCHED_FILES[@]}"
        echo "    expected build-fail at: $EXPECTED_SITE"
        SKIP=$((SKIP+1))
        echo ""
        continue
    fi

    BUILD_OUT=$(mktemp /tmp/w45_35_build.XXXXXX)
    BUILD_RC=0
    cmake --build "$CMAKE_BUILD_DIR" --target jit -j8 > "$BUILD_OUT" 2>&1 || BUILD_RC=$?

    if [ "$BUILD_RC" -eq 0 ]; then
        echo "    [FAIL] build PASSED with mutated derivation — DRIFT UNDETECTED"
        FAIL=$((FAIL+1))
        if [ "$VERBOSE" -eq 1 ]; then
            tail -20 "$BUILD_OUT" | sed 's/^/      build: /'
        fi
    else
        if grep -q "$EXPECTED_SITE" "$BUILD_OUT"; then
            echo "    [PASS] build failed at $EXPECTED_SITE (drift caught)"
            PASS=$((PASS+1))
        else
            echo "    [FAIL] build failed but NOT at expected $EXPECTED_SITE"
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
restore_files
TOUCHED_FILES=()

if [ "$DRY_RUN" -eq 0 ]; then
    BUILD_OUT=$(mktemp /tmp/w45_35_build_restore.XXXXXX)
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

echo "=== §3.5 Verdict ==="
echo "Fixtures: ${#FIXTURES[@]}  PASS: $PASS  FAIL: $FAIL  SKIP: $SKIP"

if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "GATE FAIL: $FAIL fixture(s) silently passed build despite derivation"
    echo "mutation. The derived-constant / field-layout / return-type that"
    echo "C body assumes is NOT being protected by the toolchain."
    [ "$STRICT" -eq 1 ] && exit 1
    exit 0
fi

if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY-RUN COMPLETE: $SKIP fixture(s) staged. Re-run without --dry-run to verify build behavior."
    exit 0
fi

echo "GATE PASS: all $PASS fixture(s) caught at build time. Derivation-drift surface protected."
exit 0
