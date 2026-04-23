#!/bin/bash
# check_do_not_use_callers.sh — gate against production-path callers of
# header-marked DO-NOT-USE factories.
#
# Catches the class:
#   /* WARNING: DO NOT USE for production — bypass issue. */
#   static inline void *hir_c_create_branch(...) { ... }
# being called from production code (e.g., builder_emit_c.c) instead
# of the C++-bridged hir_c_create_branch_cpp variant.
#
# Filed per W44 (D-1776973262, supervisor 19:37:05Z + librarian
# 19:36:45Z): the Edge::set_to bypass class is now 2 incidents
# (06e2ecb652→61c319ca49 + 678e9905a8 W22 cluster); header warning
# alone is insufficient — gate closes the loop.
#
# Symmetric to scripts/caller_grep.sh + scripts/numstat_bundle.sh
# (mechanizes a class supervisor previously caught only via post-hoc
# librarian audit).
#
# Usage:
#   scripts/check_do_not_use_callers.sh             # default scope
#   scripts/check_do_not_use_callers.sh --files     # show file:line per match
#   scripts/check_do_not_use_callers.sh --strict    # exit 1 on any match
#
# Behavior:
#   - Scans Python/jit/hir/*.h for "DO NOT USE" warnings preceding
#     static inline factory definitions.
#   - For each marked symbol, greps Python/ Modules/ Programs/ for
#     callers (excluding the defining header itself + verify/test
#     files where read-path testing is documented as legitimate).
#   - Reports any callers found as VIOLATIONS.
#
# Exit codes:
#   0 — no production callers found (gate clean)
#   1 — at least 1 production caller found (--strict) OR script error

set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

STRICT=0
SHOW_FILES=0
for arg in "$@"; do
    case "$arg" in
        --strict) STRICT=1 ;;
        --files)  SHOW_FILES=1 ;;
        *) echo "ERROR: unknown arg '$arg'" >&2
           echo "Usage: $0 [--strict] [--files]" >&2
           exit 1 ;;
    esac
done

# Headers to scan for DO-NOT-USE markers.
HEADER_GLOB="Python/jit/hir/*.h"

# Production search paths (excludes verify .cpp files where legitimate
# read-path testing happens, and the defining headers themselves).
SEARCH_PATHS="Python/jit Python/cinderx/Jit Modules Programs"

# Files where DO-NOT-USE callers are LEGITIMATE (read-path tests, layout
# verification, etc.). Add new entries with a brief justification.
ALLOW_LIST_REGEX='/(hir_instr_c_verify\.cpp|test_.*\.c|.*_test\.c|.*_verify.*)$'

echo "=== check_do_not_use_callers.sh — W44 gate ==="
echo "HEAD: $(git rev-parse HEAD)"
echo "Header scope: $HEADER_GLOB"
echo "Production search: $SEARCH_PATHS"
echo "Allow-list (legitimate non-prod callers): $ALLOW_LIST_REGEX"
echo ""

# Step 1: enumerate DO-NOT-USE markers in headers + extract symbol names.
# Strategy: find lines matching "DO NOT USE", then look ahead in the
# same file for the next `static inline ... <name>(...)` definition.
echo "=== Step 1: enumerate DO-NOT-USE-marked factories ==="
MARKERS_TMP=$(mktemp)
trap "rm -f $MARKERS_TMP" EXIT

for header in $HEADER_GLOB; do
    [ -f "$header" ] || continue
    # awk: when we see "DO NOT USE", remember and look for next factory
    # def. Match `static inline <ret> <name>(` patterns.
    awk -v hdr="$header" '
        /DO NOT USE/ { warned=1; warn_line=NR; next }
        warned && /^static inline / {
            # Extract symbol name: tokens are "static" "inline" "<ret>"
            # ... "<name>(...)"; the name precedes "(".
            line=$0
            # Find "(" position.
            paren=index(line, "(")
            if (paren > 0) {
                pre = substr(line, 1, paren - 1)
                # Take last whitespace-separated token as name.
                n = split(pre, parts, /[ \t*]+/)
                name = parts[n]
                # Strip leading "*" if present (pointer-return type).
                sub(/^\*+/, "", name)
                if (name != "") {
                    print hdr ":" warn_line "\t" name
                }
            }
            warned=0
        }
        warned && /^\}/ { warned=0 }
    ' "$header" >> "$MARKERS_TMP"
done

N_MARKERS=$(wc -l < "$MARKERS_TMP")
echo "Found $N_MARKERS DO-NOT-USE-marked factory(s):"
if [ "$N_MARKERS" -eq 0 ]; then
    echo "  (none)"
    echo ""
    echo "=== Verdict ==="
    echo "GATE PASS: no DO-NOT-USE markers found to enforce against."
    exit 0
fi
cat "$MARKERS_TMP" | awk -F'\t' '{print "  " $1 "  symbol=" $2}'
echo ""

# Step 2: for each marked symbol, grep production paths for callers.
echo "=== Step 2: search production callers of each marked symbol ==="
echo ""

TOTAL_VIOLATIONS=0
TOTAL_SYMBOLS_WITH_VIOLATIONS=0

while IFS=$'\t' read -r marker_loc symbol; do
    # Caller pattern: "<symbol>(" anywhere — excludes "<symbol>_cpp("
    # via word boundary check.
    PATTERN="\\b${symbol}\\("
    set +e
    CALLERS=$(grep -rnE \
        --include='*.c' --include='*.cpp' \
        "$PATTERN" $SEARCH_PATHS 2>/dev/null || true)
    set -e

    # Filter out allow-listed files (verify, tests).
    CALLERS_FILTERED=$(printf '%s\n' "$CALLERS" | grep -vE "$ALLOW_LIST_REGEX" || true)

    if [ -z "$CALLERS_FILTERED" ]; then
        echo "  [OK]  $symbol — no production callers"
        continue
    fi

    N=$(printf '%s\n' "$CALLERS_FILTERED" | wc -l)
    TOTAL_VIOLATIONS=$((TOTAL_VIOLATIONS + N))
    TOTAL_SYMBOLS_WITH_VIOLATIONS=$((TOTAL_SYMBOLS_WITH_VIOLATIONS + 1))

    echo "  [VIOLATION]  $symbol — $N production caller(s):"
    if [ "$SHOW_FILES" -eq 1 ]; then
        printf '%s\n' "$CALLERS_FILTERED" | sed 's/^/      /'
    else
        printf '%s\n' "$CALLERS_FILTERED" | cut -d: -f1 | sort -u | sed 's/^/      /'
    fi
done < "$MARKERS_TMP"

echo ""
echo "=== Verdict ==="
if [ "$TOTAL_VIOLATIONS" -eq 0 ]; then
    echo "GATE PASS: $N_MARKERS marker(s) audited, zero production callers found."
    exit 0
else
    echo "GATE FAIL: $TOTAL_VIOLATIONS production caller(s) of $TOTAL_SYMBOLS_WITH_VIOLATIONS DO-NOT-USE symbol(s)."
    echo ""
    echo "Each violation must be fixed by switching the caller to the"
    echo "_cpp-suffixed bridge variant (or other documented replacement)."
    echo "If a caller is legitimate non-production (read-path test,"
    echo "layout verification), add the file to ALLOW_LIST_REGEX above."
    [ "$STRICT" -eq 1 ] && exit 1
    exit 0
fi
