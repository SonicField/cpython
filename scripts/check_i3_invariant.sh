#!/bin/bash
# check_i3_invariant.sh — W-I3-RUNTIME-ASSERT (IV) gate. Detects
# id-mutation patterns on BasicBlock-class objects that would violate
# Tier 8 SECOND-PILOT Phase B's I3 invariant: BasicBlock::id is
# allocation-monotonic AND never mutated post-allocation.
#
# I3 underpins PhxBcBlockArray's dense-array O(1) lookup at
# Python/jit/hir/builder_state_c.h:298 (phx_bc_block_array_at). Any
# future HIR pass that renumbers ids (SSA-destruction-style cache
# locality renumbering, peephole block-splitter, speculative-inlining
# clone) would silently corrupt the lookup.
#
# Per W-I3-RUNTIME-ASSERT spec docs/w-i3-runtime-assert-spec.md §2 (IV)
# + §7 (theologian 21:57:02Z + librarian 22:05:19Z framework guidance):
# this script is the (IV) layer. (III) JIT_DCHECK pydebug sentinel
# lives in builder_state_c.h's PhxBcBlockEntry.
#
# Usage:
#   scripts/check_i3_invariant.sh             # default scope (warn-only)
#   scripts/check_i3_invariant.sh --files     # show file:line per match
#   scripts/check_i3_invariant.sh --strict    # exit 1 on any violation
#
# Behavior:
#   - Greps Python/jit/hir/ for id-mutation patterns:
#       <var>->id = ...    (pointer-style)
#       <var>.id = ...     (value-style)
#       set_id(/setId(    (setter methods)
#   - Filters via ALLOW_LIST to exclude:
#       - HirLoadAttrSpecial::id (void* attribute-id field, not BasicBlock)
#       - Test/verify files (read-path testing per W44 precedent)
#       - The BasicBlock ctor itself (initializer-list, not mutation)
#   - Reports any remaining matches as VIOLATIONS.
#
# Exit codes:
#   0 — no violations (gate clean)
#   1 — at least 1 violation (--strict) OR script error

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

SEARCH_PATHS="Python/jit/hir"

# Allow-list patterns (file or file:line) for known-OK matches.
# Each entry should be commented with WHY it is legitimate.
#
# - hir_c_api.cpp 'l->id = id' on HirLoadAttrSpecial: the .id field is
#   a void* attribute identifier on HirLoadAttrSpecial, NOT a
#   BasicBlock id. Different class, different field semantics.
# - hir.h 'next_register_id_' / 'cache_id_': not BasicBlock id; trailing
#   underscore disambiguates. Pattern uses \bid\b so these won't match.
# - hir.h 'BasicBlock(int id_) : id(id_)': initializer-list assignment
#   inside the ctor, NOT a post-allocation mutation. Pattern excludes
#   ': id(' constructor-init form.
# - test_phx_block_map.c, hir_instr_c_verify.cpp: tests/verify files
#   exercising read-path. Per W44 ALLOW_LIST precedent.
ALLOW_LIST_FILES_REGEX='/(test_.*\.c|.*_test\.c|hir_instr_c_verify\.cpp)$'

# Allow-list specific symbols whose ".id" or "->id" mutation is
# unrelated to BasicBlock::id. Add new entries with justification.
# Each entry is a regex matching the FULL grep line.
ALLOW_LIST_LINES_REGEX='HirLoadAttrSpecial|l->id\s*=\s*id;|next_register_id_|cache_id_'

echo "=== check_i3_invariant.sh — W-I3-RUNTIME-ASSERT (IV) gate ==="
echo "HEAD: $(git rev-parse HEAD)"
echo "Search scope: $SEARCH_PATHS"
echo "Allow-list files: $ALLOW_LIST_FILES_REGEX"
echo "Allow-list lines: $ALLOW_LIST_LINES_REGEX"
echo ""

# Patterns to detect (all id-mutation forms targeting a presumed-BasicBlock):
#   ->id = ...    (single = followed by non-= to exclude == comparisons)
#   .id = ...     (same; but more permissive — many false positives, hence
#                  ALLOW_LIST filtering)
#   set_id(       (setter-method form)
#   setId(        (camelCase setter)
PATTERNS_PTR='->\s*id\s*=\s*[^=]'
PATTERNS_VAL='\.\s*id\s*=\s*[^=]'
PATTERNS_SET='\b(set_id|setId)\s*\('

echo "=== Step 1: enumerate id-mutation candidates in $SEARCH_PATHS ==="
ALL_TMP=$(mktemp)
trap "rm -f $ALL_TMP" EXIT

set +e
{
    grep -rnE -e "$PATTERNS_PTR" --include='*.c' --include='*.cpp' --include='*.h' $SEARCH_PATHS 2>/dev/null
    grep -rnE -e "$PATTERNS_VAL" --include='*.c' --include='*.cpp' --include='*.h' $SEARCH_PATHS 2>/dev/null
    grep -rnE -e "$PATTERNS_SET" --include='*.c' --include='*.cpp' --include='*.h' $SEARCH_PATHS 2>/dev/null
} | sort -u > "$ALL_TMP"
set -e

N_RAW=$(wc -l < "$ALL_TMP")
echo "Raw matches: $N_RAW"

# Filter via allow-lists.
echo ""
echo "=== Step 2: filter via allow-lists ==="
FILTERED_TMP=$(mktemp)
trap "rm -f $ALL_TMP $FILTERED_TMP" EXIT

set +e
grep -vE "$ALLOW_LIST_FILES_REGEX" "$ALL_TMP" | grep -vE "$ALLOW_LIST_LINES_REGEX" > "$FILTERED_TMP"
set -e

N_VIOLATIONS=$(wc -l < "$FILTERED_TMP")
echo "After allow-list filtering: $N_VIOLATIONS violation(s)"
echo ""

if [ "$N_VIOLATIONS" -eq 0 ]; then
    echo "=== Verdict ==="
    echo "GATE PASS: $N_RAW raw matches all in allow-list; zero I3-violation candidates."
    exit 0
fi

echo "=== Violations ==="
if [ "$SHOW_FILES" -eq 1 ]; then
    cat "$FILTERED_TMP" | sed 's/^/  /'
else
    cat "$FILTERED_TMP" | cut -d: -f1-2 | sort -u | sed 's/^/  /'
fi

echo ""
echo "=== Verdict ==="
echo "GATE FAIL: $N_VIOLATIONS candidate I3 violation(s)."
echo ""
echo "I3 invariant: BasicBlock::id is allocation-monotonic AND never"
echo "mutated post-allocation. PhxBcBlockArray's dense-array O(1)"
echo "lookup depends on this. Each violation must be either:"
echo "  - Fixed (remove the id mutation)"
echo "  - Re-architected (drop PhxBcBlockArray's id-indexed shape; see"
echo "    docs/w-i3-runtime-assert-spec.md (II) generation-counter as"
echo "    fallback)"
echo "  - Allow-listed (if NOT actually a BasicBlock::id mutation; add"
echo "    to ALLOW_LIST_LINES_REGEX or ALLOW_LIST_FILES_REGEX with a"
echo "    one-line justification comment)"
[ "$STRICT" -eq 1 ] && exit 1
exit 0
