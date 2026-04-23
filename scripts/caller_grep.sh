#!/bin/bash
# caller_grep.sh — definitive caller search across the whole repo.
#
# Usage:
#   scripts/caller_grep.sh <symbol-or-regex>
#   scripts/caller_grep.sh <symbol-or-regex> --exclude=<regex>
#
# Prevents the head-truncation-fabrication class (theologian L2581 →
# medic L2583 + gatekeeper L2584 caught a 'ZERO instantiation sites'
# false claim that would have broken the build by deleting LIVE
# Pass<T> classes called from compiler.cpp).
#
# Symmetric to scripts/numstat_bundle.sh (mechanizes count-discrepancy
# class). Per pythia #88 + supervisor L2603 mechanization directive.
#
# Discipline:
#   - Searches the WHOLE built source tree (NOT just /Python/jit/).
#   - NO head/tail truncation; full output goes to stdout.
#   - Reports total count + per-file count + verbatim grep cmd, so the
#     caller can paste the result into chat without paraphrase.
#
# Example deletion-candidate workflow:
#   scripts/caller_grep.sh 'Simplify::Factory'
#   scripts/caller_grep.sh '\bRegisterSet\b' --exclude='PhyRegisterSet'
#   scripts/caller_grep.sh 'hir::Simplify\{\}'

set -e

cd /data/users/alexturner/phoenix/cpython

if [ -z "$1" ]; then
    echo "ERROR: pass <symbol-or-regex> as arg." >&2
    echo "Usage: $0 <symbol-or-regex> [--exclude=<regex>]" >&2
    echo "" >&2
    echo "Example: $0 'Simplify::Factory'" >&2
    echo "Example: $0 '\\bRegisterSet\\b' --exclude='PhyRegisterSet'" >&2
    exit 1
fi
PATTERN="$1"
shift

EXCLUDE=""
while [ -n "$1" ]; do
    case "$1" in
        --exclude=*)
            EXCLUDE="${1#--exclude=}"
            shift
            ;;
        *)
            echo "ERROR: unknown arg '$1'" >&2
            exit 1
            ;;
    esac
done

# Built source trees only (skip docs/, .nbs/, build outputs).
# Includes: code (Python/, Modules/, Programs/), headers (Include/),
# stdlib for Python-side callers (Lib/), tooling (Tools/).
SEARCH_PATHS="Python Include Tools Lib Modules Programs"

# Restrict to source files (skip generated artifacts + .o + binaries).
INCLUDES=(--include='*.c' --include='*.cpp' --include='*.h' --include='*.hpp' --include='*.cc' --include='*.hxx')

GREP_CMD=(grep -rnE "${INCLUDES[@]}" "$PATTERN" $SEARCH_PATHS)

echo "=== caller_grep.sh — definitive caller search (no head-truncation) ==="
echo "HEAD: $(git rev-parse HEAD)"
echo "Pattern: $PATTERN"
[ -n "$EXCLUDE" ] && echo "Exclude (post-grep -vE): $EXCLUDE"
echo "Search paths: $SEARCH_PATHS"
echo "Include globs: *.c *.cpp *.h *.hpp *.cc *.hxx"
echo ""
echo "=== grep cmd (verbatim) ==="
printf '%q ' "${GREP_CMD[@]}"
echo
[ -n "$EXCLUDE" ] && printf '  | grep -vE %q\n' "$EXCLUDE"
echo ""

# Execute (allow grep exit 1 = no matches without aborting).
set +e
if [ -n "$EXCLUDE" ]; then
    RESULT=$("${GREP_CMD[@]}" | grep -vE "$EXCLUDE")
else
    RESULT=$("${GREP_CMD[@]}")
fi
set -e

if [ -z "$RESULT" ]; then
    N_MATCHES=0
    N_FILES=0
else
    N_MATCHES=$(printf '%s\n' "$RESULT" | wc -l)
    N_FILES=$(printf '%s\n' "$RESULT" | cut -d: -f1 | sort -u | wc -l)
fi

echo "=== Matches ($N_MATCHES lines across $N_FILES files) ==="
if [ "$N_MATCHES" -gt 0 ]; then
    printf '%s\n' "$RESULT"
fi
echo ""

if [ "$N_FILES" -gt 0 ]; then
    echo "=== Per-file count ==="
    printf '%s\n' "$RESULT" | cut -d: -f1 | sort | uniq -c | sort -rn
    echo ""
fi

echo "=== Verdict ==="
if [ "$N_MATCHES" -eq 0 ]; then
    echo "DEAD-CODE CANDIDATE: zero callers found across $SEARCH_PATHS."
    echo "Caveats:"
    echo "  - Confirm pattern matches the actual symbol (try with/without ::, namespace, ())."
    echo "  - Run again on raw symbol-name-only if --exclude was used."
    echo "  - Search docs/ separately if non-build references matter."
else
    echo "ACTIVE: $N_MATCHES caller(s) across $N_FILES file(s)."
    echo "DO NOT delete without rewiring callers first."
fi
