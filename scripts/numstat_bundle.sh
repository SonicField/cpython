#!/bin/bash
# numstat_bundle.sh — print verbatim cumulative bundle stats for a push range.
#
# Usage:
#   scripts/numstat_bundle.sh <push-base-commit>   # prints "<base>..HEAD" stats
#   scripts/numstat_bundle.sh                      # auto-detects last remote
#                                                  # commit on phoenix-asm-integration
#
# Prevents the per-commit-sum count-discrepancy class (4 incidents in
# 2026-04-22/23 sessions, each caught by gatekeeper or medic post-hoc).
# Per-commit `git diff --numstat | sum` over-counts when commits modify
# the same lines (replacement overlap nets out under cumulative diff).
#
# Per memory feedback_re_numstat_after_bundle_change.md + supervisor L2502
# auto-numstat-script directive.

set -e

cd /data/users/alexturner/phoenix/cpython

if [ -z "$1" ]; then
    echo "ERROR: pass <push-base-commit> as arg." >&2
    echo "Usage: $0 <push-base-commit>" >&2
    echo "" >&2
    echo "The push-base is the last-pushed-to-remote commit on the branch" >&2
    echo "before the bundle started. Typically the commit hash from the" >&2
    echo "previous push status post in chat (e.g., '<base>..<HEAD>')." >&2
    echo "" >&2
    echo "Auto-detect via origin/<branch> is unreliable here because the" >&2
    echo "fwdproxy push path does not fetch refs back to the local repo." >&2
    exit 1
fi
PUSH_BASE="$1"

if ! git rev-parse "$PUSH_BASE" >/dev/null 2>&1; then
    echo "ERROR: '$PUSH_BASE' is not a valid commit." >&2
    exit 1
fi

PUSH_BASE_SHORT=$(git rev-parse --short=10 "$PUSH_BASE")
HEAD_SHORT=$(git rev-parse --short=10 HEAD)
N_COMMITS=$(git rev-list --count "${PUSH_BASE}..HEAD")

if [ "$N_COMMITS" -eq 0 ]; then
    echo "Bundle is empty (no commits between $PUSH_BASE_SHORT and HEAD)."
    exit 0
fi

echo "=== Bundle aggregate (paste verbatim into chat) ==="
echo "Range: ${PUSH_BASE_SHORT}..${HEAD_SHORT} (${N_COMMITS} commits)"
echo "git diff --shortstat ${PUSH_BASE_SHORT}..HEAD:"
git diff --shortstat "${PUSH_BASE}..HEAD"
echo ""
echo "Per-commit (sanity check, not for chat aggregate quote):"
git log --oneline --shortstat "${PUSH_BASE}..HEAD"
