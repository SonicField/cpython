#!/bin/bash
# Push phoenix-asm-integration branch to SonicField/cpython.
#
# In-tree canonical version. Lives at cpython/scripts/push.sh so that
# fresh clones inherit the structural-fix without a per-machine setup
# step (gatekeeper 2026-05-06 10:16Z + pythia #289 + librarian
# 15:33:57Z noted the prior /data/users/alexturner/phoenix/push.sh
# was scope-local-only; D-1776902911 numstat_bundle.sh established
# the in-tree wrapper precedent).
#
# Usage:
#   scripts/push.sh             # push HEAD (the safe default)
#   scripts/push.sh COMMIT      # push only if COMMIT resolves to HEAD
#                               # (degenerate-but-permitted check); else REFUSE
#
# Why the COMMIT-arg path is restricted to "must equal HEAD":
#   `git push origin COMMIT:phoenix-asm-integration` fast-forwards EVERYTHING
#   reachable from COMMIT onto the remote, NOT just the named commit. The
#   ancestor-sweep footgun has fired three times:
#     2026-04-16 D-1776242717 — first occurrence + protocol codified
#     2026-04-30 feedback_verify_head_before_push — second occurrence + memory
#     2026-05-05 D-1777993894 — third occurrence pushed 9 unauthorized commits
#                               (5.A3 c0..c7 + 5.B spec) during in-flight gate
#
# Per supervisor 18:14:38Z code-fix > codify directive (D-1778004893): refuse
# COMMIT != HEAD, point users at git-direct invocation if they truly intended
# to fast-forward to a non-HEAD commit.
#
# To deliberately push up to a non-HEAD commit (sweeping any unpushed
# ancestors), invoke git directly with full intent:
#   https_proxy=http://fwdproxy:8080 git push origin <hash>:phoenix-asm-integration
#
# To push a single non-HEAD commit without its ancestors, cherry-pick onto a
# temporary branch based on origin's tip and push that branch.

# Resolve to repo root regardless of caller cwd. scripts/push.sh lives at
# <repo>/scripts/push.sh, so .. from the script's directory is the repo root.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

if [ -n "$1" ]; then
    REQUESTED=$(/usr/bin/git rev-parse "$1" 2>/dev/null)
    if [ -z "$REQUESTED" ]; then
        echo "ERROR: cannot resolve commit '$1' in $(pwd)"
        exit 4
    fi
    HEAD_SHA=$(/usr/bin/git rev-parse HEAD)
    if [ "$REQUESTED" != "$HEAD_SHA" ]; then
        echo "ERROR: refusing partial-push."
        echo "  requested COMMIT='$1' resolves to $REQUESTED"
        echo "  HEAD is                            $HEAD_SHA"
        echo ""
        echo "  push.sh COMMIT-arg fast-forwards EVERYTHING reachable from"
        echo "  COMMIT, NOT just COMMIT. This footgun has fired 3 times"
        echo "  (D-1776242717, feedback_verify_head_before_push, D-1777993894)."
        echo ""
        echo "  - To push only HEAD ($HEAD_SHA), run scripts/push.sh with no args."
        echo "  - To deliberately push up to '$1' (sweeping any unpushed"
        echo "    ancestors), invoke git directly with full intent:"
        echo "      https_proxy=http://fwdproxy:8080 git push origin \\"
        echo "          $REQUESTED:phoenix-asm-integration"
        echo "  - To push a single non-HEAD commit without its ancestors,"
        echo "    cherry-pick onto a branch based on origin's tip and push."
        exit 1
    fi
    echo "Pushing HEAD ($HEAD_SHA, resolved from arg '$1')"
fi

https_proxy=http://fwdproxy:8080 /usr/bin/git push origin phoenix-asm-integration 2>&1
echo "PUSH_EXIT=$?"
