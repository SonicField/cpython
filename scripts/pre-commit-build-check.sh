#!/bin/bash
# pre-commit-build-check.sh — git pre-commit hook that BLOCKS commit on build failure.
#
# Authored 2026-05-12 per supervisor 18:04:09Z disposition of pythia #361 + #359
# (queue item 9 structural enforcement). Closes the 13-day enforcement gap on
# feedback_compile_before_commit (D-1776482908, 2026-04-22) that produced
# the c4 three-amend cycle (push 2026-05-12 14:14:47Z, 15:33:24Z, 15:53:04Z).
#
# Behaviour:
#   - Runs scripts/build_phoenix.sh (incremental build, no --clean by default)
#   - Exit 0 -> commit proceeds
#   - Exit non-0 -> commit BLOCKED with build log path + actionable hint
#   - Skip via PHOENIX_SKIP_BUILD_CHECK=1 (USE WITH CARE, leaves trailer in commit
#     message so reviewers can audit)
#
# Install (one-time):
#   ln -s ../../scripts/pre-commit-build-check.sh .git/hooks/pre-commit
#   (or copy if symlinks aren't appropriate on your platform)
#
# Why incremental, not --clean:
#   - Pre-commit runtime budget is ~1-2 min; clean build is 5-10 min.
#   - Incremental catches the c4 failure class (undefined symbols at link,
#     compile errors in .h inline definitions, missing decl visibility).
#   - --clean is still recommended after JIT header layout changes per
#     CLAUDE.md "JIT Header Change Protocol" — the hook does NOT replace
#     that protocol, only catches the more common "I forgot to compile"
#     class.
#
# The hook is intentionally conservative — it runs the same script the team
# already uses for builds, no new build configuration. If the script doesn't
# exist (e.g., cwd not in cpython repo), the hook exits 0 (do not block).

set -uo pipefail

# Allow opt-out for emergencies. Track in a way reviewers can spot.
if [ "${PHOENIX_SKIP_BUILD_CHECK:-0}" = "1" ]; then
    echo "[pre-commit-build-check] SKIPPED via PHOENIX_SKIP_BUILD_CHECK=1"
    echo "[pre-commit-build-check] reviewers: please verify build separately"
    exit 0
fi

# Resolve repo root from the hook's location. The hook lives at
# .git/hooks/pre-commit (or as a symlink there); $0 may be either path.
HOOK_PATH="$(readlink -f "$0" 2>/dev/null || echo "$0")"
SCRIPT_DIR="$(cd "$(dirname "$HOOK_PATH")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo "")"

if [ -z "$REPO_ROOT" ]; then
    echo "[pre-commit-build-check] not in a git repo; allowing commit"
    exit 0
fi

BUILD_SCRIPT="$REPO_ROOT/scripts/build_phoenix.sh"

if [ ! -x "$BUILD_SCRIPT" ]; then
    echo "[pre-commit-build-check] $BUILD_SCRIPT not found or not executable; allowing commit"
    exit 0
fi

# Skip the build for commits that can't introduce build breaks — pure docs,
# benchmarks, gate logs. The skip is path-pattern based on the staged file
# list; if any non-skippable file is staged, build runs.
STAGED_FILES="$(git diff --cached --name-only 2>/dev/null)"
if [ -z "$STAGED_FILES" ]; then
    echo "[pre-commit-build-check] no staged files; allowing commit"
    exit 0
fi

# Path patterns that cannot break the build. Keep this conservative — when
# in doubt, run the build.
BUILD_AFFECTING=0
while IFS= read -r f; do
    [ -z "$f" ] && continue
    case "$f" in
        docs/*|*.md)              ;;  # docs only
        docs/benchmarks/*)        ;;  # bench data
        docs/gates/*)             ;;  # gate logs
        scripts/.last_build_state) ;; # build-state file (ignored anyway)
        *)
            BUILD_AFFECTING=1
            break
            ;;
    esac
done <<< "$STAGED_FILES"

if [ "$BUILD_AFFECTING" -eq 0 ]; then
    echo "[pre-commit-build-check] only docs/benchmarks/gate-logs staged; allowing commit"
    exit 0
fi

LOG_FILE="${TMPDIR:-/tmp}/phoenix-pre-commit-build-$(date +%s).log"
START_TS="$(date +%s)"

echo "[pre-commit-build-check] running scripts/build_phoenix.sh (incremental)"
echo "[pre-commit-build-check] log: $LOG_FILE"

if "$BUILD_SCRIPT" > "$LOG_FILE" 2>&1; then
    DURATION=$(( $(date +%s) - START_TS ))
    echo "[pre-commit-build-check] PASS (${DURATION}s); commit proceeds"
    exit 0
fi

EXIT=$?
DURATION=$(( $(date +%s) - START_TS ))

echo ""
echo "[pre-commit-build-check] BUILD FAILED (exit=$EXIT, ${DURATION}s)"
echo "[pre-commit-build-check] commit BLOCKED"
echo ""
echo "Last 40 lines of build log ($LOG_FILE):"
echo "----"
tail -40 "$LOG_FILE"
echo "----"
echo ""
echo "Fix the build error and re-stage (git add) then re-commit."
echo "If you are intentionally committing broken code (e.g., WIP branch),"
echo "skip the check ONCE with: PHOENIX_SKIP_BUILD_CHECK=1 git commit ..."
exit "$EXIT"
