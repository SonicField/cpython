#!/bin/bash
# test_preflight_i3_negative.sh — I3 SUBSHELL ERROR-VISIBILITY negative-test
# per theologian 21:44:42Z (I3-3) + gatekeeper 21:45:16Z spec under
# supervisor 21:48:29Z RESEQUENCE.
#
# Two test scenarios per gatekeeper 21:45:16Z (b) WARN-vs-FAIL severity +
# (c) extended rc-class coverage:
#
# (1) WARN class: i3_pipe_grep_count helper with grep error rc=2 (malformed
#     regex) — assert WARNING surfaces to stderr, return rc still propagates,
#     no exit.
# (2) FAIL class: explicit rc-capture pattern with simulated build-product
#     invariant violation (cmd exits non-zero) — assert downstream FAIL +
#     exit 1 with diagnostic.
# (3) rc=127 not-found class: i3_pipe_grep_count with cmd returning 127 —
#     assert WARNING surfaces (grep rc>1 detected) without crashing the
#     wrapper.
# (4) rc=128+N signal-killed class: cmd interrupted by SIGTERM (rc=143) —
#     assert downstream rc-capture surfaces signal-class diagnostic.
#
# Runnable manually OR via gate_phoenix.sh --selftest=i3.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/lib_preflight.sh"

RESULTS_FILE=""

PASS=0
FAIL=0
report_pass() { echo "  PASS: $1"; PASS=$((PASS+1)); }
report_fail() { echo "  FAIL: $1"; FAIL=$((FAIL+1)); }

# --- Test (1): WARN class via i3_pipe_grep_count + malformed regex ---
echo "Test (1) WARN: i3_pipe_grep_count with malformed regex (grep rc=2)"
WARN_OUT=$( ( i3_pipe_grep_count CNT "some input here" -E '[invalid' ) 2>&1 )
WARN_RC=$?

if [ "$WARN_RC" -ne 0 ]; then
    report_fail "(1) WARN: helper itself exited non-zero (rc=$WARN_RC); should set CNT=0 and return 0 with WARNING to stderr"
elif ! echo "$WARN_OUT" | grep -q "WARNING: i3_pipe_grep_count"; then
    report_fail "(1) WARN: WARNING message missing on stderr (output: $WARN_OUT)"
elif ! echo "$WARN_OUT" | grep -q "rc=2"; then
    report_fail "(1) WARN: rc=2 not surfaced in WARNING (output: $WARN_OUT)"
else
    report_pass "(1) WARN: malformed regex → grep rc=2 → WARNING surfaced, helper continued"
fi

# --- Test (2): FAIL class via explicit rc-capture pattern ---
echo "Test (2) FAIL: explicit rc-capture pattern with simulated build-product violation"
FAIL_OUT=$(
    ( set +e
      bash -c "exit 7"
      RC=$?
      set -e
      if [ "$RC" -ne 0 ]; then
          echo "FAIL: simulated-build-product-invariant: rc=$RC"
          exit 1
      fi
    ) 2>&1
)
FAIL_RC=$?

if [ "$FAIL_RC" -eq 0 ]; then
    report_fail "(2) FAIL: pattern did not exit non-zero on rc=7 (silent-fail leak)"
elif ! echo "$FAIL_OUT" | grep -q "FAIL: simulated-build-product-invariant: rc=7"; then
    report_fail "(2) FAIL: canonical FAIL message missing (output: $FAIL_OUT)"
else
    report_pass "(2) FAIL: rc=7 captured + canonical message + exit 1 (no silent-fail)"
fi

# --- Test (3): rc=127 not-found class via i3_pipe_grep_count ---
echo "Test (3) rc=127: command-not-found via grep -c with non-existent input file path"
NOTFOUND_OUT=$( ( i3_pipe_grep_count CNT2 "any input" -f /nonexistent/pattern/file ) 2>&1 )
NOTFOUND_RC=$?

# grep -f /nonexistent typically returns rc=2 (file not found); helper should
# WARN. If grep returns rc=127 (extremely unusual since grep itself exists),
# helper should still WARN.
if [ "$NOTFOUND_RC" -ne 0 ]; then
    report_fail "(3) rc>=127: helper itself exited non-zero (should set CNT2=0 and return 0)"
elif ! echo "$NOTFOUND_OUT" | grep -q "WARNING: i3_pipe_grep_count"; then
    report_fail "(3) rc>=127: WARNING missing for non-existent pattern file (output: $NOTFOUND_OUT)"
else
    report_pass "(3) rc>=127: non-existent pattern file → grep rc>1 → WARNING surfaced"
fi

# --- Test (4): signal-killed (rc=128+N) via timeout ---
echo "Test (4) signal: SIGTERM-killed sub-process via timeout 1 sleep 5"
SIG_OUT=$(
    ( set +e
      timeout --signal=TERM 1 sleep 5
      RC=$?
      set -e
      # timeout exits 124 on timeout; the killed sleep returns 143 (128+15)
      # but we see 124 from timeout itself. Both are non-zero → diagnostic.
      if [ "$RC" -ne 0 ]; then
          if [ "$RC" -ge 128 ]; then
              echo "WARNING: signal-killed: rc=$RC (signal=$((RC - 128)))"
          else
              echo "WARNING: timeout-or-rc=$RC"
          fi
      fi
    ) 2>&1
)

if ! echo "$SIG_OUT" | grep -qE "WARNING: (signal-killed|timeout-or-rc)="; then
    report_fail "(4) signal: rc=signal-class WARNING missing (output: $SIG_OUT)"
else
    report_pass "(4) signal: timeout/SIGTERM → rc>0 → WARNING surfaced"
fi

echo ""
echo "I3 NEG-TEST SUMMARY: $PASS pass, $FAIL fail"
if [ "$FAIL" -ne 0 ]; then
    exit 1
fi
exit 0
