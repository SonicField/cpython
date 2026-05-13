#!/bin/bash
# test_preflight_i2_negative.sh — I2 ENV-FLAG READ-BACK negative-test per
# theologian 21:21:28Z (I2-3) + gatekeeper 21:22:00Z spec under supervisor
# 21:22:52Z RESEQUENCE.
#
# Procedure:
#   1. Stub a cmake-invocation log that does NOT contain a deliberately-
#      named flag (simulates B1 hard-reset class — caller passed flag,
#      script silently dropped it, log doesn't have it).
#   2. Invoke preflight_check_env_flag(EXPECTED, STUB_LOG) in subshell;
#      assert exit non-zero AND output contains 'I2 ENV-FLAG READ-BACK'.
#   3. POSITIVE-CONTROL: stub a SECOND log that DOES contain the flag;
#      assert preflight_check_env_flag returns 0 (no false-FAIL on
#      legitimate cases).
#   4. Cleanup: rm temp logs (no git state mutation).
#
# Runnable manually OR via gate_phoenix.sh --selftest=i2.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/lib_preflight.sh"

CPYTHON_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_FILE=""

# Negative scenario: log is missing the deliberately-named flag.
NEG_LOG=$(mktemp /tmp/i2_neg_log_XXXXXX)
cat > "$NEG_LOG" <<'EOF'
-- The C compiler identification is GNU 12.2.0
-- Configuring done
-- Generating done
-- Build files have been written to: /tmp/build
-- (this stub log simulates B1 hard-reset class: caller set
--  EXTRA_CMAKE_FLAGS but build_phoenix.sh dropped the value
--  before passing to cmake; the flag must NOT appear anywhere
--  in this stub or the grep will false-positive.)
EOF

# Positive scenario: log contains the expected flag.
POS_LOG=$(mktemp /tmp/i2_pos_log_XXXXXX)
cat > "$POS_LOG" <<'EOF'
-- The C compiler identification is GNU 12.2.0
-- CMAKE_CXX_FLAGS: -DPHOENIX_ASM -DDELIBERATELY_MISNAMED_FLAG=1
-- Configuring done
-- Generating done
-- Build files have been written to: /tmp/build
EOF

trap 'rm -f "$NEG_LOG" "$POS_LOG"' EXIT

EXPECTED=" -DDELIBERATELY_MISNAMED_FLAG=1"

# --- Negative test: must FAIL on missing flag. ---
NEG_OUT=$( ( preflight_check_env_flag "$EXPECTED" "$NEG_LOG" ) 2>&1 )
NEG_RC=$?

if [ "$NEG_RC" -eq 0 ]; then
    echo "I2 NEG-TEST FAIL — staleness not detected (preflight returned 0 on missing flag)"
    echo "  output: $NEG_OUT"
    exit 1
fi

if ! echo "$NEG_OUT" | grep -q "I2 ENV-FLAG READ-BACK"; then
    echo "I2 NEG-TEST FAIL — preflight rejected but did not emit canonical 'I2 ENV-FLAG READ-BACK' message"
    echo "  output: $NEG_OUT"
    exit 1
fi

# --- Positive control: must PASS when flag is present. ---
POS_OUT=$( ( preflight_check_env_flag "$EXPECTED" "$POS_LOG" ) 2>&1 )
POS_RC=$?

if [ "$POS_RC" -ne 0 ]; then
    echo "I2 NEG-TEST FAIL — false-positive: preflight rejected legitimate (flag present) log"
    echo "  output: $POS_OUT"
    exit 1
fi

echo "I2 NEG-TEST PASS — staleness detected on missing flag; positive control PASSED"
echo "  neg_rc=$NEG_RC neg_msg: $(echo "$NEG_OUT" | grep 'I2 ENV-FLAG READ-BACK' | head -1)"
echo "  pos_rc=$POS_RC pos_msg: $(echo "$POS_OUT" | grep 'I2 ENV-FLAG READ-BACK' | head -1)"
exit 0
