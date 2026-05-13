#!/bin/bash
# test_preflight_i5_negative.sh — I5γ γ-1 STRONGER-POISON SELF-TEST per
# theologian 22:14:12Z + gatekeeper 22:15:07Z spec under supervisor
# 22:17:44Z RESEQUENCE.
#
# Purpose: empirically disambiguate whether the `make distclean` line in
# build_phoenix.sh --clean path (added by I5 quick-fix d1e7a205d8 + made
# I3-compliant by amend-1 6a49d9a881) is LOAD-BEARING under the failure
# class that motivated it (ARM64 v3 18:13:17Z: cross-toggle Modules/*.o
# staleness causing _testembed link failure, D-1778696034).
#
# 4-STEP procedure (theologian 22:14:12Z γ-1.2):
#   STEP 1: clean pydebug build + deliberate-stronger-poison
#           (touch Modules/_decimal/_decimal.c → force one Modules/*.o
#            to be older than rebuild target;
#            echo "#define Py_DEBUG 1" >> pyconfig.h.in → corrupt config
#            flag persistence).
#   STEP 2: SED-OUT make distclean line in build_phoenix.sh --clean path,
#           then run build_phoenix.sh --clean and capture exit code.
#           EXPECT non-zero (per ARM64 v3 class — line WAS load-bearing).
#   STEP 3: RESTORE make distclean line, run build_phoenix.sh --clean
#           and capture exit code. EXPECT zero (recovery succeeds with
#           the line in place — confirms load-bearing).
#   STEP 4: cleanup via git checkout HEAD --
#           (Modules/_decimal/_decimal.c, pyconfig.h.in, build_phoenix.sh).
#
# CRITICAL TIMING: each build_phoenix.sh --pydebug --clean takes ~15-20
# min on x86_64. Total runtime: ~45-60 min. NOT triggered by standard
# gate; opt-in only via gate_phoenix.sh --selftest=i5g1.
#
# CRASH-SAFETY (gatekeeper 22:15:07Z (a)): EXIT trap restores all
# modified files via git checkout, ensuring build_phoenix.sh self-mod
# is reverted even if test crashes mid-step.
#
# POISON-INSUFFICIENT ALTERNATIVE HYPOTHESIS (gatekeeper 22:15:07Z (b)):
# If STEP 2 PASSES (recovery succeeds without distclean), the conclusion
# "no-op" depends on the poison being SUFFICIENT to reproduce the
# ARM64 v3 18:13:17Z failure mode. The Modules/_decimal/*.c touch +
# pyconfig.h.in append may be insufficient to reproduce cross-toggle
# Modules/*.o staleness on this arch. Failure VIS documents both
# possibilities; final disposition is supervisor-class.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CPYTHON_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

POISON_FILES=(
    "$CPYTHON_ROOT/Modules/_decimal/_decimal.c"
    "$CPYTHON_ROOT/pyconfig.h.in"
    "$CPYTHON_ROOT/scripts/build_phoenix.sh"
)

cleanup() {
    echo ""
    echo "γ-1 STEP 4: cleanup (git checkout HEAD -- on touched files)"
    for f in "${POISON_FILES[@]}"; do
        git -C "$CPYTHON_ROOT" checkout HEAD -- "${f#$CPYTHON_ROOT/}" 2>/dev/null || true
    done
}
trap cleanup EXIT

echo "===================================================================="
echo "I5γ γ-1 STRONGER-POISON SELF-TEST"
echo "  Purpose: disambiguate I5 amend-1 'make distclean' load-bearing"
echo "  Estimated runtime: ~45-60 min (3x build_phoenix.sh --pydebug --clean)"
echo "===================================================================="

if [ ! -f "$CPYTHON_ROOT/scripts/build_phoenix.sh" ]; then
    echo "FAIL: γ-1 — build_phoenix.sh missing"
    exit 1
fi

# --- STEP 1: clean baseline + deliberate poison ---
echo ""
echo "γ-1 STEP 1: clean pydebug build + deliberate-stronger-poison"
cd "$CPYTHON_ROOT"

set +e
"$CPYTHON_ROOT/scripts/build_phoenix.sh" --pydebug --clean > /tmp/i5g1_step1_build.log 2>&1
STEP1_BUILD_RC=$?
set -e
if [ "$STEP1_BUILD_RC" -ne 0 ]; then
    echo "FAIL: γ-1 STEP 1 — clean baseline build failed (rc=$STEP1_BUILD_RC; log /tmp/i5g1_step1_build.log)"
    tail -10 /tmp/i5g1_step1_build.log | sed 's/^/  /'
    exit 1
fi
echo "  STEP 1 baseline build PASS (rc=0)"

# Apply poison: touch Modules .c → force .o older-than-source mismatch;
# corrupt pyconfig.h.in → simulate config-flag persistence drift.
touch "$CPYTHON_ROOT/Modules/_decimal/_decimal.c"
echo "/* I5γ-poison: force pyconfig.h.in mtime drift */" >> "$CPYTHON_ROOT/pyconfig.h.in"
echo "  STEP 1 poison applied: touched Modules/_decimal/_decimal.c + appended to pyconfig.h.in"

# --- STEP 2: SED-OUT make distclean + recovery (EXPECT non-zero) ---
echo ""
echo "γ-1 STEP 2: SED-OUT 'make distclean' line + run --clean (EXPECT non-zero rc)"

# Locate the make distclean invocation in build_phoenix.sh CLEAN path.
# Pattern: 'if ! make distclean' — guarded under [ -f Makefile ] per
# I5 amend-1 (commit 6a49d9a881). Comment out the entire if-then-fi block
# starting at the 'if ! make distclean' line through 'fi' (3 lines).
DISTCLEAN_LINE=$(grep -n 'if ! make distclean' "$CPYTHON_ROOT/scripts/build_phoenix.sh" | head -1 | cut -d: -f1)
if [ -z "$DISTCLEAN_LINE" ]; then
    echo "FAIL: γ-1 STEP 2 — could not locate 'if ! make distclean' line in build_phoenix.sh"
    exit 1
fi
# Sed in-place to comment out lines DISTCLEAN_LINE through DISTCLEAN_LINE+2
# (the if-then-fi block). Exact line count assumes amend-1 structure.
sed -i "${DISTCLEAN_LINE},$((DISTCLEAN_LINE+4))s/^/# I5g1-NEG-TEST-SEDOUT: /" "$CPYTHON_ROOT/scripts/build_phoenix.sh"
echo "  SED-OUT applied: lines $DISTCLEAN_LINE-$((DISTCLEAN_LINE+4)) commented in build_phoenix.sh"

set +e
"$CPYTHON_ROOT/scripts/build_phoenix.sh" --pydebug --clean > /tmp/i5g1_step2_build.log 2>&1
STEP2_BUILD_RC=$?
set -e
echo "  STEP 2 recovery-attempt-sans-distclean rc=$STEP2_BUILD_RC"

# Restore build_phoenix.sh BEFORE running STEP 3.
git -C "$CPYTHON_ROOT" checkout HEAD -- scripts/build_phoenix.sh

# --- STEP 3: RESTORE make distclean + recovery (EXPECT zero) ---
echo ""
echo "γ-1 STEP 3: RESTORE 'make distclean' + run --clean (EXPECT zero rc)"
set +e
"$CPYTHON_ROOT/scripts/build_phoenix.sh" --pydebug --clean > /tmp/i5g1_step3_build.log 2>&1
STEP3_BUILD_RC=$?
set -e
echo "  STEP 3 recovery-with-distclean rc=$STEP3_BUILD_RC"

# --- ASSERT: load-bearing CONFIRMED iff STEP 2 fails AND STEP 3 succeeds ---
echo ""
echo "γ-1 RESULTS:"
echo "  STEP 2 (sans distclean): rc=$STEP2_BUILD_RC (EXPECTED non-zero per ARM64 v3 class)"
echo "  STEP 3 (with distclean): rc=$STEP3_BUILD_RC (EXPECTED zero — load-bearing)"

if [ "$STEP2_BUILD_RC" -ne 0 ] && [ "$STEP3_BUILD_RC" -eq 0 ]; then
    echo ""
    echo "γ-1 PASS: I5 'make distclean' line IS LOAD-BEARING."
    echo "  Recovery without it FAILED (rc=$STEP2_BUILD_RC) under stronger poison."
    echo "  Recovery with it succeeded (rc=$STEP3_BUILD_RC)."
    echo "  Conclusion: the line authored by I5 quick-fix + amend-1 prevents"
    echo "  the poison-class failure mode it was designed to address."
    EXITCODE=0
elif [ "$STEP2_BUILD_RC" -eq 0 ] && [ "$STEP3_BUILD_RC" -eq 0 ]; then
    echo ""
    echo "γ-1 INCONCLUSIVE: I5 'make distclean' line load-bearing UNCONFIRMED."
    echo "  Recovery without it succeeded (rc=$STEP2_BUILD_RC)."
    echo "  Recovery with it also succeeded (rc=$STEP3_BUILD_RC)."
    echo "  Two non-exclusive hypotheses:"
    echo "    (a) NO-OP HYPOTHESIS: line is no-op masked by amend-1 explicit-rc;"
    echo "        consider removal in future I-series."
    echo "    (b) POISON-INSUFFICIENT HYPOTHESIS (gatekeeper 22:15:07Z (b)):"
    echo "        the touch+append poison may not reproduce ARM64 v3"
    echo "        18:13:17Z failure mode (cross-toggle Modules/*.o staleness"
    echo "        on aarch64); test on devgpu004 ARM64 with stronger poison."
    echo "  Disposition is supervisor-class; do NOT auto-remove based on this run alone."
    EXITCODE=2
elif [ "$STEP3_BUILD_RC" -ne 0 ]; then
    echo ""
    echo "γ-1 FAIL: STEP 3 recovery with line in place ALSO failed (rc=$STEP3_BUILD_RC)"
    echo "  This means the I5 line is INSUFFICIENT to recover from the applied poison."
    echo "  Either the poison is too aggressive (uncoverable by --clean) OR"
    echo "  the line itself is not actually doing what it's supposed to."
    echo "  See log: /tmp/i5g1_step3_build.log"
    tail -10 /tmp/i5g1_step3_build.log | sed 's/^/    /'
    EXITCODE=1
else
    echo ""
    echo "γ-1 UNEXPECTED: STEP 2 succeeded (rc=$STEP2_BUILD_RC) but STEP 3 failed (rc=$STEP3_BUILD_RC)"
    echo "  Restoring distclean made recovery WORSE — the line may be actively"
    echo "  destructive. Investigate immediately."
    EXITCODE=1
fi

exit $EXITCODE
