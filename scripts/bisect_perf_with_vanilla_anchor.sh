#!/bin/bash
# bisect_perf_with_vanilla_anchor.sh — perf bisect driver with vanilla absolute anchor
#
# M-slate SR-1 (supervisor D-1777572112). Per methodology audit
# cpython/docs/methodology-audit-2026-04-30.md:89: every bisect-point snapshot
# reports vanilla-CPython absolute time alongside Phoenix-JIT time. If vanilla
# absolute baseline drifts across bisect points → machine/env contamination
# (skip via exit 125). If vanilla holds steady AND Phoenix drifts → anchor-
# point established for the perf regression.
#
# Used as: git bisect run scripts/bisect_perf_with_vanilla_anchor.sh
#
# Required env:
#   BISECT_BENCH      — comma-separated bench names (default: fibonacci,nqueens,gen_simple,func_calls)
#   BISECT_PREDICATE  — bench name whose JIT speedup is the regression signal (default: fibonacci)
#   BISECT_THRESHOLD  — speedup threshold; commit is GOOD if BISECT_PREDICATE >= this (default: 1.5)
#   BISECT_REPS       — ABBA reps per bench (default: 3)
#   BISECT_VANILLA_DRIFT_PCT — abort bisect if vanilla absolute time drifts more than this %
#                              vs first-point baseline (default: 5).
#   BISECT_VANILLA_BASELINE_FILE — path that stores the first-point vanilla baseline
#                                   (default: /tmp/bisect_vanilla_baseline.txt)
#
# Exit codes (git bisect run convention):
#   0   = good (BISECT_PREDICATE speedup >= BISECT_THRESHOLD; vanilla within drift)
#   1   = bad  (BISECT_PREDICATE speedup < BISECT_THRESHOLD; vanilla within drift)
#   125 = skip (build failure OR vanilla absolute drifted beyond BISECT_VANILLA_DRIFT_PCT)
#
# Per CLAUDE.md Build Lock: only testkeeper executes this driver.
set -uo pipefail

CPYTHON_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CPYTHON_ROOT"

VANILLA_PY="$CPYTHON_ROOT/../cpython-vanilla/python"
EXPECTED_VANILLA_MD5="fcb1dddcbf5d1edbf54c478e705deccc"

BISECT_BENCH="${BISECT_BENCH:-fibonacci,nqueens,gen_simple,func_calls}"
BISECT_PREDICATE="${BISECT_PREDICATE:-fibonacci}"
BISECT_THRESHOLD="${BISECT_THRESHOLD:-1.5}"
BISECT_REPS="${BISECT_REPS:-3}"
BISECT_VANILLA_DRIFT_PCT="${BISECT_VANILLA_DRIFT_PCT:-5}"
BISECT_VANILLA_BASELINE_FILE="${BISECT_VANILLA_BASELINE_FILE:-/tmp/bisect_vanilla_baseline.txt}"

CUR=$(git log -1 --format=%H)
SHORT=$(git log -1 --format=%h)
echo "[bisect-perf] commit $CUR"

# Vanilla anchor verification: lock to known-good vanilla binary.
if [ ! -x "$VANILLA_PY" ]; then
    echo "[bisect-perf] vanilla binary not found at $VANILLA_PY -> skip (125)"
    exit 125
fi
ACTUAL_MD5=$(md5sum "$VANILLA_PY" | awk '{print $1}')
if [ "$ACTUAL_MD5" != "$EXPECTED_VANILLA_MD5" ]; then
    echo "[bisect-perf] vanilla md5 mismatch: expected $EXPECTED_VANILLA_MD5 got $ACTUAL_MD5 -> skip (125)"
    exit 125
fi

# Build Phoenix at this bisect point.
if ! scripts/build_phoenix.sh --clean 2>&1 | tail -5; then
    echo "[bisect-perf] BUILD FAIL on $CUR -> skip (125)"
    exit 125
fi

# Run the benchmark harness; capture both vanilla and JIT per-bench times.
cp ./python ./python_bench_perf
BENCH_OUTPUT=$(VANILLA_PYTHON="$VANILLA_PY" JIT_ENABLE=1 ./python_bench_perf \
    Tools/benchmark_phoenix.py jit \
    --compile=auto --reps="$BISECT_REPS" --only="$BISECT_BENCH" 2>&1 || true)
rm -f ./python_bench_perf

# Per-bench parser: each row is "<name> <vanilla_ms>ms <jit_ms>ms <speedup>x ..."
BENCH_PARSED=$(echo "$BENCH_OUTPUT" | awk '
    /^  [a-z_]+[[:space:]]+[0-9.]+ms[[:space:]]+[0-9.]+ms[[:space:]]+[0-9.]+x/ {
        vms=$2; gsub(/ms/,"",vms);
        jms=$3; gsub(/ms/,"",jms);
        sp=$4; gsub(/x/,"",sp);
        printf "%s %s %s %s\n", $1, vms, jms, sp;
    }')

if [ -z "$BENCH_PARSED" ]; then
    echo "[bisect-perf] no bench output rows parsed; harness output:"
    echo "$BENCH_OUTPUT" | tail -20
    echo "[bisect-perf] -> skip (125)"
    exit 125
fi

echo "[bisect-perf] per-bench (name vanilla_ms jit_ms speedup):"
echo "$BENCH_PARSED" | sed 's/^/  /'

# Vanilla absolute anchor: first-point baseline is recorded; subsequent points
# must stay within BISECT_VANILLA_DRIFT_PCT of baseline. If not, bisect
# environment is contaminated (load average, thermal, BIOS, scheduler) and
# the perf signal cannot be trusted at this point — skip via 125.
CURRENT_VANILLA_GEO=$(echo "$BENCH_PARSED" | awk '
    BEGIN { sumlog=0; n=0 }
    { sumlog += log($2); n++ }
    END { if (n>0) printf "%.4f", exp(sumlog/n); else print "0"; }')
echo "[bisect-perf] vanilla geo-mean ms: $CURRENT_VANILLA_GEO"

if [ ! -f "$BISECT_VANILLA_BASELINE_FILE" ]; then
    echo "$CURRENT_VANILLA_GEO" > "$BISECT_VANILLA_BASELINE_FILE"
    echo "[bisect-perf] anchored vanilla baseline at $CURRENT_VANILLA_GEO ms (this is bisect point #1)"
else
    BASELINE=$(cat "$BISECT_VANILLA_BASELINE_FILE")
    DRIFT_PCT=$(echo "scale=2; ($CURRENT_VANILLA_GEO - $BASELINE) / $BASELINE * 100" | bc -l 2>/dev/null || echo "0")
    ABS_DRIFT=$(echo "$DRIFT_PCT" | tr -d '-')
    OVER=$(echo "$ABS_DRIFT > $BISECT_VANILLA_DRIFT_PCT" | bc -l 2>/dev/null || echo 0)
    echo "[bisect-perf] vanilla drift vs baseline ${BASELINE} ms: ${DRIFT_PCT}% (limit ±${BISECT_VANILLA_DRIFT_PCT}%)"
    if [ "$OVER" = "1" ]; then
        echo "[bisect-perf] VANILLA DRIFT > ${BISECT_VANILLA_DRIFT_PCT}% — env contamination, skip (125)"
        exit 125
    fi
fi

# Predicate evaluation: extract BISECT_PREDICATE bench's speedup; compare to threshold.
PRED_LINE=$(echo "$BENCH_PARSED" | awk -v name="$BISECT_PREDICATE" '$1==name {print}')
if [ -z "$PRED_LINE" ]; then
    echo "[bisect-perf] predicate bench '$BISECT_PREDICATE' not found in output -> skip (125)"
    exit 125
fi
PRED_SPEEDUP=$(echo "$PRED_LINE" | awk '{print $4}')
GOOD=$(echo "$PRED_SPEEDUP >= $BISECT_THRESHOLD" | bc -l 2>/dev/null || echo 0)

if [ "$GOOD" = "1" ]; then
    echo "[bisect-perf] $SHORT -> GOOD ($BISECT_PREDICATE speedup ${PRED_SPEEDUP}x >= ${BISECT_THRESHOLD}x)"
    exit 0
else
    echo "[bisect-perf] $SHORT -> BAD ($BISECT_PREDICATE speedup ${PRED_SPEEDUP}x < ${BISECT_THRESHOLD}x)"
    exit 1
fi
