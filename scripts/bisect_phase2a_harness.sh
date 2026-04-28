#!/bin/bash
# bisect_phase2a_harness.sh — WS3 Phase 2A harness-stability sub-bisect driver
#
# Per theologian 07:46:45Z disposition. Predicate: bench harness --help exits 0
# within 30s. Range: e11fc09f6e..cf49ad6da5 (38 commits, ~5-6 iters).
#
# Necessary AND sufficient: --help failure at import-machinery setup = ABBA
# failure at same point per testkeeper 07:45:55Z forensic.
#
# Exit codes (git bisect run convention):
#   0   = good (harness --help exits 0)
#   1   = bad  (harness --help crashes/hangs)
#   125 = skip (build fail)
#
# Caller exports BENCH_SCRIPT (default: /tmp/phx-bisect-staging/benchmark_phoenix.py).
# Per CLAUDE.md Build Lock: only testkeeper executes this driver.
set -uo pipefail

CPYTHON_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CPYTHON_ROOT"

BENCH_SCRIPT="${BENCH_SCRIPT:-/tmp/phx-bisect-staging/benchmark_phoenix.py}"

CUR=$(git log -1 --format=%H)
echo "[bisect-phase2a] commit $CUR"

if ! scripts/build_phoenix.sh --clean 2>&1 | tail -5; then
    echo "[bisect-phase2a] BUILD FAIL on $CUR -> skip (125)"
    exit 125
fi

if ! cp ./python ./python_bench; then
    echo "[bisect-phase2a] cp ./python ./python_bench failed -> skip (125)"
    exit 125
fi

OUT=$(mktemp)
timeout 30 ./python_bench "$BENCH_SCRIPT" --help >"$OUT" 2>&1
EX=$?
if [ "$EX" -eq 0 ]; then
    echo "[bisect-phase2a] $CUR -> GOOD (--help exit 0)"
    rm -f "$OUT"
    exit 0
fi
echo "[bisect-phase2a] $CUR -> BAD (--help exit=$EX)"
tail -5 "$OUT"
rm -f "$OUT"
exit 1
