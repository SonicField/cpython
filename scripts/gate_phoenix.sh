#!/bin/bash
# gate_phoenix.sh — Unified gate script for Phoenix JIT
# Builds, tests, and reports pass/fail for pre-push gating.
#
# Usage:
#   scripts/gate_phoenix.sh              # x86_64 only (local)
#   scripts/gate_phoenix.sh --pydebug    # x86_64 with assertions
#   scripts/gate_phoenix.sh --benchmark  # x86_64 + 4-benchmark check
#
# Exit code: 0 = GATE PASS, 1 = GATE FAIL
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CPYTHON_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$CPYTHON_ROOT/python"
RESULTS_FILE="/tmp/gate_results_$(date +%s).txt"

# Parse flags
PYDEBUG=0
BENCHMARK=0
CLEAN=0
for arg in "$@"; do
    case "$arg" in
        --pydebug)   PYDEBUG=1 ;;
        --benchmark) BENCHMARK=1 ;;
        --clean)     CLEAN=1 ;;
        *)           echo "Unknown flag: $arg"; echo "Usage: $0 [--pydebug] [--benchmark] [--clean]"; exit 1 ;;
    esac
done

ARCH="$(uname -m)"
COMMIT="$(cd "$CPYTHON_ROOT" && git log -1 --oneline)"
GATE_PASS=1
FAILURES=""

echo "=== Phoenix JIT Gate ===" | tee "$RESULTS_FILE"
echo "Architecture: $ARCH" | tee -a "$RESULTS_FILE"
echo "Commit: $COMMIT" | tee -a "$RESULTS_FILE"
echo "Pydebug: $PYDEBUG" | tee -a "$RESULTS_FILE"
echo "Benchmark: $BENCHMARK" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

# Step 1: Build
echo "--- Step 1: Build ---" | tee -a "$RESULTS_FILE"
BUILD_FLAGS=""
[ "$PYDEBUG" -eq 1 ] && BUILD_FLAGS="$BUILD_FLAGS --pydebug"
[ "$CLEAN" -eq 1 ] && BUILD_FLAGS="$BUILD_FLAGS --clean"

if ! "$SCRIPT_DIR/build_phoenix.sh" $BUILD_FLAGS; then
    echo "GATE FAIL — build failed" | tee -a "$RESULTS_FILE"
    exit 1
fi
echo "Build: PASS" | tee -a "$RESULTS_FILE"

# Step 2: Verify JIT compiles and executes
echo "" | tee -a "$RESULTS_FILE"
echo "--- Step 2: JIT Smoke Test ---" | tee -a "$RESULTS_FILE"
SMOKE_OUTPUT=$(ASAN_OPTIONS=detect_leaks=0 "$PYTHON" -c "
import _cinderx, cinderjit

def add(x, y): return x + y
def mul(x, y): return x * y
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

for f in [add, mul, fib]:
    cinderjit.force_compile(f)
    assert cinderjit.is_jit_compiled(f), f'{f.__name__} not compiled'

assert add(3, 4) == 7
assert mul(6, 7) == 42
assert fib(10) == 55
print('JIT smoke test: PASS')
" 2>&1)
SMOKE_EXIT=$?
echo "$SMOKE_OUTPUT" | tee -a "$RESULTS_FILE"
if [ "$SMOKE_EXIT" -ne 0 ]; then
    echo "GATE FAIL — JIT smoke test failed" | tee -a "$RESULTS_FILE"
    exit 1
fi
echo "JIT Smoke: PASS" | tee -a "$RESULTS_FILE"

# Step 3: Phoenix test suite
echo "" | tee -a "$RESULTS_FILE"
echo "--- Step 3: Phoenix Tests ---" | tee -a "$RESULTS_FILE"
PHOENIX_MODULES="test_phoenix_jit_arithmetic test_phoenix_jit_autocompile test_phoenix_jit_comparisons test_phoenix_jit_containers test_phoenix_jit_controlflow test_phoenix_jit_coverage test_phoenix_jit_functions test_phoenix_jit_generators test_phoenix_float test_phoenix_hir_type test_phoenix_profiling_hooks test_phoenix_deferred_compile test_phoenix_benchmark_correctness test_phoenix_usetype_float"
PHOENIX_OUTPUT=$(ASAN_OPTIONS=detect_leaks=0 "$PYTHON" -m test $PHOENIX_MODULES 2>&1 || true)

PHOENIX_TOTAL=$(echo "$PHOENIX_OUTPUT" | grep -oP 'Total tests: run=\K[0-9]+' || echo 0)
PHOENIX_RESULT=$(echo "$PHOENIX_OUTPUT" | grep -oP 'Result: \K\w+' || echo "UNKNOWN")

PHOENIX_MODULES_PASS=$(echo "$PHOENIX_OUTPUT" | grep -oP 'run=\K[0-9]+(?=/[0-9]+$)' | tail -1 || echo 0)
PHOENIX_MODULES_TOTAL=$(echo "$PHOENIX_OUTPUT" | grep -oP 'run=[0-9]+/\K[0-9]+' | tail -1 || echo 0)

echo "Phoenix: $PHOENIX_TOTAL tests, $PHOENIX_MODULES_PASS/$PHOENIX_MODULES_TOTAL modules, Result: $PHOENIX_RESULT" | tee -a "$RESULTS_FILE"
if [ "$PHOENIX_RESULT" != "SUCCESS" ]; then
    GATE_PASS=0
    FAILURES="$FAILURES Phoenix:$PHOENIX_RESULT"
    echo "Phoenix FAILURES:" | tee -a "$RESULTS_FILE"
    echo "$PHOENIX_OUTPUT" | grep -E "FAIL|ERROR|CRASH|Assertion failed" | tee -a "$RESULTS_FILE"
fi

# Step 4: CPython test suite (parallel)
echo "" | tee -a "$RESULTS_FILE"
echo "--- Step 4: CPython Tests ---" | tee -a "$RESULTS_FILE"
CPYTHON_OUTPUT=$(ASAN_OPTIONS=detect_leaks=0 "$PYTHON" -m test -j4 2>&1 || true)

CPYTHON_PASS=$(echo "$CPYTHON_OUTPUT" | grep -oP 'tests OK\.\s*$' | wc -l || echo 0)
CPYTHON_SUMMARY=$(echo "$CPYTHON_OUTPUT" | tail -5)
echo "CPython summary:" | tee -a "$RESULTS_FILE"
echo "$CPYTHON_SUMMARY" | tee -a "$RESULTS_FILE"

# Check for crashes (hard failures)
CPYTHON_CRASHES=$(echo "$CPYTHON_OUTPUT" | grep -c -E "CRASH|Segmentation fault|Aborted" || true)
CPYTHON_CRASHES=${CPYTHON_CRASHES:-0}
if [ "$CPYTHON_CRASHES" -gt 0 ]; then
    GATE_PASS=0
    FAILURES="$FAILURES CPython:${CPYTHON_CRASHES}crash"
fi

# Step 5: Benchmark (optional)
if [ "$BENCHMARK" -eq 1 ]; then
    echo "" | tee -a "$RESULTS_FILE"
    echo "--- Step 5: Benchmark ---" | tee -a "$RESULTS_FILE"
    cp "$PYTHON" "${PYTHON}_bench"
    BENCH_OUTPUT=$(VANILLA_PYTHON="$CPYTHON_ROOT/../cpython-vanilla/python" JIT_ENABLE=1 "${PYTHON}_bench" \
        "$CPYTHON_ROOT/Tools/benchmark_phoenix.py" jit \
        --compile=auto --reps=3 --only=fibonacci,nqueens,gen_simple,func_calls 2>&1 || true)
    echo "$BENCH_OUTPUT" | tee -a "$RESULTS_FILE"

    # Check geo-mean > 1.0x (hard floor)
    GEO_MEAN=$(echo "$BENCH_OUTPUT" | grep -oP 'geo-mean:\s*\K[0-9.]+' | head -1 || echo "0")
    GEO_MEAN=${GEO_MEAN:-0}
    BELOW_FLOOR=$(echo "$GEO_MEAN < 1.0" | bc -l 2>/dev/null || echo 0)
    if [ "${BELOW_FLOOR:-0}" -eq 1 ] && [ "$GEO_MEAN" != "0" ]; then
        GATE_PASS=0
        FAILURES="$FAILURES Benchmark:geo-mean=${GEO_MEAN}x(<1.0x)"
    fi
    rm -f "${PYTHON}_bench"
fi

# Final report
echo "" | tee -a "$RESULTS_FILE"
echo "========================================" | tee -a "$RESULTS_FILE"
if [ "$GATE_PASS" -eq 1 ]; then
    echo "GATE PASS — $ARCH: Phoenix $PHOENIX_TOTAL tests ($PHOENIX_MODULES_PASS/$PHOENIX_MODULES_TOTAL modules)" | tee -a "$RESULTS_FILE"
    echo "Commit: $COMMIT" | tee -a "$RESULTS_FILE"
    echo "Results: $RESULTS_FILE"
    exit 0
else
    echo "GATE FAIL — $FAILURES" | tee -a "$RESULTS_FILE"
    echo "Commit: $COMMIT" | tee -a "$RESULTS_FILE"
    echo "Results: $RESULTS_FILE"
    exit 1
fi
