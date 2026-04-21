#!/bin/bash
# gate_phoenix.sh — Unified gate script for Phoenix JIT
# Builds, tests, and reports pass/fail for pre-push gating.
#
# Usage:
#   scripts/gate_phoenix.sh              # x86_64 only (local)
#   scripts/gate_phoenix.sh --pydebug    # x86_64 with assertions
#   scripts/gate_phoenix.sh --benchmark  # x86_64 + 4-benchmark check
#   scripts/gate_phoenix.sh --wiring     # x86_64 + wiring smoke (force_compile diverse functions)
#   scripts/gate_phoenix.sh --arm64      # x86_64 + ARM64 remote gate (devgpu004)
#
# Exit code: 0 = GATE PASS, 1 = GATE FAIL
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CPYTHON_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$CPYTHON_ROOT/python"

# Parse flags
PYDEBUG=0
BENCHMARK=0
CLEAN=0
WIRING=0
ARM64=0
EXPECT_COMMIT=""
for arg in "$@"; do
    case "$arg" in
        --pydebug)   PYDEBUG=1 ;;
        --benchmark) BENCHMARK=1 ;;
        --clean)     CLEAN=1 ;;
        --wiring)    WIRING=1 ;;
        --arm64)     ARM64=1 ;;
        --commit=*)  EXPECT_COMMIT="${arg#--commit=}" ;;
        *)           echo "Unknown flag: $arg"; echo "Usage: $0 [--pydebug] [--benchmark] [--clean] [--wiring] [--arm64] [--commit=HASH]"; exit 1 ;;
    esac
done

ARCH="$(uname -m)"
COMMIT_HASH="$(cd "$CPYTHON_ROOT" && git rev-parse --short HEAD)"
COMMIT="$(cd "$CPYTHON_ROOT" && git log -1 --oneline)"
GATE_PASS=1
FAILURES=""

GATE_LOG_DIR="$CPYTHON_ROOT/docs/gates"
mkdir -p "$GATE_LOG_DIR"
GATE_LOG="$GATE_LOG_DIR/${COMMIT_HASH}.log"
RESULTS_FILE="$GATE_LOG"

if [ -n "$EXPECT_COMMIT" ]; then
    if [[ "$COMMIT_HASH" != "$EXPECT_COMMIT"* ]]; then
        echo "GATE FAIL — HEAD $COMMIT_HASH does not match expected $EXPECT_COMMIT" | tee "$RESULTS_FILE"
        exit 1
    fi
fi

echo "=== Phoenix JIT Gate ===" | tee "$RESULTS_FILE"
echo "Architecture: $ARCH" | tee -a "$RESULTS_FILE"
echo "Commit: $COMMIT" | tee -a "$RESULTS_FILE"
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RESULTS_FILE"
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

# Copy binary to avoid "Text file busy" if another process holds the original
cp "$PYTHON" "${PYTHON}_gate" 2>/dev/null || true
PYTHON="${PYTHON}_gate"

# Step 1b: _reg usage policy gate (no-FS DeoptBase factories banned in simplify_c.c)
echo "" | tee -a "$RESULTS_FILE"
echo "--- Step 1b: _reg Usage Policy ---" | tee -a "$RESULTS_FILE"
REG_HITS=$(grep -n 'hir_c_create_guard_type_reg\|hir_c_create_guard_is_reg\|hir_c_create_vectorcall_reg\|hir_c_create_check_exc_reg\|hir_c_create_call_method_reg' "$CPYTHON_ROOT/Python/jit/hir/simplify_c.c" 2>/dev/null || true)
if [ -n "$REG_HITS" ]; then
    echo "GATE FAIL — banned no-FS _reg factories in simplify_c.c:" | tee -a "$RESULTS_FILE"
    echo "$REG_HITS" | tee -a "$RESULTS_FILE"
    echo "Use FS-aware alternatives (guard_type, guard_type_fs_reg, vectorcall_fs_reg)." | tee -a "$RESULTS_FILE"
    GATE_PASS=0
    FAILURES="$FAILURES _reg_policy:BLOCKED"
else
    echo "_reg policy: PASS (0 banned factories in simplify_c.c)" | tee -a "$RESULTS_FILE"
fi

# Step 1c: Preserved-symbol gate (critical functions must still exist after deletions)
echo "" | tee -a "$RESULTS_FILE"
echo "--- Step 1c: Preserved Symbol Check ---" | tee -a "$RESULTS_FILE"
MUST_SURVIVE="simplify_emit_cond simplify_emit_cond_slow_path hir_output_type_c hir_reflow_types_c hir_remove_trampoline_blocks_c hir_remove_unreachable_blocks_c simplify_env_emit simplify_binary_op_c simplify_load_method_c simplify_load_attr_c simplify_call_method_c simplify_vectorcall_c simplify_vectorcall_isinstance_c hir_chase_assign_operand hir_simplify_redundant_cond_branches_c hir_simplify_run_c"
SURVIVE_FAIL=0
for sym in $MUST_SURVIVE; do
    if ! grep -rq "$sym" "$CPYTHON_ROOT/Python/jit/hir/" --include='*.c' --include='*.cpp' --include='*.h' 2>/dev/null; then
        echo "GATE FAIL — preserved symbol '$sym' not found in any source file" | tee -a "$RESULTS_FILE"
        SURVIVE_FAIL=1
    fi
done
if [ "$SURVIVE_FAIL" -eq 1 ]; then
    GATE_PASS=0
    FAILURES="$FAILURES preserved_symbols:MISSING"
else
    echo "Preserved symbols: PASS (all $( echo $MUST_SURVIVE | wc -w) symbols present)" | tee -a "$RESULTS_FILE"
fi

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
PHOENIX_OUTPUT=$(JIT_ENABLE=1 ASAN_OPTIONS=detect_leaks=0 "$PYTHON" -m test $PHOENIX_MODULES 2>&1 || true)

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

# Step 4: CPython test suite (parallel, JIT enabled)
echo "" | tee -a "$RESULTS_FILE"
echo "--- Step 4: CPython Tests ---" | tee -a "$RESULTS_FILE"
CPYTHON_OUTPUT=$(JIT_ENABLE=1 ASAN_OPTIONS=detect_leaks=0 "$PYTHON" -m test -j8 --timeout=120 2>&1 || true)

CPYTHON_RESULT=$(echo "$CPYTHON_OUTPUT" | grep -oP 'Result: \K\w+' || echo "UNKNOWN")
CPYTHON_TOTAL_TESTS=$(echo "$CPYTHON_OUTPUT" | grep -oP 'Total tests: run=\K[0-9,]+' | tr -d ',' || echo 0)
CPYTHON_FILES_RUN=$(echo "$CPYTHON_OUTPUT" | grep -oP 'Total test files: run=\K[0-9]+' || echo 0)
CPYTHON_FILES_TOTAL=$(echo "$CPYTHON_OUTPUT" | grep -oP 'Total test files: run=[0-9]+/\K[0-9]+' || echo 0)
CPYTHON_FAILED=$(echo "$CPYTHON_OUTPUT" | grep -oP 'failed=\K[0-9]+' || echo 0)
CPYTHON_SUMMARY=$(echo "$CPYTHON_OUTPUT" | tail -10)
echo "CPython: $CPYTHON_TOTAL_TESTS tests, $CPYTHON_FILES_RUN/$CPYTHON_FILES_TOTAL modules, Result: $CPYTHON_RESULT" | tee -a "$RESULTS_FILE"
echo "$CPYTHON_SUMMARY" | tee -a "$RESULTS_FILE"

# Check for crashes (hard failures)
CPYTHON_CRASHES=$(echo "$CPYTHON_OUTPUT" | grep -c -wE "CRASH|Segmentation fault|Aborted" || true)
CPYTHON_CRASHES=${CPYTHON_CRASHES:-0}
if [ "$CPYTHON_CRASHES" -gt 0 ]; then
    GATE_PASS=0
    FAILURES="$FAILURES CPython:${CPYTHON_CRASHES}crash"
fi

# Step 5: nbody crash check (auto-compilation path)
echo "" | tee -a "$RESULTS_FILE"
echo "--- Step 5: nbody Crash Check ---" | tee -a "$RESULTS_FILE"
NBODY_OUTPUT=$(JIT_ENABLE=1 "$PYTHON" -c "
import _cinderx, cinderjit
cinderjit.auto()
import sys; sys.path.insert(0, '$CPYTHON_ROOT/Tools')
from benchmark_phoenix import bench_nbody
for i in range(3):
    r = bench_nbody(10000)
    print(f'nbody iter {i}: {r}')
print('nbody: PASS')
" 2>&1 || true)
NBODY_EXIT=$?
echo "$NBODY_OUTPUT" | tee -a "$RESULTS_FILE"
if ! echo "$NBODY_OUTPUT" | grep -q "nbody: PASS"; then
    GATE_PASS=0
    FAILURES="$FAILURES nbody:CRASH"
    echo "nbody: FAIL (crash or incorrect output)" | tee -a "$RESULTS_FILE"
else
    echo "nbody: PASS" | tee -a "$RESULTS_FILE"
fi

# Step 6: Wiring smoke test (optional — catches sole-path divergence)
if [ "$WIRING" -eq 1 ]; then
    echo "" | tee -a "$RESULTS_FILE"
    echo "--- Step 6: Wiring Smoke Test ---" | tee -a "$RESULTS_FILE"
    WIRING_OUTPUT=$(ASAN_OPTIONS=detect_leaks=0 "$PYTHON" -c "
import _cinderx, cinderjit

def straight_add(x, y): return x + y

def recursive_fib(n):
    if n < 2: return n
    return recursive_fib(n - 1) + recursive_fib(n - 2)

def loop_sum(n):
    s = 0
    for i in range(n):
        s += i
    return s

def make_gen(n):
    def gen(n):
        for i in range(n):
            yield i * i
    return list(gen(n))

def nested_float(n):
    total = 0.0
    for i in range(n):
        for j in range(n):
            total += float(i) * float(j)
    return total

def multi_var(n):
    a, b, c = 0, 1, 2
    for i in range(n):
        a, b, c = b, c, a + b + c
    return a

def cond_loop(n):
    s = 0
    for i in range(n):
        s += i if i % 2 == 0 else -i
    return s

def make_closure(x):
    def inner(y):
        return x + y
    return inner

def container_eq():
    return ([1,2]==[1,2], [1,2]==[1,3], {1:2}=={1:2}, (1,2)==(1,2))

def tuple_fold():
    t = (10, 20, 30)
    return (t[0], t[1], t[-1], len(t), len((1,2,3,4,5)))

def dict_store():
    d = {}
    d['a'] = 1
    d['b'] = 2
    d['c'] = 3
    return d == {'a': 1, 'b': 2, 'c': 3}

def truthy_checks():
    return (bool([1,2]), bool([]), bool('hi'), bool(''),
            bool(42), bool(0), bool(3.14), bool(0.0),
            bool(None), bool(True), bool(False))

def length_checks():
    return (len([1,2,3]), len('hello'), len({1:2, 3:4}),
            len((1,2,3,4)), len({1,2,3}))

tests = [
    (straight_add, (3, 4), 7),
    (recursive_fib, (10,), 55),
    (loop_sum, (100,), 4950),
    (make_gen, (5,), [0, 1, 4, 9, 16]),
    (nested_float, (10,), 2025.0),
    (multi_var, (10,), 230),
    (cond_loop, (10,), -5),
    (make_closure(10), (5,), 15),
    (container_eq, (), (True, False, True, True)),
    (tuple_fold, (), (10, 20, 30, 3, 5)),
    (dict_store, (), True),
    (truthy_checks, (), (True, False, True, False, True, False, True, False, False, True, False)),
    (length_checks, (), (3, 5, 2, 4, 3)),
]
for func, args, expected in tests:
    cinderjit.force_compile(func)
    assert cinderjit.is_jit_compiled(func), f'{func.__name__} not compiled'
    result = func(*args)
    assert result == expected, f'{func.__name__}: got {result}, expected {expected}'
    print(f'{func.__name__}: PASS')
print('Wiring smoke: PASS')
" 2>&1 || true)
    WIRING_EXIT=$?
    echo "$WIRING_OUTPUT" | tee -a "$RESULTS_FILE"
    if ! echo "$WIRING_OUTPUT" | grep -q "Wiring smoke: PASS"; then
        GATE_PASS=0
        FAILURES="$FAILURES Wiring:CRASH"
        echo "Wiring smoke: FAIL" | tee -a "$RESULTS_FILE"
    else
        echo "Wiring smoke: PASS" | tee -a "$RESULTS_FILE"
    fi

    # Step 6b: emitCond wiring test (auto-compilation — exercises block bridges)
    echo "" | tee -a "$RESULTS_FILE"
    echo "--- Step 6b: emitCond Auto-Compile Wiring ---" | tee -a "$RESULTS_FILE"
    EMITCOND_OUTPUT=$(ASAN_OPTIONS=detect_leaks=0 JIT_ENABLE=1 "$PYTHON" -c "
import _cinderx, cinderjit
cinderjit.auto()

WARMUP = 1100

# --- kLoadMethod emitCond path ---
# Calling a method on a type triggers simplifyLoadTypeMethodCached
# which uses emitCond for type method cache fast/slow path.
class Widget:
    __slots__ = ('x', 'y')
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def area(self):
        return self.x * self.y

def load_method_test():
    w = Widget(3, 7)
    return w.area()

expected_lm = load_method_test()
for _ in range(WARMUP):
    load_method_test()
result_lm = load_method_test()
assert result_lm == expected_lm, f'load_method_test: {result_lm} != {expected_lm}'
print(f'load_method_test: PASS (result={result_lm})')

# --- kLoadAttr emitCond path ---
# Instance attribute access on a class with inline dict triggers
# simplifyLoadAttrSplitDict which uses emitCond for inline values check.
class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

def load_attr_test():
    p = Point(10, 20, 30)
    return p.x + p.y + p.z

expected_la = load_attr_test()
for _ in range(WARMUP):
    load_attr_test()
result_la = load_attr_test()
assert result_la == expected_la, f'load_attr_test: {result_la} != {expected_la}'
print(f'load_attr_test: PASS (result={result_la})')

# --- kCompare isinstance emitCond path ---
# isinstance() with a known type triggers simplifyVectorCall -> isinstance
# which uses emitCond for type check fast/slow path.
def isinstance_test():
    results = []
    results.append(isinstance(42, int))
    results.append(isinstance('hello', str))
    results.append(isinstance(3.14, int))
    results.append(isinstance([], list))
    results.append(isinstance({}, dict))
    return tuple(results)

expected_ii = isinstance_test()
for _ in range(WARMUP):
    isinstance_test()
result_ii = isinstance_test()
assert result_ii == expected_ii, f'isinstance_test: {result_ii} != {expected_ii}'
print(f'isinstance_test: PASS (result={result_ii})')

# --- Combined: method call on instance with dict attrs ---
# Exercises both LoadMethod and LoadAttr in same compilation unit.
class Counter:
    def __init__(self, start=0):
        self.count = start
    def increment(self, n=1):
        self.count += n
        return self.count

def combined_method_attr_test():
    c = Counter(10)
    c.increment(5)
    c.increment(3)
    return c.count

expected_cm = combined_method_attr_test()
for _ in range(WARMUP):
    combined_method_attr_test()
result_cm = combined_method_attr_test()
assert result_cm == expected_cm, f'combined_method_attr_test: {result_cm} != {expected_cm}'
print(f'combined_method_attr_test: PASS (result={result_cm})')

print('emitCond wiring: PASS')
" 2>&1 || true)
    echo "$EMITCOND_OUTPUT" | tee -a "$RESULTS_FILE"
    if ! echo "$EMITCOND_OUTPUT" | grep -q "emitCond wiring: PASS"; then
        GATE_PASS=0
        FAILURES="$FAILURES emitCond:FAIL"
        echo "emitCond wiring: FAIL" | tee -a "$RESULTS_FILE"
    else
        echo "emitCond wiring: PASS" | tee -a "$RESULTS_FILE"
    fi
fi

# Step 7: Benchmark (optional)
if [ "$BENCHMARK" -eq 1 ]; then
    echo "" | tee -a "$RESULTS_FILE"
    echo "--- Step 7: Benchmark ---" | tee -a "$RESULTS_FILE"
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

# Step 7: ARM64 remote gate (optional)
if [ "$ARM64" -eq 1 ]; then
    echo "" | tee -a "$RESULTS_FILE"
    echo "Step 7: ARM64 remote gate (devgpu004)" | tee -a "$RESULTS_FILE"
    ARM64_HOST="alexturner@devgpu004.kcm2.facebook.com"
    ARM64_DIR="~/local/phoenix/cpython"

    # Sync current HEAD to ARM64 via git bundle + SCP
    BUNDLE_FILE="$CPYTHON_ROOT/arm64-gate-bundle.bundle"
    REMOTE_BUNDLE="$ARM64_DIR/arm64-gate-bundle.bundle"
    (cd "$CPYTHON_ROOT" && git bundle create "$BUNDLE_FILE" HEAD~200..HEAD 2>/dev/null)
    nbs-local-run "scp $BUNDLE_FILE $ARM64_HOST:$REMOTE_BUNDLE" 2>/dev/null

    ARM64_OUTPUT=$(nbs-remote-run "$ARM64_HOST" "
        cd $ARM64_DIR &&
        git checkout --detach HEAD 2>&1 | tail -1;
        git fetch $REMOTE_BUNDLE HEAD:arm64-gate-update 2>&1 | tail -3;
        git checkout arm64-gate-update 2>&1 | tail -3;
        ARM64_COMMIT=\$(git rev-parse --short HEAD);
        echo ARM64_COMMIT=\$ARM64_COMMIT;
        scripts/build_phoenix.sh --clean 2>&1 | tail -5;
        echo BUILD_ARM64=\$?;
        chmod +x python;
        JIT_ENABLE=1 ./python -m test test_phoenix_jit_arithmetic test_phoenix_jit_autocompile test_phoenix_jit_comparisons test_phoenix_jit_containers test_phoenix_jit_controlflow test_phoenix_jit_coverage test_phoenix_jit_functions test_phoenix_jit_generators test_phoenix_float test_phoenix_hir_type test_phoenix_benchmark_correctness test_phoenix_deferred_compile test_phoenix_profiling_hooks test_phoenix_usetype_float 2>&1 | tail -10;
        echo ARM64_EXIT=\$?
    " 2>&1 || echo "ARM64_REMOTE_FAIL")

    # Verify ARM64 commit matches x86_64 commit
    ARM64_COMMIT_HASH=$(echo "$ARM64_OUTPUT" | grep -v '^\$>' | grep -oP 'ARM64_COMMIT=\K\S+' | head -1)
    if [ -n "$ARM64_COMMIT_HASH" ] && [ "$ARM64_COMMIT_HASH" != "$COMMIT_HASH" ]; then
        echo "GATE FAIL — ARM64 commit $ARM64_COMMIT_HASH does not match x86_64 commit $COMMIT_HASH" | tee -a "$RESULTS_FILE"
        GATE_PASS=0
        FAILURES="$FAILURES ARM64:commit_mismatch($ARM64_COMMIT_HASH)"
    fi

    echo "$ARM64_OUTPUT" | tee -a "$RESULTS_FILE"

    if echo "$ARM64_OUTPUT" | grep -q "ARM64_REMOTE_FAIL"; then
        GATE_PASS=0
        FAILURES="$FAILURES ARM64:remote_fail"
    elif echo "$ARM64_OUTPUT" | grep -q "FAILURE"; then
        GATE_PASS=0
        FAILURES="$FAILURES ARM64:test_failure"
    fi
    rm -f "$BUNDLE_FILE"
fi

# Final report
echo "" | tee -a "$RESULTS_FILE"
echo "========================================" | tee -a "$RESULTS_FILE"
if [ "$GATE_PASS" -eq 1 ]; then
    echo "GATE PASS — $ARCH: Phoenix $PHOENIX_TOTAL tests ($PHOENIX_MODULES_PASS/$PHOENIX_MODULES_TOTAL modules)" | tee -a "$RESULTS_FILE"
    echo "Commit: $COMMIT" | tee -a "$RESULTS_FILE"
    echo "Gate log: $GATE_LOG"
    exit 0
else
    echo "GATE FAIL — $FAILURES" | tee -a "$RESULTS_FILE"
    echo "Commit: $COMMIT" | tee -a "$RESULTS_FILE"
    echo "Gate log: $GATE_LOG"
    exit 1
fi
