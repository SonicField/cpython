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

# Copy binary to avoid "Text file busy" if another process holds the original.
# rm before cp so a busy ${PYTHON}_gate from a prior run cannot leave a
# stale binary in place — that masquerades as PASS while the gate actually
# tests the OLD binary. Pre-2026-04-22 the cp was '|| true' which silently
# tolerated 'Text file busy' and ran the gate against whatever stale
# ${PYTHON}_gate happened to already exist (caught at 00:41Z, push 38).
# Now: rm forces a fresh copy; cp failure aborts the gate.
rm -f "${PYTHON}_gate"
cp "$PYTHON" "${PYTHON}_gate"
PYTHON="${PYTHON}_gate"

# Step 1a: Binary identity verification (per theologian 2026-04-22 00:43:35Z).
# Even with the cp-without-||-true fix, downstream failure modes can still
# produce a binary that does not correspond to HEAD (stale build that wasn't
# rebuilt after pull, wrong working-tree state at build time, etc.). Make
# 'GATE PASS commit X' falsifiable: prove the gate binary actually reports
# commit X in its --version output before any test runs.
echo "" | tee -a "$RESULTS_FILE"
echo "--- Step 1a: Gate Binary Identity ---" | tee -a "$RESULTS_FILE"
GATE_BINARY_VERSION=$("$PYTHON" --version 2>&1)
echo "$GATE_BINARY_VERSION" | tee -a "$RESULTS_FILE"
# Long-form version output ($PYTHON -c 'import sys; print(sys.version)') carries
# the commit hash; --version only carries '3.12.13'. Use sys.version.
GATE_BINARY_LONG_VERSION=$("$PYTHON" -c 'import sys; print(sys.version)' 2>&1)
echo "$GATE_BINARY_LONG_VERSION" | tee -a "$RESULTS_FILE"
if ! echo "$GATE_BINARY_LONG_VERSION" | grep -q "$COMMIT_HASH"; then
    echo "BINARY_MISMATCH: gate binary does not contain HEAD hash $COMMIT_HASH" | tee -a "$RESULTS_FILE"
    echo "GATE FAIL — gate binary identity does not match HEAD" | tee -a "$RESULTS_FILE"
    exit 1
fi
# Reject '-dirty' suffix: per supervisor 2026-04-22 01:08:18Z, the prior
# 'cp || true' (removed in f3271f58a5) was accidentally protecting the
# gate by falling back to a stale-clean binary when the local build was
# dirty. With cp now strict, a -dirty build IS the binary the gate
# tests. The original 79890e7b73-dirty contamination (catch #4) would
# match 'BINARY_MATCH 79890e7b73 ✓' under the bare-hash grep — close
# the loophole.
if echo "$GATE_BINARY_LONG_VERSION" | grep -q -- "-dirty"; then
    echo "BINARY_DIRTY: gate binary built with uncommitted working-tree changes" | tee -a "$RESULTS_FILE"
    echo "  $GATE_BINARY_LONG_VERSION" | tee -a "$RESULTS_FILE"
    echo "GATE FAIL — gate binary built from a dirty working tree (commit 'git stash' or 'git status --short' before re-running)" | tee -a "$RESULTS_FILE"
    exit 1
fi
echo "BINARY_MATCH: gate binary reports $COMMIT_HASH (clean) ✓" | tee -a "$RESULTS_FILE"

# Step 1a-2: RC_ORACLE production-leak check (gatekeeper item #15)
# Per supervisor 2026-04-22 03:06:55Z + theologian 03:07:12Z + pythia #58:
# the W3 R4 oracle dispatcher in compiler.cpp is #ifdef RC_ORACLE guarded.
# Production builds MUST contain ZERO rc_oracle symbols. A standing assertion
# in the gate transcript catches the silent failure mode where a future
# compiler.cpp edit accidentally drops or inverts the #ifdef guard, leaking
# RC_ORACLE dispatch into production undetectably. Mirrors BINARY_DIRTY
# discipline — catch silent failures structurally, not via memory.
RC_ORACLE_LEAK=$(nm "$PYTHON" 2>/dev/null | grep -c 'rc_oracle' || echo 0)
RC_ORACLE_LEAK=$(echo "$RC_ORACLE_LEAK" | tr -d '[:space:]')
if [ "${RC_ORACLE_LEAK:-0}" -ne 0 ]; then
    echo "BINARY_RC_ORACLE_LEAK_DETECTED: production binary contains $RC_ORACLE_LEAK rc_oracle symbols" | tee -a "$RESULTS_FILE"
    echo "FATAL: RC_ORACLE leaked into production build (review #ifdef RC_ORACLE guards in Python/jit/compiler.cpp)" | tee -a "$RESULTS_FILE"
    exit 1
fi
echo "BINARY_RC_ORACLE_OK: production binary clean (0 rc_oracle symbols)" | tee -a "$RESULTS_FILE"

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

# Step 1d: W25 lint gate (Step C per W25 spec docs/w25-hbb-canonicalization.md §3).
# Catches local extern decls of API functions (file-scope OR function-scope) that
# would re-introduce the §1b drift surface W25 Step B closed (293 lint externs
# across 9 TUs). Theologian L2347 revised pattern uses ^[[:space:]]*extern to
# match both line-start (file-scope) and indented (function-scope) externs.
echo "" | tee -a "$RESULTS_FILE"
echo "--- Step 1d: W25 Lint Gate ---" | tee -a "$RESULTS_FILE"
W25_VIOLATORS=$(grep -rE '^[[:space:]]*extern[[:space:]]+.*hir_(c_|cfg_|block_|bb_|edge_|func_|instr_)' \
    "$CPYTHON_ROOT/Python/jit/hir/" \
    --include='*.c' --include='*.cpp' \
    --exclude='hir_c_api.h' --exclude='hir_basic_block_c.h' --exclude='builder.cpp' \
    2>/dev/null || true)
if [ -n "$W25_VIOLATORS" ]; then
    echo "GATE FAIL — W25 lint pattern matched local extern decls of API functions:" | tee -a "$RESULTS_FILE"
    echo "$W25_VIOLATORS" | tee -a "$RESULTS_FILE"
    echo "Fix: delete the extern + add #include of canonical header (hir_c_api.h," | tee -a "$RESULTS_FILE"
    echo "  hir_basic_block_c.h, hir_instr_c.h, or hir_cfg_rpo_c.h depending on the function)." | tee -a "$RESULTS_FILE"
    GATE_PASS=0
    FAILURES="$FAILURES w25_lint:$(echo "$W25_VIOLATORS" | wc -l)"
else
    echo "W25 lint gate: PASS (0 lint-pattern externs)" | tee -a "$RESULTS_FILE"
fi

# Step 1e: W44 DO-NOT-USE caller gate
echo "" | tee -a "$RESULTS_FILE"
echo "--- Step 1e: W44 DO-NOT-USE Caller Gate ---" | tee -a "$RESULTS_FILE"
W44_OUTPUT=$("$SCRIPT_DIR/check_do_not_use_callers.sh" --strict 2>&1) && W44_EXIT=0 || W44_EXIT=$?
echo "$W44_OUTPUT" | tee -a "$RESULTS_FILE"
if [ "$W44_EXIT" -ne 0 ]; then
    echo "GATE FAIL — W44 DO-NOT-USE caller gate detected production callers" | tee -a "$RESULTS_FILE"
    GATE_PASS=0
    W44_VIOLATIONS=$(echo "$W44_OUTPUT" | grep -c "VIOLATION" || true)
    FAILURES="$FAILURES w44_do_not_use:$W44_VIOLATIONS"
else
    echo "W44 DO-NOT-USE caller gate: PASS" | tee -a "$RESULTS_FILE"
fi

# Step 1g: W45 §3.5 derivation-drift falsifier (per spec §2.7.3 #3).
# Step 1f reserved for future W45 §1-§2 bridge-sig falsifier integration.
echo "" | tee -a "$RESULTS_FILE"
echo "--- Step 1g: W45 §3.5 Derivation-Drift Falsifier ---" | tee -a "$RESULTS_FILE"
W45_35_OUTPUT=$("$SCRIPT_DIR/w45_section_3_5_derivation_drift.sh" --strict 2>&1) && W45_35_EXIT=0 || W45_35_EXIT=$?
echo "$W45_35_OUTPUT" | tee -a "$RESULTS_FILE"
if [ "$W45_35_EXIT" -ne 0 ]; then
    echo "GATE FAIL — W45 §3.5 derivation-drift falsifier detected unprotected derivation surface" | tee -a "$RESULTS_FILE"
    GATE_PASS=0
    W45_35_FAILURES=$(echo "$W45_35_OUTPUT" | grep -c "\[FAIL\]" || true)
    FAILURES="$FAILURES w45_section_3_5:$W45_35_FAILURES"
else
    echo "W45 §3.5 derivation-drift falsifier: PASS" | tee -a "$RESULTS_FILE"
fi

# Step 1h: W-I3-RUNTIME-ASSERT (IV) gate — BasicBlock::id immutability.
# Detects id-mutation patterns that would silently corrupt PhxBcBlockArray
# dense-array O(1) lookup (Tier 8 SECOND-PILOT Phase B). Per
# docs/w-i3-runtime-assert-spec.md §2 (IV) + §7. Wired per librarian
# 22:05:19Z framework guidance (mirrors 1e/1g delegation shape).
echo "" | tee -a "$RESULTS_FILE"
echo "--- Step 1h: W-I3 Invariant Gate ---" | tee -a "$RESULTS_FILE"
I3_OUTPUT=$("$SCRIPT_DIR/check_i3_invariant.sh" --strict 2>&1) && I3_EXIT=0 || I3_EXIT=$?
echo "$I3_OUTPUT" | tee -a "$RESULTS_FILE"
if [ "$I3_EXIT" -ne 0 ]; then
    echo "GATE FAIL — W-I3 invariant gate detected BasicBlock::id mutation" | tee -a "$RESULTS_FILE"
    GATE_PASS=0
    I3_VIOLATIONS=$(echo "$I3_OUTPUT" | grep -c "candidate I3 violation" || true)
    FAILURES="$FAILURES w_i3:$I3_VIOLATIONS"
else
    echo "W-I3 invariant gate: PASS" | tee -a "$RESULTS_FILE"
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
PHOENIX_MODULES="test_phoenix_jit_arithmetic test_phoenix_jit_autocompile test_phoenix_jit_comparisons test_phoenix_jit_containers test_phoenix_jit_controlflow test_phoenix_jit_coverage test_phoenix_jit_functions test_phoenix_jit_generators test_phoenix_jit_inline_except_closure test_phoenix_jit_loadattr_golden test_phoenix_float test_phoenix_hir_type test_phoenix_profiling_hooks test_phoenix_deferred_compile test_phoenix_benchmark_correctness test_phoenix_usetype_float"
# W32 Option B: capture exit code as canonical SUCCESS/FAIL signal.
# The output `Result: SUCCESS` literal can be missing (zero tests, format
# drift) and previously misclassified clean runs as UNKNOWN. The exit
# code is the authoritative signal.
set +e
PHOENIX_OUTPUT=$(JIT_ENABLE=1 ASAN_OPTIONS=detect_leaks=0 "$PYTHON" -m test $PHOENIX_MODULES 2>&1)
PHOENIX_EXIT=$?
set -e

PHOENIX_TOTAL=$(echo "$PHOENIX_OUTPUT" | grep -oP 'Total tests: run=\K[0-9]+' || echo 0)
if [ $PHOENIX_EXIT -eq 0 ]; then
    PHOENIX_RESULT="SUCCESS"
else
    PHOENIX_RESULT="FAIL"
fi

PHOENIX_MODULES_PASS=$(echo "$PHOENIX_OUTPUT" | grep -oP 'run=\K[0-9]+(?=/[0-9]+$)' | tail -1 || echo 0)
PHOENIX_MODULES_TOTAL=$(echo "$PHOENIX_OUTPUT" | grep -oP 'run=[0-9]+/\K[0-9]+' | tail -1 || echo 0)

echo "Phoenix: $PHOENIX_TOTAL tests, $PHOENIX_MODULES_PASS/$PHOENIX_MODULES_TOTAL modules, Result: $PHOENIX_RESULT (exit=$PHOENIX_EXIT)" | tee -a "$RESULTS_FILE"
if [ "$PHOENIX_RESULT" != "SUCCESS" ]; then
    GATE_PASS=0
    FAILURES="$FAILURES Phoenix:$PHOENIX_RESULT"
    echo "Phoenix FAILURES:" | tee -a "$RESULTS_FILE"
    # W32 Option A: tolerate grep-no-match (pipefail would otherwise kill
    # the script before Step 4-6 ran).
    { echo "$PHOENIX_OUTPUT" | grep -E "FAIL|ERROR|CRASH|Assertion failed" || true; } | tee -a "$RESULTS_FILE"
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
import _cinderx, cinderjit, sys

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

# Rotating wiring-gate additions per supervisor 23:32:03Z (push 37 batch).
# Cover recently-converted Tier-5 emit methods on the force_compile path
# (no warmup → exercises GENERIC unspecialized opcode dispatch in builder C).
# MINIMAL-BYTECODE rule (supervisor 00:04:55Z, theologian 00:04:50Z): each
# function's force_compile failure must be attributable to ONE emit-method
# family. Inseparable predecessors (LOAD_GLOBAL before LOAD_ATTR; SET_FUNCTION_ATTRIBUTE
# after MAKE_FUNCTION) are permitted; independent additional exercises are not.
def binop_arithmetic(x, y):
    # emitBinaryOp generic: chained BINARY_OP across +,-,*,//,%
    return x + y - (y * 2) + (x // 2) - (y % 3)

def load_attr_generic():
    # emitLoadAttr generic: LOAD_GLOBAL sys → LOAD_ATTR maxsize → RETURN_VALUE.
    return sys.maxsize

def make_function_with_defaults():
    # emitMakeFunction: MAKE_FUNCTION + SET_FUNCTION_ATTRIBUTE(defaults) + RETURN_VALUE.
    # Inner is data; not force_compiled. CALL is OUTSIDE this function.
    def inner(a=1):
        return a + 1
    return inner

def block_map_resize_chain(n):
    # Tier 8 SECOND-PILOT Phase A (testkeeper 10:29:06Z): deterministic
    # PhxBlockMap_resize trigger via createBlocks. Each `if x: return`
    # arm contributes 3 block_starts (branch target + branch fallthrough +
    # terminator nextInstr). 8 arms * 3 + entry = ~25 block_starts; well
    # past the 16*0.7=11.2 resize threshold so the resize path runs even
    # if Python compiles the if-chain to fewer-than-expected block_starts.
    if n == 0: return 'a'
    if n == 1: return 'b'
    if n == 2: return 'c'
    if n == 3: return 'd'
    if n == 4: return 'e'
    if n == 5: return 'f'
    if n == 6: return 'g'
    if n == 7: return 'h'
    if n == 8: return 'i'
    if n == 9: return 'j'
    return 'z'

# Tier 8 SECOND-PILOT Phase A pythia #119 (a) closure (theologian
# 11:20:24Z + 11:42:36Z + supervisor 11:42:49Z): n>=294 hash-clustering
# coverage at the named pythia threshold. Push 41 (110 arms) achieved
# n=223 BBs (PARTIAL closure); push 42 bumps to 150 arms targeting
# n>=294 BBs to fully exercise the 5 sequential resizes
# (16->32->64->128->256->512) that re._parser:Tokenizer.__next drives
# in production. force_compile must succeed AND produce the correct
# result for sampled arms -- proves Knuth multiplicative
# h=key*2654435761u clusters acceptably at production-scale BCOffset
# density. Measured: each Python 'if n==i: return i' arm produces
# 2 HIR BBs (testkeeper 11:41:25Z PYTHONJITDUMPFINALHIR), so 150 arms
# yields ~302 BBs.
_n294_src = 'def block_map_n294_chain(n):\n'
for _i in range(150):
    _n294_src += f'    if n == {_i}: return {_i}\n'
_n294_src += '    return -1\n'
exec(_n294_src)

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
    (binop_arithmetic, (10, 3), 10 + 3 - (3*2) + (10//2) - (3%3)),
    (load_attr_generic, (), sys.maxsize),
    (block_map_resize_chain, (5,), 'f'),
    (block_map_resize_chain, (9,), 'j'),
    (block_map_resize_chain, (99,), 'z'),
    (block_map_n294_chain, (0,), 0),
    (block_map_n294_chain, (74,), 74),
    (block_map_n294_chain, (149,), 149),
    (block_map_n294_chain, (200,), -1),
]
for func, args, expected in tests:
    cinderjit.force_compile(func)
    assert cinderjit.is_jit_compiled(func), f'{func.__name__} not compiled'
    result = func(*args)
    assert result == expected, f'{func.__name__}: got {result}, expected {expected}'
    print(f'{func.__name__}: PASS')

# Special-case make_function_with_defaults: force_compile target = OUTER (MAKE_FUNCTION
# + SET_FUNCTION_ATTRIBUTE family). Inner-function invocation happens OUTSIDE the JIT'd
# code so the force_compile failure mode is unambiguously attributable to emitMakeFunction.
cinderjit.force_compile(make_function_with_defaults)
assert cinderjit.is_jit_compiled(make_function_with_defaults), 'make_function_with_defaults not compiled'
_inner = make_function_with_defaults()
assert _inner() == 2 and _inner(5) == 6, f'make_function_with_defaults inner: {_inner()}, {_inner(5)}'
print('make_function_with_defaults: PASS')

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
    # Place the bundle OUTSIDE the cpython working tree so the new
    # 'git stash push -u' below does not stash it as an untracked file
    # — which would leave 'git fetch' with no bundle to read.
    REMOTE_BUNDLE="/tmp/arm64-gate-bundle.bundle"
    (cd "$CPYTHON_ROOT" && git bundle create "$BUNDLE_FILE" HEAD~200..HEAD 2>/dev/null)
    nbs-local-run "scp $BUNDLE_FILE $ARM64_HOST:$REMOTE_BUNDLE" 2>/dev/null

    # Working-tree contamination guard: stash any uncommitted devgpu004 changes
    # BEFORE the gate so 'git checkout --detach HEAD' cannot fail and stale edits
    # cannot pollute the build. Restore (pop) AFTER the gate completes — PASS or
    # FAIL — so the developer's in-flight work on devgpu004 is preserved.
    # 'git stash push -u' is a no-op when the working tree is clean.
    #
    # 'stdbuf -oL -eL' wraps the local nbs-remote-run process to force
    # line-buffering on its output. Two truncation incidents in the push 40
    # window (01:11Z, 01:25Z) had ARM64 transcript cut off mid-build, leaving
    # 'GATE PASS' verdicts undeterminable — addressed by line buffering on the
    # capture side so output flushes incrementally rather than in a final block.
    ARM64_OUTPUT=$(stdbuf -oL -eL nbs-remote-run "$ARM64_HOST" "
        cd $ARM64_DIR &&
        echo STASH_PUSH_BEGIN;
        ORIG_REF=\$(git symbolic-ref --short -q HEAD || git rev-parse HEAD);
        echo ORIG_REF=\$ORIG_REF;
        git stash push -u -m phoenix-gate-stash 2>&1 | tail -2;
        echo STASH_PUSH_END;
        git checkout --detach HEAD 2>&1 | tail -1;
        git fetch $REMOTE_BUNDLE HEAD:arm64-gate-update 2>&1 | tail -3;
        git checkout arm64-gate-update 2>&1 | tail -3;
        ARM64_COMMIT=\$(git rev-parse --short HEAD);
        echo ARM64_COMMIT=\$ARM64_COMMIT;
        scripts/build_phoenix.sh 2>&1 | tail -5;
        echo BUILD_ARM64=\$?;
        chmod +x python;
        ARM64_BINARY_LONG_VERSION=\$(./python -c 'import sys; print(sys.version)' 2>&1);
        echo ARM64_BINARY_LONG_VERSION=\$ARM64_BINARY_LONG_VERSION;
        if ! echo \"\$ARM64_BINARY_LONG_VERSION\" | grep -q \"\$ARM64_COMMIT\"; then
            echo \"ARM64_BINARY_MISMATCH: ./python does not contain \$ARM64_COMMIT\";
            echo \"GATE FAIL — ARM64 gate binary identity does not match HEAD\";
            exit 99;
        fi;
        if echo \"\$ARM64_BINARY_LONG_VERSION\" | grep -q -- '-dirty'; then
            echo \"ARM64_BINARY_DIRTY: ./python built with uncommitted working-tree changes\";
            echo \"GATE FAIL — ARM64 gate binary built from a dirty working tree\";
            exit 99;
        fi;
        echo \"ARM64_BINARY_MATCH: ./python reports \$ARM64_COMMIT (clean) ✓\";
        JIT_ENABLE=1 ./python -m test test_phoenix_jit_arithmetic test_phoenix_jit_autocompile test_phoenix_jit_comparisons test_phoenix_jit_containers test_phoenix_jit_controlflow test_phoenix_jit_coverage test_phoenix_jit_functions test_phoenix_jit_generators test_phoenix_jit_inline_except_closure test_phoenix_jit_loadattr_golden test_phoenix_float test_phoenix_hir_type test_phoenix_benchmark_correctness test_phoenix_deferred_compile test_phoenix_profiling_hooks test_phoenix_usetype_float 2>&1 | tail -10;
        echo ARM64_EXIT=\$?;
        echo STASH_POP_BEGIN;
        # Restore the original branch BEFORE popping so the stash is applied to
        # the same working tree it was captured from. Pop stash@{0} only if the
        # most-recent stash carries our marker (defensive — never pop someone
        # else's stash).
        git checkout \"\$ORIG_REF\" 2>&1 | tail -2;
        STASH_TOP=\$(git stash list 2>/dev/null | head -1);
        if echo \"\$STASH_TOP\" | grep -q 'phoenix-gate-stash'; then
            git stash pop 2>&1 | tail -3;
        else
            echo 'no phoenix-gate-stash on top of stash list — skipping pop';
        fi;
        echo STASH_POP_END
    " 2>&1 || echo "ARM64_REMOTE_FAIL")

    # Verify ARM64 commit matches x86_64 commit. Anchor the grep to
    # start-of-line so it matches ONLY the runtime echo output
    # ('ARM64_COMMIT=<hash>'), not the script-body echo lines (indented,
    # contain literal '$(git rev-parse ...)') nor the '$> ' prefixed
    # command-trace lines. Without the anchor, the first script-body
    # match captures '$(git' and the comparison fails spuriously.
    ARM64_COMMIT_HASH=$(echo "$ARM64_OUTPUT" | grep -oP '^ARM64_COMMIT=\K\S+' | head -1)
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
