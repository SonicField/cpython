#!/bin/bash
# test_phoenix.sh — Full Phoenix JIT test suite
# Verifies JIT compilation first — if JIT isn't compiling, test suite doesn't run
set -euo pipefail

CPYTHON_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$CPYTHON_ROOT/python"

echo "=== Phoenix JIT Test Suite ==="

# Step 1: Verify JIT is compiling (PREREQUISITE)
verify_jit() {
    if [ ! -f "$PYTHON" ]; then
        echo "FAIL: python binary not found at $PYTHON"
        return 1
    fi

    echo "Binary: $PYTHON"

    ASAN_OPTIONS=detect_leaks=0 "$PYTHON" -c "
import sys, _cinderx, cinderjit

# force_compile smoke test
def add(x, y): return x + y
def mul(x, y): return x * y
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

for f in [add, mul, fib]:
    cinderjit.force_compile(f)
    if not cinderjit.is_jit_compiled(f):
        print(f'FAIL: {f.__name__} did not compile')
        sys.exit(1)

assert add(3, 4) == 7
assert mul(6, 7) == 42
assert fib(10) == 55
print(f'  force_compile: add={add(3,4)} mul={mul(6,7)} fib={fib(10)} — PASS')

# auto-compilation test
def auto_target(x):
    total = 0
    for i in range(x):
        total += i * i
    return total

for i in range(2000):
    auto_target(10)

if not cinderjit.is_jit_compiled(auto_target):
    print('FAIL: auto-compilation did not trigger')
    sys.exit(1)
print(f'  auto_compile: threshold={cinderjit.get_compile_after_n_calls()} — PASS')

# generator resume test
def gen_counter(n):
    total = 0
    for i in range(n):
        total += i
        yield total

cinderjit.force_compile(gen_counter)
if not cinderjit.is_jit_compiled(gen_counter):
    print('FAIL: gen_counter did not compile')
    sys.exit(1)
assert list(gen_counter(10)) == [sum(range(i+1)) for i in range(10)]
print('  generator_resume: PASS')

print('JIT VERIFICATION: PASS')
"
}

echo "--- Step 1: Verify JIT is compiling ---"
verify_jit || { echo "FAIL: JIT not compiling. Aborting test suite."; exit 1; }

# Step 2: Full test suite
echo ""
echo "--- Step 2: Full test suite ---"
ASAN_OPTIONS=detect_leaks=0 "$PYTHON" -m test -j4 2>&1 | tee /tmp/phoenix_test_output.txt

# Step 3: Report
echo ""
echo "--- Step 3: Summary ---"
echo "Full output: /tmp/phoenix_test_output.txt"
tail -10 /tmp/phoenix_test_output.txt
