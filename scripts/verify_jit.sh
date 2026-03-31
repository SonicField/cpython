#!/bin/bash
# verify_jit.sh — Verify JIT compilation and execution works
# Exit 0 = JIT works, Exit 1 = JIT broken
set -euo pipefail

CPYTHON_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$CPYTHON_ROOT/python"

if [ ! -f "$PYTHON" ]; then
    echo "FAIL: python binary not found at $PYTHON"
    exit 1
fi

echo "=== JIT Verification ==="
echo "Binary: $PYTHON"
echo "Binary timestamp: $(stat -c %Y "$PYTHON")"

ASAN_OPTIONS=detect_leaks=0 "$PYTHON" -c "
import sys, _cinderx, cinderjit

# Test 1: force_compile a simple function
def add(x, y): return x + y
def mul(x, y): return x * y
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

results = []
for f in [add, mul, fib]:
    cinderjit.force_compile(f)
    compiled = cinderjit.is_jit_compiled(f)
    results.append((f.__name__, compiled))
    if not compiled:
        print(f'FAIL: {f.__name__} did not compile')
        sys.exit(1)

# Test 2: execute compiled functions
assert add(3, 4) == 7, f'add(3,4)={add(3,4)}, expected 7'
assert mul(6, 7) == 42, f'mul(6,7)={mul(6,7)}, expected 42'
assert fib(10) == 55, f'fib(10)={fib(10)}, expected 55'

# Report
for name, compiled in results:
    print(f'  {name}: compiled={compiled}')
print(f'  add(3,4)={add(3,4)}')
print(f'  mul(6,7)={mul(6,7)}')
print(f'  fib(10)={fib(10)}')
print(f'  compiled_functions={len(cinderjit.get_compiled_functions())}')
print('JIT VERIFICATION: PASS')
"

echo "=== JIT Verification Complete ==="
