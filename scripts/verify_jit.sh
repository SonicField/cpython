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
print('FORCE_COMPILE VERIFICATION: PASS')

# Test 3: auto-compilation via threshold=1000
print()
print('--- Auto-compilation test (threshold=1000) ---')
threshold = cinderjit.get_compile_after_n_calls()
print(f'  compile_after_n_calls={threshold}')

def auto_target(x):
    total = 0
    for i in range(x):
        total += i * i
    return total

# Call 2000 times to exceed threshold
for i in range(2000):
    auto_target(10)

auto_compiled = cinderjit.is_jit_compiled(auto_target)
print(f'  auto_target compiled after 2000 calls: {auto_compiled}')
if not auto_compiled:
    print('FAIL: auto-compilation did not trigger at threshold=1000')
    sys.exit(1)

# Verify correctness after auto-compilation
result = auto_target(100)
expected = sum(i * i for i in range(100))
assert result == expected, f'auto_target(100)={result}, expected {expected}'
print(f'  auto_target(100)={result} (correct)')
print('AUTO_COMPILE VERIFICATION: PASS')

print()

# Test 4: generator resume (regression gate for 4ec8a085d5)
print('--- Generator resume test ---')
def gen_counter(n):
    total = 0
    for i in range(n):
        total += i
        yield total

cinderjit.force_compile(gen_counter)
gen_compiled = cinderjit.is_jit_compiled(gen_counter)
print(f'  gen_counter compiled: {gen_compiled}')
if not gen_compiled:
    print('FAIL: gen_counter did not compile')
    sys.exit(1)

# Exercise generator resume path
g = gen_counter(10)
values = list(g)
expected_values = [sum(range(i+1)) for i in range(10)]
assert values == expected_values, f'gen_counter values={values}, expected {expected_values}'
print(f'  gen_counter(10) yielded {len(values)} values: correct')

# Multiple generators from same compiled function
g1 = gen_counter(5)
g2 = gen_counter(5)
v1 = next(g1)
v2 = next(g2)
assert v1 == 0 and v2 == 0, f'parallel gen first yield: g1={v1}, g2={v2}'
v1 = next(g1)
v2 = next(g2)
assert v1 == 1 and v2 == 1, f'parallel gen second yield: g1={v1}, g2={v2}'
print(f'  parallel generators: correct')
print('GENERATOR VERIFICATION: PASS')

print()
print('JIT VERIFICATION: PASS (force_compile + auto_compile + generator)')
"

echo "=== JIT Verification Complete ==="
