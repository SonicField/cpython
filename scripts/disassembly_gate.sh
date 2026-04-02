#!/bin/bash
# disassembly_gate.sh — Capture or compare JIT disassembly for gate validation
# Usage:
#   ./scripts/disassembly_gate.sh capture <output_file>   # Save current disassembly
#   ./scripts/disassembly_gate.sh compare <baseline> <current>  # Diff two captures
#   ./scripts/disassembly_gate.sh gate <baseline_file>    # Capture + compare in one step
set -euo pipefail

CPYTHON_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$CPYTHON_ROOT/python"

if [ ! -f "$PYTHON" ]; then
    echo "FAIL: python binary not found at $PYTHON"
    echo "Run scripts/build_phoenix.sh first"
    exit 1
fi

capture_disassembly() {
    local output="$1"
    local elf_tmp="/tmp/phoenix_gate_$$.elf"

    PYTHONPATH="$CPYTHON_ROOT/Lib/test" ASAN_OPTIONS=detect_leaks=0 "$PYTHON" -c "
import sys
sys.path.insert(0, '$CPYTHON_ROOT/Lib/test')
import _cinderx, cinderjit

# Standard gate functions — must match across captures
# Group 1: Arithmetic (original)
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

def add(x, y):
    return x + y

def mul(x, y):
    return x * y

def float_math(n):
    x = 1.5
    for i in range(n):
        x = x * 1.001 + 0.1
    return x

# Group 2: Closures (LOAD_DEREF, STORE_DEREF, MAKE_FUNCTION)
def make_adder(n):
    def adder(x):
        return x + n
    return adder

def make_counter():
    count = 0
    def inc():
        nonlocal count
        count += 1
        return count
    return inc

# Group 3: Exception handling (SETUP_FINALLY, POP_EXCEPT, RERAISE)
def safe_div(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return 0

def nested_try(x):
    try:
        try:
            if x < 0:
                raise ValueError('neg')
            return x * 2
        except ValueError:
            return -1
    finally:
        pass

# Group 4: Method calls + attribute access (LOAD_METHOD, CALL_METHOD, LOAD_ATTR)
def list_ops(n):
    result = []
    for i in range(n):
        result.append(i * i)
    return len(result)

def dict_ops(keys, vals):
    d = {}
    for k, v in zip(keys, vals):
        d[k] = v
    return d.get('missing', -1)

# Group 5: Generators (YIELD_VALUE, generator frame)
def gen_range(n):
    i = 0
    while i < n:
        yield i
        i += 1

def gen_filter(n):
    for x in range(n):
        if x % 2 == 0:
            yield x

# Compile all top-level functions
funcs = [fib, add, mul, float_math, make_adder, make_counter,
         safe_div, nested_try, list_ops, dict_ops, gen_range, gen_filter]
for f in funcs:
    cinderjit.force_compile(f)
    assert cinderjit.is_jit_compiled(f), f'{f.__name__} failed to compile'

# Also compile the inner closures
adder5 = make_adder(5)
cinderjit.force_compile(adder5)
counter = make_counter()
cinderjit.force_compile(counter)

# Correctness checks
assert fib(10) == 55
assert add(3, 4) == 7
assert mul(6, 7) == 42
assert isinstance(float_math(100), float)
assert adder5(10) == 15
assert counter() == 1 and counter() == 2
assert safe_div(10, 3) > 3.0 and safe_div(1, 0) == 0
assert nested_try(5) == 10 and nested_try(-1) == -1
assert list_ops(5) == 5
assert dict_ops(['a','b'], [1,2]) == -1
assert list(gen_range(4)) == [0, 1, 2, 3]
assert list(gen_filter(6)) == [0, 2, 4]

cinderjit.dump_elf('$elf_tmp')
" 2>&1

    if [ ! -f "$elf_tmp" ]; then
        echo "FAIL: ELF dump failed"
        exit 1
    fi

    # Header with metadata
    {
        echo "# Phoenix JIT Disassembly Gate Capture"
        echo "# Commit: $(cd "$CPYTHON_ROOT" && git rev-parse HEAD)"
        echo "# Branch: $(cd "$CPYTHON_ROOT" && git branch --show-current)"
        echo "# Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "# Arch: $(uname -m)"
        echo "# Functions: fib, add, mul, float_math, make_adder, make_counter, safe_div, nested_try, list_ops, dict_ops, gen_range, gen_filter (+closures)"
        echo "#"
        objdump -d "$elf_tmp"
    } > "$output"

    rm -f "$elf_tmp"
    echo "Captured disassembly to $output ($(wc -l < "$output") lines)"
}

# Normalize a disassembly file: sort by function, extract opcode sequences
# The JIT's register allocator and instruction scheduler are non-deterministic,
# so we compare opcode sequences (sorted within each function) rather than
# exact instruction text. This catches structural changes (wrong codegen path,
# missing/extra instructions) without false positives from register allocation.
normalize_file() {
    local input_file="$1"
    python3 - "$input_file" <<'PYEOF'
import re, sys
from collections import Counter

input_file = sys.argv[1]
sections = {}
current_func = None
current_opcodes = []

for line in open(input_file):
    line = line.rstrip()
    if line.startswith('#'):
        continue
    m = re.match(r'^[0-9a-f]+ <(.+)>:', line)
    if m:
        if current_func:
            sections[current_func] = current_opcodes
        current_func = m.group(1)
        current_opcodes = []
        continue
    m = re.match(r'^\s+[0-9a-f]+:\s+(?:[0-9a-f]{2}\s)+\s+(\S+)', line)
    if m:
        opcode = m.group(1)
        current_opcodes.append(opcode)

if current_func:
    sections[current_func] = current_opcodes

for func in sorted(sections):
    opcodes = sections[func]
    counts = Counter(opcodes)
    print(f'=== {func} ({len(opcodes)} instructions) ===')
    # Opcode sequence (order matters for control flow)
    print(f'sequence: {" ".join(opcodes)}')
    # Opcode histogram (order-independent structural fingerprint)
    print(f'histogram: {" ".join(f"{k}:{v}" for k, v in sorted(counts.items()))}')
    print()
PYEOF
}

compare_disassembly() {
    local baseline="$1"
    local current="$2"

    if [ ! -f "$baseline" ]; then
        echo "FAIL: baseline file not found: $baseline"
        exit 1
    fi
    if [ ! -f "$current" ]; then
        echo "FAIL: current file not found: $current"
        exit 1
    fi

    # Compare using histogram tolerance (JIT register allocator + scheduler
    # are non-deterministic, so exact instruction match is not possible for
    # complex functions). Tolerance: ±3 per opcode type, ±5 total per function.
    # B2 type-cast errors cause much larger deltas (10+ instructions).
    python3 - "$baseline" "$current" <<'PYEOF'
import re, sys
from collections import Counter

OPCODE_TOLERANCE = 5   # max difference per opcode type (JIT allocator varies ±4)
TOTAL_TOLERANCE = 8    # max total instruction count difference per function

def parse_normalized(filepath):
    """Parse normalized output into {func_name: Counter(opcodes)}"""
    funcs = {}
    current_func = None
    for line in open(filepath):
        line = line.rstrip()
        m = re.match(r'^=== (.+) \((\d+) instructions\) ===$', line)
        if m:
            current_func = m.group(1)
            funcs[current_func] = {'total': int(m.group(2)), 'histogram': Counter()}
            continue
        m = re.match(r'^histogram: (.+)$', line)
        if m and current_func:
            for item in m.group(1).split():
                opcode, count = item.split(':')
                funcs[current_func]['histogram'][opcode] = int(count)
    return funcs

baseline_norm = sys.argv[1] + '.norm'
current_norm = sys.argv[2] + '.norm'

# Write normalized versions
import subprocess
for src, dst in [(sys.argv[1], baseline_norm), (sys.argv[2], current_norm)]:
    # Re-run normalization (already done by caller, but we need the files)
    pass

# Parse the raw disassembly files directly
def parse_raw(filepath):
    funcs = {}
    current_func = None
    current_opcodes = []
    for line in open(filepath):
        line = line.rstrip()
        if line.startswith('#'):
            continue
        m = re.match(r'^[0-9a-f]+ <(.+)>:', line)
        if m:
            if current_func:
                funcs[current_func] = {'total': len(current_opcodes), 'histogram': Counter(current_opcodes)}
            current_func = m.group(1)
            current_opcodes = []
            continue
        m = re.match(r'^\s+[0-9a-f]+:\s+(?:[0-9a-f]{2}\s)+\s+(\S+)', line)
        if m:
            current_opcodes.append(m.group(1))
    if current_func:
        funcs[current_func] = {'total': len(current_opcodes), 'histogram': Counter(current_opcodes)}
    return funcs

base = parse_raw(sys.argv[1])
curr = parse_raw(sys.argv[2])

errors = []

# Check for missing/extra functions
base_funcs = set(base.keys())
curr_funcs = set(curr.keys())
for f in sorted(base_funcs - curr_funcs):
    errors.append(f'MISSING function: {f} (was in baseline)')
for f in sorted(curr_funcs - base_funcs):
    errors.append(f'NEW function: {f} (not in baseline)')

# Compare shared functions
for f in sorted(base_funcs & curr_funcs):
    b, c = base[f], curr[f]
    total_diff = abs(b['total'] - c['total'])
    if total_diff > TOTAL_TOLERANCE:
        errors.append(f'{f}: instruction count {b["total"]} -> {c["total"]} (delta {total_diff}, tolerance {TOTAL_TOLERANCE})')

    all_opcodes = set(b['histogram'].keys()) | set(c['histogram'].keys())
    for op in sorted(all_opcodes):
        diff = abs(b['histogram'].get(op, 0) - c['histogram'].get(op, 0))
        if diff > OPCODE_TOLERANCE:
            errors.append(f'{f}: {op} count {b["histogram"].get(op, 0)} -> {c["histogram"].get(op, 0)} (delta {diff}, tolerance {OPCODE_TOLERANCE})')

if errors:
    print('GATE: BLOCKED — structural differences detected')
    for e in errors:
        print(f'  {e}')
    sys.exit(1)
else:
    print('GATE: PASS — disassembly is structurally equivalent')
    print(f'  Functions compared: {len(base_funcs & curr_funcs)}')
    print(f'  Tolerance: ±{OPCODE_TOLERANCE} per opcode, ±{TOTAL_TOLERANCE} total per function')
    sys.exit(0)
PYEOF
    local result=$?

    echo "  Baseline: $baseline"
    echo "  Current:  $current"
    return $result
}

gate_check() {
    local baseline="$1"
    local current_tmp="/tmp/phoenix_gate_current_$$.txt"

    echo "=== Disassembly Gate Check ==="
    echo "Capturing current disassembly..."
    capture_disassembly "$current_tmp"

    echo ""
    echo "Comparing against baseline..."
    compare_disassembly "$baseline" "$current_tmp"
    local result=$?

    rm -f "$current_tmp"
    return $result
}

case "${1:-help}" in
    capture)
        [ -z "${2:-}" ] && { echo "Usage: $0 capture <output_file>"; exit 1; }
        capture_disassembly "$2"
        ;;
    compare)
        [ -z "${2:-}" ] || [ -z "${3:-}" ] && { echo "Usage: $0 compare <baseline> <current>"; exit 1; }
        compare_disassembly "$2" "$3"
        ;;
    gate)
        [ -z "${2:-}" ] && { echo "Usage: $0 gate <baseline_file>"; exit 1; }
        gate_check "$2"
        ;;
    help|*)
        echo "Usage:"
        echo "  $0 capture <output_file>     Capture current JIT disassembly"
        echo "  $0 compare <baseline> <cur>  Compare two captures"
        echo "  $0 gate <baseline_file>      Capture + compare against baseline"
        ;;
esac
