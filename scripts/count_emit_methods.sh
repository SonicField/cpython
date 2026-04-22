#!/bin/bash
# count_emit_methods.sh — reproducible Tier 5/6 progress baseline.
#
# Scans Python/jit/hir/builder.cpp for HIRBuilder::emit* method definitions and
# categorizes them by C-conversion status (stub / partial / pure C++).
#
# Usage:
#   scripts/count_emit_methods.sh [BUILDER_CPP_PATH]
#   (default path: Python/jit/hir/builder.cpp relative to repo root)
#
# Output: counts + the exhaustive pure-C++ method list.
#
# Origin: replaces the unverified /144 chat-propagated baseline that propagated
# from 2026-04-21 (commit 7783df7182) through 2026-04-22 Tier 5 close. Real
# denominator at HEAD a642405a5c was 123, not 144 (pythia #78 #1 catch).
#
# Methodology:
#   denominator = count of HIRBuilder::emit* method bodies in builder.cpp
#                 (multi-line signatures, any return type, brace-balanced body)
#   stub        = body has <=8 non-comment lines AND contains hir_builder_emit_*_c
#   partial     = body has >8 non-comment lines AND contains hir_builder_emit_*_c
#   pure C++    = body does NOT contain hir_builder_emit_*_c
#
# Reference values: at HEAD a642405a5c (post-Tier-5-close, push 62):
#   total: 123
#   stub: 93
#   partial: 7
#   pure C++: 23
#   ratio: (stub + partial) / total = 100/123 = 81.3%

set -euo pipefail

BUILDER_CPP="${1:-Python/jit/hir/builder.cpp}"

if [[ ! -f "$BUILDER_CPP" ]]; then
    echo "ERROR: $BUILDER_CPP not found" >&2
    exit 1
fi

python3 - "$BUILDER_CPP" <<'PYEOF'
import re
import sys

src = open(sys.argv[1]).read()

# Brace-balanced extraction: <return type> HIRBuilder::emitXxx( ... ) { ... }
methods = []
i = 0
while i < len(src):
    m = re.search(r'^(\w[\w\s:&*<>]*?)\s+HIRBuilder::(emit\w+)\s*\(', src[i:], re.MULTILINE)
    if not m:
        break
    name = m.group(2)
    ret_type = m.group(1).strip()
    start = i + m.start()
    paren_close = src.find(')', start)
    brace = src.find('{', paren_close)
    if brace == -1:
        break
    depth = 1
    j = brace + 1
    while j < len(src) and depth > 0:
        if src[j] == '{':
            depth += 1
        elif src[j] == '}':
            depth -= 1
        j += 1
    body = src[brace + 1:j - 1]
    methods.append((name, ret_type, body))
    i = j

stubs, partial, pure_cpp = [], [], []
for name, ret_type, body in methods:
    body_lines = [l.strip() for l in body.strip().split('\n')
                  if l.strip() and not l.strip().startswith('//')]
    has_c_call = 'hir_builder_emit_' in body
    if not has_c_call:
        pure_cpp.append((name, ret_type))
    elif len(body_lines) <= 8:
        stubs.append((name, ret_type))
    else:
        partial.append((name, ret_type))

total = len(methods)
converted = len(stubs) + len(partial)
ratio = (converted / total * 100) if total else 0.0

print(f"=== HIRBuilder::emit* C-conversion baseline ({sys.argv[1]}) ===")
print(f"TOTAL: {total}")
print(f"  STUBS  (delegate to C, body <=8 lines): {len(stubs)}")
print(f"  PARTIAL (some C, some C++):             {len(partial)}")
print(f"  PURE C++ (no C call):                   {len(pure_cpp)}")
print(f"CONVERTED (stubs + partial): {converted}")
print(f"RATIO: {converted}/{total} = {ratio:.1f}%")
print()
print("=== Pure C++ method list ===")
for name, ret_type in sorted(pure_cpp):
    print(f"  {ret_type} {name}")
print()
print("=== Partial method list ===")
for name, ret_type in sorted(partial):
    print(f"  {ret_type} {name}")
PYEOF
