#!/bin/bash
# count_w25_b1b_tus.sh — reproducible W25 §1b TU baseline.
#
# Enumerates TUs in Python/jit/hir/ that include hir_basic_block_c.h
# (the W25 §1b set) and categorizes them by whether they have local
# extern decls of API functions (Step B cleanup targets) vs type-only
# inclusion (likely no-op for Step B).
#
# Usage:
#   scripts/count_w25_b1b_tus.sh [HIR_DIR]
#   (default path: Python/jit/hir/ relative to repo root)
#
# Output:
#   - Total TUs including hir_basic_block_c.h
#   - Per-TU extern-decl count (sorted descending)
#   - Subtotal: TUs with extern decls (real Step B cleanup targets)
#   - Subtotal: TUs without extern decls (type-only inclusion, no-op for Step B)
#
# Origin: pythia #80 #4 [chat 2026-04-22 19:12Z] flagged that
# count_emit_methods.sh is method-specific; the /N-class lapse pattern
# (4 occurrences in one session: /144, /38 INVOKE_*, /24 §1b, /5 §1a)
# needs pattern-general reproducible scripts. This anchors §1b TU
# counts the same way count_emit_methods.sh anchors emit-method counts.
#
# Reference values: at HEAD e6a8a2d0fb (post-W25-Step-A, push 65):
#   total: 17 §1b TUs
#   with extern decls (lint-pattern match): 7 (Step B cleanup targets)
#   type-only: 10 (no-op for Step B)
#
# Spec §1b table at line 60-72 used a BROADER grep (any "extern.*hir_")
# yielding higher per-TU counts (9 with externs / 8 type-only at b808cdada4).
# This script uses the NARROW lint-pattern grep matching spec §3 Step C —
# the actual gate that Step B will install. Differences:
#   - refcount_env_c.c + refcount_pass_c.c have non-API externs (e.g.
#     hir_liveness_*) that broad grep counts but lint-pattern excludes.
#     They're correctly classified as type-only-for-Step-B.
#
# The lint pattern matches spec §3 Step C grep:
#   ^extern[[:space:]]+.*hir_(c_|cfg_|block_|bb_|edge_|func_|instr_)
# (excludes hir_c_api.h itself + builder.cpp legitimate bridges)

set -euo pipefail

HIR_DIR="${1:-Python/jit/hir}"

if [[ ! -d "$HIR_DIR" ]]; then
    echo "ERROR: $HIR_DIR not found" >&2
    exit 1
fi

# Step 1: enumerate all TUs that include hir_basic_block_c.h
mapfile -t b1b_tus < <(
    grep -l 'cinderx/Jit/hir/hir_basic_block_c\.h' \
        "$HIR_DIR"/*.c "$HIR_DIR"/*.cpp 2>/dev/null \
        | sort
)

total=${#b1b_tus[@]}

# Step 2: per-TU extern-decl count using spec §3 Step C lint pattern
echo "=== W25 §1b TU baseline ($HIR_DIR at $(git rev-parse --short HEAD 2>/dev/null || echo 'no-git')) ==="
echo
echo "Per-TU extern API decl counts (spec §3 Step C lint pattern):"
echo

with_externs=0
type_only=0
declare -a with_externs_list
declare -a type_only_list

for tu in "${b1b_tus[@]}"; do
    base=$(basename "$tu")
    # Skip self-canonical headers (lint excludes these)
    if [[ "$base" == "hir_c_api.h" || "$base" == "builder.cpp" ]]; then
        continue
    fi
    count=$(grep -cE '^extern[[:space:]]+.*hir_(c_|cfg_|block_|bb_|edge_|func_|instr_)' "$tu" 2>/dev/null || true)
    count=${count:-0}
    if (( count > 0 )); then
        with_externs=$(( with_externs + 1 ))
        with_externs_list+=("$count $base")
    else
        type_only=$(( type_only + 1 ))
        type_only_list+=("$base")
    fi
done

# Sort by extern count desc
printf '%s\n' "${with_externs_list[@]}" | sort -rn | while read -r count name; do
    printf "  %3d  %s\n" "$count" "$name"
done

echo
echo "TUs WITHOUT extern API decls (type-only inclusion, no-op for Step B):"
for name in "${type_only_list[@]}"; do
    echo "  $name"
done

echo
echo "=== SUMMARY ==="
echo "TOTAL §1b TUs: $total"
echo "  WITH extern decls (Step B cleanup targets): $with_externs"
echo "  TYPE-ONLY inclusion (no-op for Step B):     $type_only"
