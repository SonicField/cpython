#!/bin/bash
# build_vanilla.sh — Build vanilla CPython 3.12.13 from source for benchmarking
# Uses LTO (--with-lto) to match Phoenix build quality.
# NO PGO (--enable-optimizations) — Phoenix can't do PGO (breaks JIT build).
# Compiler matches Phoenix build (clang/clang++).
set -euo pipefail

CPYTHON_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VANILLA_DIR="$CPYTHON_ROOT/../cpython-vanilla"

echo "=== Vanilla CPython 3.12.13 Build ==="
echo "Phoenix root: $CPYTHON_ROOT"
echo "Vanilla dir:  $VANILLA_DIR"

# Step 1: Create vanilla worktree from v3.12.13 tag if not exists
if [ ! -d "$VANILLA_DIR" ]; then
    echo "--- Creating worktree from v3.12.13 tag ---"
    cd "$CPYTHON_ROOT"
    git worktree add "$VANILLA_DIR" v3.12.13
else
    echo "--- Vanilla worktree exists ---"
fi

cd "$VANILLA_DIR"

# Step 2: Configure with LTO, matching Phoenix compiler
echo "--- Configuring with LTO (clang) ---"
./configure \
    --without-pydebug \
    --with-lto \
    CC=clang \
    CXX=clang++ \
    2>&1 | tail -5

# Step 3: Build with LTO
echo "--- Building with LTO ---"
make -j"$(nproc)" 2>&1 | tail -10

# Step 4: Verify
if [ ! -f "$VANILLA_DIR/python" ]; then
    echo "FAIL: vanilla python binary not built"
    exit 1
fi

echo ""
echo "=== Vanilla Build Complete ==="
echo "Binary:  $VANILLA_DIR/python"
echo "Version: $("$VANILLA_DIR/python" --version)"
echo "CC:      $("$VANILLA_DIR/python" -c "import sysconfig; print(sysconfig.get_config_var('CC'))")"
echo "OPT:     $("$VANILLA_DIR/python" -c "import sysconfig; print(sysconfig.get_config_var('OPT'))")"
echo ""
echo "Set VANILLA_PYTHON=$VANILLA_DIR/python for benchmarks."
