#!/bin/bash
# build_phoenix.sh — Clean build of Phoenix JIT with phoenix-asm
# Ensures no stale .o files or binaries contaminate the build.
set -euo pipefail

CPYTHON_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$CPYTHON_ROOT/Python/jit_build/build"

echo "=== Phoenix JIT Clean Build ==="
echo "CPython root: $CPYTHON_ROOT"
echo "Build dir: $BUILD_DIR"

# Step 1: Clean stale JIT build artifacts
echo "--- Cleaning stale JIT artifacts ---"
rm -rf "$BUILD_DIR/CMakeFiles/jit.dir" \
       "$BUILD_DIR/CMakeFiles/phoenix_jit.dir" \
       "$BUILD_DIR/libjit.a" \
       "$BUILD_DIR/libphoenix_jit.a" \
       "$BUILD_DIR/libcommon.a"

# Step 2: Configure cmake with PHOENIX_ASM
echo "--- Configuring cmake with PHOENIX_ASM=ON ---"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake .. \
    -DPHOENIX_ASM=ON \
    -DCMAKE_CXX_FLAGS="-DPHOENIX_ASM" \
    -DCMAKE_C_FLAGS="-DPHOENIX_ASM" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    > /dev/null 2>&1

# Step 3: Build JIT library
echo "--- Building JIT library ---"
cmake --build . -- -j"$(nproc)"

# Step 4: Remove stale python binary and rebuild CPython
echo "--- Building CPython ---"
cd "$CPYTHON_ROOT"
rm -f python
make -j"$(nproc)" 2>&1 | tail -3

# Step 5: Verify binary exists
if [ ! -f "$CPYTHON_ROOT/python" ]; then
    echo "FAIL: python binary not built"
    exit 1
fi

echo "=== Build complete ==="
echo "Binary: $CPYTHON_ROOT/python"
echo "Binary timestamp: $(stat -c %Y python)"
