#!/bin/bash
# build_phoenix.sh — Clean build of Phoenix JIT with phoenix-asm
# Works on both x86_64 and ARM64 (aarch64).
# Ensures no stale .o files or binaries contaminate the build.
set -euo pipefail

CPYTHON_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$CPYTHON_ROOT/Python/jit_build/build"
ARCH="$(uname -m)"

echo "=== Phoenix JIT Clean Build ==="
echo "CPython root: $CPYTHON_ROOT"
echo "Build dir: $BUILD_DIR"
echo "Architecture: $ARCH"

# Step 0: Check if local branch is behind remote (prevents stale binary builds)
cd "$CPYTHON_ROOT"
if git remote get-url origin >/dev/null 2>&1; then
    echo "--- Checking remote sync ---"
    git fetch origin 2>/dev/null || true
    BEHIND=$(git rev-list HEAD..origin/phoenix-asm-integration --count 2>/dev/null || echo 0)
    if [ "$BEHIND" -gt 0 ]; then
        echo "WARNING: local is $BEHIND commit(s) behind remote!"
        echo "Run: git pull origin phoenix-asm-integration"
        echo "Building anyway — but the binary may be stale."
    else
        echo "Local is up to date with remote."
    fi
fi

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

# Step 4: Create empty libasmjit.a stub (phoenix-asm replaces asmjit)
# Always recreate — PGO's internal `make clean` deletes the stub
mkdir -p "$BUILD_DIR/_deps/asmjit-build"
llvm-ar rcs "$BUILD_DIR/_deps/asmjit-build/libasmjit.a"

# Step 5: Configure CPython (hermetic — always reconfigure)
# ARM64: no LTO (causes issues on aarch64 devgpu builds)
# x86_64: LTO enabled for production performance
echo "--- Configuring CPython ---"
cd "$CPYTHON_ROOT"
if [ "$ARCH" = "aarch64" ]; then
    echo "ARM64 detected — configuring without LTO"
    CC=clang CXX=clang++ ./configure --without-pydebug --without-lto > /dev/null 2>&1
else
    echo "x86_64 detected — configuring with LTO"
    CC=clang CXX=clang++ ./configure --without-pydebug --with-lto > /dev/null 2>&1
fi

# Step 6: Remove stale python binary and rebuild CPython
echo "--- Building CPython ---"
rm -f python
make -j"$(nproc)" 2>&1 | tail -3

# Step 7: Verify binary exists
if [ ! -f "$CPYTHON_ROOT/python" ]; then
    echo "FAIL: python binary not built"
    exit 1
fi

echo "=== Build complete ==="
echo "Binary: $CPYTHON_ROOT/python"
echo "Binary timestamp: $(stat -c %Y python)"
