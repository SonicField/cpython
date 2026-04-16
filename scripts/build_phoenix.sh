#!/bin/bash
# build_phoenix.sh — Clean build of Phoenix JIT with phoenix-asm
# Works on both x86_64 and ARM64 (aarch64).
# Ensures no stale .o files or binaries contaminate the build.
#
# Usage:
#   scripts/build_phoenix.sh              # standard build
#   scripts/build_phoenix.sh --clean      # rm cmake cache first (use after JIT header changes)
#   scripts/build_phoenix.sh --pydebug    # build with --with-pydebug (enables JIT_DCHECK assertions)
#   scripts/build_phoenix.sh --jobs=16    # override parallelism (default: 32, max: 64)
#   scripts/build_phoenix.sh --clean --pydebug  # combine flags
set -euo pipefail

CPYTHON_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$CPYTHON_ROOT/Python/jit_build/build"
ARCH="$(uname -m)"

# Parse flags
CLEAN=0
PYDEBUG=0
ASAN=0
JOBS=32
for arg in "$@"; do
    case "$arg" in
        --clean)   CLEAN=1 ;;
        --pydebug) PYDEBUG=1 ;;
        --asan)    ASAN=1 ;;
        --jobs=*)  JOBS="${arg#--jobs=}" ;;
        *)         echo "Unknown flag: $arg"; echo "Usage: $0 [--clean] [--pydebug] [--asan] [--jobs=N]"; exit 1 ;;
    esac
done

# Cap jobs at 64 to avoid OOM on shared machines
if [ "$JOBS" -gt 64 ]; then
    JOBS=64
fi

echo "=== Phoenix JIT Clean Build ==="
echo "CPython root: $CPYTHON_ROOT"
echo "Build dir: $BUILD_DIR"
echo "Architecture: $ARCH"
echo "Jobs: $JOBS"
[ "$CLEAN" -eq 1 ] && echo "Mode: CLEAN (full cmake cache removal)"
[ "$PYDEBUG" -eq 1 ] && echo "Mode: PYDEBUG (assertions enabled)"
[ "$ASAN" -eq 1 ] && echo "Mode: ASAN (address sanitizer enabled)"

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
if [ "$CLEAN" -eq 1 ]; then
    echo "Full cmake cache removal (--clean)"
    rm -rf "$BUILD_DIR"
else
    rm -rf "$BUILD_DIR/CMakeFiles/jit.dir" \
           "$BUILD_DIR/CMakeFiles/phoenix_jit.dir" \
           "$BUILD_DIR/libjit.a" \
           "$BUILD_DIR/libphoenix_jit.a" \
           "$BUILD_DIR/libcommon.a"
fi

# Step 2: Configure cmake with PHOENIX_ASM
echo "--- Configuring cmake with PHOENIX_ASM=ON ---"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
# CRITICAL: --pydebug requires CMAKE_BUILD_TYPE=Debug so JIT_DCHECK is active.
# RelWithDebInfo compiles out JIT_DCHECK even with configure --with-pydebug.
CMAKE_BUILD_TYPE="RelWithDebInfo"
EXTRA_CMAKE_FLAGS=""
if [ "$PYDEBUG" -eq 1 ]; then
    CMAKE_BUILD_TYPE="Debug"
fi
if [ "$ASAN" -eq 1 ]; then
    EXTRA_CMAKE_FLAGS=" -fsanitize=address -fno-omit-frame-pointer"
fi
if ! cmake .. \
    -DPHOENIX_ASM=ON \
    -DCMAKE_CXX_FLAGS="-DPHOENIX_ASM${EXTRA_CMAKE_FLAGS}" \
    -DCMAKE_C_FLAGS="-DPHOENIX_ASM${EXTRA_CMAKE_FLAGS}" \
    -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++; then
    echo "FAIL: cmake configuration failed"
    exit 1
fi

# Step 3: Build JIT library
echo "--- Building JIT library ---"
cmake --build . -- -j"$JOBS"

# Step 4: Create empty libasmjit.a stub (phoenix-asm replaces asmjit)
# Always recreate — PGO's internal `make clean` deletes the stub
mkdir -p "$BUILD_DIR/_deps/asmjit-build"
AR_CMD=$(command -v llvm-ar 2>/dev/null || command -v ar)
$AR_CMD rcs "$BUILD_DIR/_deps/asmjit-build/libasmjit.a"

# Step 5: Configure CPython (hermetic — always reconfigure)
# ARM64: no LTO (causes issues on aarch64 devgpu builds)
# x86_64: LTO enabled for production performance
echo "--- Configuring CPython ---"
cd "$CPYTHON_ROOT"
PYDEBUG_FLAG="--without-pydebug"
ASAN_FLAG=""
if [ "$PYDEBUG" -eq 1 ]; then
    PYDEBUG_FLAG="--with-pydebug"
fi
if [ "$ASAN" -eq 1 ]; then
    ASAN_FLAG="--with-address-sanitizer"
fi
if [ "$ARCH" = "aarch64" ]; then
    echo "ARM64 detected — configuring without LTO"
    if ! CC=clang CXX=clang++ ./configure $PYDEBUG_FLAG $ASAN_FLAG --without-lto; then
        echo "FAIL: configure failed"
        exit 1
    fi
else
    if [ "$PYDEBUG" -eq 1 ]; then
        echo "x86_64 detected — configuring without LTO (pydebug)"
        if ! CC=clang CXX=clang++ ./configure $PYDEBUG_FLAG $ASAN_FLAG --without-lto; then
            echo "FAIL: configure failed"
            exit 1
        fi
    else
        echo "x86_64 detected — configuring with LTO"
        if ! CC=clang CXX=clang++ ./configure $PYDEBUG_FLAG $ASAN_FLAG --with-lto; then
            echo "FAIL: configure failed"
            exit 1
        fi
    fi
fi

# Step 6: Remove stale python binary and rebuild CPython
echo "--- Building CPython ---"
rm -f python
# ASAN builds: disable LeakSanitizer during make — CPython's freeze step runs
# its own Python, and ASAN flags CPython's startup leaks as errors.
if [ "$ASAN" -eq 1 ]; then
    ASAN_OPTIONS=detect_leaks=0 make -j"$JOBS" 2>&1 | tail -3
else
    make -j"$JOBS" 2>&1 | tail -3
fi

# Step 7: Verify binary exists
if [ ! -f "$CPYTHON_ROOT/python" ]; then
    echo "FAIL: python binary not built"
    exit 1
fi

echo "=== Build complete ==="
echo "Binary: $CPYTHON_ROOT/python"
echo "Commit: $(git log -1 --oneline)"
echo "Binary timestamp: $(stat -c %Y python)"
