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

# W26 gate-hardening: track previous build exit so we can force --clean
# after a recent compile-fail. Stale .o / LTO bitcode from a failed iter
# can produce a binary that builds clean but crashes at runtime — this
# was the root cause of the push-51 build-state corruption near-miss.
# State file: scripts/.last_build_state (gitignored), 2 lines: epoch_ts
# then exit_code. Window: 5 minutes (rapid-iter sessions only).
STATE_FILE="$CPYTHON_ROOT/scripts/.last_build_state"
write_build_state() {
    local exit_code=$?
    {
        date +%s
        echo "$exit_code"
    } > "$STATE_FILE" 2>/dev/null || true
    return "$exit_code"
}
trap write_build_state EXIT

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

# W26 gate-hardening: detect compile-fail-then-pass rapid-iter sequence and
# force --clean. Window matches the V3 timeline (push 51): two compile fixes
# within ~3 min produced a binary with stale-artifact corruption.
if [ "$CLEAN" -eq 0 ] && [ -f "$STATE_FILE" ]; then
    LAST_TS=$(awk 'NR==1{print; exit}' "$STATE_FILE" 2>/dev/null || echo "")
    LAST_EXIT=$(awk 'NR==2{print; exit}' "$STATE_FILE" 2>/dev/null || echo "")
    if [ -n "$LAST_TS" ] && [ -n "$LAST_EXIT" ] && [ "$LAST_EXIT" -ne 0 ]; then
        AGE=$(( $(date +%s) - LAST_TS ))
        if [ "$AGE" -lt 300 ]; then
            echo "[W26 gate-hardening] previous build failed (exit $LAST_EXIT, ${AGE}s ago)"
            echo "[W26 gate-hardening] forcing --clean to avoid build-state corruption class"
            CLEAN=1
        fi
    fi
fi

# Cap jobs at 64 to avoid OOM on shared machines
if [ "$JOBS" -gt 64 ]; then
    JOBS=64
fi

PHOENIX_CC="${PHOENIX_CC:-clang}"
PHOENIX_CXX="${PHOENIX_CXX:-clang++}"

# gcc-toolset-15 workaround: force clang to use gcc 11 headers/libs
GCC_INSTALL_FLAG=""
if [ -d /usr/lib/gcc/x86_64-redhat-linux/11 ] && [ "$ARCH" != "aarch64" ]; then
    GCC_INSTALL_FLAG="--gcc-install-dir=/usr/lib/gcc/x86_64-redhat-linux/11"
fi

echo "=== Phoenix JIT Clean Build ==="
echo "CPython root: $CPYTHON_ROOT"
echo "Build dir: $BUILD_DIR"
echo "Architecture: $ARCH"
echo "C compiler: $PHOENIX_CC ($($PHOENIX_CC --version 2>&1 | head -1))"
echo "C++ compiler: $PHOENIX_CXX ($($PHOENIX_CXX --version 2>&1 | head -1))"
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

# Step 1b: Detect stale cmake artifacts (compiler mismatch)
if [ -f "$BUILD_DIR/CMakeCache.txt" ] && [ "$CLEAN" -eq 0 ]; then
    CACHED_CC=$(grep 'CMAKE_C_COMPILER:' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null | cut -d= -f2)
    CACHED_CXX=$(grep 'CMAKE_CXX_COMPILER:' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null | cut -d= -f2)
    ACTUAL_CC=$(command -v "$PHOENIX_CC" 2>/dev/null || echo "$PHOENIX_CC")
    ACTUAL_CXX=$(command -v "$PHOENIX_CXX" 2>/dev/null || echo "$PHOENIX_CXX")
    if [ "$CACHED_CC" != "$ACTUAL_CC" ] || [ "$CACHED_CXX" != "$ACTUAL_CXX" ]; then
        echo "WARNING: compiler changed (cached: $CACHED_CC/$CACHED_CXX, current: $ACTUAL_CC/$ACTUAL_CXX)"
        echo "Forcing --clean to prevent stale object contamination."
        rm -rf "$BUILD_DIR"
        CLEAN=1
    fi
fi

# Step 2: Configure CPython (generates pyconfig.h needed by cmake)
# Skip configure if pyconfig.h already exists and is valid (has SIZEOF_VOID_P defined).
# This handles environments where configure fails (e.g., cannot run compiled programs).
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
# DSecondary-3 (W-C2 push-46 follow-up): --clean must remove pyconfig.h +
# autoconf cache so toggling --pydebug ↔ release re-configures cleanly.
# Pre-DSecondary-3 (testkeeper 17:08:14Z + 17:17:59Z forensic): --clean
# removed BUILD_DIR but pyconfig.h survived; re-run without --pydebug
# inherited stale '#define Py_DEBUG 1', JIT_DCHECK fired in 'release'
# code, Phase B I1 invariant tripped → spurious gate FAIL on push-45.
#
# I5 quick-fix (theologian 18:35:26Z + supervisor 18:35:52Z): autoconf-
# cache file removal alone insufficient when --pydebug ↔ release
# transition leaves stale Modules/*.o + generated headers + python
# binary that were built against the prior config. ARM64 v3 gate FAIL
# 18:13:17Z empirical anchor: --clean alone left _testembed link with
# stale Py_DEBUG-built objects. `make distclean` is the load-bearing
# step (CPython tree artifact cleanup); rm -rf Python/jit_build/build
# is already covered above (line 110).
if [ "$CLEAN" -eq 1 ]; then
    # I5 amend-1 (theologian 19:54:21Z negative-control + supervisor
    # 19:54:48Z disposition β): I3-compliant make distclean call.
    # Was '2>/dev/null || true' which silently swallowed errors —
    # exactly the silent-fail-under-set-euo-pipefail pattern that I3
    # SUBSHELL VISIBILITY invariant catalogues as bug-class B4.
    # Negative-control 19:54:21Z: load-bearing-vs-no-op gap — recovery
    # passed with line reverted under simple --pydebug→release toggle,
    # but poison may not have reached the cross-toggle Modules/*.o
    # staleness that ARM64 v3 18:13:17Z exhibited. Disambiguation
    # deferred to full I5 self-test stage with stronger poison.
    if [ -f Makefile ]; then
        if ! make distclean >/dev/null 2>&1; then
            echo "WARNING: make distclean failed under --clean; downstream rm may be insufficient"
        fi
    fi
    rm -f pyconfig.h config.status config.cache
fi
if [ -f pyconfig.h ] && grep -q '#define SIZEOF_VOID_P' pyconfig.h && [ "$CLEAN" -eq 0 ]; then
    echo "pyconfig.h exists and is valid — skipping configure"
else
    if [ "$ARCH" = "aarch64" ]; then
        echo "ARM64 detected — configuring without LTO"
        if ! CC="$PHOENIX_CC $GCC_INSTALL_FLAG" CXX="$PHOENIX_CXX $GCC_INSTALL_FLAG" ./configure $PYDEBUG_FLAG $ASAN_FLAG --without-lto; then
            echo "FAIL: configure failed"
            exit 1
        fi
    else
        if [ "$PYDEBUG" -eq 1 ]; then
            echo "x86_64 detected — configuring without LTO (pydebug)"
            if ! CC="$PHOENIX_CC $GCC_INSTALL_FLAG" CXX="$PHOENIX_CXX $GCC_INSTALL_FLAG" ./configure $PYDEBUG_FLAG $ASAN_FLAG --without-lto; then
                echo "FAIL: configure failed"
                exit 1
            fi
        else
            echo "x86_64 detected — configuring with LTO"
            if ! CC="$PHOENIX_CC $GCC_INSTALL_FLAG" CXX="$PHOENIX_CXX $GCC_INSTALL_FLAG" ./configure $PYDEBUG_FLAG $ASAN_FLAG --with-lto; then
                echo "FAIL: configure failed"
                exit 1
            fi
        fi
    fi
fi


# Step 3: Configure cmake with PHOENIX_ASM
echo "--- Configuring cmake with PHOENIX_ASM=ON ---"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
# CRITICAL: --pydebug requires CMAKE_BUILD_TYPE=Debug so JIT_DCHECK is active.
# RelWithDebInfo compiles out JIT_DCHECK even with configure --with-pydebug.
CMAKE_BUILD_TYPE="RelWithDebInfo"
# Preserve externally-set EXTRA_CMAKE_FLAGS so callers (e.g.,
# gate_phoenix.sh JIT_VARIADIC_BAD_PATH_VERIFY mode) can inject extra
# -D flags. Bug landed empirically when the flag was hard-reset to ""
# on every invocation, silently dropping caller-supplied -DJIT_TEST_*
# and producing a binary without the bad-path mechanism (testkeeper
# 17:03:02Z + generalist diagnosis).
EXTRA_CMAKE_FLAGS="${EXTRA_CMAKE_FLAGS:-}"
if [ "$PYDEBUG" -eq 1 ]; then
    CMAKE_BUILD_TYPE="Debug"
fi
if [ "$ASAN" -eq 1 ]; then
    EXTRA_CMAKE_FLAGS="${EXTRA_CMAKE_FLAGS} -fsanitize=address -fno-omit-frame-pointer"
fi
if ! cmake .. \
    -DPHOENIX_ASM=ON \
    -DCMAKE_CXX_FLAGS="-DPHOENIX_ASM ${GCC_INSTALL_FLAG}${EXTRA_CMAKE_FLAGS}" \
    -DCMAKE_C_FLAGS="-DPHOENIX_ASM ${GCC_INSTALL_FLAG}${EXTRA_CMAKE_FLAGS}" \
    -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
    -DCMAKE_C_COMPILER="$PHOENIX_CC" \
    -DCMAKE_CXX_COMPILER="$PHOENIX_CXX"; then
    echo "FAIL: cmake configuration failed"
    exit 1
fi

# Step 4: Build JIT library
echo "--- Building JIT library ---"
cmake --build . -- -j"$JOBS"

# Step 5: Create empty libasmjit.a stub (phoenix-asm replaces asmjit)
# Always recreate — PGO's internal `make clean` deletes the stub
mkdir -p "$BUILD_DIR/_deps/asmjit-build"
AR_CMD=$(command -v llvm-ar 2>/dev/null || command -v ar)
$AR_CMD rcs "$BUILD_DIR/_deps/asmjit-build/libasmjit.a"

# Step 6: Remove stale python binary and rebuild CPython
echo "--- Building CPython ---"
cd "$CPYTHON_ROOT"
rm -f python
# Modules/getbuildinfo.o embeds GITVERSION/GITTAG/GITBRANCH at compile time,
# baked from 'git describe --dirty' AT FIRST BUILD. Incremental rebuilds
# don't re-run that compile because the .c source hasn't changed — so a
# binary built once with a dirty working tree carries the '-dirty' marker
# forever. Force re-evaluation every build by deleting the .o, so
# Modules/getbuildinfo.o is the only object the BINARY_MATCH check
# can rely on for accurate identity.
rm -f Modules/getbuildinfo.o
# ASAN builds: disable LeakSanitizer during make — CPython's freeze step runs
# its own Python, and ASAN flags CPython's startup leaks as errors.
MAKE_LOG=$(mktemp)
if [ "$ASAN" -eq 1 ]; then
    ASAN_OPTIONS=detect_leaks=0 make -j"$JOBS" 2>&1 | tee "$MAKE_LOG" | tail -3
else
    make -j"$JOBS" 2>&1 | tee "$MAKE_LOG" | tail -3
fi
MAKE_EXIT=${PIPESTATUS[0]}
if [ "$MAKE_EXIT" -ne 0 ]; then
    echo "FAIL: make exited with code $MAKE_EXIT"
    echo "Last 20 lines of make output:"
    tail -20 "$MAKE_LOG"
    rm -f "$MAKE_LOG"
    exit 1
fi
rm -f "$MAKE_LOG"

# Step 7: Verify binary exists and is executable
if [ ! -f "$CPYTHON_ROOT/python" ]; then
    echo "FAIL: python binary not built"
    exit 1
fi
chmod +x "$CPYTHON_ROOT/python"

echo "=== Build complete ==="
echo "Binary: $CPYTHON_ROOT/python"
echo "Commit: $(git log -1 --oneline)"
echo "Binary timestamp: $(stat -c %Y python)"
