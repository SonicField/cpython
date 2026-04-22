#!/bin/bash
# build_oracle.sh — W3 R4 oracle scratch lib + python_rc_cpp builder
#
# Builds:
#   1. docs/oracle_scratch/build/libphoenix_rc_oracle.a — cc4a18e7e5
#      refcount_insertion.cpp compiled against current jit headers via
#      _rc_oracle_adapter.h shim
#   2. docs/oracle_scratch/python_rc_cpp — copy of main python build with
#      compiler.cpp's #ifdef RC_ORACLE active and libphoenix_rc_oracle.a
#      linked
#
# Per supervisor 2026-04-22 02:37:10Z hybrid design.
# Per theologian 2026-04-22 02:30:08Z W3 spec.
#
# Pre-conditions:
#   - main python build complete (./python exists at HEAD)
#   - generalist's compiler.cpp dispatcher (Step 3.5) committed (#ifdef
#     RC_ORACLE block present)
#   - docs/oracle_scratch/{rc_oracle.cpp, rc_oracle.h, _rc_oracle_adapter.{h,cpp},
#     rc_oracle_entry.cpp, CMakeLists.txt} exist (Steps 2-3 + this script's
#     CMakeLists)
#
# Falsifier:
#   - nm docs/oracle_scratch/build/libphoenix_rc_oracle.a | grep ' T rc_oracle_run'
#     must return one match (rc_oracle_run is the C entry point)
#   - nm ./python | grep rc_oracle MUST return EMPTY (production has no
#     dispatcher because RC_ORACLE undefined)
#   - nm docs/oracle_scratch/python_rc_cpp | grep rc_oracle MUST return matches
#     (python_rc_cpp has the dispatcher because RC_ORACLE defined)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CPYTHON_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ORACLE_DIR="$CPYTHON_ROOT/docs/oracle_scratch"
ORACLE_BUILD="$ORACLE_DIR/build"
ORACLE_PYTHON="$ORACLE_DIR/python_rc_cpp"

echo "=== W3 R4 oracle scratch build ==="
echo "Oracle dir:  $ORACLE_DIR"
echo "Build dir:   $ORACLE_BUILD"
echo "Output bin:  $ORACLE_PYTHON"
echo ""

# Pre-flight: required source files
for f in rc_oracle.cpp rc_oracle.h _rc_oracle_adapter.h _rc_oracle_adapter.cpp \
         rc_oracle_entry.cpp CMakeLists.txt; do
    if [ ! -f "$ORACLE_DIR/$f" ]; then
        echo "FAIL: missing $ORACLE_DIR/$f" >&2
        exit 1
    fi
done

# Pre-flight: main python must exist (we copy its build artifacts)
if [ ! -f "$CPYTHON_ROOT/python" ]; then
    echo "FAIL: main python not built — run scripts/build_phoenix.sh first" >&2
    exit 1
fi

# Pre-flight: compiler.cpp must have the #ifdef RC_ORACLE dispatcher
if ! grep -q '#ifdef RC_ORACLE' "$CPYTHON_ROOT/Python/jit/compiler.cpp"; then
    echo "FAIL: compiler.cpp does not contain #ifdef RC_ORACLE dispatcher" >&2
    echo "      Step 3.5 (generalist) must land before this build can run" >&2
    exit 1
fi

# Step 1: Build the scratch lib
echo "--- Building libphoenix_rc_oracle.a ---"
mkdir -p "$ORACLE_BUILD"
(cd "$ORACLE_BUILD" && \
    CC=clang CXX=clang++ cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
        -DPHOENIX_ASM=ON > /dev/null && \
    cmake --build . --parallel "$(nproc)")

if [ ! -f "$ORACLE_BUILD/libphoenix_rc_oracle.a" ]; then
    echo "FAIL: libphoenix_rc_oracle.a not produced" >&2
    exit 1
fi

# Falsifier: rc_oracle_run must be a T (text) symbol with C linkage in the
# scratch lib. C linkage means unmangled — easy to verify with nm.
if ! nm "$ORACLE_BUILD/libphoenix_rc_oracle.a" 2>/dev/null | \
        grep -E ' T rc_oracle_run$' > /dev/null; then
    echo "FAIL: rc_oracle_run not exposed as T symbol from libphoenix_rc_oracle.a" >&2
    nm "$ORACLE_BUILD/libphoenix_rc_oracle.a" 2>/dev/null | grep rc_oracle || true
    exit 1
fi
echo "ORACLE_LIB_OK: rc_oracle_run T symbol present in libphoenix_rc_oracle.a"

# Step 2: Print the python_rc_cpp link recipe to stderr.
#
# Per supervisor 2026-04-22 02:46:08Z option (c): generalist's Step 5
# exercises the link out-of-band (W3 is a research-time / regression-suspected
# diagnostic, NOT a continuously-runnable CI gate). build_oracle.sh prints
# the recipe + recommended commands so the link is documented but not
# automated. If W3 becomes continuously-needed, W11 (Makefile.pre.in
# conditional target) is queued.
JIT_BUILD="$CPYTHON_ROOT/Python/jit_build/build"
COMPILER_RC_ORACLE_OBJ="$ORACLE_BUILD/compiler_rc_oracle.cpp.o"

echo ""
echo "--- python_rc_cpp link recipe (run out-of-band per Step 5) ---"
cat <<EOF >&2

# 1. Recompile compiler.cpp with -DRC_ORACLE=1 (force the dispatcher block in)
#
#    The exact command line lives in $JIT_BUILD/CMakeFiles/jit.dir/build.make
#    under the 'compiler.cpp.o:' rule. Extract that line, append -DRC_ORACLE=1,
#    redirect output to:
#      $COMPILER_RC_ORACLE_OBJ
#
#    Or, minimal direct compile (mirrors Phoenix CMakeLists SHARED_FLAGS):
#
clang++ -std=c++20 -O2 -g -fPIC \\
    -DPy_BUILD_CORE -DPy_BUILD_CORE_MODULE -DENABLE_LIGHTWEIGHT_FRAMES \\
    -DRC_ORACLE=1 -DPHOENIX_ASM \\
    -I$CPYTHON_ROOT/Python \\
    -I$JIT_BUILD/generated \\
    -I$CPYTHON_ROOT \\
    -I$CPYTHON_ROOT/Include \\
    -I$CPYTHON_ROOT/Include/internal \\
    -I$CPYTHON_ROOT/Python/jit_deps/parallel-hashmap \\
    -I$CPYTHON_ROOT/Python/jit_deps/usdt/.. \\
    -I$CPYTHON_ROOT/Python/jit_deps/fmt/include \\
    -I$CPYTHON_ROOT/Python/jit_deps/asmjit/src \\
    -c $CPYTHON_ROOT/Python/jit/compiler.cpp \\
    -o $COMPILER_RC_ORACLE_OBJ

# 2. Re-link python with the swapped compiler.cpp.o + libphoenix_rc_oracle.a
#
#    The original link command lives in $CPYTHON_ROOT/Makefile under
#    \$(BUILDPYTHON). Capture it via:
#
#      make -n python | grep -E '^(\\s+)?\\\$\\(CXX\\)' | head -1
#
#    Then replace the production compiler.cpp.o path
#    ($JIT_BUILD/CMakeFiles/jit.dir/jit/compiler.cpp.o) with
#    $COMPILER_RC_ORACLE_OBJ AND insert
#    '-Wl,--whole-archive $ORACLE_BUILD/libphoenix_rc_oracle.a -Wl,--no-whole-archive'
#    immediately before the existing -Wl,--start-group of \$(JIT_LIBS).
#    Output to:
#      $ORACLE_PYTHON
#
#    --whole-archive is required because rc_oracle_run is not referenced
#    outside the dispatcher block; without --whole-archive the linker drops
#    the lib's rc_oracle_run, causing undefined-symbol error in compiler.cpp.

# 3. Verify the build (Step 5 falsifier checks):
#
#    nm $CPYTHON_ROOT/python | grep rc_oracle               # MUST be EMPTY
#    nm $ORACLE_PYTHON | grep ' T rc_oracle_run'             # MUST match
#    $ORACLE_PYTHON --version                               # should run
#    RC_ORACLE_USE_CXX=1 $ORACLE_PYTHON -c 'import _cinderx; ...'
#                                                            # should run, uses C++ path

EOF

echo ""
echo "=== build_oracle.sh complete ==="
echo "Built:    $ORACLE_BUILD/libphoenix_rc_oracle.a (rc_oracle_run T-symbol verified)"
echo "Pending:  python_rc_cpp link (Step 5, generalist, out-of-band per supervisor 02:46:08Z)"
exit 0
