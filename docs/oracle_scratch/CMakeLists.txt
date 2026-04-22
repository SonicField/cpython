cmake_minimum_required(VERSION 3.12)
project(phoenix_rc_oracle CXX C)

#
# W3 R4 ORACLE — scratch lib that compiles cc4a18e7e5:refcount_insertion.cpp
# (extracted to rc_oracle.cpp) against current Phoenix JIT headers via the
# _rc_oracle_adapter.h shim.
#
# Per supervisor 2026-04-22 02:37:10Z hybrid design:
#   - Production python build: -DRC_ORACLE undefined → dispatcher absent
#   - python_rc_cpp build: -DRC_ORACLE=1 + link libphoenix_rc_oracle.a
#     → dispatcher present, env-var RC_ORACLE_USE_CXX selects path
#
# Per theologian 2026-04-22 02:30:08Z W3 spec:
#   - Oracle pin enforced by SOURCE files being cc4a18e7e5 snapshots
#     (NOT by git checkout — pin is structural in the file content)
#

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(ORACLE_DIR ${CMAKE_CURRENT_SOURCE_DIR} ABSOLUTE)
get_filename_component(CPYTHON_ROOT ${ORACLE_DIR}/../.. ABSOLUTE)
set(JIT_ROOT ${CPYTHON_ROOT}/Python)
set(JIT_BUILD_DIR ${CPYTHON_ROOT}/Python/jit_build/build)

# Mirror Phoenix's include structure (Python/jit_build/CMakeLists.txt:44-52).
include_directories(
  ${JIT_ROOT}
  ${JIT_BUILD_DIR}/generated
  ${CPYTHON_ROOT}
  ${CPYTHON_ROOT}/Include
  ${CPYTHON_ROOT}/Include/internal
  ${JIT_ROOT}/jit_deps/parallel-hashmap
  ${JIT_ROOT}/jit_deps/usdt/..
  ${JIT_ROOT}/jit_deps/fmt/include  # fmt/chrono.h, fmt/format.h, etc.
  ${JIT_ROOT}/jit_deps/asmjit/src   # asmjit/asmjit.h
  ${ORACLE_DIR}                 # for "_rc_oracle_adapter.h" + "rc_oracle.h"
)

# Phoenix's compile flags (mirror Python/jit_build/CMakeLists.txt:11).
set(SHARED_FLAGS
  "-DPy_BUILD_CORE -DPy_BUILD_CORE_MODULE -DENABLE_LIGHTWEIGHT_FRAMES -fPIC \
-Wall -Wextra \
-Wno-c99-designator -Wno-c++11-narrowing -Wno-cast-function-type-mismatch \
-Wno-deprecated-declarations -Wno-missing-field-initializers \
-Wno-null-pointer-subtraction -Wno-sign-compare -Wno-unknown-pragmas \
-Wno-unused-function -Wno-unused-parameter")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SHARED_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SHARED_FLAGS}")

# Force-include _rc_oracle_adapter.h on rc_oracle.cpp's compile so the
# cc4a18e7e5 references to LivenessAnalysis + kEmptyRegSet resolve to the
# adapter (which wraps current HirLivenessState C API). Without this,
# rc_oracle.cpp's `#include "cinderx/Jit/hir/analysis.h"` resolves to HEAD's
# header which DOES NOT define LivenessAnalysis (deleted by R4) → compile
# fails on undefined class.
set_source_files_properties(rc_oracle.cpp PROPERTIES
  COMPILE_FLAGS "-include ${ORACLE_DIR}/_rc_oracle_adapter.h")

# Static lib: cc4a18e7e5 source + adapter + C-linkage entry point.
add_library(phoenix_rc_oracle STATIC
  rc_oracle.cpp                # cc4a18e7e5 snapshot of refcount_insertion.cpp
  _rc_oracle_adapter.cpp       # kEmptyRegSet definition
  rc_oracle_entry.cpp          # extern "C" int rc_oracle_run(void*) wrapper
)

# Falsifier check: when this library is BUILT, libphoenix_rc_oracle.a should
# expose 'rc_oracle_run' as a T (text) symbol with C linkage. Verify after
# build with: nm libphoenix_rc_oracle.a | grep ' T rc_oracle_run'
