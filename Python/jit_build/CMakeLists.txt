cmake_minimum_required(VERSION 3.12)
project(phoenix_jit CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(CPYTHON_ROOT "${CMAKE_SOURCE_DIR}/../.." ABSOLUTE)
set(JIT_ROOT ${CPYTHON_ROOT}/Python)

set(SHARED_FLAGS "-DPy_BUILD_CORE -DPy_BUILD_CORE_MODULE -DENABLE_LIGHTWEIGHT_FRAMES -fPIC -Wall -Wextra -Wno-c99-designator -Wno-c++11-narrowing -Wno-cast-function-type-mismatch -Wno-deprecated-declarations -Wno-missing-field-initializers -Wno-null-pointer-subtraction -Wno-sign-compare -Wno-unknown-pragmas -Wno-unused-function -Wno-unused-parameter")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SHARED_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SHARED_FLAGS}")

# Generated header directory
set(GENERATED_HEADER_DIR ${CMAKE_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GENERATED_HEADER_DIR})

##############################################################################
# Dependencies

# asmjit (or phoenix-asm replacement)
option(PHOENIX_ASM "Use phoenix-asm instead of asmjit" OFF)
if(NOT PHOENIX_ASM)
  set(ASMJIT_STATIC TRUE)
  add_subdirectory(${JIT_ROOT}/jit_deps/asmjit ${CMAKE_BINARY_DIR}/_deps/asmjit-build)
else()
  add_definitions(-DPHOENIX_ASM)
endif()

# fmt
add_subdirectory(${JIT_ROOT}/jit_deps/fmt ${CMAKE_BINARY_DIR}/_deps/fmt-build)

# usdt header
set(USDT_DIR ${GENERATED_HEADER_DIR}/usdt)
file(MAKE_DIRECTORY ${USDT_DIR})
file(COPY ${JIT_ROOT}/jit_deps/usdt/usdt.h DESTINATION ${USDT_DIR})

# zlib
find_package(ZLIB)

##############################################################################
# Include paths

include_directories(
  ${JIT_ROOT}                          # so "cinderx/Jit/foo.h" resolves via symlinks
  ${GENERATED_HEADER_DIR}              # generated headers
  ${CPYTHON_ROOT}                      # pyconfig.h lives in CPython root
  ${CPYTHON_ROOT}/Include              # CPython public headers
  ${CPYTHON_ROOT}/Include/internal     # CPython internal headers
  ${JIT_ROOT}/jit_deps/parallel-hashmap  # phmap headers
  ${JIT_ROOT}/jit_deps/usdt/..         # usdt parent for <usdt/usdt.h>
)

##############################################################################
# Generate Interpreter headers (copy version-specific ones)

set(INTERP_SRC_DIR ${JIT_ROOT}/jit_interp/3.12)
set(GEN_INTERP_DIR ${GENERATED_HEADER_DIR}/cinderx/Interpreter)
file(MAKE_DIRECTORY ${GEN_INTERP_DIR})

foreach(HEADER cinderx_opcode_targets.h cinder_opcode.h cinder_opcode_ids.h cinder_opcode_metadata.h)
  if(EXISTS ${INTERP_SRC_DIR}/${HEADER})
    file(READ ${INTERP_SRC_DIR}/${HEADER} FILE_CONTENTS)
    file(WRITE ${GEN_INTERP_DIR}/${HEADER} "${FILE_CONTENTS}")
  endif()
endforeach()

##############################################################################
# Common/

file(GLOB_RECURSE COMMON_SOURCES ${JIT_ROOT}/jit_common/*.cpp ${JIT_ROOT}/jit_common/*.c)
add_library(common ${COMMON_SOURCES})
if(NOT PHOENIX_ASM)
  target_link_libraries(common PRIVATE asmjit::asmjit fmt::fmt)
else()
  target_link_libraries(common PRIVATE fmt::fmt)
endif()
if(ZLIB_FOUND)
  target_link_libraries(common PRIVATE ZLIB::ZLIB)
endif()

##############################################################################
# UpstreamBorrow/

set(BORROWED_C ${JIT_ROOT}/jit_support/UpstreamBorrow/borrowed-3.12.gen_cached.c)
if(EXISTS ${BORROWED_C})
  add_library(borrowed ${BORROWED_C})
else()
  set(DUMMY_BORROWED "${GENERATED_HEADER_DIR}/dummy-borrowed.c")
  file(WRITE ${DUMMY_BORROWED} "")
  add_library(borrowed ${DUMMY_BORROWED})
endif()

##############################################################################
# Interpreter/

set(INTERP_SOURCES
  ${JIT_ROOT}/jit_interp/cinder_opcode.c
  ${JIT_ROOT}/jit_interp/interpreter_base.c
  ${JIT_ROOT}/jit_interp/iter_helpers.c
)
add_library(interpreter ${INTERP_SOURCES})
target_include_directories(interpreter PUBLIC ${INTERP_SRC_DIR}/Includes)

##############################################################################
# Jit/

file(GLOB_RECURSE JIT_SOURCES ${JIT_ROOT}/jit/*.cpp ${JIT_ROOT}/jit/*.c)
list(FILTER JIT_SOURCES EXCLUDE REGEX ".*\.gen_cached\.c")
set(JIT_GEN_CACHED "${JIT_ROOT}/jit/generators_borrowed_3.12.gen_cached.c")
if(EXISTS ${JIT_GEN_CACHED})
  list(APPEND JIT_SOURCES ${JIT_GEN_CACHED})
endif()
# Exclude jit_build directory from JIT sources
list(FILTER JIT_SOURCES EXCLUDE REGEX ".*/jit_build/.*")
# Exclude jit_deps directory
list(FILTER JIT_SOURCES EXCLUDE REGEX ".*/jit_deps/.*")
# Exclude jit_common, jit_stubs, jit_interp, jit_support (separate targets)
list(FILTER JIT_SOURCES EXCLUDE REGEX ".*/jit_common/.*")
list(FILTER JIT_SOURCES EXCLUDE REGEX ".*/jit_stubs/.*")
list(FILTER JIT_SOURCES EXCLUDE REGEX ".*/jit_interp/.*")
list(FILTER JIT_SOURCES EXCLUDE REGEX ".*/jit_support/.*")
# Exclude phoenix-asm test harness (uses real asmjit for comparison)
list(FILTER JIT_SOURCES EXCLUDE REGEX ".*/phoenix_asm/test_.*")
# Exclude standalone C unit tests (have own main(), conflict with python binary)
list(FILTER JIT_SOURCES EXCLUDE REGEX ".*/hir/test_.*")
# When not using phoenix-asm AND not on ARM64, exclude phoenix_asm sources.
# ARM64 always uses the phoenix_asm ARM64 backend (phx_a64_* functions).
if(NOT PHOENIX_ASM AND NOT "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
  list(FILTER JIT_SOURCES EXCLUDE REGEX ".*/phoenix_asm/.*")
endif()

add_library(jit ${JIT_SOURCES})
if(NOT PHOENIX_ASM)
  target_link_libraries(jit PRIVATE asmjit::asmjit interpreter fmt::fmt)
else()
  target_link_libraries(jit PRIVATE interpreter fmt::fmt)
endif()

##############################################################################
# Top-level phoenix_jit library

set(SUPPORT_SOURCES
  ${JIT_ROOT}/jit_support/phoenix_init.cpp
  ${JIT_ROOT}/jit_support/module_state.cpp
  ${JIT_ROOT}/jit_support/module_c_state.cpp
  ${JIT_ROOT}/jit_support/python_runtime.c
  ${JIT_ROOT}/jit_support/phoenix_stubs.c
)

add_library(${PROJECT_NAME} STATIC ${SUPPORT_SOURCES})
if(NOT PHOENIX_ASM)
  target_link_libraries(${PROJECT_NAME} PRIVATE
    borrowed common interpreter jit
    asmjit::asmjit fmt::fmt)
else()
  target_link_libraries(${PROJECT_NAME} PRIVATE
    borrowed common interpreter jit
    fmt::fmt)
endif()
