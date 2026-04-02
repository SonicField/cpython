/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C bridge for ThreadedCompileSerialize.
 * Implements the extern "C" functions declared in threaded_compile_c.h
 * by delegating to the C++ ThreadedCompileContext.
 */

#include "cinderx/Jit/threaded_compile_c.h"
#include "cinderx/Jit/threaded_compile.h"

extern "C" {

void jit_compile_lock(void) {
    jit::getThreadedCompileContext().lock();
}

void jit_compile_unlock(void) {
    jit::getThreadedCompileContext().unlock();
}

} /* extern "C" */
