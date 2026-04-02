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

int jit_compile_running(void) {
    return jit::getThreadedCompileContext().compileRunning() ? 1 : 0;
}

int jit_compile_can_access_shared_data(void) {
    return jit::getThreadedCompileContext().canAccessSharedData() ? 1 : 0;
}

PyInterpreterState* jit_compile_interpreter(void) {
    return jit::ThreadedCompileContext::interpreter();
}

} /* extern "C" */
