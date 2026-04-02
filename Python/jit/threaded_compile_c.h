/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C-compatible compile guard — replaces C++ ThreadedCompileSerialize.
 * Delegates to the same underlying std::recursive_mutex used by C++ code,
 * ensuring both C and C++ paths are synchronized.
 *
 * Pattern 7 in docs/phase3d-conversion-patterns.md
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/* Lock/unlock the global JIT compile mutex.
 * These call through to ThreadedCompileContext::lock()/unlock(),
 * sharing the same std::recursive_mutex as ThreadedCompileSerialize. */
void jit_compile_lock(void);
void jit_compile_unlock(void);

/* __attribute__((cleanup)) helper for scope-based guard.
 * The cleanup function receives a pointer to a dummy variable;
 * it just calls jit_compile_unlock(). */
static inline void jit_compile_unlock_cleanup(int *dummy) {
    (void)dummy;
    jit_compile_unlock();
}

/* Usage:
 *   JIT_COMPILE_GUARD();
 *   // ... protected code ...
 *   // mutex released automatically at scope exit
 *
 * Equivalent to C++: ThreadedCompileSerialize guard;
 */
#define JIT_COMPILE_GUARD() \
    jit_compile_lock(); \
    __attribute__((cleanup(jit_compile_unlock_cleanup))) \
    int _jit_compile_guard = 0

#ifdef __cplusplus
} /* extern "C" */
#endif
