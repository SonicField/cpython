/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C-compatible JIT logging and assertion macros.
 * Replaces cinderx/Common/log.h for .c files that cannot include
 * the C++ fmt::format-based macros.
 *
 * Usage: #include "cinderx/Common/jit_log_c.h" instead of log.h
 */
#pragma once

#include <stdio.h>
#include <stdlib.h>

/* ---- Assertions ---- */

/* JIT_CHECK_C: abort with file/line/condition if COND is false.
 * Accepts an optional format string + args (printf-style). */
#define JIT_CHECK_C(COND, ...) \
    do { \
        if (!(COND)) { \
            fprintf(stderr, "JIT: %s:%d -- Assertion failed: %s\n", \
                    __FILE__, __LINE__, #COND); \
            JIT_ABORT_C_IMPL(__VA_ARGS__); \
        } \
    } while (0)

/* JIT_ABORT_C: unconditional abort with message. */
#define JIT_ABORT_C(...) \
    do { \
        fprintf(stderr, "JIT: %s:%d -- Abort\n", __FILE__, __LINE__); \
        JIT_ABORT_C_IMPL(__VA_ARGS__); \
    } while (0)

/* Implementation: print optional message then abort.
 * Uses a trick to handle zero or more args after the format. */
#define JIT_ABORT_C_IMPL(...) \
    do { \
        JIT_ABORT_C_MAYBE_PRINT(__VA_ARGS__, ""); \
        fflush(stderr); \
        abort(); \
    } while (0)

/* Helper: print message if format string is non-empty */
#define JIT_ABORT_C_MAYBE_PRINT(FMT, ...) \
    do { \
        const char *_fmt = FMT; \
        if (_fmt[0] != '\0') { \
            fprintf(stderr, _fmt, ##__VA_ARGS__); \
            fprintf(stderr, "\n"); \
        } \
    } while (0)

/* ---- Logging ---- */

/* JIT_LOG_C: printf-style log to stderr.
 * For C files that can't use JIT_LOG (which uses fmt::format). */
#define JIT_LOG_C(FMT, ...) \
    do { \
        fprintf(stderr, "JIT: %s:%d -- ", __FILE__, __LINE__); \
        fprintf(stderr, FMT, ##__VA_ARGS__); \
        fprintf(stderr, "\n"); \
        fflush(stderr); \
    } while (0)

/* ---- Debug-only variants ---- */

#ifdef Py_DEBUG
#define JIT_DCHECK_C(COND, ...) JIT_CHECK_C(COND, ##__VA_ARGS__)
#define JIT_DABORT_C(...) JIT_ABORT_C(__VA_ARGS__)
#else
#define JIT_DCHECK_C(COND, ...) \
    do { if (0) { (void)(COND); } } while (0)
#define JIT_DABORT_C(...) \
    do { if (0) { JIT_ABORT_C(__VA_ARGS__); } } while (0)
#endif
