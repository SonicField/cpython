/*
 * jit_log.c — JIT chronological event log
 *
 * Real-time time-series log of JIT events: compilation, deoptimization,
 * backoff, and global deopt. Events are written to stderr (or a file
 * via JIT_LOG env var) as they happen, chronologically ordered.
 *
 * This is a LOG, not a dump — events stream in real time with timestamps,
 * enabling time-series analysis of JIT behavior.
 *
 * Enable: set JIT_LOG=1 (stderr) or JIT_LOG=/path/to/file
 */

#include "Python.h"
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>

/* Log output file — NULL means logging disabled */
static FILE *jit_log_file = NULL;
static int jit_log_initialized = 0;
static uint64_t jit_log_start_ns = 0;

/* Monotonic clock in nanoseconds */
static uint64_t
jit_log_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* Initialize logging — called lazily on first event */
static void
jit_log_init(void) {
    if (jit_log_initialized) return;
    jit_log_initialized = 1;

    const char *env = getenv("JIT_LOG");
    if (!env || !env[0]) return;

    if (strcmp(env, "1") == 0 || strcmp(env, "stderr") == 0) {
        jit_log_file = stderr;
    } else {
        jit_log_file = fopen(env, "w");
    }

    if (jit_log_file) {
        jit_log_start_ns = jit_log_now_ns();
        fprintf(jit_log_file, "JIT_LOG_START %llu\n",
                (unsigned long long)jit_log_start_ns);
        fflush(jit_log_file);
    }
}

/* Shutdown — called from module cleanup */
void
jit_log_shutdown(void) {
    if (jit_log_file && jit_log_file != stderr) {
        fclose(jit_log_file);
    }
    jit_log_file = NULL;
    jit_log_initialized = 0;
}

/* ---- Event emitters ---- */

/* Log a compilation event */
void
jit_log_compile(const char *func_name, int is_force_compile,
                size_t code_size) {
    if (!jit_log_initialized) jit_log_init();
    if (!jit_log_file) return;

    uint64_t elapsed_us = (jit_log_now_ns() - jit_log_start_ns) / 1000;
    fprintf(jit_log_file,
            "%llu COMPILE %s force=%d size=%zu\n",
            (unsigned long long)elapsed_us,
            func_name, is_force_compile, code_size);
    fflush(jit_log_file);
}

/* Log a deoptimization event */
void
jit_log_deopt(const char *func_name, size_t deopt_idx,
              const char *reason) {
    if (!jit_log_initialized) jit_log_init();
    if (!jit_log_file) return;

    uint64_t elapsed_us = (jit_log_now_ns() - jit_log_start_ns) / 1000;
    fprintf(jit_log_file,
            "%llu DEOPT %s idx=%zu reason=%s\n",
            (unsigned long long)elapsed_us,
            func_name, deopt_idx, reason ? reason : "unknown");
    fflush(jit_log_file);
}

/* Log a deopt backoff event (function permanently deopted) */
void
jit_log_backoff(const char *func_name, int guard_failures) {
    if (!jit_log_initialized) jit_log_init();
    if (!jit_log_file) return;

    uint64_t elapsed_us = (jit_log_now_ns() - jit_log_start_ns) / 1000;
    fprintf(jit_log_file,
            "%llu BACKOFF %s guard_failures=%d\n",
            (unsigned long long)elapsed_us,
            func_name, guard_failures);
    fflush(jit_log_file);
}

/* Log a global deopt event (all functions deopted) */
void
jit_log_global_deopt(int num_functions) {
    if (!jit_log_initialized) jit_log_init();
    if (!jit_log_file) return;

    uint64_t elapsed_us = (jit_log_now_ns() - jit_log_start_ns) / 1000;
    fprintf(jit_log_file,
            "%llu GLOBAL_DEOPT count=%d\n",
            (unsigned long long)elapsed_us,
            num_functions);
    fflush(jit_log_file);
}

/* Log a recompilation event (function re-attached after deopt) */
void
jit_log_reattach(const char *func_name) {
    if (!jit_log_initialized) jit_log_init();
    if (!jit_log_file) return;

    uint64_t elapsed_us = (jit_log_now_ns() - jit_log_start_ns) / 1000;
    fprintf(jit_log_file,
            "%llu REATTACH %s\n",
            (unsigned long long)elapsed_us,
            func_name);
    fflush(jit_log_file);
}
