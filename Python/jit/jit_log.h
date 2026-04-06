/*
 * jit_log.h — JIT chronological event log
 *
 * Enable: JIT_LOG=1 (stderr) or JIT_LOG=/path/to/file
 * Events: COMPILE, DEOPT, BACKOFF, GLOBAL_DEOPT, REATTACH
 * Format: <elapsed_us> <event_type> <details...>
 */

#ifndef JIT_LOG_H
#define JIT_LOG_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void jit_log_shutdown(void);

void jit_log_compile(const char *func_name, int is_force_compile,
                     size_t code_size);

void jit_log_deopt(const char *func_name, size_t deopt_idx,
                   const char *reason);

void jit_log_backoff(const char *func_name, int guard_failures);

void jit_log_global_deopt(int num_functions);

void jit_log_reattach(const char *func_name);

#ifdef __cplusplus
}
#endif

#endif /* JIT_LOG_H */
