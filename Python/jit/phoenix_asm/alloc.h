/*
 * phoenix-asm: Code Allocator
 *
 * Manages executable memory for JIT-compiled code via mmap/munmap.
 * Replaces asmjit::JitRuntime.
 *
 * Linux only. Works on both x86_64 and ARM64.
 */

#ifndef PHX_ALLOC_H
#define PHX_ALLOC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque runtime handle */
typedef struct PhxRuntime PhxRuntime;

/*
 * Create a new runtime instance.
 * Returns NULL on failure (allocation error).
 */
PhxRuntime* phx_runtime_create(void);

/*
 * Destroy the runtime and release all associated memory.
 * All code regions allocated through this runtime become invalid.
 * Passing NULL is a no-op.
 */
void phx_runtime_destroy(PhxRuntime* rt);

/*
 * Copy finalized machine code into a new executable memory region.
 *
 * Parameters:
 *   rt        - runtime instance
 *   code      - pointer to the machine code bytes to copy
 *   code_size - size of the machine code in bytes (must be > 0)
 *   out_size  - if non-NULL, receives the total allocated size
 *               (page-aligned, >= code_size)
 *
 * Returns a pointer to the executable code, or NULL on failure.
 *
 * On ARM64, an instruction cache flush (__builtin___clear_cache) is
 * performed on the allocated region before returning.
 */
void* phx_runtime_add(PhxRuntime* rt, const void* code, size_t code_size,
                       size_t* out_size);

/*
 * Release a code region previously allocated by phx_runtime_add().
 *
 * Parameters:
 *   rt       - runtime instance
 *   code_ptr - pointer returned by a prior phx_runtime_add() call.
 *              Passing NULL is a no-op.
 *
 * After this call, code_ptr is invalid and must not be executed.
 */
void phx_runtime_release(PhxRuntime* rt, void* code_ptr);

/*
 * Check whether ptr falls inside any allocation owned by this runtime.
 * Returns 1 if contained, 0 otherwise.
 */
int phx_runtime_contains(const PhxRuntime* rt, const void* ptr);

/*
 * Return the total number of bytes currently allocated (sum of all
 * page-aligned allocation sizes that have not been released).
 */
size_t phx_runtime_used_bytes(const PhxRuntime* rt);

/*
 * Return the number of live (non-released) code regions.
 */
uint32_t phx_runtime_alloc_count(const PhxRuntime* rt);

#ifdef __cplusplus
}
#endif

#endif /* PHX_ALLOC_H */
