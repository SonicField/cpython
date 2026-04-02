/*
 * writer_c.h -- Pure C interface for ELF writer
 *
 * Phase 3D conversion: writer.h C API
 */

#ifndef JIT_ELF_WRITER_C_H
#define JIT_ELF_WRITER_C_H

#include "Python.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* A code entry to write into an ELF file. */
typedef struct {
    PyCodeObject *code;             /* for bytecode hash */
    const uint8_t *compiled_code;   /* compiled machine code */
    size_t compiled_code_size;
    void *normal_entry;             /* vectorcall entry point */
    void *static_entry;             /* static entry point (NULL if none) */
    const char *func_name;          /* borrowed, null-terminated */
    const char *file_name;          /* borrowed, null-terminated */
    size_t lineno;
} JitElfCodeEntry;

/*
 * Write code entries to an ELF shared library in memory.
 *
 * On success, returns 0 and sets *out_data / *out_size to a newly
 * allocated buffer (caller frees with PyMem_RawFree).
 * On error, returns -1.
 */
int jit_elf_write_entries(const JitElfCodeEntry *entries, size_t count,
                          uint8_t **out_data, size_t *out_size);

#ifdef __cplusplus
}
#endif

#endif /* JIT_ELF_WRITER_C_H */
