/*
 * reader_c.h -- Pure C interface for ELF note section reading
 *
 * Phase 3D conversion: reader.h C API
 */

#ifndef JIT_ELF_READER_C_H
#define JIT_ELF_READER_C_H

#include "cinderx/Jit/elf/note_c.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Find an ELF section by name.
 *
 * On success, returns 0 and sets *out_data/*out_size to the section contents.
 * If the section is not found, returns 0 with *out_data=NULL, *out_size=0.
 * On error (malformed ELF, unsupported), returns -1.
 */
int jit_elf_find_section(const uint8_t *elf_data, size_t elf_size,
                         const char *name,
                         const uint8_t **out_data, size_t *out_size);

/*
 * Read all ELF notes from a note section's raw bytes.
 *
 * On success, returns 0 and populates *out (caller must free with
 * jit_elf_note_array_free).  On error, returns -1.
 */
int jit_elf_read_note_section(const uint8_t *data, size_t size,
                              JitElfNoteArray *out);

/*
 * Parse a function's code note data from a JitElfNote.
 *
 * On success, returns 0 and populates *out (caller must free with
 * jit_elf_code_note_data_free).  On error, returns -1.
 */
int jit_elf_parse_code_note(const JitElfNote *note,
                            JitElfCodeNoteData *out);

#ifdef __cplusplus
}
#endif

#endif /* JIT_ELF_READER_C_H */
