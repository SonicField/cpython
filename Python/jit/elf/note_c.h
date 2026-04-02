/*
 * note_c.h -- Pure C types for ELF notes
 *
 * Phase 3D conversion: note.h C types
 *
 * A note in an ELF file is a tuple of a string name, an integral type,
 * and an optional descriptor string.
 */

#ifndef JIT_ELF_NOTE_C_H
#define JIT_ELF_NOTE_C_H

#include "Python.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Note ---- */

typedef struct {
    char *name;         /* owned, null-terminated */
    char *desc;         /* owned, may contain embedded NULs (binary data) */
    size_t desc_len;    /* length of desc in bytes (not including trailing NUL) */
    uint32_t type;
} JitElfNote;

void jit_elf_note_init(JitElfNote *note);
void jit_elf_note_set(JitElfNote *note, const char *name, const char *desc,
                      uint32_t type);
void jit_elf_note_set_bin(JitElfNote *note, const char *name,
                          const char *desc, size_t desc_len, uint32_t type);
void jit_elf_note_free(JitElfNote *note);
size_t jit_elf_note_size_bytes(const JitElfNote *note);

/* ---- NoteArray ---- */

typedef struct {
    JitElfNote *notes;
    size_t len;
    size_t cap;
} JitElfNoteArray;

void jit_elf_note_array_init(JitElfNoteArray *arr);
void jit_elf_note_array_free(JitElfNoteArray *arr);
void jit_elf_note_array_insert(JitElfNoteArray *arr,
                               const char *name, const char *desc,
                               uint32_t type);
void jit_elf_note_array_insert_bin(JitElfNoteArray *arr,
                                   const char *name,
                                   const char *desc, size_t desc_len,
                                   uint32_t type);
size_t jit_elf_note_array_size_bytes(const JitElfNoteArray *arr);
size_t jit_elf_note_array_len(const JitElfNoteArray *arr);
const JitElfNote *jit_elf_note_array_get(const JitElfNoteArray *arr,
                                          size_t index);

/* ---- CodeNoteData ---- */

#define JIT_ELF_INVALID_STATIC_OFFSET ((uint32_t)0xFFFFFFFF)

typedef struct {
    char *file_name;            /* owned, null-terminated */
    uint32_t lineno;
    uint32_t hash;
    uint32_t size;
    uint32_t normal_entry_offset;
    uint32_t static_entry_offset;   /* JIT_ELF_INVALID_STATIC_OFFSET if none */
    int has_static_entry;
} JitElfCodeNoteData;

void jit_elf_code_note_data_init(JitElfCodeNoteData *data);
void jit_elf_code_note_data_free(JitElfCodeNoteData *data);

/* Section name constant */
#define JIT_ELF_FUNC_NOTE_SECTION_NAME ".note.pyfunc"

#ifdef __cplusplus
}
#endif

#endif /* JIT_ELF_NOTE_C_H */
