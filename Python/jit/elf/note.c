/*
 * note.c -- C implementation of ELF note types
 *
 * Phase 3D conversion: note.h inline methods -> note.c
 */

#include "cinderx/Jit/elf/note_c.h"

#include "Python.h"

#include <assert.h>
#include <string.h>

/* ---- Utility ---- */

static size_t
round_up(size_t n, size_t align)
{
    return (n + align - 1) & ~(align - 1);
}

static char *
strdup_raw(const char *s)
{
    if (s == NULL) {
        return NULL;
    }
    size_t len = strlen(s) + 1;
    char *copy = (char *)PyMem_RawMalloc(len);
    if (copy != NULL) {
        memcpy(copy, s, len);
    }
    return copy;
}

/* Duplicate binary data (may contain embedded NULs).
 * Always adds a trailing NUL for safety. */
static char *
memdup_raw(const char *s, size_t len)
{
    if (s == NULL) {
        return NULL;
    }
    char *copy = (char *)PyMem_RawMalloc(len + 1);
    if (copy != NULL) {
        memcpy(copy, s, len);
        copy[len] = '\0';
    }
    return copy;
}

/* ---- Note ---- */

void
jit_elf_note_init(JitElfNote *note)
{
    note->name = NULL;
    note->desc = NULL;
    note->desc_len = 0;
    note->type = 0;
}

void
jit_elf_note_set(JitElfNote *note, const char *name, const char *desc,
                 uint32_t type)
{
    PyMem_RawFree(note->name);
    PyMem_RawFree(note->desc);
    note->name = strdup_raw(name ? name : "");
    note->desc = strdup_raw(desc ? desc : "");
    note->desc_len = desc ? strlen(desc) : 0;
    note->type = type;
}

void
jit_elf_note_set_bin(JitElfNote *note, const char *name,
                     const char *desc, size_t desc_len, uint32_t type)
{
    PyMem_RawFree(note->name);
    PyMem_RawFree(note->desc);
    note->name = strdup_raw(name ? name : "");
    note->desc = memdup_raw(desc, desc_len);
    note->desc_len = desc_len;
    note->type = type;
}

void
jit_elf_note_free(JitElfNote *note)
{
    PyMem_RawFree(note->name);
    PyMem_RawFree(note->desc);
    note->name = NULL;
    note->desc = NULL;
    note->desc_len = 0;
}

size_t
jit_elf_note_size_bytes(const JitElfNote *note)
{
    /* Three uint32 fields: name_size, desc_size, type */
    size_t s = sizeof(uint32_t) * 3;

    /* Name is assumed present, includes NUL terminator, padded to 4 bytes */
    if (note->name != NULL) {
        s += round_up(strlen(note->name) + 1, 4);
    }

    /* Descriptor may be empty */
    if (note->desc_len > 0) {
        s += round_up(note->desc_len + 1, 4);
    }

    return s;
}

/* ---- NoteArray ---- */

void
jit_elf_note_array_init(JitElfNoteArray *arr)
{
    arr->notes = NULL;
    arr->len = 0;
    arr->cap = 0;
}

void
jit_elf_note_array_free(JitElfNoteArray *arr)
{
    for (size_t i = 0; i < arr->len; i++) {
        jit_elf_note_free(&arr->notes[i]);
    }
    PyMem_RawFree(arr->notes);
    arr->notes = NULL;
    arr->len = 0;
    arr->cap = 0;
}

void
jit_elf_note_array_insert(JitElfNoteArray *arr,
                          const char *name, const char *desc,
                          uint32_t type)
{
    if (arr->len >= arr->cap) {
        size_t new_cap = arr->cap ? arr->cap * 2 : 8;
        arr->notes = (JitElfNote *)PyMem_RawRealloc(
            arr->notes, new_cap * sizeof(JitElfNote));
        arr->cap = new_cap;
    }
    JitElfNote *note = &arr->notes[arr->len];
    jit_elf_note_init(note);
    jit_elf_note_set(note, name, desc, type);
    arr->len++;
}

void
jit_elf_note_array_insert_bin(JitElfNoteArray *arr,
                              const char *name,
                              const char *desc, size_t desc_len,
                              uint32_t type)
{
    if (arr->len >= arr->cap) {
        size_t new_cap = arr->cap ? arr->cap * 2 : 8;
        arr->notes = (JitElfNote *)PyMem_RawRealloc(
            arr->notes, new_cap * sizeof(JitElfNote));
        arr->cap = new_cap;
    }
    JitElfNote *note = &arr->notes[arr->len];
    jit_elf_note_init(note);
    jit_elf_note_set_bin(note, name, desc, desc_len, type);
    arr->len++;
}

size_t
jit_elf_note_array_size_bytes(const JitElfNoteArray *arr)
{
    size_t sum = 0;
    for (size_t i = 0; i < arr->len; i++) {
        sum += jit_elf_note_size_bytes(&arr->notes[i]);
    }
    return sum;
}

size_t
jit_elf_note_array_len(const JitElfNoteArray *arr)
{
    return arr->len;
}

const JitElfNote *
jit_elf_note_array_get(const JitElfNoteArray *arr, size_t index)
{
    assert(index < arr->len);
    return &arr->notes[index];
}

/* ---- CodeNoteData ---- */

void
jit_elf_code_note_data_init(JitElfCodeNoteData *data)
{
    data->file_name = NULL;
    data->lineno = 0;
    data->hash = 0;
    data->size = 0;
    data->normal_entry_offset = 0;
    data->static_entry_offset = JIT_ELF_INVALID_STATIC_OFFSET;
    data->has_static_entry = 0;
}

void
jit_elf_code_note_data_free(JitElfCodeNoteData *data)
{
    PyMem_RawFree(data->file_name);
    data->file_name = NULL;
}
