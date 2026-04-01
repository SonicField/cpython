/*
 * string.c -- ELF string table (pure C)
 *
 * Phase 3D conversion: string.cpp -> string.c
 */

#include "Python.h"

#include <stdint.h>
#include <string.h>

typedef struct {
    uint8_t *data;
    size_t len;
    size_t cap;
} JitElfStrTab;

void
jit_elf_strtab_init(JitElfStrTab *st) {
    st->cap = 64;
    st->data = (uint8_t *)PyMem_RawMalloc(st->cap);
    st->data[0] = 0;  /* all string tables begin with NUL */
    st->len = 1;
}

void
jit_elf_strtab_free(JitElfStrTab *st) {
    PyMem_RawFree(st->data);
    st->data = NULL;
    st->len = st->cap = 0;
}

uint32_t
jit_elf_strtab_insert(JitElfStrTab *st, const char *s, size_t slen) {
    size_t start_off = st->len;
    size_t needed = st->len + slen + 1;
    if (needed > st->cap) {
        size_t new_cap = st->cap;
        while (new_cap < needed) new_cap *= 2;
        st->data = (uint8_t *)PyMem_RawRealloc(st->data, new_cap);
        st->cap = new_cap;
    }
    memcpy(&st->data[start_off], s, slen);
    st->data[start_off + slen] = 0;
    st->len = needed;
    return (uint32_t)start_off;
}

const char *
jit_elf_strtab_string_at(const JitElfStrTab *st, size_t offset) {
    return (const char *)&st->data[offset];
}

const uint8_t *
jit_elf_strtab_data(const JitElfStrTab *st) {
    return st->data;
}

size_t
jit_elf_strtab_size(const JitElfStrTab *st) {
    return st->len;
}
