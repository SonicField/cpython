/*
 * symbol.c -- ELF symbol table (pure C)
 *
 * Phase 3D conversion: symbol.cpp -> symbol.c
 */

#include "Python.h"

#include <stdint.h>
#include <string.h>

/* Symbol struct — must match symbol.h layout */
typedef struct {
    uint32_t name_offset;
    uint8_t info;
    uint8_t other;
    uint16_t section_index;
    uint64_t address;
    uint64_t size;
} JitElfSymbol;

typedef struct {
    JitElfSymbol *syms;
    size_t len;
    size_t cap;
} JitElfSymTab;

void
jit_elf_symtab_init(JitElfSymTab *st) {
    st->cap = 16;
    st->syms = (JitElfSymbol *)PyMem_RawCalloc(st->cap, sizeof(JitElfSymbol));
    /* Symbol table must always start with an undefined symbol (all zeros) */
    st->len = 1;
}

void
jit_elf_symtab_free(JitElfSymTab *st) {
    PyMem_RawFree(st->syms);
    st->syms = NULL;
    st->len = st->cap = 0;
}

void
jit_elf_symtab_insert(JitElfSymTab *st, const JitElfSymbol *sym) {
    if (st->len >= st->cap) {
        size_t new_cap = st->cap * 2;
        st->syms = (JitElfSymbol *)PyMem_RawRealloc(
            st->syms, new_cap * sizeof(JitElfSymbol));
        st->cap = new_cap;
    }
    st->syms[st->len++] = *sym;
}

const JitElfSymbol *
jit_elf_symtab_get(const JitElfSymTab *st, size_t idx) {
    return &st->syms[idx];
}

size_t
jit_elf_symtab_size(const JitElfSymTab *st) {
    return st->len;
}

const uint8_t *
jit_elf_symtab_data(const JitElfSymTab *st) {
    return (const uint8_t *)st->syms;
}

size_t
jit_elf_symtab_data_size(const JitElfSymTab *st) {
    return st->len * sizeof(JitElfSymbol);
}
