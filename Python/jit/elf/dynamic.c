/*
 * dynamic.c -- ELF dynamic section table (pure C)
 *
 * Phase 3D conversion: dynamic.cpp -> dynamic.c
 */

#include "Python.h"

#include <stdint.h>
#include <string.h>
#include <assert.h>

/* Dyn entry — must match dynamic.h layout */
typedef struct {
    uint64_t tag;
    uint64_t val;
} JitElfDyn;

typedef struct {
    JitElfDyn *dyns;
    size_t len;
    size_t cap;
} JitElfDynTab;

void
jit_elf_dyntab_init(JitElfDynTab *dt) {
    dt->cap = 8;
    dt->dyns = (JitElfDyn *)PyMem_RawCalloc(dt->cap, sizeof(JitElfDyn));
    /* Table must always end with a null dynamic item (tag=0, val=0) */
    dt->len = 1;
}

void
jit_elf_dyntab_free(JitElfDynTab *dt) {
    PyMem_RawFree(dt->dyns);
    dt->dyns = NULL;
    dt->len = dt->cap = 0;
}

void
jit_elf_dyntab_insert(JitElfDynTab *dt, uint64_t tag, uint64_t val) {
    if (dt->len >= dt->cap) {
        size_t new_cap = dt->cap * 2;
        dt->dyns = (JitElfDyn *)PyMem_RawRealloc(
            dt->dyns, new_cap * sizeof(JitElfDyn));
        dt->cap = new_cap;
    }
    /* Insert before the trailing null entry */
    assert(dt->len >= 1);
    dt->dyns[dt->len] = dt->dyns[dt->len - 1];  /* move null to end */
    dt->dyns[dt->len - 1].tag = tag;
    dt->dyns[dt->len - 1].val = val;
    dt->len++;
}

const uint8_t *
jit_elf_dyntab_data(const JitElfDynTab *dt) {
    return (const uint8_t *)dt->dyns;
}

size_t
jit_elf_dyntab_data_size(const JitElfDynTab *dt) {
    return dt->len * sizeof(JitElfDyn);
}
