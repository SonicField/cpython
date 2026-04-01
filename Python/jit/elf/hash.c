/*
 * hash.c -- ELF symbol hash table (pure C)
 *
 * Phase 3D conversion: hash.cpp -> hash.c
 * Standard ELF hash (SysV) with bucket/chain structure.
 */

#include "cinderx/Jit/elf/hash.h"

#include "Python.h"

#include <assert.h>

/* ELF standard hash function */
uint32_t
jit_elf_hash(const char *name) {
    uint32_t h = 0;
    for (; *name; name++) {
        h = (h << 4) + (uint8_t)*name;
        uint32_t g = h & 0xf0000000;
        if (g) {
            h ^= g >> 24;
        }
        h &= ~g;
    }
    return h;
}

void
jit_elf_hashtab_init(JitElfHashTab *ht) {
    ht->buckets = NULL;
    ht->chains = NULL;
    ht->nbuckets = 0;
    ht->nchains = 0;
}

void
jit_elf_hashtab_free(JitElfHashTab *ht) {
    PyMem_RawFree(ht->buckets);
    PyMem_RawFree(ht->chains);
    ht->buckets = NULL;
    ht->chains = NULL;
    ht->nbuckets = ht->nchains = 0;
}

static uint32_t
chase_chain(const JitElfHashTab *ht, uint32_t idx) {
    uint32_t limit = ht->nchains;
    uint32_t count;
    for (count = 0; ht->chains[idx] != 0 && count < limit; ++count) {
        idx = ht->chains[idx];
    }
    assert(count < limit);
    return idx;
}

void
jit_elf_hashtab_build(JitElfHashTab *ht,
                      const JitElfSymTab *syms,
                      const JitElfStrTab *strings) {
    size_t nsyms = jit_elf_symtab_size(syms);
    uint32_t nb = (uint32_t)(nsyms / 2);
    if (nb == 0) nb = 1;

    ht->nbuckets = nb;
    ht->nchains = (uint32_t)nsyms;
    ht->buckets = (uint32_t *)PyMem_RawCalloc(nb, sizeof(uint32_t));
    ht->chains = (uint32_t *)PyMem_RawCalloc(nsyms, sizeof(uint32_t));

    /* Skip element zero (undefined symbol) */
    for (size_t i = 1; i < nsyms; i++) {
        const JitElfSymbol *sym = jit_elf_symtab_get(syms, i);
        const char *name = jit_elf_strtab_string_at(strings, sym->name_offset);
        uint32_t bucket_idx = jit_elf_hash(name) % nb;

        uint32_t first = ht->buckets[bucket_idx];
        if (first == 0) {
            ht->buckets[bucket_idx] = (uint32_t)i;
        } else {
            ht->chains[chase_chain(ht, first)] = (uint32_t)i;
        }
    }
}

size_t
jit_elf_hashtab_size_bytes(const JitElfHashTab *ht) {
    return (sizeof(uint32_t) * 2) +
           ((size_t)ht->nbuckets * sizeof(uint32_t)) +
           ((size_t)ht->nchains * sizeof(uint32_t));
}
