// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/Jit/elf/string.h"
#include "cinderx/Jit/elf/symbol.h"

#include <stdint.h>
#include <stddef.h>

/* ---- C types and API (implemented in hash.c) ---- */

typedef struct {
    uint32_t *buckets;
    uint32_t *chains;
    uint32_t nbuckets;
    uint32_t nchains;
} JitElfHashTab;

#ifdef __cplusplus
extern "C" {
#endif

uint32_t jit_elf_hash(const char *name);
void jit_elf_hashtab_init(JitElfHashTab *ht);
void jit_elf_hashtab_free(JitElfHashTab *ht);
void jit_elf_hashtab_build(JitElfHashTab *ht,
                           const JitElfSymTab *syms,
                           const JitElfStrTab *strings);
size_t jit_elf_hashtab_size_bytes(const JitElfHashTab *ht);

#ifdef __cplusplus
} /* extern "C" */
#endif

#ifdef __cplusplus
#include <cstdint>
#include <span>

namespace jit::elf {

// ELF standard hash function.
constexpr uint32_t hash(const char* name) {
  uint32_t h = 0;
  for (; *name; name++) {
    h = (h << 4) + *name;
    uint32_t g = h & 0xf0000000;
    if (g) {
      h ^= g >> 24;
    }
    h &= ~g;
  }
  return h;
}

// C++ wrapper around JitElfHashTab.
class HashTable {
 public:
  HashTable() { jit_elf_hashtab_init(&tab_); }
  ~HashTable() { jit_elf_hashtab_free(&tab_); }

  void build(const SymbolTable& syms, const StringTable& strings) {
    jit_elf_hashtab_build(&tab_, syms.c_tab(), strings.c_tab());
  }

  std::span<const uint32_t> buckets() const {
    return {tab_.buckets, tab_.nbuckets};
  }

  std::span<const uint32_t> chains() const {
    return {tab_.chains, tab_.nchains};
  }

  size_t size_bytes() const {
    return jit_elf_hashtab_size_bytes(&tab_);
  }

 private:
  JitElfHashTab tab_;
};

} // namespace jit::elf
#endif
