// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <stddef.h>
#include <stdint.h>

/* Symbol flags */
#define JIT_ELF_SYM_GLOBAL 0x10
#define JIT_ELF_SYM_FUNC   0x02

/* ---- C API (implemented in symbol.c) ---- */
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

#ifdef __cplusplus
extern "C" {
#endif

void jit_elf_symtab_init(JitElfSymTab *st);
void jit_elf_symtab_free(JitElfSymTab *st);
void jit_elf_symtab_insert(JitElfSymTab *st, const JitElfSymbol *sym);
const JitElfSymbol *jit_elf_symtab_get(const JitElfSymTab *st, size_t idx);
size_t jit_elf_symtab_size(const JitElfSymTab *st);
const uint8_t *jit_elf_symtab_data(const JitElfSymTab *st);
size_t jit_elf_symtab_data_size(const JitElfSymTab *st);

#ifdef __cplusplus
} /* extern "C" */
#endif

#ifdef __cplusplus
#include <span>

namespace jit::elf {

constexpr uint8_t kGlobal = JIT_ELF_SYM_GLOBAL;
constexpr uint8_t kFunc = JIT_ELF_SYM_FUNC;

using Symbol = JitElfSymbol;

class SymbolTable {
 public:
  SymbolTable() { jit_elf_symtab_init(&tab_); }
  ~SymbolTable() { jit_elf_symtab_free(&tab_); }

  template <class... Args>
  void insert(Args&&... args) {
    Symbol sym{std::forward<Args>(args)...};
    jit_elf_symtab_insert(&tab_, &sym);
  }

  const Symbol& operator[](size_t idx) const {
    return *jit_elf_symtab_get(&tab_, idx);
  }

  std::span<const std::byte> bytes() const {
    return std::as_bytes(
        std::span<const uint8_t>{jit_elf_symtab_data(&tab_),
                                 jit_elf_symtab_data_size(&tab_)});
  }

  size_t size() const { return jit_elf_symtab_size(&tab_); }

 private:
  JitElfSymTab tab_;
};

} // namespace jit::elf
#endif
