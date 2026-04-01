// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <stddef.h>
#include <stdint.h>

/* ---- C API (implemented in string.c) ---- */
typedef struct {
    uint8_t *data;
    size_t len;
    size_t cap;
} JitElfStrTab;

#ifdef __cplusplus
extern "C" {
#endif

void jit_elf_strtab_init(JitElfStrTab *st);
void jit_elf_strtab_free(JitElfStrTab *st);
uint32_t jit_elf_strtab_insert(JitElfStrTab *st, const char *s, size_t slen);
const char *jit_elf_strtab_string_at(const JitElfStrTab *st, size_t offset);
const uint8_t *jit_elf_strtab_data(const JitElfStrTab *st);
size_t jit_elf_strtab_size(const JitElfStrTab *st);

#ifdef __cplusplus
} /* extern "C" */
#endif

#ifdef __cplusplus
#include <span>
#include <string_view>

namespace jit::elf {

// C++ wrapper around the C string table implementation.
class StringTable {
 public:
  StringTable() { jit_elf_strtab_init(&tab_); }
  ~StringTable() { jit_elf_strtab_free(&tab_); }

  uint32_t insert(std::string_view s) {
    return jit_elf_strtab_insert(&tab_, s.data(), s.size());
  }

  std::string_view string_at(size_t offset) const {
    return jit_elf_strtab_string_at(&tab_, offset);
  }

  std::span<const std::byte> bytes() const {
    return std::as_bytes(
        std::span<const uint8_t>{tab_.data, tab_.len});
  }

 private:
  JitElfStrTab tab_;
};

} // namespace jit::elf
#endif
