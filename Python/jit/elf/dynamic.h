// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <stddef.h>
#include <stdint.h>

/* DynTag constants */
#define JIT_ELF_DYN_NULL    0
#define JIT_ELF_DYN_NEEDED  1
#define JIT_ELF_DYN_HASH    4
#define JIT_ELF_DYN_STRTAB  5
#define JIT_ELF_DYN_SYMTAB  6
#define JIT_ELF_DYN_STRSZ   10
#define JIT_ELF_DYN_SYMENT  11

/* ---- C API (implemented in dynamic.c) ---- */
typedef struct {
    uint64_t tag;
    uint64_t val;
} JitElfDyn;

typedef struct {
    JitElfDyn *dyns;
    size_t len;
    size_t cap;
} JitElfDynTab;

#ifdef __cplusplus
extern "C" {
#endif

void jit_elf_dyntab_init(JitElfDynTab *dt);
void jit_elf_dyntab_free(JitElfDynTab *dt);
void jit_elf_dyntab_insert(JitElfDynTab *dt, uint64_t tag, uint64_t val);
const uint8_t *jit_elf_dyntab_data(const JitElfDynTab *dt);
size_t jit_elf_dyntab_data_size(const JitElfDynTab *dt);

#ifdef __cplusplus
} /* extern "C" */
#endif

#ifdef __cplusplus
#include <span>

namespace jit::elf {

enum class DynTag : uint64_t {
  kNull = JIT_ELF_DYN_NULL,
  kNeeded = JIT_ELF_DYN_NEEDED,
  kHash = JIT_ELF_DYN_HASH,
  kStrtab = JIT_ELF_DYN_STRTAB,
  kSymtab = JIT_ELF_DYN_SYMTAB,
  kStrSz = JIT_ELF_DYN_STRSZ,
  kSymEnt = JIT_ELF_DYN_SYMENT,
};

struct Dyn {
  constexpr Dyn() = default;
  constexpr Dyn(DynTag tag, uint64_t val)
      : tag{tag}, val{val} {}

  DynTag tag{DynTag::kNull};
  uint64_t val{0};
};

class DynamicTable {
 public:
  DynamicTable() { jit_elf_dyntab_init(&tab_); }
  ~DynamicTable() { jit_elf_dyntab_free(&tab_); }

  void insert(DynTag tag, uint64_t val) {
    jit_elf_dyntab_insert(&tab_,
        static_cast<uint64_t>(tag), val);
  }

  std::span<const std::byte> bytes() const {
    return std::as_bytes(
        std::span<const uint8_t>{jit_elf_dyntab_data(&tab_),
                                 jit_elf_dyntab_data_size(&tab_)});
  }

 private:
  JitElfDynTab tab_;
};

} // namespace jit::elf
#endif
