// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/Jit/disassembler_c.h"

#ifdef __cplusplus

#include <ostream>
#include <sstream>

namespace jit {

// C++ Disassembler — delegates to C implementation (disassembler_c.c).
struct Disassembler {
  Disassembler(const char* buf, size_t size) {
    jit_disasm_init(&impl_, buf, size);
  }

  void disassembleOne(std::ostream& os) {
    printToStream(os, [&](FILE* f) { jit_disasm_one(&impl_, f); });
  }

  void disassembleAll(std::ostream& os) {
    printToStream(os, [&](FILE* f) { jit_disasm_all(&impl_, f); });
  }

  const char* cursor() const {
    return jit_disasm_cursor(&impl_);
  }

  void setPrintAddr(bool print) {
    jit_disasm_set_print_addr(&impl_, print ? 1 : 0);
  }

  void setPrintInstBytes(bool print) {
    jit_disasm_set_print_instr_bytes(&impl_, print ? 1 : 0);
  }

 private:
  JitDisassembler impl_;

  template <typename Fn>
  void printToStream(std::ostream& out, Fn&& fn) {
    char buf[8192];
    FILE* f = fmemopen(buf, sizeof(buf), "w");
    if (!f) return;
    fn(f);
    fclose(f);
    out << buf;
  }
};

} // namespace jit

#endif /* __cplusplus */
