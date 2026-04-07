// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/Jit/lir/printer_c.h"

#ifdef __cplusplus

#include "cinderx/Jit/lir/block.h"
#include "cinderx/Jit/lir/function.h"
#include "cinderx/Jit/lir/instruction.h"
#include "cinderx/Jit/lir/operand.h"
#include "fmt/ostream.h"

#include <iosfwd>
#include <sstream>

namespace jit::lir {

// C++ Printer class — delegates to C implementation (printer_c.c).
// Wraps FILE*-based C functions for std::ostream callers.
class Printer {
 public:
  Printer() {}

  void print(std::ostream& out, const Function& func) {
    printToStream(out, [&](FILE* f) {
      lir_print_function(f, const_cast<Function*>(&func));
    });
  }

  void print(std::ostream& out, const BasicBlock& block) {
    const JitConfig* cfg = jit_get_config();
    int show_hir = cfg ? cfg->log.lir_origin : 0;
    printToStream(out, [&](FILE* f) {
      lir_print_block(f, const_cast<BasicBlock*>(&block), show_hir);
    });
  }

  void print(std::ostream& out, const Instruction& instr) {
    printToStream(out, [&](FILE* f) {
      lir_print_instruction(f,
          reinterpret_cast<const LirInstruction*>(&instr));
    });
  }

  void print(std::ostream& out, const OperandBase& operand) {
    printToStream(out, [&](FILE* f) {
      lir_print_operand(f,
          reinterpret_cast<const LirOperand*>(&operand));
    });
  }

  void print(std::ostream& out, const MemoryIndirect& memind) {
    printToStream(out, [&](FILE* f) {
      lir_print_memind(f,
          reinterpret_cast<const LirMemoryIndirect*>(&memind));
    });
  }

 private:
  template <typename Fn>
  void printToStream(std::ostream& out, Fn&& fn) {
    char buf[4096];
    FILE* f = fmemopen(buf, sizeof(buf), "w");
    if (!f) return;
    fn(f);
    fclose(f);
    out << buf;
  }
};

inline std::ostream& operator<<(std::ostream& out, const Function& func) {
  Printer().print(out, func);
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const BasicBlock& block) {
  Printer().print(out, block);
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const Instruction& instr) {
  Printer().print(out, instr);
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const OperandBase& operand) {
  Printer().print(out, operand);
  return out;
}

inline std::ostream& operator<<(
    std::ostream& out,
    const MemoryIndirect& memind) {
  Printer().print(out, memind);
  return out;
}

} // namespace jit::lir

template <>
struct fmt::formatter<jit::lir::Function> : fmt::ostream_formatter {};
template <>
struct fmt::formatter<jit::lir::BasicBlock> : fmt::ostream_formatter {};
template <>
struct fmt::formatter<jit::lir::Instruction> : fmt::ostream_formatter {};
template <>
struct fmt::formatter<jit::lir::OperandBase> : fmt::ostream_formatter {};
template <>
struct fmt::formatter<jit::lir::Operand> : fmt::ostream_formatter {};
template <>
struct fmt::formatter<jit::lir::MemoryIndirect> : fmt::ostream_formatter {};

#endif /* __cplusplus */
