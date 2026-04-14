// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Phase 3D: Reduced — simple methods inlined to function.h.
// Remaining: constructor, destructor, setCode, CountInstrs, canDeopt,
// codeFor, count_opcodes (need C++ features).

#include "cinderx/Jit/hir/function.h"
#include "cinderx/Jit/jit_config_c.h"

namespace jit::hir {

#ifndef _LIBCPP_VERSION
static_assert(sizeof(Function) == 48 * kPointerSize);
static_assert(sizeof(CFG) == 5 * kPointerSize);
static_assert(sizeof(BasicBlock) == 20 * kPointerSize);
static_assert(sizeof(Instr) == 5 * kPointerSize);
#endif

Function::Function() {}

Function::~Function() {
  ThreadedCompileSerialize guard;
  code.reset();
  builtins.reset();
  globals.reset();
  prim_args_info.reset();
}

void Function::setCode(BorrowedRef<PyCodeObject> code_2) {
  this->code.reset(code_2);
  uses_runtime_func = usesRuntimeFunc(code_2);
  frameMode = static_cast<FrameMode>(jit_get_config()->frame_mode);
}

std::size_t Function::CountInstrs(InstrPredicate pred) const {
  std::size_t result = 0;
  for (const auto& block : cfg.blocks) {
    for (const auto& instr : block) {
      if (pred(instr)) {
        result++;
      }
    }
  }
  return result;
}

bool Function::canDeopt() const {
  for (const BasicBlock& block : cfg.blocks) {
    for (const Instr& instr : block) {
      if (instr.asDeoptBase()) {
        return true;
      }
    }
  }
  return false;
}

BorrowedRef<PyCodeObject> Function::codeFor(const Instr& instr) const {
  if (instr.IsBeginInlinedFunction()) {
    auto bif = static_cast<const BeginInlinedFunction*>(&instr);
    return bif->func()->func_code;
  }
  if (instr.IsLoadGlobalCached()) {
    auto load_global = static_cast<const LoadGlobalCached*>(&instr);
    return load_global->code();
  }
  if (auto deopt_base = instr.asDeoptBase()) {
    auto fs = deopt_base->frameState();
    return fs != nullptr ? fs->code : nullptr;
  }
  const FrameState* fs = instr.getDominatingFrameState();
  return fs == nullptr ? code : fs->code;
}

OpcodeCounts count_opcodes(const Function& func) {
  OpcodeCounts counts{};
  for (const BasicBlock& block : func.cfg.blocks) {
    for (const Instr& instr : block) {
      counts[static_cast<size_t>(instr.opcode())]++;
    }
  }
  return counts;
}

} // namespace jit::hir
