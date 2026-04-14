// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Phase 3D: Reduced — compileTime, setCompileTime, setCodePatchers,
// setHirFunc, staticEntry inlined to compiled_function.h.
// Remaining: isJitCompiled (extern C), ~CompiledFunction, disassemble,
// printHIR (need Disassembler/HIRPrinter/ModuleState).

#include "cinderx/Jit/compiled_function.h"

#include "cinderx/Common/extra-py-flags.h"
#include "cinderx/Common/log.h"
#include "cinderx/Jit/disassembler.h"
#include "cinderx/Jit/hir/printer.h"
#include "cinderx/Jit/context_iface.h"
#include "cinderx/module_state.h"

#include <iostream>

extern "C" {

bool isJitCompiled(const PyFunctionObject* func) {
  cinderx::ModuleState* mod_state = cinderx::getModuleState();
  if (mod_state == nullptr) {
    return false;
  }
  jit::ICodeAllocator* code_allocator = mod_state->codeAllocator();
  if (code_allocator != nullptr &&
      code_allocator->contains(reinterpret_cast<const void*>(func->vectorcall))) {
    return true;
  }
  jit::IJitContext* jit_ctx = mod_state->jitContext();
  if (jit_ctx != nullptr) {
    auto* mutable_func = const_cast<PyFunctionObject*>(func);
    return jit_ctx->didCompile(mutable_func)
        && !jit_ctx->isDeoptimized(mutable_func);
  }
  return false;
}

} // extern "C"

namespace jit {

CompiledFunction::~CompiledFunction() {
  if (data_.runtime != nullptr) {
    data_.runtime->releaseReferences();
  }
  auto code_allocator = cinderx::getModuleState()->codeAllocator();
  code_allocator->releaseCode(const_cast<std::byte*>(data_.code.data()));
}

void CompiledFunction::disassemble() const {
  auto start = reinterpret_cast<const char*>(vectorcallEntry());
  Disassembler dis{start, codeSize()};
  dis.disassembleAll(std::cout);
}

void CompiledFunction::printHIR() const {
  JIT_CHECK(
      data_.irfunc != nullptr,
      "Can only call CompiledFunction::printHIR() from a debug build");
  jit::hir::HIRPrinter printer;
  printer.Print(std::cout, *data_.irfunc);
}

} // namespace jit
