// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/Jit/containers.h"
#include "cinderx/Jit/hir/cfg.h"
#include "cinderx/Jit/hir/hir.h"
#include "cinderx/Jit/jit_config_c.h"
#include "cinderx/Jit/jit_time_log.h"
#include "cinderx/StaticPython/typed-args-info.h"

namespace jit::hir {

class Function {
 public:
  using InlineFailureStats =
      UnorderedMap<InlineFailureType, UnorderedSet<std::string>>;
  Function() {}
  ~Function() {
    ThreadedCompileSerialize guard;
    code.reset();
    builtins.reset();
    globals.reset();
    prim_args_info.reset();
  }

  ThreadedRef<PyCodeObject> code;
  ThreadedRef<PyDictObject> builtins;
  ThreadedRef<PyDictObject> globals;

  // for primitive args only, null if there are none
  ThreadedRef<_PyTypedArgsInfo> prim_args_info;

  // Fully-qualified name of the function
  std::string fullname;

  // Does this function need its PyFunctionObject* at runtime?
  // (This is always the case in 3.12 as it is used to quickly access the
  // _PyInterpreterFrame)
  bool uses_runtime_func{
#if PY_VERSION_HEX < 0x030C0000
      false
#else
      true
#endif
  };

  // Does this function have primitive args?
  bool has_primitive_args{false};

  // is the first argument a primitive?
  bool has_primitive_first_arg{false};

  struct InlineFunctionStats {
    int num_inlined_functions{0};
    // map of {inline_failure_type -> function_names}
    InlineFailureStats failure_stats;
  } inline_function_stats;

  // vector of {locals_idx, type, optional}
  // in argument order, may have gaps for unchecked args
  std::vector<TypedArgument> typed_args;

  // Return type
  Type return_type{TObject};

  FrameMode frameMode{FrameMode::kNormal};

  CFG cfg;

  Environment env;

  // All the code patchers pointing to patch points in this function.
  //
  // These will be moved over to the CompiledFunction after compilation is
  // complete.
  std::vector<std::unique_ptr<CodePatcher>> code_patchers;

  // Optional property used to track time taken for individual compilation
  // phases
  std::unique_ptr<CompilationPhaseTimer> compilation_phase_timer;

  // Return the total number of arguments (positional + kwonly + varargs +
  // varkeywords)
  int numArgs() const {
    if (code == nullptr) return 0;
    return code->co_argcount + code->co_kwonlyargcount +
        bool(code->co_flags & CO_VARARGS) + bool(code->co_flags & CO_VARKEYWORDS);
  }

  Py_ssize_t numVars() const {
    return code != nullptr ? numLocalsplus(code) : 0;
  }

  // Set code and a number of other members that are derived from it.
  void setCode(BorrowedRef<PyCodeObject> code_2) {
    this->code.reset(code_2);
    uses_runtime_func = usesRuntimeFunc(code_2);
    frameMode = static_cast<FrameMode>(jit_get_config()->frame_mode);
  }

  // Count the number of instructions that match the predicate
  std::size_t CountInstrs(InstrPredicate pred) const {
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

  bool returnsPrimitive() const { return return_type <= TPrimitive; }
  bool returnsPrimitiveDouble() const { return return_type <= TCDouble; }

  void setCompilationPhaseTimer(std::unique_ptr<CompilationPhaseTimer> cpt) {
    compilation_phase_timer = std::move(cpt);
  }

  bool canDeopt() const {
    for (const BasicBlock& block : cfg.blocks) {
      for (const Instr& instr : block) {
        if (instr.asDeoptBase()) {
          return true;
        }
      }
    }
    return false;
  }

  template <typename T, typename... Args>
  T* allocateCodePatcher(Args&&... args) {
    code_patchers.emplace_back(
        std::make_unique<T>(std::forward<Args>(args)...));
    return static_cast<T*>(code_patchers.back().get());
  }

  // Get the code object for the given instruction.  Handles inlined functions
  // but assumes that inlined functions have a dominating FrameState from
  // BeginInlinedFunction to use.  If we start optimizing that out for inlined
  // functions that cannot deopt, we will have to do something different.
  //
  // The instruction must be part of this function.
  BorrowedRef<PyCodeObject> codeFor(const Instr& instr) const {
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

  ThreadedRef<> reifier;

 private:
  DISALLOW_COPY_AND_ASSIGN(Function);
};

using OpcodeCounts = std::array<int, kNumOpcodes>;
inline OpcodeCounts count_opcodes(const Function& func) {
  OpcodeCounts counts{};
  for (const BasicBlock& block : func.cfg.blocks) {
    for (const Instr& instr : block) {
      counts[static_cast<size_t>(instr.opcode())]++;
    }
  }
  return counts;
}

#ifndef _LIBCPP_VERSION
static_assert(sizeof(Function) == 48 * kPointerSize);
static_assert(sizeof(CFG) == 5 * kPointerSize);
static_assert(sizeof(BasicBlock) == 20 * kPointerSize);
static_assert(sizeof(Instr) == 5 * kPointerSize);
#endif

} // namespace jit::hir
