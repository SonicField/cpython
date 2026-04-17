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
    Py_XDECREF(code);
    code = nullptr;
    Py_XDECREF(builtins);
    builtins = nullptr;
    Py_XDECREF(globals);
    globals = nullptr;
    Py_XDECREF(prim_args_info);
    prim_args_info = nullptr;
    Py_XDECREF(reifier);
    reifier = nullptr;
    free(fullname);
    for (size_t i = 0; i < typed_args_count; i++) {
      typed_args_data[i].~TypedArgument();
    }
    free(typed_args_data);
  }

  PyCodeObject* code{nullptr};
  PyDictObject* builtins{nullptr};
  PyDictObject* globals{nullptr};

  _PyTypedArgsInfo* prim_args_info{nullptr};

  char* fullname{nullptr};

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

  TypedArgument* typed_args_data{nullptr};
  size_t typed_args_count{0};
  size_t typed_args_capacity{0};

  void typed_args_push(const TypedArgument& arg) {
    if (typed_args_count == typed_args_capacity) {
      size_t new_cap = typed_args_capacity ? typed_args_capacity * 2 : 4;
      auto* new_data = static_cast<TypedArgument*>(
          malloc(new_cap * sizeof(TypedArgument)));
      for (size_t i = 0; i < typed_args_count; i++) {
        new (&new_data[i]) TypedArgument(typed_args_data[i]);
        typed_args_data[i].~TypedArgument();
      }
      free(typed_args_data);
      typed_args_data = new_data;
      typed_args_capacity = new_cap;
    }
    new (&typed_args_data[typed_args_count]) TypedArgument(arg);
    typed_args_count++;
  }

  // Return type
  Type return_type{TObject};

  FrameMode frameMode{FrameMode::kNormal};

  // env MUST be declared before cfg: C++ destroys in reverse declaration
  // order, so env outlives cfg. FrameState holds borrowed Register* into
  // env — if env destructs first, cfg teardown reads freed memory.
  Environment env;

  CFG cfg;

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
    Py_XINCREF(code_2.get());
    Py_XDECREF(this->code);
    this->code = code_2;
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
    return fs == nullptr ? BorrowedRef<PyCodeObject>(code) : fs->code;
  }

  PyObject* reifier{nullptr};

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
static_assert(sizeof(Function) == 41 * kPointerSize);
static_assert(sizeof(CFG) == 5 * kPointerSize);
/* Edge→C: BasicBlock shrank — unordered_set (7 ptrs each) → PhxEdgePtrArray
 * (3 ptrs each). 20 - (2×7) + (2×3) = 12. Verify after first successful build. */
static_assert(sizeof(BasicBlock) == 12 * kPointerSize);
static_assert(sizeof(Instr) == 5 * kPointerSize);
#endif

} // namespace jit::hir
