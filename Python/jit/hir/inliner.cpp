// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cinderx/Jit/hir/inliner.h"
#include "cinderx/Jit/hir/hir_c_api.h"
#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/hir/hir_type_c.h"
#include "cinderx/Jit/jit_config_c.h"

#include "internal/pycore_code.h"

#include "cinderx/Common/extra-py-flags.h"
#include "cinderx/Jit/hir/builder.h"
#include "cinderx/Jit/hir/clean_cfg.h"
#include "cinderx/Jit/hir/copy_propagation.h"
#include "cinderx/Jit/hir/instr_effects.h"
#include "cinderx/Jit/hir/preload.h"
#include "cinderx/Jit/inline_cache.h"
#include "cinderx/Jit/context.h"
#include "cinderx/Jit/type_deopt_patchers.h"
#include "cinderx/StaticPython/strictmoduleobject.h"

namespace jit::hir {

#define LOG_INLINER(...) JIT_LOGIF(jit_get_config()->log.debug_inliner, __VA_ARGS__)

namespace {

struct AbstractCall {
  AbstractCall(
      BorrowedRef<PyFunctionObject> func,
      size_t nargs,
      DeoptBase* instr,
      Register* target = nullptr)
      : func{func}, nargs{nargs}, instr{instr}, target{target} {}

  Register* arg(std::size_t i) const {
    if (instr->IsInvokeStaticFunction()) {
      auto f = static_cast<InvokeStaticFunction*>(instr);
      return f->arg(i + 1);
    }
    if (instr->IsVectorCall()) {
      auto f = static_cast<VectorCall*>(instr);
      return f->arg(i);
    }
    if (instr->IsCallMethod()) {
      auto f = static_cast<CallMethod*>(instr);
      // arg(0) is self, arg(1+) are the regular arguments
      if (i == 0) {
        return f->self();
      }
      return f->arg(i - 1);
    }
    JIT_ABORT("Unsupported call type {}", instr->opname());
  }

  BorrowedRef<PyFunctionObject> func;
  size_t nargs{0};
  DeoptBase* instr{nullptr};
  Register* target{nullptr};
};

void dlogAndCollectFailureStats(
    Function& caller,
    AbstractCall* call_instr,
    InlineFailureType failure_type) {
  BorrowedRef<PyFunctionObject> func = call_instr->func;
  std::string callee_name = funcFullname(func);
  Function::InlineFailureStats& inline_failure_stats =
      caller.inline_function_stats.failure_stats;
  inline_failure_stats[failure_type].insert(callee_name);
  LOG_INLINER(
      "Can't inline {} into {} because {}",
      callee_name,
      caller.fullname,
      getInlineFailureMessage(failure_type));
}

void dlogAndCollectFailureStats(
    Function& caller,
    AbstractCall* call_instr,
    InlineFailureType failure_type,
    const char* tp_name) {
  BorrowedRef<PyFunctionObject> func = call_instr->func;
  std::string callee_name = funcFullname(func);
  Function::InlineFailureStats& inline_failure_stats =
      caller.inline_function_stats.failure_stats;
  inline_failure_stats[failure_type].insert(callee_name);
  LOG_INLINER(
      "Can't inline {} into {} because {} but a {:.200s}",
      callee_name,
      caller.fullname,
      getInlineFailureMessage(failure_type),
      tp_name);
}

// Assigns a cost to every function, to be used when determining whether it
// makes sense to inline or not.
size_t codeCost(BorrowedRef<PyCodeObject> code) {
  // Manually iterating through the code block to count real opcodes and not
  // inline caches.  Not the best metric but it's something to start with.
  size_t num_opcodes = 0;
  for ([[maybe_unused]] auto& instr : BytecodeInstructionBlock{code}) {
    num_opcodes++;
  }
  return num_opcodes;
}

// Most of these checks are only temporary and do not in perpetuity prohibit
// inlining.
bool canInline(Function& caller, AbstractCall* call_instr) {
  BorrowedRef<PyFunctionObject> func = call_instr->func;

  BorrowedRef<> globals = func->func_globals;
  if (!PyDict_Check(globals)) {
    dlogAndCollectFailureStats(
        caller,
        call_instr,
        InlineFailureType::kGlobalsNotDict,
        Py_TYPE(globals)->tp_name);
    return false;
  }

  BorrowedRef<> builtins = func->func_builtins;
  if (!PyDict_CheckExact(builtins)) {
    dlogAndCollectFailureStats(
        caller,
        call_instr,
        InlineFailureType::kBuiltinsNotDict,
        Py_TYPE(builtins)->tp_name);
    return false;
  }

  auto fail = [&](InlineFailureType failure_type) {
    dlogAndCollectFailureStats(caller, call_instr, failure_type);
    return false;
  };

  if (func->func_kwdefaults != nullptr) {
    return fail(InlineFailureType::kHasKwdefaults);
  }

  BorrowedRef<PyCodeObject> code{func->func_code};
  JIT_CHECK(PyCode_Check(code), "Expected PyCodeObject");

  if (code->co_kwonlyargcount > 0) {
    return fail(InlineFailureType::kHasKwOnlyArgs);
  }
  if (code->co_flags & CO_VARKEYWORDS) {
    return fail(InlineFailureType::kHasVarkwargs);
  }
  JIT_DCHECK(code->co_argcount >= 0, "argcount must be positive");
  if (code->co_flags & CO_VARARGS) {
    // Allow inlining *args functions when the call site provides a
    // statically-known argument count >= co_argcount. Excess arguments
    // will be packed into a MakeTuple in inlineFunctionCall().
    if (call_instr->nargs < static_cast<size_t>(code->co_argcount)) {
      return fail(InlineFailureType::kCalledWithMismatchedArgs);
    }
  } else {
    if (call_instr->nargs != static_cast<size_t>(code->co_argcount)) {
      return fail(InlineFailureType::kCalledWithMismatchedArgs);
    }
  }
  if (code->co_flags & kCoFlagsAnyGenerator) {
    return fail(InlineFailureType::kIsGenerator);
  }
#if PY_VERSION_HEX >= 0x030C0000
  // Avoid the allocation that can happen in
  // PyCode_GetCellvars and PyCode_GetFreevars
  for (int offset = 0; offset < code->co_nlocalsplus; offset++) {
    _PyLocals_Kind k = _PyLocals_GetKind(code->co_localspluskinds, offset);
    if (k & CO_FAST_CELL) {
      return fail(InlineFailureType::kHasCellvars);
    } else if (k & CO_FAST_FREE) {
      return fail(InlineFailureType::kHasFreevars);
    }
  }
#else
  Py_ssize_t ncellvars = PyTuple_GET_SIZE(PyCode_GetCellvars(code));
  if (ncellvars > 0) {
    return fail(InlineFailureType::kHasCellvars);
  }
  Py_ssize_t nfreevars = PyTuple_GET_SIZE(PyCode_GetFreevars(code));
  if (nfreevars > 0) {
    return fail(InlineFailureType::kHasFreevars);
  }
#endif

  if constexpr (PY_VERSION_HEX >= 0x030E0000 && PY_VERSION_HEX < 0x030F0000) {
    // On 3.14, EagerImportName LIR uses env_->asm_interpreter_frame which is
    // wrong for inlined functions. On 3.12 (PyImport_Import) and 3.15+
    // (JITRT_ImportName), the import path does not use the interpreter frame
    // directly. Note: both paths use PyEval_GetGlobals() for __builtins__
    // lookup, which returns the caller frame globals under inlining — same
    // pre-existing behaviour as regular ImportName (JITRT_ImportName).
    for (auto& bci : BytecodeInstructionBlock{code}) {
      if (bci.opcode() == EAGER_IMPORT_NAME) {
        return fail(InlineFailureType::kHasEagerImportName);
      }
    }
  }


  return true;
}

// As canInline() for checks which require a preloader.
bool canInlineWithPreloader(
    Function& caller,
    AbstractCall* call_instr,
    const Preloader& preloader) {
  if (call_instr->instr->IsVectorCall() &&
      (preloader.code()->co_flags & CI_CO_STATICALLY_COMPILED) &&
      (preloader.returnType() <= TPrimitive || preloader.hasPrimitiveArgs())) {
    // TASK(T122371281) remove this constraint
    dlogAndCollectFailureStats(
        caller, call_instr, InlineFailureType::kIsVectorCallWithPrimitives);
    return false;
  }

  return true;
}

void inlineFunctionCall(Function& caller, AbstractCall* call_instr) {
  if (!canInline(caller, call_instr)) {
    return;
  }

  auto caller_frame_state =
      std::make_unique<FrameState>(*call_instr->instr->frameState());

  BorrowedRef<PyFunctionObject> callee = call_instr->func;

  // We are only able to inline functions that were already preloaded, since we
  // can't safely preload anything mid-compile (preloading can execute arbitrary
  // Python code and raise Python exceptions). Currently this means that in
  // single-function-compile mode we are limited to inlining functions loaded as
  // globals, or statically invoked. See `preloadFuncAndDeps` for what
  // dependencies we will preload. In batch-compile mode we can inline anything
  // that is part of the batch.
  Preloader* preloader = preloaderManager().find(callee);
  if (!preloader) {
    dlogAndCollectFailureStats(
        caller, call_instr, InlineFailureType::kNeedsPreload);
    return;
  }

  if (!canInlineWithPreloader(caller, call_instr, *preloader)) {
    return;
  }

  HIRBuilder hir_builder(*preloader);
  std::string callee_name = funcFullname(callee);

  InlineResult result;
  try {
    result = hir_builder.inlineHIR(&caller, caller_frame_state.get());
  } catch (const std::exception& exn) {
    LOG_INLINER(
        "Tried to inline {} into {}, but failed with {}",
        callee_name,
        caller.fullname,
        exn.what());
    return;
  }

  // This logging is parsed by jitlist_bisect.py to find inlined functions.
  JIT_LOGIF(
      jit_get_config()->log.debug_inliner || jit_get_config()->log.debug,
      "Inlining function {} into {}",
      callee_name,
      caller.fullname);

  BorrowedRef<PyCodeObject> callee_code = preloader->code();
  BasicBlock* tail = caller.cfg.splitAfter(*call_instr->instr);
  auto begin_inlined_function = BeginInlinedFunction::create(
      callee, std::move(caller_frame_state), callee_name, preloader->reifier());
  auto callee_branch = static_cast<Instr*>(hir_c_create_branch_cpp(result.entry));
  if (call_instr->target != nullptr) {
    // Not a static call. Check that __code__ has not been swapped out since
    // the function was inlined.
    // VectorCall -> {LoadField, GuardIs, BeginInlinedFunction, Branch to
    // callee CFG}
    //
    // Consider emitting a DeoptPatchpoint here to catch the case where someone
    // swaps out function.__code__.
    Register* code_obj = caller.env.AllocateRegister();
    auto load_code = static_cast<Instr*>(hir_c_create_load_field_reg(
        code_obj, call_instr->target, "func_code",
        offsetof(PyFunctionObject, func_code),
        Type::toHirType(TObject), 0));
    Register* guarded_code = caller.env.AllocateRegister();
    auto guard_code = static_cast<Instr*>(hir_c_create_guard_is_reg(guarded_code, callee_code, code_obj));
    call_instr->instr->ExpandInto(
        {load_code, guard_code, begin_inlined_function, callee_branch});
  } else {
    call_instr->instr->ExpandInto({begin_inlined_function, callee_branch});
  }
  tail->push_front(static_cast<Instr*>(hir_c_create_end_inlined_function(begin_inlined_function)));

  // Transform LoadArg into Assign (or MakeTuple for *args)
  int starargs_idx = (callee_code->co_flags & CO_VARARGS)
      ? callee_code->co_argcount + callee_code->co_kwonlyargcount
      : -1;
  for (auto it = result.entry->begin(); it != result.entry->end();) {
    auto& instr = *it;
    ++it;

    if (instr.IsLoadArg()) {
      auto load_arg = static_cast<LoadArg*>(&instr);
      int arg_idx = load_arg->arg_idx();
      if (arg_idx == starargs_idx) {
        // *args parameter: pack excess caller arguments into a tuple.
        // This is semantics-preserving -- a non-inlined call creates the
        // same tuple in the interpreter. Escape analysis (Step 2) can
        // later eliminate the allocation for constant-index access.
        size_t num_excess = call_instr->nargs
            - static_cast<size_t>(callee_code->co_argcount);
        std::vector<Register*> excess_args;
        excess_args.reserve(num_excess);
        for (size_t i = 0; i < num_excess; i++) {
          excess_args.push_back(
              call_instr->arg(callee_code->co_argcount + i));
        }
        auto* make_tuple = MakeTuple::create(
            num_excess,
            instr.output(),
            excess_args,
            *call_instr->instr->frameState());
        instr.ReplaceWith(*make_tuple);
        Instr::Destroy(&instr);
      } else {
        auto assign =
            static_cast<Instr*>(hir_c_create_assign(instr.output(), call_instr->arg(arg_idx)));
        instr.ReplaceWith(*assign);
        Instr::Destroy(&instr);
      }
    }
  }

  // Transform Return into Assign+Branch
  auto return_instr = result.exit->GetTerminator();
  JIT_CHECK(
      return_instr->IsReturn(),
      "terminator from inlined function should be Return");
  auto assign =
      static_cast<Instr*>(hir_c_create_assign(call_instr->instr->output(), return_instr->GetOperand(0)));
  auto return_branch = static_cast<Instr*>(hir_c_create_branch_cpp(tail));
  return_instr->ExpandInto({assign, return_branch});
  Instr::Destroy(return_instr);

  Instr::Destroy(call_instr->instr);
  caller.inline_function_stats.num_inlined_functions++;
}

void tryEliminateBeginEnd(EndInlinedFunction* end) {
  BeginInlinedFunction* begin = end->matchingBegin();
  if (begin->block() != end->block()) {
    // Elimination across basic blocks not supported yet.
    return;
  }
  auto it = begin->block()->iterator_to(*begin);
  it++;
  std::vector<Instr*> to_delete{begin, end};
  for (; &*it != end; it++) {
    // Snapshots reference the FrameState owned by BeginInlinedFunction and, if
    // not removed, will contain bad pointers.
    if (it->IsSnapshot()) {
      to_delete.push_back(&*it);
      continue;
    }
    // Instructions that either deopt or otherwise materialize a PyFrameObject
    // need the shadow frames to exist. Everything that materializes a
    // PyFrameObject should also be marked as deopting.

    if (it->asDeoptBase()
#if PY_VERSION_HEX >= 0x030C0000
        // Updating the previous instruction needs the frame too.
        || hasArbitraryExecution(*it)
#endif
    ) {
      return;
    }
  }
  for (Instr* instr : to_delete) {
    instr->unlink();
    Instr::Destroy(instr);
  }
}

} // namespace

void InlineFunctionCalls::Run(Function& irfunc) {
  if (irfunc.code == nullptr) {
    // In tests, irfunc may not have bytecode.
    return;
  }
  if (irfunc.code->co_flags & kCoFlagsAnyGenerator) {
    // TASK(T109706798): Support inlining into generators
    LOG_INLINER(
        "Refusing to inline functions into {}: function is a generator",
        irfunc.fullname);
    return;
  }

  // Scan through all function calls in `irfunc` and mark the ones that are
  // suitable for inlining.
  std::vector<AbstractCall> to_inline;
  for (auto& block : irfunc.cfg.blocks) {
    for (auto& instr : block) {
      if (instr.IsVectorCall()) {
        auto call = static_cast<VectorCall*>(&instr);
        Register* target = call->func();
        const std::string& caller_name = irfunc.fullname;
        if (!target->isA(TFunc)) {
          LOG_INLINER(
              "Can't inline non-function {}:{} into {}",
              *target,
              target->type(),
              caller_name);
          // Speculative inlining: check IC for resolved method
          HirType target_hir = Type::toHirType(target->type());
          PyTypeObject* recv_type = hir_type_runtime_py_type(&target_hir);
          if (recv_type != nullptr) {
            auto* fs = call->asDeoptBase() ? call->asDeoptBase()->frameState() : nullptr;
            if (fs != nullptr) {
              auto* ic = jit::getContext()->allocateLoadMethodCache(BorrowedRef<PyCodeObject>(fs->code), instr.bytecodeOffset().value());
              for (const auto& entry : ic->entries()) {
                if (entry.value != nullptr && PyFunction_Check(entry.value)) {
                  BorrowedRef<PyFunctionObject> callee{entry.value};
                  to_inline.emplace_back(callee, call->numArgs(), call, target);
                  break;
                }
              }
            }
          }
          continue;
        }
        if (!target->type().hasValueSpec(TFunc)) {
          LOG_INLINER(
              "Can't inline unknown function {}:{} into {}",
              *target,
              target->type(),
              caller_name);
          continue;
        }
        if (call->flags() & CallFlags::KwArgs) {
          LOG_INLINER(
              "Can't inline {}:{} into {} because it has kwargs",
              *target,
              target->type(),
              caller_name);
          continue;
        }

        auto target_ty2 = target->type();
        HirType target_hir2 = Type::toHirType(target_ty2);
        BorrowedRef<PyFunctionObject> callee{hir_type_object_spec(&target_hir2)};
        to_inline.emplace_back(callee, call->numArgs(), call, target);
      } else if (instr.IsInvokeStaticFunction()) {
        auto call = static_cast<InvokeStaticFunction*>(&instr);
        to_inline.emplace_back(call->func(), call->NumArgs() - 1, call);
      } else if (instr.IsCallMethod()) {
        // Speculative inlining for method calls: check IC type feedback
        // for monomorphic call sites.
        auto* cm = static_cast<CallMethod*>(&instr);
        Register* func_reg = cm->func();
        Instr* def = func_reg->instr();
        if (def != nullptr && def->opcode() == Opcode::kLoadMethodCached) {
          auto* fs = def->asDeoptBase()->frameState();
          if (fs != nullptr) {
            BorrowedRef<PyCodeObject> code = fs->code;
            int bc_off = def->bytecodeOffset().value();
            auto* ic = jit::getContext()->allocateLoadMethodCache(code, bc_off);
            // Check for monomorphic type in IC entries
            PyTypeObject* mono_type = nullptr;
            bool is_monomorphic = false;
            for (const auto& entry : ic->entries()) {
              if (entry.type != nullptr) {
                if (mono_type == nullptr) {
                  mono_type = entry.type;
                  is_monomorphic = true;
                } else if (entry.type != mono_type) {
                  is_monomorphic = false;
                  break;
                }
              }
            }
            if (is_monomorphic && mono_type != nullptr) {
              // Get the resolved function from the IC entry
              for (const auto& entry : ic->entries()) {
                if (entry.value != nullptr && PyFunction_Check(entry.value)) {
                  BorrowedRef<PyFunctionObject> callee{entry.value};
                  LOG_INLINER(
                      "Speculative inline: monomorphic CallMethod via IC type {}",
                      mono_type->tp_name);
                  to_inline.emplace_back(callee, cm->NumArgs() + 1, cm, func_reg);  // +1 for self
                  break;
                }
              }
            }
          }
        }
      }
    }
  }

  if (to_inline.empty()) {
    return;
  }

  size_t cost_limit = jit_get_config()->inliner_cost_limit;
  size_t cost = codeCost(irfunc.code);

  // Inline as many calls as possible, starting from the top of the function and
  // working down.
  for (auto& call : to_inline) {
    BorrowedRef<PyCodeObject> call_code{call.func->func_code};
    size_t new_cost = cost + codeCost(call_code);
    if (new_cost > cost_limit) {
      LOG_INLINER(
          "Inliner reached cost limit of {} when trying to inline {} into {}, "
          "inlining stopping early",
          new_cost,
          funcFullname(call.func),
          irfunc.fullname);
      break;
    }
    cost = new_cost;

    // For speculative IC-based inlining, insert GuardType on the receiver
    // to deopt if the speculated type doesn't match at runtime.
    if (call.target != nullptr) {
      Register* receiver = nullptr;
      bool is_speculative = false;

      if (call.instr->IsVectorCall() && !call.target->isA(TFunc)) {
        // VectorCall with non-TFunc target (IC-resolved)
        auto* vc = static_cast<VectorCall*>(call.instr);
        if (vc->numArgs() > 0) {
          receiver = vc->arg(0);
          is_speculative = true;
        }
      } else if (call.instr->IsCallMethod()) {
        // CallMethod with IC-resolved function
        auto* cm = static_cast<CallMethod*>(call.instr);
        receiver = cm->self();
        is_speculative = true;
      }

      if (is_speculative && receiver != nullptr) {
        // Look up the monomorphic type from the IC
        Instr* def = call.target->instr();
        if (def != nullptr && def->opcode() == Opcode::kLoadMethodCached) {
          auto* fs = def->asDeoptBase()->frameState();
          if (fs != nullptr) {
            BorrowedRef<PyCodeObject> code = fs->code;
            int bc_off = def->bytecodeOffset().value();
            auto* ic = jit::getContext()->allocateLoadMethodCache(code, bc_off);
            PyTypeObject* mono_type = nullptr;
            for (const auto& entry : ic->entries()) {
              if (entry.type != nullptr) {
                mono_type = entry.type;
                break;
              }
            }
            if (mono_type != nullptr) {
              auto& env = irfunc.env;
              Register* guarded = env.AllocateRegister();
              Type guard_type = Type::fromTypeExact(mono_type);
              // Construct deopt FrameState at the LOAD_METHOD bytecode
              // offset with the receiver on the operand stack.
              // LoadMethodCached FrameState has the correct bytecodeOffset
              // (LOAD_METHOD) but an empty stack because emitLoadMethod()
              // pops the receiver before emitting the instruction. Clone
              // the FrameState and push the receiver back so that deopt
              // re-executes LOAD_METHOD from the correct interpreter state.
              FrameState deopt_fs(*def->asDeoptBase()->frameState());
              phx_ptr_arr_push(&deopt_fs.stack, receiver);
              auto* guard = GuardType::create(
                  guarded, guard_type, receiver, deopt_fs);
              // Insert Snapshot with the same corrected FrameState before
              // the guard. refcount_insertion's snapshot resolution overwrites
              // guard FrameStates with the dominating Snapshot's FrameState.
              auto* snapshot = Snapshot::create(deopt_fs);
              snapshot->copyBytecodeOffset(*def);
              snapshot->InsertBefore(*call.instr);
              guard->InsertBefore(*call.instr);
              LOG_INLINER(
                  "Inserted GuardType for speculative inline, type={}",
                  mono_type->tp_name);
            }
          }
        }
      }
    }
    // Eliminate redundant LoadMethodCached for speculative inlining.
    // The IC lookup (Py_TYPE, entries iteration, version check, 2x Py_INCREF)
    // is redundant when the guard chain (GuardType on receiver, GuardIs on
    // func.__code__) provides the same safety guarantees.
    // For mutable types, add TypeAttrDeoptPatcher to detect method reassignment.
    if (call.target != nullptr) {
      Instr* target_def = call.target->instr();
      if (target_def != nullptr &&
          target_def->opcode() == Opcode::kLoadMethodCached) {
        auto* lmc = static_cast<LoadMethodCached*>(target_def);
        Register* receiver = target_def->GetOperand(0);

        // Get the attribute name from co_names for the DeoptPatchpoint.
        auto* fs = target_def->asDeoptBase()->frameState();
        PyObject* attr_name = PyTuple_GetItem(fs->code->co_names, lmc->name_idx());

        // For mutable types, install a TypeAttrDeoptPatcher to detect
        // method reassignment at runtime. If Dog.speak is reassigned,
        // the patcher fires and deopts to the interpreter.
        PyTypeObject* mono_type = nullptr;
        auto* ic = jit::getContext()->allocateLoadMethodCache(
            fs->code, target_def->bytecodeOffset().value());
        for (const auto& entry : ic->entries()) {
          if (entry.type != nullptr) {
            mono_type = entry.type;
            break;
          }
        }

        if (mono_type != nullptr &&
            !_PyClassLoader_IsImmutable(reinterpret_cast<PyObject*>(mono_type))) {
          PyObject* func_obj_raw = reinterpret_cast<PyObject*>(call.func.get());
          auto* patcher = irfunc.allocateCodePatcher<TypeAttrDeoptPatcher>(
              BorrowedRef<PyTypeObject>{mono_type},
              BorrowedRef<PyUnicodeObject>{attr_name},
              BorrowedRef<>{func_obj_raw});
          auto* patchpoint = static_cast<DeoptPatchpoint*>(hir_c_create_deopt_patchpoint(patcher));
          patchpoint->copyBytecodeOffset(*target_def);
          auto cloned_fs = std::make_unique<FrameState>(*fs);
          patchpoint->setFrameState(std::move(cloned_fs));
          patchpoint->setGuiltyReg(receiver);
          patchpoint->setDescr("speculative inline method guard");
          patchpoint->InsertBefore(*target_def);
          LOG_INLINER(
              "Added TypeAttrDeoptPatcher for speculative inline of {}",
              funcFullname(call.func));
        }

        // Find and replace GetSecondOutput (self extraction) before replacing
        // LoadMethodCached. Replace with Assign from the original receiver.
        for (auto& block : irfunc.cfg.blocks) {
          for (auto it = block.begin(); it != block.end();) {
            auto& inst = *it;
            ++it;
            if (inst.IsGetSecondOutput() && inst.GetOperand(0) == call.target) {
              auto* assign = static_cast<Instr*>(hir_c_create_assign(inst.output(), receiver));
              inst.ReplaceWith(*assign);
              Instr::Destroy(&inst);
            }
          }
        }

        // Replace LoadMethodCached with LoadConst of the resolved function.
        PyObject* func_obj = reinterpret_cast<PyObject*>(call.func.get());
        Type func_type = Type::fromObject(irfunc.env.addReference(func_obj));
        auto* load_const = static_cast<Instr*>(hir_c_create_load_const(call.target, Type::toHirType(func_type)));
        target_def->ReplaceWith(*load_const);
        Instr::Destroy(target_def);
        LOG_INLINER(
            "Eliminated LoadMethodCached for speculative inline of {}",
            funcFullname(call.func));
      }
    }

    inlineFunctionCall(irfunc, &call);


    // We need to reflow types after every inline to propagate new type
    // information from the callee.
    reflowTypes(irfunc);
  }

  // The inliner will make some blocks unreachable and we need to remove them
  // to make the CFG valid again. While inlining might make some blocks
  // unreachable and therefore make less work (less to inline), we cannot
  // remove unreachable blocks in the above loop. It might delete instructions
  // pointed to by `to_inline`.
  CopyPropagation{}.Run(irfunc);
  CleanCFG{}.Run(irfunc);
}

void BeginInlinedFunctionElimination::Run(Function& irfunc) {
  std::vector<EndInlinedFunction*> ends;
  for (auto& block : irfunc.cfg.blocks) {
    for (auto& instr : block) {
      if (!instr.IsEndInlinedFunction()) {
        continue;
      }
      ends.push_back(static_cast<EndInlinedFunction*>(&instr));
    }
  }
  for (EndInlinedFunction* end : ends) {
    tryEliminateBeginEnd(end);
  }
}

} // namespace jit::hir
