// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cinderx/Jit/hir/builtin_load_method_elimination_c.h"
#include "cinderx/Jit/hir/builtin_load_method_elimination.h"

#include "cinderx/Common/py-portability.h"
#include "cinderx/Jit/hir/hir_c_api.h"
#include "cinderx/Jit/hir/blme_helpers_c.h"

extern "C" void hir_reflow_types_c(void *func, void *start_block);
#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/hir/hir_type_c.h"
#include "cinderx/Jit/threaded_compile.h"
#include "cinderx/module_state.h"

#include <string.h>

namespace jit::hir {

namespace {

/* Convert C++ Type to C HirType via field-by-field conversion. */
inline HirType to_hir(Type t) {
  return Type::toHirType(t);
}

struct MethodInvoke {
  LoadMethodBase* load_method{nullptr};
  GetSecondOutput* get_instance{nullptr};
  CallMethod* call_method{nullptr};
};

/* immutableMultithreadedTypeLookup + getMethodObjectFromType: extracted
 * to Python/jit/hir/blme_helpers_c.{c,h} as pure-C functions
 * (phx_immutable_multithreaded_type_lookup +
 * phx_get_method_object_from_type). W42-class Cat-A extraction per
 * theologian 20:03Z + supervisor 20:04Z + W27e PARTIAL precedent.
 *
 * tryEliminateLoadMethod (below) calls phx_get_method_object_from_type
 * via Type::toHirType conversion. Heavier methods (tryEliminateLoadMethod
 * + Run) remain Cat-B accepted-residual due to ~17-bridge surface
 * required for full conversion (exceeds W25b minimal-bridge budget). */

// Returns true if LoadMethod/CallMethod/GetSecondOutput were removed.
// Returns false if they could not be removed.
bool tryEliminateLoadMethod(Function& irfunc, MethodInvoke& invoke) {
  ThreadedCompileSerialize guard;
  PyCodeObject* code = invoke.load_method->frameState()->code;
  PyObject* names = code->co_names;
  PyObject* name = PyTuple_GetItem(names, invoke.load_method->name_idx());
  JIT_DCHECK(name != nullptr, "name must not be null");

  Register* receiver = invoke.load_method->receiver();
  Type receiver_type = receiver->type();
  /* getMethodObjectFromType extracted to blme_helpers_c.c — call the
   * pure-C version with HirType conversion. */
  PyObject* method_obj = phx_get_method_object_from_type(to_hir(receiver_type), name);
  /* Mirror the original .cpp's DCHECK on TBottom-when-no-runtime-type:
   * the C helper returns NULL early when hir_type_runtime_py_type is
   * NULL, but the JIT_DCHECK on receiver_type == TBottom belongs in
   * the C++ caller where Type is in scope. */
  if (method_obj == nullptr) {
    HirType recv_hir = to_hir(receiver_type);
    if (hir_type_has_type_exact_spec(&recv_hir) &&
        hir_type_runtime_py_type(&recv_hir) == nullptr) {
      JIT_DCHECK(
          receiver_type == TBottom,
          "Type {} expected to have PyTypeObject*",
          receiver_type);
    }
  }
  if (method_obj == nullptr) {
    // No such method. Let the LoadMethod fail at runtime. _PyType_Lookup does
    // not raise an exception.
    return false;
  }
  if (Py_TYPE(method_obj) == &PyStaticMethod_Type) {
    // This is slightly tricky and nobody uses this except for
    // bytearray/bytes/str.maketrans. Not worth optimizing.
    return false;
  }
  Register* method_reg = invoke.load_method->output();
  auto load_const = static_cast<Instr*>(hir_c_create_load_const(
      method_reg, Type::toHirType(Type::fromObject(irfunc.env.addReference(method_obj)))));
  auto call_static = static_cast<VectorCall*>(hir_c_create_vectorcall_fs_reg(
      invoke.call_method->NumOperands(),
      invoke.call_method->output(),
      static_cast<uint32_t>(invoke.call_method->flags() | CallFlags::Static),
      invoke.call_method->frameState()));
  call_static->SetOperand(0, method_reg);
  if (Py_TYPE(method_obj) == &PyClassMethodDescr_Type) {
    // Pass the type as the first argument (e.g. dict.fromkeys).
    Register* type_reg = irfunc.env.AllocateRegister();
    auto load_type = static_cast<Instr*>(hir_c_create_load_const(
        type_reg,
        Type::toHirType(Type::fromObject(
            reinterpret_cast<PyObject*>(
                [&]{ HirType h = to_hir(receiver_type);
                     return hir_type_runtime_py_type(&h); }())))));
    load_type->setBytecodeOffset(invoke.load_method->bytecodeOffset());
    load_type->InsertBefore(*invoke.call_method);
    call_static->SetOperand(1, type_reg);
  } else {
    JIT_DCHECK(
        Py_TYPE(method_obj) == &PyMethodDescr_Type ||
            Py_TYPE(method_obj) == &PyWrapperDescr_Type ||
            Py_TYPE(method_obj) == &PyFunction_Type,
        "unexpected type");
    // Pass the instance as the first argument (e.g. str.join, str.__mod__).
    call_static->SetOperand(1, receiver);
  }
  for (std::size_t i = 2; i < invoke.call_method->NumOperands(); i++) {
    call_static->SetOperand(i, invoke.call_method->GetOperand(i));
  }
  auto use_type = static_cast<Instr*>(hir_c_create_use_type(receiver, Type::toHirType(receiver_type.unspecialized())));
  invoke.load_method->ExpandInto({use_type, load_const});
  invoke.get_instance->ReplaceWith(
      *static_cast<Instr*>(hir_c_create_assign(invoke.get_instance->output(), receiver)));
  invoke.call_method->ReplaceWith(*call_static);
  Instr::Destroy(invoke.load_method);
  Instr::Destroy(invoke.get_instance);
  Instr::Destroy(invoke.call_method);
  return true;
}

} // namespace

void BuiltinLoadMethodElimination::Run(Function& irfunc) {
  bool changed = true;
  while (changed) {
    changed = false;
    UnorderedMap<LoadMethodBase*, MethodInvoke> invokes;
    for (auto& block : irfunc.cfg.blocks) {
      for (auto& instr : block) {
        if (!instr.IsCallMethod()) {
          continue;
        }
        auto cm = static_cast<CallMethod*>(&instr);
        auto func_instr = cm->func()->instr();
        if (func_instr->IsLoadMethodSuper()) {
          continue;
        }

        if (!isLoadMethodBase(*func_instr)) {
          // {FillTypeMethodCache | LoadTypeMethodCacheEntryValue} and
          // CallMethod represent loading and invoking methods off a type (e.g.
          // dict.fromkeys(...)) which do not need to follow
          // LoadMethod/CallMethod pairing invariant and do not benefit from
          // tryEliminateLoadMethod which only handles eliminating of method
          // calls on the instance
          continue;
        }

        auto lm = static_cast<LoadMethodBase*>(func_instr);

        JIT_DCHECK(
            cm->self()->instr()->IsGetSecondOutput(),
            "GetSecondOutput/CallMethod should be paired but got "
            "{}/CallMethod",
            cm->self()->instr()->opname());
        auto glmi = static_cast<GetSecondOutput*>(cm->self()->instr());
        auto result = invokes.emplace(lm, MethodInvoke{lm, glmi, cm});
        if (!result.second) {
          // This pass currently only handles 1:1 LoadMethod/CallMethod
          // combinations. If there are multiple CallMethod for a given
          // LoadMethod, bail out.
          // TASK(T138839090): support multiple CallMethod
          invokes.erase(result.first);
        }
      }
    }
    for (auto [lm, invoke] : invokes) {
      changed |= tryEliminateLoadMethod(irfunc, invoke);
    }
    hir_reflow_types_c(&irfunc, irfunc.cfg.entry_block);
  }
}

} // namespace jit::hir

extern "C" void hir_builtin_load_method_elimination_run(HirFunction func) {
  jit::hir::BuiltinLoadMethodElimination{}.Run(
      *static_cast<jit::hir::Function*>(func));
}
