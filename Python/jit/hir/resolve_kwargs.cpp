// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cinderx/Jit/hir/resolve_kwargs_c.h"
#include "cinderx/Jit/hir/resolve_kwargs.h"

#include "cinderx/Jit/hir/hir.h"
#include "cinderx/Jit/hir/hir_c_api.h"
#include "cinderx/Jit/hir/hir_type_c.h"
#include "cinderx/Common/log.h"
#include "cinderx/Common/code.h"

#include <Python.h>

#include <vector>

namespace jit::hir {

static inline HirType to_hir(Type t) { return Type::toHirType(t); }

namespace {

// Try to resolve keyword arguments in a VectorCall to positional order.
// Returns true if the instruction was replaced.
bool resolveVectorCallKwargs(VectorCall* call) {
  Register* target = call->func();

  // Need a known function to get parameter names.
  auto target_ty = target->type();
  HirType target_hir = to_hir(target_ty);
  HirType tfunc_hir = to_hir(TFunc);
  if (!hir_type_has_value_spec(&target_hir, tfunc_hir)) {
    return false;
  }

  BorrowedRef<PyFunctionObject> callee{hir_type_object_spec(&target_hir)};
  BorrowedRef<PyCodeObject> code{callee->func_code};

  // Skip callees with **kwargs or *args — cannot resolve statically.
  if (code->co_flags & (CO_VARKEYWORDS | CO_VARARGS)) {
    return false;
  }

  // Get the kwnames tuple from the last operand.
  std::size_t total_operands = call->NumOperands();
  if (total_operands < 2) {
    return false;  // Need at least func + 1 arg
  }

  Register* kwnames_reg = call->GetOperand(total_operands - 1);
  auto kwnames_ty = kwnames_reg->type();
  HirType kwnames_hir = to_hir(kwnames_ty);
  if (!hir_type_has_object_spec(&kwnames_hir)) {
    return false;  // kwnames not a known constant
  }

  PyObject* kwnames_obj = hir_type_object_spec(&kwnames_hir);
  if (!PyTuple_Check(kwnames_obj)) {
    return false;
  }

  Py_ssize_t n_kw = PyTuple_GET_SIZE(kwnames_obj);
  if (n_kw == 0) {
    return false;
  }

  // Operand layout: [func, arg0, arg1, ..., argN-1, kwnames]
  // Total args = total_operands - 2 (subtract func and kwnames)
  std::size_t total_args = total_operands - 2;
  std::size_t n_pos = total_args - static_cast<std::size_t>(n_kw);

  // Get parameter names from callee code object.
  int co_argcount = code->co_argcount;
  int co_kwonly = code->co_kwonlyargcount;
  int total_params = co_argcount + co_kwonly;

  // The total args provided must not exceed the callee parameter count.
  if (static_cast<int>(total_args) > total_params) {
    return false;
  }

  // Build mapping: for each callee param position, which call-site operand
  // provides the value?
  // First n_pos call-site args map to first n_pos callee params (positional).
  // Remaining args are keyword — look up their position in co_varnames.
  std::vector<Register*> reordered(total_args, nullptr);

  // Copy positional args (they are already in the right position).
  for (std::size_t i = 0; i < n_pos; i++) {
    reordered[i] = call->GetOperand(i + 1);  // +1 to skip func
  }

  // Map keyword args to callee parameter positions.
  PyObject* varnames = PyCode_GetVarnames(code);
  if (!PyTuple_Check(varnames)) {
    return false;
  }

  for (Py_ssize_t kw_idx = 0; kw_idx < n_kw; kw_idx++) {
    PyObject* kwname = PyTuple_GET_ITEM(kwnames_obj, kw_idx);
    Register* kw_arg = call->GetOperand(n_pos + kw_idx + 1);  // +1 for func

    // Find this keyword name in the callee parameters.
    bool found = false;
    for (int param_idx = 0; param_idx < total_params; param_idx++) {
      PyObject* param_name = PyTuple_GET_ITEM(varnames, param_idx);
      int cmp = PyUnicode_Compare(kwname, param_name);
      if (cmp == 0 && !PyErr_Occurred()) {
        if (reordered[param_idx] != nullptr) {
          // Duplicate argument — bail out, let Python handle the error.
          return false;
        }
        reordered[param_idx] = kw_arg;
        found = true;
        break;
      }
      if (PyErr_Occurred()) {
        PyErr_Clear();
        return false;
      }
    }

    if (!found) {
      // Keyword arg does not match any callee parameter — bail out.
      return false;
    }
  }

  // Verify all provided positions are filled (no gaps).
  for (std::size_t i = 0; i < total_args; i++) {
    if (reordered[i] == nullptr) {
      return false;  // Missing argument — let runtime handle it.
    }
  }

  // Create replacement VectorCall without kwargs.
  // New operand count: func + total_args (no kwnames).
  std::size_t new_num_operands = total_args + 1;
  CallFlags new_flags = static_cast<CallFlags>(static_cast<uint32_t>(call->flags()) & ~static_cast<uint32_t>(CallFlags::KwArgs));

  auto* new_call = static_cast<VectorCall*>(hir_c_create_vectorcall_reg(new_num_operands, call->output(), static_cast<uint32_t>(new_flags)));
  new_call->SetOperand(0, target);  // func
  for (std::size_t i = 0; i < total_args; i++) {
    new_call->SetOperand(i + 1, reordered[i]);
  }

  // Copy frame state and bytecode offset.
  if (auto* fs = call->frameState()) {
    new_call->setFrameState(*fs);
  }
  new_call->setBytecodeOffset(call->bytecodeOffset());

  call->ReplaceWith(*new_call);
  return true;
}

// Try to resolve keyword arguments in a CallMethod to positional order.
bool resolveCallMethodKwargs(CallMethod* call) {
  Register* target = call->func();

  auto target_ty2 = target->type();
  HirType target_hir2 = to_hir(target_ty2);
  HirType tfunc_hir2 = to_hir(TFunc);
  if (!hir_type_has_value_spec(&target_hir2, tfunc_hir2)) {
    return false;
  }

  BorrowedRef<PyFunctionObject> callee{hir_type_object_spec(&target_hir2)};
  BorrowedRef<PyCodeObject> code{callee->func_code};

  if (code->co_flags & (CO_VARKEYWORDS | CO_VARARGS)) {
    return false;
  }

  // CallMethod operand layout: [func, self, arg0, ..., argN-1, kwnames]
  std::size_t total_operands = call->NumOperands();
  if (total_operands < 3) {
    return false;
  }

  Register* kwnames_reg = call->GetOperand(total_operands - 1);
  auto kwnames_ty2 = kwnames_reg->type();
  HirType kwnames_hir2 = to_hir(kwnames_ty2);
  if (!hir_type_has_object_spec(&kwnames_hir2)) {
    return false;
  }

  PyObject* kwnames_obj = hir_type_object_spec(&kwnames_hir2);
  if (!PyTuple_Check(kwnames_obj)) {
    return false;
  }

  Py_ssize_t n_kw = PyTuple_GET_SIZE(kwnames_obj);
  if (n_kw == 0) {
    return false;
  }

  // Total args = total_operands - 3 (subtract func, self, kwnames)
  std::size_t total_args = total_operands - 3;
  std::size_t n_pos = total_args - static_cast<std::size_t>(n_kw);

  int co_argcount = code->co_argcount;
  int co_kwonly = code->co_kwonlyargcount;
  // For methods, first param is self which is separate in CallMethod.
  // co_argcount includes self for bound methods, but CallMethod has self
  // as a separate operand. Adjust: effective params = co_argcount - 1 + co_kwonly.
  // Actually, for module-level functions called via CALL, self is NULL/unused.
  // This needs careful handling — for now, use the args as-is.
  int total_params = co_argcount + co_kwonly;

  if (static_cast<int>(total_args) > total_params) {
    return false;
  }

  std::vector<Register*> reordered(total_args, nullptr);

  for (std::size_t i = 0; i < n_pos; i++) {
    reordered[i] = call->GetOperand(i + 2);  // +2 to skip func and self
  }

  PyObject* varnames = PyCode_GetVarnames(code);
  if (!PyTuple_Check(varnames)) {
    return false;
  }

  for (Py_ssize_t kw_idx = 0; kw_idx < n_kw; kw_idx++) {
    PyObject* kwname = PyTuple_GET_ITEM(kwnames_obj, kw_idx);
    Register* kw_arg = call->GetOperand(n_pos + kw_idx + 2);

    bool found = false;
    for (int param_idx = 0; param_idx < total_params; param_idx++) {
      PyObject* param_name = PyTuple_GET_ITEM(varnames, param_idx);
      int cmp = PyUnicode_Compare(kwname, param_name);
      if (cmp == 0 && !PyErr_Occurred()) {
        if (reordered[param_idx] != nullptr) {
          return false;
        }
        reordered[param_idx] = kw_arg;
        found = true;
        break;
      }
      if (PyErr_Occurred()) {
        PyErr_Clear();
        return false;
      }
    }

    if (!found) {
      return false;
    }
  }

  for (std::size_t i = 0; i < total_args; i++) {
    if (reordered[i] == nullptr) {
      return false;
    }
  }

  // Create replacement CallMethod without kwargs.
  std::size_t new_num_operands = total_args + 2;  // func + self + args
  CallFlags new_flags = static_cast<CallFlags>(static_cast<uint32_t>(call->flags()) & ~static_cast<uint32_t>(CallFlags::KwArgs));

  auto* new_call = static_cast<CallMethod*>(hir_c_create_call_method_reg(new_num_operands, call->output(), static_cast<uint32_t>(new_flags)));
  new_call->SetOperand(0, target);
  new_call->SetOperand(1, call->self());
  for (std::size_t i = 0; i < total_args; i++) {
    new_call->SetOperand(i + 2, reordered[i]);
  }

  if (auto* fs = call->frameState()) {
    new_call->setFrameState(*fs);
  }
  new_call->setBytecodeOffset(call->bytecodeOffset());

  call->ReplaceWith(*new_call);
  return true;
}

}  // namespace

void ResolveKwargs::Run(Function& irfunc) {
  int resolved = 0;

  for (auto& block : irfunc.cfg.blocks) {
    // Collect instructions to process (can not modify while iterating).
    std::vector<Instr*> kwargs_instrs;
    for (auto& instr : block) {
      if (instr.IsVectorCall()) {
        auto* call = static_cast<VectorCall*>(&instr);
        if (call->flags() & CallFlags::KwArgs) {
          kwargs_instrs.push_back(&instr);
        }
      } else if (instr.IsCallMethod()) {
        auto* call = static_cast<CallMethod*>(&instr);
        if (call->flags() & CallFlags::KwArgs) {
          kwargs_instrs.push_back(&instr);
        }
      }
    }

    for (auto* instr : kwargs_instrs) {
      if (instr->IsVectorCall()) {
        if (resolveVectorCallKwargs(static_cast<VectorCall*>(instr))) {
          resolved++;
        }
      } else if (instr->IsCallMethod()) {
        if (resolveCallMethodKwargs(static_cast<CallMethod*>(instr))) {
          resolved++;
        }
      }
    }
  }

  if (resolved > 0) {
    JIT_DLOG("ResolveKwargs: resolved {} keyword calls to positional in {}",
             resolved, irfunc.fullname);
  }
}

}  // namespace jit::hir

extern "C" void hir_resolve_kwargs_run(HirFunction func) {
  jit::hir::ResolveKwargs{}.Run(
      *static_cast<jit::hir::Function*>(func));
}
