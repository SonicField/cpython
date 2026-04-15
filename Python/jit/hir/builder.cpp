// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cinderx/Jit/hir/builder.h"
#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/hir/hir_c_api.h"
#include "cinderx/Jit/hir/hir_type_c.h"
#include "cinderx/Jit/jit_config_c.h"
#include "cinderx/Jit/jit_rt.h"

#include "ceval.h"

#include "cinderx/python_runtime.h"

#if PY_VERSION_HEX >= 0x030C0000
#include "internal/pycore_intrinsics.h"
#include "internal/pycore_long.h"
#include "internal/pycore_pystate.h"
#include "internal/pycore_runtime.h"
#include "internal/pycore_typeobject.h"
#endif

#include "cinderx/Common/code.h"
#include "cinderx/Common/dict.h"
#include "cinderx/Common/py-portability.h"
#include "cinderx/Common/ref.h"
#include "cinderx/Interpreter/cinder_opcode.h"
#include "cinderx/Jit/containers.h"
#include "cinderx/Jit/context.h"
#include "cinderx/Jit/iterator_types.h"
#include "cinderx/Jit/hir/annotation_index_c.h"
#include "cinderx/Jit/hir/ssa.h"
#include "cinderx/Jit/hir/type.h"
#include "cinderx/StaticPython/checked_dict.h"
#include "cinderx/StaticPython/checked_list.h"
#include "cinderx/StaticPython/classloader.h"
#include "cinderx/StaticPython/static_array.h"
#include "cinderx/StaticPython/typed_method_def.h"
#include "cinderx/module_state.h"

#include <algorithm>
#include <deque>
#include <memory>
#include <optional>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>

namespace jit::hir {

namespace {

// Check that an opcode is one we know how to translate into HIR.
bool isSupportedOpcode(int opcode) {
  switch (opcode) {
    case BEFORE_ASYNC_WITH:
    case BEFORE_WITH:
    case BINARY_ADD:
    case BINARY_AND:
    case BINARY_FLOOR_DIVIDE:
    case BINARY_LSHIFT:
    case BINARY_MATRIX_MULTIPLY:
    case BINARY_MODULO:
    case BINARY_MULTIPLY:
    case BINARY_OP:
    case BINARY_OR:
    case BINARY_POWER:
    case BINARY_RSHIFT:
    case BINARY_SLICE:
    case BINARY_SUBSCR:
    case BINARY_SUBTRACT:
    case BINARY_TRUE_DIVIDE:
    case BINARY_XOR:
    case BUILD_CHECKED_LIST:
    case BUILD_CHECKED_MAP:
    case BUILD_CONST_KEY_MAP:
    case BUILD_LIST:
    case BUILD_MAP:
    case BUILD_SET:
    case BUILD_SLICE:
    case BUILD_STRING:
    case BUILD_INTERPOLATION:
    case BUILD_TEMPLATE:
    case BUILD_TUPLE:
    case CONVERT_VALUE:
    case CALL:
    case CALL_FUNCTION:
    case CALL_FUNCTION_EX:
    case CALL_FUNCTION_KW:
    case CALL_INTRINSIC_1:
    case CALL_INTRINSIC_2:
    case CALL_KW:
    case CALL_METHOD:
    case CAST:
    case CHECK_EG_MATCH:
    case CHECK_EXC_MATCH:
    case CLEANUP_THROW:
    case COMPARE_OP:
    case CONVERT_PRIMITIVE:
    case CONTAINS_OP:
    case COPY:
    case COPY_DICT_WITHOUT_KEYS:
    case COPY_FREE_VARS:
    case DELETE_ATTR:
    case DELETE_FAST:
    case DELETE_SUBSCR:
    case DICT_MERGE:
    case DICT_UPDATE:
    case DUP_TOP:
    case DUP_TOP_TWO:
    // EAGER_IMPORT_NAME: disabled — JIT does not set up frame globals
    // correctly for PyImport_Import, causing SIGSEGV or TypeError.
    // case EAGER_IMPORT_NAME:
    case END_ASYNC_FOR:
    case END_FOR:
    case END_SEND:
    case EXTENDED_ARG:
    case FAST_LEN:
    case FORMAT_SIMPLE:
    case FORMAT_VALUE:
    case FORMAT_WITH_SPEC:
    case FOR_ITER:
    case GEN_START:
    case GET_AITER:
    case GET_ANEXT:
    case GET_AWAITABLE:
    case GET_ITER:
    case GET_LEN:
    case GET_YIELD_FROM_ITER:
#if PY_VERSION_HEX >= 0x030E0000 || ENABLE_LAZY_IMPORTS
    case IMPORT_FROM:
      // LIR generation for IMPORT_FROM depends on access to _PyEval_ImportFrom
      // (added in 3.14) or the_PyImport_ImportFrom function that's only added
      // by Lazy Imports.
#endif
    // IMPORT_NAME: disabled — JIT does not set up frame globals
    // correctly for PyImport_Import, causing SIGSEGV or TypeError.
    // case IMPORT_NAME:
    case INPLACE_ADD:
    case INPLACE_AND:
    case INPLACE_FLOOR_DIVIDE:
    case INPLACE_LSHIFT:
    case INPLACE_MATRIX_MULTIPLY:
    case INPLACE_MODULO:
    case INPLACE_MULTIPLY:
    case INPLACE_OR:
    case INPLACE_POWER:
    case INPLACE_RSHIFT:
    case INPLACE_SUBTRACT:
    case INPLACE_TRUE_DIVIDE:
    case INPLACE_XOR:
    case INVOKE_FUNCTION:
    case INVOKE_METHOD:
    case INVOKE_NATIVE:
    case IS_OP:
    case JUMP_ABSOLUTE:
    case JUMP_BACKWARD:
    case JUMP_BACKWARD_NO_INTERRUPT:
    case JUMP_FORWARD:
    case JUMP_IF_FALSE_OR_POP:
    case JUMP_IF_NONZERO_OR_POP:
    case JUMP_IF_NOT_EXC_MATCH:
    case JUMP_IF_TRUE_OR_POP:
    case JUMP_IF_ZERO_OR_POP:
    case KW_NAMES:
    case LIST_APPEND:
    case LIST_EXTEND:
    case LIST_TO_TUPLE:
    case LOAD_ASSERTION_ERROR:
    case LOAD_ATTR:
    case LOAD_ATTR_SUPER:
    case LOAD_BUILD_CLASS:
    case LOAD_CLOSURE:
    case LOAD_COMMON_CONSTANT:
    case LOAD_CONST:
    case LOAD_DEREF:
    case LOAD_FAST:
    case LOAD_FAST_AND_CLEAR:
    case LOAD_FAST_BORROW:
    case LOAD_FAST_BORROW_LOAD_FAST_BORROW:
    case LOAD_FAST_LOAD_FAST:
    case LOAD_FAST_CHECK:
    case LOAD_FIELD:
    case LOAD_GLOBAL:
    case LOAD_ITERABLE_ARG:
    case LOAD_LOCAL:
    case LOAD_METHOD:
    case LOAD_METHOD_STATIC:
    case LOAD_METHOD_SUPER:
    case LOAD_SMALL_INT:
    case LOAD_SPECIAL:
    case LOAD_SUPER_ATTR:
    case LOAD_TYPE:
    case MAKE_CELL:
    case MAKE_FUNCTION:
    case MAP_ADD:
    case MATCH_CLASS:
    case MATCH_KEYS:
    case MATCH_MAPPING:
    case MATCH_SEQUENCE:
    case NOP:
    case NOT_TAKEN:
    case POP_BLOCK:
    case POP_EXCEPT:
    case POP_ITER:
    case POP_JUMP_IF_FALSE:
    case POP_JUMP_IF_NONE:
    case POP_JUMP_IF_NONZERO:
    case POP_JUMP_IF_NOT_NONE:
    case POP_JUMP_IF_TRUE:
    case POP_JUMP_IF_ZERO:
    case POP_TOP:
    case PRIMITIVE_BINARY_OP:
    case PRIMITIVE_BOX:
    case PRIMITIVE_COMPARE_OP:
    case PRIMITIVE_LOAD_CONST:
    case PRIMITIVE_UNARY_OP:
    case PRIMITIVE_UNBOX:
    case PUSH_EXC_INFO:
    case PUSH_NULL:
    case RAISE_VARARGS:
    case REFINE_TYPE:
    case RERAISE:
    case RESUME:
    case RETURN_CONST:
    case RETURN_GENERATOR:
    case RETURN_PRIMITIVE:
    case RETURN_VALUE:
    case ROT_FOUR:
    case ROT_N:
    case ROT_THREE:
    case ROT_TWO:
    case SEND:
    case SEQUENCE_GET:
    case SEQUENCE_SET:
    case SET_ADD:
    case SET_FUNCTION_ATTRIBUTE:
    case SET_UPDATE:
    case SETUP_ASYNC_WITH:
    case SETUP_FINALLY:
    case SETUP_WITH:
    case STORE_ATTR:
    case STORE_DEREF:
    case STORE_FAST:
    case STORE_FAST_LOAD_FAST:
    case STORE_FAST_STORE_FAST:
    case STORE_FIELD:
    case STORE_GLOBAL:
    case STORE_LOCAL:
    case STORE_SLICE:
    case STORE_SUBSCR:
    case SWAP:
    case TO_BOOL:
    case TP_ALLOC:
    case UNARY_INVERT:
    case UNARY_NEGATIVE:
    case UNARY_NOT:
    case UNARY_POSITIVE:
    case UNPACK_EX:
    case UNPACK_SEQUENCE:
    case WITH_EXCEPT_START:
    case YIELD_FROM:
    case YIELD_VALUE:
      return true;
    default:
      break;
  }
  return false;
}

// Check that a symbol/name is one that the JIT has banned.
bool isBannedName(std::string_view name) {
  return name == "eval" || name == "exec" || name == "locals";
}

} // namespace

// Allocate a temp register that may be used for the stack. It should not be a
// register that will be treated specially in the FrameState (e.g. tracked as
// containing a local or cell.)
Register* TempAllocator::AllocateStack() {
  Register* reg = env_->AllocateRegister();
  cache_.emplace_back(reg);
  return reg;
}

// Get the i-th stack temporary or allocate one.
Register* TempAllocator::GetOrAllocateStack(std::size_t idx) {
  if (idx < cache_.size()) {
    Register* reg = cache_[idx];
    return reg;
  }
  return AllocateStack();
}

// Allocate a temp register that will not be used for a stack value.
Register* TempAllocator::AllocateNonStack() {
  return env_->AllocateRegister();
}

void HIRBuilder::allocateLocalsplus(Environment* env, FrameState& state) {
  int nlocalsplus = numLocalsplus(code_);
  state.localsplus.clear();
  state.localsplus.reserve(nlocalsplus);
  for (int i = 0; i < nlocalsplus; ++i) {
    state.localsplus.emplace_back(env->AllocateRegister());
  }

  state.nlocals = numLocals(code_);
}

static inline HirType to_hir(Type t) {
  return Type::toHirType(t);
}

// Holds the current state of translation for a given basic block
struct HIRBuilder::TranslationContext {
  TranslationContext(BasicBlock* b, const FrameState& fs)
      : block(b), frame(fs) {}

  template <typename T, typename... Args>
  T* emit(Args&&... args) {
    auto instr = block->appendWithOff<T>(
        frame.instrOffset(), std::forward<Args>(args)...);
    return instr;
  }

  template <typename T, typename... Args>
  T* emitChecked(Args&&... args) {
    auto instr = emit<T>(std::forward<Args>(args)...);
    auto out = instr->output();
    emit<CheckExc>(out, out, frame);
    return instr;
  }

  template <typename T, typename... Args>
  T* emitVariadic(
      TempAllocator& temps,
      std::size_t num_operands,
      Args&&... args) {
    Register* out = temps.AllocateStack();
    auto call = emit<T>(num_operands, out, std::forward<Args>(args)...);
    for (auto i = num_operands; i > 0; i--) {
      Register* operand = frame.stack.pop();
      call->SetOperand(i - 1, operand);
    }
    call->setFrameState(frame);
    frame.stack.push(out);
    return call;
  }

  void emitSnapshot() {
    emitC(static_cast<Instr*>(hir_c_create_snapshot(
        const_cast<void*>(static_cast<const void*>(&frame)))));
  }

  // Insert a pre-created instruction from a C factory.
  // Sets bytecode offset and appends to current block.
  Instr* emitC(Instr* instr) {
    instr->setBytecodeOffset(frame.instrOffset());
    block->Append(instr);
    return instr;
  }

  // Convenience: emit LoadConst via pure C factory.
  void emitLoadConst(Register* dst, Type type) {
    emitC(static_cast<Instr*>(hir_c_create_load_const(dst, to_hir(type))));
  }

  // GuardType via C++ bridge (no FrameState).
  void emitGuardType(Register* dst, Type target, Register* src) {
    emitC(static_cast<Instr*>(
        hir_c_create_guard_type_reg(dst, to_hir(target), src)));
  }

  // GuardType via C++ bridge (with FrameState).
  void emitGuardType(Register* dst, Type target, Register* src,
                     const FrameState& fs) {
    auto* instr = emitC(static_cast<Instr*>(
        hir_c_create_guard_type_reg(dst, to_hir(target), src)));
    static_cast<DeoptBase*>(instr)->setFrameState(fs);
  }

  // RefineType via C factory (caller-provided register).
  void emitRefineType(Register* dst, Type type, Register* src) {
    emitC(static_cast<Instr*>(
        hir_c_create_refine_type_reg(dst, to_hir(type), src)));
  }

  // CheckExc via C factory (no FrameState).
  void emitCheckExc(Register* dst, Register* src) {
    emitC(static_cast<Instr*>(hir_c_create_check_exc_reg(dst, src)));
  }

  // CheckExc via C factory (with FrameState).
  void emitCheckExc(Register* dst, Register* src, const FrameState& fs) {
    auto* instr = emitC(static_cast<Instr*>(hir_c_create_check_exc_reg(dst, src)));
    static_cast<DeoptBase*>(instr)->setFrameState(fs);
  }

  // Branch via C++ bridge (Edge::set_to). Returns instruction.
  Instr* emitBranch(BasicBlock* target) {
    return emitC(static_cast<Instr*>(hir_c_create_branch_cpp(target)));
  }

  // CondBranch via C++ bridge (Edge::set_to). Returns instruction.
  Instr* emitCondBranch(Register* cond, BasicBlock* true_bb, BasicBlock* false_bb) {
    return emitC(static_cast<Instr*>(
        hir_c_create_cond_branch_cpp(cond, true_bb, false_bb)));
  }

  // Deopt via C factory (0 operands). Returns Instr* for post-creation mutation.
  Instr* emitDeopt() {
    return emitC(static_cast<Instr*>(hir_c_create_deopt()));
  }

  // Return via C factory.
  void emitReturn(Register* src, Type type) {
    emitC(static_cast<Instr*>(hir_c_create_return(src, to_hir(type))));
  }

  // VectorCall via C++ bridge (returns VectorCall* for operand wiring).
  VectorCall* emitVectorCall(size_t n_operands, Register* dst, CallFlags flags) {
    return static_cast<VectorCall*>(emitC(static_cast<Instr*>(
        hir_c_create_vectorcall_reg(n_operands, dst,
                                     static_cast<uint32_t>(flags)))));
  }

  // CondBranchCheckType via C++ bridge (Edge::set_to). Returns instruction.
  Instr* emitCondBranchCheckType(Register* target, Type type,
                                  BasicBlock* true_bb, BasicBlock* false_bb) {
    return emitC(static_cast<Instr*>(
        hir_c_create_cond_branch_check_type_cpp(target, to_hir(type),
                                                 true_bb, false_bb)));
  }

  // Assign via C factory.
  void emitAssign(Register* dst, Register* src) {
    emitC(static_cast<Instr*>(hir_assign_create(dst, src)));
  }

  // PrimitiveCompare via pure C factory (hir_instr_c.h).
  void emitPrimitiveCompare(Register* dst, PrimitiveCompareOp op,
                             Register* left, Register* right) {
    emitC(static_cast<Instr*>(hir_c_create_primitive_compare(
        dst, static_cast<int32_t>(op), left, right)));
  }

  // PrimitiveBoxBool via pure C factory.
  void emitPrimitiveBoxBool(Register* dst, Register* src) {
    emitC(static_cast<Instr*>(hir_c_create_primitive_box_bool(dst, src)));
  }

  // IntBinaryOp via pure C factory.
  void emitIntBinaryOp(Register* dst, BinaryOpKind op,
                        Register* left, Register* right) {
    emitC(static_cast<Instr*>(hir_c_create_int_binary_op(
        dst, static_cast<int32_t>(op), left, right)));
  }

  // CheckSequenceBounds via C++ bridge (DeoptBase + FrameState).
  void emitCheckSequenceBounds(Register* dst, Register* seq, Register* idx,
                                const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_check_seq_bounds_reg(
        dst, seq, idx, const_cast<void*>(static_cast<const void*>(&fs)))));
  }

  // CheckField via C++ bridge (DeoptBase + FrameState). Returns DeoptBase* for mutation.
  DeoptBase* emitCheckField(Register* dst, Register* src, PyObject* name,
                             const FrameState& fs) {
    return static_cast<DeoptBase*>(emitC(static_cast<Instr*>(
        hir_c_create_check_field_reg(
            dst, src, name,
            const_cast<void*>(static_cast<const void*>(&fs))))));
  }

  // BinaryOp via C++ bridge (DeoptBase + FrameState).
  void emitBinaryOp(Register* dst, BinaryOpKind op, Register* left,
                     Register* right, const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_binary_op_reg(
        dst, static_cast<int32_t>(op), left, right,
        const_cast<void*>(static_cast<const void*>(&fs)))));
  }

  // GuardIs via C++ bridge (caller-provided register). Returns DeoptBase*.
  DeoptBase* emitGuardIs(Register* dst, PyObject* target, Register* src) {
    return static_cast<DeoptBase*>(emitC(static_cast<Instr*>(
        hir_c_create_guard_is_reg(dst, target, src))));
  }

  // SetDictItem via C++ bridge (DeoptBase, 3 operands + FrameState).
  void emitSetDictItem(Register* dst, Register* dict, Register* key,
                        Register* value, const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_set_dict_item_reg(
        dst, dict, key, value,
        const_cast<void*>(static_cast<const void*>(&fs)))));
  }

  // LoadTupleItem via C factory (HasOutput, 1 operand + index).
  void emitLoadTupleItem(Register* dst, Register* tuple, Py_ssize_t idx) {
    emitC(static_cast<Instr*>(
        hir_c_create_load_tuple_item_reg(dst, tuple, static_cast<int32_t>(idx))));
  }

  // LoadFieldAddress via C factory.
  void emitLoadFieldAddress(Register* dst, Register* object, Register* offset) {
    emitC(static_cast<Instr*>(
        hir_c_create_load_field_address_reg(dst, object, offset)));
  }

  // YieldValue via C++ bridge (DeoptBase + FrameState).
  void emitYieldValue(Register* dst, Register* src, const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_yield_value_reg(
        dst, src, const_cast<void*>(static_cast<const void*>(&fs)))));
  }

  // SetCurrentAwaiter via C factory (1 operand, no output).
  void emitSetCurrentAwaiter(Register* src) {
    emitC(static_cast<Instr*>(hir_c_create_set_current_awaiter_reg(src)));
  }

  // Decref or XDecref depending on nullability.
  // Uses Decref for non-nullable (TObject), XDecref for nullable (TOptObject).
  void emitDecref(Register* src) {
    if (src->type() <= TObject) {
      emitC(static_cast<Instr*>(hir_c_create_decref_reg(src)));
    } else {
      emit<XDecref>(src);
    }
  }

  // MakeCell via C++ bridge (DeoptBase + FrameState).
  void emitMakeCell(Register* dst, Register* src, const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_make_cell_reg(
        dst, src, const_cast<void*>(static_cast<const void*>(&fs)))));
  }

  // InitialYield via C++ bridge (DeoptBase + FrameState).
  void emitInitialYield(Register* dst, const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_initial_yield_reg(
        dst, const_cast<void*>(static_cast<const void*>(&fs)))));
  }

  // LoadArg via C factory (HasOutput, index + type).
  void emitLoadArg(Register* dst, int idx, Type type) {
    emitC(static_cast<Instr*>(
        hir_c_create_load_arg_reg(dst, static_cast<int32_t>(idx), to_hir(type))));
  }

  // YieldFrom via C++ bridge (DeoptBase + FrameState).
  void emitYieldFrom(Register* dst, Register* send_value, Register* iter,
                      const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_yield_from_reg(
        dst, send_value, iter,
        const_cast<void*>(static_cast<const void*>(&fs)))));
  }

  // CheckVar via C++ bridge (DeoptBase + FrameState).
  void emitCheckVar(Register* dst, Register* src, PyObject* name,
                     const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_check_var_reg(
        dst, src, name,
        const_cast<void*>(static_cast<const void*>(&fs)))));
  }

  // CondBranchIterNotDone via C++ bridge (Edge::set_to).
  void emitCondBranchIterNotDone(Register* src, BasicBlock* body, BasicBlock* done) {
    emitC(static_cast<Instr*>(
        hir_c_create_cond_branch_iter_not_done_cpp(src, body, done)));
  }

  // IntConvert via C factory (HasOutput, type field).
  void emitIntConvert(Register* dst, Register* src, Type type) {
    emitC(static_cast<Instr*>(
        hir_c_create_int_convert_reg(dst, src, to_hir(type))));
  }

  // GetIter via C++ bridge (DeoptBase + FrameState).
  void emitGetIter(Register* dst, Register* src, const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_get_iter_reg(
        dst, src, const_cast<void*>(static_cast<const void*>(&fs)))));
  }

  // Batch: simple DEFINE_SIMPLE_INSTR wrappers
  void emitRaise(const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_raise_reg(
        const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  void emitWaitHandleRelease(Register* src) {
    emitC(static_cast<Instr*>(hir_c_create_wait_handle_release_reg(src)));
  }
  void emitMakeSet(Register* dst, const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_make_set_reg(
        dst, const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  void emitDeleteAttr(Register* receiver, int name_idx, const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_delete_attr_reg(
        receiver, name_idx, const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  void emitDeleteSubscr(Register* container, Register* sub, const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_delete_subscr_reg(
        container, sub, const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  void emitStoreAttr(Register* receiver, Register* value, int name_idx, const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_store_attr_reg(
        receiver, value, name_idx, const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  void emitSwapCellItem(Register* dst, Register* cell, Register* value) {
    emitC(static_cast<Instr*>(hir_c_create_swap_cell_item_reg(dst, cell, value)));
  }
  void emitStealCellItem(Register* dst, Register* cell) {
    emitC(static_cast<Instr*>(hir_c_create_steal_cell_item_reg(dst, cell)));
  }
  void emitSetCellItem(Register* cell, Register* value, Register* old) {
    emitC(static_cast<Instr*>(hir_c_create_set_cell_item_reg(cell, value, old)));
  }
  void emitAtQuiescentState() {
    emitC(static_cast<Instr*>(hir_c_create_at_quiescent_state_reg()));
  }
  void emitRunPeriodicTasks(Register* dst, const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_run_periodic_tasks_reg(
        dst, const_cast<void*>(static_cast<const void*>(&fs)))));
  }

  // Batch 2 wrappers
  void emitWaitHandleLoadWaiter(Register* dst, Register* src) {
    emitC(static_cast<Instr*>(hir_c_create_wait_handle_load_waiter_reg(dst, src)));
  }
  void emitWaitHandleLoadCoroOrResult(Register* dst, Register* src) {
    emitC(static_cast<Instr*>(hir_c_create_wait_handle_load_coro_reg(dst, src)));
  }
  void emitSetUpdate(Register* dst, Register* set, Register* iter, const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_set_update_reg(dst, set, iter, const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  void emitDictUpdate(Register* dst, Register* dict, Register* update, const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_dict_update_reg(dst, dict, update, const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  void emitListExtend(Register* dst, Register* list, Register* iter, const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_list_extend_reg(dst, list, iter, const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  void emitCopyDictWithoutKeys(Register* dst, Register* subj, Register* keys, const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_copy_dict_without_keys_reg(dst, subj, keys, const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  void emitMakeTupleFromList(Register* dst, Register* list, const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_make_tuple_from_list_reg(dst, list, const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  void emitListAppend(Register* dst, Register* list, Register* item, const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_list_append_reg(dst, list, item, const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  void emitCheckFreevar(Register* dst, Register* src, PyObject* name, const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_check_freevar_reg(dst, src, name, const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  void emitLoadGlobal(Register* dst, int name_idx, const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_load_global_reg(dst, name_idx, const_cast<void*>(static_cast<const void*>(&fs)))));
  }

  // StoreSubscr via C++ bridge (DeoptBase, no output).
  void emitStoreSubscr(Register* container, Register* sub, Register* value,
                        const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_store_subscr_reg(
        container, sub, value,
        const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  // SetSetItem via C++ bridge (DeoptBase, HasOutput).
  void emitSetSetItem(Register* dst, Register* set, Register* item,
                       const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_set_set_item_reg(
        dst, set, item, const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  // InPlaceOp via C++ bridge.
  void emitInPlaceOp(Register* dst, InPlaceOpKind op, Register* left,
                      Register* right, const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_in_place_op_reg(
        dst, static_cast<int32_t>(op), left, right,
        const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  // Compare via C++ bridge.
  void emitCompare(Register* dst, CompareOp op, Register* left,
                    Register* right, const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_compare_reg(
        dst, static_cast<int32_t>(op), left, right,
        const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  // FormatWithSpec via C++ bridge.
  void emitFormatWithSpec(Register* dst, Register* value, Register* fmt_spec,
                           const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_format_with_spec_reg(
        dst, value, fmt_spec,
        const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  // MakeDict via C++ bridge.
  void emitMakeDict(Register* dst, Py_ssize_t dict_size, const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_make_dict_reg(
        dst, static_cast<int32_t>(dict_size),
        const_cast<void*>(static_cast<const void*>(&fs)))));
  }

  // Batch 3 wrappers
  void emitDictMerge(Register* dst, Register* dict, Register* update, Register* func, const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_dict_merge_reg(dst, dict, update, func, const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  void emitDictSubscr(Register* dst, Register* dict, Register* key, const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_dict_subscr_reg(dst, dict, key, const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  void emitSend(Register* iter, Register* vout, Register* vin, const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_send_reg(iter, vout, vin, const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  void emitConvertValue(Register* dst, Register* value, int conversion, const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_convert_value_reg(dst, value, conversion, const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  void emitUnaryOp(Register* dst, UnaryOpKind op, Register* operand, const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_unary_op_reg(dst, static_cast<int32_t>(op), operand, const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  void emitImportFrom(Register* dst, Register* name, int name_idx, const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_import_from_reg(dst, name, name_idx, const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  void emitInvokeIterNext(Register* dst, Register* iter, const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_invoke_iter_next_reg(dst, iter, const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  void emitPrimitiveUnbox(Register* dst, Register* src, Type type) {
    emitC(static_cast<Instr*>(hir_c_create_primitive_unbox_reg(dst, src, to_hir(type))));
  }

  // Batch: 1-op HasOutput DeoptBase (dst, src, frame)
  void emitGetAIter(Register* dst, Register* src, const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_get_a_iter_reg(
        dst, src, const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  void emitGetANext(Register* dst, Register* src, const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_get_a_next_reg(
        dst, src, const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  void emitGetTuple(Register* dst, Register* src, const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_get_tuple_reg(
        dst, src, const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  void emitIsNegativeAndErrOccurred(Register* dst, Register* src, const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_is_neg_and_err_reg(
        dst, src, const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  // 0/1-op no-frame
  void emitLoadCellItem(Register* dst, Register* src) {
    emitC(static_cast<Instr*>(hir_c_create_load_cell_item_reg(dst, src)));
  }
  void emitLoadCurrentFunc(Register* dst) {
    emitC(static_cast<Instr*>(hir_c_create_load_current_func_reg(dst)));
  }
  void emitLoadEvalBreaker(Register* dst) {
    emitC(static_cast<Instr*>(hir_c_create_load_eval_breaker_reg(dst)));
  }
  void emitLoadFrame() {
    emitC(static_cast<Instr*>(hir_c_create_load_frame_reg()));
  }
  void emitLoadVarObjectSize(Register* dst, Register* src) {
    emitC(static_cast<Instr*>(hir_c_create_load_var_object_size_reg(dst, src)));
  }
  void emitCheckErrOccurred(const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_check_err_occurred_reg(
        const_cast<void*>(static_cast<const void*>(&fs)))));
  }

  // IsTruthy via C++ bridge (DeoptBase + FrameState).
  void emitIsTruthy(Register* dst, Register* src, const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_is_truthy_reg(
        dst, src, const_cast<void*>(static_cast<const void*>(&fs)))));
  }

  // GetSecondOutput via C factory.
  void emitGetSecondOutput(Register* dst, Type type, Register* src) {
    emitC(static_cast<Instr*>(
        hir_c_create_get_second_output_reg(dst, to_hir(type), src)));
  }

  // SetFunctionAttr via C factory.
  void emitSetFunctionAttr(Register* value, Register* base, FunctionAttr field) {
    emitC(static_cast<Instr*>(hir_c_create_set_function_attr_reg(
        value, base, static_cast<int32_t>(field))));
  }

  // CheckNeg via C++ bridge (DeoptBase + FrameState).
  void emitCheckNeg(Register* dst, Register* src, const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_check_neg_reg(
        dst, src, const_cast<void*>(static_cast<const void*>(&fs)))));
  }

  // GetLength via C++ bridge (DeoptBase + FrameState).
  void emitGetLength(Register* dst, Register* src, const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_get_length_reg(
        dst, src, const_cast<void*>(static_cast<const void*>(&fs)))));
  }

  // PrimitiveBox via C++ bridge (DeoptBase + FrameState).
  void emitPrimitiveBox(Register* dst, Register* src, Type type,
                         const FrameState& fs) {
    emitC(static_cast<Instr*>(hir_c_create_primitive_box_reg(
        dst, src, to_hir(type),
        const_cast<void*>(static_cast<const void*>(&fs)))));
  }

  // LoadArrayItem via C++ bridge (caller-provided register).
  void emitLoadArrayItem(Register* dst, Register* arr, Register* idx,
                          Register* container, intptr_t offset, Type type) {
    emitC(static_cast<Instr*>(hir_c_create_load_array_item_reg(
        dst, arr, idx, container, offset, to_hir(type))));
  }

  // Guard via C factory (with FrameState). Returns instruction for mutation.
  Instr* emitGuard(Register* src, const FrameState& fs) {
    auto* instr = emitC(static_cast<Instr*>(hir_c_create_guard(src)));
    static_cast<DeoptBase*>(instr)->setFrameState(fs);
    return instr;
  }

  // BitCast via pure C factory.
  void emitBitCast(Register* dst, Register* src, Type type) {
    emitC(static_cast<Instr*>(hir_c_create_bit_cast(dst, src, to_hir(type))));
  }

  // DoubleBinaryOp via pure C factory.
  void emitDoubleBinaryOp(Register* dst, BinaryOpKind op,
                           Register* left, Register* right) {
    emitC(static_cast<Instr*>(hir_c_create_double_binary_op(
        dst, static_cast<int32_t>(op), left, right)));
  }

  // PrimitiveUnaryOp via pure C factory.
  void emitPrimitiveUnaryOp(Register* dst, PrimitiveUnaryOpKind op,
                              Register* src) {
    emitC(static_cast<Instr*>(hir_c_create_primitive_unary_op(
        dst, static_cast<int32_t>(op), src)));
  }

  // LoadField via C++ bridge (caller-provided register, no FrameState).
  void emitLoadField(Register* dst, Register* receiver, const char* name,
                      intptr_t offset, Type type, bool borrowed = false) {
    emitC(static_cast<Instr*>(hir_c_create_load_field_reg(
        dst, receiver, name, offset, to_hir(type), borrowed ? 1 : 0)));
  }

  // CallStatic via C++ bridge (returns CallStatic* for operand wiring).
  CallStatic* emitCallStatic(size_t n_operands, Register* dst, void* addr,
                              Type ret_type) {
    return static_cast<CallStatic*>(emitC(static_cast<Instr*>(
        hir_c_create_call_static_reg(n_operands, dst, addr, to_hir(ret_type)))));
  }

  // UseType via pure C factory.
  void emitUseType(Register* val, Type type) {
    emitC(static_cast<Instr*>(hir_c_create_use_type(val, to_hir(type))));
  }

  BasicBlock* block{nullptr};
  FrameState frame;
};

void HIRBuilder::addInitialYield(TranslationContext& tc) {
  auto out = temps_.AllocateNonStack();
  tc.emitInitialYield(out, tc.frame);
}

// Add LoadArg instructions for each function argument. This ensures that the
// corresponding variables are always assigned and allows for a uniform
// treatment of registers that correspond to arguments (vs locals) during
// definite assignment analysis.
void HIRBuilder::addLoadArgs(TranslationContext& tc, int num_args) {
  PyCodeObject* code = tc.frame.code;
  int starargs_idx = (code->co_flags & CO_VARARGS)
      ? code->co_argcount + code->co_kwonlyargcount
      : -1;
  for (int i = 0; i < num_args; i++) {
    // Arguments in CPython are the first N locals.
    Register* dst = tc.frame.localsplus[i];
    JIT_CHECK(dst != nullptr, "No register for argument {}", i);
    if (i == starargs_idx) {
      tc.emitLoadArg(dst, i, TTupleExact);
    } else {
      Type type = preloader_.checkArgType(i);
      tc.emitLoadArg(dst, i, type);
    }
  }
}

// Add a MakeCell for each cellvar and load each freevar from closure.
//
// Note: This is only necessary for 3.10.  For 3.12 we have the explicit
// MAKE_CELL and COPY_FREE_VARS instructions.
void HIRBuilder::addInitializeCells([[maybe_unused]] TranslationContext& tc) {
#if PY_VERSION_HEX < 0x030C0000
  int nlocals = tc.frame.nlocals;
  int ncellvars = numCellvars(code_);
  int nfreevars = numFreevars(code_);

  Register* null_reg = ncellvars > 0 ? temps_.AllocateNonStack() : nullptr;
  for (int i = 0; i < ncellvars; ++i) {
    int arg = CO_CELL_NOT_AN_ARG;
    Register* dst = tc.frame.localsplus[i + nlocals];
    JIT_CHECK(dst != nullptr, "No register for cell {}", i);
    Register* cell_contents = null_reg;
    if (code_->co_cell2arg != nullptr &&
        (arg = code_->co_cell2arg[i]) != CO_CELL_NOT_AN_ARG) {
      // cell is for argument local number `arg`
      JIT_CHECK(
          static_cast<unsigned>(arg) < tc.frame.nlocals,
          "co_cell2arg says cell {} is local {} but locals size is {}",
          i,
          arg,
          tc.frame.nlocals);
      cell_contents = tc.frame.localsplus[arg];
    }
    tc.emitMakeCell(dst, cell_contents, tc.frame);
    if (arg != CO_CELL_NOT_AN_ARG) {
      // Clear the local once we have it in a cell.
      tc.frame.localsplus[arg] = null_reg;
    }
  }

  if (nfreevars != 0) {
    emitCopyFreeVars(tc, nfreevars);
  }
#endif
}

static bool should_snapshot(
    const BytecodeInstruction& bci,
    bool is_in_async_for_header_block) {
  // Taking a snapshot after a terminator doesn't make sense, as control either
  // transfers to another basic block or the function ends.
  if (bci.isTerminator()) {
    return false;
  }

  switch (bci.opcode()) {
    // These instructions only modify frame state and are always safe to
    // replay. We don't snapshot these in order to limit the amount of
    // unnecessary metadata in the lowered IR.
    case CONVERT_PRIMITIVE:
    case COPY:
    case DUP_TOP_TWO:
    case DUP_TOP:
    case END_FOR:
    case EXTENDED_ARG:
    case IS_OP:
    case KW_NAMES:
    case LOAD_ASSERTION_ERROR:
    case LOAD_CLOSURE:
    case LOAD_CONST:
    case LOAD_FAST_AND_CLEAR:
    case LOAD_FAST_BORROW_LOAD_FAST_BORROW:
    case LOAD_FAST_BORROW:
    case LOAD_FAST_CHECK:
    case LOAD_FAST_LOAD_FAST:
    case LOAD_FAST:
    case LOAD_LOCAL:
    case NOP:
    case POP_ITER:
    case POP_TOP:
    case PRIMITIVE_BOX:
    case PRIMITIVE_LOAD_CONST:
    case PRIMITIVE_UNARY_OP:
    case PRIMITIVE_UNBOX:
    case PUSH_NULL:
    case REFINE_TYPE:
    case ROT_FOUR:
    case ROT_N:
    case ROT_THREE:
    case ROT_TWO:
    case STORE_FAST_LOAD_FAST:
    case STORE_FAST_STORE_FAST:
    case STORE_FAST:
    case STORE_LOCAL:
    case SWAP: {
      return false;
    }
    // In an async-for header block YIELD_FROM controls whether we end the loop
    case YIELD_FROM: {
      return !is_in_async_for_header_block;
    }
    case JUMP_IF_NOT_EXC_MATCH:
    case RERAISE:
    case WITH_EXCEPT_START: {
      JIT_ABORT(
          "Should not be compiling except blocks (opcode {}, {})\n",
          bci.opcode(),
          opcodeName(bci.opcode()));
    }
    // Take a snapshot after translating all other bytecode instructions. This
    // may generate unnecessary deoptimization metadata but will always be
    // correct.
    default: {
      return true;
    }
  }
}

// Compute basic block boundaries and allocate corresponding HIR blocks
HIRBuilder::BlockMap HIRBuilder::createBlocks(
    Function& irfunc,
    const BytecodeInstructionBlock& bc_block) {
  BlockMap block_map;

  // Mark the beginning of each basic block in the bytecode
  std::set<BCIndex> block_starts = {BCIndex{0}};
  auto maybe_add_next_instr = [&](const BytecodeInstruction& bc_instr) {
    BCIndex next_instr_idx = bc_instr.nextInstrOffset();
    if (next_instr_idx < bc_block.size()) {
      block_starts.insert(next_instr_idx);
    }
  };
  for (auto bc_instr : bc_block) {
    if (bc_instr.isBranch()) {
      maybe_add_next_instr(bc_instr);
      BCIndex target = bc_instr.getJumpTarget();
      block_starts.insert(target);
    } else {
      auto opcode = bc_instr.opcode();
      if (
          // We always split after YIELD_FROM to handle the case where it's the
          // top of an async-for loop and so generate a HIR conditional jump.
          bc_instr.isTerminator() || (opcode == YIELD_FROM)) {
        maybe_add_next_instr(bc_instr);
      } else {
        JIT_CHECK(!bc_instr.isTerminator(), "Terminator should split block");
      }
    }
  }

  // Parse co_exceptiontable and add handler targets as block starts.
  // This ensures exception handler basic blocks are created in the HIR,
  // even though Python 3.12+ does not emit SETUP_FINALLY opcodes.
  parseExceptionTable();
  for (const auto& entry : exception_table_) {
    block_starts.insert(entry.target.asIndex());
    // B2: Also add except body start so we can branch to it.
    SimpleExceptInfo info;
    if (getSimpleExceptInfo(entry, info)) {
      block_starts.insert(info.except_body.asIndex());
    }
  }

  // Allocate blocks
  auto it = block_starts.begin();
  while (it != block_starts.end()) {
    BCIndex start_idx = *it;
    ++it;
    BCIndex end_idx;
    if (it != block_starts.end()) {
      end_idx = *it;
    } else {
      end_idx = BCIndex{bc_block.size()};
    }
    auto block = irfunc.cfg.AllocateBlock();
    block_map.blocks[start_idx] = block;
    block_map.bc_blocks.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(block),
        std::forward_as_tuple(bc_block.code(), start_idx, end_idx));
  }

  return block_map;
}


void HIRBuilder::parseExceptionTable() {
  PyObject* table_obj = code_->co_exceptiontable;
  if (table_obj == nullptr || !PyBytes_Check(table_obj)) {
    return;
  }
  const uint8_t* table =
      reinterpret_cast<const uint8_t*>(PyBytes_AS_STRING(table_obj));
  Py_ssize_t length = PyBytes_GET_SIZE(table_obj);
  Py_ssize_t pos = 0;

    // Variable-length integer decoder matching CPython 3.12 format (MSB first).
    // Each byte: bits 0-5 = payload, bit 6 = continuation.
    // First byte holds the most-significant bits of the value.
    auto parse_varint = [&]() -> int {
      int val = 0;
      uint8_t b;
      do {
        JIT_DCHECK(pos < length, "Truncated exception table");
        b = table[pos++];
        val = (val << 6) | (b & 0x3F);
      } while (b & 0x40);
      return val;
    };
  while (pos < length) {
    int start = parse_varint();
    int size = parse_varint();
    int target = parse_varint();
    int depth_lasti = parse_varint();

    // Exception table values are in instruction units (word offsets).
    // Convert to byte offsets by multiplying by sizeof(_Py_CODEUNIT).
    constexpr int scale = sizeof(_Py_CODEUNIT);
    exception_table_.push_back(ExceptionTableEntry{
        BCOffset{start * scale},
        BCOffset{(start + size) * scale},
        BCOffset{target * scale},
        depth_lasti >> 1,
        (depth_lasti & 1) != 0});
  }
}

const HIRBuilder::ExceptionTableEntry* HIRBuilder::findExceptionHandler(
    BCOffset off) const {
  for (const auto& entry : exception_table_) {
    if (off >= entry.start && off < entry.end) {
      return &entry;
    }
  }
  return nullptr;
}

bool HIRBuilder::getSimpleExceptInfo(
    const ExceptionTableEntry& handler,
    SimpleExceptInfo& info) const {
  // Scan handler bytecodes for the pattern:
  //   PUSH_EXC_INFO, LOAD_GLOBAL <type>, CHECK_EXC_MATCH,
  //   POP_JUMP_IF_FALSE, POP_TOP
  BytecodeInstruction bc{code_, handler.target};

  if (bc.opcode() != PUSH_EXC_INFO) {
    return false;
  }
  bc = bc.nextInstr();

  if (bc.opcode() != LOAD_GLOBAL) {
    return false;
  }
  int name_idx = loadGlobalIndex(bc.oparg());
  bc = bc.nextInstr();

  if (bc.opcode() != CHECK_EXC_MATCH) {
    return false;
  }
  bc = bc.nextInstr();

  if (bc.opcode() != POP_JUMP_IF_FALSE) {
    return false;
  }
  bc = bc.nextInstr();

  if (bc.opcode() != POP_TOP) {
    return false;
  }
  BCOffset except_body = bc.nextInstrOffset();

  // Resolve exception type at JIT compile time via preloader.
  BorrowedRef<> exc_type = preloader_.global(name_idx);
  if (exc_type == nullptr) {
    return false;
  }
  if (!PyExceptionClass_Check(exc_type.get())) {
    return false;
  }

  info.name_idx = name_idx;
  info.exc_type = exc_type;
  info.except_body = except_body;
  return true;
}

void HIRBuilder::emitInlineExceptionMatch(
    CFG& cfg,
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr,
    const ExceptionTableEntry& handler,
    const SimpleExceptInfo& info,
    Register* left,
    Register* right,
    Register* result) {
  // Emit dict lookup via CallStatic. For BINARY_SUBSCR_DICT (known dict
  // type), use JITRT_DictGetItem which calls PyDict_GetItemWithError
  // directly, avoiding PyObject_GetItem's generic type dispatch overhead.
  void* getitem_fn = (bc_instr.opcode() == BINARY_SUBSCR_DICT)
      ? reinterpret_cast<void*>(JITRT_DictGetItem)
      : reinterpret_cast<void*>(PyObject_GetItem);
  auto call = tc.emitCallStatic(2, result, getitem_fn, TOptObject);
  call->SetOperand(0, left);
  call->SetOperand(1, right);

  BasicBlock* ok_block = cfg.AllocateBlock();
  BasicBlock* exc_match_block = cfg.AllocateBlock();

  // Branch: non-null -> success, null -> exception path
  tc.emitCondBranch(result, ok_block, exc_match_block);

  // === Exception match block ===
  // Use a SEPARATE TranslationContext so we do not corrupt tc.frame.stack.
  {
    TranslationContext exc_tc{exc_match_block, tc.frame};

    // Pop excess stack items above handler depth.
    // Refcount insertion will handle decrement automatically.
    while (static_cast<int>(exc_tc.frame.stack.size()) > handler.depth) {
      exc_tc.frame.stack.pop();
    }

    // Load exception type as a constant (resolved at compile time).
    Register* exc_type_reg = temps_.AllocateNonStack();
    exc_tc.emitLoadConst(exc_type_reg, Type::fromObject(info.exc_type));

    // Call JITRT_MatchAndClearException(exc_type) via CallStatic.
    // Returns int (TCInt32): 1 = matched, 0 = no match.
    Register* match_result = temps_.AllocateNonStack();
    auto match_call = exc_tc.emitCallStatic(
        1, match_result,
        reinterpret_cast<void*>(JITRT_MatchAndClearException),
        TCInt32);
    match_call->SetOperand(0, exc_type_reg);

    BasicBlock* match_block = cfg.AllocateBlock();
    BasicBlock* deopt_block = cfg.AllocateBlock();
    exc_tc.emitCondBranch(match_result, match_block, deopt_block);

    // === Match block: emit except body bytecodes inline ===
    // Delegate to existing emit* methods to handle all edge cases.
    {
      TranslationContext match_tc{match_block, exc_tc.frame};
      match_tc.frame.cur_instr_offs = info.except_body;

      // Push Py_None as the "previous exception" placeholder.
      // In CPython's interpreter, PUSH_EXC_INFO pushes the old
      // exc_info->exc_value onto the stack so POP_EXCEPT can restore
      // it. The JIT skips PUSH_EXC_INFO but the handler body bytecodes
      // (SWAP, POP_EXCEPT) still expect prev_exc on the stack. If the
      // handler body hits an unsupported opcode and deopts, the
      // interpreter must find a valid prev_exc here — otherwise
      // POP_EXCEPT writes a garbage pointer into exc_info->exc_value,
      // corrupting the exception context chain.
      Register* prev_exc_reg = temps_.AllocateStack();
      match_tc.emitLoadConst(prev_exc_reg, TNoneType);
      match_tc.frame.stack.push(prev_exc_reg);

      BytecodeInstruction ebc{code_, info.except_body};
      bool emitted_terminator = false;

      while (!emitted_terminator) {
        switch (ebc.opcode()) {
          case POP_EXCEPT:
            // Pop the prev_exc placeholder we pushed above.
            // In CPython, POP_EXCEPT pops the saved exception and
            // restores it to exc_info->exc_value. Our placeholder is
            // Py_None, which is correct: there was no active exception
            // before the JIT's inline handler.
            match_tc.frame.stack.pop();
            break;

          case POP_TOP:
            match_tc.frame.stack.pop();
            break;

          case SWAP:
            emitSwap(match_tc, ebc.oparg());
            break;

          case LOAD_FAST:
          case LOAD_FAST_CHECK:
          case LOAD_FAST_AND_CLEAR:
            emitLoadFast(match_tc, ebc);
            break;

          case LOAD_CONST: {
            Register* reg = temps_.AllocateStack();
            Type type = Type::fromObject(
                PyTuple_GET_ITEM(code_->co_consts, ebc.oparg()));
            match_tc.emitLoadConst(reg, type);
            match_tc.frame.stack.push(reg);
            break;
          }

          case STORE_FAST:
            emitStoreFast(match_tc, ebc);
            break;

          case BINARY_OP:
            emitBinaryOp(cfg, match_tc, ebc);
            break;

          case RETURN_CONST: {
            Register* ret_reg = temps_.AllocateStack();
            Type type = Type::fromObject(
                PyTuple_GET_ITEM(code_->co_consts, ebc.oparg()));
            match_tc.emitLoadConst(ret_reg, type);
            match_tc.emitReturn(ret_reg, type);
            emitted_terminator = true;
            break;
          }

          case RETURN_VALUE: {
            Register* ret_val = match_tc.frame.stack.pop();
            match_tc.emitReturn(ret_val, preloader_.returnType());
            emitted_terminator = true;
            break;
          }

          case JUMP_BACKWARD:
          case JUMP_BACKWARD_NO_INTERRUPT: {
            BCOffset target = ebc.getJumpTarget();
            auto* target_block = getBlockAtOff(target);
            match_tc.emitBranch(target_block);
            emitted_terminator = true;
            break;
          }

          default:
            // Unsupported opcode: deopt to interpreter.
            match_tc.frame.cur_instr_offs = ebc.baseOffset();
            match_tc.emitSnapshot();
            match_tc.emitDeopt();
            emitted_terminator = true;
            break;
        }

        if (!emitted_terminator) {
          ebc = ebc.nextInstr();
        }
      }
    }

    // === Deopt block (no match -- let interpreter handle) ===
    // JITRT_MatchAndClearException returned 0 and restored the pending
    // exception via PyErr_SetRaisedException. We deopt back to the
    // interpreter at the BINARY_SUBSCR offset with left/right re-pushed
    // (emitBinaryOp already popped them, but the interpreter expects them
    // on stack at this offset for correct exception_unwind depth).
    //
    // Deopt uses kUnhandledException (default for Deopt instruction),
    // which causes the interpreter to enter the error handler directly
    // (throwflag=1). The interpreter calls exception_unwind, walks
    // co_exceptiontable, and finds the correct handler -- including
    // outer handlers for nested try blocks.
    {
      TranslationContext deopt_tc{deopt_block, tc.frame};
      // Push left (container) and right (key) back onto the stack.
      deopt_tc.frame.stack.push(left);
      deopt_tc.frame.stack.push(right);
      deopt_tc.frame.cur_instr_offs = bc_instr.baseOffset();
      deopt_tc.emitSnapshot();
      deopt_tc.emitDeopt();
    }
  }

  // === OK block (no exception, result is non-null) ===
  tc.block = ok_block;
  tc.emitRefineType(result, TObject, result);
}

void HIRBuilder::emitCallExceptionHandler(
    CFG& cfg,
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr,
    const ExceptionTableEntry& handler,
    const SimpleExceptInfo& info,
    DeoptBase* call_instr,
    Register* result) {
  // Suppress the auto null-check deopt in LIR codegen. We handle
  // exceptions inline via CondBranch instead. FrameState is preserved
  // for Simplify, register allocation, and other deopt paths.
  call_instr->setSuppressExceptionDeopt(true);

  // Pop the result that emitAnyCall pushed onto the stack.
  tc.frame.stack.pop();

  BasicBlock* ok_block = cfg.AllocateBlock();
  BasicBlock* exc_match_block = cfg.AllocateBlock();

  // Branch: non-null -> success, null -> exception path
  tc.emitCondBranch(result, ok_block, exc_match_block);

  // === Exception match block ===
  {
    TranslationContext exc_tc{exc_match_block, tc.frame};

    // Pop excess stack items above handler depth.
    // Refcount insertion will handle decrement automatically.
    while (static_cast<int>(exc_tc.frame.stack.size()) > handler.depth) {
      exc_tc.frame.stack.pop();
    }

    // Load exception type as a constant (resolved at compile time).
    Register* exc_type_reg = temps_.AllocateNonStack();
    exc_tc.emitLoadConst(exc_type_reg, Type::fromObject(info.exc_type));

    // Call JITRT_MatchAndClearException(exc_type) via CallStatic.
    // Returns int (TCInt32): 1 = matched, 0 = no match.
    Register* match_result = temps_.AllocateNonStack();
    auto match_call = exc_tc.emitCallStatic(
        1, match_result,
        reinterpret_cast<void*>(JITRT_MatchAndClearException),
        TCInt32);
    match_call->SetOperand(0, exc_type_reg);

    BasicBlock* match_block = cfg.AllocateBlock();
    BasicBlock* deopt_block = cfg.AllocateBlock();
    exc_tc.emitCondBranch(match_result, match_block, deopt_block);

    // === Match block: emit except body bytecodes inline ===
    {
      TranslationContext match_tc{match_block, exc_tc.frame};
      match_tc.frame.cur_instr_offs = info.except_body;

      // Push Py_None as "previous exception" placeholder — see comment
      // in the first emitInlineExceptionMatch for full rationale.
      Register* prev_exc_reg = temps_.AllocateStack();
      match_tc.emitLoadConst(prev_exc_reg, TNoneType);
      match_tc.frame.stack.push(prev_exc_reg);

      BytecodeInstruction ebc{code_, info.except_body};
      bool emitted_terminator = false;

      while (!emitted_terminator) {
        switch (ebc.opcode()) {
          case POP_EXCEPT:
            // Pop the prev_exc placeholder.
            match_tc.frame.stack.pop();
            break;

          case POP_TOP:
            match_tc.frame.stack.pop();
            break;

          case SWAP:
            emitSwap(match_tc, ebc.oparg());
            break;

          case LOAD_FAST:
          case LOAD_FAST_CHECK:
          case LOAD_FAST_AND_CLEAR:
            emitLoadFast(match_tc, ebc);
            break;

          case LOAD_CONST: {
            Register* reg = temps_.AllocateStack();
            Type type = Type::fromObject(
                PyTuple_GET_ITEM(code_->co_consts, ebc.oparg()));
            match_tc.emitLoadConst(reg, type);
            match_tc.frame.stack.push(reg);
            break;
          }

          case STORE_FAST:
            emitStoreFast(match_tc, ebc);
            break;

          case BINARY_OP:
            emitBinaryOp(cfg, match_tc, ebc);
            break;

          case RETURN_CONST: {
            Register* ret_reg = temps_.AllocateStack();
            Type type = Type::fromObject(
                PyTuple_GET_ITEM(code_->co_consts, ebc.oparg()));
            match_tc.emitLoadConst(ret_reg, type);
            match_tc.emitReturn(ret_reg, type);
            emitted_terminator = true;
            break;
          }

          case RETURN_VALUE: {
            Register* ret_val = match_tc.frame.stack.pop();
            match_tc.emitReturn(ret_val, preloader_.returnType());
            emitted_terminator = true;
            break;
          }

          case JUMP_BACKWARD:
          case JUMP_BACKWARD_NO_INTERRUPT: {
            BCOffset target = ebc.getJumpTarget();
            auto* target_block = getBlockAtOff(target);
            match_tc.emitBranch(target_block);
            emitted_terminator = true;
            break;
          }

          default:
            // Unsupported opcode: deopt to interpreter.
            match_tc.frame.cur_instr_offs = ebc.baseOffset();
            match_tc.emitSnapshot();
            match_tc.emitDeopt();
            emitted_terminator = true;
            break;
        }

        if (!emitted_terminator) {
          ebc = ebc.nextInstr();
        }
      }
    }

    // === Deopt block (no match -- let interpreter handle) ===
    // JITRT_MatchAndClearException returned 0 and restored the pending
    // exception. Deopt back to the interpreter at the CALL offset.
    // The interpreter calls exception_unwind, walks co_exceptiontable,
    // and finds the correct handler.
    {
      TranslationContext deopt_tc{deopt_block, tc.frame};
      deopt_tc.frame.cur_instr_offs = bc_instr.baseOffset();
      deopt_tc.emitSnapshot();
      deopt_tc.emitDeopt();
    }
  }

  // === OK block (no exception, result is non-null) ===
  tc.block = ok_block;
  tc.emitRefineType(result, TObject, result);
  tc.frame.stack.push(result);
}

BasicBlock* HIRBuilder::getBlockAtOff(BCOffset off) {
  auto it = block_map_.blocks.find(off);
  JIT_DCHECK(it != block_map_.blocks.end(), "No block for offset {}", off);
  return it->second;
}

std::unique_ptr<Function> buildHIR(const Preloader& preloader) {
  return HIRBuilder{preloader}.buildHIR();
}

// This performs an abstract interpretation over the bytecode for func in order
// to translate it from a stack to register machine. The translation proceeds
// in two passes over the bytecode. First, basic block boundaries are
// enumerated and a mapping from block start offset to basic block is
// created. Next, basic blocks are filled in by simulating the effect that each
// instruction has on the stack.
//
// The correctness of the translation depends on the invariant that the depth
// the operand stack is be constant at each program point.  All of the CPython
// bytecode that we currently support maintain this invariant. However, there
// are a few bytecodes that do not (e.g. SETUP_FINALLY). We will need to deal
// with that if we ever want to support compiling them.
std::unique_ptr<Function> HIRBuilder::buildHIR() {
  checkTranslate();

  std::unique_ptr<Function> irfunc = preloader_.makeFunction();
  buildHIRImpl(irfunc.get(), /*frame_state=*/nullptr);
  // Use removeTrampolineBlocks and removeUnreachableBlocks directly instead of
  // Run because the rest of CleanCFG requires SSA.
  removeTrampolineBlocks(&irfunc->cfg);
  removeUnreachableBlocks(*irfunc);
  return irfunc;
}

// Loop through each of the arguments on the current translation context and
// check and see if there is any annotation to guard against.
void HIRBuilder::emitTypeAnnotationGuards(TranslationContext& tc) {
  HirAnnotationIndex* index = preloader_.annotations();

  // Bail out if there are no annotations.
  if (!index) {
    return;
  }

  PyCodeObject* const code = tc.frame.code;
  bool first = true;

  for (int arg_idx = 0; arg_idx < preloader_.numArgs(); arg_idx++) {
    PyObject* annotation = hir_annotation_index_find(index, getVarname(code, arg_idx));

    // If there is no annotation or if the annotation is an unexpected type,
    // then skip over this argument.
    //
    // Note that this also skips over more complex types like unions. It could
    // be beneficial in the future to support runtime checks for these kinds of
    // annotations.
    if (!annotation || !PyType_Check(annotation)) {
      continue;
    }

    // If we have an annotation that we are going to guard against, we need to
    // emit a snapshot for the guard.
    //
    // It's likely that no bytecode instructions have been compiled yet, meaning
    // the instruction offset has not yet been set. Setting it to zero here
    // ensures that if we need to deopt that it starts executing the first
    // instruction.
    if (first) {
      first = false;
      tc.frame.cur_instr_offs = BCOffset(0);
      tc.emitSnapshot();
    }

    // Now guard against the type of the argument.
    auto arg = tc.frame.localsplus.at(arg_idx);
    JIT_CHECK(arg != nullptr, "No register for argument {}", arg_idx);

    Type type =
        Type::fromTypeExact(reinterpret_cast<PyTypeObject*>(annotation));

    tc.emitGuardType(arg, type, arg);
  }
}

BasicBlock* HIRBuilder::buildHIRImpl(
    Function* irfunc,
    FrameState* frame_state) {
  temps_ = TempAllocator(&irfunc->env);

  BytecodeInstructionBlock bc_instrs{code_};
  block_map_ = createBlocks(*irfunc, bc_instrs);
  if (frame_state != nullptr) {
    // Suppress exception table for inlined callees to prevent B2
    // (emitBinaryOp -> findExceptionHandler -> emitInlineExceptionMatch)
    // from creating reachable handler blocks. All exceptions deopt.
    exception_table_.clear();
  }

  // Ensure that the entry block isn't a loop header
  BasicBlock* entry_block = getBlockAtOff(BCOffset{0});
  for (const auto& bci : bc_instrs) {
    if (bci.isBranch() && bci.getJumpTarget() == 0) {
      entry_block = irfunc->cfg.AllocateBlock();
      break;
    }
  }
  if (frame_state == nullptr) {
    // Function is not being inlined (irfunc matches code) so set the whole
    // CFG's entry block.
    irfunc->cfg.entry_block = entry_block;
  }

  // Insert LoadArg, LoadClosureCell, and MakeCell/MakeNullCell instructions
  // for the entry block
  TranslationContext entry_tc{
      entry_block,
      FrameState{
          code_,
          preloader_.globals(),
          preloader_.builtins(),
          /*parent=*/frame_state}};
  allocateLocalsplus(&irfunc->env, entry_tc.frame);

  addLoadArgs(entry_tc, preloader_.numArgs());

  // Consider checking if the code object or preloader uses runtime func and
  // drop the frame_state == nullptr check.  Inlined functions should load a
  // const instead of using LoadCurrentFunc.
  if (frame_state == nullptr && irfunc->uses_runtime_func) {
    func_ = temps_.AllocateNonStack();
    entry_tc.emitLoadCurrentFunc(func_);
  }

#if PY_VERSION_HEX >= 0x030C0000
  if (frame_state == nullptr) {
    entry_tc.emitLoadFrame();
  }
#endif

  emitTypeAnnotationGuards(entry_tc);

  addInitializeCells(entry_tc);

  // In 3.12+ "Initial Yield" has an explicit bytecode instruction in
  // "RETURN_GENERATOR" and so is emitted at the appropriate time.
  if (PY_VERSION_HEX < 0x030C0000 && code_->co_flags & kCoFlagsAnyGenerator) {
    // InitialYield must be after args are loaded so they can be spilled to
    // the suspendable state. It must also come before anything which can
    // deopt as generator deopt assumes we're running from state stored
    // in a generator object.
    addInitialYield(entry_tc);
  }

  BasicBlock* first_block = getBlockAtOff(BCOffset{0});
  if (entry_block != first_block) {
    entry_block->appendWithOff<Branch>(BCOffset{0}, first_block);
  }

  entry_tc.block = first_block;
  translate(*irfunc, bc_instrs, entry_tc);

  return entry_block;
}

InlineResult HIRBuilder::inlineHIR(
    Function* caller,
    FrameState* caller_frame_state) {
  checkTranslate();

  BasicBlock* entry_block = buildHIRImpl(caller, caller_frame_state);
  // Make one block with a Return that merges the return branches from the
  // callee. After SSA, it will turn into a massive Phi. The caller can find
  // the Return and use it as the output of the call instruction.
  Register* return_val = caller->env.AllocateRegister();
  BasicBlock* exit_block = caller->cfg.AllocateBlock();
  if (preloader_.returnType() <= TPrimitive) {
    exit_block->append<Return>(return_val, preloader_.returnType());
  } else {
    exit_block->append<Return>(return_val);
  }
  for (auto block : caller->cfg.GetRPOTraversal(entry_block)) {
    auto instr = block->GetTerminator();
    if (instr->IsReturn()) {
      auto assign = Assign::create(return_val, instr->GetOperand(0));
      auto branch = Branch::create(exit_block);
      instr->ExpandInto({assign, branch});
      Instr::Destroy(instr);
    }
  }

  // Map of FrameState to parent pointers. We must completely disconnect the
  // inlined function's CFG from its caller for SSAify to run properly: it will
  // find uses (in FrameState) before defs and insert LoadConst<Nullptr>.
  UnorderedMap<FrameState*, FrameState*> framestate_parent;
  for (BasicBlock* block : caller->cfg.GetRPOTraversal(entry_block)) {
    for (Instr& instr : *block) {
      JIT_CHECK(
          !instr.IsBeginInlinedFunction(),
          "there should be no BeginInlinedFunction in inlined functions");
      JIT_CHECK(
          !instr.IsEndInlinedFunction(),
          "there should be no EndInlinedFunction in inlined functions");
      FrameState* fs = nullptr;
      if (auto db = instr.asDeoptBase()) {
        fs = db->frameState();
      } else if (instr.IsSnapshot()) {
        auto snap = static_cast<Snapshot*>(&instr);
        fs = snap->frameState();
      }
      if (fs == nullptr || fs->parent == nullptr) {
        continue;
      }
      bool inserted = framestate_parent.emplace(fs, fs->parent).second;
      JIT_CHECK(inserted, "there should not be duplicate FrameState pointers");
      fs->parent = nullptr;
    }
  }

  // The caller function has already been converted to SSA form and all HIR
  // passes require input to be in SSA form. SSAify the inlined function.
  SSAify{}.Run(*caller, entry_block);

  // Re-link the CFG.
  for (auto& [fs, parent] : framestate_parent) {
    fs->parent = parent;
  }

  return {entry_block, exit_block};
}

void HIRBuilder::advancePastYieldInstr(TranslationContext& tc) {
  // A YIELD_VALUE/RETURN_GENERATOR doesn't directly fail, however we may want
  // to throw into the generator which means we'd deopt. In this case we need
  // bytecode pointer to the following instruction which is where the
  // interpreter should pick-up execution.
  BCOffset next_bc_offs{
      BytecodeInstruction{code_, tc.frame.cur_instr_offs}.nextInstrOffset()};
  tc.frame.cur_instr_offs = next_bc_offs;
  JIT_DCHECK(
      next_bc_offs.asIndex().value() < countIndices(code_),
      "Yield should not be end of instruction stream");
}

void HIRBuilder::translate(
    Function& irfunc,
    const jit::BytecodeInstructionBlock& bc_instrs,
    const TranslationContext& initial_tc) {
  current_func_ = &irfunc;
  std::deque<TranslationContext> queue = {initial_tc};
  std::unordered_set<BasicBlock*> processed;
  std::unordered_set<BasicBlock*> loop_headers;

  while (!queue.empty()) {
    auto tc = std::move(queue.front());
    queue.pop_front();
    if (processed.contains(tc.block)) {
      continue;
    }
    processed.emplace(tc.block);

    // Translate remaining instructions into HIR
    auto& bc_block = map_get(block_map_.bc_blocks, tc.block);

    // Safety: skip unreachable END_FOR blocks. With _PyOpcode_Deopt in
    // opcode(), getJumpTarget() correctly skips past END_FOR after
    // FOR_ITER. createBlocks() still creates a block boundary at END_FOR
    // (from JUMP_BACKWARD fall-through), but no CFG edge leads to it.
    // If it somehow enters the queue, skip it entirely to avoid stack
    // assertions on subsequent instructions (STORE_FAST etc.).
    {
      auto first_it = bc_block.begin();
      if (first_it != bc_block.end() && (*first_it).opcode() == END_FOR) {
        continue;
      }
    }

    auto is_in_async_for_header_block = [&tc, &bc_instrs]() {
      if (tc.frame.block_stack.isEmpty()) {
        return false;
      }
      const ExecutionBlock& block_top = tc.frame.block_stack.top();
      return block_top.isAsyncForHeaderBlock(bc_instrs);
    };

    BytecodeInstruction prev_bc_instr{code_, BCOffset{-2}};
    for (auto bc_it = bc_block.begin(); bc_it != bc_block.end(); ++bc_it) {
      BytecodeInstruction bc_instr = *bc_it;

      tc.frame.cur_instr_offs = bc_instr.baseOffset();
      Instr* prev_hir_instr = tc.block->GetTerminator();
      // Outputting too many snapshots is safe but noisy so try to cull.
      // Note in some cases we'll have a non-empty block without yet having
      // translated any bytecodes. For example, if this is the first block and
      // there were prologue HIR instructions.
      if (
          // A completely empty block always gets a snapshot.
          prev_hir_instr == nullptr ||
          (
              // If we already have HIR instructions but haven't processed a
              // bytecode yet then conservatively emit a Snapshot.
              (prev_bc_instr.baseOffset() < 0 ||
               // Only emit a Snapshot after bytecode instructions which might
               // change the frame state.
               should_snapshot(
                   prev_bc_instr, is_in_async_for_header_block())))) {
        if (prev_hir_instr && prev_hir_instr->IsSnapshot()) {
          auto snapshot = static_cast<Snapshot*>(prev_hir_instr);
          snapshot->setFrameState(tc.frame);
        } else {
          tc.emitSnapshot();
        }
      }
      prev_bc_instr = bc_instr;

      // Translate instruction
      auto opcode = bc_instr.opcode();
      switch (opcode) {
        case NOP:
        case NOT_TAKEN: {
          break;
        }
        case PUSH_NULL: {
          emitPushNull(tc);
          break;
        }
        case BINARY_ADD:
        case BINARY_AND:
        case BINARY_FLOOR_DIVIDE:
        case BINARY_LSHIFT:
        case BINARY_MATRIX_MULTIPLY:
        case BINARY_MODULO:
        case BINARY_MULTIPLY:
        case BINARY_OP:
        case BINARY_OR:
        case BINARY_POWER:
        case BINARY_RSHIFT:
        case BINARY_SUBSCR:
        case BINARY_SUBTRACT:
        case BINARY_TRUE_DIVIDE:
        case BINARY_XOR: {
          emitBinaryOp(irfunc.cfg, tc, bc_instr);
          break;
        }
        case INPLACE_ADD:
        case INPLACE_AND:
        case INPLACE_FLOOR_DIVIDE:
        case INPLACE_LSHIFT:
        case INPLACE_MATRIX_MULTIPLY:
        case INPLACE_MODULO:
        case INPLACE_MULTIPLY:
        case INPLACE_OR:
        case INPLACE_POWER:
        case INPLACE_RSHIFT:
        case INPLACE_SUBTRACT:
        case INPLACE_TRUE_DIVIDE:
        case INPLACE_XOR: {
          emitInPlaceOp(tc, bc_instr);
          break;
        }
        case UNARY_NOT:
#if PY_VERSION_HEX >= 0x030E0000
          emitUnaryNot(tc);
          break;
#endif
        case UNARY_NEGATIVE:
        case UNARY_POSITIVE:
        case UNARY_INVERT: {
          emitUnaryOp(tc, bc_instr);
          break;
        }
        case BUILD_LIST:
        case BUILD_TUPLE:
          emitMakeListTuple(tc, bc_instr);
          break;
        case BUILD_CHECKED_LIST: {
          emitBuildCheckedList(tc, bc_instr);
          break;
        }
        case BUILD_CHECKED_MAP: {
          emitBuildCheckedMap(tc, bc_instr);
          break;
        }
        case BUILD_MAP: {
          emitBuildMap(tc, bc_instr);
          break;
        }
        case BUILD_SET: {
          emitBuildSet(tc, bc_instr);
          break;
        }
        case BUILD_CONST_KEY_MAP: {
          emitBuildConstKeyMap(tc, bc_instr);
          break;
        }
        case CALL:
        case CALL_FUNCTION:
        case CALL_FUNCTION_EX:
        case CALL_FUNCTION_KW:
        case CALL_KW:
        case CALL_METHOD:
        case INVOKE_FUNCTION:
        case INVOKE_METHOD:
        case INVOKE_NATIVE: {
          emitAnyCall(irfunc.cfg, tc, bc_it, bc_instrs);
          break;
        }
        case CALL_INTRINSIC_1:
        case CALL_INTRINSIC_2: {
          emitCallInstrinsic(tc, bc_instr);
          break;
        }
        case RESUME: {
          emitResume(irfunc.cfg, tc, bc_instr);
          break;
        }
        case KW_NAMES: {
          emitKwNames(tc, bc_instr);
          break;
        }
        case MAKE_CELL: {
          emitMakeCell(tc, bc_instr.oparg());
          break;
        }
        case COPY: {
          emitCopy(tc, bc_instr.oparg());
          break;
        }
        case COPY_FREE_VARS: {
          emitCopyFreeVars(tc, bc_instr.oparg());
          break;
        }
        case SWAP: {
          emitSwap(tc, bc_instr.oparg());
          break;
        }
        case IS_OP: {
          emitIsOp(tc, bc_instr.oparg());
          break;
        }
        case CONTAINS_OP: {
          emitContainsOp(tc, bc_instr.oparg());
          break;
        }
        case COMPARE_OP: {
          emitCompareOp(tc, bc_instr);
          break;
        }
        case TO_BOOL: {
          emitToBool(tc);
          break;
        }
        case COPY_DICT_WITHOUT_KEYS: {
          emitCopyDictWithoutKeys(tc);
          break;
        }
        case GET_LEN: {
          emitGetLen(tc);
          break;
        }
        case DELETE_ATTR: {
          emitDeleteAttr(tc, bc_instr);
          break;
        }
        case LOAD_ATTR: {
          emitLoadAttr(tc, bc_instr);
          break;
        }
        case LOAD_METHOD: {
          emitLoadMethod(tc, bc_instr.oparg());
          break;
        }
        case LOAD_METHOD_STATIC: {
          emitLoadMethodStatic(tc, bc_instr);
          break;
        }
        case LOAD_METHOD_SUPER: {
          emitLoadMethodOrAttrSuper(irfunc.cfg, tc, bc_instr, true);
          break;
        }
        case LOAD_ASSERTION_ERROR: {
          emitLoadAssertionError(tc, irfunc.env);
          break;
        }
        case LOAD_ATTR_SUPER:
        case LOAD_SUPER_ATTR: {
          emitLoadMethodOrAttrSuper(irfunc.cfg, tc, bc_instr, false);
          break;
        }
        case LOAD_CLOSURE: {
          // <3.11, the oparg was the cell index.  >=3.11 it's the same index as
          // any other local / frame value.
          int idx = bc_instr.oparg();
          if constexpr (PY_VERSION_HEX < 0x030B0000) {
            idx += tc.frame.nlocals;
          }
          tc.frame.stack.push(tc.frame.localsplus[idx]);
          break;
        }
        case LOAD_DEREF: {
          emitLoadDeref(tc, bc_instr);
          break;
        }
        case STORE_DEREF: {
          emitStoreDeref(tc, bc_instr);
          break;
        }
        case LOAD_CLASS: {
          emitLoadClass(tc, bc_instr);
          break;
        }
        case LOAD_CONST: {
          emitLoadConst(tc, bc_instr);
          break;
        }
        case LOAD_FAST:
        case LOAD_FAST_AND_CLEAR:
        case LOAD_FAST_CHECK:
        case LOAD_FAST_BORROW: {
          emitLoadFast(tc, bc_instr);
          break;
        }
        case LOAD_FAST_LOAD_FAST:
        case LOAD_FAST_BORROW_LOAD_FAST_BORROW: {
          emitLoadFastLoadFast(tc, bc_instr);
          break;
        }
        case LOAD_LOCAL: {
          emitLoadLocal(tc, bc_instr);
          break;
        }
        case LOAD_SMALL_INT: {
          emitLoadSmallInt(tc, bc_instr);
          break;
        }
        case LOAD_SPECIAL: {
          emitLoadSpecial(tc, bc_instr);
          break;
        }
        case LOAD_TYPE: {
          emitLoadType(tc, bc_instr);
          break;
        }
        case CONVERT_PRIMITIVE: {
          emitConvertPrimitive(tc, bc_instr);
          break;
        }
        case PRIMITIVE_LOAD_CONST: {
          emitPrimitiveLoadConst(tc, bc_instr);
          break;
        }
        case PRIMITIVE_BOX: {
          emitPrimitiveBox(tc, bc_instr);
          break;
        }
        case PRIMITIVE_UNBOX: {
          emitPrimitiveUnbox(tc, bc_instr);
          break;
        }
        case PRIMITIVE_BINARY_OP: {
          emitPrimitiveBinaryOp(tc, bc_instr);
          break;
        }
        case PRIMITIVE_COMPARE_OP: {
          emitPrimitiveCompare(tc, bc_instr);
          break;
        }
        case PRIMITIVE_UNARY_OP: {
          emitPrimitiveUnaryOp(tc, bc_instr);
          break;
        }
        case FAST_LEN: {
          emitFastLen(irfunc.cfg, tc, bc_instr);
          break;
        }
        case REFINE_TYPE: {
          emitRefineType(tc, bc_instr);
          break;
        }
        case SEQUENCE_GET: {
          emitSequenceGet(tc, bc_instr);
          break;
        }
        case SEQUENCE_SET: {
          emitSequenceSet(tc, bc_instr);
          break;
        }
        case LOAD_GLOBAL: {
          emitLoadGlobal(tc, bc_instr);
          break;
        }
        case JUMP_ABSOLUTE:
        case JUMP_BACKWARD: {
          BCOffset target_off = bc_instr.getJumpTarget();
          BasicBlock* target = getBlockAtOff(target_off);
          if (target_off <= bc_instr.baseOffset() || opcode != JUMP_ABSOLUTE) {
            loop_headers.emplace(target);
          }
          tc.emitBranch(target);
          break;
        }
        case JUMP_BACKWARD_NO_INTERRUPT:
        case JUMP_FORWARD: {
          BCOffset target_off = bc_instr.getJumpTarget();
          BasicBlock* target = getBlockAtOff(target_off);
          tc.emitBranch(target);
          break;
        }
        case JUMP_IF_FALSE_OR_POP:
        case JUMP_IF_NONZERO_OR_POP:
        case JUMP_IF_TRUE_OR_POP:
        case JUMP_IF_ZERO_OR_POP: {
          emitJumpIf(tc, bc_instr);
          break;
        }
        case POP_BLOCK: {
          popBlock(irfunc.cfg, tc);
          break;
        }
        case POP_JUMP_IF_FALSE:
        case POP_JUMP_IF_TRUE: {
          BCOffset target_off = bc_instr.getJumpTarget();
          BasicBlock* target = getBlockAtOff(target_off);
          if (target_off <= bc_instr.baseOffset()) {
            loop_headers.emplace(target);
          }
          emitPopJumpIf(tc, bc_instr);
          break;
        }
        case POP_JUMP_IF_NONE:
        case POP_JUMP_IF_NOT_NONE: {
          BCOffset target_off = bc_instr.getJumpTarget();
          BasicBlock* target = getBlockAtOff(target_off);
          if (target_off <= bc_instr.baseOffset()) {
            loop_headers.emplace(target);
          }
          emitPopJumpIfNone(tc, bc_instr);
          break;
        }
        case POP_ITER:
          if constexpr (PY_VERSION_HEX >= 0x030F0000) {
            tc.frame.stack.pop();
          }
          tc.frame.stack.pop();
          break;
        case POP_EXCEPT: {
          // B2: no-op — we never pushed exc_info in the JIT.
          break;
        }
        case POP_TOP: {
          tc.frame.stack.pop();
          break;
        }
        case RETURN_CONST: {
          Register* reg = temps_.AllocateStack();
          JIT_CHECK(
              bc_instr.oparg() < PyTuple_Size(code_->co_consts),
              "RETURN_CONST index out of bounds");
          Type type = Type::fromObject(
              PyTuple_GET_ITEM(code_->co_consts, bc_instr.oparg()));
          tc.emitLoadConst(reg, type);
          if (jit_get_config()->refine_static_python && type < TObject) {
            tc.emitRefineType(reg, type, reg);
          }
          tc.emitReturn(reg, type);
          break;
        }
        case RETURN_PRIMITIVE: {
          Type type = prim_type_to_type(bc_instr.oparg());
          JIT_CHECK(
              type <= preloader_.returnType(),
              "bad return type {}, expected {}",
              type,
              preloader_.returnType());
          Register* reg = tc.frame.stack.pop();
          tc.emitReturn(reg, type);
          break;
        }
        case RETURN_VALUE: {
          JIT_CHECK(
              tc.frame.block_stack.isEmpty(),
              "Returning with non-empty block stack");
          Register* reg = tc.frame.stack.pop();
          Type ret_type = preloader_.returnType();
          if (jit_get_config()->refine_static_python && ret_type < TObject) {
            tc.emitRefineType(reg, ret_type, reg);
          }
          tc.emitReturn(reg, ret_type);
          break;
        }
        case ROT_N: {
          int oparg = bc_instr.oparg();
          if (oparg <= 1) {
            break;
          }
          OperandStack& stack = tc.frame.stack;
          Register* top = stack.top();

          std::copy_backward(stack.end() - oparg, stack.end() - 1, stack.end());
          stack.topPut(oparg - 1, top);
          break;
        }
        case END_ASYNC_FOR: {
          emitEndAsyncFor(tc);
          break;
        }
        case END_FOR: {
          // END_FOR is unreachable: getJumpTarget() skips past it after
          // FOR_ITER. With _PyOpcode_Deopt in opcode(), specialised
          // FOR_ITER variants (FOR_ITER_LIST etc.) are correctly mapped
          // back to FOR_ITER, so getJumpTarget() always fires.
          // Safety no-op if the block is somehow reached.
          break;
        }
        case SETUP_FINALLY: {
          emitSetupFinally(tc, bc_instr);
          break;
        }
        case STORE_ATTR: {
          emitStoreAttr(tc, bc_instr);
          break;
        }
        case STORE_FAST: {
          emitStoreFast(tc, bc_instr);
          break;
        }
        case STORE_FAST_STORE_FAST: {
          emitStoreFastStoreFast(tc, bc_instr);
          break;
        }
        case STORE_FAST_LOAD_FAST: {
          emitStoreFastLoadFast(tc, bc_instr);
          break;
        }
        case STORE_LOCAL: {
          emitStoreLocal(tc, bc_instr);
          break;
        }
        case BINARY_SLICE: {
          emitBinarySlice(tc);
          break;
        }
        case STORE_SLICE: {
          emitStoreSlice(tc);
          break;
        }
        case STORE_SUBSCR: {
          emitStoreSubscr(tc, bc_instr);
          break;
        }
        case BUILD_SLICE: {
          emitBuildSlice(tc, bc_instr);
          break;
        }
        case GET_AITER: {
          emitGetAIter(tc);
          break;
        }
        case GET_ANEXT: {
          emitGetANext(tc);
          break;
        }
        case GET_ITER: {
          emitGetIter(tc, bc_instr);
          break;
        }
        case GET_YIELD_FROM_ITER: {
          emitGetYieldFromIter(irfunc.cfg, tc);
          break;
        }
        case MAKE_FUNCTION: {
          emitMakeFunction(tc, bc_instr);
          break;
        }
        case LIST_APPEND: {
          emitListAppend(tc, bc_instr);
          break;
        }
        case LIST_EXTEND: {
          emitListExtend(tc, bc_instr);
          break;
        }
        case LIST_TO_TUPLE: {
          emitListToTuple(tc);
          break;
        }
        case LOAD_ITERABLE_ARG: {
          emitLoadIterableArg(irfunc.cfg, tc, bc_instr);
          break;
        }
        case DUP_TOP: {
          auto& stack = tc.frame.stack;
          stack.push(stack.top());
          break;
        }
        case DUP_TOP_TWO: {
          auto& stack = tc.frame.stack;
          Register* top = stack.top();
          Register* snd = stack.top(1);
          stack.push(snd);
          stack.push(top);
          break;
        }
        case ROT_TWO: {
          auto& stack = tc.frame.stack;
          Register* top = stack.pop();
          Register* snd = stack.pop();
          stack.push(top);
          stack.push(snd);
          break;
        }
        case ROT_THREE: {
          auto& stack = tc.frame.stack;
          Register* top = stack.pop();
          Register* snd = stack.pop();
          Register* thd = stack.pop();
          stack.push(top);
          stack.push(thd);
          stack.push(snd);
          break;
        }
        case ROT_FOUR: {
          auto& stack = tc.frame.stack;
          Register* r1 = stack.pop();
          Register* r2 = stack.pop();
          Register* r3 = stack.pop();
          Register* r4 = stack.pop();
          stack.push(r1);
          stack.push(r4);
          stack.push(r3);
          stack.push(r2);
          break;
        }
        case FOR_ITER: {
          emitForIter(tc, bc_instr);
          break;
        }
        case LOAD_FIELD: {
          emitLoadField(tc, bc_instr);
          break;
        }
        case CAST: {
          emitCast(tc, bc_instr);
          break;
        }
        case TP_ALLOC: {
          emitTpAlloc(tc, bc_instr);
          break;
        }
        case STORE_FIELD: {
          emitStoreField(tc, bc_instr);
          break;
        }
        case POP_JUMP_IF_ZERO:
        case POP_JUMP_IF_NONZERO: {
          emitPopJumpIf(tc, bc_instr);
          break;
        }
        case IMPORT_FROM: {
          emitImportFrom(tc, bc_instr);
          break;
        }
        case EAGER_IMPORT_NAME:
        case IMPORT_NAME: {
          emitImportName(tc, bc_instr);
          break;
        }
        case RAISE_VARARGS: {
          emitRaiseVarargs(tc);
          break;
        }
        case YIELD_VALUE: {
          emitYieldValue(tc, bc_instr);
          break;
        }
        case YIELD_FROM: {
          if (is_in_async_for_header_block()) {
            emitAsyncForHeaderYieldFrom(tc, bc_instr);
          } else {
            emitYieldFrom(tc, temps_.AllocateStack());
          }
          break;
        }
        case GET_AWAITABLE: {
          emitGetAwaitable(irfunc.cfg, tc, bc_instrs, bc_instr);
          break;
        }
        case BUILD_STRING: {
          emitBuildString(tc, bc_instr);
          break;
        }
        case FORMAT_VALUE: {
          emitFormatValue(tc, bc_instr);
          break;
        }
        case FORMAT_WITH_SPEC: {
          emitFormatWithSpec(tc);
          break;
        }
        case MAP_ADD: {
          emitMapAdd(tc, bc_instr);
          break;
        }
        case SET_ADD: {
          emitSetAdd(tc, bc_instr);
          break;
        }
        case SET_UPDATE: {
          emitSetUpdate(tc, bc_instr);
          break;
        }
        case UNPACK_EX: {
          emitUnpackEx(tc, bc_instr);
          break;
        }
        case UNPACK_SEQUENCE: {
          emitUnpackSequence(irfunc.cfg, tc, bc_instr);
          break;
        }
        case DELETE_SUBSCR: {
          Register* sub = tc.frame.stack.pop();
          Register* container = tc.frame.stack.pop();
          tc.emitDeleteSubscr(container, sub, tc.frame);
          break;
        }
        case DELETE_FAST: {
          int var_idx = bc_instr.oparg();
          Register* var = tc.frame.localsplus[var_idx];
          moveOverwrittenStackRegisters(tc, var);
          tc.emitLoadConst(var, TNullptr);
          break;
        }
        case BEFORE_ASYNC_WITH:
        case BEFORE_WITH: {
          emitBeforeWith(tc, bc_instr);
          break;
        }
        case SETUP_ASYNC_WITH: {
          emitSetupAsyncWith(tc, bc_instr);
          break;
        }
        case SETUP_WITH: {
          emitSetupWith(tc, bc_instr);
          break;
        }
        case MATCH_CLASS: {
          emitMatchClass(irfunc.cfg, tc, bc_instr);
          break;
        }
        case MATCH_KEYS: {
          emitMatchKeys(irfunc.cfg, tc);
          break;
        }
        case MATCH_MAPPING: {
          emitMatchMappingSequence(irfunc.cfg, tc, Py_TPFLAGS_MAPPING);
          break;
        }
        case MATCH_SEQUENCE: {
          emitMatchMappingSequence(irfunc.cfg, tc, Py_TPFLAGS_SEQUENCE);
          break;
        }
        case GEN_START: {
          // In the interpreter this instruction behaves like POP_TOP because it
          // assumes a generator will always be sent a superfluous None value to
          // start execution via the stack. We skip doing this for JIT
          // functions. This should be fine as long as we can't de-opt after the
          // function is started but before GEN_START. This check ensures this.
          JIT_DCHECK(
              bc_instr.baseIndex() == 0, "GEN_START must be first instruction");
          break;
        }
        case DICT_UPDATE: {
          emitDictUpdate(tc, bc_instr);
          break;
        }
        case DICT_MERGE: {
          emitDictMerge(tc, bc_instr);
          break;
        }
        case RETURN_GENERATOR: {
          auto out = temps_.AllocateStack();
          if constexpr (
              PY_VERSION_HEX < 0x030C0000 || PY_VERSION_HEX >= 0x030E0000) {
            advancePastYieldInstr(tc);
          }
          tc.emitInitialYield(out, tc.frame);
          tc.frame.stack.push(out);
          break;
        }
        case SEND: {
          emitSend(tc, bc_instr);
          break;
        }
        case END_SEND: {
          // Pop the value and iterator off the stack and then push back the
          // value.
          Register* value = tc.frame.stack.pop();
          tc.frame.stack.pop();
          tc.frame.stack.push(value);
          break;
        }
        case BUILD_INTERPOLATION: {
          emitBuildInterpolation(tc, bc_instr);
          break;
        }
        case BUILD_TEMPLATE: {
          emitBuildTemplate(tc);
          break;
        }
        case CONVERT_VALUE: {
          emitConvertValue(tc, bc_instr);
          break;
        }
        case FORMAT_SIMPLE: {
          emitFormatSimple(irfunc.cfg, tc);
          break;
        }
        case LOAD_COMMON_CONSTANT: {
          emitLoadCommonConstant(tc, bc_instr);
          break;
        }
        case SET_FUNCTION_ATTRIBUTE: {
          emitSetFunctionAttribute(tc, bc_instr);
          break;
        }
        case LOAD_BUILD_CLASS: {
          emitLoadBuildClass(tc);
          break;
        }
        case STORE_GLOBAL: {
          emitStoreGlobal(tc, bc_instr);
          break;
        }
        case CHECK_EG_MATCH:
        case CHECK_EXC_MATCH:
        case CLEANUP_THROW:
        case PUSH_EXC_INFO:
          // Graceful fallback: these opcodes appear in with-statement
          // try/finally blocks that the builder does not handle.
          // Throwing allows compilePreloaderImpl to catch and fall back
          // to the interpreter instead of aborting the process.
          throw std::runtime_error(fmt::format(
              "Cannot compile: opcode {} ({}) in non-handler context",
              opcode,
              opcodeName(opcode)));
        default: {
          throw std::runtime_error(fmt::format("Cannot compile: unhandled opcode {} ({})", opcode, opcodeName(opcode)));
        }
      }
    }
    // Insert jumps for blocks that fall through.
    auto last_instr = tc.block->GetTerminator();
    if ((last_instr == nullptr) || !last_instr->IsTerminator()) {
      auto off = bc_block.endOffset();
      last_instr = tc.emitBranch(getBlockAtOff(off));
    }

    // Make sure any values left on the stack are in the registers that we
    // expect
    BlockCanonicalizer bc;
    bc.Run(tc.block, temps_, tc.frame.stack);

    // Add successors to be processed
    //
    // These bytecodes alter the operand stack along one branch and leave it
    // untouched along the other. Thus, they must be special cased.
    switch (prev_bc_instr.opcode()) {
      case FOR_ITER: {
        auto condbr = static_cast<CondBranchIterNotDone*>(last_instr);
        auto new_frame = tc.frame;
        if constexpr (PY_VERSION_HEX >= 0x030E0000) {
          // Just pop the sentinel value. The target POP_ITER will pop the
          // iterator.
          new_frame.stack.discard(1);
        } else {
          // Pop both the sentinel value signaling iteration is complete
          // and the iterator itself.
          new_frame.stack.discard(2);
        }
        queue.emplace_back(condbr->true_bb(), tc.frame);
        queue.emplace_back(condbr->false_bb(), new_frame);
        break;
      }
      case JUMP_IF_FALSE_OR_POP:
      case JUMP_IF_ZERO_OR_POP: {
        auto condbr = static_cast<CondBranch*>(last_instr);
        auto new_frame = tc.frame;
        new_frame.stack.pop();
        queue.emplace_back(condbr->true_bb(), new_frame);
        queue.emplace_back(condbr->false_bb(), tc.frame);
        break;
      }
      case JUMP_IF_NONZERO_OR_POP:
      case JUMP_IF_TRUE_OR_POP: {
        auto condbr = static_cast<CondBranch*>(last_instr);
        auto new_frame = tc.frame;
        new_frame.stack.pop();
        queue.emplace_back(condbr->true_bb(), tc.frame);
        queue.emplace_back(condbr->false_bb(), new_frame);
        break;
      }
      default: {
        if (prev_bc_instr.opcode() == YIELD_FROM &&
            is_in_async_for_header_block()) {
          JIT_CHECK(
              last_instr->IsCondBranchIterNotDone(),
              "Async-for header should end with CondBranchIterNotDone");
          auto condbr = static_cast<CondBranchIterNotDone*>(last_instr);
          FrameState new_frame = tc.frame;
          // Pop sentinel value signaling that iteration is complete
          new_frame.stack.pop();
          queue.emplace_back(condbr->true_bb(), tc.frame);
          queue.emplace_back(condbr->false_bb(), std::move(new_frame));
          break;
        }
        for (std::size_t i = 0; i < last_instr->numEdges(); i++) {
          auto succ = last_instr->successor(i);
          queue.emplace_back(succ, tc.frame);
        }
        break;
      }
    }
    // B2: drain pending blocks from emitInlineExceptionMatch.
    for (auto& pb : pending_b2_blocks_) {
      queue.emplace_back(pb.block, std::move(pb.frame));
    }
    pending_b2_blocks_.clear();

    JIT_DCHECK(
        tc.block->GetTerminator() != nullptr &&
            !tc.block->GetTerminator()->IsSnapshot(),
        "opcodes should not end with a snapshot");
  }

  JIT_CHECK(
      kwnames_ == nullptr,
      "Stashed a KW_NAMES value for function {} but never consumed it",
      irfunc.fullname);

  for (auto block : loop_headers) {
    insertRunPeriodicActivitesForLoop(irfunc.cfg, block);
  }
}

void BlockCanonicalizer::InsertCopies(
    Register* reg,
    TempAllocator& temps,
    Instr& terminator,
    std::vector<Register*>& alloced) {
  if (done_.contains(reg)) {
    return;
  } else if (processing_.contains(reg)) {
    // We've detected a cycle. Move the register to a new home
    // in order to break the cycle.
    auto tmp = temps.AllocateStack();
    auto mov = Assign::create(tmp, reg);
    mov->copyBytecodeOffset(terminator);
    mov->InsertBefore(terminator);
    moved_[reg] = tmp;
    alloced.emplace_back(tmp);
    return;
  }

  auto orig_reg = reg;
  for (auto dst : copies_[reg]) {
    auto it = copies_.find(dst);
    if (it != copies_.end()) {
      // The destination also needs to be moved. So deal with it first.
      processing_.insert(reg);
      InsertCopies(dst, temps, terminator, alloced);
      processing_.erase(reg);
      // It's possible that the register we were processing was moved
      // because it participated in a cycle
      auto it2 = moved_.find(reg);
      if (it2 != moved_.end()) {
        reg = it2->second;
      }
    }
    auto mov = Assign::create(dst, reg);
    mov->copyBytecodeOffset(terminator);
    mov->InsertBefore(terminator);
  }

  done_.insert(orig_reg);
}

void BlockCanonicalizer::Run(
    BasicBlock* block,
    TempAllocator& temps,
    OperandStack& stack) {
  if (stack.isEmpty()) {
    return;
  }

  processing_.clear();
  copies_.clear();
  moved_.clear();

  // Compute the desired stack layout
  std::vector<Register*> dsts;
  dsts.reserve(stack.size());
  for (std::size_t i = 0; i < stack.size(); i++) {
    auto reg = temps.GetOrAllocateStack(i);
    dsts.emplace_back(reg);
  }

  // Compute the minimum number of copies that need to happen
  std::vector<Register*> need_copy;
  auto term = block->GetTerminator();
  std::vector<Register*> alloced;
  for (std::size_t i = 0; i < stack.size(); i++) {
    auto src = stack.at(i);
    auto dst = dsts[i];
    if (src != dst) {
      need_copy.emplace_back(src);
      copies_[src].emplace_back(dst);

      if (term->Uses(src)) {
        term->ReplaceUsesOf(src, dst);
      } else if (term->Uses(dst)) {
        auto tmp = temps.AllocateStack();
        alloced.emplace_back(tmp);
        auto mov = Assign::create(tmp, dst);
        mov->InsertBefore(*term);
        term->ReplaceUsesOf(dst, tmp);
      }
    }
  }
  if (need_copy.empty()) {
    return;
  }

  for (auto reg : need_copy) {
    InsertCopies(reg, temps, *term, alloced);
  }

  // Put the stack in canonical form
  for (std::size_t i = 0; i < stack.size(); i++) {
    stack.atPut(i, dsts[i]);
  }
}

static std::optional<BinaryOpKind> getBinaryOpKindFromOpcode(int opcode) {
  switch (opcode) {
    case BINARY_ADD:
      return BinaryOpKind::kAdd;
    case BINARY_AND:
      return BinaryOpKind::kAnd;
    case BINARY_FLOOR_DIVIDE:
      return BinaryOpKind::kFloorDivide;
    case BINARY_LSHIFT:
      return BinaryOpKind::kLShift;
    case BINARY_MATRIX_MULTIPLY:
      return BinaryOpKind::kMatrixMultiply;
    case BINARY_MODULO:
      return BinaryOpKind::kModulo;
    case BINARY_MULTIPLY:
      return BinaryOpKind::kMultiply;
    case BINARY_OR:
      return BinaryOpKind::kOr;
    case BINARY_POWER:
      return BinaryOpKind::kPower;
    case BINARY_RSHIFT:
      return BinaryOpKind::kRShift;
    case BINARY_SUBSCR:
      return BinaryOpKind::kSubscript;
    case BINARY_SUBTRACT:
      return BinaryOpKind::kSubtract;
    case BINARY_TRUE_DIVIDE:
      return BinaryOpKind::kTrueDivide;
    case BINARY_XOR:
      return BinaryOpKind::kXor;
    default:
      return std::nullopt;
  }
}

static std::optional<BinaryOpKind> getBinaryOpKindFromOparg(int oparg) {
  switch (oparg) {
    case NB_ADD:
      return BinaryOpKind::kAdd;
    case NB_AND:
      return BinaryOpKind::kAnd;
    case NB_FLOOR_DIVIDE:
      return BinaryOpKind::kFloorDivide;
    case NB_LSHIFT:
      return BinaryOpKind::kLShift;
    case NB_MATRIX_MULTIPLY:
      return BinaryOpKind::kMatrixMultiply;
    case NB_MULTIPLY:
      return BinaryOpKind::kMultiply;
    case NB_REMAINDER:
      return BinaryOpKind::kModulo;
    case NB_OR:
      return BinaryOpKind::kOr;
    case NB_POWER:
      return BinaryOpKind::kPower;
    case NB_RSHIFT:
      return BinaryOpKind::kRShift;
    case NB_SUBTRACT:
      return BinaryOpKind::kSubtract;
    case NB_TRUE_DIVIDE:
      return BinaryOpKind::kTrueDivide;
    case NB_XOR:
      return BinaryOpKind::kXor;
#if PY_VERSION_HEX >= 0x030E0000
    case NB_SUBSCR:
      return BinaryOpKind::kSubscript;
#endif
    default:
      return std::nullopt;
  }
}

static std::optional<InPlaceOpKind> getInPlaceOpKindFromOpcode(int opcode) {
  switch (opcode) {
    case INPLACE_ADD:
      return InPlaceOpKind::kAdd;
    case INPLACE_AND:
      return InPlaceOpKind::kAnd;
    case INPLACE_FLOOR_DIVIDE:
      return InPlaceOpKind::kFloorDivide;
    case INPLACE_LSHIFT:
      return InPlaceOpKind::kLShift;
    case INPLACE_MATRIX_MULTIPLY:
      return InPlaceOpKind::kMatrixMultiply;
    case INPLACE_MODULO:
      return InPlaceOpKind::kModulo;
    case INPLACE_MULTIPLY:
      return InPlaceOpKind::kMultiply;
    case INPLACE_OR:
      return InPlaceOpKind::kOr;
    case INPLACE_POWER:
      return InPlaceOpKind::kPower;
    case INPLACE_RSHIFT:
      return InPlaceOpKind::kRShift;
    case INPLACE_SUBTRACT:
      return InPlaceOpKind::kSubtract;
    case INPLACE_TRUE_DIVIDE:
      return InPlaceOpKind::kTrueDivide;
    case INPLACE_XOR:
      return InPlaceOpKind::kXor;
    default:
      return std::nullopt;
  }
}

static std::optional<InPlaceOpKind> getInPlaceOpKindFromOparg(int oparg) {
  switch (oparg) {
    case NB_INPLACE_ADD:
      return InPlaceOpKind::kAdd;
    case NB_INPLACE_AND:
      return InPlaceOpKind::kAnd;
    case NB_INPLACE_FLOOR_DIVIDE:
      return InPlaceOpKind::kFloorDivide;
    case NB_INPLACE_LSHIFT:
      return InPlaceOpKind::kLShift;
    case NB_INPLACE_MATRIX_MULTIPLY:
      return InPlaceOpKind::kMatrixMultiply;
    case NB_INPLACE_MULTIPLY:
      return InPlaceOpKind::kMultiply;
    case NB_INPLACE_REMAINDER:
      return InPlaceOpKind::kModulo;
    case NB_INPLACE_OR:
      return InPlaceOpKind::kOr;
    case NB_INPLACE_POWER:
      return InPlaceOpKind::kPower;
    case NB_INPLACE_RSHIFT:
      return InPlaceOpKind::kRShift;
    case NB_INPLACE_SUBTRACT:
      return InPlaceOpKind::kSubtract;
    case NB_INPLACE_TRUE_DIVIDE:
      return InPlaceOpKind::kTrueDivide;
    case NB_INPLACE_XOR:
      return InPlaceOpKind::kXor;
    default:
      return std::nullopt;
  }
}

void HIRBuilder::emitPushNull(TranslationContext& tc) {
  auto& stack = tc.frame.stack;
  Register* tmp = temps_.AllocateStack();
  tc.emitLoadConst(tmp, TNullptr);
  stack.push(tmp);
}

void HIRBuilder::emitAnyCall(
    CFG& cfg,
    TranslationContext& tc,
    jit::BytecodeInstructionBlock::Iterator& bc_it,
    const jit::BytecodeInstructionBlock& bc_instrs) {
  BytecodeInstruction bc_instr = *bc_it;
  bool is_awaited;
  if constexpr (PY_VERSION_HEX >= 0x030C0000) {
    is_awaited = false;
  } else {
    is_awaited = code_->co_flags & CO_COROUTINE &&
        // We only need to be followed by GET_AWAITABLE to know we are awaited,
        // but we also need to ensure the following LOAD_CONST and YIELD_FROM
        // are inside this BytecodeInstructionBlock. This may not be the case if
        // the 'await' is shared as in 'await (x if y else z)'.
        bc_it.remainingIndices() >= 3 &&
        bc_instr.nextInstr().opcode() == GET_AWAITABLE;
  }
  auto flags = is_awaited ? CallFlags::Awaited : CallFlags::None;
  bool call_used_is_awaited = true;

  auto opcode = bc_instr.opcode();
  switch (opcode) {
    case CALL_FUNCTION:
    case CALL_FUNCTION_KW: {
      // Operands include the function arguments plus the function itself.
      auto num_operands = static_cast<std::size_t>(bc_instr.oparg()) + 1;
      // Add one more operand for the kwnames tuple at the end.
      if (opcode == CALL_FUNCTION_KW) {
        num_operands++;
        flags |= CallFlags::KwArgs;
      }
      tc.emitVariadic<VectorCall>(temps_, num_operands, flags);
      break;
    }
    case CALL_FUNCTION_EX: {
      emitCallEx(tc, bc_instr, flags);
      break;
    }
    case CALL:
    case CALL_KW:
    case CALL_METHOD: {
      auto num_operands = static_cast<std::size_t>(bc_instr.oparg()) + 2;
      auto num_stack_inputs = num_operands;
      bool is_call_kw = opcode == CALL_KW;
      if (kwnames_ != nullptr || is_call_kw) {
        if (is_call_kw) {
          num_stack_inputs++;
        }
        num_operands++;
        flags |= CallFlags::KwArgs;
      }

      // Manually set up the instruction instead of using emitVariadic.
      // kwnames_ isn't on the stack, but it has to be part of the operand
      // count.
      Register* out = temps_.AllocateStack();
      auto call = tc.emit<CallMethod>(num_operands, out, flags);
      for (auto i = num_stack_inputs; i > 0; i--) {
        Register* arg = tc.frame.stack.pop();
        call->SetOperand(i - 1, arg);
      }
      if (kwnames_ != nullptr) {
        JIT_CHECK(
            call->GetOperand(num_operands - 1) == nullptr,
            "Somehow already set the kwnames argument");
        call->SetOperand(num_operands - 1, kwnames_);
        kwnames_ = nullptr;
      }
      call->setFrameState(tc.frame);

      tc.frame.stack.push(out);

      // B2: If this CALL is inside a try block with a simple except pattern,
      // inline the exception handler instead of deopting on exception.
      // This prevents the deopt cascade that occurs when JIT-compiled
      // functions raise caught exceptions (e.g. StopIteration from next()
      // in contextlib._GeneratorContextManager.__exit__).
      {
        BCOffset cur_off = bc_instr.baseOffset();
        auto* handler = findExceptionHandler(cur_off);
        if (handler != nullptr) {
          SimpleExceptInfo info;
          if (getSimpleExceptInfo(*handler, info)) {
            emitCallExceptionHandler(
                cfg, tc, bc_instr, *handler, info, call, out);
          }
        }
      }
      break;
    }
    case INVOKE_FUNCTION: {
      call_used_is_awaited = emitInvokeFunction(tc, bc_instr, flags);
      break;
    }
    case INVOKE_NATIVE: {
      call_used_is_awaited = emitInvokeNative(tc, bc_instr);
      break;
    }
    case INVOKE_METHOD: {
      call_used_is_awaited = emitInvokeMethod(tc, bc_instr, is_awaited);
      break;
    }
    default:
      JIT_ABORT("Unhandled call opcode {} ({})", opcode, opcodeName(opcode));
  }
  if (is_awaited && call_used_is_awaited) {
    Register* out = temps_.AllocateStack();
    TranslationContext await_block{cfg.AllocateBlock(), tc.frame};
    TranslationContext post_await_block{cfg.AllocateBlock(), tc.frame};

    emitDispatchEagerCoroResult(
        cfg, tc, out, await_block.block, post_await_block.block);

    tc.block = await_block.block;

    ++bc_it;
    JIT_CHECK(
        bc_it->opcode() == GET_AWAITABLE,
        "Awaited function call must be followed by GET_AWAITABLE");
    emitGetAwaitable(cfg, tc, bc_instrs, *bc_it);

    ++bc_it;
    JIT_CHECK(
        bc_it->opcode() == LOAD_CONST,
        "GET_AWAITABLE must be followed by LOAD_CONST");
    emitLoadConst(tc, *bc_it);

    ++bc_it;
    JIT_CHECK(
        bc_it->opcode() == YIELD_FROM,
        "GET_AWAITABLE should always be followed by LOAD_CONST+YIELD_FROM");
    emitYieldFrom(tc, out);
    tc.emitBranch(post_await_block.block);

    tc.block = post_await_block.block;
  }
}

void HIRBuilder::emitCallInstrinsic(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  auto oparg = bc_instr.oparg();
  auto num_operands = 1;

  Register* value = tc.frame.stack.pop();
  Register* res = temps_.AllocateStack();
  std::vector<Register*> args;
#if PY_VERSION_HEX >= 0x030C0000
  if (bc_instr.opcode() == CALL_INTRINSIC_2) {
    JIT_CHECK(
        oparg <= MAX_INTRINSIC_2,
        "Invalid oparg for binary intrinsic function: {}",
        oparg);
    Register* value2 = tc.frame.stack.pop();
    args.push_back(value2);
    num_operands = 2;
  } else {
    JIT_CHECK(
        oparg <= MAX_INTRINSIC_1,
        "Invalid oparg for unary intrinsic function: {}",
        oparg);
  }
#endif
  args.push_back(value);
  tc.emit<CallIntrinsic>(num_operands, res, oparg, args);
  tc.frame.stack.push(res);
}

void HIRBuilder::emitResume(
    CFG& cfg,
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  if (bc_instr.oparg() >= 2) {
    return;
  }
  TranslationContext succ(cfg.AllocateBlock(), tc.frame);
  succ.emitSnapshot();
  insertRunPeriodicActivites(cfg, tc.block, succ.block, tc.frame);
  tc.block = succ.block;
}

void HIRBuilder::emitKwNames(
    TranslationContext& tc,
    const BytecodeInstruction& bc_instr) {
  auto index = bc_instr.oparg();
  auto consts_len = PyTuple_Size(code_->co_consts);
  JIT_CHECK(
      index < consts_len,
      "KW_NAMES index {} is greater than co_consts length {}",
      index,
      consts_len);
  JIT_CHECK(
      kwnames_ == nullptr,
      "Trying to save KW_NAMES({}) but previous kwnames_ value wasn't consumed "
      "by a CALL* opcode yet",
      index);

  kwnames_ = temps_.AllocateNonStack();
  tc.emitLoadConst(
      kwnames_, Type::fromObject(PyTuple_GET_ITEM(code_->co_consts, index)));
}

void HIRBuilder::emitBinaryOp(
    CFG& cfg,
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  auto& stack = tc.frame.stack;
  if (jit_get_config()->specialized_opcodes) {
    // Bug 7 fix: Snapshot BEFORE popping operands — deopt re-executes instruction
    tc.emitSnapshot();
  }
  Register* right = stack.pop();
  Register* left = stack.pop();
  Register* result = temps_.AllocateStack();

  int opcode = bc_instr.opcode();
  int oparg = bc_instr.oparg();

  if (jit_get_config()->specialized_opcodes) {
    switch (bc_instr.specializedOpcode()) {
      case BINARY_OP_ADD_INT:
      case BINARY_OP_MULTIPLY_INT:
      case BINARY_OP_SUBTRACT_INT:
        tc.emitGuardType(left, TLongExact, left);
        tc.emitGuardType(right, TLongExact, right);
        break;
      case BINARY_OP_ADD_FLOAT:
      case BINARY_OP_MULTIPLY_FLOAT:
      case BINARY_OP_SUBTRACT_FLOAT:
        tc.emitGuardType(left, TFloatExact, left);
        tc.emitGuardType(right, TFloatExact, right);
        break;
      case BINARY_OP_ADD_UNICODE:
        tc.emitGuardType(left, TUnicodeExact, left);
        tc.emitGuardType(right, TUnicodeExact, right);
        break;
      case BINARY_SUBSCR_DICT:
        tc.emitGuardType(left, TDictExact, left);
        break;
      case BINARY_SUBSCR_LIST_INT:
        tc.emitGuardType(left, TListExact, left);
        tc.emitGuardType(right, TLongExact, right);
        break;
      case BINARY_SUBSCR_TUPLE_INT:
        tc.emitGuardType(left, TTupleExact, left);
        tc.emitGuardType(right, TLongExact, right);
        break;
      default:
        break;
    }
  }

  BinaryOpKind op_kind;
  if (opcode == BINARY_OP) {
    auto opt_op_kind = getBinaryOpKindFromOparg(oparg);
    if (opt_op_kind) {
      op_kind = *opt_op_kind;
    } else {
      // BINARY_OP can also contain inplace opargs.
      auto inplace_opt_op_kind = getInPlaceOpKindFromOparg(oparg);
      JIT_CHECK(
          inplace_opt_op_kind.has_value(),
          "Unrecognized oparg for BINARY_OP: {}",
          oparg);
      InPlaceOpKind inplace_op_kind = *inplace_opt_op_kind;
      tc.emitInPlaceOp(result, inplace_op_kind, left, right, tc.frame);
      stack.push(result);
      return;
    }
  } else {
    auto opt_op_kind = getBinaryOpKindFromOpcode(opcode);
    JIT_CHECK(
        opt_op_kind.has_value(),
        "Unrecognized opcode {} ({}) for binary operation",
        opcode,
        opcodeName(opcode));
    op_kind = *opt_op_kind;
  }

  // B2: For subscript inside try block with simple except pattern,
  // emit inline exception match instead of BinaryOp (which auto-deopts).
  if (op_kind == BinaryOpKind::kSubscript) {
    BCOffset cur_off = bc_instr.baseOffset();
    auto* handler = findExceptionHandler(cur_off);
    if (handler != nullptr) {
      SimpleExceptInfo info;
      if (getSimpleExceptInfo(*handler, info)) {
        emitInlineExceptionMatch(
            cfg, tc, bc_instr, *handler, info,
            left, right, result);
        stack.push(result);
        return;
      }
    }
  }

  tc.emitBinaryOp(result, op_kind, left, right, tc.frame);
  stack.push(result);
}

void HIRBuilder::emitInPlaceOp(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  auto& stack = tc.frame.stack;
  Register* right = stack.pop();
  Register* left = stack.pop();
  Register* result = temps_.AllocateStack();
  int opcode = bc_instr.opcode();
  auto opt_op_kind = getInPlaceOpKindFromOpcode(opcode);
  JIT_CHECK(
      opt_op_kind.has_value(),
      "Unrecognized opcode {} ({}) for inplace operation",
      opcode,
      opcodeName(opcode));
  InPlaceOpKind op_kind = *opt_op_kind;
  tc.emitInPlaceOp(result, op_kind, left, right, tc.frame);
  stack.push(result);
}

static inline UnaryOpKind get_unary_op_kind(
    const jit::BytecodeInstruction& bc_instr) {
  auto opcode = bc_instr.opcode();
  switch (opcode) {
    case UNARY_NOT:
      return UnaryOpKind::kNot;

    case UNARY_NEGATIVE:
      return UnaryOpKind::kNegate;

    case UNARY_POSITIVE:
      return UnaryOpKind::kPositive;

    case UNARY_INVERT:
      return UnaryOpKind::kInvert;

    default:
      break;
  }
  JIT_ABORT("Unhandled unary op {} ({})", opcode, opcodeName(opcode));
}

void HIRBuilder::emitUnaryNot(TranslationContext& tc) {
  Register* operand = tc.frame.stack.pop();
  Register* is_false = temps_.AllocateNonStack();
  Register* const_false = temps_.AllocateNonStack();
  Register* result = temps_.AllocateStack();
  tc.emitLoadConst(const_false, Type::fromObject(Py_False));
  tc.emitPrimitiveCompare(
      is_false, PrimitiveCompareOp::kEqual, const_false, operand);
  tc.emitPrimitiveBoxBool(result, is_false);
  tc.frame.stack.push(result);
}

void HIRBuilder::emitUnaryOp(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  Register* operand = tc.frame.stack.pop();
  Register* result = temps_.AllocateStack();
  UnaryOpKind op_kind = get_unary_op_kind(bc_instr);
  tc.emitUnaryOp(result, op_kind, operand, tc.frame);
  tc.frame.stack.push(result);
}

void HIRBuilder::emitCallEx(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr,
    CallFlags flags) {
  Register* dst = temps_.AllocateStack();
  OperandStack& stack = tc.frame.stack;
  // In 3.14+ we always have kwargs on the stack but it may be null.
  bool has_kwargs = (PY_VERSION_HEX >= 0x030E0000) || bc_instr.oparg() & 0x1;
  Register* kwargs = nullptr;
  if (has_kwargs) {
    kwargs = stack.pop();
    flags |= CallFlags::KwArgs;
  } else {
    Register* nullp = temps_.AllocateNonStack();
    tc.emitLoadConst(nullp, TNullptr);
    kwargs = nullp;
  }
  Register* pargs = stack.pop();
  Register* func;
  // CALL_FUNCTION_EX has an unused value on the stack, starting with 3.12.
  // In 3.14 this swapped location.
  if constexpr (PY_VERSION_HEX >= 0x030E0000) {
    stack.pop();
    func = stack.pop();
  } else if constexpr (PY_VERSION_HEX >= 0x030C0000) {
    func = stack.pop();
    stack.pop();
  } else {
    func = stack.pop();
  }
  tc.emit<CallEx>(dst, func, pargs, kwargs, flags, tc.frame);
  stack.push(dst);
}

void HIRBuilder::emitBuildSlice(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  std::size_t num_operands = static_cast<std::size_t>(bc_instr.oparg());
  tc.emitVariadic<BuildSlice>(temps_, num_operands);
}

void HIRBuilder::emitListAppend(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  auto item = tc.frame.stack.pop();
  auto list = tc.frame.stack.peek(bc_instr.oparg());
  auto dst = temps_.AllocateStack();
  tc.emitListAppend(dst, list, item, tc.frame);
}

void HIRBuilder::emitLoadIterableArg(
    CFG& cfg,
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  auto iterable = tc.frame.stack.pop();
  Register* tuple;
  if (iterable->type() != TTupleExact) {
    TranslationContext tuple_path{cfg.AllocateBlock(), tc.frame};
    tuple_path.emitSnapshot();
    TranslationContext non_tuple_path{cfg.AllocateBlock(), tc.frame};
    non_tuple_path.emitSnapshot();
    tc.emitCondBranchCheckType(
        iterable, TTuple, tuple_path.block, non_tuple_path.block);
    tc.block = cfg.AllocateBlock();
    tc.emitSnapshot();

    tuple = temps_.AllocateStack();

    tuple_path.emitAssign(tuple, iterable);
    tuple_path.emitBranch(tc.block);

    non_tuple_path.emitGetTuple(tuple, iterable, tc.frame);
    non_tuple_path.emitBranch(tc.block);
  } else {
    tuple = iterable;
  }

  auto tmp = temps_.AllocateStack();
  auto tup_idx = temps_.AllocateStack();
  auto element = temps_.AllocateStack();
  tc.emitLoadConst(tmp, Type::fromCInt(bc_instr.oparg(), TCInt64));
  tc.emitPrimitiveBox(tup_idx, tmp, TCInt64, tc.frame);
  tc.emitBinaryOp(
      element, BinaryOpKind::kSubscript, tuple, tup_idx, tc.frame);
  tc.frame.stack.push(element);
  tc.frame.stack.push(tuple);
}

bool HIRBuilder::tryEmitDirectMethodCall(
    const InvokeTarget& target,
    TranslationContext& tc,
    long nargs) {
  if (target.is_statically_typed || nargs == target.builtin_expected_nargs) {
    Instr* staticCall;
    Register* out = nullptr;
    if (target.builtin_returns_void) {
      staticCall = tc.emit<CallStaticRetVoid>(nargs, target.builtin_c_func);
    } else {
      out = temps_.AllocateStack();
      Type ret_type =
          target.builtin_returns_error_code ? TCInt32 : target.return_type;
      staticCall =
          tc.emitCallStatic(nargs, out, target.builtin_c_func, ret_type);
    }

    auto& stack = tc.frame.stack;
    for (auto i = nargs - 1; i >= 0; i--) {
      Register* operand = stack.pop();
      staticCall->SetOperand(i, operand);
    }

    if (target.builtin_returns_error_code) {
      tc.emitCheckNeg(out, out, tc.frame);
    } else if (out != nullptr) {
      auto ret_ty = target.return_type;
      HirType h_ret = to_hir(ret_ty), h_prim = to_hir(TPrimitive);
      if (!hir_type_could_be(&h_ret, &h_prim)) {
        tc.emitCheckExc(out, out, tc.frame);
      }
    }
    if (target.builtin_returns_void || target.builtin_returns_error_code) {
      // We could update the compiler so that void returning functions either
      // are only used in void contexts, or explicitly emit a LOAD_CONST None
      // when not used in a void context. For now we just produce None here (and
      // in _PyClassLoader_ConvertRet).
      Register* tmp = temps_.AllocateStack();
      tc.emitLoadConst(tmp, TNoneType);
      stack.push(tmp);
    } else {
      stack.push(out);
    }
    return true;
  }

  return false;
}

std::vector<Register*> HIRBuilder::setupStaticArgs(
    TranslationContext& tc,
    const InvokeTarget& target,
    long nargs,
    bool statically_invoked) {
  auto arg_regs = std::vector<Register*>(nargs, nullptr);

  for (auto i = nargs - 1; i >= 0; i--) {
    arg_regs[i] = tc.frame.stack.pop();
  }

  // If we have patched a function that accepts/returns primitives,
  // but we couldn't emit a direct x64 call, we have to box any primitive args
  if (!target.primitive_arg_types.empty() && !statically_invoked) {
    for (auto [argnum, type] : target.primitive_arg_types) {
      Register* reg = arg_regs.at(argnum);
      auto boxed_primitive_tmp = temps_.AllocateStack();
      boxPrimitive(tc, boxed_primitive_tmp, reg, type);
      arg_regs[argnum] = boxed_primitive_tmp;
    }
  }

  return arg_regs;
}

void HIRBuilder::fixStaticReturn(
    TranslationContext& tc,
    Register* ret_val,
    Type ret_type) {
  Type boxed_ret = ret_type;
  if (boxed_ret <= TPrimitive) {
    boxed_ret = boxed_ret.asBoxed();
  }
  if (jit_get_config()->refine_static_python && boxed_ret < TObject) {
    tc.emitRefineType(ret_val, boxed_ret, ret_val);
  }

  // Since we are not doing an x64 call, we will get a boxed value; if the
  // function is supposed to return a primitive, we need to unbox it because
  // later code in the function will expect the primitive.
  if (ret_type <= TPrimitive) {
    unboxPrimitive(tc, ret_val, ret_val, ret_type);
  }
}

bool HIRBuilder::isStaticRand(const InvokeTarget& target) {
  return target.builtin_c_func == (void*)Ci_static_rand;
}

bool HIRBuilder::tryEmitStaticRandCall(
    const InvokeTarget& target,
    TranslationContext& tc,
    long nargs) {
  // Special case for static function call
  //     rand() -> int32
  //
  // This is a hack to support __static__.rand for now, since it's the most
  // common case. Eventually we'll get the typed method def support into
  // upstream CPython or CinderX and then we'll be able to have generic strongly
  // typed methods.

  if (nargs != 0) {
    return false;
  }

  Register* out = temps_.AllocateStack();
  Type ret_type = TCInt32;
  // Ci_static_rand() boxes the return value; call rand() directly instead.
  tc.emitCallStatic(nargs, out, (void*)rand, ret_type);
  tc.frame.stack.push(out);
  return true;
}

bool HIRBuilder::emitInvokeFunction(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr,
    CallFlags flags) {
  BorrowedRef<> arg = constArg(bc_instr);
  BorrowedRef<> descr = PyTuple_GET_ITEM(arg.get(), 0);
  long nargs = PyLong_AsLong(PyTuple_GET_ITEM(arg.get(), 1));

  const InvokeTarget& target = preloader_.invokeFunctionTarget(descr);

  if constexpr (PY_VERSION_HEX >= 0x030C0000) {
    // Hack to support a static type signature for __static__.rand(). Since we
    // don't have typed method defs in 3.12 we special case it here, by ignoring
    // all the metadata generated by the compiler pipeline and simply checking
    // that we are calling the Ci_static_rand function.
    if (isStaticRand(target) && tryEmitStaticRandCall(target, tc, nargs)) {
      return false;
    }
  }

  Register* funcreg = temps_.AllocateStack();

  if (target.container_is_immutable) {
    // try to emit a direct x64 call (InvokeStaticFunction/CallStatic) if we can

    if (target.is_function && target.is_statically_typed) {
      // Direct invoke is safe whether we succeeded in JIT-compiling or not,
      // it'll just have an extra indirection if not JIT compiled.
      Register* out = temps_.AllocateStack();
      Type typ = target.return_type;
      tc.emitLoadConst(funcreg, Type::fromObject(target.callable));

      auto call =
          tc.emit<InvokeStaticFunction>(nargs + 1, out, target.func(), typ);

      call->SetOperand(0, funcreg);

      for (auto i = nargs - 1; i >= 0; i--) {
        Register* operand = tc.frame.stack.pop();
        call->SetOperand(i + 1, operand);
      }
      call->setFrameState(tc.frame);

      tc.frame.stack.push(out);

      return false;
    } else if (
        target.is_builtin && tryEmitDirectMethodCall(target, tc, nargs)) {
      return false;
    }
    // we couldn't emit an x64 call, but we know what object we'll vectorcall,
    // so load it directly
    tc.emitLoadConst(funcreg, Type::fromObject(target.callable));
  } else {
    // The target is patchable so we have to load it indirectly
    tc.emit<LoadFunctionIndirect>(
        target.indirect_ptr, descr, funcreg, tc.frame);
  }

  std::vector<Register*> arg_regs =
      setupStaticArgs(tc, target, nargs, false /*statically_invoked*/);

  Register* out = temps_.AllocateStack();
  if (target.container_is_immutable) {
    flags |= CallFlags::Static;
  }

  // Add one for the function argument.
  auto call = tc.emitVectorCall(nargs + 1, out, flags);
  for (auto i = 0; i < nargs; i++) {
    call->SetOperand(i + 1, arg_regs.at(i));
  }
  call->SetOperand(0, funcreg);
  call->setFrameState(tc.frame);

  fixStaticReturn(tc, out, target.return_type);
  tc.frame.stack.push(out);

  return true;
}

bool HIRBuilder::emitInvokeNative(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  BorrowedRef<> arg = constArg(bc_instr);
  BorrowedRef<> native_target_descr = PyTuple_GET_ITEM(arg.get(), 0);
  const NativeTarget& target =
      preloader_.invokeNativeTarget(native_target_descr);

  BorrowedRef<> signature = PyTuple_GET_ITEM(arg.get(), 1);

  // The last entry in the signature is the return type, so subtract 1
  Py_ssize_t nargs = PyTuple_GET_SIZE(signature.get()) - 1;

  Register* out = temps_.AllocateStack();
  Type typ = target.return_type;
  auto call = tc.emitCallStatic(nargs, out, target.callable, typ);
  for (auto i = nargs - 1; i >= 0; i--) {
    Register* operand = tc.frame.stack.pop();
    call->SetOperand(i, operand);
  }

  tc.frame.stack.push(out);
  return false;
}

void HIRBuilder::emitInvokeMethodVectorCall(
    TranslationContext& tc,
    bool is_awaited,
    std::vector<Register*>& arg_regs,
    const InvokeTarget& target) {
  Register* out = temps_.AllocateStack();

  auto vectorCall = tc.emitVectorCall(
      arg_regs.size(), out, is_awaited ? CallFlags::Awaited : CallFlags::None);
  for (auto i = 0; i < arg_regs.size(); i++) {
    vectorCall->SetOperand(i, arg_regs.at(i));
  }
  vectorCall->setFrameState(tc.frame);

  fixStaticReturn(tc, out, target.return_type);
  tc.frame.stack.push(out);
}

void HIRBuilder::emitLoadMethodStatic(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  BorrowedRef<> arg = constArg(bc_instr);
  BorrowedRef<> descr = PyTuple_GET_ITEM(arg.get(), 0);
  bool is_classmethod = _PyClassLoader_IsClassMethodDescr(arg.get());

  const InvokeTarget& target = preloader_.invokeMethodTarget(descr);

  Register* self = tc.frame.stack.pop();
  auto type = temps_.AllocateStack();
  if (!is_classmethod) {
    tc.emitLoadField(
        type, self, "ob_type", offsetof(PyObject, ob_type), TType);
  } else {
    type = self;
  }

  Register* vtable = temps_.AllocateNonStack();
  Register* func_obj = temps_.AllocateNonStack();

  tc.emitLoadField(
      vtable, type, "tp_cache", offsetof(PyTypeObject, tp_cache), TObject);
  size_t entry_offset = offsetof(_PyType_VTable, vt_entries) +
      target.slot * sizeof(_PyType_VTableEntry);

  tc.emitLoadField(
      func_obj,
      vtable,
      "vte_state",
      entry_offset + offsetof(_PyType_VTableEntry, vte_state),
      TObject);

  // If this is natively callable then we'll want to get load_func for
  // the dispatch later. Otherwise we'll just vectorcall to the function.
  Register* entry_func = temps_.AllocateNonStack();
  Register* vtable_load = temps_.AllocateNonStack();

  tc.emitLoadField(
      vtable_load,
      vtable,
      "vte_load",
      entry_offset + offsetof(_PyType_VTableEntry, vte_load),
      TCPtr);

  auto call = tc.emit<CallInd>(
      3, func_obj, "vte_load", TOptObject, vtable_load, func_obj, self);
  call->setFrameState(tc.frame);

  if (target.is_statically_typed) {
    // the entry func isn't used by the interpreter and can't be de-opted but
    // we can have a LOAD_METHOD_STATIC that has another LOAD_METHOD_STATIC
    // before we get to the invokes.
    tc.emitGetSecondOutput(entry_func, TCPtr, func_obj);

    static_method_stack_.push(entry_func);
  }

  tc.frame.stack.push(func_obj);
  tc.frame.stack.push(self);
}

bool HIRBuilder::emitInvokeMethod(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr,
    bool is_awaited) {
  BorrowedRef<> arg = constArg(bc_instr);
  BorrowedRef<> descr = PyTuple_GET_ITEM(arg.get(), 0);
  long nargs = PyLong_AsLong(PyTuple_GET_ITEM(arg.get(), 1)) + 2; // thunk, self

  const InvokeTarget& target = preloader_.invokeMethodTarget(descr);

  if (target.is_builtin && tryEmitDirectMethodCall(target, tc, nargs - 1)) {
    auto res = tc.frame.stack.pop();
    tc.frame.stack.pop(); // pop the thunk
    tc.frame.stack.push(res);
    return false;
  }

  std::vector<Register*> arg_regs =
      setupStaticArgs(tc, target, nargs, target.is_statically_typed);

  if (target.is_statically_typed) {
    Register* out = temps_.AllocateNonStack();
    auto entry = static_method_stack_.pop();
    auto invoke =
        tc.emit<CallInd>(nargs + 1, out, "vtable invoke", target.return_type);
    invoke->SetOperand(0, entry);
    for (size_t i = 0; i < arg_regs.size(); i++) {
      invoke->SetOperand(i + 1, arg_regs[i]);
    }

    invoke->setFrameState(tc.frame);
    tc.frame.stack.push(out);
  } else {
    emitInvokeMethodVectorCall(tc, is_awaited, arg_regs, target);
  }

  return true;
}

void HIRBuilder::emitIsOp(TranslationContext& tc, int oparg) {
  auto& stack = tc.frame.stack;
  Register* right = stack.pop();
  Register* left = stack.pop();
  Register* unboxed_result = temps_.AllocateStack();
  Register* result = temps_.AllocateStack();
  auto op =
      oparg == 0 ? PrimitiveCompareOp::kEqual : PrimitiveCompareOp::kNotEqual;
  tc.emitPrimitiveCompare(unboxed_result, op, left, right);
  tc.emitPrimitiveBoxBool(result, unboxed_result);
  stack.push(result);
}

void HIRBuilder::emitContainsOp(TranslationContext& tc, int oparg) {
  auto& stack = tc.frame.stack;
  Register* right = stack.pop();
  Register* left = stack.pop();
  Register* result = temps_.AllocateStack();
  CompareOp op = oparg == 0 ? CompareOp::kIn : CompareOp::kNotIn;
  tc.emitCompare(result, op, left, right, tc.frame);
  stack.push(result);
}

void HIRBuilder::emitCompareOp(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  int compare_op = bc_instr.oparg();

  if constexpr (PY_VERSION_HEX >= 0x030E0000) {
    compare_op >>= 5;
  } else if constexpr (PY_VERSION_HEX >= 0x030B0000) {
    compare_op >>= 4;
  }

  JIT_CHECK(compare_op >= Py_LT, "Invalid op {}", compare_op);
  JIT_CHECK(compare_op <= Py_GE, "Invalid op {}", compare_op);
  auto& stack = tc.frame.stack;
  if (jit_get_config()->specialized_opcodes) {
    // Bug 7 fix: Snapshot BEFORE popping operands — deopt re-executes instruction
    tc.emitSnapshot();
  }
  Register* right = stack.pop();
  Register* left = stack.pop();
  Register* result = temps_.AllocateStack();
  CompareOp op = static_cast<CompareOp>(compare_op);

  if (jit_get_config()->specialized_opcodes) {
    switch (bc_instr.specializedOpcode()) {
      case COMPARE_OP_FLOAT:
        tc.emitGuardType(left, TFloatExact, left);
        tc.emitGuardType(right, TFloatExact, right);
        break;
      case COMPARE_OP_INT:
        tc.emitGuardType(left, TLongExact, left);
        tc.emitGuardType(right, TLongExact, right);
        break;
      case COMPARE_OP_STR:
        tc.emitGuardType(left, TUnicodeExact, left);
        tc.emitGuardType(right, TUnicodeExact, right);
        break;
      default:
        break;
    }
  }

  tc.emitCompare(result, op, left, right, tc.frame);
  stack.push(result);
  if (PY_VERSION_HEX >= 0x030E0000 && bc_instr.oparg() & 0x10) {
    emitToBool(tc);
  }
}

void HIRBuilder::emitToBool(TranslationContext& tc) {
  Register* operand = tc.frame.stack.pop();
  Register* truthy_result = temps_.AllocateStack();
  tc.emitIsTruthy(truthy_result, operand, tc.frame);

  Register* coerced_result = temps_.AllocateStack();
  tc.emitPrimitiveBoxBool(coerced_result, truthy_result);
  tc.frame.stack.push(coerced_result);
}

void HIRBuilder::emitCopyDictWithoutKeys(TranslationContext& tc) {
  auto& stack = tc.frame.stack;
  Register* keys = stack.top();
  Register* subject = stack.top(1);
  Register* rest = temps_.AllocateStack();
  tc.emitCopyDictWithoutKeys(rest, subject, keys, tc.frame);
  stack.topPut(0, rest);
}

void HIRBuilder::emitGetLen(TranslationContext& tc) {
  FrameState state = tc.frame;
  auto& stack = tc.frame.stack;
  Register* obj = stack.top();
  Register* result = temps_.AllocateStack();
  tc.emitGetLength(result, obj, state);
  stack.push(result);
}

void HIRBuilder::emitJumpIf(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  Register* var = tc.frame.stack.top();

  BCOffset true_offset, false_offset;
  bool check_truthy = true;
  auto opcode = bc_instr.opcode();
  switch (opcode) {
    case JUMP_IF_NONZERO_OR_POP:
      check_truthy = false;
      [[fallthrough]];
    case JUMP_IF_TRUE_OR_POP: {
      true_offset = bc_instr.getJumpTarget();
      false_offset = bc_instr.nextInstrOffset();
      break;
    }
    case JUMP_IF_ZERO_OR_POP:
      check_truthy = false;
      [[fallthrough]];
    case JUMP_IF_FALSE_OR_POP: {
      false_offset = bc_instr.getJumpTarget();
      true_offset = bc_instr.nextInstrOffset();
      break;
    }
    default: {
      // NOTREACHED
      JIT_ABORT(
          "Trying to translate non-jump-if bytecode {} ({})",
          opcode,
          opcodeName(opcode));
    }
  }

  BasicBlock* true_block = getBlockAtOff(true_offset);
  BasicBlock* false_block = getBlockAtOff(false_offset);

  if (check_truthy) {
    Register* tval = temps_.AllocateNonStack();
    // Registers that hold the result of `IsTruthy` are guaranteed to never be
    // the home of a value left on the stack at the end of a basic block, so we
    // don't need to worry about potentially storing a PyObject in them.
    tc.emitIsTruthy(tval, var, tc.frame);
    tc.emitCondBranch(tval, true_block, false_block);
  } else {
    tc.emitCondBranch(var, true_block, false_block);
  }
}

namespace {

// Walk the type hierarchy starting from 'base' to find a type whose
// tp_version_tag matches 'version'. Used at JIT compile time to recover
// the PyTypeObject* from CPython's inline cache (which only stores the
// version tag, not the type pointer).
//
// Cost: O(number_of_types) but runs once per LOAD_ATTR_SLOT at JIT compile
// time, not at runtime.
// Get the tp_subclasses dict for a type, handling the static builtin
// indirection (static builtin types store subclasses in interpreter state,
// not directly in tp_subclasses).
PyObject* getTypeSubclasses(PyTypeObject* type) {
  if (type->tp_flags & _Py_TPFLAGS_STATIC_BUILTIN) {
    PyInterpreterState* interp = _PyInterpreterState_GET();
    static_builtin_state* state = _PyStaticType_GetState(interp, type);
    return state ? state->tp_subclasses : nullptr;
  }
  return (PyObject*)type->tp_subclasses;
}

PyTypeObject* findTypeByVersionTagImpl(
    PyTypeObject* base,
    uint32_t version,
    int depth) {
  if (depth > 50) {
    return nullptr;
  }
  if (base->tp_version_tag == version) {
    return base;
  }
  // Iterate tp_subclasses directly using PyDict_Next — zero pymalloc
  // allocation. The previous _PyType_GetSubclasses() call allocated a
  // temporary list via PyList_New, which corrupted pymalloc pool metadata
  // when called during auto-compilation on ARM64.
  PyObject* subclasses = getTypeSubclasses(base);
  if (subclasses == nullptr || !PyDict_Check(subclasses)) {
    return nullptr;
  }
  Py_ssize_t pos = 0;
  PyObject* ref;
  while (PyDict_Next(subclasses, &pos, nullptr, &ref)) {
    PyObject* sub = PyWeakref_GetObject(ref);
    if (sub == Py_None || sub == nullptr || !PyType_Check(sub)) {
      continue;
    }
    PyTypeObject* found =
        findTypeByVersionTagImpl((PyTypeObject*)sub, version, depth + 1);
    if (found != nullptr) {
      return found;
    }
  }
  return nullptr;
}

PyTypeObject* findTypeByVersionTag(uint32_t version) {
  if (version == 0) {
    return nullptr;
  }
  return findTypeByVersionTagImpl(&PyBaseObject_Type, version, 0);
}

} // namespace

void HIRBuilder::emitDeleteAttr(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  Register* receiver = tc.frame.stack.pop();
  tc.emitDeleteAttr(receiver, bc_instr.oparg(), tc.frame);
}

void HIRBuilder::emitLoadAttr(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  int oparg = bc_instr.oparg();
  int name_idx = loadAttrIndex(oparg);

  // In 3.12 LOAD_METHOD has been merged into LOAD_ATTR, and the oparg tells you
  // which one it should be.
  if constexpr (PY_VERSION_HEX >= 0x030C0000) {
    if (oparg & 1) {
      emitLoadMethod(tc, name_idx);
      return;
    }
  }

  if (jit_get_config()->specialized_opcodes) {
    // Bug 7 fix: Snapshot BEFORE popping operands — deopt re-executes instruction
    tc.emitSnapshot();
  }
  Register* receiver = tc.frame.stack.pop();

  if (jit_get_config()->specialized_opcodes) {
    switch (bc_instr.specializedOpcode()) {
      case LOAD_ATTR_MODULE: {
        // Guard receiver is a module
        Type mod_type = Type::fromTypeExact(&PyModule_Type);
        tc.emitGuardType(receiver, mod_type, receiver);

        // Read dict_version and index from CPython's inline cache
        _Py_CODEUNIT* code_units = codeUnit(code_);
        int instr_idx = bc_instr.opcodeIndex().value();
        const _PyAttrCache* cache =
            reinterpret_cast<const _PyAttrCache*>(
                &code_units[instr_idx + 1]);
        uint32_t dict_version = cache->version[0] |
            (static_cast<uint32_t>(cache->version[1]) << 16);
        uint16_t index = cache->index;

        if (dict_version != 0) {
          // Inline dict access: module->md_dict->ma_keys->dk_version
          Register* dict = temps_.AllocateStack();
          tc.emitLoadField(
              dict, receiver, "md_dict",
              offsetof(PyModuleObject, md_dict), TObject);

          Register* keys = temps_.AllocateStack();
          tc.emitLoadField(
              keys, dict, "ma_keys",
              offsetof(PyDictObject, ma_keys), TCPtr);

          Register* loaded_version = temps_.AllocateStack();
          tc.emitLoadField(
              loaded_version, keys, "dk_version",
              offsetof(PyDictKeysObject, dk_version), TCUInt32);

          Register* expected_version = temps_.AllocateStack();
          tc.emitLoadConst(
              expected_version,
              Type::fromCUInt(dict_version, TCUInt32));

          Register* version_match = temps_.AllocateStack();
          tc.emitPrimitiveCompare(
              version_match, PrimitiveCompareOp::kEqual,
              loaded_version, expected_version);
          tc.emitGuard(version_match, tc.frame);

          // Load entry value via helper (computes DK_UNICODE_ENTRIES)
          Register* index_reg = temps_.AllocateStack();
          tc.emitLoadConst(
              index_reg,
              Type::fromCInt(static_cast<int64_t>(index), TCInt64));

          Register* result = temps_.AllocateStack();
          auto call = tc.emitCallStatic(
              2, result,
              reinterpret_cast<void*>(JITRT_LoadModuleDictEntry),
              TOptObject);
          call->SetOperand(0, keys);
          call->SetOperand(1, index_reg);

          // Deopt if value is NULL (attribute deleted)
          BorrowedRef<> attr_name =
              PyTuple_GET_ITEM(code_->co_names, name_idx);
          auto cf = tc.emitCheckField(
              result, result, attr_name, tc.frame);
          cf->setGuiltyReg(receiver);

          tc.frame.stack.push(result);
          return;
        }
        break;
      }
      case LOAD_ATTR_SLOT: {
        // LOAD_ATTR_SLOT direct LoadField specialisation.
        // For __slots__ types, cache->index is the byte offset from the
        // object pointer to the slot value. Emit GuardType + direct
        // LoadField at that offset, bypassing the generic C API path.
        _Py_CODEUNIT* code_units = codeUnit(code_);
        int instr_idx = bc_instr.opcodeIndex().value();
        const _PyAttrCache* cache =
            reinterpret_cast<const _PyAttrCache*>(&code_units[instr_idx + 1]);
        uint32_t type_version =
            cache->version[0] |
            (static_cast<uint32_t>(cache->version[1]) << 16);
        uint16_t slot_offset = cache->index;

        PyTypeObject* slot_type = findTypeByVersionTag(type_version);
        if (slot_type != nullptr &&
            (slot_type->tp_subclasses == nullptr ||
             PyDict_GET_SIZE(slot_type->tp_subclasses) == 0)) {
          // Root the type so it lives as long as the compiled code.
          // Without this, GC may collect the type during auto-compilation,
          // causing GuardType to compare against a freed pointer.
          current_func_->env.addReference(BorrowedRef<>(
              reinterpret_cast<PyObject*>(slot_type)));
          Type type = Type::fromTypeExact(slot_type);
          tc.emitGuardType(receiver, type, receiver);

          BorrowedRef<> attr_name =
              PyTuple_GET_ITEM(code_->co_names, name_idx);

          // Direct load at the slot byte offset.
          // CPython LOAD_ATTR_SLOT does: *(PyObject**)((char*)owner + index)
          Register* attr = temps_.AllocateStack();
          tc.emitLoadField(
              attr, receiver, "slot",
              static_cast<int>(slot_offset), TOptObject);

          // Check attribute is not NULL (uninitialised slot).
          Register* result = temps_.AllocateStack();
          auto cf = tc.emitCheckField(
              result, attr, attr_name, tc.frame);
          cf->setGuiltyReg(receiver);

          tc.frame.stack.push(result);
          return;
        }
        // Fallback: no type found or has subclasses — generic LoadAttr.
        break;
      }
      case LOAD_ATTR_INSTANCE_VALUE: {
        // LOAD_ATTR_INSTANCE_VALUE direct LoadField specialisation.
        // Instead of falling through to generic LoadAttr (which goes through
        // C++ inline cache -> dict lookup -> branch mispredictions), emit
        // a direct field load at the cached offset from CPython's adaptive
        // cache. This matches what simplifyLoadAttrSplitDict does in the
        // Simplify pass, but at build time.
        _Py_CODEUNIT* code_units = codeUnit(code_);
        int instr_idx = bc_instr.opcodeIndex().value();
        const _PyAttrCache* cache =
            reinterpret_cast<const _PyAttrCache*>(&code_units[instr_idx + 1]);
        uint32_t type_version =
            cache->version[0] |
            (static_cast<uint32_t>(cache->version[1]) << 16);
        uint16_t index = cache->index;

        PyTypeObject* slot_type = findTypeByVersionTag(type_version);
        if (slot_type != nullptr &&
            (slot_type->tp_subclasses == nullptr ||
             PyDict_GET_SIZE(slot_type->tp_subclasses) == 0) &&
            PyType_HasFeature(slot_type, Py_TPFLAGS_MANAGED_DICT)) {

          BorrowedRef<PyHeapTypeObject> ht(reinterpret_cast<PyHeapTypeObject*>(slot_type));
          if (ht->ht_cached_keys != nullptr) {
            // Root the type so it lives as long as the compiled code.
            current_func_->env.addReference(BorrowedRef<>(
                reinterpret_cast<PyObject*>(slot_type)));
            Type type = Type::fromTypeExact(slot_type);
            tc.emitGuardType(receiver, type, receiver);

            BorrowedRef<> attr_name =
                PyTuple_GET_ITEM(code_->co_names, name_idx);

            // Load PyDictOrValues from managed dict slot (offset -3).
            // In CPython 3.12, managed dict pointer is at
            // -3 * sizeof(PyObject*) from the object.
            // Must remain borrowed: dorv can be a tagged pointer (low bit
            // set for inline values) which is not a valid PyObject*.
            Register* dorv = temps_.AllocateStack();
            tc.emitLoadField(
                dorv, receiver, "__dict__",
                -3 * static_cast<int>(sizeof(PyObject*)), TOptDict, true);

            // Check dorv is not NULL (object has dict/values allocated).
            Register* checked_dorv = temps_.AllocateStack();
            auto cf = tc.emitCheckField(
                checked_dorv, dorv, attr_name, tc.frame);
            cf->setGuiltyReg(receiver);

            // Check low bit: 1 means inline values, 0 means regular dict.
            Register* one = temps_.AllocateStack();
            tc.emitLoadConst(one, Type::fromCUInt(1, TCUInt64));
            Register* dorv_int = temps_.AllocateStack();
            tc.emitBitCast(dorv_int, checked_dorv, TCUInt64);
            Register* is_values = temps_.AllocateStack();
            tc.emitIntBinaryOp(
                is_values, BinaryOpKind::kAnd, dorv_int, one);
            auto* guard = static_cast<DeoptBase*>(tc.emitGuard(is_values, tc.frame));
            guard->setGuiltyReg(receiver);
            guard->setDescr("dict values check");

            // Get values pointer: dorv + 1 (see simplifyLoadAttrSplitDict).
            // The tagged pointer layout makes dorv+1 point to the start
            // of the values array for LoadField indexing.
            Register* values_ptr = temps_.AllocateStack();
            tc.emitIntBinaryOp(
                values_ptr, BinaryOpKind::kAdd, dorv_int, one);
            Register* values_obj = temps_.AllocateStack();
            tc.emitBitCast(values_obj, values_ptr, TOptObject);

            // Load attribute at cached index.
            Register* attr = temps_.AllocateStack();
            tc.emitLoadField(
                attr, values_obj, "attr",
                static_cast<int>(index) * static_cast<int>(sizeof(PyObject*)),
                TOptObject);

            // Check attribute is not NULL (not deleted).
            Register* result = temps_.AllocateStack();
            auto cf2 = tc.emitCheckField(
                result, attr, attr_name, tc.frame);
            cf2->setGuiltyReg(receiver);

            tc.frame.stack.push(result);
            return;
          }
        }
        // Fallback: emit GuardType only if we found the type but couldn't
        // do the direct load (e.g., no MANAGED_DICT, has subclasses).
        if (slot_type != nullptr) {
          if (slot_type->tp_subclasses == nullptr ||
              PyDict_GET_SIZE(slot_type->tp_subclasses) == 0) {
            Type type = Type::fromTypeExact(slot_type);
            tc.emitGuardType(receiver, type, receiver);
          }
        }
        break;
      }
      default:
        break;
    }
  }

  Register* result = temps_.AllocateStack();
  tc.emit<LoadAttr>(result, receiver, name_idx, tc.frame);
  tc.frame.stack.push(result);
}

void HIRBuilder::emitLoadMethod(TranslationContext& tc, int name_idx) {
  Register* receiver = tc.frame.stack.pop();
  Register* result = temps_.AllocateStack();
  Register* method_instance = temps_.AllocateStack();
  tc.emit<LoadMethod>(result, receiver, name_idx, tc.frame);
  tc.emitGetSecondOutput(method_instance, TOptObject, result);
  tc.frame.stack.push(result);
  tc.frame.stack.push(method_instance);
}

void HIRBuilder::emitLoadMethodOrAttrSuper(
    CFG& cfg,
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr,
    bool load_method) {
  TranslationContext deopt_path{cfg.AllocateBlock(), tc.frame};
  Register* receiver = tc.frame.stack.pop();
  Register* type = tc.frame.stack.pop();
  Register* global_super = tc.frame.stack.pop();
  Register* result = temps_.AllocateStack();

#if PY_VERSION_HEX >= 0x030B0000
  int oparg = bc_instr.oparg();
  int name_idx = oparg >> 2;
  load_method = oparg & 1;
  bool no_args_in_super_call = !(oparg & 2);
#else
  PyObject* oparg = PyTuple_GET_ITEM(code_->co_consts, bc_instr.oparg());
  int name_idx = PyLong_AsLong(PyTuple_GET_ITEM(oparg, 0));
  bool no_args_in_super_call = PyTuple_GET_ITEM(oparg, 1) == Py_True;
#endif

  // This is assumed to be a type object by the rest of the JIT.  Ideally it
  // would be typed by whatever pushes it onto the stack.
  deopt_path.frame.cur_instr_offs = bc_instr.baseOffset();
  deopt_path.emitSnapshot();
  deopt_path.emitDeopt();
  BasicBlock* fast_path = cfg.AllocateBlock();
  tc.emitCondBranchCheckType(type, TType, fast_path, deopt_path.block);
  tc.block = fast_path;
  tc.emitRefineType(type, TType, type);

  if (!load_method) {
    tc.emit<LoadAttrSuper>(
        result,
        global_super,
        type,
        receiver,
        name_idx,
        no_args_in_super_call,
        tc.frame);
    tc.frame.stack.push(result);
    return;
  }

  Register* method_instance = temps_.AllocateStack();
  tc.emit<LoadMethodSuper>(
      result,
      global_super,
      type,
      receiver,
      name_idx,
      no_args_in_super_call,
      tc.frame);
  tc.emitGetSecondOutput(method_instance, TOptObject, result);
  tc.frame.stack.push(result);
  tc.frame.stack.push(method_instance);
}

void HIRBuilder::emitMakeCell(TranslationContext& tc, int local_idx) {
  Register* local = tc.frame.localsplus[local_idx];
  Register* cell = temps_.AllocateNonStack();
  tc.emitMakeCell(cell, local, tc.frame);
  moveOverwrittenStackRegisters(tc, local);
  tc.emitAssign(local, cell);
}

void HIRBuilder::emitCopy(TranslationContext& tc, int item_idx) {
  JIT_CHECK(item_idx > 0, "The index ({}) must be positive!", item_idx);
  Register* item = tc.frame.stack.peek(item_idx);
  tc.frame.stack.push(item);
}

void HIRBuilder::emitCopyFreeVars(TranslationContext& tc, int nfreevars) {
  JIT_CHECK(nfreevars > 0, "Can't initialize {} freevars", nfreevars);
  JIT_CHECK(
      nfreevars == numFreevars(code_),
      "COPY_FREE_VARS oparg doesn't match the function's freevars tuple");
  JIT_CHECK(func_ != nullptr, "No func_ in function with freevars");

  Register* func_closure = temps_.AllocateNonStack();
  tc.emitLoadField(
      func_closure,
      func_,
      "func_closure",
      offsetof(PyFunctionObject, func_closure),
      TTuple);
  int offset = numLocalsplus(code_) - nfreevars;
  for (int i = 0; i < nfreevars; ++i) {
    Register* dst = tc.frame.localsplus[offset + i];
    JIT_CHECK(dst != nullptr, "No register for free var {}", i);
    tc.emitLoadTupleItem(dst, func_closure, i);
  }
  if constexpr (PY_VERSION_HEX >= 0x030C0000) {
    tc.emit<InitFrameCellVars>(func_, nfreevars);
  }
}

void HIRBuilder::emitSwap(TranslationContext& tc, int item_idx) {
  JIT_CHECK(
      item_idx >= 2, "The index ({}) must be greater or equal to 2.", item_idx);
  Register* item = tc.frame.stack.peek(item_idx);
  Register* top = tc.frame.stack.top();
  tc.frame.stack.topPut(0, item);
  tc.frame.stack.topPut(item_idx - 1, top);
}

void HIRBuilder::emitLoadDeref(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  // <3.11, the oparg was the cell index.  >=3.11 it's the same index as any
  // other local / frame value.
  int idx = bc_instr.oparg();
  if constexpr (PY_VERSION_HEX < 0x030B0000) {
    idx += tc.frame.nlocals;
  }

  Register* src = tc.frame.localsplus[idx];
  Register* dst = temps_.AllocateStack();

  tc.emitLoadCellItem(dst, src);

  BorrowedRef<> name = getVarname(code_, idx);
#if PY_VERSION_HEX < 0x030C0000
  tc.emitCheckVar(dst, dst, name, tc.frame);
#else
  if (idx < PyCode_GetFirstFree(code_)) {
    tc.emitCheckVar(dst, dst, name, tc.frame);
  } else {
    tc.emitCheckFreevar(dst, dst, name, tc.frame);
  }
#endif

  tc.frame.stack.push(dst);
}

void HIRBuilder::emitStoreDeref(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  // <3.11, the oparg was the cell index.  >=3.11 it's the same index as any
  // other local / frame value.
  int idx = bc_instr.oparg();
  if constexpr (PY_VERSION_HEX < 0x030B0000) {
    idx += tc.frame.nlocals;
  }

  Register* old = temps_.AllocateStack();
  Register* dst = tc.frame.localsplus[idx];
  Register* src = tc.frame.stack.pop();
#ifdef Py_GIL_DISABLED
  // Use atomic swap for thread-safe cell access in FT-Python.
  tc.emitSwapCellItem(old, dst, src);
#else
  tc.emitStealCellItem(old, dst);
  tc.emitSetCellItem(dst, src, old);
#endif
}

void HIRBuilder::emitLoadAssertionError(
    TranslationContext& tc,
    Environment& env) {
  Register* result = temps_.AllocateStack();
  tc.emitLoadConst(
      result, Type::fromObject(env.addReference(PyExc_AssertionError)));
  tc.frame.stack.push(result);
}

void HIRBuilder::emitLoadClass(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  Register* tmp = temps_.AllocateStack();
  auto pytype = preloader_.pyType(constArg(bc_instr));
  auto pytype_as_pyobj = BorrowedRef(pytype);
  tc.emitLoadConst(tmp, Type::fromObject(pytype_as_pyobj));
  tc.frame.stack.push(tmp);
}

void HIRBuilder::emitLoadConst(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  Register* tmp = temps_.AllocateStack();
  JIT_CHECK(
      bc_instr.oparg() < PyTuple_Size(code_->co_consts),
      "LOAD_CONST index out of bounds");
  tc.emitLoadConst(
      tmp,
      Type::fromObject(PyTuple_GET_ITEM(code_->co_consts, bc_instr.oparg())));
  tc.frame.stack.push(tmp);
}

void HIRBuilder::emitLoadFast(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  int var_idx = bc_instr.oparg();
  Register* var = tc.frame.localsplus[var_idx];
  // Pre-3.12, LOAD_FAST behaves like LOAD_FAST_CHECK.
  if (bc_instr.opcode() == LOAD_FAST_CHECK || PY_VERSION_HEX < 0x030C0000) {
    tc.emitCheckVar(var, var, getVarname(code_, var_idx), tc.frame);
  }
  tc.frame.stack.push(var);
  if (bc_instr.opcode() == LOAD_FAST_AND_CLEAR) {
    moveOverwrittenStackRegisters(tc, var);
    tc.emitLoadConst(var, TNullptr);
  }
}

void HIRBuilder::emitLoadFastLoadFast(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  int var_idx1 = bc_instr.oparg() >> 4;
  int var_idx2 = bc_instr.oparg() & 0xf;
  size_t localsplus_size = tc.frame.localsplus.size();
  JIT_CHECK(
      var_idx1 < localsplus_size && var_idx2 < localsplus_size,
      "LOAD_FAST_LOAD_FAST ({}, {}) out of bounds for localsplus array size {}",
      var_idx1,
      var_idx2,
      tc.frame.localsplus.size());
  Register* var1 = tc.frame.localsplus[var_idx1];
  tc.frame.stack.push(var1);

  Register* var2 = tc.frame.localsplus[var_idx2];
  tc.frame.stack.push(var2);
}

void HIRBuilder::emitLoadLocal(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  PyObject* index_and_descr =
      PyTuple_GET_ITEM(code_->co_consts, bc_instr.oparg());
  int index = PyLong_AsLong(PyTuple_GET_ITEM(index_and_descr, 0));

  auto var = tc.frame.localsplus[index];
  tc.frame.stack.push(var);
}

void HIRBuilder::emitLoadSmallInt(
    [[maybe_unused]] TranslationContext& tc,
    [[maybe_unused]] const jit::BytecodeInstruction& bc_instr) {
#if PY_VERSION_HEX >= 0x030E0000
  Register* tmp = temps_.AllocateStack();
  JIT_CHECK(
      bc_instr.oparg() < _PY_NSMALLPOSINTS, "LOAD_SMALL_INT out of range");
  tc.emitLoadConst(
      tmp,
      Type::fromObject(
          reinterpret_cast<PyObject*>(
              &_PyLong_SMALL_INTS[_PY_NSMALLNEGINTS + bc_instr.oparg()])));
  tc.frame.stack.push(tmp);
#else
  JIT_ABORT("LOAD_SMALL_INT not supported on this Python version");
#endif
}

void HIRBuilder::emitStoreLocal(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  Register* src = tc.frame.stack.pop();
  PyObject* index_and_descr =
      PyTuple_GET_ITEM(code_->co_consts, bc_instr.oparg());
  int index = PyLong_AsLong(PyTuple_GET_ITEM(index_and_descr, 0));
  auto dst = tc.frame.localsplus[index];
  moveOverwrittenStackRegisters(tc, dst);
  tc.emitAssign(dst, src);
}

void HIRBuilder::emitLoadType(
    TranslationContext& tc,
    const jit::BytecodeInstruction&) {
  Register* instance = tc.frame.stack.pop();
  auto type = temps_.AllocateStack();
  tc.emitLoadField(
      type, instance, "ob_type", offsetof(PyObject, ob_type), TType);
  tc.frame.stack.push(type);
}

void HIRBuilder::emitConvertPrimitive(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  Register* val = tc.frame.stack.pop();
  Register* out = temps_.AllocateStack();
  Type to_type = prim_type_to_type(bc_instr.oparg() >> 4);
  tc.emitIntConvert(out, val, to_type);
  tc.frame.stack.push(out);
}

void HIRBuilder::emitPrimitiveLoadConst(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  Register* tmp = temps_.AllocateStack();
  int index = bc_instr.oparg();
  JIT_CHECK(
      index < PyTuple_Size(code_->co_consts),
      "PRIMITIVE_LOAD_CONST index out of bounds");
  PyObject* num_and_type = PyTuple_GET_ITEM(code_->co_consts, index);
  JIT_CHECK(
      PyTuple_Size(num_and_type) == 2,
      "wrong size for PRIMITIVE_LOAD_CONST arg tuple")
  PyObject* num = PyTuple_GET_ITEM(num_and_type, 0);
  Type size =
      prim_type_to_type(PyLong_AsSsize_t(PyTuple_GET_ITEM(num_and_type, 1)));
  Type type = TBottom;
  if (size == TCDouble) {
    type = Type::fromCDouble(PyFloat_AsDouble(num));
  } else if (size <= TCBool) {
    type = Type::fromCBool(num == Py_True);
  } else {
    type = (size <= TCUnsigned)
        ? Type::fromCUInt(PyLong_AsUnsignedLong(num), size)
        : Type::fromCInt(PyLong_AsLong(num), size);
  }
  tc.emitLoadConst(tmp, type);
  tc.frame.stack.push(tmp);
}

void HIRBuilder::emitPrimitiveBox(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  Register* tmp = temps_.AllocateStack();
  Register* src = tc.frame.stack.pop();
  Type typ = prim_type_to_type(bc_instr.oparg());
  boxPrimitive(tc, tmp, src, typ);
  tc.frame.stack.push(tmp);
}

void HIRBuilder::emitPrimitiveUnbox(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  Register* tmp = temps_.AllocateStack();
  Register* src = tc.frame.stack.pop();
  Type typ = prim_type_to_type(bc_instr.oparg());
  unboxPrimitive(tc, tmp, src, typ);
  tc.frame.stack.push(tmp);
}

void HIRBuilder::boxPrimitive(
    TranslationContext& tc,
    Register* dst,
    Register* src,
    Type type) {
  if (type <= TCBool) {
    tc.emitPrimitiveBoxBool(dst, src);
  } else {
    tc.emitPrimitiveBox(dst, src, type, tc.frame);
  }
}

void HIRBuilder::unboxPrimitive(
    TranslationContext& tc,
    Register* dst,
    Register* src,
    Type type) {
  tc.emitPrimitiveUnbox(dst, src, type);
  if (!(type <= (TCBool | TCDouble))) {
    Register* did_unbox_work = temps_.AllocateStack();
    tc.emitIsNegativeAndErrOccurred(did_unbox_work, dst, tc.frame);
  }
}

static inline BinaryOpKind get_primitive_bin_op_kind(
    const jit::BytecodeInstruction& bc_instr) {
  switch (bc_instr.oparg()) {
    case PRIM_OP_ADD_DBL:
    case PRIM_OP_ADD_INT: {
      return BinaryOpKind::kAdd;
    }
    case PRIM_OP_AND_INT: {
      return BinaryOpKind::kAnd;
    }
    case PRIM_OP_DIV_INT: {
      return BinaryOpKind::kFloorDivide;
    }
    case PRIM_OP_DIV_UN_INT: {
      return BinaryOpKind::kFloorDivideUnsigned;
    }
    case PRIM_OP_LSHIFT_INT: {
      return BinaryOpKind::kLShift;
    }
    case PRIM_OP_MOD_INT: {
      return BinaryOpKind::kModulo;
    }
    case PRIM_OP_MOD_UN_INT: {
      return BinaryOpKind::kModuloUnsigned;
    }
    case PRIM_OP_MUL_DBL:
    case PRIM_OP_MUL_INT: {
      return BinaryOpKind::kMultiply;
    }
    case PRIM_OP_OR_INT: {
      return BinaryOpKind::kOr;
    }
    case PRIM_OP_RSHIFT_INT: {
      return BinaryOpKind::kRShift;
    }
    case PRIM_OP_RSHIFT_UN_INT: {
      return BinaryOpKind::kRShiftUnsigned;
    }
    case PRIM_OP_SUB_DBL:
    case PRIM_OP_SUB_INT: {
      return BinaryOpKind::kSubtract;
    }
    case PRIM_OP_XOR_INT: {
      return BinaryOpKind::kXor;
    }
    case PRIM_OP_DIV_DBL: {
      return BinaryOpKind::kTrueDivide;
    }
    case PRIM_OP_POW_UN_INT: {
      return BinaryOpKind::kPowerUnsigned;
    }
    case PRIM_OP_POW_INT:
    case PRIM_OP_POW_DBL: {
      return BinaryOpKind::kPower;
    }
    default: {
      JIT_ABORT("Unhandled binary op {}", bc_instr.oparg());
      // NOTREACHED
    }
  }
}

static inline bool is_double_binop(int oparg) {
  switch (oparg) {
    case PRIM_OP_ADD_INT:
    case PRIM_OP_AND_INT:
    case PRIM_OP_DIV_INT:
    case PRIM_OP_DIV_UN_INT:
    case PRIM_OP_LSHIFT_INT:
    case PRIM_OP_MOD_INT:
    case PRIM_OP_MOD_UN_INT:
    case PRIM_OP_POW_INT:
    case PRIM_OP_POW_UN_INT:
    case PRIM_OP_MUL_INT:
    case PRIM_OP_OR_INT:
    case PRIM_OP_RSHIFT_INT:
    case PRIM_OP_RSHIFT_UN_INT:
    case PRIM_OP_SUB_INT:
    case PRIM_OP_XOR_INT: {
      return false;
    }
    case PRIM_OP_ADD_DBL:
    case PRIM_OP_SUB_DBL:
    case PRIM_OP_DIV_DBL:
    case PRIM_OP_MUL_DBL:
    case PRIM_OP_POW_DBL: {
      return true;
    }
    default: {
      JIT_ABORT("Invalid binary op {}", oparg);
      // NOTREACHED
    }
  }
}

static inline Type element_type_from_seq_type(int seq_type) {
  switch (seq_type) {
    case SEQ_LIST:
    case SEQ_LIST_INEXACT:
    case SEQ_CHECKED_LIST:
    case SEQ_TUPLE:
      return TObject;
    case SEQ_ARRAY_INT64:
      return TCInt64;
    default:
      JIT_ABORT("Invalid sequence type: ({})", seq_type);
      // NOTREACHED
  }
}

void HIRBuilder::emitPrimitiveBinaryOp(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  auto& stack = tc.frame.stack;
  Register* right = stack.pop();
  Register* left = stack.pop();
  Register* result = temps_.AllocateStack();

  BinaryOpKind op_kind = get_primitive_bin_op_kind(bc_instr);

  if (is_double_binop(bc_instr.oparg())) {
    tc.emitDoubleBinaryOp(result, op_kind, left, right);
  } else {
    tc.emitIntBinaryOp(result, op_kind, left, right);
  }

  stack.push(result);
}

void HIRBuilder::emitPrimitiveCompare(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  auto& stack = tc.frame.stack;
  Register* right = stack.pop();
  Register* left = stack.pop();
  Register* result = temps_.AllocateStack();
  PrimitiveCompareOp op;
  switch (bc_instr.oparg()) {
    case PRIM_OP_EQ_INT:
    case PRIM_OP_EQ_DBL:
      op = PrimitiveCompareOp::kEqual;
      break;
    case PRIM_OP_NE_INT:
    case PRIM_OP_NE_DBL:
      op = PrimitiveCompareOp::kNotEqual;
      break;
    case PRIM_OP_LT_INT:
      op = PrimitiveCompareOp::kLessThan;
      break;
    case PRIM_OP_LE_INT:
      op = PrimitiveCompareOp::kLessThanEqual;
      break;
    case PRIM_OP_GT_INT:
      op = PrimitiveCompareOp::kGreaterThan;
      break;
    case PRIM_OP_GE_INT:
      op = PrimitiveCompareOp::kGreaterThanEqual;
      break;
    case PRIM_OP_LT_UN_INT:
    case PRIM_OP_LT_DBL:
      op = PrimitiveCompareOp::kLessThanUnsigned;
      break;
    case PRIM_OP_LE_UN_INT:
    case PRIM_OP_LE_DBL:
      op = PrimitiveCompareOp::kLessThanEqualUnsigned;
      break;
    case PRIM_OP_GT_UN_INT:
    case PRIM_OP_GT_DBL:
      op = PrimitiveCompareOp::kGreaterThanUnsigned;
      break;
    case PRIM_OP_GE_UN_INT:
    case PRIM_OP_GE_DBL:
      op = PrimitiveCompareOp::kGreaterThanEqualUnsigned;
      break;
    default:
      JIT_ABORT("unsupported comparison");
  }
  tc.emitPrimitiveCompare(result, op, left, right);
  stack.push(result);
}

void HIRBuilder::emitPrimitiveUnaryOp(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  Register* value = tc.frame.stack.pop();
  Register* result = temps_.AllocateStack();
  PrimitiveUnaryOpKind op;
  switch (bc_instr.oparg()) {
    case PRIM_OP_NEG_INT: {
      op = PrimitiveUnaryOpKind::kNegateInt;
      tc.emitPrimitiveUnaryOp(result, op, value);
      break;
    }
    case PRIM_OP_INV_INT: {
      op = PrimitiveUnaryOpKind::kInvertInt;
      tc.emitPrimitiveUnaryOp(result, op, value);
      break;
    }
    case PRIM_OP_NOT_INT: {
      op = PrimitiveUnaryOpKind::kNotInt;
      tc.emitPrimitiveUnaryOp(result, op, value);
      break;
    }
    case PRIM_OP_NEG_DBL: {
      // For doubles, there's no easy way to unary negate a value, so just
      // multiply it by -1
      auto tmp = temps_.AllocateStack();
      tc.emitLoadConst(tmp, Type::fromCDouble(-1.0));
      tc.emitDoubleBinaryOp(result, BinaryOpKind::kMultiply, tmp, value);
      break;
    }
    default: {
      JIT_ABORT("unsupported unary op");
    }
  }
  tc.frame.stack.push(result);
}

void HIRBuilder::emitFastLen(
    CFG& cfg,
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  auto result = temps_.AllocateStack();
  Register* collection;
  auto oparg = bc_instr.oparg();
  int inexact = oparg & FAST_LEN_INEXACT;
  std::size_t offset = 0;
  auto type = TBottom;

  oparg &= ~FAST_LEN_INEXACT;
  const char* name = "";
  if (oparg == FAST_LEN_LIST) {
    type = TListExact;
    offset = offsetof(PyVarObject, ob_size);
    name = "ob_size";
  } else if (oparg == FAST_LEN_TUPLE) {
    type = TTupleExact;
    offset = offsetof(PyVarObject, ob_size);
    name = "ob_size";
  } else if (oparg == FAST_LEN_ARRAY) {
    type = TArray;
    offset = offsetof(PyVarObject, ob_size);
    name = "ob_size";
  } else if (oparg == FAST_LEN_DICT) {
    type = TDictExact;
    offset = offsetof(PyDictObject, ma_used);
    name = "ma_used";
  } else if (oparg == FAST_LEN_SET) {
    type = TSetExact;
    offset = offsetof(PySetObject, used);
    name = "used";
  } else if (oparg == FAST_LEN_STR) {
    type = TUnicodeExact;
    // Note: In debug mode, the interpreter has an assert that
    // ensures the string is "ready", check PyUnicode_GET_LENGTH
    offset = offsetof(PyASCIIObject, length);
    name = "length";
  }
  JIT_CHECK(offset > 0, "Bad oparg for FAST_LEN");

  if (inexact) {
    TranslationContext deopt_path{cfg.AllocateBlock(), tc.frame};
    deopt_path.frame.cur_instr_offs = bc_instr.baseOffset();
    deopt_path.emitSnapshot();
    deopt_path.emitDeopt();
    collection = tc.frame.stack.pop();
    BasicBlock* fast_path = cfg.AllocateBlock();
    tc.emitCondBranchCheckType(collection, type, fast_path, deopt_path.block);
    tc.block = fast_path;
    // TASK(T105038867): Remove once we have RefineTypeInsertion
    tc.emitRefineType(collection, type, collection);
  } else {
    collection = tc.frame.stack.pop();
  }

  tc.emitLoadField(result, collection, name, offset, TCInt64);
  tc.frame.stack.push(result);
}

void HIRBuilder::emitRefineType(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  Type type = preloader_.type(constArg(bc_instr));
  Register* dst = tc.frame.stack.top();
  tc.emitRefineType(dst, type, dst);
}

void HIRBuilder::emitSequenceGet(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  auto& stack = tc.frame.stack;
  auto idx = stack.pop();
  auto sequence = stack.pop();
  auto oparg = bc_instr.oparg();
  if (oparg == SEQ_LIST_INEXACT) {
    auto type = temps_.AllocateStack();
    tc.emitLoadField(
        type, sequence, "ob_type", offsetof(PyObject, ob_type), TType);
    tc.emitGuardIs(type, (PyObject*)&PyList_Type, type);
    tc.emitRefineType(sequence, TListExact, sequence);
  }

  Register* adjusted_idx;
  int unchecked = oparg & SEQ_SUBSCR_UNCHECKED;
  if (!unchecked) {
    adjusted_idx = temps_.AllocateStack();
    tc.emitCheckSequenceBounds(adjusted_idx, sequence, idx, tc.frame);
  } else {
    adjusted_idx = idx;
    oparg &= ~SEQ_SUBSCR_UNCHECKED;
  }
  auto ob_item = temps_.AllocateStack();
  auto result = temps_.AllocateStack();
  if (oparg == SEQ_LIST || oparg == SEQ_LIST_INEXACT ||
      oparg == SEQ_CHECKED_LIST) {
    int offset = offsetof(PyListObject, ob_item);
    tc.emitLoadField(ob_item, sequence, "ob_item", offset, TCPtr);
  } else if (oparg == SEQ_ARRAY_INT64) {
    Register* offset_reg = temps_.AllocateStack();
    tc.emitLoadConst(
        offset_reg,
        Type::fromCInt(offsetof(PyStaticArrayObject, ob_item), TCInt64));
    tc.emitLoadFieldAddress(ob_item, sequence, offset_reg);
  } else {
    JIT_ABORT("Unsupported oparg for SEQUENCE_GET: {}", oparg);
  }

  auto type = element_type_from_seq_type(oparg);
  tc.emitLoadArrayItem(
      result, ob_item, adjusted_idx, sequence, /*offset=*/0, type);
  stack.push(result);
}

void HIRBuilder::emitSequenceSet(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  auto& stack = tc.frame.stack;
  auto idx = stack.pop();
  auto sequence = stack.pop();
  auto value = stack.pop();
  auto adjusted_idx = temps_.AllocateStack();
  auto oparg = bc_instr.oparg();
  if (oparg == SEQ_LIST_INEXACT) {
    auto type = temps_.AllocateStack();
    tc.emitLoadField(
        type, sequence, "ob_type", offsetof(PyObject, ob_type), TType);
    tc.emitGuardIs(type, (PyObject*)&PyList_Type, type);
    tc.emitRefineType(sequence, TListExact, sequence);
  }
  tc.emitCheckSequenceBounds(adjusted_idx, sequence, idx, tc.frame);
  auto ob_item = temps_.AllocateStack();
  if (oparg == SEQ_ARRAY_INT64) {
    Register* offset_reg = temps_.AllocateStack();
    tc.emitLoadConst(
        offset_reg,
        Type::fromCInt(offsetof(PyStaticArrayObject, ob_item), TCInt64));
    tc.emitLoadFieldAddress(ob_item, sequence, offset_reg);
  } else if (oparg == SEQ_LIST || oparg == SEQ_LIST_INEXACT) {
    int offset = offsetof(PyListObject, ob_item);
    tc.emitLoadField(ob_item, sequence, "ob_item", offset, TCPtr);
  } else {
    JIT_ABORT("Unsupported oparg for SEQUENCE_SET: {}", oparg);
  }
  tc.emit<StoreArrayItem>(
      ob_item,
      adjusted_idx,
      value,
      sequence,
      element_type_from_seq_type(oparg));
}

void HIRBuilder::emitLoadGlobal(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  int name_idx = loadGlobalIndex(bc_instr.oparg());
  Register* result = temps_.AllocateStack();

  if constexpr (PY_VERSION_HEX >= 0x030B0000 && PY_VERSION_HEX < 0x030E0000) {
    if (bc_instr.oparg() & 1) {
      emitPushNull(tc);
    }
  }

  auto try_fast_path = [&] {
    if (!jit_get_config()->stable_frame) {
      return false;
    }
    BorrowedRef<> value = preloader_.global(name_idx);
    if (value == nullptr) {
      return false;
    }
    tc.emit<LoadGlobalCached>(
        result, code_, preloader_.builtins(), preloader_.globals(), name_idx);
    auto guard_is = tc.emitGuardIs(result, value, result);
    BorrowedRef<> name = PyTuple_GET_ITEM(code_->co_names, name_idx);
    guard_is->setDescr(fmt::format("LOAD_GLOBAL: {}", PyUnicode_AsUTF8(name)));
    return true;
  };

  if (!try_fast_path()) {
    tc.emitLoadGlobal(result, name_idx, tc.frame);
  }

  tc.frame.stack.push(result);

  if constexpr (PY_VERSION_HEX >= 0x030E0000) {
    if (bc_instr.oparg() & 1) {
      emitPushNull(tc);
    }
  }
}

void HIRBuilder::emitMakeFunction(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  int oparg = bc_instr.oparg();
  Register* func = temps_.AllocateStack();

  // In 3.10 the function's qualname is on the stack.  In 3.11+ it's computed
  // from the code object, so we use a sentinel Nullptr value here.
  Register* qualname = nullptr;
  if constexpr (PY_VERSION_HEX < 0x030B0000) {
    qualname = tc.frame.stack.pop();
  } else {
    qualname = temps_.AllocateNonStack();
    tc.emitLoadConst(qualname, TNullptr);
  }

  Register* codeobj = tc.frame.stack.pop();

  // make a function
  tc.emit<MakeFunction>(func, codeobj, qualname, tc.frame);

  if (oparg & MAKE_FUNCTION_CLOSURE) {
    Register* closure = tc.frame.stack.pop();
    tc.emitSetFunctionAttr(closure, func, FunctionAttr::kClosure);
  }
  if (oparg & MAKE_FUNCTION_ANNOTATIONS) {
    Register* annotations = tc.frame.stack.pop();
    tc.emitSetFunctionAttr(annotations, func, FunctionAttr::kAnnotations);
  }
  if (oparg & MAKE_FUNCTION_KWDEFAULTS) {
    Register* kwdefaults = tc.frame.stack.pop();
    tc.emitSetFunctionAttr(kwdefaults, func, FunctionAttr::kKwDefaults);
  }
  if (oparg & MAKE_FUNCTION_DEFAULTS) {
    Register* defaults = tc.frame.stack.pop();
    tc.emitSetFunctionAttr(defaults, func, FunctionAttr::kDefaults);
  }

  tc.frame.stack.push(func);
}

void HIRBuilder::emitMakeListTuple(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  auto num_elems = static_cast<size_t>(bc_instr.oparg());
  auto dst = temps_.AllocateStack();
  Instr* instr;
  if (bc_instr.opcode() == BUILD_TUPLE) {
    instr = tc.emit<MakeTuple>(num_elems, dst, tc.frame);
  } else {
    instr = tc.emit<MakeList>(num_elems, dst, tc.frame);
  }
  for (size_t i = num_elems; i > 0; i--) {
    auto opnd = tc.frame.stack.pop();
    instr->SetOperand(i - 1, opnd);
  }
  tc.frame.stack.push(dst);
}

void HIRBuilder::emitListExtend(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  Register* iterable = tc.frame.stack.pop();
  Register* list = tc.frame.stack.peek(bc_instr.oparg());
  Register* none = temps_.AllocateStack();
  tc.emitListExtend(none, list, iterable, tc.frame);
}

void HIRBuilder::emitListToTuple(TranslationContext& tc) {
  Register* list = tc.frame.stack.pop();
  Register* tuple = temps_.AllocateStack();
  tc.emitMakeTupleFromList(tuple, list, tc.frame);
  tc.frame.stack.push(tuple);
}

void HIRBuilder::emitBuildCheckedList(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  BorrowedRef<> arg = constArg(bc_instr);
  BorrowedRef<> descr = PyTuple_GET_ITEM(arg.get(), 0);
  Py_ssize_t list_size = PyLong_AsLong(PyTuple_GET_ITEM(arg.get(), 1));

  Type type = preloader_.type(descr);
  JIT_CHECK(
      Ci_CheckedList_TypeCheck(type.uniquePyType()),
      "expected CheckedList type");

  Register* list = temps_.AllocateStack();
  auto instr = tc.emit<MakeCheckedList>(list_size, list, type, tc.frame);
  // Fill list
  for (size_t i = list_size; i > 0; i--) {
    auto operand = tc.frame.stack.pop();
    instr->SetOperand(i - 1, operand);
  }
  tc.frame.stack.push(list);
}

void HIRBuilder::emitBuildCheckedMap(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  BorrowedRef<> arg = constArg(bc_instr);
  BorrowedRef<> descr = PyTuple_GET_ITEM(arg.get(), 0);
  Py_ssize_t dict_size = PyLong_AsLong(PyTuple_GET_ITEM(arg.get(), 1));

  Type type = preloader_.type(descr);
  JIT_CHECK(
      Ci_CheckedDict_TypeCheck(type.uniquePyType()),
      "expected CheckedDict type");

  Register* dict = temps_.AllocateStack();
  tc.emit<MakeCheckedDict>(dict, dict_size, type, tc.frame);
  // Fill dict
  auto& stack = tc.frame.stack;
  for (auto i = stack.size() - dict_size * 2, end = stack.size(); i < end;
       i += 2) {
    auto key = stack.at(i);
    auto value = stack.at(i + 1);
    auto result = temps_.AllocateStack();
    tc.emitSetDictItem(result, dict, key, value, tc.frame);
  }
  stack.discard(dict_size * 2);
  stack.push(dict);
}

void HIRBuilder::emitBuildMap(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  auto dict_size = bc_instr.oparg();
  Register* dict = temps_.AllocateStack();
  tc.emitMakeDict(dict, dict_size, tc.frame);
  // Fill dict
  auto& stack = tc.frame.stack;
  for (auto i = stack.size() - dict_size * 2, end = stack.size(); i < end;
       i += 2) {
    auto key = stack.at(i);
    auto value = stack.at(i + 1);
    auto result = temps_.AllocateStack();
    tc.emitSetDictItem(result, dict, key, value, tc.frame);
  }
  stack.discard(dict_size * 2);
  stack.push(dict);
}

void HIRBuilder::emitBuildSet(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  Register* set = temps_.AllocateStack();
  tc.emitMakeSet(set, tc.frame);

  int oparg = bc_instr.oparg();
  for (int i = oparg; i > 0; i--) {
    auto item = tc.frame.stack.peek(i);

    auto result = temps_.AllocateStack();
    tc.emitSetSetItem(result, set, item, tc.frame);
  }

  tc.frame.stack.discard(oparg);

  tc.frame.stack.push(set);
}

void HIRBuilder::emitBuildConstKeyMap(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  auto dict_size = bc_instr.oparg();
  Register* dict = temps_.AllocateStack();
  tc.emitMakeDict(dict, dict_size, tc.frame);
  // Fill dict
  auto& stack = tc.frame.stack;
  Register* keys = stack.pop();
  // ceval.c checks the type and size of the keys tuple before proceeding; we
  // intentionally skip that here.
  for (auto i = 0; i < dict_size; ++i) {
    Register* key = temps_.AllocateStack();
    tc.emitLoadTupleItem(key, keys, i);
    Register* value = stack.at(stack.size() - dict_size + i);
    Register* result = temps_.AllocateStack();
    tc.emitSetDictItem(result, dict, key, value, tc.frame);
  }
  stack.discard(dict_size);
  stack.push(dict);
}

void HIRBuilder::emitPopJumpIf(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  Register* var = tc.frame.stack.pop();
  BCOffset true_offset, false_offset;
  auto opcode = bc_instr.opcode();
  switch (opcode) {
    case POP_JUMP_IF_ZERO:
    case POP_JUMP_IF_FALSE: {
      true_offset = bc_instr.nextInstrOffset();
      false_offset = bc_instr.getJumpTarget();
      break;
    }
    case POP_JUMP_IF_NONZERO:
    case POP_JUMP_IF_TRUE: {
      true_offset = bc_instr.getJumpTarget();
      false_offset = bc_instr.nextInstrOffset();
      break;
    }
    default: {
      // NOTREACHED
      JIT_ABORT(
          "Trying to translate non pop-jump bytecode {} ({})",
          opcode,
          opcodeName(opcode));
    }
  }

  BasicBlock* true_block = getBlockAtOff(true_offset);
  BasicBlock* false_block = getBlockAtOff(false_offset);

  if (bc_instr.opcode() == POP_JUMP_IF_FALSE ||
      bc_instr.opcode() == POP_JUMP_IF_TRUE) {
    Register* is_true = temps_.AllocateNonStack();
    // In 3.14+ coercion to exactly Py_True or Py_False is performed by earlier
    // instructions. See GH-106008.
    if constexpr (PY_VERSION_HEX >= 0x030E0000) {
      Register* const_true = temps_.AllocateNonStack();
      tc.emitLoadConst(const_true, Type::fromObject(Py_True));
      tc.emitPrimitiveCompare(
          is_true, PrimitiveCompareOp::kEqual, var, const_true);
    } else {
      tc.emitIsTruthy(is_true, var, tc.frame);
    }
    tc.emitCondBranch(is_true, true_block, false_block);
  } else {
    tc.emitCondBranch(var, true_block, false_block);
  }
}

void HIRBuilder::emitPopJumpIfNone(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  Register* var = tc.frame.stack.pop();
  BCOffset true_offset = bc_instr.getJumpTarget();
  BCOffset false_offset = bc_instr.nextInstrOffset();

  BasicBlock* true_block = getBlockAtOff(true_offset);
  BasicBlock* false_block = getBlockAtOff(false_offset);

  auto none = temps_.AllocateNonStack();
  tc.emitLoadConst(none, Type::fromObject(Py_None));
  auto is_true = temps_.AllocateNonStack();
  auto op = bc_instr.opcode() == POP_JUMP_IF_NONE
      ? PrimitiveCompareOp::kEqual
      : PrimitiveCompareOp::kNotEqual;
  tc.emitPrimitiveCompare(is_true, op, var, none);
  tc.emitCondBranch(is_true, true_block, false_block);
}

void HIRBuilder::emitStoreAttr(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  Register* receiver = tc.frame.stack.pop();
  Register* value = tc.frame.stack.pop();

  tc.emitStoreAttr(receiver, value, bc_instr.oparg(), tc.frame);
}

void HIRBuilder::moveOverwrittenStackRegisters(
    TranslationContext& tc,
    Register* dst) {
  // If we're about to overwrite a register that is on the stack, move it to a
  // new register.
  Register* tmp = nullptr;
  auto& stack = tc.frame.stack;
  for (std::size_t i = 0, stack_size = stack.size(); i < stack_size; i++) {
    if (stack.at(i) == dst) {
      if (tmp == nullptr) {
        tmp = temps_.AllocateStack();
        tc.emitAssign(tmp, dst);
      }
      stack.atPut(i, tmp);
    }
  }
}
void HIRBuilder::emitStoreFast(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  Register* src = tc.frame.stack.pop();
  Register* dst = tc.frame.localsplus[bc_instr.oparg()];
  JIT_DCHECK(dst != nullptr, "no register");
  moveOverwrittenStackRegisters(tc, dst);
  tc.emitAssign(dst, src);
}

void HIRBuilder::emitStoreFastStoreFast(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  int var_idx1 = bc_instr.oparg() >> 4;
  int var_idx2 = bc_instr.oparg() & 0xf;
  size_t localsplus_size = tc.frame.localsplus.size();
  JIT_CHECK(
      var_idx1 < localsplus_size && var_idx2 < localsplus_size,
      "STORE_FAST_STORE_FAST ({}, {}) out of bounds for localsplus array size "
      "{}",
      var_idx1,
      var_idx2,
      tc.frame.localsplus.size());
  Register* src = tc.frame.stack.pop();
  Register* dst = tc.frame.localsplus[var_idx1];
  moveOverwrittenStackRegisters(tc, dst);
  tc.emitAssign(dst, src);

  src = tc.frame.stack.pop();
  dst = tc.frame.localsplus[var_idx2];
  moveOverwrittenStackRegisters(tc, dst);
  tc.emitAssign(dst, src);
}

void HIRBuilder::emitStoreFastLoadFast(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  int var_idx1 = bc_instr.oparg() >> 4;
  int var_idx2 = bc_instr.oparg() & 0xf;
  size_t localsplus_size = tc.frame.localsplus.size();
  JIT_CHECK(
      var_idx1 < localsplus_size && var_idx2 < localsplus_size,
      "STORE_FAST_LOAD_FAST ({}, {}) out of bounds for localsplus array size "
      "{}",
      var_idx1,
      var_idx2,
      tc.frame.localsplus.size());
  Register* src = tc.frame.stack.pop();
  Register* dst = tc.frame.localsplus[var_idx1];
  moveOverwrittenStackRegisters(tc, dst);
  tc.emitAssign(dst, src);

  Register* var = tc.frame.localsplus[var_idx2];
  tc.frame.stack.push(var);
}

void HIRBuilder::emitBinarySlice(TranslationContext& tc) {
  auto& stack = tc.frame.stack;
  tc.emitVariadic<BuildSlice>(temps_, 2);
  Register* slice = stack.pop();
  Register* container = stack.pop();
  Register* result = temps_.AllocateStack();
  tc.emitBinaryOp(
      result, BinaryOpKind::kSubscript, container, slice, tc.frame);
  tc.frame.stack.push(result);
}

void HIRBuilder::emitStoreSlice(TranslationContext& tc) {
  auto& stack = tc.frame.stack;
  tc.emitVariadic<BuildSlice>(temps_, 2);
  Register* slice = stack.pop();
  Register* container = stack.pop();
  Register* values = stack.pop();
  tc.emitStoreSubscr(container, slice, values, tc.frame);
}

void HIRBuilder::emitStoreSubscr(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  auto& stack = tc.frame.stack;
  if (jit_get_config()->specialized_opcodes) {
    // Bug 7 fix: Snapshot BEFORE popping operands — deopt re-executes instruction
    tc.emitSnapshot();
  }
  Register* sub = stack.pop();
  Register* container = stack.pop();
  Register* value = stack.pop();

  if (jit_get_config()->specialized_opcodes) {
    switch (bc_instr.specializedOpcode()) {
      case STORE_SUBSCR_DICT:
        tc.emitGuardType(container, TDictExact, container);
        break;
      case STORE_SUBSCR_LIST_INT:
        tc.emitGuardType(container, TListExact, container);
        tc.emitGuardType(sub, TLongExact, sub);
        break;
      default:
        break;
    }
  }

  tc.emitStoreSubscr(container, sub, value, tc.frame);
}

void HIRBuilder::emitGetIter(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  Register* iterable = tc.frame.stack.pop();
  Register* result = temps_.AllocateStack();
  tc.emitGetIter(result, iterable, tc.frame);
  // FOR_ITER specialisation: if the next instruction is a specialised FOR_ITER,
  // guard the iterator type here (once, before the loop) rather than inside
  // the loop body. This enables the Simplify pass to replace generic
  // InvokeIterNext with CallStatic(JITRT_InvokeIterNext).
  if (jit_get_config()->specialized_opcodes) {
    // Bug 7 fix: The Snapshot/GuardType FrameState must reflect the
    // interpreter state AT FOR_ITER (not GET_ITER), because:
    // - GET_ITER already executed successfully (we have a valid iterator)
    // - If the type guard fails, the interpreter should resume at FOR_ITER
    //   with the iterator on the stack (FOR_ITER handles any iterator type)
    // - The old code had cur_instr_offs=GET_ITER with iterable popped,
    //   causing the interpreter to re-execute GET_ITER with garbage stack.
    tc.frame.stack.push(result);  // iterator must be on stack for FOR_ITER
    auto saved_offs = tc.frame.cur_instr_offs;
    tc.frame.cur_instr_offs = bc_instr.nextInstr().baseOffset();  // FOR_ITER
    tc.emitSnapshot();
    auto next_instr = bc_instr.nextInstr();
    auto next_opcode = next_instr.specializedOpcode();
    if (next_opcode == FOR_ITER_RANGE &&
        jit::g_range_iterator_type != nullptr) {
      Type range_iter_type =
          Type::fromTypeExact(jit::g_range_iterator_type);
      tc.emitGuardType(result, range_iter_type, result, tc.frame);
    } else if (next_opcode == FOR_ITER_LIST &&
               jit::g_list_iterator_type != nullptr) {
      Type list_iter_type =
          Type::fromTypeExact(jit::g_list_iterator_type);
      tc.emitGuardType(result, list_iter_type, result, tc.frame);
    } else if (next_opcode == FOR_ITER_TUPLE &&
               jit::g_tuple_iterator_type != nullptr) {
      Type tuple_iter_type =
          Type::fromTypeExact(jit::g_tuple_iterator_type);
      tc.emitGuardType(result, tuple_iter_type, result, tc.frame);
    }
    tc.frame.cur_instr_offs = saved_offs;  // restore for subsequent processing
    // result already pushed above - do NOT push again
  } else {
    tc.frame.stack.push(result);
  }
  if constexpr (PY_VERSION_HEX >= 0x030F0000) {
    // TASK(T243355471): We should support virtual indexing
    emitPushNull(tc);
  }
}

void HIRBuilder::emitForIter(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  Register* iterator;
  if constexpr (PY_VERSION_HEX >= 0x030F0000) {
    iterator = tc.frame.stack.top(1);
  } else {
    iterator = tc.frame.stack.top();
  }
  Register* next_val = temps_.AllocateStack();
  tc.emitInvokeIterNext(next_val, iterator, tc.frame);
  tc.frame.stack.push(next_val);
  BasicBlock* footer = getBlockAtOff(bc_instr.getJumpTarget());
  BasicBlock* body = getBlockAtOff(bc_instr.nextInstrOffset());
  tc.emitCondBranchIterNotDone(next_val, body, footer);
}

void HIRBuilder::emitGetYieldFromIter(CFG& cfg, TranslationContext& tc) {
  Register* iter_in = tc.frame.stack.pop();

  bool in_coro = code_->co_flags & (CO_COROUTINE | CO_ITERABLE_COROUTINE);
  BasicBlock* done_block = cfg.AllocateBlock();
  BasicBlock* next_block = cfg.AllocateBlock();
  BasicBlock* nop_block = cfg.AllocateBlock();
  BasicBlock* is_coro_block = in_coro ? nop_block : cfg.AllocateBlock();

#if PY_VERSION_HEX >= 0x030C0000
  BasicBlock* check_coro_block = cfg.AllocateBlock();
  tc.emitCondBranchCheckType(
      iter_in,
      Type::fromTypeExact(cinderx::getModuleState()->coroType()),
      is_coro_block,
      check_coro_block);

  tc.block = check_coro_block;
#endif
  tc.emitCondBranchCheckType(
      iter_in, Type::fromTypeExact(&PyCoro_Type), is_coro_block, next_block);

  if (!in_coro) {
    tc.block = is_coro_block;
    tc.emit<RaiseStatic>(
        0,
        PyExc_TypeError,
        "cannot 'yield from' a coroutine object in a non-coroutine generator",
        tc.frame);
  }

  tc.block = next_block;

  BasicBlock* slow_path = cfg.AllocateBlock();
  Register* iter_out = temps_.AllocateStack();
  tc.emitCondBranchCheckType(iter_in, TGen, nop_block, slow_path);

  tc.block = slow_path;
  tc.emitGetIter(iter_out, iter_in, tc.frame);
  tc.emitBranch(done_block);

  tc.block = nop_block;
  tc.emitAssign(iter_out, iter_in);
  tc.emitBranch(done_block);

  tc.block = done_block;
  tc.frame.stack.push(iter_out);
}

void HIRBuilder::emitUnpackEx(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  int oparg = bc_instr.oparg();
  int arg_before = oparg & 0xff;
  int arg_after = oparg >> 8;

  auto& stack = tc.frame.stack;
  Register* seq = stack.pop();

  Register* tuple = temps_.AllocateStack();
  tc.emit<UnpackExToTuple>(tuple, seq, arg_before, arg_after, tc.frame);

  int total_args = arg_before + arg_after + 1;
  for (int i = total_args - 1; i >= 0; i--) {
    Register* item = temps_.AllocateStack();
    tc.emitLoadTupleItem(item, tuple, i);
    stack.push(item);
  }
}

void HIRBuilder::emitUnpackSequence(
    CFG& cfg,
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  auto& stack = tc.frame.stack;
  Register* seq = stack.top();

  if (jit_get_config()->specialized_opcodes) {
    // Bug 6 fix: ensure dominating Snapshot for specialised-opcode GuardType
    tc.emitSnapshot();
    switch (bc_instr.specializedOpcode()) {
      case UNPACK_SEQUENCE_LIST:
        tc.emitGuardType(seq, TListExact, seq, tc.frame);
        break;
      case UNPACK_SEQUENCE_TUPLE:
      case UNPACK_SEQUENCE_TWO_TUPLE:
        tc.emitGuardType(seq, TTupleExact, seq, tc.frame);
        break;
      default:
        break;
    }
  }

  TranslationContext deopt_path{cfg.AllocateBlock(), tc.frame};
  deopt_path.frame.cur_instr_offs = bc_instr.baseOffset();
  deopt_path.emitSnapshot();
  auto* deopt = static_cast<Deopt*>(deopt_path.emitDeopt());
  deopt->setGuiltyReg(seq);
  deopt->setDescr("UNPACK_SEQUENCE");

  BasicBlock* fast_path = cfg.AllocateBlock();
  BasicBlock* list_check_path = cfg.AllocateBlock();
  BasicBlock* list_fast_path = cfg.AllocateBlock();
  BasicBlock* tuple_fast_path = cfg.AllocateBlock();
  Register* list_mem = temps_.AllocateStack();
  stack.pop();

  // TODO: The manual type checks and branches should go away once we get
  // PGO support to be able to optimize to known types.

  //---
  // +-main------------------------------+         +-tuple_fast_path------+
  // | CondBranchCheckType (TTupleExact) |-truthy->| LoadConst (ob_item)  |
  // +-----------------------------------+         | LoadFieldAddress     |
  //    |                                          | Branch               |--+
  //  falsy                                        +----------------------+  |
  //    |                                                                    |
  //    v                                                                    |
  // +-list_check_path------------------+         +-list_fast_path------+    |
  // | CondBranchCheckType (TListExact) |-truthy->| LoadField (ob_item) |    |
  // +----------------------------------+         | Branch              |----+
  //   |                                          +---------------------+    |
  //  falsy                                                                  |
  //   |                                          +-fast_path---------+      |
  //   |                                          | LoadVarObjectSize |<-----+
  //   v                                          | LoadConst         |
  // +-deopt_path-+                               | PrimitiveCompare  |
  // | Deopt      |<----------falsy---------------| CondBranch        |------+
  // +------------+                               +-------------------+      |
  //                                                                         |
  //                                              +-fast_path-----+          |
  //                                              | LoadConst     |<-truthy--+
  //                                              | LoadArrayItem |
  //                                              +---------------+
  //---

  if (seq->isA(TTupleExact)) {
    tc.emitBranch(tuple_fast_path);
  } else if (seq->isA(TListExact)) {
// TODO(T255264577). Enable this again. See P2169677587.
#ifdef Py_GIL_DISABLED
    tc.emitBranch(deopt_path.block);
#else
    tc.emitBranch(list_fast_path);
#endif
  } else {
    tc.emitCondBranchCheckType(
        seq, TTupleExact, tuple_fast_path, list_check_path);

    tc.block = list_check_path;
// TODO(T255264577). Enable this again. See P2169677587.
#ifdef Py_GIL_DISABLED
    tc.emitBranch(deopt_path.block);
#else
    tc.emitCondBranchCheckType(
        seq, TListExact, list_fast_path, deopt_path.block);
#endif
  }

  tc.block = tuple_fast_path;
  Register* offset_reg = temps_.AllocateStack();
  tc.emitLoadConst(
      offset_reg, Type::fromCInt(offsetof(PyTupleObject, ob_item), TCInt64));
  tc.emitLoadFieldAddress(list_mem, seq, offset_reg);
  tc.emitBranch(fast_path);

  tc.block = list_fast_path;
  tc.emitLoadField(
      list_mem, seq, "ob_item", offsetof(PyListObject, ob_item), TCPtr);
  tc.emitBranch(fast_path);

  tc.block = fast_path;

  Register* seq_size = temps_.AllocateStack();
  Register* target_size = temps_.AllocateStack();
  Register* is_equal = temps_.AllocateStack();
  tc.emitLoadVarObjectSize(seq_size, seq);
  tc.emitLoadConst(target_size, Type::fromCInt(bc_instr.oparg(), TCInt64));
  tc.emitPrimitiveCompare(
      is_equal, PrimitiveCompareOp::kEqual, seq_size, target_size);
  fast_path = cfg.AllocateBlock();
  tc.emitCondBranch(is_equal, fast_path, deopt_path.block);
  tc.block = fast_path;

  Register* idx_reg = temps_.AllocateStack();
  for (int idx = bc_instr.oparg() - 1; idx >= 0; --idx) {
    Register* item = temps_.AllocateStack();
    tc.emitLoadConst(idx_reg, Type::fromCInt(idx, TCInt64));
    tc.emitLoadArrayItem(item, list_mem, idx_reg, seq, 0, TObject);
    stack.push(item);
  }
}

void HIRBuilder::emitSetupFinally(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  BCOffset handler_off =
      bc_instr.nextInstrOffset() + BCIndex{bc_instr.oparg()}.asOffset();
  int stack_level = tc.frame.stack.size();
  tc.frame.block_stack.push(
      ExecutionBlock{SETUP_FINALLY, handler_off, stack_level});
}

void HIRBuilder::emitAsyncForHeaderYieldFrom(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  Register* send_value = tc.frame.stack.pop();
  Register* awaitable = tc.frame.stack.top();
  Register* out = temps_.AllocateStack();
  if (code_->co_flags & CO_COROUTINE) {
    tc.emitSetCurrentAwaiter(awaitable);
  }
  tc.emit<YieldFromHandleStopAsyncIteration>(
      out, send_value, awaitable, tc.frame);
  tc.frame.stack.pop();
  tc.frame.stack.push(out);

  BasicBlock* yf_cont_block = getBlockAtOff(bc_instr.nextInstrOffset());
  BCOffset handler_off{tc.frame.block_stack.top().handler_off};
  BasicBlock* yf_done_block = getBlockAtOff(handler_off);
  tc.emitCondBranchIterNotDone(out, yf_cont_block, yf_done_block);
}

void HIRBuilder::emitEndAsyncFor(TranslationContext& tc) {
  // Pop finally block and discard exhausted async iterator.
  const ExecutionBlock& b = tc.frame.block_stack.top();
  JIT_CHECK(
      static_cast<int>(tc.frame.stack.size()) == b.stack_level,
      "Bad stack depth in END_ASYNC_FOR: block stack expects {}, stack is {}",
      b.stack_level,
      tc.frame.stack.size());
  tc.frame.block_stack.pop();
  tc.frame.stack.pop();
}

void HIRBuilder::emitGetAIter(TranslationContext& tc) {
  Register* obj = tc.frame.stack.pop();
  Register* out = temps_.AllocateStack();
  tc.emitGetAIter(out, obj, tc.frame);
  tc.frame.stack.push(out);
}

void HIRBuilder::emitGetANext(TranslationContext& tc) {
  Register* obj = tc.frame.stack.top();
  Register* out = temps_.AllocateStack();
  tc.emitGetANext(out, obj, tc.frame);
  tc.frame.stack.push(out);
}

Register* HIRBuilder::emitSetupWithCommon(
    TranslationContext& tc,
#if PY_VERSION_HEX < 0x030C0000
    _Py_Identifier* enter_id,
    _Py_Identifier* exit_id,
#else
    PyObject* enter_id,
    PyObject* exit_id,
#endif
    bool is_async) {
  // Load the enter and exit attributes from the manager, push exit, and return
  // the result of calling enter().
  auto& stack = tc.frame.stack;
  Register* manager = stack.pop();
  Register* enter = temps_.AllocateStack();
  Register* exit = temps_.AllocateStack();
  tc.emit<LoadAttrSpecial>(
      enter,
      manager,
      enter_id,
      is_async
          ? "'%.200s' object does not support the asynchronous context manager "
            "protocol"
          : "'%.200s' object does not support the context manager protocol",
      tc.frame);
  tc.emit<LoadAttrSpecial>(
      exit,
      manager,
      exit_id,
      is_async
          ? "'%.200s' object does not support the asynchronous context manager "
            "protocol (missed __aexit__ method)"
          : "'%.200s' object does not support the context manager protocol "
            "(missed __exit__ method)",
      tc.frame);
  stack.push(exit);

  Register* enter_result = temps_.AllocateStack();
  auto call = tc.emitVectorCall(1, enter_result, CallFlags::None);
  call->setFrameState(tc.frame);
  call->SetOperand(0, enter);
  return enter_result;
}

void HIRBuilder::emitBeforeWith(
    TranslationContext& tc,
    [[maybe_unused]] const jit::BytecodeInstruction& bc_instr) {
#if PY_VERSION_HEX < 0x030C0000
  _Py_IDENTIFIER(__aenter__);
  _Py_IDENTIFIER(__aexit__);
  tc.frame.stack.push(
      emitSetupWithCommon(tc, &PyId___aenter__, &PyId___aexit__, true));
#else
  if (bc_instr.opcode() == BEFORE_ASYNC_WITH) {
    tc.frame.stack.push(
        emitSetupWithCommon(tc, &_Py_ID(__aenter__), &_Py_ID(__aexit__), true));
  } else {
    tc.frame.stack.push(
        emitSetupWithCommon(tc, &_Py_ID(__enter__), &_Py_ID(__exit__), false));
  }
#endif
}

void HIRBuilder::emitSetupAsyncWith(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  // The finally block should be above the result of __aenter__.
  Register* top = tc.frame.stack.pop();
  emitSetupFinally(tc, bc_instr);
  tc.frame.stack.push(top);
}

void HIRBuilder::emitSetupWith(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
#if PY_VERSION_HEX < 0x030C0000
  _Py_IDENTIFIER(__enter__);
  _Py_IDENTIFIER(__exit__);
  Register* enter_result =
      emitSetupWithCommon(tc, &PyId___enter__, &PyId___exit__, false);
#else
  Register* enter_result =
      emitSetupWithCommon(tc, &_Py_ID(__aenter__), &_Py_ID(__aexit__), true);
#endif
  emitSetupFinally(tc, bc_instr);
  tc.frame.stack.push(enter_result);
}

void HIRBuilder::emitLoadField(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  auto& [offset, type, name] = preloader_.fieldInfo(constArg(bc_instr));

  Register* receiver = tc.frame.stack.pop();
  Register* result = temps_.AllocateStack();
  const char* field_name = PyUnicode_AsUTF8(name);
  if (field_name == nullptr) {
    PyErr_Clear();
    field_name = "";
  }
  tc.emitLoadField(result, receiver, field_name, offset, type);
  HirType h_type = to_hir(type), h_null = to_hir(TNullptr);
  if (hir_type_could_be(&h_type, &h_null)) {
    auto* cf = tc.emitCheckField(result, result, name, tc.frame);
    cf->setGuiltyReg(receiver);
  }
  tc.frame.stack.push(result);
}

void HIRBuilder::emitStoreField(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  auto& [offset, type, name] = preloader_.fieldInfo(constArg(bc_instr));
  const char* field_name = PyUnicode_AsUTF8(name);
  if (field_name == nullptr) {
    PyErr_Clear();
    field_name = "";
  }

  Register* receiver = tc.frame.stack.pop();
  Register* value = tc.frame.stack.pop();
  Register* previous = temps_.AllocateStack();
  if (type <= TPrimitive) {
    Register* converted = temps_.AllocateStack();
    tc.emitLoadConst(previous, TNullptr);
    tc.emitIntConvert(converted, value, type);
    value = converted;
  } else {
    tc.emitLoadField(previous, receiver, field_name, offset, type, false);
  }
  tc.emit<StoreField>(receiver, field_name, offset, value, type, previous);
}

void HIRBuilder::emitCast(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  auto const& preloaded_type = preloader_.preloadedType(constArg(bc_instr));
  Register* value = tc.frame.stack.pop();
  Register* result = temps_.AllocateStack();
  tc.emit<Cast>(
      result,
      value,
      preloaded_type.type,
      preloaded_type.optional,
      preloaded_type.exact,
      tc.frame);
  tc.frame.stack.push(result);
}

void HIRBuilder::emitTpAlloc(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  auto pytype = preloader_.pyType(constArg(bc_instr));

  Register* result = temps_.AllocateStack();
  tc.emit<TpAlloc>(result, pytype, tc.frame);
  tc.frame.stack.push(result);
}

void HIRBuilder::emitImportFrom(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  auto& stack = tc.frame.stack;
  Register* name = stack.top();
  Register* res = temps_.AllocateStack();
  tc.emitImportFrom(res, name, bc_instr.oparg(), tc.frame);
  stack.push(res);
}

// Adjusts the oparg for import name to be the name index.
int importNameIdx(int oparg) {
  if constexpr (PY_VERSION_HEX >= 0x030F0000) {
    return oparg >> 2;
  } else {
    return oparg;
  }
}

void HIRBuilder::emitImportName(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  auto& stack = tc.frame.stack;
  Register* fromlist = stack.pop();
  Register* level = stack.pop();
  Register* res = temps_.AllocateStack();
  if (bc_instr.opcode() == EAGER_IMPORT_NAME) {
    tc.emit<EagerImportName>(res, bc_instr.oparg(), fromlist, level, tc.frame);
  } else {
    tc.emit<ImportName>(
        res, importNameIdx(bc_instr.oparg()), fromlist, level, tc.frame);
  }
  stack.push(res);
}

void HIRBuilder::emitRaiseVarargs(TranslationContext& tc) {
  tc.emitRaise(tc.frame);
}

void HIRBuilder::emitYieldFrom(TranslationContext& tc, Register* out) {
  auto& stack = tc.frame.stack;
  auto send_value = stack.pop();
  auto iter = stack.top();
  if (code_->co_flags & CO_COROUTINE) {
    tc.emitSetCurrentAwaiter(iter);
  }
  tc.emitYieldFrom(out, send_value, iter, tc.frame);
  stack.pop();
  stack.push(out);
}

void HIRBuilder::emitYieldValue(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  auto& stack = tc.frame.stack;
  auto in = stack.pop();
  auto out = temps_.AllocateStack();
  if (code_->co_flags & CO_ASYNC_GENERATOR) {
    tc.emitChecked<CallCFunc>(
        1,
        out,
        CallCFunc::Func::kCix_PyAsyncGenValueWrapperNew,
        std::vector<Register*>{in});
    in = out;
    out = temps_.AllocateStack();
  }
  if constexpr (PY_VERSION_HEX < 0x030C0000) {
    advancePastYieldInstr(tc);
    tc.emitYieldValue(out, in, tc.frame);
  } else if constexpr (PY_VERSION_HEX < 0x030E0000) {
    auto next_bc =
        BytecodeInstruction{code_, tc.frame.cur_instr_offs}.nextInstr();

    // This mirrors what _PyGen_yf() does. I assume the RESUME oparg exists
    // primarily for this check - values 2 and 3 indicate a "yield from" and
    // "await" respectively.
    if (next_bc.opcode() == RESUME && next_bc.oparg() >= 2) {
      tc.emitYieldFrom(out, in, stack.top(), tc.frame);
    } else {
      tc.emitYieldValue(out, in, tc.frame);
    }
  } else {
    advancePastYieldInstr(tc);
    if (bc_instr.oparg() == 1) {
      tc.emitYieldFrom(out, in, stack.top(), tc.frame);
    } else {
      JIT_CHECK(bc_instr.oparg() == 0, "Invalid oparg {}", bc_instr.oparg());
      tc.emitYieldValue(out, in, tc.frame);
    }
  }
  stack.push(out);
}

static std::pair<bool, bool> checkAsyncWithError(
    const BytecodeInstructionBlock& bc_instrs,
    BytecodeInstruction bc_instr) {
  bool error_aenter = false;
  bool error_aexit = false;
  if constexpr (PY_VERSION_HEX < 0x030C0000) {
    BCIndex idx = bc_instr.baseIndex();
    int prev_prev_op = idx > 1 ? bc_instrs.at(idx - 2).opcode() : 0;
    int prev_op = idx != 0 ? bc_instrs.at(idx - 1).opcode() : 0;
    if (prev_op == BEFORE_ASYNC_WITH) {
      error_aenter = true;
    } else if (
        prev_op == WITH_EXCEPT_START ||
        (prev_op == CALL_FUNCTION && prev_prev_op == DUP_TOP)) {
      error_aexit = true;
    }
  } else {
    error_aenter = bc_instr.oparg() == 1;
    error_aexit = bc_instr.oparg() == 2;
  }
  return std::make_pair(error_aenter, error_aexit);
}

void HIRBuilder::emitGetAwaitable(
    CFG& cfg,
    TranslationContext& tc,
    const BytecodeInstructionBlock& bc_instrs,
    BytecodeInstruction bc_instr) {
  OperandStack& stack = tc.frame.stack;
  Register* iterable = stack.pop();
  Register* iter = temps_.AllocateStack();

  // Most work is done by existing JitPyCoro_GetAwaitableIter() utility.
  tc.emit<CallCFunc>(
      1,
      iter,
#if PY_VERSION_HEX >= 0x030C0000
      CallCFunc::Func::kJitCoro_GetAwaitableIter,
#else
      CallCFunc::Func::kCix_PyCoro_GetAwaitableIter,
#endif
      std::vector<Register*>{iterable});

  auto [error_aenter, error_aexit] = checkAsyncWithError(bc_instrs, bc_instr);
  if (error_aenter || error_aexit) {
    BasicBlock* error_block = cfg.AllocateBlock();
    BasicBlock* ok_block = cfg.AllocateBlock();
    tc.emitCondBranch(iter, ok_block, error_block);
    tc.block = error_block;
    Register* type = temps_.AllocateStack();
    tc.emitLoadField(
        type, iterable, "ob_type", offsetof(PyObject, ob_type), TType);
    tc.emit<RaiseAwaitableError>(type, error_aenter, tc.frame);

    tc.block = ok_block;
    // TASK(T105038867): Remove once we have RefineTypeInsertion
    tc.emitRefineType(iter, TObject, iter);
  } else {
    tc.emitCheckExc(iter, iter, tc.frame);
  }

  // For coroutines only, runtime assert it isn't already awaiting by checking
  // if it has a sub-iterator using *Gen_yf().
  BasicBlock* block_assert_not_awaited_coro = cfg.AllocateBlock();
  BasicBlock* block_done = cfg.AllocateBlock();
#if PY_VERSION_HEX >= 0x030C0000
  BasicBlock* block_check_coro = cfg.AllocateBlock();
  tc.emitCondBranchCheckType(
      iter,
      Type::fromTypeExact(cinderx::getModuleState()->coroType()),
      block_assert_not_awaited_coro,
      block_check_coro);
  tc.block = block_check_coro;
#endif
  tc.emitCondBranchCheckType(
      iter,
      Type::fromTypeExact(&PyCoro_Type),
      block_assert_not_awaited_coro,
      block_done);
  Register* yf = temps_.AllocateStack();
  tc.block = block_assert_not_awaited_coro;
  tc.emit<CallCFunc>(
      1,
      yf,
#if PY_VERSION_HEX >= 0x030C0000
      CallCFunc::Func::kJitGen_yf,
#else
      CallCFunc::Func::kCix_PyGen_yf,
#endif
      std::vector<Register*>{iter});
  BasicBlock* block_coro_already_awaited = cfg.AllocateBlock();
  tc.emitCondBranch(yf, block_coro_already_awaited, block_done);
  tc.block = block_coro_already_awaited;
  tc.emit<RaiseStatic>(
      0, PyExc_RuntimeError, "coroutine is being awaited already", tc.frame);

  stack.push(iter);

  tc.block = block_done;
}

void HIRBuilder::emitBuildString(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  auto num_operands = bc_instr.oparg();
  tc.emitVariadic<BuildString>(temps_, num_operands);
}

void HIRBuilder::emitFormatValue(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  auto oparg = bc_instr.oparg();

  int have_fmt_spec = (oparg & FVS_MASK) == FVS_HAVE_SPEC;
  Register* fmt_spec;
  if (have_fmt_spec) {
    fmt_spec = tc.frame.stack.pop();
  } else {
    fmt_spec = temps_.AllocateStack();
    tc.emitLoadConst(fmt_spec, TNullptr);
  }
  Register* value = tc.frame.stack.pop();
  Register* dst = temps_.AllocateStack();
  int which_conversion = oparg & FVC_MASK;

  tc.emit<FormatValue>(dst, fmt_spec, value, which_conversion, tc.frame);
  tc.frame.stack.push(dst);
}

void HIRBuilder::emitFormatWithSpec(TranslationContext& tc) {
  OperandStack& stack = tc.frame.stack;
  Register* fmt_spec = stack.pop();
  Register* value = stack.pop();
  Register* out = temps_.AllocateStack();
  tc.emitFormatWithSpec(out, value, fmt_spec, tc.frame);
  stack.push(out);
}

void HIRBuilder::emitMapAdd(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  auto oparg = bc_instr.oparg();
  auto& stack = tc.frame.stack;
  auto value = stack.pop();
  auto key = stack.pop();

  auto map = stack.peek(oparg);

  auto result = temps_.AllocateStack();
  tc.emitSetDictItem(result, map, key, value, tc.frame);
}

void HIRBuilder::emitSetAdd(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  auto oparg = bc_instr.oparg();
  auto& stack = tc.frame.stack;

  auto* v = stack.pop();
  auto* set = stack.peek(oparg);

  auto result = temps_.AllocateStack();
  tc.emitSetSetItem(result, set, v, tc.frame);
}

void HIRBuilder::emitSetUpdate(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  auto oparg = bc_instr.oparg();
  auto& stack = tc.frame.stack;
  auto* iterable = stack.pop();
  auto* set = stack.peek(oparg);
  auto result = temps_.AllocateStack();
  tc.emitSetUpdate(result, set, iterable, tc.frame);
}

void HIRBuilder::emitDispatchEagerCoroResult(
    CFG& cfg,
    TranslationContext& tc,
    Register* out,
    BasicBlock* await_block,
    BasicBlock* post_await_block) {
  Register* stack_top = tc.frame.stack.top();

  TranslationContext has_wh_block{cfg.AllocateBlock(), tc.frame};
  tc.emitCondBranchCheckType(
      stack_top, TWaitHandle, has_wh_block.block, await_block);

  Register* wait_handle = stack_top;
  Register* wh_coro_or_result = temps_.AllocateStack();
  Register* wh_waiter = temps_.AllocateStack();
  has_wh_block.emitWaitHandleLoadCoroOrResult(wh_coro_or_result, wait_handle);
  has_wh_block.emitWaitHandleLoadWaiter(wh_waiter, wait_handle);
  has_wh_block.emitWaitHandleRelease(wait_handle);

  TranslationContext coro_block{cfg.AllocateBlock(), tc.frame};
  TranslationContext res_block{cfg.AllocateBlock(), tc.frame};
  has_wh_block.emitCondBranch(wh_waiter, coro_block.block, res_block.block);

  if (code_->co_flags & CO_COROUTINE) {
    coro_block.emitSetCurrentAwaiter(wh_coro_or_result);
  }
  coro_block.emit<YieldAndYieldFrom>(
      out, wh_waiter, wh_coro_or_result, tc.frame);
  coro_block.emitBranch(post_await_block);

  res_block.emitAssign(out, wh_coro_or_result);
  res_block.emitBranch(post_await_block);
}

void HIRBuilder::emitMatchMappingSequence(
    CFG& cfg,
    TranslationContext& tc,
    uint64_t tf_flag) {
  Register* top = tc.frame.stack.top();
  auto type = temps_.AllocateStack();
  tc.emitLoadField(type, top, "ob_type", offsetof(PyObject, ob_type), TType);
  auto tp_flags = temps_.AllocateStack();
  tc.emitLoadField(
      tp_flags, type, "tp_flags", offsetof(PyTypeObject, tp_flags), TCUInt64);
  auto flag = temps_.AllocateStack();
  tc.emitLoadConst(flag, Type::fromCUInt(tf_flag, TCUInt64));

  auto and_result = temps_.AllocateStack();
  tc.emitIntBinaryOp(and_result, BinaryOpKind::kAnd, tp_flags, flag);

  auto true_block = cfg.AllocateBlock();
  auto false_block = cfg.AllocateBlock();
  tc.emitCondBranch(and_result, true_block, false_block);

  auto result = temps_.AllocateStack();
  tc.block = true_block;
  tc.emitLoadConst(result, Type::fromObject(Py_True));
  auto done = cfg.AllocateBlock();
  tc.emitBranch(done);

  tc.block = false_block;
  tc.emitLoadConst(result, Type::fromObject(Py_False));
  tc.emitBranch(done);

  tc.block = done;

  tc.frame.stack.push(result);
}

void HIRBuilder::emitMatchClass(
    CFG& cfg,
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  auto& stack = tc.frame.stack;
  Register* names = stack.pop();
  Register* type = stack.pop();
  Register* subject = stack.pop();
  auto oparg = bc_instr.oparg();

  auto nargs = temps_.AllocateStack();
  tc.emitLoadConst(nargs, Type::fromCUInt(oparg, TCUInt64));

  auto attrs_tuple = temps_.AllocateStack();
  tc.emit<MatchClass>(attrs_tuple, subject, type, nargs, names);
  tc.emitRefineType(attrs_tuple, TOptTupleExact, attrs_tuple);

  Register* tuple_or_none = temps_.AllocateStack();
  stack.push(tuple_or_none);
  Register* if_success = nullptr;
  if constexpr (PY_VERSION_HEX < 0x030C0000) {
    if_success = temps_.AllocateStack();
    stack.push(if_success);
  }

  auto true_block = cfg.AllocateBlock();
  auto false_block = cfg.AllocateBlock();
  auto done = cfg.AllocateBlock();

  tc.emitCondBranch(attrs_tuple, true_block, false_block);
  tc.block = true_block;
  tc.emitRefineType(tuple_or_none, TTupleExact, attrs_tuple);
  if constexpr (PY_VERSION_HEX < 0x030C0000) {
    tc.emitLoadConst(if_success, Type::fromObject(Py_True));
  }
  tc.emitBranch(done);

  tc.block = false_block;
  tc.emitCheckErrOccurred(tc.frame);
  if constexpr (PY_VERSION_HEX < 0x030C0000) {
    tc.emitLoadConst(if_success, Type::fromObject(Py_False));
    tc.emitAssign(tuple_or_none, subject);
  } else {
    Register* none = temps_.AllocateNonStack();
    tc.emitLoadConst(none, Type::fromObject(Py_None));
    tc.emitAssign(tuple_or_none, none);
  }
  tc.emitBranch(done);

  tc.block = done;
}

void HIRBuilder::emitMatchKeys(CFG& cfg, TranslationContext& tc) {
  auto& stack = tc.frame.stack;
  Register* keys = stack.top();
  Register* subject = stack.top(1);

  auto values_or_none = temps_.AllocateStack();
  tc.emit<MatchKeys>(values_or_none, subject, keys, tc.frame);
  stack.push(values_or_none);

  auto none = temps_.AllocateStack();
  tc.emitLoadConst(none, Type::fromObject(Py_None));
  auto is_none = temps_.AllocateStack();
  tc.emitPrimitiveCompare(
      is_none, PrimitiveCompareOp::kEqual, values_or_none, none);

  auto true_block = cfg.AllocateBlock();
  auto false_block = cfg.AllocateBlock();
  auto done = cfg.AllocateBlock();

  tc.emitCondBranch(is_none, true_block, false_block);
  Register* if_success = nullptr;
  if constexpr (PY_VERSION_HEX < 0x030C0000) {
    if_success = temps_.AllocateStack();
  }
  tc.block = true_block;
  tc.emitRefineType(values_or_none, TNoneType, values_or_none);
  if constexpr (PY_VERSION_HEX < 0x030C0000) {
    tc.emitLoadConst(if_success, Type::fromObject(Py_False));
  }
  tc.emitBranch(done);

  tc.block = false_block;
  tc.emitRefineType(values_or_none, TTupleExact, values_or_none);
  if constexpr (PY_VERSION_HEX < 0x030C0000) {
    tc.emitLoadConst(if_success, Type::fromObject(Py_True));
  }
  tc.emitBranch(done);
  if constexpr (PY_VERSION_HEX < 0x030C0000) {
    stack.push(if_success);
  }
  tc.block = done;
}

void HIRBuilder::emitDictUpdate(
    TranslationContext& tc,
    const BytecodeInstruction& bc_instr) {
  auto& stack = tc.frame.stack;
  Register* update = stack.pop();
  Register* dict = stack.top(bc_instr.oparg() - 1);
  Register* out = temps_.AllocateStack();
  tc.emitDictUpdate(out, dict, update, tc.frame);
}

void HIRBuilder::emitDictMerge(
    TranslationContext& tc,
    const BytecodeInstruction& bc_instr) {
  auto& stack = tc.frame.stack;
  Register *dict, *func;
  if constexpr (PY_VERSION_HEX < 0x030E0000) {
    dict = stack.top(bc_instr.oparg());
    func = stack.top(bc_instr.oparg() + 2);
  } else {
    // According to bytecodes.c, at this point on the stack we have:
    //  update (top of the stack)
    //  [unused if oparg is 0]
    //  dict
    //  unused
    //  unused
    //  callable
    // Looking at codegen.c for 3.14, oparg is only ever 1 so the optional
    // "unused" slot is never present. So the 1 and 4 offsets skip to "dict" and
    // "callable" respectively.
    JIT_CHECK(bc_instr.oparg() == 1, "oparg must be 1");
    dict = stack.top(1);
    func = stack.top(4);
  }
  Register* update = stack.pop();
  Register* out = temps_.AllocateStack();
  tc.emitDictMerge(out, dict, update, func, tc.frame);
}

void HIRBuilder::emitSend(
    TranslationContext& tc,
    const BytecodeInstruction& bc_instr) {
  OperandStack& stack = tc.frame.stack;
  Register* value_out = stack.pop();
  Register* iter = stack.top();
  Register* value_in = temps_.AllocateStack();
  tc.emitSend(iter, value_out, value_in, tc.frame);
  Register* is_done = temps_.AllocateNonStack();
  tc.emitGetSecondOutput(is_done, TCInt64, value_in);
  stack.push(value_in);
  BasicBlock* done_block = getBlockAtOff(bc_instr.getJumpTarget());
  BasicBlock* continue_block = getBlockAtOff(bc_instr.nextInstrOffset());
  tc.emitCondBranch(is_done, done_block, continue_block);
}

void HIRBuilder::emitBuildInterpolation(
    [[maybe_unused]] TranslationContext& tc,
    [[maybe_unused]] const jit::BytecodeInstruction& bc_instr) {
#if PY_VERSION_HEX >= 0x030E0000
  OperandStack& stack = tc.frame.stack;
  auto oparg = bc_instr.oparg();
  int conversion = oparg >> 2;

  Register* format;
  if (oparg & 1) {
    format = stack.pop();
  } else {
    PyObject* empty = &_Py_STR(empty);
    format = temps_.AllocateStack();
    tc.emitLoadConst(format, Type::fromObject(empty));
  }

  Register* str = stack.pop();
  Register* value = stack.pop();
  Register* out = temps_.AllocateStack();
  tc.emit<BuildInterpolation>(out, value, str, format, conversion, tc.frame);
  stack.push(out);
#endif
}

void HIRBuilder::emitBuildTemplate(TranslationContext& tc) {
  OperandStack& stack = tc.frame.stack;
  Register* interpolations = stack.pop();
  Register* strings = stack.pop();
  Register* out = temps_.AllocateStack();
  tc.emit<BuildTemplate>(strings, interpolations, out, tc.frame);
  stack.push(out);
}

void HIRBuilder::emitConvertValue(
    TranslationContext& tc,
    const jit::BytecodeInstruction& bc_instr) {
  OperandStack& stack = tc.frame.stack;
  Register* value = stack.pop();
  Register* out = temps_.AllocateStack();
  tc.emitConvertValue(out, value, bc_instr.oparg(), tc.frame);
  stack.push(out);
}

void HIRBuilder::emitFormatSimple(CFG& cfg, TranslationContext& tc) {
  OperandStack& stack = tc.frame.stack;
  Register* value = stack.pop();

  BasicBlock* done_block = cfg.AllocateBlock();
  BasicBlock* do_fmt_block = cfg.AllocateBlock();
  BasicBlock* pass_through_block = cfg.AllocateBlock();

  tc.emitCondBranchCheckType(
      value, TUnicodeExact, pass_through_block, do_fmt_block);
  Register* out = temps_.AllocateStack();

  tc.block = do_fmt_block;
  Register* fmt_spec = temps_.AllocateStack();
  tc.emitLoadConst(fmt_spec, TNullptr);
  tc.emitFormatWithSpec(out, value, fmt_spec, tc.frame);
  tc.emitBranch(done_block);

  tc.block = pass_through_block;
  tc.emitRefineType(out, TUnicodeExact, value);
  tc.emitBranch(done_block);

  tc.block = done_block;
  stack.push(out);
}

void HIRBuilder::emitLoadCommonConstant(
    TranslationContext& tc,
    const BytecodeInstruction& bc_instr) {
  Register* out = temps_.AllocateStack();
  tc.emitLoadConst(
      out, getContext()->typeForCommonConstant(bc_instr.oparg()));
  tc.frame.stack.push(out);
}

void HIRBuilder::emitLoadSpecial(
    TranslationContext& tc,
    const BytecodeInstruction& bc_instr) {
  OperandStack& stack = tc.frame.stack;
  Register* self = stack.pop();
  Register* method = temps_.AllocateStack();
  Register* null_or_self = temps_.AllocateStack();
  tc.emit<LoadSpecial>(method, self, bc_instr.oparg(), tc.frame);
  tc.emitGetSecondOutput(null_or_self, TOptObject, method);
  stack.push(method);
  stack.push(null_or_self);
}

void HIRBuilder::emitSetFunctionAttribute(
    TranslationContext& tc,
    const BytecodeInstruction& bc_instr) {
  OperandStack& stack = tc.frame.stack;
  Register* func = stack.pop();
  Register* value = stack.pop();

  // Map the bytecode oparg to FunctionAttr enum
  FunctionAttr attr;
  switch (bc_instr.oparg()) {
    case MAKE_FUNCTION_DEFAULTS:
      attr = FunctionAttr::kDefaults;
      break;
    case MAKE_FUNCTION_KWDEFAULTS:
      attr = FunctionAttr::kKwDefaults;
      break;
    case MAKE_FUNCTION_ANNOTATIONS:
      attr = FunctionAttr::kAnnotations;
      break;
    case MAKE_FUNCTION_CLOSURE:
      attr = FunctionAttr::kClosure;
      break;
#if PY_VERSION_HEX >= 0x030E0000
    case MAKE_FUNCTION_ANNOTATE:
      attr = FunctionAttr::kAnnotate;
      break;
#endif
    default:
      JIT_ABORT(
          "Unsupported SET_FUNCTION_ATTRIBUTE oparg: {}", bc_instr.oparg());
  }

  tc.emitSetFunctionAttr(value, func, attr);
  stack.push(func);
}

void HIRBuilder::emitLoadBuildClass(TranslationContext& tc) {
  Register* result = temps_.AllocateStack();
  Register* builtins = temps_.AllocateNonStack();
  Register* key = temps_.AllocateNonStack();
  tc.emitLoadConst(builtins, Type::fromObject(tc.frame.builtins));
  // Starting at the preloader the JIT seems to assume builtins will be a
  // dictionary, however I'm not sure there's any guarantee of this.
  Register* builtins_dict = temps_.AllocateNonStack();
  tc.emitGuardType(builtins_dict, TDictExact, builtins, tc.frame);
  tc.emitLoadConst(key, Type::fromObject(getContext()->strBuildClass()));
  tc.emitDictSubscr(result, builtins_dict, key, tc.frame);
  tc.frame.stack.push(result);
}

void HIRBuilder::emitStoreGlobal(
    TranslationContext& tc,
    const BytecodeInstruction& bc_instr) {
  Register* globals = temps_.AllocateNonStack();
  Register* key = temps_.AllocateNonStack();

  tc.emitLoadConst(globals, Type::fromObject(tc.frame.globals));
  // Starting at the preloader the JIT seems to assume globals will be a
  // dictionary, however I'm not sure there's any guarantee of this.
  Register* globals_dict = temps_.AllocateNonStack();
  tc.emitGuardType(globals_dict, TDictExact, globals, tc.frame);
  tc.emitLoadConst(
      key,
      Type::fromObject(PyTuple_GET_ITEM(code_->co_names, bc_instr.oparg())));
  Register* value = tc.frame.stack.pop();
  Register* result = temps_.AllocateNonStack();
  tc.emitSetDictItem(result, globals_dict, key, value, tc.frame);
}

void HIRBuilder::insertRunPeriodicActivites(
    CFG& cfg,
    BasicBlock* check_block,
    BasicBlock* succ,
    const FrameState& frame) {
  TranslationContext check(check_block, frame);
  TranslationContext body(cfg.AllocateBlock(), frame);
#ifdef Py_GIL_DISABLED
  check.emitAtQuiescentState();
#endif
  // Check if the eval breaker has been set
  Register* eval_breaker = temps_.AllocateStack();
  check.emitLoadEvalBreaker(eval_breaker);
  check.emitCondBranch(eval_breaker, body.block, succ);
  // If set, run periodic tasks
  body.emitSnapshot();
  body.emitRunPeriodicTasks(temps_.AllocateStack(), body.frame);
  body.emitBranch(succ);
}

void HIRBuilder::insertRunPeriodicActivitesForLoop(
    CFG& cfg,
    BasicBlock* loop_header) {
  auto snap = loop_header->entrySnapshot();
  JIT_CHECK(snap != nullptr, "block {} has no entry snapshot", loop_header->id);
  auto fs = snap->frameState();
  JIT_CHECK(
      fs != nullptr,
      "entry snapshot for block {} has no FrameState",
      loop_header->id);
  auto check_block = cfg.AllocateBlock();
  loop_header->retargetPreds(check_block);
  insertRunPeriodicActivites(cfg, check_block, loop_header, *fs);
}

void HIRBuilder::insertRunPeriodicActivitesForExcept(
    CFG& cfg,
    TranslationContext& tc) {
  TranslationContext succ(cfg.AllocateBlock(), tc.frame);
  succ.emitSnapshot();
  insertRunPeriodicActivites(cfg, tc.block, succ.block, tc.frame);
  tc.block = succ.block;
}

ExecutionBlock HIRBuilder::popBlock(CFG& cfg, TranslationContext& tc) {
  if (tc.frame.block_stack.top().opcode == SETUP_FINALLY) {
    insertRunPeriodicActivitesForExcept(cfg, tc);
  }
  return tc.frame.block_stack.pop();
}

BorrowedRef<> HIRBuilder::constArg(const BytecodeInstruction& bc_instr) {
  return PyTuple_GET_ITEM(code_->co_consts, bc_instr.oparg());
}

void HIRBuilder::checkTranslate() {
  PyObject* names = code_->co_names;
  std::unordered_set<Py_ssize_t> banned_name_ids;
  auto name_at = [&](Py_ssize_t i) {
    return std::string_view(PyUnicode_AsUTF8(PyTuple_GET_ITEM(names, i)));
  };
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(names); i++) {
    if (isBannedName(name_at(i))) {
      banned_name_ids.insert(i);
    }
  }
  for (auto& bci : BytecodeInstructionBlock{code_}) {
    auto opcode = bci.opcode();
    int oparg = bci.oparg();
    if (!isSupportedOpcode(opcode)) {
      throw std::runtime_error{fmt::format(
          "Cannot compile {} to HIR because it contains unsupported opcode {} "
          "({})",
          preloader_.fullname(),
          opcode,
          opcodeName(opcode))};
    } else if (opcode == LOAD_GLOBAL) {
      if constexpr (PY_VERSION_HEX >= 0x030B0000) {
        if ((oparg & 0x01) && name_at(oparg >> 1) == "super") {
          // LOAD_GLOBAL NULL + super, super isn't being used with a
          // LOAD_SUPER_ATTR.
          throw std::runtime_error{fmt::format(
              "Cannot compile {} to HIR because it uses super() without an "
              "attribute or method after it",
              preloader_.fullname())};
        }
        oparg = oparg >> 1;
      }
      if (banned_name_ids.contains(oparg)) {
        throw std::runtime_error{fmt::format(
            "Cannot compile {} to HIR because it uses banned global '{}'",
            preloader_.fullname(),
            name_at(oparg))};
      }
    }
  }
}

} // namespace jit::hir

