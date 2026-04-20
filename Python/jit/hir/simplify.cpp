// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cinderx/Jit/hir/simplify.h"
#include "cinderx/Jit/hir/simplify_c.h"
#include "cinderx/Jit/jit_config_c.h"
#include "cinderx/Jit/hir/hir_type_c.h"
#include "cinderx/Jit/hir/hir_c_api.h"
#include "cinderx/Jit/hir/hir_instr_c.h"

#include "pycore_long.h"

#include <string.h>

#include "cinderx/Common/dict.h"
#include "cinderx/Common/log.h"
#include "cinderx/Common/property.h"
#include "cinderx/Common/py-portability.h"
#include "cinderx/Common/type.h"
#include "cinderx/Jit/context.h"
#include "cinderx/Jit/global_deopt_patcher.h"
#include "cinderx/Jit/hir/analysis.h"
#include "cinderx/Jit/hir/clean_cfg.h"
#include "cinderx/Jit/hir/copy_propagation.h"
#include "cinderx/Jit/hir/printer.h"
#include "cinderx/Jit/hir/type.h"
#include "cinderx/Jit/iterator_types.h"
#include "cinderx/Jit/jit_rt.h"
#include "cinderx/Jit/threaded_compile.h"
#include "cinderx/Jit/hir/preload.h"
#include "cinderx/StaticPython/strictmoduleobject.h"

#include <fmt/ostream.h>

namespace jit::hir {

/* Convert C++ Type to C HirType via field-by-field conversion. */
static inline HirType to_hir(Type t) {
  return Type::toHirType(t);
}

/* Wrappers for C query functions that take Type directly (avoids
 * type which is layout-dependent). */
static inline int type_could_be(Type a, Type b) {
  HirType ha = to_hir(a), hb = to_hir(b);
  return hir_type_could_be(&ha, &hb);
}
static inline int type_has_object_spec(Type t) {
  HirType h = to_hir(t);
  return hir_type_has_object_spec(&h);
}
static inline int type_has_int_spec(Type t) {
  HirType h = to_hir(t);
  return hir_type_has_int_spec(&h);
}
static inline PyObject* type_object_spec(Type t) {
  HirType h = to_hir(t);
  return hir_type_object_spec(&h);
}
static inline intptr_t type_int_spec(Type t) {
  HirType h = to_hir(t);
  return hir_type_int_spec(&h);
}
static inline PyObject* type_as_object(Type t) {
  HirType h = to_hir(t);
  return hir_type_as_object(&h);
}

// This file contains the Simplify pass, which is a collection of
// strength-reduction optimizations. An optimization should be added as a case
// in Simplify rather than a standalone pass if and only if it meets these
// criteria:
// - It operates on one instruction at a time, with no global analysis or
//   state.

// Convert InPlaceOpKind to the corresponding BinaryOpKind.
// InPlaceOpKind and BinaryOpKind share names but have different ordinals
// (BinaryOpKind has kSubscript at position 10, which InPlaceOpKind lacks).
static std::optional<BinaryOpKind> inPlaceOpToBinaryOp(InPlaceOpKind op) {
  switch (op) {
    case InPlaceOpKind::kAdd: return BinaryOpKind::kAdd;
    case InPlaceOpKind::kSubtract: return BinaryOpKind::kSubtract;
    case InPlaceOpKind::kMultiply: return BinaryOpKind::kMultiply;
    case InPlaceOpKind::kTrueDivide: return BinaryOpKind::kTrueDivide;
    case InPlaceOpKind::kFloorDivide: return BinaryOpKind::kFloorDivide;
    case InPlaceOpKind::kModulo: return BinaryOpKind::kModulo;
    case InPlaceOpKind::kPower: return BinaryOpKind::kPower;
    default: return std::nullopt;
  }
}
// - Optimizable instructions are replaced with 0 or more new instructions that
//   define an equivalent value while doing less work.
//
// To add support for a new instruction Foo, add a function simplifyFoo(Env&
// env, const Foo* instr) (env can be left out if you don't need it) containing
// the optimization and call it from a new case in
// simplifyInstr(). simplifyFoo() should analyze the given instruction, then do
// one of the following:
// - If the instruction is not optimizable, return nullptr and do not call any
//   functions on env.
// - If the instruction is redundant and can be elided, return the existing
//   value that should replace its output (this is often one of the
//   instruction's inputs).
// - If the instruction can be replaced with a cheaper sequence of
//   instructions, emit those instructions using env.emit<T>(...). For
//   instructions that define an output, emit<T> will allocate and return an
//   appropriately-typed Register* for you, to ease chaining multiple
//   instructions. As with the previous case, return the Register* that should
//   replace the current output of the instruction.
// - If the instruction can be elided but does not produce an output, set
//   env.optimized = true and return nullptr.
//
// Do not modify, unlink, or delete the existing instruction; all of those
// details are handled by existing code outside of the individual optimization
// functions.

namespace {

struct Env {
  explicit Env(Function& f)
      : func{f},
        type_object(
            Type::fromObject(reinterpret_cast<PyObject*>(&PyType_Type))) {}

  // The current function.
  Function& func;

  // The current block being emitted into. Might not be the block originally
  // containing the instruction being optimized, if more blocks have been
  // inserted by the simplify function.
  BasicBlock* block{nullptr};

  // Insertion cursor for new instructions. Must belong to block's Instr::List,
  // and except for brief critical sections during emit functions on Env,
  // should always point to the original, unoptimized instruction.
  Instr::List::iterator cursor;

  // Bytecode instruction of the instruction being optimized, automatically set
  // on all replacement instructions.
  BCOffset bc_off{-1};

  // Set to true by emit<T>() to indicate that the original instruction should
  // be removed.
  bool optimized{false};

  // The object that corresponds to "type".
  Type type_object{TTop};

  // Number of new basic blocks added by the simplifier.
  size_t new_blocks{0};

  // Create and insert the specified instruction. If the instruction has an
  // output, a new Register* will be created and returned.
  template <typename T, typename... Args>
  Register* emit(Args&&... args) {
    return emitInstr<T>(std::forward<Args>(args)...)->output();
  }

  // Similar to emit(), but returns the instruction itself. Useful for
  // instructions with no output, when you need to manipulate the instruction
  // after creation.
  template <typename T, typename... Args>
  T* emitInstr(Args&&... args) {
    if constexpr (T::has_output) {
      return emitRawInstr<T>(
          func.env.AllocateRegister(), std::forward<Args>(args)...);
    } else {
      return emitRawInstr<T>(std::forward<Args>(args)...);
    }
  }

  // Similar to emitRawInstr<T>(), but does not automatically create an output
  // Create and insert the specified instruction. If the instruction has an
  // output, a new Register* will be created and returned.
  template <typename T, typename... Args>
  Register* emitVariadic(std::size_t arity, Args&&... args) {
    if constexpr (T::has_output) {
      return emitRawInstr<T>(
                 arity,
                 func.env.AllocateRegister(),
                 std::forward<Args>(args)...)
          ->output();
    } else {
      return emitRawInstr<T>(arity, std::forward<Args>(args)...)->output();
    }
  }

  // Convenience: create + insert a LoadConst via pure C factory.
  Register* emitLoadConst(Type type) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_load_const(
        func.env.AllocateRegister(), to_hir(type))));
  }

  // Convenience: create + insert a GuardType via C factory bridge.
  Register* emitGuardType(Type target, Register* src, const FrameState* fs) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_guard_type(
        &func, to_hir(target), src, const_cast<void*>(static_cast<const void*>(fs)))));
  }

  // Convenience: create + insert a CheckExc via C factory bridge.
  Register* emitCheckExc(Register* src, const FrameState& fs) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_check_exc(
        &func, src, const_cast<void*>(static_cast<const void*>(&fs)))));
  }

  // Convenience: create + insert a Branch via C++ bridge factory.
  // Must use C++ bridge (not pure C) because Edge::set_to manages
  // target block's in_edges_ set.
  Register* emitBranch(BasicBlock* target) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_branch_cpp(target)));
  }

  // Convenience: create + insert a CondBranch via C++ bridge factory.
  // Must use C++ bridge (not pure C) because Edge::set_to manages
  // target block's in_edges_ set.
  Register* emitCondBranch(Register* cond, BasicBlock* true_bb, BasicBlock* false_bb) {
    return emitCInstr(static_cast<Instr*>(
        hir_c_create_cond_branch_cpp(cond, true_bb, false_bb)));
  }

  // Convenience: create + insert a UseType via pure C factory.
  Register* emitUseType(Register* val, Type type) {
    return emitCInstr(static_cast<Instr*>(
        hir_c_create_use_type(val, to_hir(type))));
  }

  // Batch of pure C factory helpers for non-DeoptBase, non-terminator instructions.
  Register* emitPrimitiveCompare(PrimitiveCompareOp op, Register* left, Register* right) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_primitive_compare(
        func.env.AllocateRegister(), static_cast<int32_t>(op), left, right)));
  }
  Register* emitIntBinaryOp(BinaryOpKind op, Register* left, Register* right) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_int_binary_op(
        func.env.AllocateRegister(), static_cast<int32_t>(op), left, right)));
  }
  Register* emitDoubleBinaryOp(BinaryOpKind op, Register* left, Register* right) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_double_binary_op(
        func.env.AllocateRegister(), static_cast<int32_t>(op), left, right)));
  }
  Register* emitPrimitiveUnaryOp(PrimitiveUnaryOpKind op, Register* src) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_primitive_unary_op(
        func.env.AllocateRegister(), static_cast<int32_t>(op), src)));
  }
  Register* emitFloatCompare(CompareOp op, Register* left, Register* right) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_float_compare(
        func.env.AllocateRegister(), static_cast<int32_t>(op), left, right)));
  }
  Register* emitLongCompare(CompareOp op, Register* left, Register* right) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_long_compare(
        func.env.AllocateRegister(), static_cast<int32_t>(op), left, right)));
  }
  Register* emitUnicodeCompare(CompareOp op, Register* left, Register* right) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_unicode_compare(
        func.env.AllocateRegister(), static_cast<int32_t>(op), left, right)));
  }
  Register* emitPrimitiveBoxBool(Register* src) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_primitive_box_bool(
        func.env.AllocateRegister(), src)));
  }
  Register* emitCIntToCBool(Register* src) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_cint_to_cbool(
        func.env.AllocateRegister(), src)));
  }
  Register* emitBitCast(Register* src, Type type) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_bit_cast(
        func.env.AllocateRegister(), src, to_hir(type))));
  }
  Register* emitRefineType(Type type, Register* src) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_refine_type(
        func.env.AllocateRegister(), to_hir(type), src)));
  }
  Register* emitPrimitiveUnbox(Register* src, Type type) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_primitive_unbox(
        func.env.AllocateRegister(), src, to_hir(type))));
  }
  Register* emitIndexUnbox(Register* src, PyObject* exc = PyExc_IndexError) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_index_unbox(
        func.env.AllocateRegister(), src, exc)));
  }

  // DeoptBase C++ bridge helpers.
  Register* emitFloatBinaryOp(BinaryOpKind op, Register* left, Register* right,
                               const FrameState& fs) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_float_binary_op(
        &func, static_cast<int32_t>(op), left, right,
        const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  Register* emitLongBinaryOp(BinaryOpKind op, Register* left, Register* right,
                              const FrameState& fs) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_long_binary_op(
        &func, static_cast<int32_t>(op), left, right,
        const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  Register* emitIsNegativeAndErrOccurred(Register* src, const FrameState& fs) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_is_neg_and_err(
        &func, src, const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  Register* emitLoadField(Register* receiver, const char* name,
                           std::size_t offset, Type type, bool borrowed = false) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_load_field(
        &func, receiver, name, static_cast<intptr_t>(offset),
        to_hir(type), borrowed ? 1 : 0)));
  }
  Register* emitGuardIs(PyObject* target, Register* src) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_guard_is(
        &func, target, src)));
  }
  Register* emitCheckNeg(Register* src, const FrameState& fs) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_check_neg(
        &func, src, const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  Register* emitPrimitiveBox(Register* src, Type type, const FrameState& fs) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_primitive_box(
        &func, src, to_hir(type),
        const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  Register* emitCheckSequenceBounds(Register* seq, Register* idx,
                                     const FrameState& fs) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_check_seq_bounds(
        &func, seq, idx,
        const_cast<void*>(static_cast<const void*>(&fs)))));
  }

  // Simple DeoptBase 2-operand + FrameState helpers.
  Register* emitDictSubscr(Register* lhs, Register* rhs, const FrameState& fs) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_dict_subscr(
        &func, lhs, rhs, const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  Register* emitUnicodeSubscr(Register* lhs, Register* rhs, const FrameState& fs) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_unicode_subscr(
        &func, lhs, rhs, const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  Register* emitUnicodeRepeat(Register* lhs, Register* rhs, const FrameState& fs) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_unicode_repeat(
        &func, lhs, rhs, const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  Register* emitUnicodeConcat(Register* lhs, Register* rhs, const FrameState& fs) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_unicode_concat(
        &func, lhs, rhs, const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  Register* emitListAppend(Register* list, Register* item, const FrameState& fs) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_list_append(
        &func, list, item, const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  Register* emitIsInstance(Register* obj, Register* type, const FrameState& fs) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_is_instance(
        &func, obj, type, const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  Register* emitGetLength(Register* src, const FrameState& fs) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_get_length(
        &func, src, const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  Register* emitLongInPlaceOp(InPlaceOpKind op, Register* left, Register* right,
                               const FrameState& fs) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_long_in_place_op(
        &func, static_cast<int32_t>(op), left, right,
        const_cast<void*>(static_cast<const void*>(&fs)))));
  }

  // Cache entry + scalar-field pure C factories.
  Register* emitLoadTypeAttrCacheEntryType(int cache_id) {
    return emitCInstr(static_cast<Instr*>(
        hir_c_create_load_type_attr_cache_entry_type(
            func.env.AllocateRegister(), cache_id)));
  }
  Register* emitLoadTypeAttrCacheEntryValue(int cache_id) {
    return emitCInstr(static_cast<Instr*>(
        hir_c_create_load_type_attr_cache_entry_value(
            func.env.AllocateRegister(), cache_id)));
  }
  Register* emitLoadTypeMethodCacheEntryType(int cache_id) {
    return emitCInstr(static_cast<Instr*>(
        hir_c_create_load_type_method_cache_entry_type(
            func.env.AllocateRegister(), cache_id)));
  }
  Register* emitLoadTypeMethodCacheEntryValue(int cache_id, Register* receiver) {
    return emitCInstr(static_cast<Instr*>(
        hir_c_create_load_type_method_cache_entry_value(
            func.env.AllocateRegister(), cache_id, receiver)));
  }
  Register* emitLoadSplitDictItem(Register* src, Py_ssize_t item_idx) {
    return emitCInstr(static_cast<Instr*>(
        hir_c_create_load_split_dict_item(
            func.env.AllocateRegister(), src, item_idx)));
  }

  // DeoptBaseWithNameIdx helpers.
  Register* emitLoadModuleMethodCached(Register* receiver, int name_idx,
                                        const FrameState& fs) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_load_module_method_cached(
        &func, receiver, name_idx,
        const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  Register* emitLoadMethodCached(Register* receiver, int name_idx,
                                  const FrameState& fs) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_load_method_cached(
        &func, receiver, name_idx,
        const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  Register* emitLoadModuleAttrCached(Register* receiver, int name_idx,
                                      const FrameState& fs) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_load_module_attr_cached(
        &func, receiver, name_idx,
        const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  Register* emitLoadAttrCached(Register* receiver, int name_idx,
                                const FrameState& fs) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_load_attr_cached(
        &func, receiver, name_idx,
        const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  Register* emitStoreAttrCached(Register* obj, Register* value, int name_idx,
                                 const FrameState& fs) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_store_attr_cached(
        &func, obj, value, name_idx,
        const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  Register* emitFillTypeAttrCache(Register* receiver, int name_idx,
                                   int cache_id, const FrameState& fs) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_fill_type_attr_cache(
        &func, receiver, name_idx, cache_id,
        const_cast<void*>(static_cast<const void*>(&fs)))));
  }
  Register* emitFillTypeMethodCache(Register* receiver, int name_idx,
                                     int cache_id, const FrameState& fs) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_fill_type_method_cache(
        &func, receiver, name_idx, cache_id,
        const_cast<void*>(static_cast<const void*>(&fs)))));
  }

  // Tier 3: CheckField (with guilty_reg), LoadAttr, LoadArrayItem.
  Register* emitCheckField(Register* src, PyObject* name, const FrameState& fs,
                            Register* guilty_reg = nullptr) {
    Register* out = emitCInstr(static_cast<Instr*>(hir_c_create_check_field(
        &func, src, name,
        const_cast<void*>(static_cast<const void*>(&fs)))));
    if (guilty_reg) {
      hir_c_set_guilty_reg(out->instr(), guilty_reg);
    }
    return out;
  }
  Register* emitLoadAttr(Register* receiver, int name_idx,
                          const FrameState& fs, bool already_optimized = false) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_load_attr(
        &func, receiver, name_idx,
        const_cast<void*>(static_cast<const void*>(&fs)),
        already_optimized ? 1 : 0)));
  }
  Register* emitLoadArrayItem(Register* arr, Register* idx, Register* container,
                               ssize_t offset, Type type) {
    return emitCInstr(static_cast<Instr*>(hir_c_create_load_array_item(
        &func, arr, idx, container, offset, to_hir(type))));
  }

  // Emit Guard via C bridge, returns instruction for post-creation mutation.
  Instr* emitGuard(Register* src) {
    Instr* guard = static_cast<Instr*>(hir_c_create_guard(src));
    emitCInstr(guard);
    return guard;
  }

  // Emit Snapshot via C bridge.
  Instr* emitSnapshot(const FrameState& fs) {
    Instr* snap = static_cast<Instr*>(hir_c_create_snapshot(
        const_cast<void*>(static_cast<const void*>(&fs))));
    emitCInstr(snap);
    return snap;
  }

  // Emit DeoptPatchpoint via C bridge, returns for post-creation mutation.
  Instr* emitDeoptPatchpoint(void* patcher) {
    Instr* pp = static_cast<Instr*>(hir_c_create_deopt_patchpoint(patcher));
    emitCInstr(pp);
    return pp;
  }

  // Emit VectorCall via C factory, returns instruction for operand wiring.
  // Output type is set to TObject (placeholder — depends on operands).
  VectorCall* emitVectorCallInstr(size_t n_operands, CallFlags flags,
                                   const FrameState& fs) {
    auto* instr = static_cast<VectorCall*>(static_cast<Instr*>(
        hir_c_create_vectorcall(
            &func, n_operands, static_cast<uint32_t>(flags),
            const_cast<void*>(static_cast<const void*>(&fs)))));
    optimized = true;
    instr->setBytecodeOffset(bc_off);
    block->insert(instr, cursor);
    // VectorCall output type depends on operands; use TObject placeholder.
    instr->output()->set_type(TObject);
    return instr;
  }

  // Emit CallStatic via C factory, returns instruction for operand wiring.
  CallStatic* emitCallStaticInstr(size_t n_operands, void* addr, Type ret_type) {
    auto* instr = static_cast<CallStatic*>(static_cast<Instr*>(
        hir_c_create_call_static(&func, n_operands, addr, to_hir(ret_type))));
    optimized = true;
    instr->setBytecodeOffset(bc_off);
    block->insert(instr, cursor);
    if (instr->output()) {
      instr->output()->set_type(outputType(*instr));
    }
    return instr;
  }

  // Emit CondBranch via C++ bridge, returns instruction (not register).
  // Needed for emitCondSlowPath which patches true_bb after creation.
  CondBranchBase* emitCondBranchInstr(Register* cond, BasicBlock* true_bb,
                                   BasicBlock* false_bb) {
    auto* instr = static_cast<CondBranchBase*>(static_cast<Instr*>(
        hir_c_create_cond_branch_cpp(cond, true_bb, false_bb)));
    optimized = true;
    instr->setBytecodeOffset(bc_off);
    block->insert(instr, cursor);
    return instr;
  }

  // Insert a pre-created instruction from a C factory into the current block.
  // Sets bytecode offset, inserts at cursor, computes output type.
  // Returns the output register (or nullptr if no output).
  Register* emitCInstr(Instr* instr) {
    optimized = true;
    instr->setBytecodeOffset(bc_off);
    block->insert(instr, cursor);
    if (instr->output()) {
      instr->output()->set_type(outputType(*instr));
    }
    return instr->output();
  }

  // Similar to emit<T>(), but does not automatically create an output
  // register.
  template <typename T, typename... Args>
  T* emitRawInstr(Args&&... args) {
    optimized = true;
    T* instr = T::create(std::forward<Args>(args)...);
    instr->setBytecodeOffset(bc_off);
    block->insert(instr, cursor);

    if constexpr (T::has_output) {
      Register* output = instr->output();
      switch (instr->opcode()) {
        case Opcode::kVectorCall:
          // We don't know the exact output type until its operands are
          // populated.
          output->set_type(TObject);
          break;
        default:
          output->set_type(outputType(*instr));
          break;
      }
    }

    return instr;
  }

  // Create and return a conditional value. Expects three callables:
  // - do_branch is given two BasicBlock* and should emit a conditional branch
  //   instruction using them.
  // - do_bb1 should emit code for the first successor, returning the computed
  //   value.
  // - do_bb2 should do the same for the second successor.
  template <typename BranchFn, typename Bb1Fn, typename Bb2Fn>
  Register* emitCond(BranchFn do_branch, Bb1Fn do_bb1, Bb2Fn do_bb2) {
    // bb1, bb2, and the new tail block that's split from the original.
    new_blocks += 3;

    BasicBlock* bb1 = func.cfg.AllocateBlock();
    BasicBlock* bb2 = func.cfg.AllocateBlock();
    do_branch(bb1, bb2);
    JIT_CHECK(
        cursor != block->begin(),
        "block should not be empty after calling do_branch()");
    BasicBlock* tail = func.cfg.splitAfter(*std::prev(cursor));

    block = bb1;
    cursor = bb1->end();
    Register* bb1_reg = do_bb1();
    emitBranch(tail);

    block = bb2;
    cursor = bb2->end();
    Register* bb2_reg = do_bb2();
    emitBranch(tail);

    block = tail;
    cursor = tail->begin();
    std::unordered_map<BasicBlock*, Register*> phi_srcs{
        {bb1, bb1_reg},
        {bb2, bb2_reg},
    };
    return emit<Phi>(phi_srcs);
  }

  // Create and return a conditional value that could go through a slow path if
  // it matches a certain condition. Expects two callables:
  //
  // - do_branch is given a BasicBlock* and it is expected that it will
  //   conditionally branch to that block if it needs to. The true_bb will be
  //   patched after the fast path is split. It should return the branch
  //   instruction so that it can be patched.
  // - do_slow_path should emit code for the slow path, returning the computed
  //   value.
  //
  // It is expected that the slow path will jump back to the default path at the
  // end of its block.
  template <typename BranchFn, typename SlowPathFn>
  Phi* emitCondSlowPath(
      Register* output,
      Register* previous_path_value,
      BranchFn do_branch,
      SlowPathFn do_slow_path) {
    new_blocks += 2;

    BasicBlock* previous_path = block;
    BasicBlock* slow_path = func.cfg.AllocateBlock();

    auto branch = do_branch(slow_path);
    BasicBlock* fast_path = func.cfg.splitAfter(*branch);
    branch->set_true_bb(fast_path);

    block = slow_path;
    cursor = slow_path->begin();
    auto slow_path_value = do_slow_path();
    emitBranch(fast_path);

    block = fast_path;
    cursor = fast_path->begin();
    std::unordered_map<BasicBlock*, Register*> args{
        {previous_path, previous_path_value},
        {slow_path, slow_path_value},
    };

    return emitRawInstr<Phi>(output, args);
  }
};

Register* simplifyCheck(const CheckBase* instr) {
  // These all check their input for null.
  if (instr->GetOperand(0)->isA(TObject)) {
    // No UseType is necessary because we never guard potentially-null values.
    return instr->GetOperand(0);
  }
  return nullptr;
}

Register* simplifyCheckSequenceBounds(
    Env& env,
    const Instr* instr) {
  Register* sequence = instr->GetOperand(0);
  Register* idx = instr->GetOperand(1);
  if (sequence->isA(TTupleExact) && sequence->instr()->IsMakeTuple() &&
      idx->isA(TCInt)) {
    HirType idx_hir = to_hir(idx->type());
    if (!hir_type_has_int_spec(&idx_hir)) return nullptr;
    size_t length = static_cast<const MakeTuple*>(sequence->instr())->nvalues();
    intptr_t idx_value = hir_type_int_spec(&idx_hir);
    bool adjusted = false;
    if (idx_value < 0) {
      idx_value += length;
      adjusted = true;
    }
    if (static_cast<size_t>(idx_value) < length) {
      env.emitUseType(sequence, sequence->type());
      env.emitUseType(idx, idx->type());
      if (adjusted) {
        return env.emitLoadConst(Type::fromCInt(idx_value, TCInt64));
      } else {
        return idx;
      }
    }
  }
  return nullptr;
}

Register* simplifyGuardType(Env& env, const GuardType* instr) {
  Register* input = instr->GetOperand(0);
  Type type = instr->target();
  if (input->isA(type)) {
    // We don't need a UseType: If an instruction cares about the type of this
    // GuardType's output, it will express that through its operand type
    // constraints. Once this GuardType is removed, those constraints will
    // apply to input's instruction rather than this GuardType, and any
    // downstream instructions will still be satisfied.
    return input;
  }
  if (type == TNoneType) {
    return env.emitGuardIs(Py_None, input);
  }
  return nullptr;
}

Register* simplifyRefineType(const RefineType* instr) {
  Register* input = instr->GetOperand(0);
  if (input->isA(instr->type())) {
    // No UseType for the same reason as GuardType above: RefineType itself
    // doesn't care about the input's type, only users of its output do, and
    // they're unchanged.
    return input;
  }
  return nullptr;
}

Register* simplifyCast(const Cast* instr) {
  Register* input = instr->GetOperand(0);
  Type type = instr->exact() ? Type::fromTypeExact(instr->pytype())
                             : Type::fromType(instr->pytype());
  if (instr->optional()) {
    type |= TNoneType;
  }
  if (input->isA(type)) {
    // No UseType for the same reason as GuardType above: Cast itself
    // doesn't care about the input's type, only users of its output do, and
    // they're unchanged.
    return input;
  }
  return nullptr;
}

Register* emitGetLengthInt64(Env& env, Register* obj) {
  Type ty = obj->type();
  if (
// TODO(T255264007). Enable this again. See P2169677410.
#ifndef Py_GIL_DISABLED
      ty <= TListExact || ty <= TArray ||
#endif
      ty <= TTupleExact) {
    HirType ty_hir_us = to_hir(ty);
    HirType ty_unspec = hir_type_unspecialized(&ty_hir_us);
    env.emitUseType(obj, Type::fromHirType(ty_unspec));
    return env.emitLoadField(
        obj, "ob_size", offsetof(PyVarObject, ob_size), TCInt64);
  }
  if (
// TODO(T255264007). Enable this again. See P2169677410.
#ifndef Py_GIL_DISABLED
      ty <= TDictExact || ty <= TSetExact ||
#endif
      ty <= TUnicodeExact) {
    std::size_t offset = 0;
    const char* name = nullptr;
    if (ty <= TDictExact) {
      offset = offsetof(PyDictObject, ma_used);
      name = "ma_used";
    } else if (ty <= TSetExact) {
      offset = offsetof(PySetObject, used);
      name = "used";
    } else if (ty <= TUnicodeExact) {
      // Note: In debug mode, the interpreter has an assert that ensures the
      // string is "ready", check PyUnicode_GET_LENGTH for strings.
      offset = offsetof(PyASCIIObject, length);
      name = "length";
    } else {
      JIT_ABORT("unexpected type");
    }
    HirType ty_hir_us2 = to_hir(ty);
    HirType ty_unspec2 = hir_type_unspecialized(&ty_hir_us2);
    env.emitUseType(obj, Type::fromHirType(ty_unspec2));
    return env.emitLoadField(obj, name, offset, TCInt64);
  }
  return nullptr;
}

Register* simplifyGetLength(Env& env, const Instr* instr) {
  Register* obj = instr->GetOperand(0);
  if (Register* size = emitGetLengthInt64(env, obj)) {
    return env.emitPrimitiveBox(size, TCInt64, *instr->asDeoptBase()->frameState());
  }
  return nullptr;
}

Register* simplifyIntConvert(Env& env, const IntConvert* instr) {
  Register* src = instr->GetOperand(0);
  if (src->isA(instr->type())) {
    env.emitUseType(src, instr->type());
    return instr->GetOperand(0);
  }
  return nullptr;
}

Register* simplifyCompare(Env& env, const Compare* instr) {
  Register* left = instr->GetOperand(0);
  Register* right = instr->GetOperand(1);
  CompareOp op = instr->op();

  if (left->isA(TNoneType) && right->isA(TNoneType)) {
    if (op == CompareOp::kEqual || op == CompareOp::kNotEqual) {
      env.emitUseType(left, TNoneType);
      env.emitUseType(right, TNoneType);
      return env.emitLoadConst(
          Type::fromObject(op == CompareOp::kEqual ? Py_True : Py_False));
    }
  }

  // Can compare booleans for equality with primitive operations.
  if (left->isA(TBool) && right->isA(TBool) &&
      (op == CompareOp::kEqual || op == CompareOp::kNotEqual)) {
    if (auto prim_op = toPrimitiveCompareOp(op)) {
      env.emitUseType(left, TBool);
      env.emitUseType(right, TBool);
      Register* result = env.emitPrimitiveCompare(*prim_op, left, right);
      return env.emitPrimitiveBoxBool(result);
    }
  }

  // Emit FloatCompare if both args are FloatExact and the op is supported
  // between two longs.
  if (left->isA(TFloatExact) && right->isA(TFloatExact) &&
      !(op == CompareOp::kIn || op == CompareOp::kNotIn ||
        op == CompareOp::kExcMatch)) {
    return env.emitFloatCompare(instr->op(), left, right);
  }

  // Emit LongCompare if both args are LongExact and the op is supported between
  // two longs.
  if (left->isA(TLongExact) && right->isA(TLongExact) &&
      !(op == CompareOp::kIn || op == CompareOp::kNotIn ||
        op == CompareOp::kExcMatch)) {
    return env.emitLongCompare(instr->op(), left, right);
  }

  // Emit UnicodeCompare if both args are UnicodeExact and the op is supported
  // between two strings.
  if (left->isA(TUnicodeExact) && right->isA(TUnicodeExact) &&
      !(op == CompareOp::kIn || op == CompareOp::kNotIn ||
        op == CompareOp::kExcMatch)) {
    return env.emitUnicodeCompare(instr->op(), left, right);
  }

  return nullptr;
}

Register* simplifyCondBranch(Env& env, const Instr* instr) {
  Register* cond = instr->GetOperand(0);
  Type cond_type = cond->type();
  HirType cond_hir = to_hir(cond_type);
  auto cb = static_cast<const CondBranchBase*>(instr);
  // Constant condition folds into an unconditional jump.
  if (hir_type_has_int_spec(&cond_hir)) {
    auto spec = hir_type_int_spec(&cond_hir);
    return env.emitBranch(spec ? cb->true_bb() : cb->false_bb());
  }
  // Common pattern of CondBranch getting its condition from an IntConvert,
  // which had been simplified down from an IsTruthy.  Can forward the value
  // only if it's being widened.  Narrowing an integer might change it from
  // non-zero to zero.
  if (cond->instr()->IsIntConvert()) {
    auto convert = static_cast<IntConvert*>(cond->instr());
    Register* src = convert->src();
    if (convert->type().sizeInBytes() >= src->type().sizeInBytes()) {
      return env.emitCondBranch(src, cb->true_bb(), cb->false_bb());
    }
  }
  return nullptr;
}

Register* simplifyCondBranchCheckType(
    Env& env,
    const CondBranchCheckType* instr) {
  Register* value = instr->GetOperand(0);
  Type actual_type = value->type();
  Type expected_type = instr->type();
  if (actual_type <= expected_type) {
    env.emitUseType(value, actual_type);
    return env.emitBranch(instr->true_bb());
  }
  HirType actual_hir = to_hir(actual_type);
  HirType expected_hir = to_hir(expected_type);
  if (!hir_type_could_be(&actual_hir, &expected_hir)) {
    env.emitUseType(value, actual_type);
    return env.emitBranch(instr->false_bb());
  }
  return nullptr;
}

Register* simplifyIsTruthy(Env& env, const Instr* instr) {
  Type ty = instr->GetOperand(0)->type();
  HirType ty_hir_it = to_hir(ty);
  PyObject* obj = hir_type_as_object(&ty_hir_it);
  if (obj != nullptr) {
    // Should only consider immutable Objects
    static const std::unordered_set<PyTypeObject*> kTrustedTypes{
        &PyBool_Type,
        &PyFloat_Type,
        &PyLong_Type,
        &PyFrozenSet_Type,
        &PySlice_Type,
        &PyTuple_Type,
        &PyUnicode_Type,
        Py_TYPE(Py_None),
    };
    if (kTrustedTypes.contains(Py_TYPE(obj))) {
      int res = PyObject_IsTrue(obj);
      JIT_CHECK(res >= 0, "PyObject_IsTrue failed on trusted type");
      // Since we no longer use instr->GetOperand(0), we need to make sure that
      // we don't lose any associated type checks
      env.emitUseType(instr->GetOperand(0), ty);
      return env.emitLoadConst(Type::fromCBool(res));
    }
  }
  if (ty <= TBool) {
    Register* left = instr->GetOperand(0);
    env.emitUseType(left, TBool);
    Register* right = env.emitLoadConst(Type::fromObject(Py_True));
    Register* result =
        env.emitPrimitiveCompare(PrimitiveCompareOp::kEqual, left, right);
    return result;
  }
  if (Register* size = emitGetLengthInt64(env, instr->GetOperand(0))) {
    return env.emitCIntToCBool(size);
  }
  if (ty <= TLongExact) {
    Register* left = instr->GetOperand(0);
    env.emitUseType(left, ty);
    Register* right = env.emitLoadConst(Type::fromObject(_PyLong_GetZero()));
    Register* result =
        env.emitPrimitiveCompare(PrimitiveCompareOp::kNotEqual, left, right);
    return result;
  }
  return nullptr;
}

Register* simplifyLoadTupleItem(Env& env, const LoadTupleItem* instr) {
  Register* src = instr->GetOperand(0);
  Type src_ty = src->type();
  HirType src_hir = to_hir(src_ty);
  if (!hir_type_has_value_spec(&src_hir, to_hir(TTuple))) {
    return nullptr;
  }
  env.emitUseType(src, src_ty);
  PyObject* item = PyTuple_GET_ITEM(hir_type_object_spec(&src_hir), instr->idx());
  return env.emitLoadConst(Type::fromObject(env.func.env.addReference(item)));
}

Register* simplifyLoadArrayItem(Env& env, const LoadArrayItem* instr) {
  Register* src = instr->seq();
  HirType idx_arr_hir = to_hir(instr->idx()->type());
  if (!hir_type_has_int_spec(&idx_arr_hir)) {
    return nullptr;
  }
  intptr_t idx_signed = hir_type_int_spec(&idx_arr_hir);
  JIT_CHECK(idx_signed >= 0, "LoadArrayItem should not have negative index");
  uintptr_t idx = static_cast<uintptr_t>(idx_signed);
  // We can only do this for tuples because lists and arrays, the other
  // sequence types, are mutable. A more general LoadElimination pass could
  // accomplish that, though.
  if (src->instr()->IsMakeTuple()) {
    size_t length = static_cast<const MakeTuple*>(src->instr())->nvalues();
    if (idx < length) {
      env.emitUseType(src, TTupleExact);
      env.emitUseType(instr->idx(), instr->idx()->type());
      return src->instr()->GetOperand(idx);
    }
  }
  HirType src_arr_hir = to_hir(src->type());
  if (hir_type_has_value_spec(&src_arr_hir, to_hir(TTupleExact))) {
    if (idx_signed < PyTuple_GET_SIZE(hir_type_object_spec(&src_arr_hir))) {
      env.emitUseType(src, src->type());
      env.emitUseType(instr->idx(), instr->idx()->type());
      PyObject* item = PyTuple_GET_ITEM(hir_type_object_spec(&src_arr_hir), idx);
      return env.emitLoadConst(
          Type::fromObject(env.func.env.addReference(item)));
    }
  }
  return nullptr;
}

Register* simplifyLoadVarObjectSize(Env& env, const Instr* instr) {
  Register* obj_reg = instr->GetOperand(0);
  Type type = obj_reg->type();
  // We can only do this for tuples because lists and arrays, the other
  // sequence types, are mutable. A more general LoadElimination pass could
  // accomplish that, though.
  if (obj_reg->instr()->IsMakeTuple()) {
    env.emitUseType(obj_reg, type);
    size_t size = static_cast<const MakeTuple*>(obj_reg->instr())->nvalues();
    Type output_type = instr->output()->type();
    return env.emitLoadConst(Type::fromCInt(size, output_type));
  }
  HirType type_hvs = to_hir(type);
  if (hir_type_has_value_spec(&type_hvs, to_hir(TTupleExact)) ||
      hir_type_has_value_spec(&type_hvs, to_hir(TBytesExact))) {
    PyVarObject* obj = reinterpret_cast<PyVarObject*>(hir_type_as_object(&type_hvs));
    Py_ssize_t size = obj->ob_size;
    env.emitUseType(obj_reg, type);
    Type output_type = instr->output()->type();
    return env.emitLoadConst(Type::fromCInt(size, output_type));
  }
  return nullptr;
}


Register* simplifyInPlaceOp(Env& env, const InPlaceOp* instr) {
  Register* lhs = instr->left();
  Register* rhs = instr->right();
  if (lhs->isA(TLongExact) && rhs->isA(TLongExact)) {
    // All binary ops on TLong's return mutable so can be freely simplified with
    // no explicit check.
    switch (instr->op()) {
      case InPlaceOpKind::kAdd:
      case InPlaceOpKind::kAnd:
      case InPlaceOpKind::kFloorDivide:
      case InPlaceOpKind::kLShift:
      case InPlaceOpKind::kModulo:
      case InPlaceOpKind::kMultiply:
      case InPlaceOpKind::kOr:
      case InPlaceOpKind::kRShift:
      case InPlaceOpKind::kSubtract:
      case InPlaceOpKind::kXor:
      case InPlaceOpKind::kPower:
      case InPlaceOpKind::kTrueDivide:
        env.emitUseType(lhs, TLongExact);
        env.emitUseType(rhs, TLongExact);
        return env.emitLongInPlaceOp(
            instr->op(), lhs, rhs, *instr->frameState());
      case InPlaceOpKind::kMatrixMultiply:
        // These will generate an error at runtime.
        break;
    }
  }
  // Change 4: InPlaceOp Object speculation for float accumulators.
  // When the LHS is Object (accumulator through Phi) and RHS is FloatExact,
  // guard LHS as FloatExact. This enables the Phi cascade: the guard narrows
  // the accumulator type, which propagates through the Phi back-edge.
  if (!lhs->isA(TFloatExact) && rhs->isA(TFloatExact)) {
    std::optional<BinaryOpKind> binop;
    switch (instr->op()) {
      case InPlaceOpKind::kAdd: binop = BinaryOpKind::kAdd; break;
      case InPlaceOpKind::kSubtract: binop = BinaryOpKind::kSubtract; break;
      case InPlaceOpKind::kMultiply: binop = BinaryOpKind::kMultiply; break;
      case InPlaceOpKind::kTrueDivide: binop = BinaryOpKind::kTrueDivide; break;
      case InPlaceOpKind::kFloorDivide: binop = BinaryOpKind::kFloorDivide; break;
      case InPlaceOpKind::kModulo: binop = BinaryOpKind::kModulo; break;
      case InPlaceOpKind::kPower: binop = BinaryOpKind::kPower; break;
      default: break;
    }
    if (binop && (FloatBinaryOp::slotMethod(*binop) || *binop == BinaryOpKind::kPower)) {
      Register* guarded_lhs = env.emitGuardType(TFloatExact, lhs, instr->frameState());
      env.emitUseType(rhs, TFloatExact);
      return env.emitFloatBinaryOp(*binop, guarded_lhs, rhs, *instr->frameState());
    }
  }

  // Phase 2: Float in-place ops. Convert InPlaceOpKind to BinaryOpKind
  // for FloatBinaryOp emission.
  if (lhs->isA(TFloatExact) && rhs->isA(TFloatExact)) {
    std::optional<BinaryOpKind> binop;
    switch (instr->op()) {
      case InPlaceOpKind::kAdd: binop = BinaryOpKind::kAdd; break;
      case InPlaceOpKind::kSubtract: binop = BinaryOpKind::kSubtract; break;
      case InPlaceOpKind::kMultiply: binop = BinaryOpKind::kMultiply; break;
      case InPlaceOpKind::kTrueDivide: binop = BinaryOpKind::kTrueDivide; break;
      case InPlaceOpKind::kFloorDivide: binop = BinaryOpKind::kFloorDivide; break;
      case InPlaceOpKind::kModulo: binop = BinaryOpKind::kModulo; break;
      case InPlaceOpKind::kPower: binop = BinaryOpKind::kPower; break;
      default: break;
    }
    if (binop && (FloatBinaryOp::slotMethod(*binop) || *binop == BinaryOpKind::kPower)) {
      env.emitUseType(lhs, TFloatExact);
      env.emitUseType(rhs, TFloatExact);
      return env.emitFloatBinaryOp(*binop, lhs, rhs, *instr->frameState());
    }
  }

  // InPlaceOp float speculation: guard Object operand to FloatExact.
  // Handles accumulators (total += val) where total is Object through Phi.
  auto rhs_type_ip = rhs->type();
  if (lhs->isA(TFloatExact) && !rhs->isA(TFloatExact) &&
      type_could_be(rhs_type_ip,
                        TFloatExact)) {
    auto binop = inPlaceOpToBinaryOp(instr->op());
    if (binop && (FloatBinaryOp::slotMethod(*binop) ||
                  *binop == BinaryOpKind::kPower)) {
      env.emitUseType(lhs, TFloatExact);
      Register* guarded = env.emitGuardType(TFloatExact, rhs, instr->frameState());
      return env.emitFloatBinaryOp(*binop, lhs, guarded,
                                      *instr->frameState());
    }
  }
  auto lhs_type_ip = lhs->type();
  if (rhs->isA(TFloatExact) && !lhs->isA(TFloatExact) &&
      type_could_be(lhs_type_ip,
                        TFloatExact)) {
    auto binop = inPlaceOpToBinaryOp(instr->op());
    if (binop && (FloatBinaryOp::slotMethod(*binop) ||
                  *binop == BinaryOpKind::kPower)) {
      env.emitUseType(rhs, TFloatExact);
      Register* guarded = env.emitGuardType(TFloatExact, lhs, instr->frameState());
      return env.emitFloatBinaryOp(*binop, guarded, rhs,
                                      *instr->frameState());
    }
  }

  return nullptr;
}

Register* simplifyLongBinaryOp(Env& env, const LongBinaryOp* instr) {
  // This isn't safe in the multi-threaded compilation on 3.12 because
  // we don't hold the GIL which is required for allocation.
  RETURN_MULTITHREADED_COMPILE(nullptr);

  Type left_type = instr->left()->type();
  Type right_type = instr->right()->type();
  HirType left_hir_lb = to_hir(left_type);
  HirType right_hir_lb = to_hir(right_type);
  if (hir_type_has_object_spec(&left_hir_lb) && hir_type_has_object_spec(&right_hir_lb)) {
    ThreadedCompileSerialize guard;
    Ref<> result;
    if (instr->op() == BinaryOpKind::kPower) {
      result = Ref<>::steal(PyLong_Type.tp_as_number->nb_power(
          hir_type_object_spec(&left_hir_lb), hir_type_object_spec(&right_hir_lb), Py_None));
    } else {
      binaryfunc helper = instr->slotMethod();
      result = Ref<>::steal(
          (*helper)(hir_type_object_spec(&left_hir_lb), hir_type_object_spec(&right_hir_lb)));
    }
    if (result == nullptr) {
      PyErr_Clear();
      return nullptr;
    }
    env.emitUseType(instr->left(), left_type);
    env.emitUseType(instr->right(), right_type);
    return env.emitLoadConst(
        Type::fromObject(env.func.env.addReference(std::move(result))));
  }
  return nullptr;
}

Register* simplifyFloatBinaryOp(Env& env, const FloatBinaryOp* instr) {
  // Convert FloatBinaryOp to native double arithmetic:
  // PrimitiveUnbox(PyFloat) + DoubleBinaryOp(fadd/fsub/fmul/fdiv) + PrimitiveBox
  // This avoids the C slot method call and heap allocation per operation.
  if (instr->op() != BinaryOpKind::kPower &&
      FloatBinaryOp::slotMethod(instr->op())) {
    Register* left_unboxed = env.emitPrimitiveUnbox(instr->left(), TCDouble);
    Register* right_unboxed = env.emitPrimitiveUnbox(instr->right(), TCDouble);
    Register* result = env.emitDoubleBinaryOp(instr->op(), left_unboxed, right_unboxed);
    return env.emitPrimitiveBox(result, TFloatExact, *instr->frameState());
  }

  // Constant folding (requires known values at compile time).
  // This isn't safe in the multi-threaded compilation on 3.12 because
  // we don't hold the GIL which is required for allocation.
  RETURN_MULTITHREADED_COMPILE(nullptr);

  Type left_type = instr->left()->type();
  Type right_type = instr->right()->type();

  HirType left_hir_fb = to_hir(left_type);
  HirType right_hir_fb = to_hir(right_type);

  if (!hir_type_has_object_spec(&left_hir_fb) || !hir_type_has_object_spec(&right_hir_fb)) {
    return nullptr;
  }

  ThreadedCompileSerialize guard;
  Ref<> result;

  if (instr->op() == BinaryOpKind::kPower) {
    result = Ref<>::steal(PyFloat_Type.tp_as_number->nb_power(
        hir_type_object_spec(&left_hir_fb), hir_type_object_spec(&right_hir_fb), Py_None));
  } else {
    binaryfunc helper = instr->slotMethod();
    result = Ref<>::steal(
        (*helper)(hir_type_object_spec(&left_hir_fb), hir_type_object_spec(&right_hir_fb)));
  }

  if (result == nullptr) {
    PyErr_Clear();
    return nullptr;
  }

  env.emitUseType(instr->left(), left_type);
  env.emitUseType(instr->right(), right_type);
  return env.emitLoadConst(
      Type::fromObject(env.func.env.addReference(std::move(result))));
}

Register* simplifyUnaryOp(Env& env, const UnaryOp* instr) {
  Register* operand = instr->operand();

  if (instr->op() == UnaryOpKind::kNot && operand->isA(TBool)) {
    env.emitUseType(operand, TBool);
    Register* unboxed = env.emitPrimitiveUnbox(operand, TCBool);
    Register* negated =
        env.emitPrimitiveUnaryOp(PrimitiveUnaryOpKind::kNotInt, unboxed);
    return env.emitPrimitiveBoxBool(negated);
  }

  return nullptr;
}

Register* simplifyPrimitiveCompare(Env& env, const PrimitiveCompare* instr) {
  Register* left = instr->GetOperand(0);
  Register* right = instr->GetOperand(1);
  if (instr->op() == PrimitiveCompareOp::kEqual ||
      instr->op() == PrimitiveCompareOp::kNotEqual) {
    auto do_cbool = [&](bool value) {
      env.emitUseType(left, left->type());
      env.emitUseType(right, right->type());
      return env.emitLoadConst(Type::fromCBool(
          instr->op() == PrimitiveCompareOp::kNotEqual ? !value : value));
    };
    auto left_type_cmp = left->type();
    auto right_type_cmp = right->type();
    if (!type_could_be(left_type_cmp,
                           right_type_cmp)) {
      return do_cbool(false);
    }
    if (type_has_int_spec(left_type_cmp) &&
        type_has_int_spec(right_type_cmp)) {
      return do_cbool(type_int_spec(left_type_cmp) ==
                      type_int_spec(right_type_cmp));
    }
    if (type_has_object_spec(left_type_cmp) &&
        type_has_object_spec(right_type_cmp)) {
      return do_cbool(type_object_spec(left_type_cmp) ==
                      type_object_spec(right_type_cmp));
    }
  }
  // box(b) == True --> b
  auto right_type_box = right->type();
  if (instr->op() == PrimitiveCompareOp::kEqual &&
      left->instr()->IsPrimitiveBoxBool() &&
      type_as_object(right_type_box) == Py_True) {
    return left->instr()->GetOperand(0);
  }
  return nullptr;
}

Register* simplifyPrimitiveBoxBool(Env& env, const Instr* instr) {
  Register* input = instr->GetOperand(0);
  HirType input_hir_bb = to_hir(input->type());
  if (hir_type_has_int_spec(&input_hir_bb)) {
    env.emitUseType(input, input->type());
    auto bool_obj = hir_type_int_spec(&input_hir_bb) ? Py_True : Py_False;
    return env.emitLoadConst(Type::fromObject(bool_obj));
  }
  return nullptr;
}

Register* simplifyUnbox(Env& env, const Instr* instr) {
  Register* input_value = instr->GetOperand(0);
  Type output_type = instr->output()->type();
  if (input_value->instr()->IsPrimitiveBox()) {
    // Simplify unbox(box(x)) -> x
    const auto box = static_cast<PrimitiveBox*>(input_value->instr());
    if (box->type() == output_type) {
      // We can't optimize away the potential overflow in unboxing.
      return box->GetOperand(0);
    }
  }
  // Ensure that we are dealing with either a integer or a double.
  Type input_value_type = input_value->type();
  HirType ivt_hir = to_hir(input_value_type);
  if (!hir_type_has_object_spec(&ivt_hir)) {
    return nullptr;
  }
  PyObject* value = hir_type_object_spec(&ivt_hir);
  if (output_type <= (TCSigned | TCUnsigned)) {
    if (!PyLong_Check(value)) {
      return nullptr;
    }
    int overflow = 0;
    long number =
        PyLong_AsLongAndOverflow(hir_type_object_spec(&ivt_hir), &overflow);
    if (overflow != 0) {
      return nullptr;
    }
    if (output_type <= TCSigned) {
      if (!Type::CIntFitsType(number, output_type)) {
        return nullptr;
      }
      return env.emitLoadConst(Type::fromCInt(number, output_type));
    } else {
      if (!Type::CUIntFitsType(number, output_type)) {
        return nullptr;
      }
      return env.emitLoadConst(Type::fromCUInt(number, output_type));
    }
  } else if (output_type <= TCDouble) {
    if (!PyFloat_Check(value)) {
      return nullptr;
    }
    double number = PyFloat_AS_DOUBLE(hir_type_object_spec(&ivt_hir));
    return env.emitLoadConst(Type::fromCDouble(number));
  }
  return nullptr;
}


// If we're loading ob_fval from a known float into a double, this can be
// simplified into a LoadConst.
Register* simplifyLoadField(Env& env, const LoadField* instr) {
  Register* loadee = instr->GetOperand(0);
  Type load_output_type = instr->output()->type();
  // Ensure that we are dealing with either a integer or a double.
  Type loadee_type = loadee->type();
  HirType loadee_hir = to_hir(loadee_type);
  if (!hir_type_has_object_spec(&loadee_hir)) {
    return nullptr;
  }
  PyObject* value = hir_type_object_spec(&loadee_hir);
  if (PyFloat_Check(value) && load_output_type <= TCDouble &&
      instr->offset() == offsetof(PyFloatObject, ob_fval)) {
    double number = PyFloat_AS_DOUBLE(hir_type_object_spec(&loadee_hir));
    env.emitUseType(loadee, loadee_type);
    return env.emitLoadConst(Type::fromCDouble(number));
  }
  return nullptr;
}

Register* simplifyIsNegativeAndErrOccurred(
    Env& env,
    const Instr* instr) {
  if (!instr->GetOperand(0)->instr()->IsLoadConst()) {
    return nullptr;
  }
  // Other optimizations might reduce the strength of global loads, etc. to load
  // consts. If this is the case, we know that there can't be an active
  // exception. In this case, the IsNegativeAndErrOccurred instruction has a
  // known result. Instead of deleting it, we replace it with load of false -
  // the idea is that if there are other downstream consumers of it, they will
  // still have access to the result. Otherwise, DCE will take care of this.
  Type output_type = instr->output()->type();
  return env.emitLoadConst(Type::fromCInt(0, output_type));
}

Register* simplifyStoreAttr(Env& env, const Instr* instr) {
  if (jit_get_config()->attr_caches) {
    auto db = static_cast<const DeoptBaseWithNameIdx*>(instr);
    return env.emitStoreAttrCached(
        instr->GetOperand(0),
        instr->GetOperand(1),
        db->name_idx(),
        *instr->asDeoptBase()->frameState());
  }
  return nullptr;
}

static bool isBuiltin(PyMethodDef* meth, const char* name) {
  // To make sure we have the right function, look up the PyMethodDef in the
  // fixed builtins. Any joker can make a new C method called "len", for
  // example.
  const Builtins& builtins = getContext()->builtins();
  return builtins.find(meth) == name;
}

static bool isBuiltin(Register* callable, const char* name) {
  Type callable_type = callable->type();
  HirType callable_hir = to_hir(callable_type);
  if (!hir_type_has_object_spec(&callable_hir)) {
    return false;
  }
  PyObject* callable_obj = hir_type_object_spec(&callable_hir);
  if (Py_TYPE(callable_obj) == &PyCFunction_Type) {
    PyCFunctionObject* func =
        reinterpret_cast<PyCFunctionObject*>(callable_obj);
    return isBuiltin(func->m_ml, name);
  }
  if (Py_TYPE(callable_obj) == &PyMethodDescr_Type) {
    PyMethodDescrObject* meth =
        reinterpret_cast<PyMethodDescrObject*>(callable_obj);
    return isBuiltin(meth->d_method, name);
  }
  return false;
}

// This is inspired by _PyEval_EvalCodeWithName in 3.8's Python/ceval.c
// We have a vector of Register* (resolved_args) that gets populated with
// already-provided arguments from call instructions alongside the function's
// default arguments, when such defaults are needed
static Register* resolveArgs(
    Env& env,
    const VectorCall* instr,
    PyFunctionObject* target) {
  PyCodeObject* code = (PyCodeObject*)target->func_code;
  JIT_CHECK(!(code->co_flags & CO_VARARGS), "can't resolve varargs");
  // number of positional args (including args with default values)
  size_t co_argcount = static_cast<size_t>(code->co_argcount);
  if (instr->numArgs() > co_argcount) {
    // TASK(T143644311): support varargs and check if non-varargs here
    return nullptr;
  }

  size_t num_positional = std::min(co_argcount, instr->numArgs());
  std::vector<Register*> resolved_args(co_argcount, nullptr);

  JIT_CHECK(!(code->co_flags & CO_VARKEYWORDS), "can't resolve varkwargs");

  // grab default positional arguments
  PyTupleObject* defaults = (PyTupleObject*)target->func_defaults;

  // TASK(T143644350): support kwargs and kwdefaults
  size_t num_defaults =
      defaults == nullptr ? 0 : static_cast<size_t>(PyTuple_GET_SIZE((PyObject*)defaults));

  if (num_positional + num_defaults < co_argcount) {
    // function was called with too few arguments
    return nullptr;
  }
  // TASK(T143644377): support kwonly args
  JIT_CHECK(code->co_kwonlyargcount == 0, " can't resolve kwonly args");
  for (size_t i = 0; i < co_argcount; i++) {
    if (i < num_positional) {
      resolved_args[i] = instr->arg(i);
    } else {
      size_t num_non_defaults = co_argcount - num_defaults;
      size_t default_idx = i - num_non_defaults;

      ThreadedCompileSerialize guard;
      auto def = PyTuple_GET_ITEM((PyObject*)defaults, default_idx);
      JIT_CHECK(def != nullptr, "expected non-null default");
      auto type = Type::fromObject(env.func.env.addReference(def));
      resolved_args[i] = env.emitLoadConst(type);
    }
    JIT_CHECK(resolved_args.at(i) != nullptr, "expected non-null arg");
  }

  Register* defaults_obj = env.emitLoadField(
      instr->GetOperand(0),
      "func_defaults",
      offsetof(PyFunctionObject, func_defaults),
      TTuple);
  env.emitGuardIs((PyObject*)defaults, defaults_obj);
  auto new_instr = env.emitVectorCallInstr(
      resolved_args.size() + 1, CallFlags::None, *instr->frameState());
  Register* result = new_instr->output();

  new_instr->SetOperand(0, instr->func());
  for (size_t i = 0; i < resolved_args.size(); i++) {
    new_instr->SetOperand(i + 1, resolved_args.at(i));
  }
  result->set_type(outputType(*new_instr));
  return result;
}


// Translate VectorCall to CallStatic whenever possible, saving stack
// manipulation costs (pushing args to stack).
static Register* trySpecializeCCall(Env& env, const VectorCall* instr) {
  if (instr->flags() & CallFlags::Awaited) {
    // We can't pass the awaited flag outside of vectorcall.
    return nullptr;
  }
  Register* callable = instr->func();
  Type callable_type = callable->type();
  HirType callable_hir_co = to_hir(callable_type);
  PyObject* callable_obj = hir_type_as_object(&callable_hir_co);
  if (callable_obj == nullptr) {
    return nullptr;
  }

  // Non METH_STATIC and METH_CLASS tp_methods on types are stored as
  // PyMethodDescr inside tp_dict. Check out:
  // Objects/typeobject.c#type_add_method
  if (Py_TYPE(callable_obj) == &PyMethodDescr_Type) {
    auto meth = reinterpret_cast<PyMethodDescrObject*>(callable_obj);
    PyMethodDef* def = meth->d_method;
    if (def->ml_flags & METH_NOARGS && instr->numArgs() == 1) {
      auto call = env.emitCallStaticInstr(
          1, reinterpret_cast<void*>(def->ml_meth),
          instr->output()->type() | TNullptr);
      call->SetOperand(0, instr->arg(0));
      return env.emitCheckExc(call->output(), *instr->frameState());
    }
    if (def->ml_flags & METH_O && instr->numArgs() == 2) {
      auto call = env.emitCallStaticInstr(
          2, reinterpret_cast<void*>(def->ml_meth),
          instr->output()->type() | TNullptr);
      call->SetOperand(0, instr->arg(0));
      call->SetOperand(1, instr->arg(1));
      return env.emitCheckExc(call->output(), *instr->frameState());
    }
  }
  return nullptr;
}

Register* simplifyVectorCallStatic(Env& env, const VectorCall* instr) {
  if (!(instr->flags() & CallFlags::Static)) {
    return nullptr;
  }
  Register* func = instr->func();
  if (isBuiltin(func, "list.append") && instr->numArgs() == 2) {
    env.emitUseType(func, func->type());
    env.emitListAppend(instr->arg(0), instr->arg(1), *instr->frameState());
    return env.emitLoadConst(TNoneType);
  }

  return trySpecializeCCall(env, instr);
}

// Special case here where we are testing `if isinstance`. In that case we do
// not want to go through the boxing and then unboxing that we are about to do.
// Instead, we want to directly provide the result of the unboxed comparison.
std::optional<std::pair<Instr*, std::vector<Instr*>>> isVectorCallIfIsInstance(
    Env& env,
    const VectorCall* instr) {
  std::vector<Instr*> snapshots;

  LivenessAnalysis::LastUses last_uses;
  Register* output = nullptr;

  enum state { kInitial, kCondBranch, kIsTruthy, kFailed };
  auto state = kInitial;

  auto block = instr->block();
  for (auto current = block->rbegin();
       current != block->rend() && state != kFailed;
       ++current) {
    switch (state) {
      case kInitial: {
        if (!current->IsCondBranch()) {
          state = kFailed;
          break;
        }

        LivenessAnalysis analysis{env.func};
        analysis.Run();

        last_uses = analysis.GetLastUses();
        auto lu_at_condbranch = last_uses.find(&*current);
        if (lu_at_condbranch == last_uses.end() ||
            lu_at_condbranch->second.size() != 1) {
          // If the CondBranch instruction is not the last use of the
          // IsTruthy output, then we cannot perform this optimization.
          state = kFailed;
          break;
        }

        state = kCondBranch;
        output = current->GetOperand(0);
        break;
      }
      case kCondBranch: {
        if (current->IsIsTruthy() && output == current->output() &&
            current->GetOperand(0) == instr->output()) {
          auto lu_at_istruthy = last_uses.find(&*current);
          if (lu_at_istruthy == last_uses.end() ||
              lu_at_istruthy->second.size() != 1) {
            // If the IsTruthy instruction is not the last use of the VectorCall
            // output, then we cannot perform this optimization.
            state = kFailed;
          } else {
            state = kIsTruthy;
          }
          break;
        }

        if (current->IsSnapshot()) {
          snapshots.push_back(&*current);
          break;
        }

        state = kFailed;
        break;
      }
      case kIsTruthy: {
        if (&*current == instr) {
          JIT_CHECK(output != nullptr, "output should have been set");
          return std::make_optional(std::make_pair(output->instr(), snapshots));
        }

        if (current->IsSnapshot()) {
          // Leave these snapshots in place.
          break;
        }

        state = kFailed;
        break;
      }
      case kFailed:
        JIT_ABORT("Hit kFailed state but it should not be reachable");
    }
  }

  // If we found anything else between the VectorCall, IsTruthy, and CondBranch
  // besides the expected instructions and some snapshots, then we cannot
  // perform this optimization.
  return std::nullopt;
}


// Simplify VectorCall where the function operand was produced by
// LoadAttrSpecial (used for __enter__/__aenter__ in with-statements).
// If the receiver type is exact and the special method resolves to a
// PyFunctionObject, replace the bound-method call with a static call
// to the resolved function with self as the first argument.
Register* simplifyVectorCallBoundMethod(Env& env, const VectorCall* instr) {
#if PY_VERSION_HEX < 0x030C0000
  (void)env;
  (void)instr;
  return nullptr;
#else
  // Only handle simple calls -- no kwargs, static, or awaited.
  if (instr->flags() &
      (CallFlags::KwArgs | CallFlags::Static | CallFlags::Awaited)) {
    return nullptr;
  }

  // Check if the function operand was produced by LoadAttrSpecial.
  Register* func_reg = instr->func();
  Instr* func_def = func_reg->instr();
  if (func_def == nullptr || !func_def->IsLoadAttrSpecial()) {
    return nullptr;
  }

  auto load_attr_special = static_cast<const LoadAttrSpecial*>(func_def);
  PyObject* attr_id = load_attr_special->id();

  // Handle __enter__/__aenter__ and __exit__/__aexit__ -- the special methods
  // used by with-statements. Resolving both sides lets the inliner eliminate
  // all context-manager call overhead.
  if (attr_id != &_Py_ID(__enter__) && attr_id != &_Py_ID(__aenter__) &&
      attr_id != &_Py_ID(__exit__) && attr_id != &_Py_ID(__aexit__)) {
    return nullptr;
  }

  // The receiver is the object being used as a context manager.
  Register* receiver = load_attr_special->GetOperand(0);
  Type receiver_type = receiver->type();
  HirType recv_hir2 = to_hir(receiver_type);
  PyTypeObject* py_type{hir_type_runtime_py_type(&recv_hir2)};

  // Bail if receiver type is not exact or not ready.
  if (!hir_type_is_exact(&recv_hir2) || py_type == nullptr ||
      !PyType_HasFeature(py_type, Py_TPFLAGS_READY)) {
    return nullptr;
  }

  // Ensure the type has a valid version tag for deopt safety.
  if (getThreadedCompileContext().compileRunning()) {
    if (!Ci_Type_HasValidVersionTag(py_type)) {
      return nullptr;
    }
  } else if (!ensureVersionTag(py_type)) {
    return nullptr;
  }

  // Resolve the special method through the MRO at compile time.
  PyObject* method{typeLookupSafe(py_type, attr_id)};
  if (method == nullptr) {
    return nullptr;
  }

  // Only handle plain Python functions -- reject C method descriptors,
  // classmethod, staticmethod, property, etc.
  if (!PyFunction_Check(method)) {
    return nullptr;
  }

  // Ensure a preloader exists for the resolved callee so the inliner can
  // inline it. preloadFuncAndDeps only discovers globals and static
  // invocations, not type attribute methods from LoadAttrSpecial.
  // Safe to call here: in single-function mode we hold the GIL, and for
  // trivial methods (e.g. __enter__(self): return self) the preloader
  // matches no bytecodes -- no Python code execution occurs.
  if (!getThreadedCompileContext().compileRunning()) {
    PyFunctionObject* py_func = (PyFunctionObject*)method;
    if (preloaderManager().find(py_func) == nullptr) {
      auto callee_preloader = Preloader::makePreloader(py_func);
      if (callee_preloader) {
        preloaderManager().add(
            (PyCodeObject*)py_func->func_code,
            std::move(callee_preloader));
      }
    }
  }

  // Emit a Snapshot to provide a FrameState for the DeoptPatchpoint.
  // bindGuards resets fs to nullptr on non-replayable instructions
  // (LoadAttrSpecial), so our DeoptPatchpoint needs its own Snapshot.
  env.emitSnapshot(*instr->frameState());

  if (!_PyClassLoader_IsImmutable((PyObject*)py_type)) {
    auto patchpoint = env.emitDeoptPatchpoint(
        env.func.allocateCodePatcher<TypeAttrDeoptPatcher>(
            py_type, (PyUnicodeObject*)attr_id, method));
    hir_c_set_guilty_reg(patchpoint,receiver);
    hir_c_set_descr(patchpoint,"LoadAttrSpecial method resolution");
  }
  HirType recv_unspec2 = hir_type_unspecialized(&recv_hir2);
  env.emitUseType(receiver, Type::fromHirType(recv_unspec2));

  // Load the resolved function as a constant.
  Register* func_const = env.emitLoadConst(
      Type::fromObject(env.func.env.addReference(method)));

  // Build a new VectorCall with the function, self (receiver), and
  // original arguments. Mark as Static since we're calling directly.
  size_t orig_nargs = instr->numArgs();
  auto new_call = env.emitVectorCallInstr(
      2 + orig_nargs, instr->flags() | CallFlags::Static, *instr->frameState());
  new_call->SetOperand(0, func_const);
  new_call->SetOperand(1, receiver);
  for (size_t i = 0; i < orig_nargs; ++i) {
    new_call->SetOperand(2 + i, instr->arg(i));
  }

  return new_call->output();
#endif
}

// Eliminate the function-identity GuardIs for calls to global functions that
// are loaded via LoadGlobalCached.  Instead of checking function identity at
// runtime (GuardIs), we install a GlobalDeoptPatcher that invalidates the
// compiled code if the global is rebound.  The code-object GuardIs in the
// inliner is kept as a safety net.
//
// Pattern matched:
//   LoadGlobalCached -> GuardIs(expected_func) -> VectorCall
// Replaced with:
//   Snapshot -> DeoptPatchpoint(GlobalDeoptPatcher) -> LoadConst(expected_func) -> VectorCall(Static)
Register* simplifyVectorCallGlobal(Env& env, const VectorCall* instr) {
  // Only handle simple calls -- no kwargs, static, or awaited.
  if (instr->flags() &
      (CallFlags::KwArgs | CallFlags::Static | CallFlags::Awaited)) {
    return nullptr;
  }

  // Check if the function operand was produced by a GuardIs.
  Register* func_reg = instr->func();
  Instr* func_def = func_reg->instr();
  if (func_def == nullptr || !func_def->IsGuardIs()) {
    return nullptr;
  }

  auto guard_is = static_cast<const GuardIs*>(func_def);

  // Check if the GuardIs operand came from LoadGlobalCached.
  Register* guarded_input = guard_is->GetOperand(0);
  Instr* input_def = guarded_input->instr();
  if (input_def == nullptr || !input_def->IsLoadGlobalCached()) {
    return nullptr;
  }

  // Get the expected value from the GuardIs.
  PyObject* expected = guard_is->target();
  if (!PyFunction_Check(expected)) {
    return nullptr;
  }

  // Get the globals dict and name from the LoadGlobalCached instruction.
  auto load_global = static_cast<const LoadGlobalCached*>(input_def);
  PyDictObject* globals{load_global->globals()};
  PyObject* name = PyTuple_GET_ITEM(load_global->code()->co_names,
                                     load_global->name_idx());
  if (!PyUnicode_CheckExact(name)) {
    return nullptr;
  }
  PyUnicodeObject* key_name = (PyUnicodeObject*)name;

  // Don't simplify during threaded compilation -- we need the GIL to
  // safely register watchers and access the preloader.
  if (getThreadedCompileContext().compileRunning()) {
    return nullptr;
  }

  // Emit a Snapshot to provide a FrameState for the DeoptPatchpoint.
  env.emitSnapshot(*instr->frameState());

  // Install a GlobalDeoptPatcher that fires if this global is rebound.
  auto* patcher = env.func.allocateCodePatcher<GlobalDeoptPatcher>(
      globals, key_name, (PyObject*)expected);
  auto patchpoint = env.emitDeoptPatchpoint(patcher);
  hir_c_set_guilty_reg(patchpoint,func_reg);
  hir_c_set_descr(patchpoint,"Global callee guard elimination");

  // Load the resolved function as a constant.  This gives the register
  // TFunc[expected] type so the inliner can determine the inline target.
  Register* func_const = env.emitLoadConst(
      Type::fromObject(env.func.env.addReference(expected)));

  // Build a new VectorCall with the constant function and original arguments.
  size_t orig_nargs = instr->numArgs();
  auto new_call = env.emitVectorCallInstr(
      1 + orig_nargs, instr->flags() | CallFlags::Static, *instr->frameState());
  new_call->SetOperand(0, func_const);
  for (size_t i = 0; i < orig_nargs; ++i) {
    new_call->SetOperand(1 + i, instr->arg(i));
  }

  return new_call->output();
}

Register* simplifyVectorCall(Env& env, const VectorCall* instr) {
  if (Register* result = simplifyVectorCallStatic(env, instr)) {
    return result;
  }
  if (Register* result = simplifyVectorCallBoundMethod(env, instr)) {
    return result;
  }
  if (Register* result = simplifyVectorCallGlobal(env, instr)) {
    return result;
  }
  if (instr->flags() & CallFlags::KwArgs) {
    return nullptr;
  }

  Register* target = instr->GetOperand(0);
  Type target_type = target->type();
  if (target_type == env.type_object && instr->NumOperands() == 2) {
    env.emitUseType(target, env.type_object);
    return env.emitLoadField(
        instr->GetOperand(1), "ob_type", offsetof(PyObject, ob_type), TType);
  }
  if (isBuiltin(target, "len") && instr->numArgs() == 1) {
    env.emitUseType(target, target->type());
    return env.emitGetLength(instr->arg(0), *instr->frameState());
  }
  if (isBuiltin(target, "isinstance") && instr->numArgs() == 2 &&
      instr->GetOperand(2)->type() <= TType &&
      !(instr->GetOperand(2)->type() <= TTuple)) {
    auto obj_op = instr->GetOperand(1);
    auto type_op = instr->GetOperand(2);

    auto obj_type = env.emitLoadField(
        obj_op, "ob_type", offsetof(PyObject, ob_type), TType);

    auto compare_type = env.emitPrimitiveCompare(
        PrimitiveCompareOp::kEqual, obj_type, type_op);

    // If this is a VectorCall to isinstance and it's being used as the
    // predicate of an if statement, it will look like:
    //
    //     o1 = VectorCall
    //     o2 = IsTruthy o1
    //     CondBranch o2
    //
    // Below, this would then expand into boxing the bool on both sides of the
    // conditional, then unboxing it again to do another comparison. Instead, we
    // can circumvent that by directly using the result of the primitive
    // compare.
    auto data = isVectorCallIfIsInstance(env, instr);
    if (data.has_value()) {
      auto& [is_truthy, snapshots] = data.value();
      auto result = is_truthy->output();

      // We no longer need the IsTruthy instruction.
      is_truthy->unlink();
      Instr::Destroy(is_truthy);

      // We also no longer need the Snapshot instructions contained between the
      // IsTruthy instruction and the CondBranch instruction.
      for (auto snapshot : snapshots) {
        snapshot->unlink();
        Instr::Destroy(snapshot);
      }

      env.emitCondSlowPath(
          result,
          compare_type,
          [&](auto slow_path) {
            return env.emitCondBranchInstr(compare_type, nullptr, slow_path);
          },
          [&] {
            return env.emitIsInstance(obj_op, type_op, *instr->frameState());
          });

      // The output of the VectorCall instruction was previously a TBool, but we
      // are replacing it with a TCBool since we are now doing a primitive
      // compare instead. This works, but requires that we change the
      // instruction's output type to match in order to pass the assertions that
      // come after the call to simplifyInstr.
      instr->output()->set_type(TCBool);

      return result;
    }

    Register* cbool_res = env.emitCond(
        [&](BasicBlock* fast_path, BasicBlock* slow_path) {
          env.emitCondBranch(compare_type, fast_path, slow_path);
        },
        [&] { // Fast path
          return compare_type;
        },
        [&] { // Slow path
          return env.emitIsInstance(obj_op, type_op, *instr->frameState());
        });
    return env.emitPrimitiveBoxBool(cbool_res);
  }
  HirType target_hir = to_hir(target_type);
  if (hir_type_has_value_spec(&target_hir, to_hir(TFunc))) {
    PyFunctionObject* func = (PyFunctionObject*)hir_type_object_spec(&target_hir);
    PyCodeObject* code = (PyCodeObject*)func->func_code;
    if (code->co_kwonlyargcount > 0 || (code->co_flags & CO_VARARGS) ||
        (code->co_flags & CO_VARKEYWORDS)) {
      // TASK(T143644854): full argument resolution
      return nullptr;
    }

    JIT_CHECK(
        code->co_argcount >= 0,
        "argcount must be greater than or equal to zero");
    if (instr->numArgs() != static_cast<size_t>(code->co_argcount)) {
      return resolveArgs(env, instr, func);
    }
  }
  return nullptr;
}

Register* simplifyStoreSubscr(Env& env, const Instr* instr) {
  if (instr->GetOperand(0)->isA(TDictExact)) {
    auto call = env.emitCallStaticInstr(
        3,
        reinterpret_cast<void*>(PyDict_Type.tp_as_mapping->mp_ass_subscript),
        TCInt32);
    call->SetOperand(0, instr->GetOperand(0));
    call->SetOperand(1, instr->GetOperand(1));
    call->SetOperand(2, instr->GetOperand(2));

    env.emitCheckNeg(call->output(), *instr->asDeoptBase()->frameState());
    return nullptr;
  }

  return nullptr;
}

Register* simplifyCIntToCBool(Env& env, const Instr* instr) {
  Type input_type = instr->GetOperand(0)->type();
  HirType input_hir_cb = to_hir(input_type);
  if (hir_type_has_int_spec(&input_hir_cb)) {
    return env.emitLoadConst(Type::fromCBool(hir_type_int_spec(&input_hir_cb)));
  }
  return nullptr;
}

Register* simplifyInstr(Env& env, const Instr* instr) {
  auto make_c_env = [&]() -> SimplifyEnv {
    return {&env.func, env.block, const_cast<Instr*>(&*env.cursor),
            env.bc_off.value(), 0, 0};
  };
  auto sync_c_env = [&](const SimplifyEnv& cenv) {
    if (cenv.optimized) env.optimized = true;
    if (cenv.new_blocks) {
      env.new_blocks += cenv.new_blocks;
      env.block = static_cast<BasicBlock*>(cenv.block);
      env.cursor = env.block->iterator_to(
          *static_cast<Instr*>(cenv.cursor_instr));
    }
  };
  switch (instr->opcode()) {
    case Opcode::kCheckVar:
    case Opcode::kCheckExc:
    case Opcode::kCheckField:
      return static_cast<Register*>(simplify_check_c(instr));
    case Opcode::kCheckSequenceBounds: {
      SimplifyEnv cenv = make_c_env();
      auto *r = static_cast<Register*>(simplify_check_sequence_bounds_c(&cenv, instr));
      sync_c_env(cenv);
      return r;
    }
    case Opcode::kGuardType: {
      SimplifyEnv cenv = make_c_env();
      auto *r = static_cast<Register*>(simplify_guard_type_c(&cenv, instr));
      sync_c_env(cenv);
      return r;
    }
    case Opcode::kRefineType:
      return static_cast<Register*>(simplify_refine_type_c(instr));
    case Opcode::kCast:
      return static_cast<Register*>(simplify_cast_c(instr));

    case Opcode::kCompare: {
      SimplifyEnv cenv = make_c_env();
      auto *r = static_cast<Register*>(simplify_compare_c(&cenv, instr));
      sync_c_env(cenv);
      return r;
    }

    case Opcode::kCondBranch: {
      SimplifyEnv cenv = make_c_env();
      auto *r = static_cast<Register*>(simplify_cond_branch_const_c(&cenv, instr));
      sync_c_env(cenv);
      return r;
    }
    case Opcode::kCondBranchCheckType: {
      SimplifyEnv cenv = make_c_env();
      auto *r = static_cast<Register*>(simplify_cond_branch_check_type_c(&cenv, instr));
      sync_c_env(cenv);
      return r;
    }

    case Opcode::kGetLength: {
      SimplifyEnv cenv = make_c_env();
      auto *r = static_cast<Register*>(simplify_get_length_c(&cenv, instr));
      sync_c_env(cenv);
      return r;
    }

    case Opcode::kIntConvert: {
      SimplifyEnv cenv = make_c_env();
      auto *r = static_cast<Register*>(simplify_int_convert_c(&cenv, instr));
      sync_c_env(cenv);
      return r;
    }

    case Opcode::kIsTruthy: {
      SimplifyEnv cenv = make_c_env();
      auto *r = static_cast<Register*>(simplify_is_truthy_c(&cenv, instr));
      sync_c_env(cenv);
      return r;
    }

// TODO(T255262756) - Enable this again. See P2169675076 and P2184559031 (same
// pattern but applied to simplifyLoadAttrTypeReceiver).
#ifndef Py_GIL_DISABLED
    case Opcode::kLoadAttr: {
      SimplifyEnv cenv = make_c_env();
      auto *r = static_cast<Register*>(simplify_load_attr_c(&cenv, instr));
      sync_c_env(cenv);
      return r;
    }
#endif
// TODO(T255263721) - Enable this again. See P2169673579 and P2184559031.
#ifndef Py_GIL_DISABLED
    case Opcode::kLoadMethod: {
      SimplifyEnv cenv = make_c_env();
      auto *r = static_cast<Register*>(simplify_load_method_c(&cenv, instr));
      sync_c_env(cenv);
      return r;
    }
#endif
    case Opcode::kLoadField: {
      SimplifyEnv cenv = make_c_env();
      auto *r = static_cast<Register*>(simplify_load_field_float_c(&cenv, instr));
      sync_c_env(cenv);
      return r;
    }
    case Opcode::kLoadTupleItem: {
      SimplifyEnv cenv = make_c_env();
      auto *r = static_cast<Register*>(simplify_load_tuple_item_c(&cenv, instr));
      sync_c_env(cenv);
      return r;
    }
    case Opcode::kLoadArrayItem: {
      SimplifyEnv cenv = make_c_env();
      auto *r = static_cast<Register*>(simplify_load_array_item_tuple_c(&cenv, instr));
      sync_c_env(cenv);
      return r;
    }
    case Opcode::kLoadVarObjectSize: {
      SimplifyEnv cenv = make_c_env();
      auto *r = static_cast<Register*>(simplify_load_var_object_size_c(&cenv, instr));
      sync_c_env(cenv);
      return r;
    }

    case Opcode::kBinaryOp: {
      SimplifyEnv cenv = make_c_env();
      auto *r = static_cast<Register*>(simplify_binary_op_c(&cenv, instr));
      sync_c_env(cenv);
      return r;
    }
    case Opcode::kInPlaceOp: {
      SimplifyEnv cenv = make_c_env();
      auto *r = static_cast<Register*>(simplify_in_place_op_c(&cenv, instr));
      sync_c_env(cenv);
      return r;
    }
    case Opcode::kLongBinaryOp: {
      SimplifyEnv cenv = make_c_env();
      auto *r = static_cast<Register*>(simplify_long_binary_op_c(&cenv, instr));
      sync_c_env(cenv);
      return r;
    }
    case Opcode::kFloatBinaryOp: {
      SimplifyEnv cenv = make_c_env();
      auto *r = static_cast<Register*>(simplify_float_binary_op_c(&cenv, instr));
      sync_c_env(cenv);
      return r;
    }
    case Opcode::kUnaryOp: {
      SimplifyEnv cenv = make_c_env();
      auto *r = static_cast<Register*>(simplify_unary_op_c(&cenv, instr));
      sync_c_env(cenv);
      return r;
    }

    case Opcode::kPrimitiveCompare: {
      SimplifyEnv cenv = make_c_env();
      auto *r = static_cast<Register*>(simplify_primitive_compare_c(&cenv, instr));
      sync_c_env(cenv);
      return r;
    }
    case Opcode::kPrimitiveBoxBool: {
      SimplifyEnv cenv = make_c_env();
      auto *r = static_cast<Register*>(simplify_primitive_box_bool_c(&cenv, instr));
      sync_c_env(cenv);
      return r;
    }
    case Opcode::kIndexUnbox:
    case Opcode::kPrimitiveUnbox: {
      SimplifyEnv cenv = make_c_env();
      auto *r = static_cast<Register*>(simplify_unbox_box_c(&cenv, instr));
      sync_c_env(cenv);
      return r;
    }

    case Opcode::kIsNegativeAndErrOccurred: {
      SimplifyEnv cenv = make_c_env();
      auto *r = static_cast<Register*>(simplify_is_neg_and_err_c(&cenv, instr));
      sync_c_env(cenv);
      return r;
    }

    case Opcode::kStoreAttr: {
      SimplifyEnv cenv = make_c_env();
      auto *r = static_cast<Register*>(simplify_store_attr_c(&cenv, instr));
      sync_c_env(cenv);
      return r;
    }

    case Opcode::kCallMethod: {
      SimplifyEnv cenv = make_c_env();
      auto *r = static_cast<Register*>(simplify_call_method_c(&cenv, instr));
      sync_c_env(cenv);
      return r;
    }

    case Opcode::kVectorCall: {
      SimplifyEnv cenv = make_c_env();
      auto *r = static_cast<Register*>(simplify_vectorcall_c(&cenv, instr));
      sync_c_env(cenv);
      if (r) return r;
      return simplifyVectorCall(env, static_cast<const VectorCall*>(instr));
    }

    case Opcode::kStoreSubscr: {
      SimplifyEnv cenv = make_c_env();
      auto *r = static_cast<Register*>(simplify_store_subscr_c(&cenv, instr));
      sync_c_env(cenv);
      return r;
    }

    case Opcode::kCIntToCBool: {
      SimplifyEnv cenv = make_c_env();
      auto *r = static_cast<Register*>(simplify_cint_to_cbool_c(&cenv, instr));
      sync_c_env(cenv);
      return r;
    }

    case Opcode::kGetIter: {
      SimplifyEnv cenv = make_c_env();
      simplify_get_iter_c(&cenv, instr);
      sync_c_env(cenv);
      return nullptr;
    }
    case Opcode::kInvokeIterNext: {
      SimplifyEnv cenv = make_c_env();
      auto *r = static_cast<Register*>(simplify_invoke_iter_next_c(&cenv, instr));
      sync_c_env(cenv);
      return r;
    }
    default:
      return nullptr;
  }
}

} // namespace

void Simplify::Run(Function& irfunc) {
  Env env{irfunc};

  const JitSimplifierCfg& config = jit_get_config()->simplifier;
  size_t new_block_limit = config.new_block_limit;
  size_t iteration_limit = config.iteration_limit;

  // Iterate the simplifier until the CFG stops changing, or we hit limits on
  // total number of iterations or the number of new blocks added.
  bool changed = true;
  for (size_t i = 0;
       changed && i < iteration_limit && env.new_blocks < new_block_limit;
       ++i) {
    changed = false;
    for (auto& block : irfunc.cfg.blocks) {
      env.block = &block;

      for (auto blk_it = block.begin(); blk_it != block.end();) {
        Instr& instr = *blk_it;
        ++blk_it;

        env.optimized = false;
        env.cursor = block.iterator_to(instr);
        env.bc_off = instr.bytecodeOffset();
        Register* new_output = simplifyInstr(env, &instr);
        JIT_CHECK(
            env.cursor == env.block->iterator_to(instr),
            "Simplify functions are expected to leave env.cursor pointing to "
            "the original instruction, with new instructions inserted before "
            "it.");
        if (new_output == nullptr && !env.optimized) {
          continue;
        }

        changed = true;
        JIT_CHECK(
            (new_output == nullptr) == (instr.output() == nullptr),
            "Simplify function should return a new output if and only if the "
            "existing instruction has an output");
        if (new_output != nullptr) {
          JIT_CHECK(
              new_output->type() <= instr.output()->type(),
              "New output type {} isn't compatible with old output type {}",
              new_output->type(),
              instr.output()->type());
          env.emitCInstr(static_cast<Instr*>(
              hir_assign_create(instr.output(), new_output)));
        }

        if (instr.IsCondBranch() || instr.IsCondBranchIterNotDone() ||
            instr.IsCondBranchCheckType()) {
          JIT_CHECK(env.cursor != env.block->begin(), "Unexpected empty block");
          Instr& prev_instr = *std::prev(env.cursor);
          JIT_CHECK(
              instr.opcode() == prev_instr.opcode() || prev_instr.IsBranch(),
              "The only supported simplification for CondBranch* is to a "
              "Branch or a different CondBranch, got unexpected '{}'",
              prev_instr);

          // If we've optimized a CondBranchBase into a Branch, we also need to
          // remove any Phi references to the current block from the block that
          // we no longer visit.
          if (prev_instr.IsBranch()) {
            auto cond = static_cast<CondBranchBase*>(&instr);
            BasicBlock* new_dst = prev_instr.successor(0);
            BasicBlock* old_branch_block = cond->false_bb() == new_dst
                ? cond->true_bb()
                : cond->false_bb();
            old_branch_block->removePhiPredecessor(cond->block());
          }
        }

        instr.unlink();
        Instr::Destroy(&instr);

        if (env.block != &block) {
          // If we're now in a different block, `block' should only contain the
          // newly-emitted instructions, with no more old instructions to
          // process. Continue to the next block in the list; any newly-created
          // blocks were added to the end of the list and will be processed
          // later.
          break;
        }
      }

      // Check for going past the new block limit only upon leaving a block.  We
      // might go past the limit, but not by too much.
      if (env.new_blocks > new_block_limit) {
        break;
      }
    }

    if (changed) {
      // Perform some simple cleanup between each pass.
      CopyPropagation{}.Run(irfunc);
      reflowTypes(irfunc);
      CleanCFG{}.Run(irfunc);
    }
  }
}

} // namespace jit::hir
