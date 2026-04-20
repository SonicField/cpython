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
      instr->output()->set_type(Type::fromHirType(hir_output_type(instr)));
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
      instr->output()->set_type(Type::fromHirType(hir_output_type(instr)));
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
          output->set_type(Type::fromHirType(hir_output_type(instr)));
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
      return r;
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
