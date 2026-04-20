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
