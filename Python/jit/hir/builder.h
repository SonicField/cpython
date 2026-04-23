// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/python.h"

#include "cinderx/Common/util.h"
#include "cinderx/Jit/bytecode.h"
#include "cinderx/Jit/bytecode_offsets.h"
#include "cinderx/Jit/hir/hir.h"
#include "cinderx/Jit/hir/preload.h"

#include <memory>
#include <unordered_map>
#include <unordered_set>

extern "C" {
void *hir_builder_get_block_at_off(void *builder, int byte_offset);
void *hir_builder_preloader_annotations(void *builder);
int hir_builder_preloader_num_args(void *builder);
HirType hir_builder_preloader_return_type(void *builder);
HirType hir_builder_preloader_type(void *builder, PyObject *descr);
void *hir_builder_preloader_py_type(void *builder, PyObject *descr);
void hir_builder_insert_run_periodic_activities_c(
    void *builder, void *func,
    void *check_block, void *succ_block, void *frame_state);
void *hir_builder_get_kwnames(void *builder);
void hir_builder_set_kwnames(void *builder, void *reg);
void *hir_builder_temps_alloc_stack(void *builder);
void hir_builder_fix_static_return_c(
    void *builder, void *tc, void *ret_val, HirType ret_type);

/* INVOKE_* Phase 2 #2 (theologian L2430): bridges for emitInvokeMethod C body. */
void hir_builder_invoke_method_target_c(
    void *builder, PyObject *descr,
    int *out_is_builtin, int *out_is_statically_typed, HirType *out_return_type);
int hir_builder_try_emit_direct_method_call_c(
    void *builder, void *tc, PyObject *descr, long nargs);
void hir_builder_setup_static_args_c(
    void *builder, void *tc, PyObject *descr, long nargs, int statically_typed,
    void **out_arg_regs, size_t *out_count);
void *hir_builder_static_method_stack_pop_c(void *builder);

/* W26 (theologian L2462+L2466): bridges for emitAnyCall full conversion +
 * 149b7e2d40 PartialConversion reabsorb. 4 NEW bridges: combined exception-
 * handler emit (folds findExceptionHandler+getSimpleExceptInfo+emit per (B)
 * decision), checkAsyncWithError, bc_it advance+opcode, bc_it oparg.
 *
 * PhxCallKind: opcode-to-kind mapping done in C++ stub, so the C body can
 * dispatch on a small enum without needing opcode constants in scope (which
 * would conflict with cinder_opcode.h's Py_OPCODE_H header guard, breaking
 * #ifdef BINARY_OP_ADD_INT in the BINARY_OP specialization path — root
 * cause of W26 first-attempt W21 golden regression at 95c9f9b891). */
enum PhxCallKind {
    PHX_CALL_KIND_VECTOR_CALL = 0,    /* CALL_FUNCTION / CALL_FUNCTION_KW */
    PHX_CALL_KIND_CALL_EX,            /* CALL_FUNCTION_EX */
    PHX_CALL_KIND_CALL_METHOD,        /* CALL / CALL_KW / CALL_METHOD */
    PHX_CALL_KIND_INVOKE_FUNCTION,    /* INVOKE_FUNCTION (Cinder static) */
    PHX_CALL_KIND_INVOKE_NATIVE,      /* INVOKE_NATIVE (Cinder static) */
    PHX_CALL_KIND_INVOKE_METHOD       /* INVOKE_METHOD (Cinder static) */
};
void hir_builder_emit_call_method_exception_handler_inline_c(
    void *builder, void *tc, void *cfg, int base_offset,
    void *call_instr, void *result_reg);
void hir_builder_check_async_with_error_c(
    void *bc_instrs, void *bc_it,
    int *out_error_aenter, int *out_error_aexit);
int hir_builder_bc_it_advance_and_opcode_c(void *bc_it);
int hir_builder_bc_it_oparg_c(void *bc_it);

/* INVOKE_* Phase 2 #3 (theologian L2430): bridges for emitInvokeFunction C
 * body. Function-target variants of #2 bridges (different preloader lookup
 * path: invokeFunctionTarget vs invokeMethodTarget) + invoke-target query +
 * static-rand conditional. */
void hir_builder_invoke_function_target_c(
    void *builder, PyObject *descr,
    int *out_container_is_immutable,
    int *out_is_function,
    int *out_is_statically_typed,
    int *out_is_builtin,
    void **out_callable,
    void **out_func,
    void **out_indirect_ptr,
    HirType *out_return_type);
int hir_builder_try_emit_direct_method_call_for_function_c(
    void *builder, void *tc, PyObject *descr, long nargs);
void hir_builder_setup_static_args_for_function_c(
    void *builder, void *tc, PyObject *descr, long nargs, int statically_typed,
    void **out_arg_regs, size_t *out_count);
int hir_builder_is_static_rand_and_try_emit_c(
    void *builder, void *tc, PyObject *descr, long nargs);

/* W27d #1 (theologian L2544): bridge for emitCopyFreeVars C body —
 * grants access to private Register* func_ field. */
void *hir_builder_func_register_c(void *builder);
} // extern "C"
#include <vector>

namespace jit::hir {

class BasicBlock;
class Environment;
class Function;
class Register;

// Helper class for managing temporary variables
class TempAllocator {
 public:
  explicit TempAllocator(Environment* env) : env_(env) {}

  // Allocate a temp register that may be used for the stack. It should not be a
  // register that will be treated specially in the FrameState (e.g. tracked as
  // containing a local or cell.)
  Register* AllocateStack();

  // Get the i-th stack temporary or allocate one
  Register* GetOrAllocateStack(std::size_t idx);

  // Allocate a temp register that will not be used for a stack value.
  Register* AllocateNonStack();

 private:
  Environment* env_;
  std::vector<Register*> cache_;
};

// We expect that on exit from a basic block the stack only contains temporaries
// in increasing order (called the canonical form). For example,
//
//    t0
//    t1
//    t2 <- top of stack
//
// It may be the case that temporaries are re-ordered, duplicated, or the stack
// contains locals. This class is responsible for inserting the necessary
// register moves such that the stack is in canonical form.
class BlockCanonicalizer {
 public:
  BlockCanonicalizer() : processing_(), done_(), copies_(), moved_() {}

  void Run(BasicBlock* block, TempAllocator& temps, PhxPtrArray& stack);

 private:
  DISALLOW_COPY_AND_ASSIGN(BlockCanonicalizer);

  void InsertCopies(
      Register* reg,
      TempAllocator& temps,
      Instr& terminator,
      std::vector<Register*>& alloced);

  std::unordered_set<Register*> processing_;
  std::unordered_set<Register*> done_;
  std::unordered_map<Register*, std::vector<Register*>> copies_;
  std::unordered_map<Register*, Register*> moved_;
};

// Translate the bytecode for preloader.code into HIR, in the context of the
// preloaded globals and classloader lookups in the preloader.
//
// The resulting HIR is un-optimized, not in SSA form, and does not yet have
// refcount operations or types flowed through it. Later passes will transform
// to SSA, flow types, optimize, and insert refcount operations using liveness
// analysis.
std::unique_ptr<Function> buildHIR(const Preloader& preloader);

// Inlining merges all of the different callee Returns (which terminate blocks,
// leading to a bunch of distinct exit blocks) into Branches to one Return
// block (one exit block), which the caller can transform into an Assign to the
// output register of the original call instruction.
struct InlineResult {
  BasicBlock* entry{nullptr};
  BasicBlock* exit{nullptr};
};

class HIRBuilder {
  friend void* ::hir_builder_get_block_at_off(void*, int);
  friend void ::hir_builder_insert_run_periodic_activities_c(
      void*, void*, void*, void*, void*);
  friend void* ::hir_builder_get_kwnames(void*);
  friend void ::hir_builder_set_kwnames(void*, void*);
  friend void* ::hir_builder_temps_alloc_stack(void*);
  friend void ::hir_builder_fix_static_return_c(void*, void*, void*, HirType);
  // INVOKE_* Phase 2 #2 (theologian L2430): bridges into emitInvokeMethod's
  // C++ helpers (tryEmitDirectMethodCall, setupStaticArgs, static_method_stack_).
  friend void ::hir_builder_invoke_method_target_c(
      void*, PyObject*, int*, int*, HirType*);
  friend int ::hir_builder_try_emit_direct_method_call_c(
      void*, void*, PyObject*, long);
  friend void ::hir_builder_setup_static_args_c(
      void*, void*, PyObject*, long, int, void**, size_t*);
  friend void* ::hir_builder_static_method_stack_pop_c(void*);
  // INVOKE_* Phase 2 #3 (theologian L2430): function-target variants for
  // emitInvokeFunction C body.
  friend void ::hir_builder_invoke_function_target_c(
      void*, PyObject*, int*, int*, int*, int*, void**, void**, void**, HirType*);
  friend int ::hir_builder_try_emit_direct_method_call_for_function_c(
      void*, void*, PyObject*, long);
  friend void ::hir_builder_setup_static_args_for_function_c(
      void*, void*, PyObject*, long, int, void**, size_t*);
  friend int ::hir_builder_is_static_rand_and_try_emit_c(
      void*, void*, PyObject*, long);
  // W26 (theologian L2462+L2466): emitAnyCall full conversion + 149b7e2d40
  // reabsorb. Friends grant private-member access (exception_table_,
  // findExceptionHandler, getSimpleExceptInfo, emitCallExceptionHandler).
  friend void ::hir_builder_emit_call_method_exception_handler_inline_c(
      void*, void*, void*, int, void*, void*);
  // W27d #1 (theologian L2544): grants C body access to private Register* func_.
  friend void* ::hir_builder_func_register_c(void*);
 public:
  const Preloader& preloader() const { return preloader_; }
  explicit HIRBuilder(const Preloader& preloader)
      : code_(preloader.code()), preloader_(preloader) {}

  // Translate the bytecode for code_ into HIR, in the context of the preloaded
  // globals and classloader lookups from preloader_.
  //
  // The resulting HIR is un-optimized, not in SSA form, and does not yet have
  // refcount operations or types flowed through it. Later passes will transform
  // to SSA, flow types, optimize, and insert refcount operations using liveness
  // analysis.
  std::unique_ptr<Function> buildHIR();

  // Given the preloader for the callee (passed into the constructor),
  // construct the CFG for the callee in the caller's CFG. Does not link the
  // two CFGs, except for FrameState parent pointers.  Use caller_frame_state
  // as the starting FrameState for the callee.
  //
  // Use InlineResult::succeeded to check if inlining succeeded.
  InlineResult inlineHIR(Function* caller, FrameState* caller_frame_state);

 private:
  DISALLOW_COPY_AND_ASSIGN(HIRBuilder);

  // Used by buildHIR and inlineHIR.
  // irfunc is the function being compiled or the caller function.
  // frame_state should be nullptr if irfunc matches the preloader (not
  // inlining) and non-nullptr otherwise (inlining).
  // Returns the entry block.
  BasicBlock* buildHIRImpl(Function* irfunc, FrameState* frame_state);

  struct TranslationContext;
  void translate(
      Function& irfunc,
      const jit::BytecodeInstructionBlock& bc_instrs,
      const TranslationContext& tc);

  void emitPushNull(TranslationContext& tc);

  void emitBinaryOp(
      CFG& cfg,
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitUnaryNot(TranslationContext& tc);
  void emitUnaryOp(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitAnyCall(
      CFG& cfg,
      TranslationContext& tc,
      jit::BytecodeInstructionBlock::Iterator& bc_it,
      const jit::BytecodeInstructionBlock& bc_instrs);
  void emitCallEx(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr,
      CallFlags flags);
  void emitCallInstrinsic(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitResume(
      CFG& cfg,
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitKwNames(TranslationContext& tc, const BytecodeInstruction& bc_instr);
  void emitIsOp(TranslationContext& tc, int oparg);
  void emitContainsOp(TranslationContext& tc, int oparg);
  void emitCompareOp(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitToBool(TranslationContext& tc);
  void emitCopyDictWithoutKeys(TranslationContext& tc);
  void emitGetLen(TranslationContext& tc);
  void emitJumpIf(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitDeleteAttr(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitLoadAttr(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitLoadMethod(TranslationContext& tc, int name_idx);
  void emitLoadMethodOrAttrSuper(
      CFG& cfg,
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr,
      bool load_method);
  void emitCopy(TranslationContext& tc, int item_idx);
  void emitCopyFreeVars(TranslationContext& tc, int nfreevars);
  void emitSwap(TranslationContext& tc, int item_idx);
  void emitMakeCell(TranslationContext& tc, int local_idx);
  void emitLoadDeref(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitStoreDeref(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitLoadAssertionError(TranslationContext& tc, Environment& env);
  void emitLoadClass(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitLoadConst(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitLoadFast(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitLoadFastLoadFast(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitLoadGlobal(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitLoadType(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitMakeFunction(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitFunctionCredential(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitMakeListTuple(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitBuildCheckedList(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitBuildCheckedMap(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitBuildMap(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitBuildSet(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitBuildConstKeyMap(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitPopJumpIf(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitPopJumpIfNone(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitStoreAttr(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitStoreFast(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitStoreFastStoreFast(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitStoreFastLoadFast(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitBinarySlice(TranslationContext& tc);
  void emitStoreSlice(TranslationContext& tc);
  void emitStoreSubscr(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitInPlaceOp(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitBuildSlice(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitLoadIterableArg(
      CFG& cfg,
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  bool emitInvokeFunction(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr,
      CallFlags flags);
  bool emitInvokeNative(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitGetIter(TranslationContext& tc, const jit::BytecodeInstruction& bc_instr);
  void emitGetYieldFromIter(CFG& cfg, TranslationContext& tc);
  void emitListAppend(
      TranslationContext& tc,
      const BytecodeInstruction& bc_instr);
  void emitListExtend(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitListToTuple(TranslationContext& tc);
  void emitForIter(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitInvokeMethodVectorCall(
      TranslationContext& tc,
      bool is_awaited,
      std::vector<Register*>& arg_regs,
      const InvokeTarget& target);
  void emitLoadMethodStatic(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  bool emitInvokeMethod(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr,
      bool is_awaited);
  void emitLoadField(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitStoreField(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitCast(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitTpAlloc(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitLoadSmallInt(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitStoreLocal(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitLoadLocal(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitConvertPrimitive(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitPrimitiveLoadConst(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitIntLoadConstOld(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitPrimitiveBinaryOp(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitPrimitiveCompare(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitPrimitiveBox(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitPrimitiveUnbox(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitImportFrom(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitImportName(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitPrimitiveUnaryOp(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitFastLen(
      CFG& cfg,
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitRaiseVarargs(TranslationContext& tc);
  void emitRefineType(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitSequenceGet(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitSequenceRepeat(
      CFG& cfg,
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitSequenceSet(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitYieldValue(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitGetAwaitable(
      CFG& cfg,
      TranslationContext& tc,
      const BytecodeInstructionBlock& bc_instrs,
      BytecodeInstruction bc_instr);
  void emitUnpackEx(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitUnpackSequence(
      CFG& cfg,
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitSetupFinally(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitAsyncForHeaderYieldFrom(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitEndAsyncFor(TranslationContext& tc);
  void emitGetAIter(TranslationContext& tc);
  void emitGetANext(TranslationContext& tc);
  Register* emitSetupWithCommon(
      TranslationContext& tc,
#if PY_VERSION_HEX < 0x030C0000
      _Py_Identifier* enter_id,
      _Py_Identifier* exit_id,
#else
      PyObject* enter_id,
      PyObject* exit_id,
#endif
      bool is_async);
  void emitBeforeWith(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitSetupAsyncWith(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitSetupWith(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitYieldFrom(TranslationContext& tc, Register* out);
  void emitDispatchEagerCoroResult(
      CFG& cfg,
      TranslationContext& tc,
      Register* out,
      BasicBlock* await_block,
      BasicBlock* post_await_block);

  void emitBuildString(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitFormatValue(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitFormatWithSpec(TranslationContext& tc);

  void emitMapAdd(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitSetAdd(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitSetUpdate(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);

  void
  emitMatchMappingSequence(CFG& cfg, TranslationContext& tc, uint64_t tf_flag);

  void emitMatchClass(
      CFG& cfg,
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitMatchKeys(CFG& cfg, TranslationContext& tc);

  void emitDictUpdate(
      TranslationContext& tc,
      const BytecodeInstruction& bc_instr);
  void emitDictMerge(
      TranslationContext& tc,
      const BytecodeInstruction& bc_instr);

  void emitSend(TranslationContext& tc, const BytecodeInstruction& bc_instr);

  void emitSetFunctionAttribute(
      TranslationContext& tc,
      const BytecodeInstruction& bc_instr);

  void emitTypeAnnotationGuards(TranslationContext& tc);

  void emitBuildInterpolation(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);

  void emitBuildTemplate(TranslationContext& tc);

  void emitConvertValue(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);

  void emitFormatSimple(CFG& cfg, TranslationContext& tc);

  void emitLoadCommonConstant(
      TranslationContext& tc,
      const BytecodeInstruction& bc_instr);

  void emitLoadSpecial(
      TranslationContext& tc,
      const BytecodeInstruction& bc_instr);

  void emitLoadBuildClass(TranslationContext& tc);

  void emitStoreGlobal(
      TranslationContext& tc,
      const BytecodeInstruction& bc_instr);

  PyObject* constArg(const jit::BytecodeInstruction& bc_instr);

  ExecutionBlock popBlock(CFG& cfg, TranslationContext& tc);
  void insertRunPeriodicActivitesForLoop(CFG& cfg, BasicBlock* loop_header);
  void insertRunPeriodicActivitesForExcept(CFG& cfg, TranslationContext& tc);
  void insertRunPeriodicActivites(
      CFG& cfg,
      BasicBlock* check_block,
      BasicBlock* succ,
      const FrameState& frame);
  void addInitialYield(TranslationContext& tc);
  void addLoadArgs(TranslationContext& tc, int num_args);
  void addInitializeCells(TranslationContext& tc);
  void allocateLocalsplus(Environment* env, FrameState& state);
  void moveOverwrittenStackRegisters(TranslationContext& tc, Register* dst);
  bool tryEmitDirectMethodCall(
      const InvokeTarget& target,
      TranslationContext& tc,
      long nargs);
  bool isStaticRand(const InvokeTarget& target);
  bool tryEmitStaticRandCall(
      const InvokeTarget& target,
      TranslationContext& tc,
      long nargs);
  struct BlockMap {
    std::unordered_map<BCOffset, BasicBlock*> blocks;
    std::unordered_map<BasicBlock*, BytecodeInstructionBlock> bc_blocks;
  };
  BlockMap createBlocks(
      Function& irfunc,
      const BytecodeInstructionBlock& bc_block);
  BasicBlock* getBlockAtOff(BCOffset off);

  // When a static function calls another static function indirectly, all args
  // are passed boxed and the return value will come back boxed, so we must
  // box primitive args and and unbox primitive return values. These functions
  // take care of these two, respectively.
  std::vector<Register*> setupStaticArgs(
      TranslationContext& tc,
      const InvokeTarget& target,
      long nargs,
      bool statically_invoked);
  void fixStaticReturn(TranslationContext& tc, Register* reg, Type ret_type);

  // Box the primitive value from src into dst, using the given type.
  void
  boxPrimitive(TranslationContext& tc, Register* dst, Register* src, Type type);

  // Unbox the primitive value from src into dst, using the given type. Similar
  // to TranslationContext::emitChecked(), but uses IsNegativeAndErrOccurred
  // instead of the normal CheckExc because of the primitive output value.
  void unboxPrimitive(
      TranslationContext& tc,
      Register* dst,
      Register* src,
      Type type);

  // Check that a code object can be compiled into HIR.
  void checkTranslate();

  void advancePastYieldInstr(TranslationContext& tc);

  PyCodeObject* code_;
  BlockMap block_map_;

  // Parsed exception table entries from co_exceptiontable (Layer 1).
  struct ExceptionTableEntry {
    BCOffset start;   // Start of try range (byte offset, inclusive)
    BCOffset end;     // End of try range (byte offset, exclusive)
    BCOffset target;  // Handler entry point (byte offset)
    int depth;        // Stack depth at handler entry
    bool lasti;       // Whether to push lasti
  };
  std::vector<ExceptionTableEntry> exception_table_;

  // B2: blocks that need to be added to the translation queue.
  // emitInlineExceptionMatch populates this; translate() drains it.
  struct PendingBlock {
    BasicBlock* block;
    FrameState frame;
  };
  std::vector<PendingBlock> pending_b2_blocks_;

  // Parse co_exceptiontable into exception_table_
  void parseExceptionTable();

  // Find exception handler for a given bytecode offset
  const ExceptionTableEntry* findExceptionHandler(BCOffset off) const;

  // B2: Info about a simple except pattern suitable for inlining.
  struct SimpleExceptInfo {
    int name_idx;           // Index into co_names for the exc type
    PyObject* exc_type;     // Resolved exception type (borrowed ref)
    BCOffset except_body;   // Offset of except body (after POP_TOP)
  };

  // B2: Check if handler has simple except pattern and extract info.
  bool getSimpleExceptInfo(
      const ExceptionTableEntry& handler,
      SimpleExceptInfo& info) const;

  // B2: Emit inline exception match for subscript inside try block.
  void emitInlineExceptionMatch(
      CFG& cfg,
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr,
      const ExceptionTableEntry& handler,
      const SimpleExceptInfo& info,
      Register* left,
      Register* right,
      Register* result);

  // B2: Emit inline exception handler for CALL inside try block.
  // Unlike emitInlineExceptionMatch (which replaces the operation with a
  // CallStatic), this works with an already-emitted call instruction.
  // It takes the FrameState from the call (preventing auto-deopt) and
  // emits a CondBranch on the call result to inline the exception handler.
  void emitCallExceptionHandler(
      CFG& cfg,
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr,
      const ExceptionTableEntry& handler,
      const SimpleExceptInfo& info,
      DeoptBase* call_instr,
      Register* result);

  const Preloader& preloader_;

  // Set during translate() — gives emitLoadAttr etc. access to the environment
  // for addReference calls on type objects found during compilation.
  Function* current_func_{nullptr};

  TempAllocator temps_{nullptr};

  // Tracks the function for compilations that require it.
  Register* func_{nullptr};

  // Tracks the most recent constant read from a KW_NAMES opcode.
  Register* kwnames_{nullptr};

  OperandStack static_method_stack_;
};

} // namespace jit::hir
