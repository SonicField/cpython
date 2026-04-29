// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/python.h"

#include "cinderx/Common/util.h"
#include "cinderx/Jit/bytecode.h"
#include "cinderx/Jit/bytecode_offsets.h"
#include "cinderx/Jit/hir/builder_state_c.h"
#include "cinderx/Jit/hir/hir.h"
#include "cinderx/Jit/hir/preload.h"

#include <memory>
#include <unordered_map>
#include <unordered_set>

extern "C" {
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
 * returns the func() Register* (post-§4.A.5c Pilot 5: reads state_.func
 * via the public getter; previously friend access to private func_). */
void *hir_builder_func_register_c(void *builder);

/* (D) emitLoadMethodStatic full PURE conversion bridges. Forward decls
 * required so friend lines in HIRBuilder match a real global declaration
 * (build error otherwise). */
int hir_builder_preloader_invoke_method_slot_c(void *builder, PyObject *descr);
void hir_builder_state_static_method_stack_push_cpp(void *builder, void *reg);
} // extern "C"
#include <vector>

namespace jit::hir {

class BasicBlock;
class Environment;
class Function;
class Register;

// Helper class for managing temporary variables.
// Phase 4.C Pilot 3 step 1 (Batch 42): now a thin C++ shim over the
// pure-C HirTempAllocator; methods delegate to hir_c_temps_*. The
// PhxRegisterArray cache (typedef of PhxPtrArray) replaces the prior
// std::vector<Register*>; move-only semantics + explicit dtor manage
// the malloc-backed storage.
class TempAllocator {
 public:
  explicit TempAllocator(Environment* env) {
    state_.env = env;
    phx_ptr_arr_init(&state_.cache);
  }
  ~TempAllocator() {
    phx_ptr_arr_destroy(&state_.cache);
  }

  // Move-only: cache_ owns malloc'd storage.
  TempAllocator(TempAllocator&& o) noexcept : state_(o.state_) {
    phx_ptr_arr_init(&o.state_.cache);
  }
  TempAllocator& operator=(TempAllocator&& o) noexcept {
    if (this != &o) {
      phx_ptr_arr_destroy(&state_.cache);
      state_ = o.state_;
      phx_ptr_arr_init(&o.state_.cache);
    }
    return *this;
  }
  TempAllocator(const TempAllocator&) = delete;
  TempAllocator& operator=(const TempAllocator&) = delete;

  // Allocate a temp register that may be used for the stack. It should not be a
  // register that will be treated specially in the FrameState (e.g. tracked as
  // containing a local or cell.)
  Register* AllocateStack();

  // Get the i-th stack temporary or allocate one
  Register* GetOrAllocateStack(std::size_t idx);

  // Allocate a temp register that will not be used for a stack value.
  Register* AllocateNonStack();

 private:
  HirTempAllocator state_;
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
  friend void ::hir_builder_insert_run_periodic_activities_c(
      void*, void*, void*, void*, void*);
  friend void* ::hir_builder_get_kwnames(void*);
  friend void ::hir_builder_set_kwnames(void*, void*);
  friend void* ::hir_builder_state_temps_alloc_stack_cpp(void*);
  friend void ::hir_builder_fix_static_return_c(void*, void*, void*, HirType);
  // INVOKE_* Phase 2 #2 (theologian L2430): bridges into emitInvokeMethod's
  // C++ helpers (tryEmitDirectMethodCall, setupStaticArgs, static_method_stack_).
  friend void ::hir_builder_invoke_method_target_c(
      void*, PyObject*, int*, int*, HirType*);
  friend int ::hir_builder_try_emit_direct_method_call_c(
      void*, void*, PyObject*, long);
  friend void ::hir_builder_setup_static_args_c(
      void*, void*, PyObject*, long, int, void**, size_t*);
  friend void* ::hir_builder_state_static_method_stack_pop_cpp(void*);
  // (D) emitLoadMethodStatic full PURE conversion (theologian 00:02Z + 00:03Z
  // + supervisor 00:04Z hold-lift). Bridges grant C body access to private
  // preloader_ (slot lookup) and static_method_stack_ (push).
  friend int ::hir_builder_preloader_invoke_method_slot_c(void*, PyObject*);
  friend void ::hir_builder_state_static_method_stack_push_cpp(void*, void*);
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
  // Tier 8 pilot Phase A (theologian 01:17:48Z + supervisor 01:18:35Z +
  // 03:44:19Z patch-apply): exception_table_ migrated to PhxExceptionTable
  // (pure-C container in PhxHirBuilderState); 3 _cpp bridges
  // (push/size/entry) DELETED. findExceptionHandler + parseExceptionTable
  // C++ shims rewired internally to PhxExceptionTable; Phase B will
  // delete those shims.
  // Tier 8 SECOND-PILOT Phases A + B: block_map_ ENTIRELY migrated.
  // Phase A: .blocks → PhxBlockMap (custom open-addressed hash); the
  // _cpp lookup bridge + hir_builder_get_block_at_off C-API DELETED;
  // C-side callers use phx_hir_builder_state + phx_block_map_lookup
  // directly. Phase B (theologian 11:13:21Z + supervisor 11:13:46Z):
  // .bc_blocks → PhxBcBlockArray (dense array indexed by
  // BasicBlock::id, exploiting allocation-monotonic id invariant);
  // BlockMap struct + block_map_ field DELETED.
  friend PhxHirBuilderState *::phx_hir_builder_state(void*);
 public:
  const Preloader& preloader() const {
    return *static_cast<const Preloader*>(state_.preloader);
  }
  // §4.A.5c PROBE-2 2026-04-27: current_func_ migrated to
  // state_.current_func; getter reads from state, write site
  // (translate(): line ~1636) writes state_.current_func directly.
  Function* current_func() const {
    return static_cast<Function*>(state_.current_func);
  }
  // §4.A.5c Pilot 5 2026-04-28: code_ + func_ migrated to state_.code +
  // state_.func. code() reads (immutable post-ctor); func()/set_func()
  // wrap the mutable Register* (write at builder.cpp:~1518, reads at
  // :~1519 + bridge :~3562). Bundle commit per theologian 08:39:09Z.
  PyCodeObject* code() const {
    return static_cast<PyCodeObject*>(state_.code);
  }
  Register* func() const {
    return static_cast<Register*>(state_.func);
  }
  void set_func(Register* f) {
    state_.func = static_cast<void*>(f);
  }
  explicit HIRBuilder(const Preloader& preloader) {
    hir_builder_state_init(
        &state_,
        static_cast<void*>(preloader.code()),
        static_cast<const void*>(&preloader));
  }

  // Translate the bytecode for code() into HIR, in the context of the preloaded
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
  /* emitInvokeFunction + emitInvokeNative deleted 2026-04-23 (Phase 1
   * batch #1, builder.cpp burndown). 9-line delegation stubs had ZERO
   * C++ callers — bytecode dispatch goes through hir_builder_emit_
   * invoke_function_c + hir_builder_emit_invoke_native_c (in
   * builder_emit_c.c) directly. Per supervisor 21:11:49Z exhaustive
   * caller-search gate + theologian 21:15:36Z methodology cross-check
   * PASS (4 pattern variants, 0 callers across 17 matches). */
  void emitGetIter(TranslationContext& tc, const jit::BytecodeInstruction& bc_instr);
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
  /* emitInvokeMethod deleted 2026-04-23 (Phase 1 batch #1, see
   * comment block at emitInvokeFunction deletion above). */
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
  void emitIntLoadConstOld(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitImportFrom(
      TranslationContext& tc,
      const jit::BytecodeInstruction& bc_instr);
  void emitImportName(
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
  void emitSequenceRepeat(
      CFG& cfg,
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
  void emitSetupAsyncWith(
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

  void emitMatchKeys(CFG& cfg, TranslationContext& tc);

  void emitDictUpdate(
      TranslationContext& tc,
      const BytecodeInstruction& bc_instr);
  void emitDictMerge(
      TranslationContext& tc,
      const BytecodeInstruction& bc_instr);


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


  void emitLoadCommonConstant(
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
  /* Tier 8 SECOND-PILOT Phase A + B: BlockMap struct DELETED. Both
   * sub-fields (.blocks, .bc_blocks) migrated to PhxHirBuilderState
   * (block_map_phx + bc_block_array_phx). createBlocks now writes
   * directly into state_; return type void. */
  void createBlocks(
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

  // §4.A.5c Pilot 5 2026-04-28: code_ field migrated to state_.code;
  // 46 reads in builder.cpp converted to code() getter.

  // Tier 8 pilot Phase A + Phase B: ExceptionTableEntry struct +
  // std::vector<...> exception_table_ field migrated to
  // PhxExceptionTable in PhxHirBuilderState.exception_table_phx
  // (builder_state_c.h). C++ shims findExceptionHandler +
  // parseExceptionTable DELETED (Phase B per spec §5 #5).
  // HIRBuilder no longer has accessor methods; all access via
  // bridges (parse_exception_table_c + find_exception_handler_c +
  // phx_exception_table_*).

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

  // §4.A.5c PROBE-1 2026-04-27: preloader_ migrated to state_.preloader
  // (smallest >15 single-setter — purest const-after-ctor shape; per-field
  // bracket validation per pythia #190 + supervisor 14:36:42Z + theologian
  // 14:36:45Z). C++ field deleted; preloader() getter reads from
  // state_.preloader. Probe-1 of the §4.A.5c bracket-validation gate.

  TempAllocator temps_{nullptr};

  // §4.A.5c Pilot 5 2026-04-28: func_ field migrated to state_.func;
  // 1 write (~1518) + 1 read (~1519) + 1 bridge read (~3562) converted
  // to func()/set_func().

  OperandStack static_method_stack_;

  // Phase 3 Batch 1: Class A state mirror; all Class A fields now live
  // exclusively in state_ (no parallel C++ duplicates).
  // §4.A.5 PROBE 2026-04-27: kwnames_ migrated to state_.kwnames; bridges
  // hir_builder_get/set_kwnames read/write state_.kwnames directly.
  // §4.A.5c PROBE-1 2026-04-27: preloader_ migrated to state_.preloader;
  // preloader() getter reads from state_.preloader. Smallest >15
  // single-setter (26 reads, const-after-ctor); validates the >15 bracket
  // single-commit-deletion pattern per supervisor 14:36:42Z.
  // §4.A.5c PROBE-2 2026-04-27: current_func_ migrated to
  // state_.current_func; current_func() getter reads from state, write site
  // (translate(): builder.cpp:~1636) writes state_.current_func directly.
  // 96 reads + 1 write (single-direct-write shape) — validates >15 bracket
  // for the structurally distinct shape vs PROBE-1's const-after-ctor.
  // §4.A.5c Pilot 5 2026-04-28: code_ + func_ migrated to state_.code +
  // state_.func. code() (46 reads, immutable post-ctor) + func()/set_func()
  // (1w + 2r). Bundle commit per theologian 08:39:09Z; closes Class A
  // state-mirror migration.
  PhxHirBuilderState state_{};
};

} // namespace jit::hir
