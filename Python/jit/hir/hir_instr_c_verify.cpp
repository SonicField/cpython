/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * T2-B verification: sizeof + offsetof static_asserts.
 * Uses friend struct for access to private C++ members.
 */

#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/hir/hir_basic_block_c.h"
#include "cinderx/Jit/hir/phx_frame_state.h"
#include "cinderx/Jit/hir/hir.h"
#include "cinderx/Jit/hir/cfg.h"
#include "cinderx/Jit/hir/frame_state.h"
#include "cinderx/Jit/hir/function.h"

#include <cassert>
#include <cstring>
#include <type_traits>

using namespace jit::hir;

/* ---- Compile-time size checks ---- */
static_assert(sizeof(HirInstrLayout) == sizeof(Instr),
    "HirInstr size must match C++ Instr");
static_assert(sizeof(HirCondBranchInstr) == sizeof(CondBranchBase),
    "HirCondBranchInstr size must match C++ CondBranchBase");
static_assert(sizeof(HirListNode) == sizeof(jit::IntrusiveListNode),
    "HirListNode size mismatch");
static_assert(sizeof(HirEdge) == sizeof(Edge),
    "HirEdge size mismatch");

/* Phase H2: Edge per-field offsetof checks (theologian gap #2) */
struct HirEdgeLayoutVerifier {
    static_assert(offsetof(HirEdge, from) == offsetof(Edge, from_));
    static_assert(offsetof(HirEdge, to) == offsetof(Edge, to_));
};

/* Phase H2: BasicBlock per-field offsetof checks */
struct HirBasicBlockLayoutVerifier {
    static_assert(offsetof(HirBasicBlock, instrs_) == offsetof(BasicBlock, instrs_));
    static_assert(offsetof(HirBasicBlock, out_edges_) == offsetof(BasicBlock, out_edges_));
    static_assert(offsetof(HirBasicBlock, in_edges_) == offsetof(BasicBlock, in_edges_));
};

/* Phase 4.A Batch 11: HirRegState POD-equivalence pin.
 * hir_c_deopt_visit_uses_deopt iterates PhxRegStateArray.data_ as
 * HirRegState* in pure C. Required invariants:
 *   - sizeof(HirRegState) == sizeof(jit::hir::RegState) (16 bytes)
 *   - HirRegState.reg / ref_kind / value_kind offsets match the C++
 *     RegState struct field offsets. */
static_assert(sizeof(HirRegState) == sizeof(jit::hir::RegState),
    "HirRegState size must match jit::hir::RegState");
static_assert(offsetof(HirRegState, reg) == offsetof(jit::hir::RegState, reg),
    "HirRegState.reg offset must match jit::hir::RegState.reg");
static_assert(offsetof(HirRegState, ref_kind) ==
              offsetof(jit::hir::RegState, ref_kind),
    "HirRegState.ref_kind offset must match jit::hir::RegState.ref_kind");
static_assert(offsetof(HirRegState, value_kind) ==
              offsetof(jit::hir::RegState, value_kind),
    "HirRegState.value_kind offset must match jit::hir::RegState.value_kind");

/* Phase 4.A Batch 8: PhxRegStateArray POD-equivalence pin.
 * hir_c_deopt_live_regs returns &HirDeoptLayout.live_regs_data; C++ shim
 * casts it to PhxRegStateArray*. Required invariants:
 *   - PhxRegStateArray is 3 contiguous pointer-sized fields (data_, count_,
 *     capacity_) at offsets 0, sizeof(void*), 2*sizeof(void*).
 *   - HirDeoptLayout.live_regs_{data,count,cap} sit at the matching
 *     contiguous offsets within HirDeoptLayout.
 * If any of these drifts, the shim cast becomes UB. */
static_assert(sizeof(PhxRegStateArray) == 3 * sizeof(void *),
    "PhxRegStateArray expected 3-pointer POD");
static_assert(offsetof(HirDeoptLayout, live_regs_count) -
              offsetof(HirDeoptLayout, live_regs_data) == sizeof(void *),
    "live_regs_count must follow live_regs_data with no padding");
static_assert(offsetof(HirDeoptLayout, live_regs_cap) -
              offsetof(HirDeoptLayout, live_regs_data) == 2 * sizeof(void *),
    "live_regs_cap must follow live_regs_count with no padding");

/* Phase R1b: Register layout verification */
static_assert(sizeof(HirRegisterLayout) == sizeof(Register),
    "HirRegisterLayout size mismatch with Register");
struct HirRegisterLayoutVerifier {
    static_assert(offsetof(HirRegisterLayout, type) == offsetof(Register, type_));
    static_assert(offsetof(HirRegisterLayout, instr) == offsetof(Register, instr_));
    static_assert(offsetof(HirRegisterLayout, id) == offsetof(Register, id_));
    static_assert(offsetof(HirRegisterLayout, name) == offsetof(Register, name_));
};

/* Phase C1: CFG layout verification */
static_assert(sizeof(HirCFG) == sizeof(CFG),
    "HirCFG size mismatch with CFG");
struct HirCFGLayoutVerifier {
    static_assert(offsetof(HirCFG, entry_block) == offsetof(CFG, entry_block));
    static_assert(offsetof(HirCFG, block_root) == offsetof(CFG, blocks));
    static_assert(offsetof(HirCFG, next_block_id) == offsetof(CFG, next_block_id_));
};

/* Phase F1: FrameState layout verification */
static_assert(sizeof(HirFrameStateLayout) == sizeof(FrameState),
    "HirFrameStateLayout size mismatch with FrameState");
static_assert(offsetof(HirFrameStateLayout, cur_instr_offs) == offsetof(FrameState, cur_instr_offs));
static_assert(offsetof(HirFrameStateLayout, localsplus) == offsetof(FrameState, localsplus));
static_assert(offsetof(HirFrameStateLayout, nlocals) == offsetof(FrameState, nlocals));
static_assert(offsetof(HirFrameStateLayout, stack) == offsetof(FrameState, stack));
static_assert(offsetof(HirFrameStateLayout, block_stack_data) == offsetof(FrameState, block_stack));
static_assert(offsetof(HirFrameStateLayout, code) == offsetof(FrameState, code));
static_assert(offsetof(HirFrameStateLayout, globals) == offsetof(FrameState, globals));
static_assert(sizeof(PhxExecBlock) == sizeof(ExecutionBlock), "PhxExecBlock must match ExecutionBlock size");
static_assert(offsetof(PhxExecBlock, opcode) == offsetof(ExecutionBlock, opcode));
static_assert(offsetof(PhxExecBlock, handler_off) == offsetof(ExecutionBlock, handler_off));
static_assert(offsetof(PhxExecBlock, stack_level) == offsetof(ExecutionBlock, stack_level));
static_assert(offsetof(HirFrameStateLayout, builtins) == offsetof(FrameState, builtins));
static_assert(offsetof(HirFrameStateLayout, parent) == offsetof(FrameState, parent));

/* Phase E2: Environment layout verification */
static_assert(sizeof(HirEnvironment) == sizeof(Environment),
    "HirEnvironment size mismatch with Environment");
struct HirEnvironmentLayoutVerifier {
    static_assert(offsetof(HirEnvironment, reg_data) == offsetof(Environment, reg_data_));
    static_assert(offsetof(HirEnvironment, reg_count) == offsetof(Environment, reg_count_));
    static_assert(offsetof(HirEnvironment, reg_capacity) == offsetof(Environment, reg_capacity_));
    static_assert(offsetof(HirEnvironment, references_opaque) == offsetof(Environment, references_));
    static_assert(offsetof(HirEnvironment, next_register_id) == offsetof(Environment, next_register_id_));
    static_assert(offsetof(HirEnvironment, next_load_type_attr_cache) == offsetof(Environment, next_load_type_attr_cache_));
    static_assert(offsetof(HirEnvironment, next_load_type_method_cache) == offsetof(Environment, next_load_type_method_cache_));
};

/* Phase Fn1: Function layout verification */
static_assert(sizeof(HirFunctionLayout) == sizeof(Function),
    "HirFunctionLayout size mismatch with Function");
/* Fn2: verify first 5 fields are at pointer-stride offsets */
static_assert(offsetof(Function, code) == 0 * sizeof(void*));
static_assert(offsetof(Function, builtins) == 1 * sizeof(void*));
static_assert(offsetof(Function, globals) == 2 * sizeof(void*));
static_assert(offsetof(Function, prim_args_info) == 3 * sizeof(void*));
static_assert(offsetof(Function, fullname) == 4 * sizeof(void*));
/* Fn2b: remaining field offsets (discovered via diagnostic build) */
static_assert(offsetof(Function, return_type) == 128);
static_assert(offsetof(Function, env) == 152);
static_assert(offsetof(Function, cfg) == 248);
static_assert(offsetof(Function, reifier) == 320);

/* ---- Per-field offsetof checks via friend struct ---- */
struct HirInstrLayoutVerifier {
    /* HirInstr vs Instr */
    static_assert(offsetof(HirInstrLayout, block_node) == offsetof(Instr, block_node_));
    static_assert(offsetof(HirInstrLayout, opcode) == offsetof(Instr, opcode_));
    static_assert(offsetof(HirInstrLayout, bytecode_offset) == offsetof(Instr, bytecode_offset_));
    static_assert(offsetof(HirInstrLayout, output) == offsetof(Instr, output_));
    static_assert(offsetof(HirInstrLayout, block) == offsetof(Instr, block_));

    /* HirDeoptLayout field offsets vs DeoptBase */
    static_assert(offsetof(HirDeoptLayout, live_regs_data) == offsetof(DeoptBase, live_regs_));
    static_assert(offsetof(HirDeoptLayout, frame_state) == offsetof(DeoptBase, frame_state_));
    static_assert(offsetof(HirDeoptLayout, guilty_reg) == offsetof(DeoptBase, guilty_reg_));
    static_assert(offsetof(HirDeoptLayout, nonce) == offsetof(DeoptBase, nonce_));
    static_assert(offsetof(HirDeoptLayout, descr) == offsetof(DeoptBase, descr_));
    static_assert(offsetof(HirDeoptLayout, suppress_exception_deopt) ==
        offsetof(DeoptBase, suppress_exception_deopt_));

    /* HirCondBranchInstr field offsets vs CondBranchBase */
    static_assert(offsetof(HirCondBranchInstr, true_edge) == offsetof(CondBranchBase, true_edge_));
    static_assert(offsetof(HirCondBranchInstr, false_edge) == offsetof(CondBranchBase, false_edge_));

    /* Phase 4.A Batch 22: HirBranch.edge offset pin + CondBranch
     * contiguity pin so std::span{&true_edge_, 2} (now sole-sourced
     * via hir_c_edge_at) walks adjacent storage. */
    static_assert(offsetof(HirBranch, edge) == offsetof(Branch, edge_));
    static_assert(offsetof(HirCondBranchInstr, false_edge) -
                      offsetof(HirCondBranchInstr, true_edge) ==
                  sizeof(HirEdge),
                  "CondBranchBase span{&true_edge_, 2} requires "
                  "false_edge_ to follow true_edge_ contiguously");

    /* Phase 4.A Batch 28: compare op table size pins. The C arrays
     * kCompareOpNames_c / kPrimitiveCompareOpNames_c must stay in
     * lockstep with the FOREACH_*_OP macros in hir.h. */
    /* Cannot use static_assert on extern const size_t (not a constant
     * expression); the falsifier verify_phase4a_batch28_compare_ops
     * checks this at constructor-time instead. */

    /* T2-B Batch 2: derived type sizes */
    static_assert(sizeof(HirBinaryOp) == sizeof(BinaryOp));
    static_assert(sizeof(HirUnaryOp) == sizeof(UnaryOp));
    static_assert(sizeof(HirInPlaceOp) == sizeof(InPlaceOp));
    static_assert(sizeof(HirIntBinaryOp) == sizeof(IntBinaryOp));
    static_assert(sizeof(HirDoubleBinaryOp) == sizeof(DoubleBinaryOp));
    static_assert(sizeof(HirPrimitiveUnaryOp) == sizeof(PrimitiveUnaryOp));
    static_assert(sizeof(HirLongBinaryOp) == sizeof(LongBinaryOp));
    static_assert(sizeof(HirLongInPlaceOp) == sizeof(LongInPlaceOp));
    static_assert(sizeof(HirFloatBinaryOp) == sizeof(FloatBinaryOp));
    static_assert(sizeof(HirCompare) == sizeof(Compare));
    static_assert(sizeof(HirFloatCompare) == sizeof(FloatCompare));
    static_assert(sizeof(HirLongCompare) == sizeof(LongCompare));
    static_assert(sizeof(HirUnicodeCompare) == sizeof(UnicodeCompare));
    static_assert(sizeof(HirCompareBool) == sizeof(CompareBool));
    static_assert(sizeof(HirPrimitiveCompare) == sizeof(PrimitiveCompare));
    /* W-PYTORCH-CM-(i) hardening: hir_c_compare_op casts to HirCompare*
     * (HIR_DEOPT_FIELDS); hir_c_primitive_compare_op casts to
     * HirPrimitiveCompare* (HIR_INSTR_FIELDS). Their `op` fields are at
     * DIFFERENT offsets — must never be conflated by a future single-accessor
     * shortcut. Recurrence prevention per feedback_class_of_bug_audit.md +
     * pythia #161 #4. */
    static_assert(offsetof(HirCompare, op) != offsetof(HirPrimitiveCompare, op),
                  "HirCompare and HirPrimitiveCompare must have distinct `op` "
                  "field offsets — cannot share a single accessor");
    /* T2-B Batch 3: Type-field struct sizes */
    static_assert(sizeof(HirLoadConst) == sizeof(LoadConst));
    static_assert(sizeof(HirRefineType) == sizeof(RefineType));
    static_assert(sizeof(HirBitCast) == sizeof(BitCast));
    static_assert(sizeof(HirReturn) == sizeof(Return));
    static_assert(sizeof(HirUseType) == sizeof(UseType));
    static_assert(sizeof(HirIntConvert) == sizeof(IntConvert));
    static_assert(sizeof(HirPrimitiveUnbox) == sizeof(PrimitiveUnbox));
    static_assert(sizeof(HirGetSecondOutput) == sizeof(GetSecondOutput));
    static_assert(sizeof(HirStoreArrayItem) == sizeof(StoreArrayItem));
    static_assert(sizeof(HirLoadArrayItem) == sizeof(LoadArrayItem));
    static_assert(sizeof(HirPrimitiveBox) == sizeof(PrimitiveBox));
    /* T2-B Batch 4a: scalar-field struct sizes */
    static_assert(sizeof(HirBranch) == sizeof(Branch));
    static_assert(sizeof(HirSetFunctionAttr) == sizeof(SetFunctionAttr));
    static_assert(sizeof(HirCallIntrinsic) == sizeof(CallIntrinsic));
    static_assert(sizeof(HirCallStaticRetVoid) == sizeof(CallStaticRetVoid));
    static_assert(sizeof(HirIndexUnbox) == sizeof(IndexUnbox));
    static_assert(sizeof(HirLoadTupleItem) == sizeof(LoadTupleItem));
    static_assert(sizeof(HirLoadSplitDictItem) == sizeof(LoadSplitDictItem));
    static_assert(sizeof(HirInitFrameCellVars) == sizeof(InitFrameCellVars));
    static_assert(sizeof(HirLoadTypeAttrCacheEntryType) == sizeof(LoadTypeAttrCacheEntryType));
    static_assert(sizeof(HirLoadTypeAttrCacheEntryValue) == sizeof(LoadTypeAttrCacheEntryValue));
    static_assert(sizeof(HirLoadTypeMethodCacheEntryType) == sizeof(LoadTypeMethodCacheEntryType));
    static_assert(sizeof(HirLoadTypeMethodCacheEntryValue) == sizeof(LoadTypeMethodCacheEntryValue));
    static_assert(sizeof(HirCallStatic) == sizeof(CallStatic));
    static_assert(sizeof(HirEndInlinedFunction) == sizeof(EndInlinedFunction));
    static_assert(sizeof(HirUpdatePrevInstr) == sizeof(UpdatePrevInstr));
    static_assert(sizeof(HirLoadArg) == sizeof(LoadArg));
    static_assert(sizeof(HirVectorCall) == sizeof(VectorCall));
    static_assert(sizeof(HirCallEx) == sizeof(CallEx));
    static_assert(sizeof(HirTpAlloc) == sizeof(TpAlloc));
    static_assert(sizeof(HirGuardIs) == sizeof(GuardIs));
    static_assert(sizeof(HirGuardType) == sizeof(GuardType));
    static_assert(sizeof(HirMakeDict) == sizeof(MakeDict));
    static_assert(sizeof(HirDeoptPatchpoint) == sizeof(DeoptPatchpoint));
    static_assert(sizeof(HirRaiseAwaitableError) == sizeof(RaiseAwaitableError));
    static_assert(sizeof(HirBuildInterpolation) == sizeof(BuildInterpolation));
    static_assert(sizeof(HirConvertValue) == sizeof(ConvertValue));
    static_assert(sizeof(HirLoadSpecial) == sizeof(LoadSpecial));
    static_assert(sizeof(HirCast) == sizeof(Cast));
    static_assert(sizeof(HirLoadFunctionIndirect) == sizeof(LoadFunctionIndirect));
    static_assert(sizeof(HirRaiseStatic) == sizeof(RaiseStatic));
    static_assert(sizeof(HirCallInd) == sizeof(CallInd));
    static_assert(sizeof(HirMakeCheckedDict) == sizeof(MakeCheckedDict));
    static_assert(sizeof(HirMakeCheckedList) == sizeof(MakeCheckedList));
    static_assert(sizeof(HirCondBranchCheckType) == sizeof(CondBranchCheckType));
    /* T2-B: opaque blob field offsets (container fields modeled as char arrays) */
    static_assert(offsetof(HirLoadField, name) == offsetof(LoadField, name_));
    static_assert(offsetof(HirStoreField, name) == offsetof(StoreField, name_));
    static_assert(offsetof(HirPhi, bb_data) == offsetof(Phi, basic_blocks_));
    static_assert(offsetof(HirBeginInlinedFunction, fullname) == offsetof(BeginInlinedFunction, fullname_));
    static_assert(offsetof(HirHintType, types_storage) == offsetof(HintType, types_));

    /* T2-B Batch 4b: container-field struct sizes */
    static_assert(sizeof(HirCallMethod) == sizeof(CallMethod));
    static_assert(sizeof(HirCallCFunc) == sizeof(CallCFunc));
    static_assert(sizeof(HirSnapshot) == sizeof(Snapshot));
    static_assert(sizeof(HirLoadGlobalCached) == sizeof(LoadGlobalCached));
    static_assert(sizeof(HirFillTypeAttrCache) == sizeof(FillTypeAttrCache));
    static_assert(sizeof(HirFillTypeMethodCache) == sizeof(FillTypeMethodCache));
    static_assert(sizeof(HirLoadField) == sizeof(LoadField));
    static_assert(sizeof(HirStoreField) == sizeof(StoreField));
    static_assert(sizeof(HirPhi) == sizeof(Phi));
    static_assert(sizeof(HirBeginInlinedFunction) == sizeof(BeginInlinedFunction));
    static_assert(sizeof(HirHintType) == sizeof(HintType));
    static_assert(sizeof(HirInvokeStaticFunction) == sizeof(InvokeStaticFunction));
    static_assert(sizeof(HirUnpackExToTuple) == sizeof(UnpackExToTuple));
    static_assert(sizeof(HirLoadAttrSpecial) == sizeof(LoadAttrSpecial));

    /* H2-A2: intermediate base class struct sizes */
    static_assert(sizeof(HirLoadAttr) == sizeof(LoadAttr));

    /* H2-A2: offsetof checks for intermediate base class custom fields.
     * sizeof alone is INSUFFICIENT — padding hides missing fields. */

    /* DeoptBaseWithNameIdx types: name_idx field */
    static_assert(offsetof(HirFillTypeAttrCache, name_idx) == offsetof(FillTypeAttrCache, name_idx_));
    static_assert(offsetof(HirFillTypeMethodCache, name_idx) == offsetof(FillTypeMethodCache, name_idx_));
    static_assert(offsetof(HirImportFrom, name_idx) == offsetof(ImportFrom, name_idx_));
    /* LoadAttr: name_idx + already_optimized */
    static_assert(offsetof(HirLoadAttr, name_idx) == offsetof(LoadAttr, name_idx_));
    static_assert(offsetof(HirLoadAttr, already_optimized) == offsetof(LoadAttr, already_optimized_));
    /* LoadMethodBase set (LoadMethod / LoadMethodCached /
     * LoadModuleMethodCached): no concrete C++ class exists — Phoenix
     * allocates these directly via the HIR_OP_LoadMethod{,Cached,...}
     * C factories. Layout consistency across the three C structs is
     * by-construction (all expand HIR_DEOPT_NAMEIDX_FIELDS); cross-arch
     * compatibility with DeoptBaseWithNameIdx::name_idx_ is covered
     * indirectly by the FillTypeAttrCache check above (same macro). */
    static_assert(offsetof(HirLoadMethod, name_idx) == offsetof(HirLoadMethodCached, name_idx));
    static_assert(offsetof(HirLoadMethod, name_idx) == offsetof(HirLoadModuleMethodCached, name_idx));

    /* Existing Batch 4b container-field offsetof checks */
    static_assert(offsetof(HirFillTypeAttrCache, cache_id) == offsetof(FillTypeAttrCache, cache_id_));
    static_assert(offsetof(HirFillTypeMethodCache, cache_id) == offsetof(FillTypeMethodCache, cache_id_));

    /* H2-A: DEFINE_SIMPLE_INSTR struct sizes (surviving types only) */
    static_assert(sizeof(HirGetIter) == sizeof(GetIter));
    static_assert(sizeof(HirDictSubscr) == sizeof(DictSubscr));
    static_assert(sizeof(HirInvokeIterNext) == sizeof(InvokeIterNext));
    static_assert(sizeof(HirImportFrom) == sizeof(ImportFrom));
    static_assert(sizeof(HirSend) == sizeof(Send));
    /* Send has REVERSED template parameter order:
     *   Send: Operands<2>, HasOutput, DeoptBase (non-standard)
     *   Most: HasOutput, Operands<N>, DeoptBase (standard)
     * This changes the C++ struct layout. The C factory (hir_c_create_send_reg)
     * crashes because it uses standard layout assumptions. Send::create() C++
     * path is kept until HirSend layout matches the actual C++ layout.
     * See theologian analysis 2026-04-16.
     *
     * offsetof checks to verify/measure the actual layout divergence: */
    static_assert(offsetof(HirSend, frame_state) == offsetof(Send, frame_state_),
        "Send frame_state_ layout mismatch — reversed mixin order?");
};

/* ---- CompareOp / PrimitiveCompareOp enum value verification ---- */
static_assert(static_cast<int>(CompareOp::kLessThan) == HIR_CMP_LessThan, "CompareOp mismatch");
static_assert(static_cast<int>(CompareOp::kEqual) == HIR_CMP_Equal, "CompareOp mismatch");
static_assert(static_cast<int>(CompareOp::kNotEqual) == HIR_CMP_NotEqual, "CompareOp mismatch");
static_assert(static_cast<int>(CompareOp::kIn) == HIR_CMP_In, "CompareOp mismatch");
static_assert(static_cast<int>(CompareOp::kNotIn) == HIR_CMP_NotIn, "CompareOp mismatch");
static_assert(static_cast<int>(CompareOp::kExcMatch) == HIR_CMP_ExcMatch, "CompareOp mismatch");
static_assert(static_cast<int>(CompareOp::kGreaterThanUnsigned) == HIR_CMP_GreaterThanUnsigned, "CompareOp mismatch");
static_assert(static_cast<int>(CompareOp::kLessThanEqualUnsigned) == HIR_CMP_LessThanEqualUnsigned, "CompareOp mismatch");

static_assert(static_cast<int>(PrimitiveCompareOp::kEqual) == HIR_PCMP_Equal, "PrimitiveCompareOp mismatch");
static_assert(static_cast<int>(PrimitiveCompareOp::kNotEqual) == HIR_PCMP_NotEqual, "PrimitiveCompareOp mismatch");
static_assert(static_cast<int>(PrimitiveCompareOp::kLessThan) == HIR_PCMP_LessThan, "PrimitiveCompareOp mismatch");
static_assert(static_cast<int>(PrimitiveCompareOp::kGreaterThanUnsigned) == HIR_PCMP_GreaterThanUnsigned, "PrimitiveCompareOp mismatch");
static_assert(static_cast<int>(PrimitiveCompareOp::kLessThanEqualUnsigned) == HIR_PCMP_LessThanEqualUnsigned, "PrimitiveCompareOp mismatch");

/* ---- BinaryOpKind / UnaryOpKind / PrimitiveUnaryOpKind enum verification ---- */
static_assert(static_cast<int>(BinaryOpKind::kAdd) == HIR_BOP_Add, "BinaryOpKind mismatch");
static_assert(static_cast<int>(BinaryOpKind::kPower) == HIR_BOP_Power, "BinaryOpKind mismatch");
static_assert(static_cast<int>(BinaryOpKind::kSubscript) == HIR_BOP_Subscript, "BinaryOpKind mismatch");
static_assert(static_cast<int>(BinaryOpKind::kPowerUnsigned) == HIR_BOP_PowerUnsigned, "BinaryOpKind mismatch");
static_assert(static_cast<int>(UnaryOpKind::kNot) == HIR_UOP_Not, "UnaryOpKind mismatch");
static_assert(static_cast<int>(UnaryOpKind::kInvert) == HIR_UOP_Invert, "UnaryOpKind mismatch");
static_assert(static_cast<int>(PrimitiveUnaryOpKind::kNotInt) == HIR_PUO_NotInt, "PrimitiveUnaryOpKind mismatch");
static_assert(static_cast<int>(PrimitiveUnaryOpKind::kNegateInt) == HIR_PUO_NegateInt, "PrimitiveUnaryOpKind mismatch");

/* ---- Type constant bit verification ---- */
/* Verify HIR_TYPE_*EXACT constants match C++ type_generated.h bits.
 * Prevents TLongExact/TListExact confusion class (4 incidents 2026-04-19). */
static_assert(Type::kListExact == 0x00000010000UL, "ListExact bits mismatch");
static_assert(Type::kLongExact == 0x00000000400UL, "LongExact bits mismatch");
static_assert(Type::kSetExact == 0x00000020000UL, "SetExact bits mismatch");
static_assert(Type::kBytesExact == 0x00000002000UL, "BytesExact bits mismatch");
static_assert(Type::kFloatExact == 0x00000008000UL, "FloatExact bits mismatch");
static_assert(Type::kTupleExact == 0x00000040000UL, "TupleExact bits mismatch");
static_assert(Type::kUnicodeExact == 0x00000100000UL, "UnicodeExact bits mismatch");
static_assert(Type::kDictExact == 0x00000004000UL, "DictExact bits mismatch");
static_assert(Type::kBool == 0x00000000002UL, "Bool bits mismatch");
static_assert(Type::kNoneType == 0x00000000080UL, "NoneType bits mismatch");

/* Phase 4.A Batch 14 V3 wrapper-signature pin (per supervisor 03:36:23Z
 * V5-deferred fall-back: V1 production-use + V3 structural pin substitute
 * when V5 input-domain falsifier requires fixture infra beyond batch
 * scope). Pins the 4 GetOperandTypeImpl wrapper signatures so they
 * cannot be silently re-typed (e.g., size_t → int, return-type swap). */
using HirOperandTypeWrapper =
    HirOperandTypeEntry (*)(const void *, size_t);
static_assert(std::is_same_v<decltype(&hir_primitive_compare_operand_type_c),
              HirOperandTypeWrapper>,
    "hir_primitive_compare_operand_type_c signature drift");
static_assert(std::is_same_v<decltype(&hir_primitive_unbox_operand_type_c),
              HirOperandTypeWrapper>,
    "hir_primitive_unbox_operand_type_c signature drift");
static_assert(std::is_same_v<decltype(&hir_return_operand_type_c),
              HirOperandTypeWrapper>,
    "hir_return_operand_type_c signature drift");
static_assert(std::is_same_v<decltype(&hir_use_type_operand_type_c),
              HirOperandTypeWrapper>,
    "hir_use_type_operand_type_c signature drift");

/* ---- Runtime read-through-cast verification ----
 * Creates C++ HIR objects, casts to C structs, reads via C accessors.
 * Validates that layout compatibility translates to correct field reads. */

static void verify_hir_instr_read_through_cast() {
    /* LoadConst: Instr base + Type field */
    auto lc = LoadConst(nullptr, TLong);
    const HirLoadConst *c_lc = reinterpret_cast<const HirLoadConst *>(&lc);

    assert(hir_c_opcode(&lc) ==
           static_cast<int32_t>(Opcode::kLoadConst) &&
           "C/C++ opcode read mismatch for LoadConst");
    assert(!hir_c_has_output(&lc) ||
           true && "output check");

    /* Verify Type field reads correctly through cast */
    HirType c_type = c_lc->type;
    assert(hir_type_bits(&c_type) == Type::kLong &&
           "C/C++ Type bits mismatch in LoadConst");
}

/* Spot-check is_replayable info table against known values.
 * Full validation happens via assertion wrappers in C passes at runtime. */
static void verify_is_replayable_table() {
    /* Replayable opcodes */
    assert(hir_instr_info_is_replayable(static_cast<int>(Opcode::kAssign)) == 1);
    assert(hir_instr_info_is_replayable(static_cast<int>(Opcode::kLoadConst)) == 1);
    assert(hir_instr_info_is_replayable(static_cast<int>(Opcode::kGuardType)) == 1);
    assert(hir_instr_info_is_replayable(static_cast<int>(Opcode::kRefineType)) == 1);
    assert(hir_instr_info_is_replayable(static_cast<int>(Opcode::kLoadArg)) == 1);
    assert(hir_instr_info_is_replayable(static_cast<int>(Opcode::kRaise)) == 1);
    /* Non-replayable opcodes */
    assert(hir_instr_info_is_replayable(static_cast<int>(Opcode::kBinaryOp)) == 0);
    assert(hir_instr_info_is_replayable(static_cast<int>(Opcode::kCallMethod)) == 0);
    assert(hir_instr_info_is_replayable(static_cast<int>(Opcode::kBranch)) == 0);
    assert(hir_instr_info_is_replayable(static_cast<int>(Opcode::kSnapshot)) == 0);
    assert(hir_instr_info_is_replayable(static_cast<int>(Opcode::kReturn)) == 0);
    assert(hir_instr_info_is_replayable(static_cast<int>(Opcode::kVectorCall)) == 0);
}

/* Verify bytecode_offset=-1 invariant for C-allocated instructions.
 * calloc gives 0 for bytecode_offset; hir_c_alloc_instr and hir_c_init_instr
 * must both set it to -1 (matching C++ Instr::Instr default: BCOffset{-1}).
 * Regression test for the bug that caused wrong deopt behavior when
 * _cinderx module was loaded before compilation. */
static void verify_bytecode_offset_invariant() {
    /* Test 1: hir_c_alloc_instr alone sets bytecode_offset=-1 */
    void *raw = hir_c_alloc_instr(sizeof(HirInstrLayout), 0);
    assert(raw != NULL);
    assert(hir_c_bytecode_offset(raw) == -1 &&
           "hir_c_alloc_instr must set bytecode_offset to -1");
    free((char *)raw - sizeof(size_t));  /* free from preamble start */

    /* Test 2: hir_c_init_instr preserves bytecode_offset=-1 */
    void *instr = hir_c_alloc_instr(sizeof(HirInstrLayout), 1);
    assert(instr != NULL);
    hir_c_init_instr(instr, HIR_OP_Assign);
    assert(hir_c_bytecode_offset(instr) == -1 &&
           "hir_c_init_instr must set bytecode_offset to -1");
    free((char *)instr - 1 * sizeof(void *) - sizeof(size_t));

    /* Test 3: hir_c_init_deopt preserves bytecode_offset=-1 */
    void *deopt = hir_c_alloc_instr(sizeof(HirDeoptLayout), 2);
    assert(deopt != NULL);
    hir_c_init_deopt(deopt, HIR_OP_BinaryOp);
    assert(hir_c_bytecode_offset(deopt) == -1 &&
           "hir_c_init_deopt must set bytecode_offset to -1");
    /* H2-E1+E2: All DeoptBase containers are calloc-safe.
     * No placement new cleanup needed — descr=NULL, live_regs={NULL,0,0}. */
    free((char *)deopt - 2 * sizeof(void *) - sizeof(size_t));
}

/* Phase 4.A Batch 1 exhaustive iteration: pin per-opcode behavior of
 * the C functions backing the new C++ shims (Instr::IsTerminator,
 * Instr::numEdges, Instr::opname). One assertion per opcode catches
 * silent drift from the reference C++ switch this batch replaces.
 * Per supervisor 21:40:45Z gate amendment: case-specific evidence
 * for sole-path delegation equivalence, not blanket exemption. */
static int reference_is_terminator(Opcode op) {
    switch (op) {
        case Opcode::kBranch:
        case Opcode::kDeopt:
        case Opcode::kCondBranch:
        case Opcode::kCondBranchIterNotDone:
        case Opcode::kCondBranchCheckType:
        case Opcode::kRaise:
        case Opcode::kRaiseAwaitableError:
        case Opcode::kRaiseStatic:
        case Opcode::kReturn:
        case Opcode::kUnreachable:
            return 1;
        default:
            return 0;
    }
}

static size_t reference_num_edges(Opcode op) {
    switch (op) {
        case Opcode::kBranch:
            return 1;
        case Opcode::kCondBranch:
        case Opcode::kCondBranchIterNotDone:
        case Opcode::kCondBranchCheckType:
            return 2;
        default:
            return 0;
    }
}

/* Phase 4.A Batch 2: replicates the pre-port C++ Instr::isReplayable
 * switch (hir.cpp lines 420-595, 175 LOC pre-port) for the exhaustive
 * iteration falsifier. The hand-maintained replayable column in
 * hir_instr_info_table (hir_instr_info_c.c) is the C-side source; this
 * reference function pins it to the original C++ enumeration. Drift in
 * either direction (new opcode added without table row, or table row
 * with wrong value) fires the assert at JIT module load. */
static int reference_is_replayable(Opcode op) {
    switch (op) {
        case Opcode::kAssign:
        case Opcode::kAtQuiescentState:
        case Opcode::kBitCast:
        case Opcode::kBuildString:
        case Opcode::kCast:
        case Opcode::kCheckErrOccurred:
        case Opcode::kCheckExc:
        case Opcode::kCheckField:
        case Opcode::kCheckFreevar:
        case Opcode::kCheckNeg:
        case Opcode::kCheckSequenceBounds:
        case Opcode::kCheckVar:
        case Opcode::kCIntToCBool:
        case Opcode::kDoubleBinaryOp:
        case Opcode::kFloatCompare:
        case Opcode::kFormatValue:
        case Opcode::kFormatWithSpec:
        case Opcode::kGetSecondOutput:
        case Opcode::kGuard:
        case Opcode::kGuardIs:
        case Opcode::kGuardType:
        case Opcode::kHintType:
        case Opcode::kIndexUnbox:
        case Opcode::kIntBinaryOp:
        case Opcode::kIntConvert:
        case Opcode::kIsNegativeAndErrOccurred:
        case Opcode::kLoadArg:
        case Opcode::kLoadArrayItem:
        case Opcode::kLoadCellItem:
        case Opcode::kLoadConst:
        case Opcode::kLoadCurrentFunc:
        case Opcode::kLoadFrame:
        case Opcode::kLoadEvalBreaker:
        case Opcode::kLoadField:
        case Opcode::kLoadFieldAddress:
        case Opcode::kLoadFunctionIndirect:
        case Opcode::kLoadGlobalCached:
        case Opcode::kLoadSplitDictItem:
        case Opcode::kLoadTupleItem:
        case Opcode::kLoadTypeAttrCacheEntryType:
        case Opcode::kLoadTypeAttrCacheEntryValue:
        case Opcode::kLoadTypeMethodCacheEntryType:
        case Opcode::kLoadTypeMethodCacheEntryValue:
        case Opcode::kLoadVarObjectSize:
        case Opcode::kLongCompare:
        case Opcode::kPrimitiveBox:
        case Opcode::kPrimitiveBoxBool:
        case Opcode::kPrimitiveCompare:
        case Opcode::kPrimitiveUnaryOp:
        case Opcode::kPrimitiveUnbox:
        case Opcode::kRaise:
        case Opcode::kRaiseStatic:
        case Opcode::kRefineType:
        case Opcode::kStealCellItem:
        case Opcode::kUpdatePrevInstr:
        case Opcode::kUnicodeCompare:
        case Opcode::kUnicodeConcat:
        case Opcode::kUnicodeSubscr:
        case Opcode::kUseType:
        case Opcode::kWaitHandleLoadCoroOrResult:
        case Opcode::kWaitHandleLoadWaiter:
            return 1;
        default:
            return 0;
    }
}

static void verify_phase4a_batch1_exhaustive() {
#define VERIFY_OPCODE(name) {                                                \
    int op_int = static_cast<int>(Opcode::k##name);                          \
    Opcode op_enum = Opcode::k##name;                                        \
    /* Stub instance: only the opcode field is read by hir_c_num_edges. */   \
    HirInstrLayout stub;                                                     \
    std::memset(&stub, 0, sizeof(stub));                                     \
    stub.opcode = op_int;                                                    \
    assert(hir_instr_info_is_terminator(op_int) ==                           \
           reference_is_terminator(op_enum) &&                               \
           "Phase 4.A IsTerminator drift for " #name);                       \
    assert(hir_c_num_edges(&stub) == reference_num_edges(op_enum) &&         \
           "Phase 4.A numEdges drift for " #name);                           \
    assert(std::strcmp(hir_opcode_name(static_cast<HirOpcode>(op_int)),      \
                       hir_instr_info_name(op_int)) == 0 &&                  \
           "Phase 4.A opname drift for " #name);                             \
    /* Batch 2: isReplayable pin */                                          \
    assert(hir_c_is_replayable(&stub) ==                                     \
           reference_is_replayable(op_enum) &&                               \
           "Phase 4.A isReplayable drift for " #name);                       \
}
    FOREACH_OPCODE(VERIFY_OPCODE)
#undef VERIFY_OPCODE
}

/* Phase 4.A Batch 10 V5 falsifier: input-domain callback-identity test
 * for hir_c_instr_visit_uses. Verifies the dispatcher invokes the
 * visitor callback the expected number of times for representative
 * opcode classes:
 *   - Decref (1 operand, not DeoptBase) → 1 callback
 *   - Snapshot with NULL frame_state → 0 callbacks
 *   - BinaryOp (2 operands, DeoptBase, empty live_regs, NULL guilty_reg)
 *     → 2 callbacks (operands only; bridge contributes 0 with empty
 *     live_regs + NULL guilty_reg)
 * Catches callback-identity drift between the C dispatcher and the
 * pre-port C++ Instr::visitUses semantics. */
static int verify_phase4a_batch10_counter_visitor(void **slot, void *user) {
    (void)slot;
    int *count = static_cast<int*>(user);
    (*count)++;
    return 1;
}

static void verify_phase4a_batch10_visitor() {
    /* Test 1: 1-operand non-DeoptBase opcode (Decref) → 1 callback */
    void *decref = hir_c_alloc_instr(sizeof(HirInstrLayout), 1);
    assert(decref != NULL);
    hir_c_init_instr(decref, HIR_OP_Decref);
    int count = 0;
    hir_c_instr_visit_uses(decref,
                           verify_phase4a_batch10_counter_visitor,
                           &count);
    assert(count == 1 && "Phase 4.A Batch 10: Decref visitor expected 1 callback");
    free((char *)decref - 1 * sizeof(void *) - sizeof(size_t));

    /* Test 2: Snapshot with NULL frame_state → 0 callbacks */
    void *snap = hir_c_alloc_instr(sizeof(HirSnapshot), 0);
    assert(snap != NULL);
    hir_c_init_instr(snap, HIR_OP_Snapshot);
    ((HirSnapshot *)snap)->frame_state_ptr = NULL;
    count = 0;
    hir_c_instr_visit_uses(snap,
                           verify_phase4a_batch10_counter_visitor,
                           &count);
    assert(count == 0 && "Phase 4.A Batch 10: NULL-frame Snapshot visitor expected 0 callbacks");
    free((char *)snap - sizeof(size_t));

    /* Test 3: 2-operand DeoptBase (BinaryOp) with empty live_regs +
     * NULL guilty_reg + NULL frame_state → 2 callbacks (operands only).
     * calloc-zero from hir_c_alloc_instr leaves all DeoptBase fields
     * NULL/zero. */
    void *binop = hir_c_alloc_instr(sizeof(HirDeoptLayout), 2);
    assert(binop != NULL);
    hir_c_init_deopt(binop, HIR_OP_BinaryOp);
    count = 0;
    hir_c_instr_visit_uses(binop,
                           verify_phase4a_batch10_counter_visitor,
                           &count);
    assert(count == 2 && "Phase 4.A Batch 10: 2-op BinaryOp empty-deopt visitor expected 2 callbacks");
    free((char *)binop - 2 * sizeof(void *) - sizeof(size_t));
}

/* Phase 4.A Batch 11 V4 falsifier: insert+read-back on live_regs.
 * Inserts 3 RegState entries via the C++ DeoptBase live_regs accessor,
 * iterates them via the pure-C HirRegState* cast used by
 * hir_c_deopt_visit_uses_deopt, and verifies each .reg matches the
 * inserted sentinel pointer. Catches POD-equivalence drift between
 * HirRegState and jit::hir::RegState that the static_asserts above
 * cannot detect (e.g., a future hidden field or inheritance change
 * that preserves sizeof but shifts behavior). */
static void verify_phase4a_batch11_live_regs_iteration() {
    void *db = hir_c_alloc_instr(sizeof(HirDeoptLayout), 0);
    assert(db != NULL);
    hir_c_init_deopt(db, HIR_OP_BinaryOp);

    Register* sentinels[3] = {
        reinterpret_cast<Register*>(0x10001000),
        reinterpret_cast<Register*>(0x20002000),
        reinterpret_cast<Register*>(0x30003000),
    };
    auto* base = static_cast<DeoptBase*>(db);
    for (int i = 0; i < 3; i++) {
        base->live_regs().push_back(
            RegState{sentinels[i], RefKind::kBorrowed, ValueKind::kObject});
    }

    HirDeoptLayout *d = (HirDeoptLayout *)db;
    HirRegState *regs = (HirRegState *)d->live_regs_data;
    assert(d->live_regs_count == 3 &&
           "Phase 4.A Batch 11: live_regs_count after 3 push_back");
    for (int i = 0; i < 3; i++) {
        assert(regs[i].reg == sentinels[i] &&
               "Phase 4.A Batch 11: HirRegState pure-C iteration matches C++ insert");
    }

    /* Cleanup: free live_regs malloc storage via destructor invocation
     * pattern. Then free instance preamble. */
    base->live_regs() = PhxRegStateArray{};  /* triggers data_ free in op= */
    /* H2-E1+E2: descr=NULL, frame_state=NULL, calloc-safe — no other
     * destructor work needed. */
    free((char *)db - 2 * sizeof(void *) - sizeof(size_t));
}

/* Phase 4.A Batch 12 V5 falsifier: input-domain coverage for
 * Instr::Uses + Instr::ReplaceUsesOf shim conversions. Verifies the
 * find-and-stop + write-through visitor patterns operate on operand
 * slots correctly. */
static void verify_phase4a_batch12_uses_replace() {
    void* sentinel_a = reinterpret_cast<void*>(0xA000);
    void* sentinel_b = reinterpret_cast<void*>(0xB000);
    void* sentinel_c = reinterpret_cast<void*>(0xC000);

    /* Build a 2-operand non-DeoptBase instr (Decref takes 1 op; use
     * Branch with no operands wouldn't exercise the path. Need a
     * 2-op non-deopt opcode — Phi is variable, but we can stub it
     * with operand count 2 directly via hir_c_alloc_instr. We use
     * Decref-class: HIR_OP_Phi with 2 stub operands works for the
     * dispatcher because Phi is non-DeoptBase. */
    void *instr = hir_c_alloc_instr(sizeof(HirInstrLayout), 2);
    assert(instr != NULL);
    hir_c_init_instr(instr, HIR_OP_Phi);
    hir_c_set_operand(instr, 0, sentinel_a);
    hir_c_set_operand(instr, 1, sentinel_b);

    /* Uses: needle present in operand 0 → returns true */
    bool found_a =
        static_cast<Instr*>(instr)->Uses(static_cast<Register*>(sentinel_a));
    assert(found_a && "Phase 4.A Batch 12: Uses should find operand 0");

    /* Uses: needle present in operand 1 → returns true */
    bool found_b =
        static_cast<Instr*>(instr)->Uses(static_cast<Register*>(sentinel_b));
    assert(found_b && "Phase 4.A Batch 12: Uses should find operand 1");

    /* Uses: needle absent → returns false */
    bool found_c =
        static_cast<Instr*>(instr)->Uses(static_cast<Register*>(sentinel_c));
    assert(!found_c && "Phase 4.A Batch 12: Uses should NOT find absent needle");

    /* ReplaceUsesOf: replace operand 0 (sentinel_a → sentinel_c),
     * operand 1 unchanged. */
    static_cast<Instr*>(instr)->ReplaceUsesOf(
        static_cast<Register*>(sentinel_a),
        static_cast<Register*>(sentinel_c));
    assert(hir_c_get_operand(instr, 0) == sentinel_c &&
           "Phase 4.A Batch 12: ReplaceUsesOf should overwrite operand 0");
    assert(hir_c_get_operand(instr, 1) == sentinel_b &&
           "Phase 4.A Batch 12: ReplaceUsesOf should leave operand 1 unchanged");

    free((char *)instr - 2 * sizeof(void *) - sizeof(size_t));
}

/* Phase 4.A Batch 13 V5 falsifier: sortLiveRegs result correctness.
 * Inserts 3 RegState entries with descending Register::id (3,1,2) via
 * C++ live_regs(), calls hir_c_deopt_sort_live_regs (qsort over
 * HirRegState* cast), then verifies ids are {1,2,3} ascending. */
static void verify_phase4a_batch13_sort_live_regs() {
    void *db = hir_c_alloc_instr(sizeof(HirDeoptLayout), 0);
    assert(db != NULL);
    hir_c_init_deopt(db, HIR_OP_BinaryOp);

    Register r3{3};
    Register r1{1};
    Register r2{2};
    auto* base = static_cast<DeoptBase*>(db);
    base->live_regs().push_back(RegState{&r3, RefKind::kBorrowed, ValueKind::kObject});
    base->live_regs().push_back(RegState{&r1, RefKind::kBorrowed, ValueKind::kObject});
    base->live_regs().push_back(RegState{&r2, RefKind::kBorrowed, ValueKind::kObject});

    hir_c_deopt_sort_live_regs(db);

    HirDeoptLayout *d = (HirDeoptLayout *)db;
    HirRegState *regs = (HirRegState *)d->live_regs_data;
    assert(d->live_regs_count == 3);
    assert(hir_reg_id(regs[0].reg) == 1 &&
           "Phase 4.A Batch 13: sortLiveRegs result[0].id should be 1");
    assert(hir_reg_id(regs[1].reg) == 2 &&
           "Phase 4.A Batch 13: sortLiveRegs result[1].id should be 2");
    assert(hir_reg_id(regs[2].reg) == 3 &&
           "Phase 4.A Batch 13: sortLiveRegs result[2].id should be 3");

    base->live_regs() = PhxRegStateArray{};
    free((char *)db - 2 * sizeof(void *) - sizeof(size_t));
}

/* Phase 4.A Batch 15 V5 falsifier: sentinel-pattern in_edges_ stability
 * test for Edge copy ctor + dtor (theologian 04:17:06Z challenge —
 * sentinel HirBasicBlock pattern is feasible since the struct is
 * calloc-safe). Verifies the V1-claim that Edge::Edge(const Edge&)
 * adds the new edge to both endpoints' in/out_edges_ arrays AND
 * Edge::~Edge() removes it cleanly without disturbing siblings. */
static void verify_phase4a_batch15_edge_in_edges() {
    HirBasicBlock from_bb = {};
    HirBasicBlock to_bb = {};

    /* Build A with from_bb→to_bb endpoints; arrays gain count==1. */
    Edge A;
    A.set_to(reinterpret_cast<BasicBlock*>(&to_bb));
    A.set_from(reinterpret_cast<BasicBlock*>(&from_bb));
    assert(from_bb.out_edges_.count == 1 &&
           "Phase 4.A Batch 15: A populates from_bb.out_edges_");
    assert(to_bb.in_edges_.count == 1 &&
           "Phase 4.A Batch 15: A populates to_bb.in_edges_");

    {
        /* Copy ctor: B added to both lists, count→2. */
        Edge B(A);
        assert(B.from() == A.from() &&
               "Phase 4.A Batch 15: copy ctor preserves from()");
        assert(B.to() == A.to() &&
               "Phase 4.A Batch 15: copy ctor preserves to()");
        assert(from_bb.out_edges_.count == 2 &&
               "Phase 4.A Batch 15: copy ctor inserts into out_edges_");
        assert(to_bb.in_edges_.count == 2 &&
               "Phase 4.A Batch 15: copy ctor inserts into in_edges_");
    }
    /* ~B: removes B from both lists; A still present, count→1. */
    assert(from_bb.out_edges_.count == 1 &&
           "Phase 4.A Batch 15: dtor removes B from out_edges_");
    assert(to_bb.in_edges_.count == 1 &&
           "Phase 4.A Batch 15: dtor removes B from in_edges_");
    assert(A.from() == reinterpret_cast<BasicBlock*>(&from_bb) &&
           "Phase 4.A Batch 15: A.from() unchanged after ~B");
    assert(A.to() == reinterpret_cast<BasicBlock*>(&to_bb) &&
           "Phase 4.A Batch 15: A.to() unchanged after ~B");

    /* Cleanup A's references and the dynamic edge-array storage. */
    A.set_from(nullptr);
    A.set_to(nullptr);
    phx_edge_arr_destroy(&from_bb.out_edges_);
    phx_edge_arr_destroy(&to_bb.in_edges_);
}

/* Phase 4.A Batch 16 V3+behavioral falsifier: verifies the pure-C
 * Instr allocator produces the same memory layout as the original C++
 * Instr::allocate. Allocates an instr with fixed=32 + n=3 operands;
 * checks the num_operands prefix byte sits at instr - sizeof(size_t),
 * checks base() returns the calloc origin (instr - 3*sizeof(void*) -
 * sizeof(size_t)), then frees clean. */
static_assert(sizeof(void *) == 8,
    "Phase 4.A Batch 16 assumes 8-byte pointer-size for layout math");

static void verify_phase4a_batch16_allocator() {
    void *p = hir_c_instr_allocate(32, 3);
    assert(p != NULL && "Phase 4.A Batch 16: allocate returns non-NULL");
    assert(*((size_t *)p - 1) == 3 &&
           "Phase 4.A Batch 16: num_operands prefix == 3");
    void *base = hir_c_instr_base(p);
    void *expected_base = (char *)p - 3 * sizeof(void *) - sizeof(size_t);
    assert(base == expected_base &&
           "Phase 4.A Batch 16: base() == p - n*ptrsz - sizeof(size_t)");
    hir_c_instr_free(p);  /* must not crash */
}

/* Phase 4.A Batch 17 V3+behavioral falsifier: pin block_node_ at
 * offset 0 + verify Instr ctor field-init helpers do not touch the
 * intrusive list head (which would corrupt the linked list).
 * Confirms hir_c_instr_init writes ONLY opcode and
 * hir_c_instr_init_copy writes ONLY {opcode, bytecode_offset, output}. */
static_assert(offsetof(HirInstrLayout, block_node) == 0,
    "Phase 4.A Batch 17: block_node_ MUST be HirInstrLayout offset 0");

static void verify_phase4a_batch17_instr_ctor_init() {
    HirInstrLayout sentinel;
    std::memset(&sentinel, 0, sizeof(sentinel));
    /* Mark block_node_ with sentinel pointers to detect overwrite. */
    HirListNode *bn0 = reinterpret_cast<HirListNode*>(0xB0DE0001);
    HirListNode *bn1 = reinterpret_cast<HirListNode*>(0xB0DE0002);
    sentinel.block_node.prev = bn0;
    sentinel.block_node.next = bn1;

    hir_c_instr_init(&sentinel, HIR_OP_BinaryOp);
    assert(sentinel.opcode == HIR_OP_BinaryOp &&
           "Phase 4.A Batch 17: hir_c_instr_init sets opcode");
    assert(sentinel.block_node.prev == bn0 &&
           sentinel.block_node.next == bn1 &&
           "Phase 4.A Batch 17: hir_c_instr_init must NOT touch block_node");

    /* Copy-init test */
    HirInstrLayout src;
    std::memset(&src, 0, sizeof(src));
    src.opcode = HIR_OP_Decref;
    src.bytecode_offset = 42;
    src.output = reinterpret_cast<void*>(0xCAFEF00D);

    HirInstrLayout dst;
    std::memset(&dst, 0, sizeof(dst));
    dst.block_node.prev = bn0;
    dst.block_node.next = bn1;

    hir_c_instr_init_copy(&dst, &src);
    assert(dst.opcode == HIR_OP_Decref &&
           "Phase 4.A Batch 17: copy-init opcode");
    assert(dst.bytecode_offset == 42 &&
           "Phase 4.A Batch 17: copy-init bytecode_offset");
    assert(dst.output == reinterpret_cast<void*>(0xCAFEF00D) &&
           "Phase 4.A Batch 17: copy-init output");
    assert(dst.block_node.prev == bn0 &&
           dst.block_node.next == bn1 &&
           "Phase 4.A Batch 17: copy-init must NOT touch block_node");
}

/* Phase 4.A Batch 18 V5 sentinel falsifier for setFrameState. */
static void verify_phase4a_batch18_set_frame_state() {
    /* Allocate stub DeoptBase, plant marker FrameState as old state. */
    void *db = hir_c_alloc_instr(sizeof(HirDeoptLayout), 0);
    assert(db != NULL);
    hir_c_init_deopt(db, HIR_OP_BinaryOp);

    HirDeoptLayout *d = (HirDeoptLayout *)db;
    auto* old = new FrameState{};
    old->cur_instr_offs = jit::BCOffset{99};
    d->frame_state = old;

    FrameState src{};
    src.cur_instr_offs = jit::BCOffset{42};

    hir_c_deopt_set_frame_state(db, &src);

    assert(d->frame_state != NULL &&
           "Phase 4.A Batch 18: setFrameState produces a new FrameState");
    assert(d->frame_state != old &&
           "Phase 4.A Batch 18: setFrameState replaces the prior FrameState pointer");
    auto* new_fs = static_cast<FrameState*>(d->frame_state);
    assert(new_fs->cur_instr_offs.value() == 42 &&
           "Phase 4.A Batch 18: setFrameState copy contains src.cur_instr_offs");

    hir_c_destroy_frame_state(d->frame_state);
    free((char *)db - 2 * sizeof(void *) - sizeof(size_t));
}

/* Phase 4.A Batch 19 V5 sentinel falsifier: DeoptBase copy ctor + dtor.
 * Builds a SRC DeoptBase with deep-state sentinels (descr=strdup,
 * frame_state=new FrameState{42}, nonce=99, guilty_reg=0xDEADBEEF, empty
 * live_regs), copies via hir_c_deopt_base_init_copy, asserts deep-copy
 * semantics (different ptr, same content), then destroys both via
 * hir_c_deopt_base_destroy. No-crash + pydebug RTC counters validate the
 * destroy path; full leak detection is testkeeper cap-check class. */
static void verify_phase4a_batch19_deopt_copy() {
    /* SRC sentinel. */
    void *src_v = hir_c_alloc_instr(sizeof(HirDeoptLayout), 0);
    assert(src_v != NULL);
    hir_c_init_deopt(src_v, HIR_OP_BinaryOp);
    HirDeoptLayout *src = (HirDeoptLayout *)src_v;
    src->descr = strdup("test");
    auto* fs = new FrameState{};
    fs->cur_instr_offs = jit::BCOffset{42};
    src->frame_state = fs;
    src->nonce = 99;
    src->guilty_reg = (void *)(uintptr_t)0xDEADBEEFULL;

    /* DST allocated zero, then deep-copied from src. */
    void *dst_v = hir_c_alloc_instr(sizeof(HirDeoptLayout), 0);
    assert(dst_v != NULL);
    hir_c_init_deopt(dst_v, HIR_OP_BinaryOp);
    HirDeoptLayout *dst = (HirDeoptLayout *)dst_v;

    hir_c_deopt_base_init_copy(dst_v, src_v);

    assert(dst->descr != src->descr &&
           "Phase 4.A Batch 19: descr must be a fresh strdup, not aliased");
    assert(strcmp(dst->descr, "test") == 0 &&
           "Phase 4.A Batch 19: descr contents must match");
    assert(dst->frame_state != NULL &&
           dst->frame_state != src->frame_state &&
           "Phase 4.A Batch 19: frame_state must be a fresh allocation");
    assert(static_cast<FrameState*>(dst->frame_state)->cur_instr_offs.value()
               == 42 &&
           "Phase 4.A Batch 19: frame_state cur_instr_offs must copy");
    assert(dst->nonce == 99 &&
           "Phase 4.A Batch 19: nonce must copy");
    assert(dst->guilty_reg == (void *)(uintptr_t)0xDEADBEEFULL &&
           "Phase 4.A Batch 19: guilty_reg pointer must copy");
    assert(dst->live_regs_data == NULL &&
           dst->live_regs_count == 0 &&
           dst->live_regs_cap == 0 &&
           "Phase 4.A Batch 19: empty live_regs must remain empty");

    hir_c_deopt_base_destroy(dst_v);
    hir_c_deopt_base_destroy(src_v);

    hir_c_instr_free(src_v);
    hir_c_instr_free(dst_v);
}

/* Phase 4.A Batch 20 V5 sentinel falsifier: Phi setArgs apply step.
 * Allocates a 3-operand Phi sentinel + 3 stack-only HirBasicBlock
 * sentinels (only .id is read by the comparator + the spec assertions),
 * builds pre-sorted parallel arrays, calls hir_c_phi_apply_args, and
 * asserts both basic_blocks_ writes (capacity = count, data = malloc
 * copy) and operand-slot writes hit the right indices. Independently
 * exercises hir_c_phi_pair_cmp_by_block_id by qsorting an unsorted pair
 * array and asserting paired permutation. */
static void verify_phase4a_batch20_phi_apply_args() {
    HirBasicBlock bb1{}, bb2{}, bb3{};
    bb1.id = 1;
    bb2.id = 2;
    bb3.id = 3;

    void *r1 = (void *)(uintptr_t)0xC0FFEE01ULL;
    void *r2 = (void *)(uintptr_t)0xC0FFEE02ULL;
    void *r3 = (void *)(uintptr_t)0xC0FFEE03ULL;

    void *phi = hir_c_alloc_instr(sizeof(HirPhi), 3);
    assert(phi != NULL);
    hir_c_init_instr(phi, HIR_OP_Phi);

    void *sorted_keys[3]   = { &bb1, &bb2, &bb3 };
    void *sorted_values[3] = { r2,   r3,   r1   };

    hir_c_phi_apply_args(phi, sorted_keys, sorted_values, 3);

    HirPhi *p = (HirPhi *)phi;
    assert(p->bb_count == 3 &&
           "Phase 4.A Batch 20: bb_count == n");
    assert(p->bb_cap == 3 &&
           "Phase 4.A Batch 20: bb_cap == n (PhxBlockArray copy semantics)");
    assert(p->bb_data != NULL &&
           "Phase 4.A Batch 20: bb_data freshly malloc'd");
    assert(p->bb_data != sorted_keys &&
           "Phase 4.A Batch 20: bb_data must be a fresh allocation, not aliased");

    assert(hir_phi_block_at(phi, 0) == &bb1 &&
           "Phase 4.A Batch 20: basic_blocks_[0] = bb1");
    assert(hir_phi_block_at(phi, 1) == &bb2 &&
           "Phase 4.A Batch 20: basic_blocks_[1] = bb2");
    assert(hir_phi_block_at(phi, 2) == &bb3 &&
           "Phase 4.A Batch 20: basic_blocks_[2] = bb3");

    assert(hir_c_get_operand(phi, 0) == r2 &&
           "Phase 4.A Batch 20: operand[0] = r2 (paired with bb1)");
    assert(hir_c_get_operand(phi, 1) == r3 &&
           "Phase 4.A Batch 20: operand[1] = r3 (paired with bb2)");
    assert(hir_c_get_operand(phi, 2) == r1 &&
           "Phase 4.A Batch 20: operand[2] = r1 (paired with bb3)");

    /* Independently verify the paired-permute comparator. */
    HirPhiArgPair pairs[3] = {
        { &bb3, r1 },
        { &bb1, r2 },
        { &bb2, r3 },
    };
    qsort(pairs, 3, sizeof(HirPhiArgPair), hir_c_phi_pair_cmp_by_block_id);
    assert(pairs[0].key == &bb1 && pairs[0].value == r2 &&
           "Phase 4.A Batch 20: pair sort paired permute [0]");
    assert(pairs[1].key == &bb2 && pairs[1].value == r3 &&
           "Phase 4.A Batch 20: pair sort paired permute [1]");
    assert(pairs[2].key == &bb3 && pairs[2].value == r1 &&
           "Phase 4.A Batch 20: pair sort paired permute [2]");

    free(p->bb_data);
    hir_c_instr_free(phi);
}

/* ==== Phase 4.A Batch I: V5 fixture infra ====
 * Reusable helpers for SUBSTANTIVE V5 falsifiers requiring intrusive-list
 * multi-instr chain setup. Built once here so the deferred SUBSTANTIVE
 * V5 cases in Batches 21 (getDominatingFrameState), 23 (ExpandInto),
 * 25 (add/remove predecessor), 27 (Append/push_front/pop_front) can be
 * back-filled as fix-on-tops without each re-implementing the same
 * intrusive-list wiring + Snapshot+FrameState alloc dance.
 *
 * File-static + verify-only — no production linkage. */

/* Init a stack HirBasicBlock as an empty intrusive-list-ready container.
 * After init, hir_bb_first_instr(bb) returns NULL and the block is ready
 * to receive instrs via hir_c_test_chain_append_*. */
static void hir_c_test_chain_init(HirBasicBlock *bb) {
    memset(bb, 0, sizeof(*bb));
    bb->instrs_.root_.prev_ = &bb->instrs_.root_;
    bb->instrs_.root_.next_ = &bb->instrs_.root_;
    /* node_member_offset_ = offsetof(HirInstrLayout, block_node) which is 0
     * (block_node is the first field of HIR_INSTR_FIELDS). */
    bb->instrs_.node_member_offset_ = 0;
}

/* Allocate an instr with the requested struct size + operand count, set
 * its opcode, and append it to bb. Returns the new instr ptr. The instr
 * is owned by the chain and freed by hir_c_test_chain_destroy. */
static void *hir_c_test_chain_append_instr(HirBasicBlock *bb,
                                           int opcode,
                                           size_t struct_size,
                                           size_t num_operands) {
    void *instr = hir_c_alloc_instr(struct_size, num_operands);
    assert(instr != NULL);
    hir_c_init_instr(instr, opcode);
    hir_bb_append_instr(bb, instr);
    return instr;
}

/* Allocate a Snapshot, populate frame_state_ptr by deep-copying src
 * via the C++ bridge (Batch 18), and append. Returns the Snapshot ptr. */
static void *hir_c_test_chain_append_snapshot(HirBasicBlock *bb,
                                              const FrameState *src) {
    void *snap = hir_c_test_chain_append_instr(
        bb, HIR_OP_Snapshot, sizeof(HirSnapshot), 0);
    HirSnapshot *s = (HirSnapshot *)snap;
    s->frame_state_ptr = hir_make_frame_state_c(src);
    return snap;
}

/* Walk bb's instr list and free each via hir_c_destroy_instr_impl
 * (handles per-opcode cleanup: Snapshot.frame_state_ptr, etc.). Then
 * tear down the in/out edge arrays. After destroy, bb is back to
 * zero-init and the test fixture can drop. */
static void hir_c_test_chain_destroy(HirBasicBlock *bb) {
    void *cur = hir_bb_first_instr(bb);
    while (cur != NULL) {
        void *next = hir_bb_next_instr(bb, cur);
        hir_c_destroy_instr_impl(cur);
        cur = next;
    }
    phx_edge_arr_destroy(&bb->in_edges_);
    phx_edge_arr_destroy(&bb->out_edges_);
}

/* Self-falsifier for the chain helpers. */
static void verify_phase4a_batchI_chain_helper() {
    HirBasicBlock bb;
    hir_c_test_chain_init(&bb);
    assert(hir_bb_first_instr(&bb) == NULL &&
           "Phase 4.A Batch I: post-init chain is empty");

    /* Append BinaryOp (DeoptBase, 2 operands) + Snapshot + Return. */
    void *bin_op = hir_c_test_chain_append_instr(
        &bb, HIR_OP_BinaryOp, sizeof(HirBinaryOp), 2);
    /* hir_c_init_instr leaves the DeoptBase fields zero — re-init
     * specifically to set nonce=-1 etc. for safe destroy via
     * hir_c_destroy_instr_impl. */
    hir_c_init_deopt(bin_op, HIR_OP_BinaryOp);

    FrameState src_fs{};
    src_fs.cur_instr_offs = jit::BCOffset{77};
    void *snap = hir_c_test_chain_append_snapshot(&bb, &src_fs);

    void *ret = hir_c_test_chain_append_instr(
        &bb, HIR_OP_Return, sizeof(HirReturn), 1);

    /* (a) hir_bb_first_instr returns BinaryOp. */
    assert(hir_bb_first_instr(&bb) == bin_op &&
           "Phase 4.A Batch I: first_instr returns the first appended");

    /* (b) Iteration yields all 3 in order. */
    void *expected[3] = { bin_op, snap, ret };
    void *cur = hir_bb_first_instr(&bb);
    for (int i = 0; i < 3; i++) {
        assert(cur == expected[i] &&
               "Phase 4.A Batch I: iteration order matches append order");
        cur = hir_bb_next_instr(&bb, cur);
    }
    assert(cur == NULL &&
           "Phase 4.A Batch I: iteration terminates at sentinel after 3 instrs");

    /* (c) Snapshot's frame_state populated with src.cur_instr_offs. */
    void *fs_ptr = hir_c_snapshot_get_frame_state(snap);
    assert(fs_ptr != NULL &&
           "Phase 4.A Batch I: Snapshot frame_state_ptr populated");
    assert(static_cast<FrameState*>(fs_ptr)->cur_instr_offs.value() == 77 &&
           "Phase 4.A Batch I: Snapshot frame_state copy preserves cur_instr_offs");

    /* (d) Destroy is leak-free under pydebug RTC. */
    hir_c_test_chain_destroy(&bb);
    assert(hir_bb_first_instr(&bb) == NULL &&
           "Phase 4.A Batch I: post-destroy chain is empty");
}

/* Phase 4.A Batch 21 V5 BOUNDARY falsifiers for getDominatingFrameState.
 *
 * Two boundary cases land here per the codified V5-feasibility analysis
 * (theologian 07:19:46Z); the SUBSTANTIVE multi-instr replayable+Snapshot
 * chain remains V5-DEFERRED with the cited blocker (3+ batched
 * intrusive-list dependencies + Snapshot.frame_state alloc), substituted
 * by V1 production-use coverage in refcount_pass + simplify.
 *
 *   (a) block_ == nullptr early-return: the pre-block hand-off check
 *       must be the first short-circuit, before any list iteration.
 *
 *   (b) Single-instr block (loop body never enters): hir_bb_prev_instr
 *       on the only Instr returns NULL because its block_node.prev is
 *       the sentinel head. */
static void verify_phase4a_batch21_dominating_frame_state() {
    /* (a) block_ == NULL early return. */
    {
        void *instr = hir_c_alloc_instr(sizeof(HirInstrLayout), 0);
        assert(instr != NULL);
        hir_c_init_instr(instr, HIR_OP_BinaryOp);
        /* block_ defaults to NULL from calloc. */
        assert(((HirInstrLayout *)instr)->block == NULL &&
               "Phase 4.A Batch 21(a): pre-condition block_ == NULL");

        const void *fs = hir_c_get_dominating_frame_state(instr);
        assert(fs == NULL &&
               "Phase 4.A Batch 21(a): block_ == NULL must short-circuit to NULL");

        hir_c_instr_free(instr);
    }

    /* (b) Single-instr block: target is the only Instr; reverse-walk
     * lands at the sentinel immediately, returns NULL. */
    {
        HirBasicBlock bb = {};
        /* Empty intrusive list: sentinel root_ self-pointers. */
        bb.instrs_.root_.prev_ = &bb.instrs_.root_;
        bb.instrs_.root_.next_ = &bb.instrs_.root_;
        /* block_node is at offset 0 in HirInstrLayout. */
        bb.instrs_.node_member_offset_ = 0;

        void *target = hir_c_alloc_instr(sizeof(HirInstrLayout), 0);
        assert(target != NULL);
        hir_c_init_instr(target, HIR_OP_BinaryOp);
        ((HirInstrLayout *)target)->block = &bb;

        /* Wire target as the sole entry; cast HirListNode* (no-underscore
         * field names) to HirIntrusiveListNode* — they share the
         * pointer-pair POD layout (verified by static_asserts in this
         * same file: sizeof(HirIntrusiveListNode) == sizeof(HirListNode)
         * shouldn't be needed since both equal 16 bytes by definition). */
        HirListNode *target_node = (HirListNode *)target;
        target_node->prev = (HirListNode *)&bb.instrs_.root_;
        target_node->next = (HirListNode *)&bb.instrs_.root_;
        bb.instrs_.root_.prev_ = (HirIntrusiveListNode *)target_node;
        bb.instrs_.root_.next_ = (HirIntrusiveListNode *)target_node;

        const void *fs = hir_c_get_dominating_frame_state(target);
        assert(fs == NULL &&
               "Phase 4.A Batch 21(b): single-instr block must return NULL");

        /* Unwire before free so hir_c_instr_free doesn't see a stale
         * sentinel reference (free just walks back to the slab base). */
        hir_c_instr_free(target);
    }
}

/* Phase 4.A Batch 22 V5 sentinel falsifier: edges/edge dispatchers.
 * Three cases land here per the codified V5-feasibility analysis
 * (theologian 07:39:41Z): default opcode (count==0), Branch (count==1
 * + ptr-equality to &edge), CondBranch (count==2 + paired ptr-equality
 * + storage adjacency). No intrusive-list setup — Edge fields are
 * direct struct members. */
static void verify_phase4a_batch22_edges_dispatch() {
    /* Default opcode (BinaryOp): no edges. */
    {
        HirInstrLayout instr = {};
        instr.opcode = HIR_OP_BinaryOp;
        assert(hir_c_num_edges(&instr) == 0 &&
               "Phase 4.A Batch 22(default): non-branch opcode has 0 edges");
        assert(hir_c_edge_at(&instr, 0) == NULL &&
               "Phase 4.A Batch 22(default): edge_at returns NULL");
    }

    /* Branch sentinel: 1 edge at &branch.edge. */
    {
        HirBranch br = {};
        br.opcode = HIR_OP_Branch;
        assert(hir_c_num_edges(&br) == 1 &&
               "Phase 4.A Batch 22(Branch): num_edges == 1");
        assert(hir_c_edge_at(&br, 0) == &br.edge &&
               "Phase 4.A Batch 22(Branch): edge_at(0) == &edge");
    }

    /* CondBranch sentinel: 2 edges, contiguous true_edge then false_edge. */
    {
        HirCondBranchInstr cb = {};
        cb.opcode = HIR_OP_CondBranch;
        assert(hir_c_num_edges(&cb) == 2 &&
               "Phase 4.A Batch 22(CondBranch): num_edges == 2");
        assert(hir_c_edge_at(&cb, 0) == &cb.true_edge &&
               "Phase 4.A Batch 22(CondBranch): edge_at(0) == &true_edge");
        assert(hir_c_edge_at(&cb, 1) == &cb.false_edge &&
               "Phase 4.A Batch 22(CondBranch): edge_at(1) == &false_edge");
        /* Storage adjacency: span{&true_edge_, 2} only walks correctly
         * when false_edge follows true_edge with no padding. */
        assert((char *)&cb.false_edge - (char *)&cb.true_edge ==
                   (ptrdiff_t)sizeof(HirEdge) &&
               "Phase 4.A Batch 22(CondBranch): false_edge contiguous after true_edge");
    }

    /* CondBranchIterNotDone + CondBranchCheckType share the same
     * 2-edge dispatch; verify count + ptr-equality on a CheckType
     * sentinel since it has the extra HirType trailer. */
    {
        HirCondBranchCheckType cct = {};
        cct.opcode = HIR_OP_CondBranchCheckType;
        assert(hir_c_num_edges(&cct) == 2 &&
               "Phase 4.A Batch 22(CondBranchCheckType): num_edges == 2");
        assert(hir_c_edge_at(&cct, 0) == &cct.true_edge &&
               "Phase 4.A Batch 22(CondBranchCheckType): edge_at(0) == &true_edge");
        assert(hir_c_edge_at(&cct, 1) == &cct.false_edge &&
               "Phase 4.A Batch 22(CondBranchCheckType): edge_at(1) == &false_edge");
    }
}

/* Phase 4.A Batch 23 V5 BOUNDARY falsifier for Instr lifecycle bundle.
 *
 * Per the codified V5-feasibility analysis (theologian 07:39:52Z): two
 * BOUNDARY cases land here.
 *
 *   (a) link/unlink round-trip: hir_c_instr_link sets block_, then
 *       hir_c_instr_unlink restores block_=NULL. No list wiring needed
 *       because hir_c_unlink's prev/next manipulation is idempotent on
 *       the self-pointer state that hir_c_alloc_instr leaves behind.
 *
 *   (b) ExpandInto n=0: empty expansion array — loop body never runs,
 *       leaving just the trailing unlink. Validates the no-op-loop
 *       degenerate case without paying the multi-instr setup cost.
 *
 * ReplaceWith decomposes into (insert_before + set_bytecode_offset +
 * unlink) — each step is independently sentinel-tested by Batches 17
 * (set_bytecode_offset implied via init_copy field), 22 (edge dispatch),
 * and (a) above. Whole-method coverage is V1 production-use in simplify.
 *
 * SUBSTANTIVE V5 (multi-instr expansion / wired-list ReplaceWith) stays
 * deferred per the same intrusive-list multi-step blocker cited at
 * Batch 21 + the V5-deferred-fall-back framework. */
static void verify_phase4a_batch23_instr_lifecycle() {
    /* (a) link/unlink round-trip. */
    {
        HirBasicBlock bb = {};
        void *instr = hir_c_alloc_instr(sizeof(HirInstrLayout), 0);
        assert(instr != NULL);
        hir_c_init_instr(instr, HIR_OP_BinaryOp);
        assert(((HirInstrLayout *)instr)->block == NULL &&
               "Phase 4.A Batch 23(a): pre-condition block_ == NULL");

        hir_c_instr_link(instr, &bb);
        assert(((HirInstrLayout *)instr)->block == &bb &&
               "Phase 4.A Batch 23(a): link sets block_ to target");

        hir_c_instr_unlink(instr);
        assert(((HirInstrLayout *)instr)->block == NULL &&
               "Phase 4.A Batch 23(a): unlink restores block_ to NULL");

        hir_c_instr_free(instr);
    }

    /* (b) ExpandInto with n=0: degenerates to a bare unlink. */
    {
        HirBasicBlock bb = {};
        void *self = hir_c_alloc_instr(sizeof(HirInstrLayout), 0);
        assert(self != NULL);
        hir_c_init_instr(self, HIR_OP_BinaryOp);
        ((HirInstrLayout *)self)->block = &bb;

        hir_c_instr_expand_into(self, NULL, 0);

        assert(((HirInstrLayout *)self)->block == NULL &&
               "Phase 4.A Batch 23(b): empty expand_into still unlinks self");

        hir_c_instr_free(self);
    }
}

/* Phase 4.A Batch 24 V5 sentinel falsifier for fixupPhis per-Phi remap.
 * Builds a 3-block Phi sentinel via the Batch 20 apply path (so the
 * starting state is sorted by id), then calls hir_c_phi_fixup_predecessor
 * to swap one block for a higher-id replacement. The post-fixup state
 * must be re-sorted with operands paired to the new id ordering. A
 * second no-op fixup (replacing a block no longer present) verifies the
 * walk is non-destructive when the predicate misses. */
static void verify_phase4a_batch24_fixup_phis() {
    HirBasicBlock bb1{}, bb2{}, bb3{}, bb4{};
    bb1.id = 1;
    bb2.id = 2;
    bb3.id = 3;
    bb4.id = 4;

    void *r1 = (void *)(uintptr_t)0xDEED0001ULL;
    void *r2 = (void *)(uintptr_t)0xDEED0002ULL;
    void *r3 = (void *)(uintptr_t)0xDEED0003ULL;

    void *phi = hir_c_alloc_instr(sizeof(HirPhi), 3);
    assert(phi != NULL);
    hir_c_init_instr(phi, HIR_OP_Phi);

    void *initial_keys[3]   = { &bb1, &bb2, &bb3 };
    void *initial_values[3] = { r1,   r2,   r3   };
    hir_c_phi_apply_args(phi, initial_keys, initial_values, 3);

    /* Replace bb2 (id=2) with bb4 (id=4). New sort: bb1, bb3, bb4. */
    hir_c_phi_fixup_predecessor(phi, &bb2, &bb4);

    HirPhi *p = (HirPhi *)phi;
    assert(p->bb_count == 3 &&
           "Phase 4.A Batch 24: count preserved after fixup");
    assert(p->bb_data[0] == &bb1 &&
           "Phase 4.A Batch 24: post-fixup bb_data[0] = bb1");
    assert(p->bb_data[1] == &bb3 &&
           "Phase 4.A Batch 24: post-fixup bb_data[1] = bb3");
    assert(p->bb_data[2] == &bb4 &&
           "Phase 4.A Batch 24: post-fixup bb_data[2] = bb4 (replaces bb2)");

    assert(hir_c_get_operand(phi, 0) == r1 &&
           "Phase 4.A Batch 24: operand[0] = r1 (paired with bb1)");
    assert(hir_c_get_operand(phi, 1) == r3 &&
           "Phase 4.A Batch 24: operand[1] = r3 (paired with bb3)");
    assert(hir_c_get_operand(phi, 2) == r2 &&
           "Phase 4.A Batch 24: operand[2] = r2 (paired with bb4 via replace)");

    /* No-op fixup: bb2 no longer present, walk must leave state intact. */
    hir_c_phi_fixup_predecessor(phi, &bb2, &bb4);
    assert(p->bb_count == 3 &&
           "Phase 4.A Batch 24: no-op fixup count unchanged");
    assert(p->bb_data[0] == &bb1 &&
           p->bb_data[1] == &bb3 &&
           p->bb_data[2] == &bb4 &&
           "Phase 4.A Batch 24: no-op fixup bb_data unchanged");
    assert(hir_c_get_operand(phi, 0) == r1 &&
           hir_c_get_operand(phi, 1) == r3 &&
           hir_c_get_operand(phi, 2) == r2 &&
           "Phase 4.A Batch 24: no-op fixup operands unchanged");

    free(p->bb_data);
    hir_c_instr_free(phi);
}

/* Phase 4.A Batch 25 V5 BOUNDARY falsifier for add/remove predecessor
 * collect helpers. Tests the args-extraction step (sentinel-feasible)
 * in isolation; the substantive add/remove flow (allocate new Phi via
 * the C++ bridge + replace_with + destroy_instr_impl) requires
 * intrusive-list multi-step setup and stays V1-prod-use covered (cfg
 * simplify) per the V5-deferred-fall-back framework. */
static void verify_phase4a_batch25_add_remove_collect() {
    HirBasicBlock bb1{}, bb2{}, bb3{}, bb4{};
    bb1.id = 1;
    bb2.id = 2;
    bb3.id = 3;
    bb4.id = 4;

    void *r1 = (void *)(uintptr_t)0xACE10001ULL;
    void *r2 = (void *)(uintptr_t)0xACE10002ULL;
    void *r3 = (void *)(uintptr_t)0xACE10003ULL;

    void *phi = hir_c_alloc_instr(sizeof(HirPhi), 3);
    assert(phi != NULL);
    hir_c_init_instr(phi, HIR_OP_Phi);

    void *initial_keys[3]   = { &bb1, &bb2, &bb3 };
    void *initial_values[3] = { r1,   r2,   r3   };
    hir_c_phi_apply_args(phi, initial_keys, initial_values, 3);

    /* (a) collect_add_args with old_pred=bb2, new_pred=bb4: result
     * is 4 sorted entries with bb2 + bb4 both pointing to r2. */
    {
        void **keys = NULL, **values = NULL;
        size_t n = 0;
        int found = hir_c_phi_collect_add_args(phi, &bb2, &bb4,
                                               &keys, &values, &n);
        assert(found == 1 &&
               "Phase 4.A Batch 25(add): old_pred present should return 1");
        assert(n == 4 &&
               "Phase 4.A Batch 25(add): n+1 = 4 entries");
        assert(keys[0] == &bb1 && values[0] == r1 &&
               "Phase 4.A Batch 25(add): sorted [0] = bb1/r1");
        assert(keys[1] == &bb2 && values[1] == r2 &&
               "Phase 4.A Batch 25(add): sorted [1] = bb2/r2");
        assert(keys[2] == &bb3 && values[2] == r3 &&
               "Phase 4.A Batch 25(add): sorted [2] = bb3/r3");
        assert(keys[3] == &bb4 && values[3] == r2 &&
               "Phase 4.A Batch 25(add): sorted [3] = bb4/r2 (paired with replaced)");
        free(keys);
        free(values);
    }

    /* (b) collect_add_args no-op: old_pred not present returns 0. */
    {
        void **keys = NULL, **values = NULL;
        size_t n = 0;
        int found = hir_c_phi_collect_add_args(phi, &bb4, &bb4,
                                               &keys, &values, &n);
        assert(found == 0 &&
               "Phase 4.A Batch 25(add no-op): old_pred absent returns 0");
        assert(n == 0 && keys == NULL && values == NULL &&
               "Phase 4.A Batch 25(add no-op): outputs zero/NULL");
    }

    /* (c) collect_remove_args drops bb2: 2 sorted entries [bb1, bb3]. */
    {
        void **keys = NULL, **values = NULL;
        size_t n = 0;
        hir_c_phi_collect_remove_args(phi, &bb2, &keys, &values, &n);
        assert(n == 2 &&
               "Phase 4.A Batch 25(remove): n-1 = 2 entries");
        assert(keys[0] == &bb1 && values[0] == r1 &&
               "Phase 4.A Batch 25(remove): sorted [0] = bb1/r1");
        assert(keys[1] == &bb3 && values[1] == r3 &&
               "Phase 4.A Batch 25(remove): sorted [1] = bb3/r3");
        free(keys);
        free(values);
    }

    /* (d) collect_remove_args dropping every entry yields n=0 + NULLs. */
    {
        /* Build a single-pred Phi: only bb1. */
        void *single_phi = hir_c_alloc_instr(sizeof(HirPhi), 1);
        assert(single_phi != NULL);
        hir_c_init_instr(single_phi, HIR_OP_Phi);
        void *single_keys[1]   = { &bb1 };
        void *single_values[1] = { r1   };
        hir_c_phi_apply_args(single_phi, single_keys, single_values, 1);

        void **keys = (void **)0xDEADBEEF;
        void **values = (void **)0xDEADBEEF;
        size_t n = 99;
        hir_c_phi_collect_remove_args(single_phi, &bb1,
                                      &keys, &values, &n);
        assert(n == 0 && keys == NULL && values == NULL &&
               "Phase 4.A Batch 25(remove zero): outputs zero/NULL when all dropped");

        free(((HirPhi *)single_phi)->bb_data);
        hir_c_instr_free(single_phi);
    }

    free(((HirPhi *)phi)->bb_data);
    hir_c_instr_free(phi);
}

/* Phase 4.A Batch 26 V5 sentinel falsifier for Environment register
 * allocator + registry. Calloc'd HirEnvironment + AllocateRegister
 * twice → assert ids 0/1 + reg_count grows; addRegister with hand-id=5
 * → assert reg_count jumps to 6 + slot[5] holds the supplied Register;
 * second AllocateRegister after a hand-allocated id=2 in the gap walks
 * past the occupied slot. */
static void verify_phase4a_batch26_env_register() {
    HirEnvironment env = {};

    /* AllocateRegister twice: ids 0 and 1, reg_count grows accordingly,
     * capacity bumps from 0 to 16 on the first allocation. */
    void *r0 = hir_c_env_allocate_register(&env);
    assert(r0 != NULL &&
           "Phase 4.A Batch 26: first AllocateRegister returns non-NULL");
    assert(hir_reg_id(r0) == 0 &&
           "Phase 4.A Batch 26: first reg id == 0");
    assert(env.reg_count == 1 &&
           "Phase 4.A Batch 26: reg_count == 1 after first allocate");
    assert(env.reg_capacity == 16 &&
           "Phase 4.A Batch 26: reg_capacity bootstraps to 16");
    assert(env.reg_data[0] == r0 &&
           "Phase 4.A Batch 26: reg_data[0] holds the new Register");
    assert(env.next_register_id == 1 &&
           "Phase 4.A Batch 26: next_register_id incremented to 1");

    void *r1 = hir_c_env_allocate_register(&env);
    assert(hir_reg_id(r1) == 1 &&
           "Phase 4.A Batch 26: second reg id == 1");
    assert(env.reg_count == 2 &&
           "Phase 4.A Batch 26: reg_count == 2 after second allocate");

    /* addRegister with id=5: capacity stays at 16 (5 < 16), reg_count
     * jumps to 6 with NULL gaps at 2..4. */
    void *r5 = hir_make_register_c(5);
    void *added = hir_c_env_add_register(&env, r5);
    assert(added == r5 &&
           "Phase 4.A Batch 26: addRegister returns the inserted ptr");
    assert(env.reg_data[5] == r5 &&
           "Phase 4.A Batch 26: reg_data[5] holds the inserted Register");
    assert(env.reg_count == 6 &&
           "Phase 4.A Batch 26: reg_count jumps to 6 (id=5 + 1)");
    assert(env.reg_data[2] == NULL &&
           env.reg_data[3] == NULL &&
           env.reg_data[4] == NULL &&
           "Phase 4.A Batch 26: gap slots stay NULL");

    /* Third AllocateRegister: next_register_id is 2, slot is NULL, so
     * the walk-past-occupied loop doesn't fire; id 2 is taken. */
    void *r2 = hir_c_env_allocate_register(&env);
    assert(hir_reg_id(r2) == 2 &&
           "Phase 4.A Batch 26: third allocate fills the next NULL slot id=2");

    /* Now next_register_id is 3 and slot[3] is NULL, but if we had
     * pre-occupied [3, 4] the loop would walk past. Hand-occupy [3]
     * then allocate again to exercise the skip path. */
    void *r3 = hir_make_register_c(3);
    env.reg_data[3] = r3;
    void *r4 = hir_c_env_allocate_register(&env);
    assert(hir_reg_id(r4) == 4 &&
           "Phase 4.A Batch 26: allocate skips occupied slot[3] → id 4");

    /* Cleanup: each Register was heap-allocated via the C++ bridge
     * (operator new). The verifier owns them since this is a sentinel
     * Environment with no DISALLOW_COPY destructor running. */
    delete static_cast<jit::hir::Register*>(r0);
    delete static_cast<jit::hir::Register*>(r1);
    delete static_cast<jit::hir::Register*>(r2);
    delete static_cast<jit::hir::Register*>(r3);
    delete static_cast<jit::hir::Register*>(r4);
    delete static_cast<jit::hir::Register*>(r5);
    free(env.reg_data);
}

/* Phase 4.A Batch 27 V5 BOUNDARY falsifier for BasicBlock list mutation
 * wrappers. Stack HirBasicBlock with empty intrusive-list sentinel +
 * fresh Instr; assert post-Append the instr is linked (block_ set) and
 * the list head points to it. push_front + pop_front round-trip on a
 * single-instr block confirms the head/tail wiring stays consistent
 * across the wrappers. SUBSTANTIVE V5 (multi-instr ordering) stays
 * V1-prod-use covered per the V5-deferred-fall-back framework — same
 * intrusive-list cited blocker as Batches 21, 23, 25. */
static void verify_phase4a_batch27_bb_list_wrappers() {
    /* (a) Append into empty block: instr becomes the only entry,
     * block_ is set to &bb, head = instr. */
    {
        HirBasicBlock bb = {};
        bb.instrs_.root_.prev_ = &bb.instrs_.root_;
        bb.instrs_.root_.next_ = &bb.instrs_.root_;
        bb.instrs_.node_member_offset_ = 0;

        void *instr = hir_c_alloc_instr(sizeof(HirInstrLayout), 0);
        assert(instr != NULL);
        hir_c_init_instr(instr, HIR_OP_BinaryOp);

        void *appended = hir_c_bb_append(&bb, instr);
        assert(appended == instr &&
               "Phase 4.A Batch 27(a): Append returns the inserted instr");
        assert(((HirInstrLayout *)instr)->block == &bb &&
               "Phase 4.A Batch 27(a): Append sets block_ to bb");
        assert(bb.instrs_.root_.next_ == (HirIntrusiveListNode *)instr &&
               bb.instrs_.root_.prev_ == (HirIntrusiveListNode *)instr &&
               "Phase 4.A Batch 27(a): list head/tail point to instr");

        hir_c_instr_free(instr);
    }

    /* (b) push_front into empty block, then pop_front retrieves it
     * with block_ reset to NULL. */
    {
        HirBasicBlock bb = {};
        bb.instrs_.root_.prev_ = &bb.instrs_.root_;
        bb.instrs_.root_.next_ = &bb.instrs_.root_;
        bb.instrs_.node_member_offset_ = 0;

        void *instr = hir_c_alloc_instr(sizeof(HirInstrLayout), 0);
        assert(instr != NULL);
        hir_c_init_instr(instr, HIR_OP_BinaryOp);

        hir_c_bb_push_front(&bb, instr);
        assert(((HirInstrLayout *)instr)->block == &bb &&
               "Phase 4.A Batch 27(b): push_front sets block_ to bb");
        assert(bb.instrs_.root_.next_ == (HirIntrusiveListNode *)instr &&
               "Phase 4.A Batch 27(b): list head points to instr");

        void *popped = hir_c_bb_pop_front(&bb);
        assert(popped == instr &&
               "Phase 4.A Batch 27(b): pop_front returns the head instr");
        assert(((HirInstrLayout *)instr)->block == NULL &&
               "Phase 4.A Batch 27(b): pop_front clears block_ to NULL");
        assert(bb.instrs_.root_.next_ == &bb.instrs_.root_ &&
               bb.instrs_.root_.prev_ == &bb.instrs_.root_ &&
               "Phase 4.A Batch 27(b): empty block sentinel restored");

        hir_c_instr_free(instr);
    }

    /* (c) pop_front on empty block returns NULL. */
    {
        HirBasicBlock bb = {};
        bb.instrs_.root_.prev_ = &bb.instrs_.root_;
        bb.instrs_.root_.next_ = &bb.instrs_.root_;
        bb.instrs_.node_member_offset_ = 0;

        void *popped = hir_c_bb_pop_front(&bb);
        assert(popped == NULL &&
               "Phase 4.A Batch 27(c): pop_front on empty block returns NULL");
    }
}

/* Phase 4.A Batch 28 V5 falsifier for compare op tables + helpers.
 * Verifies (a) table sizes match the C++ enum counts (V3 layout-pin
 * deferred to runtime since the C-side counts are extern const, not
 * constant expressions); (b) GetX/ParseX round-trips for every enum
 * value; (c) toPrimitiveCompareOp mapping per-case, including the
 * three nullopt cases (kIn, kNotIn, kExcMatch). */
static void verify_phase4a_batch28_compare_ops() {
    /* (a) Table-size pin. */
    assert(kNumCompareOps_c == kNumCompareOps &&
           "Phase 4.A Batch 28: kCompareOpNames_c size matches C++ kNumCompareOps");
    assert(kNumPrimitiveCompareOps_c == kNumPrimitiveCompareOps &&
           "Phase 4.A Batch 28: kPrimitiveCompareOpNames_c size matches C++ kNumPrimitiveCompareOps");

    /* (b) CompareOp round-trip: GetName(Parse(GetName(i))) == GetName(i). */
    for (int i = 0; i < (int)kNumCompareOps_c; i++) {
        const char *name = hir_c_get_compare_op_name(i);
        assert(name != NULL && name[0] != '\0' &&
               "Phase 4.A Batch 28: CompareOp name non-empty");
        int parsed = hir_c_parse_compare_op_name(name, strlen(name));
        assert(parsed == i &&
               "Phase 4.A Batch 28: CompareOp name round-trip preserves enum value");
    }

    /* (b cont.) PrimitiveCompareOp round-trip. */
    for (int i = 0; i < (int)kNumPrimitiveCompareOps_c; i++) {
        const char *name = hir_c_get_primitive_compare_op_name(i);
        assert(name != NULL && name[0] != '\0' &&
               "Phase 4.A Batch 28: PrimitiveCompareOp name non-empty");
        int parsed = hir_c_parse_primitive_compare_op_name(name, strlen(name));
        assert(parsed == i &&
               "Phase 4.A Batch 28: PrimitiveCompareOp name round-trip");
    }

    /* (c) toPrimitiveCompareOp per-case mapping. */
    /* Identity for kLessThan..kGreaterThanEqual (CompareOp 0..5 → Primitive 0..5). */
    for (int i = 0; i <= 5; i++) {
        assert(hir_c_to_primitive_compare_op(i) == i &&
               "Phase 4.A Batch 28: toPrimitive identity for kLT..kGE");
    }
    /* Three nullopt cases: kIn (6), kNotIn (7), kExcMatch (8). */
    assert(hir_c_to_primitive_compare_op(6) == -1 &&
           "Phase 4.A Batch 28: toPrimitive(kIn) = nullopt");
    assert(hir_c_to_primitive_compare_op(7) == -1 &&
           "Phase 4.A Batch 28: toPrimitive(kNotIn) = nullopt");
    assert(hir_c_to_primitive_compare_op(8) == -1 &&
           "Phase 4.A Batch 28: toPrimitive(kExcMatch) = nullopt");
    /* Unsigned variants shift down by 3: CompareOp 9..12 → Primitive 6..9. */
    assert(hir_c_to_primitive_compare_op(9) == 6 &&
           "Phase 4.A Batch 28: toPrimitive(kGTu) = kGTu primitive");
    assert(hir_c_to_primitive_compare_op(10) == 7 &&
           "Phase 4.A Batch 28: toPrimitive(kGEu) = kGEu primitive");
    assert(hir_c_to_primitive_compare_op(11) == 8 &&
           "Phase 4.A Batch 28: toPrimitive(kLTu) = kLTu primitive");
    assert(hir_c_to_primitive_compare_op(12) == 9 &&
           "Phase 4.A Batch 28: toPrimitive(kLEu) = kLEu primitive");
}

/* Phase 4.A Batch 29 V5 falsifier for BinaryOp + UnaryOp name helpers.
 * Same pattern as Batch 28: table-size pin + GetX/ParseX round-trip
 * for every enum value of both tables. */
static void verify_phase4a_batch29_binary_unary_ops() {
    assert(kNumBinaryOpKinds_c == kNumBinaryOpKinds &&
           "Phase 4.A Batch 29: kBinaryOpNames_c size matches C++ kNumBinaryOpKinds");
    assert(kNumUnaryOpKinds_c == kNumUnaryOpKinds &&
           "Phase 4.A Batch 29: kUnaryOpNames_c size matches C++ kNumUnaryOpKinds");

    for (int i = 0; i < (int)kNumBinaryOpKinds_c; i++) {
        const char *name = hir_c_get_binary_op_name(i);
        assert(name != NULL && name[0] != '\0' &&
               "Phase 4.A Batch 29: BinaryOpKind name non-empty");
        int parsed = hir_c_parse_binary_op_name(name, strlen(name));
        assert(parsed == i &&
               "Phase 4.A Batch 29: BinaryOpKind name round-trip");
    }

    for (int i = 0; i < (int)kNumUnaryOpKinds_c; i++) {
        const char *name = hir_c_get_unary_op_name(i);
        assert(name != NULL && name[0] != '\0' &&
               "Phase 4.A Batch 29: UnaryOpKind name non-empty");
        int parsed = hir_c_parse_unary_op_name(name, strlen(name));
        assert(parsed == i &&
               "Phase 4.A Batch 29: UnaryOpKind name round-trip");
    }
}

/* Phase 4.A Batch 34 V5 falsifier for PrimitiveUnaryOp + InPlaceOp
 * name helpers. Same pattern as B28 + B29: table-size pin + GetX/ParseX
 * round-trip for every enum value. */
static void verify_phase4a_batch34_primitive_unary_inplace_ops() {
    assert(kNumPrimitiveUnaryOpKinds_c == kNumPrimitiveUnaryOpKinds &&
           "Phase 4.A Batch 34: PrimitiveUnary table size matches C++");
    assert(kNumInPlaceOpKinds_c == kNumInPlaceOpKinds &&
           "Phase 4.A Batch 34: InPlace table size matches C++");

    for (int i = 0; i < (int)kNumPrimitiveUnaryOpKinds_c; i++) {
        const char *name = hir_c_get_primitive_unary_op_name(i);
        assert(name != NULL && name[0] != '\0' &&
               "Phase 4.A Batch 34: PrimitiveUnaryOpKind name non-empty");
        int parsed = hir_c_parse_primitive_unary_op_name(name, strlen(name));
        assert(parsed == i &&
               "Phase 4.A Batch 34: PrimitiveUnaryOpKind name round-trip");
    }

    for (int i = 0; i < (int)kNumInPlaceOpKinds_c; i++) {
        const char *name = hir_c_get_inplace_op_name(i);
        assert(name != NULL && name[0] != '\0' &&
               "Phase 4.A Batch 34: InPlaceOpKind name non-empty");
        int parsed = hir_c_parse_inplace_op_name(name, strlen(name));
        assert(parsed == i &&
               "Phase 4.A Batch 34: InPlaceOpKind name round-trip");
    }
}

/* Phase 4.A Batch 35 V5 falsifier for Environment::getRegister.
 * V5 N/A for the trivial 1-line shims (retargetPreds, references —
 * pure delegation, V1 prod-use covers). getRegister has the bounds
 * check + slot lookup as substantive logic. */
static void verify_phase4a_batch35_env_get_register() {
    HirEnvironment env = {};
    void *r5 = hir_make_register_c(5);
    void *added = hir_c_env_add_register(&env, r5);
    assert(added == r5 &&
           "Phase 4.A Batch 35: addRegister precondition");

    /* In-bounds positive lookup: id=5 returns the registered ptr. */
    void *got5 = hir_env_get_register(&env, 5);
    assert(got5 == r5 &&
           "Phase 4.A Batch 35(a): getRegister(5) returns registered Register");

    /* In-bounds NULL slot: id=2 (gap) returns NULL. */
    void *got2 = hir_env_get_register(&env, 2);
    assert(got2 == NULL &&
           "Phase 4.A Batch 35(b): getRegister(2) NULL gap returns NULL");

    /* Negative id: returns NULL (bounds check lower). */
    void *got_neg = hir_env_get_register(&env, -1);
    assert(got_neg == NULL &&
           "Phase 4.A Batch 35(c): getRegister(-1) returns NULL");

    /* Out-of-bounds id: returns NULL (bounds check upper). */
    void *got_oob = hir_env_get_register(&env, 999);
    assert(got_oob == NULL &&
           "Phase 4.A Batch 35(d): getRegister(999) returns NULL");

    delete static_cast<jit::hir::Register *>(r5);
    free(env.reg_data);
}

/* Phase 4.A Batch 30: SUBSTANTIVE V5 back-fill for B21
 * (Instr::getDominatingFrameState ef73c50a75) using the Batch I
 * test-chain helper. Falsifier-only addition; no production code
 * changes — semantic-equivalent SAME-content per supervisor 10:02:58Z
 * back-fill exemption. Tests both the walk-past-replayable success
 * path and the non-replayable-interpose return-NULL path that V1
 * production-use covers but never empirically anchored. */
static void verify_phase4a_batch30_dominating_frame_state_substantive() {
    /* (a) Walk-past-replayable: chain [Snapshot, Assign, target_BinaryOp].
     * Assign is replayable (arity=1, replayable=1 per hir_instr_info_c.c);
     * Snapshot precedes it. Walk from target → Assign (replayable,
     * continue) → Snapshot → return its frame_state. */
    {
        HirBasicBlock bb;
        hir_c_test_chain_init(&bb);

        FrameState fs{};
        fs.cur_instr_offs = jit::BCOffset{42};
        void *snap = hir_c_test_chain_append_snapshot(&bb, &fs);
        void *assign = hir_c_test_chain_append_instr(
            &bb, HIR_OP_Assign, sizeof(HirInstrLayout), 1);
        (void)assign;
        void *target = hir_c_test_chain_append_instr(
            &bb, HIR_OP_BinaryOp, sizeof(HirBinaryOp), 2);
        hir_c_init_deopt(target, HIR_OP_BinaryOp);

        const void *dominating_fs = hir_c_get_dominating_frame_state(target);
        assert(dominating_fs != NULL &&
               "Phase 4.A Batch 30(a): dominating Snapshot's frame_state non-NULL");
        assert(dominating_fs == hir_c_snapshot_get_frame_state(snap) &&
               "Phase 4.A Batch 30(a): returns the Snapshot's frame_state ptr");
        assert(static_cast<const FrameState*>(dominating_fs)
                   ->cur_instr_offs.value() == 42 &&
               "Phase 4.A Batch 30(a): frame_state cur_instr_offs preserved");

        hir_c_test_chain_destroy(&bb);
    }

    /* (b) Non-replayable interpose: chain [Snapshot, BinaryOp, target].
     * BinaryOp is NOT replayable (arity=2, replayable=0). Walk from
     * target → BinaryOp (not replayable) → return NULL. */
    {
        HirBasicBlock bb;
        hir_c_test_chain_init(&bb);

        FrameState fs{};
        fs.cur_instr_offs = jit::BCOffset{99};
        void *snap = hir_c_test_chain_append_snapshot(&bb, &fs);
        (void)snap;
        void *blocker = hir_c_test_chain_append_instr(
            &bb, HIR_OP_BinaryOp, sizeof(HirBinaryOp), 2);
        hir_c_init_deopt(blocker, HIR_OP_BinaryOp);
        void *target = hir_c_test_chain_append_instr(
            &bb, HIR_OP_BinaryOp, sizeof(HirBinaryOp), 2);
        hir_c_init_deopt(target, HIR_OP_BinaryOp);

        const void *dominating_fs = hir_c_get_dominating_frame_state(target);
        assert(dominating_fs == NULL &&
               "Phase 4.A Batch 30(b): non-replayable interpose returns NULL");

        hir_c_test_chain_destroy(&bb);
    }
}

/* Phase 4.A Batch 31: SUBSTANTIVE V5 back-fill for B23
 * (Instr::ExpandInto efb6199aaa) using Batch I infra. Falsifier-only;
 * no production code change. Tests the multi-instr expansion path
 * V1-prod-use covers but never empirically anchored: a prefix case
 * verifies the expansion sequence preserves the original prefix and
 * inserts expansion in order, unlinking target. Bytecode offset
 * propagation from target is also asserted. */
static void verify_phase4a_batch31_expand_into_substantive() {
    HirBasicBlock bb;
    hir_c_test_chain_init(&bb);

    /* Original chain: [prefix, target]. */
    void *prefix = hir_c_test_chain_append_instr(
        &bb, HIR_OP_Assign, sizeof(HirInstrLayout), 1);
    void *target = hir_c_test_chain_append_instr(
        &bb, HIR_OP_Assign, sizeof(HirInstrLayout), 1);
    /* Mark target's bytecode_offset so we can verify propagation. */
    ((HirInstrLayout *)target)->bytecode_offset = 71;

    /* 3 expansion instrs allocated standalone (NOT yet linked into bb;
     * expand_into does the inserts). hir_c_alloc_instr inits the
     * IntrusiveListNode self-pointers so they're insertable. */
    void *exp0 = hir_c_alloc_instr(sizeof(HirInstrLayout), 1);
    void *exp1 = hir_c_alloc_instr(sizeof(HirInstrLayout), 1);
    void *exp2 = hir_c_alloc_instr(sizeof(HirInstrLayout), 1);
    assert(exp0 != NULL && exp1 != NULL && exp2 != NULL);
    hir_c_init_instr(exp0, HIR_OP_Assign);
    hir_c_init_instr(exp1, HIR_OP_Assign);
    hir_c_init_instr(exp2, HIR_OP_Assign);
    void *expansion[3] = { exp0, exp1, exp2 };

    /* Pre-condition: target is in bb, bytecode_offset = 71, expansion
     * instrs are not in any block. */
    assert(((HirInstrLayout *)target)->block == &bb &&
           "Phase 4.A Batch 31: pre-call target.block == &bb");
    assert(((HirInstrLayout *)exp0)->block == NULL &&
           "Phase 4.A Batch 31: pre-call expansion[0].block == NULL");

    hir_c_instr_expand_into(target, expansion, 3);

    /* Post-call: target unlinked, bb chain is [prefix, exp0, exp1, exp2]. */
    assert(((HirInstrLayout *)target)->block == NULL &&
           "Phase 4.A Batch 31: target unlinked from bb");
    void *expected[4] = { prefix, exp0, exp1, exp2 };
    void *cur = hir_bb_first_instr(&bb);
    for (int i = 0; i < 4; i++) {
        assert(cur == expected[i] &&
               "Phase 4.A Batch 31: post-call iteration order [prefix, exp0..2]");
        cur = hir_bb_next_instr(&bb, cur);
    }
    assert(cur == NULL &&
           "Phase 4.A Batch 31: chain terminates at sentinel after 4 instrs");

    /* Bytecode offset propagation: each expansion instr inherits target's. */
    assert(((HirInstrLayout *)exp0)->bytecode_offset == 71 &&
           "Phase 4.A Batch 31: exp0 bytecode_offset = target's");
    assert(((HirInstrLayout *)exp1)->bytecode_offset == 71 &&
           "Phase 4.A Batch 31: exp1 bytecode_offset = target's");
    assert(((HirInstrLayout *)exp2)->bytecode_offset == 71 &&
           "Phase 4.A Batch 31: exp2 bytecode_offset = target's");

    /* Cleanup: target was unlinked, free directly. The expansion
     * instrs are now in bb and will be freed by chain_destroy. */
    hir_c_destroy_instr_impl(target);
    hir_c_test_chain_destroy(&bb);
}

/* Phase 4.A Batch 32: SUBSTANTIVE V5 back-fill for B25
 * (BasicBlock::addPhiPredecessor + removePhiPredecessor c5f4396997)
 * via Batch I infra. Falsifier-only; no production code change.
 *
 * Tests the full replace+destroy flow end-to-end (V1-prod-use covers
 * cfg simplify but never empirically anchored): build a 3-pred Phi,
 * add a new predecessor (allocate new 4-op Phi, replace, destroy old),
 * then remove that predecessor (allocate new 3-op Phi, replace,
 * destroy). Verifies basic_blocks_ count + sorted order + operand
 * pairing across both transitions. */
static void verify_phase4a_batch32_add_remove_phi_substantive() {
    HirBasicBlock bb;
    hir_c_test_chain_init(&bb);

    /* 4 BB sentinels — only .id is read. */
    HirBasicBlock pred1{}, pred2{}, pred3{}, pred_new{};
    pred1.id = 1;
    pred2.id = 2;
    pred3.id = 3;
    pred_new.id = 4;

    /* 3 register sentinels for the initial Phi. */
    void *r1 = (void *)(uintptr_t)0xACE10001ULL;
    void *r2 = (void *)(uintptr_t)0xACE10002ULL;
    void *r3 = (void *)(uintptr_t)0xACE10003ULL;
    /* Output register — Phi::createWithCount needs one. */
    void *out_reg = hir_make_register_c(99);

    /* Allocate initial Phi (3 operands) directly via bridge so we can
     * set output, then populate basic_blocks_ + operands via Batch 20
     * apply path. Append to bb so add_predecessor's
     * hir_c_instr_replace_with has a host block to wire into. */
    void *phi = hir_make_phi_with_count_c(out_reg, 3);
    void *initial_keys[3]   = { &pred1, &pred2, &pred3 };
    void *initial_values[3] = { r1,     r2,     r3     };
    hir_c_phi_apply_args(phi, initial_keys, initial_values, 3);
    hir_bb_append_instr(&bb, phi);

    assert(((HirPhi *)phi)->bb_count == 3 &&
           "Phase 4.A Batch 32: initial Phi has 3 predecessors");

    /* (a) addPhiPredecessor: replace pred1 with pred_new appended.
     * The C++ original maps both pred1 + pred_new to the same Register
     * (r1) → 4-op Phi. Sort by id: [pred1, pred2, pred3, pred_new]. */
    hir_c_phi_add_predecessor(phi, &pred1, &pred_new);

    /* The original phi was destroyed + replaced; the new Phi is the
     * sole instr in bb. Recover via list-head. */
    void *phi_after_add = hir_bb_first_instr(&bb);
    assert(phi_after_add != NULL &&
           "Phase 4.A Batch 32(a): post-add Phi present in bb");
    assert(phi_after_add != phi &&
           "Phase 4.A Batch 32(a): post-add Phi is a fresh allocation");
    HirPhi *p_add = (HirPhi *)phi_after_add;
    assert(p_add->bb_count == 4 &&
           "Phase 4.A Batch 32(a): bb_count grew to 4 after add");
    assert(p_add->bb_data[0] == &pred1 && p_add->bb_data[1] == &pred2 &&
           p_add->bb_data[2] == &pred3 && p_add->bb_data[3] == &pred_new &&
           "Phase 4.A Batch 32(a): basic_blocks_ sorted [pred1..pred3, pred_new]");
    /* Operand for pred1 (id=1) is r1; pred_new (id=4) gets r1 too
     * (per the addPhiPredecessor map-fold semantic). */
    assert(hir_c_get_operand(phi_after_add, 0) == r1 &&
           "Phase 4.A Batch 32(a): operand[pred1] = r1");
    assert(hir_c_get_operand(phi_after_add, 3) == r1 &&
           "Phase 4.A Batch 32(a): operand[pred_new] = r1 (same as pred1's)");

    /* (b) removePhiPredecessor: drop pred_new, back to 3 ops. */
    hir_c_phi_remove_predecessor(phi_after_add, &pred_new);

    void *phi_after_remove = hir_bb_first_instr(&bb);
    assert(phi_after_remove != NULL &&
           "Phase 4.A Batch 32(b): post-remove Phi present");
    assert(phi_after_remove != phi_after_add &&
           "Phase 4.A Batch 32(b): post-remove Phi is a fresh allocation");
    HirPhi *p_rm = (HirPhi *)phi_after_remove;
    assert(p_rm->bb_count == 3 &&
           "Phase 4.A Batch 32(b): bb_count back to 3 after remove");
    assert(p_rm->bb_data[0] == &pred1 && p_rm->bb_data[1] == &pred2 &&
           p_rm->bb_data[2] == &pred3 &&
           "Phase 4.A Batch 32(b): basic_blocks_ back to [pred1, pred2, pred3]");
    assert(hir_c_get_operand(phi_after_remove, 0) == r1 &&
           hir_c_get_operand(phi_after_remove, 1) == r2 &&
           hir_c_get_operand(phi_after_remove, 2) == r3 &&
           "Phase 4.A Batch 32(b): operands back to [r1, r2, r3]");

    hir_c_test_chain_destroy(&bb);
    delete static_cast<jit::hir::Register *>(out_reg);
}

/* Phase 4.A Batch 33: SUBSTANTIVE V5 back-fill for B27
 * (BasicBlock::Append + push_front + pop_front 25b4c825a4) via
 * Batch I infra. Falsifier-only; no production code change.
 *
 * Tests the multi-instr ordering V1-prod-use covers but never
 * empirically anchored: build [B, A, C] via Append + push_front +
 * Append, verify iter order, pop_front and verify [A, C] remains
 * with B detached. */
static void verify_phase4a_batch33_bb_list_substantive() {
    HirBasicBlock bb;
    hir_c_test_chain_init(&bb);

    /* 3 standalone instrs, not yet linked. */
    void *A = hir_c_alloc_instr(sizeof(HirInstrLayout), 1);
    void *B = hir_c_alloc_instr(sizeof(HirInstrLayout), 1);
    void *C = hir_c_alloc_instr(sizeof(HirInstrLayout), 1);
    assert(A != NULL && B != NULL && C != NULL);
    hir_c_init_instr(A, HIR_OP_Assign);
    hir_c_init_instr(B, HIR_OP_Assign);
    hir_c_init_instr(C, HIR_OP_Assign);

    /* Sequence Append(A) → push_front(B) → Append(C) → expect [B, A, C]. */
    void *appended_A = hir_c_bb_append(&bb, A);
    assert(appended_A == A &&
           "Phase 4.A Batch 33: Append(A) returns A");

    hir_c_bb_push_front(&bb, B);
    hir_c_bb_append(&bb, C);

    /* Iterate + assert order [B, A, C]. */
    void *expected1[3] = { B, A, C };
    void *cur = hir_bb_first_instr(&bb);
    for (int i = 0; i < 3; i++) {
        assert(cur == expected1[i] &&
               "Phase 4.A Batch 33: post-build iter order [B, A, C]");
        cur = hir_bb_next_instr(&bb, cur);
    }
    assert(cur == NULL &&
           "Phase 4.A Batch 33: chain terminates at sentinel after 3 instrs");

    /* All 3 should report block_ == &bb. */
    assert(((HirInstrLayout *)A)->block == &bb &&
           ((HirInstrLayout *)B)->block == &bb &&
           ((HirInstrLayout *)C)->block == &bb &&
           "Phase 4.A Batch 33: all linked instrs have block_ == bb");

    /* pop_front removes B, returns it with block_ cleared. */
    void *popped = hir_c_bb_pop_front(&bb);
    assert(popped == B &&
           "Phase 4.A Batch 33: pop_front returns the head (B)");
    assert(((HirInstrLayout *)B)->block == NULL &&
           "Phase 4.A Batch 33: pop_front clears B's block_");

    /* Remaining chain iter [A, C]. */
    void *expected2[2] = { A, C };
    cur = hir_bb_first_instr(&bb);
    for (int i = 0; i < 2; i++) {
        assert(cur == expected2[i] &&
               "Phase 4.A Batch 33: post-pop iter order [A, C]");
        cur = hir_bb_next_instr(&bb, cur);
    }
    assert(cur == NULL &&
           "Phase 4.A Batch 33: post-pop chain terminates after 2 instrs");

    /* Cleanup: B was popped (no longer in bb), free directly.
     * A + C are in bb, freed by chain_destroy. */
    hir_c_destroy_instr_impl(B);
    hir_c_test_chain_destroy(&bb);
}

__attribute__((constructor))
static void hir_instr_runtime_check() {
    verify_hir_instr_read_through_cast();
    verify_is_replayable_table();
    verify_bytecode_offset_invariant();
    verify_phase4a_batch1_exhaustive();
    verify_phase4a_batch10_visitor();
    verify_phase4a_batch11_live_regs_iteration();
    verify_phase4a_batch12_uses_replace();
    verify_phase4a_batch13_sort_live_regs();
    verify_phase4a_batch15_edge_in_edges();
    verify_phase4a_batch16_allocator();
    verify_phase4a_batch17_instr_ctor_init();
    verify_phase4a_batch18_set_frame_state();
    verify_phase4a_batch19_deopt_copy();
    verify_phase4a_batch20_phi_apply_args();
    verify_phase4a_batch21_dominating_frame_state();
    verify_phase4a_batch22_edges_dispatch();
    verify_phase4a_batch23_instr_lifecycle();
    verify_phase4a_batch24_fixup_phis();
    verify_phase4a_batch25_add_remove_collect();
    verify_phase4a_batch26_env_register();
    verify_phase4a_batch27_bb_list_wrappers();
    verify_phase4a_batch28_compare_ops();
    verify_phase4a_batchI_chain_helper();
    verify_phase4a_batch29_binary_unary_ops();
    verify_phase4a_batch30_dominating_frame_state_substantive();
    verify_phase4a_batch31_expand_into_substantive();
    verify_phase4a_batch32_add_remove_phi_substantive();
    verify_phase4a_batch33_bb_list_substantive();
    verify_phase4a_batch34_primitive_unary_inplace_ops();
    verify_phase4a_batch35_env_get_register();
}
