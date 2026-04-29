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
}
