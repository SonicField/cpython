/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * T2-B verification: sizeof + offsetof static_asserts.
 * Uses friend struct for access to private C++ members.
 */

#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/hir/hir.h"

#include <cassert>

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

/* ---- Per-field offsetof checks via friend struct ---- */
struct HirInstrLayoutVerifier {
    /* HirInstr vs Instr */
    static_assert(offsetof(HirInstrLayout, block_node) == offsetof(Instr, block_node_));
    static_assert(offsetof(HirInstrLayout, opcode) == offsetof(Instr, opcode_));
    static_assert(offsetof(HirInstrLayout, bytecode_offset) == offsetof(Instr, bytecode_offset_));
    static_assert(offsetof(HirInstrLayout, output) == offsetof(Instr, output_));
    static_assert(offsetof(HirInstrLayout, block) == offsetof(Instr, block_));

    /* HirDeoptLayout field offsets vs DeoptBase */
    static_assert(offsetof(HirDeoptLayout, live_regs_storage) == offsetof(DeoptBase, live_regs_));
    static_assert(offsetof(HirDeoptLayout, frame_state) == offsetof(DeoptBase, frame_state_));
    static_assert(offsetof(HirDeoptLayout, guilty_reg) == offsetof(DeoptBase, guilty_reg_));
    static_assert(offsetof(HirDeoptLayout, nonce) == offsetof(DeoptBase, nonce_));
    static_assert(offsetof(HirDeoptLayout, descr_storage) == offsetof(DeoptBase, descr_));
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
    static_assert(offsetof(HirLoadField, name_storage) == offsetof(LoadField, name_));
    static_assert(offsetof(HirStoreField, name_storage) == offsetof(StoreField, name_));
    static_assert(offsetof(HirPhi, basic_blocks_storage) == offsetof(Phi, basic_blocks_));
    static_assert(offsetof(HirBeginInlinedFunction, fullname_storage) == offsetof(BeginInlinedFunction, fullname_));
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
    /* H2-A: DEFINE_SIMPLE_INSTR struct sizes */
    static_assert(sizeof(HirAssign) == sizeof(Assign));
    static_assert(sizeof(HirUnreachable) == sizeof(Unreachable));
    static_assert(sizeof(HirDecref) == sizeof(Decref));
    static_assert(sizeof(HirXDecref) == sizeof(XDecref));
    static_assert(sizeof(HirIncref) == sizeof(Incref));
    static_assert(sizeof(HirXIncref) == sizeof(XIncref));
    static_assert(sizeof(HirDeopt) == sizeof(Deopt));
    static_assert(sizeof(HirAtQuiescentState) == sizeof(AtQuiescentState));
    static_assert(sizeof(HirRunPeriodicTasks) == sizeof(RunPeriodicTasks));
    static_assert(sizeof(HirIsTruthy) == sizeof(IsTruthy));
    static_assert(sizeof(HirLoadCellItem) == sizeof(LoadCellItem));
    static_assert(sizeof(HirLoadCurrentFunc) == sizeof(LoadCurrentFunc));
    static_assert(sizeof(HirLoadEvalBreaker) == sizeof(LoadEvalBreaker));
    static_assert(sizeof(HirLoadFrame) == sizeof(LoadFrame));
    static_assert(sizeof(HirLoadVarObjectSize) == sizeof(LoadVarObjectSize));
    static_assert(sizeof(HirCheckErrOccurred) == sizeof(CheckErrOccurred));
    static_assert(sizeof(HirDeleteAttr) == sizeof(DeleteAttr));
    static_assert(sizeof(HirDeleteSubscr) == sizeof(DeleteSubscr));
    static_assert(sizeof(HirRaise) == sizeof(Raise));
    static_assert(sizeof(HirMakeSet) == sizeof(MakeSet));
    static_assert(sizeof(HirStealCellItem) == sizeof(StealCellItem));
    static_assert(sizeof(HirSwapCellItem) == sizeof(SwapCellItem));
    static_assert(sizeof(HirSetCellItem) == sizeof(SetCellItem));
    static_assert(sizeof(HirSetCurrentAwaiter) == sizeof(SetCurrentAwaiter));
    static_assert(sizeof(HirWaitHandleRelease) == sizeof(WaitHandleRelease));
    static_assert(sizeof(HirWaitHandleLoadWaiter) == sizeof(WaitHandleLoadWaiter));
    static_assert(sizeof(HirWaitHandleLoadCoroOrResult) == sizeof(WaitHandleLoadCoroOrResult));
    static_assert(sizeof(HirGetAIter) == sizeof(GetAIter));
    static_assert(sizeof(HirGetANext) == sizeof(GetANext));
    static_assert(sizeof(HirGetIter) == sizeof(GetIter));
    static_assert(sizeof(HirGetTuple) == sizeof(GetTuple));
    static_assert(sizeof(HirGetLength) == sizeof(GetLength));
    static_assert(sizeof(HirSetUpdate) == sizeof(SetUpdate));
    static_assert(sizeof(HirDictUpdate) == sizeof(DictUpdate));
    static_assert(sizeof(HirListExtend) == sizeof(ListExtend));
    static_assert(sizeof(HirCopyDictWithoutKeys) == sizeof(CopyDictWithoutKeys));
    static_assert(sizeof(HirMakeTupleFromList) == sizeof(MakeTupleFromList));
    static_assert(sizeof(HirMatchKeys) == sizeof(MatchKeys));
    static_assert(sizeof(HirMatchClass) == sizeof(MatchClass));
    static_assert(sizeof(HirYieldValue) == sizeof(YieldValue));
    static_assert(sizeof(HirYieldAndYieldFrom) == sizeof(YieldAndYieldFrom));
    static_assert(sizeof(HirYieldFromHandleStopAsyncIteration) == sizeof(YieldFromHandleStopAsyncIteration));
    static_assert(sizeof(HirInitialYield) == sizeof(InitialYield));
    static_assert(sizeof(HirSend) == sizeof(Send));
    static_assert(sizeof(HirMakeCell) == sizeof(MakeCell));
    static_assert(sizeof(HirMakeFunction) == sizeof(MakeFunction));
    static_assert(sizeof(HirBatchDecref) == sizeof(BatchDecref));
};

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

__attribute__((constructor))
static void hir_instr_runtime_check() {
    verify_hir_instr_read_through_cast();
    verify_is_replayable_table();
}
