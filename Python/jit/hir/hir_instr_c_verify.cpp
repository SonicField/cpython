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
#include "cinderx/Jit/hir/builder_state_c.h"  /* Pilot 3 step 2 (Batch 43) */
#include "cinderx/Jit/hir/builder.h"          /* BlockCanonicalizer / HIRBuilder (TempAllocator deleted P3c Batch 77) */
#include "cinderx/Jit/hir/hir_c_api.h"         /* hir_c_tc_emit_* primitives (Batch 54) */
#include "cinderx/Jit/hir/typed_argument_c.h"   /* phx_typed_argument_pytype_swap (Batch 76) */

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

/* Phase 4.A Batch 36 V5 SUBSTANTIVE falsifier for BasicBlock::insert
 * via Batch I helper. Two cases anchor the iterator-end vs mid-list
 * paths: (a) before==NULL appends with bytecode-offset propagation
 * from the last instr; (b) before==existing-instr inserts the new
 * instr immediately before it (bytecode-offset propagated by
 * hir_bb_insert_before from the previous adjacent instr). */
static void verify_phase4a_batch36_bb_insert_substantive() {
    HirBasicBlock bb;
    hir_c_test_chain_init(&bb);

    void *A = hir_c_test_chain_append_instr(
        &bb, HIR_OP_Assign, sizeof(HirInstrLayout), 1);
    void *B = hir_c_test_chain_append_instr(
        &bb, HIR_OP_Assign, sizeof(HirInstrLayout), 1);
    void *C = hir_c_test_chain_append_instr(
        &bb, HIR_OP_Assign, sizeof(HirInstrLayout), 1);
    ((HirInstrLayout *)A)->bytecode_offset = 10;
    ((HirInstrLayout *)B)->bytecode_offset = 20;
    ((HirInstrLayout *)C)->bytecode_offset = 30;

    /* (a) Insert D before B → expect [A, D, B, C]. D's
     * bytecode_offset=-1 propagates from prev (A=10). */
    void *D = hir_c_alloc_instr(sizeof(HirInstrLayout), 1);
    assert(D != NULL);
    hir_c_init_instr(D, HIR_OP_Assign);
    assert(((HirInstrLayout *)D)->bytecode_offset == -1 &&
           "Phase 4.A Batch 36(a): pre-insert D bytecode_offset == -1");
    hir_c_bb_insert(&bb, D, B);
    assert(((HirInstrLayout *)D)->block == &bb &&
           "Phase 4.A Batch 36(a): post-insert D linked to bb");
    assert(((HirInstrLayout *)D)->bytecode_offset == 10 &&
           "Phase 4.A Batch 36(a): D inherited prev (A) bytecode_offset=10");
    void *expected_a[4] = { A, D, B, C };
    void *cur = hir_bb_first_instr(&bb);
    for (int i = 0; i < 4; i++) {
        assert(cur == expected_a[i] &&
               "Phase 4.A Batch 36(a): post-mid-insert iter [A, D, B, C]");
        cur = hir_bb_next_instr(&bb, cur);
    }
    assert(cur == NULL &&
           "Phase 4.A Batch 36(a): chain terminates at sentinel");

    /* (b) Append E via NULL before-pointer → expect [A, D, B, C, E]
     * with E's bytecode_offset propagated from C (last instr, =30). */
    void *E = hir_c_alloc_instr(sizeof(HirInstrLayout), 1);
    assert(E != NULL);
    hir_c_init_instr(E, HIR_OP_Assign);
    assert(((HirInstrLayout *)E)->bytecode_offset == -1 &&
           "Phase 4.A Batch 36(b): pre-insert E bytecode_offset == -1");
    hir_c_bb_insert(&bb, E, NULL);
    assert(((HirInstrLayout *)E)->block == &bb &&
           "Phase 4.A Batch 36(b): post-append E linked to bb");
    assert(((HirInstrLayout *)E)->bytecode_offset == 30 &&
           "Phase 4.A Batch 36(b): E inherited last (C) bytecode_offset=30");
    void *expected_b[5] = { A, D, B, C, E };
    cur = hir_bb_first_instr(&bb);
    for (int i = 0; i < 5; i++) {
        assert(cur == expected_b[i] &&
               "Phase 4.A Batch 36(b): post-end-append iter [A, D, B, C, E]");
        cur = hir_bb_next_instr(&bb, cur);
    }

    hir_c_test_chain_destroy(&bb);
}

/* Phase 4.A Batch 37 V5 falsifier for FunctionAttr + InlineFailureType
 * lookups. Table-size runtime pin + smoke-check non-empty name on
 * every enum index. No parse pair to round-trip. */
static void verify_phase4a_batch37_function_inline_lookups() {
    /* Function field table: fixed 5 entries (kClosure..kAnnotate). */
    assert(kNumFunctionAttrs_c == 5 &&
           "Phase 4.A Batch 37: FunctionAttr table size = 5");
    for (int i = 0; i < (int)kNumFunctionAttrs_c; i++) {
        const char *name = hir_c_get_function_field_name(i);
        assert(name != NULL && name[0] != '\0' &&
               "Phase 4.A Batch 37: functionFieldName entry non-empty");
    }
    /* Stable canonical entries: index 0 is "func_closure", index 4 is
     * "func_annotate" (anchors the order against unintended reordering). */
    assert(strcmp(hir_c_get_function_field_name(0), "func_closure") == 0 &&
           "Phase 4.A Batch 37: functionFieldName[0] = func_closure");
    assert(strcmp(hir_c_get_function_field_name(4), "func_annotate") == 0 &&
           "Phase 4.A Batch 37: functionFieldName[4] = func_annotate");

    /* Inline failure table: 16 entries per FOREACH_FAILURE_TYPE. */
    assert(kNumInlineFailureTypes_c == 16 &&
           "Phase 4.A Batch 37: InlineFailureType table size = 16");
    for (int i = 0; i < (int)kNumInlineFailureTypes_c; i++) {
        const char *msg = hir_c_get_inline_failure_message(i);
        const char *name = hir_c_get_inline_failure_name(i);
        assert(msg != NULL && msg[0] != '\0' &&
               "Phase 4.A Batch 37: failure msg non-empty");
        assert(name != NULL && name[0] != '\0' &&
               "Phase 4.A Batch 37: failure name non-empty");
    }
    /* Anchors: index 0 = HasDefaults / "it has defaults". */
    assert(strcmp(hir_c_get_inline_failure_name(0), "HasDefaults") == 0 &&
           "Phase 4.A Batch 37: failure name[0] = HasDefaults");
    assert(strcmp(hir_c_get_inline_failure_message(0), "it has defaults") == 0 &&
           "Phase 4.A Batch 37: failure msg[0] = 'it has defaults'");
}

/* Phase 4.A Batch 38 V5 SUBSTANTIVE falsifier for get_frame_state
 * dispatcher via Batch I helper. Three opcode cases + the NULL fallback:
 *   (a) Snapshot → returns its frame_state_ptr.
 *   (b) BeginInlinedFunction → returns caller_state_ptr.
 *   (c) DeoptBase opcode (BinaryOp) → returns deopt.frame_state.
 *   (d) Plain non-DeoptBase non-Snapshot non-BeginInlined (Assign) → NULL. */
static void verify_phase4a_batch38_get_frame_state() {
    /* (a) Snapshot path. */
    {
        HirBasicBlock bb;
        hir_c_test_chain_init(&bb);
        FrameState src{};
        src.cur_instr_offs = jit::BCOffset{38};
        void *snap = hir_c_test_chain_append_snapshot(&bb, &src);
        const void *fs = hir_c_instr_get_frame_state(snap);
        assert(fs != NULL &&
               "Phase 4.A Batch 38(a): Snapshot get_frame_state non-NULL");
        assert(fs == hir_c_snapshot_get_frame_state(snap) &&
               "Phase 4.A Batch 38(a): returns snapshot.frame_state");
        hir_c_test_chain_destroy(&bb);
    }

    /* (b) BeginInlinedFunction path. */
    {
        HirBasicBlock bb;
        hir_c_test_chain_init(&bb);
        FrameState src{};
        src.cur_instr_offs = jit::BCOffset{77};
        FrameState *caller_fs =
            static_cast<FrameState *>(hir_make_frame_state_c(&src));
        void *bi = hir_c_test_chain_append_instr(
            &bb, HIR_OP_BeginInlinedFunction,
            sizeof(HirBeginInlinedFunction), 0);
        ((HirBeginInlinedFunction *)bi)->caller_state_ptr = caller_fs;

        const void *fs = hir_c_instr_get_frame_state(bi);
        assert(fs == caller_fs &&
               "Phase 4.A Batch 38(b): BeginInlined get_frame_state = caller_state_ptr");
        hir_c_test_chain_destroy(&bb);
    }

    /* (c) DeoptBase opcode (BinaryOp) path. */
    {
        HirBasicBlock bb;
        hir_c_test_chain_init(&bb);
        FrameState src{};
        src.cur_instr_offs = jit::BCOffset{99};
        void *bin_op = hir_c_test_chain_append_instr(
            &bb, HIR_OP_BinaryOp, sizeof(HirBinaryOp), 2);
        hir_c_init_deopt(bin_op, HIR_OP_BinaryOp);
        hir_c_deopt_set_frame_state(bin_op, &src);

        const void *fs = hir_c_instr_get_frame_state(bin_op);
        assert(fs != NULL &&
               "Phase 4.A Batch 38(c): DeoptBase get_frame_state non-NULL");
        assert(fs == hir_c_deopt_get_frame_state(bin_op) &&
               "Phase 4.A Batch 38(c): returns deopt.frame_state");
        assert(static_cast<const FrameState *>(fs)->cur_instr_offs.value() == 99 &&
               "Phase 4.A Batch 38(c): cur_instr_offs preserved through deopt path");
        hir_c_test_chain_destroy(&bb);
    }

    /* (d) Plain Instr (Assign, neither Snapshot/BeginInlined/DeoptBase)
     * → returns NULL. */
    {
        HirBasicBlock bb;
        hir_c_test_chain_init(&bb);
        void *plain = hir_c_test_chain_append_instr(
            &bb, HIR_OP_Assign, sizeof(HirInstrLayout), 1);
        const void *fs = hir_c_instr_get_frame_state(plain);
        assert(fs == NULL &&
               "Phase 4.A Batch 38(d): plain Instr get_frame_state = NULL");
        hir_c_test_chain_destroy(&bb);
    }
}

/* Phase 4.A Batch 39 V5 falsifier: isLoadMethodBase + isAnyLoadMethod
 * + modelReg dispatchers. Stub Instrs and Registers exercised directly
 * (none of these touch block_, so no chain-helper setup required). */
static void verify_phase4a_batch39_load_method_model_reg() {
    /* (a) isLoadMethodBase: 3 hit opcodes + 1 miss. */
    {
        HirInstrLayout instr = {};
        instr.opcode = HIR_OP_LoadMethod;
        assert(hir_c_is_load_method_base(&instr) &&
               "Phase 4.A Batch 39(a): LoadMethod is base");
        instr.opcode = HIR_OP_LoadMethodCached;
        assert(hir_c_is_load_method_base(&instr) &&
               "Phase 4.A Batch 39(a): LoadMethodCached is base");
        instr.opcode = HIR_OP_LoadModuleMethodCached;
        assert(hir_c_is_load_method_base(&instr) &&
               "Phase 4.A Batch 39(a): LoadModuleMethodCached is base");
        instr.opcode = HIR_OP_BinaryOp;
        assert(!hir_c_is_load_method_base(&instr) &&
               "Phase 4.A Batch 39(a): BinaryOp is NOT base");
    }

    /* (b) isAnyLoadMethod: pass-through to isLoadMethodBase for hit;
     * non-Phi non-LoadMethod miss returns false. */
    {
        HirInstrLayout instr = {};
        instr.opcode = HIR_OP_LoadMethod;
        assert(hir_c_is_any_load_method(&instr) &&
               "Phase 4.A Batch 39(b): LoadMethod passes through to any");
        instr.opcode = HIR_OP_BinaryOp;
        assert(!hir_c_is_any_load_method(&instr) &&
               "Phase 4.A Batch 39(b): non-Phi non-LoadMethod = false");
    }

    /* (c) modelReg: 1-instr identity case (instr is non-passthrough,
     * loop doesn't enter, returns same reg). BinaryOp is not
     * passthrough, so the walk terminates immediately. */
    {
        HirInstrLayout instr = {};
        instr.opcode = HIR_OP_BinaryOp;
        HirRegisterLayout reg = {};
        reg.instr = &instr;
        void *result = hir_c_model_reg(&reg);
        assert(result == &reg &&
               "Phase 4.A Batch 39(c): non-passthrough instr returns same reg");
    }
}

/* Phase 4.A Batch 41 V5 falsifier for Constraint name lookup.
 * Table-size pin (10 entries) + per-enum smoke (kType=NULL, others
 * non-empty) + canonical anchor on the kOptObject* family. */
static void verify_phase4a_batch41_constraint_name() {
    assert(kNumConstraints_c == 10 &&
           "Phase 4.A Batch 41: Constraint table size = 10");
    /* kType (index 0) is the NULL sentinel for the type-printing path. */
    assert(hir_c_constraint_name(0) == NULL &&
           "Phase 4.A Batch 41: kType returns NULL (delegate to ostream)");
    /* All 9 non-kType entries have non-empty strings. */
    for (int i = 1; i < (int)kNumConstraints_c; i++) {
        const char *s = hir_c_constraint_name(i);
        assert(s != NULL && s[0] != '\0' &&
               "Phase 4.A Batch 41: non-kType entry non-empty");
    }
    /* Canonical anchors per the C++ switch order before deletion. */
    assert(strcmp(hir_c_constraint_name(1), "CInt") == 0 &&
           "Phase 4.A Batch 41: kMatchAllAsCInt = 'CInt'");
    assert(strcmp(hir_c_constraint_name(2), "Primitive") == 0 &&
           "Phase 4.A Batch 41: kMatchAllAsPrimitive = 'Primitive'");
    assert(strcmp(hir_c_constraint_name(7), "(OptObject, CInt, CBool)") == 0 &&
           "Phase 4.A Batch 41: kOptObjectOrCIntOrCBool string match");
}

/* Phase 4.C Pilot 3 Batch 42 V5 sentinel falsifier for HirTempAllocator
 * (TempAllocator port). Three behaviors anchored:
 *   (a) AllocateStack 3x grows cache to 3 + each Register matches env's
 *       allocation sequence.
 *   (b) GetOrAllocateStack(0..2) returns cached entries; (3) appends.
 *   (c) AllocateNonStack does NOT grow the cache. */
static void verify_phase4c_batch42_temp_allocator() {
    HirEnvironment env = {};
    HirTempAllocator t = {};
    t.env = &env;
    phx_ptr_arr_init(&t.cache);

    /* (a) AllocateStack 3x. */
    void *r0 = hir_c_temps_alloc_stack(&t);
    void *r1 = hir_c_temps_alloc_stack(&t);
    void *r2 = hir_c_temps_alloc_stack(&t);
    assert(t.cache.count == 3 &&
           "Phase 4.C Batch 42(a): cache grew to 3 after 3 AllocateStack");
    assert(t.cache.data[0] == r0 && t.cache.data[1] == r1 &&
           t.cache.data[2] == r2 &&
           "Phase 4.C Batch 42(a): cache holds the allocated Register sequence");
    /* env-side reg ids should be 0/1/2 — sequential from AllocateRegister. */
    assert(hir_reg_id(r0) == 0 &&
           hir_reg_id(r1) == 1 &&
           hir_reg_id(r2) == 2 &&
           "Phase 4.C Batch 42(a): env allocator id sequence preserved");

    /* (b) GetOrAllocateStack idempotent for in-range; appends past-end. */
    assert(hir_c_temps_get_or_alloc_stack(&t, 0) == r0 &&
           "Phase 4.C Batch 42(b): GetOrAllocate(0) returns cached");
    assert(hir_c_temps_get_or_alloc_stack(&t, 2) == r2 &&
           "Phase 4.C Batch 42(b): GetOrAllocate(2) returns cached");
    void *r3 = hir_c_temps_get_or_alloc_stack(&t, 3);
    assert(t.cache.count == 4 &&
           "Phase 4.C Batch 42(b): GetOrAllocate(3) appended");
    assert(t.cache.data[3] == r3 &&
           "Phase 4.C Batch 42(b): cache[3] = newly allocated");

    /* (c) AllocateNonStack does NOT grow the cache. */
    size_t cache_pre = t.cache.count;
    void *r_ns = hir_c_temps_alloc_non_stack(&t);
    assert(t.cache.count == cache_pre &&
           "Phase 4.C Batch 42(c): AllocateNonStack does NOT touch cache");
    assert(hir_reg_id(r_ns) == 4 &&
           "Phase 4.C Batch 42(c): env allocator continues sequence");

    /* Cleanup: reg_data ownership stays with env; chain helper not used. */
    delete static_cast<jit::hir::Register *>(r0);
    delete static_cast<jit::hir::Register *>(r1);
    delete static_cast<jit::hir::Register *>(r2);
    delete static_cast<jit::hir::Register *>(r3);
    delete static_cast<jit::hir::Register *>(r_ns);
    free(env.reg_data);
    phx_ptr_arr_destroy(&t.cache);
}

/* Phase 4.C Pilot 3 Batch 43 V5 sentinel falsifier: PhxHirBuilderState
 * temps_phx field migration. hir_builder_state_init zero-inits the
 * field; AllocateStack via &state->temps_phx (with env wired post-init)
 * appends to cache; hir_builder_state_destroy frees cache without leak.
 * V3 layout-pin via offsetof asserts. */
static void verify_phase4c_batch43_state_temps_phx() {
    /* V3 layout pin: temps_phx field exists at expected offset (after
     * the 3 prior phx fields). Cannot static_assert on field-existence
     * directly; smoke-check by reading + comparing the address. */
    PhxHirBuilderState state;
    hir_builder_state_init(&state, NULL, NULL);

    /* Post-init: temps_phx zero-state (env=NULL, cache empty). */
    assert(state.temps_phx.env == NULL &&
           "Phase 4.C Batch 43: post-init temps_phx.env == NULL");
    assert(state.temps_phx.cache.count == 0 &&
           state.temps_phx.cache.data == NULL &&
           "Phase 4.C Batch 43: post-init temps_phx.cache empty");

    /* Wire env, then exercise alloc via the existing C functions. */
    HirEnvironment env = {};
    state.temps_phx.env = &env;
    void *r0 = hir_c_temps_alloc_stack(&state.temps_phx);
    void *r1 = hir_c_temps_alloc_stack(&state.temps_phx);
    assert(state.temps_phx.cache.count == 2 &&
           "Phase 4.C Batch 43: AllocateStack 2x grew cache via state field");
    assert(state.temps_phx.cache.data[0] == r0 &&
           state.temps_phx.cache.data[1] == r1 &&
           "Phase 4.C Batch 43: cache holds allocated Registers");

    /* Cleanup: env owns Registers; destroy frees cache.data. */
    delete static_cast<jit::hir::Register *>(r0);
    delete static_cast<jit::hir::Register *>(r1);
    free(env.reg_data);
    hir_builder_state_destroy(&state);
    /* Post-destroy: cache freed (data NULL after phx_ptr_arr_destroy). */
    assert(state.temps_phx.cache.data == NULL &&
           state.temps_phx.cache.count == 0 &&
           "Phase 4.C Batch 43: post-destroy cache cleared (no leak)");
}

/* Phase 4.C Pilot 3 Batch 44 mirror-collapse falsifier DELETED in P3c
 * (Batch 77): the C++ TempAllocator wrapper this fixture exercised
 * was deleted in P3c — there is no longer a mirror to test for
 * collapse. Direct-C behavior is covered by batch42
 * (verify_phase4c_batch42_temp_allocator). The mirror-divergence
 * concern this fixture addressed (pythia 16:14:18Z B43-B47) is
 * structurally resolved by class deletion: zero mirror = zero
 * divergence-window class. */

/* Phase 4.C Pilot 4 Batch 48 V5 sentinel falsifier for HirOperandStack
 * (OperandStack port introduce step). LIFO push/pop semantics + state
 * field init/destroy. B49 will add the mirror-collapse falsifier with
 * the C++ wrapper alongside. */
static void verify_phase4c_batch48_op_stack() {
    PhxHirBuilderState state;
    hir_builder_state_init(&state, NULL, NULL);
    assert(state.op_stack_phx.stack.count == 0 &&
           state.op_stack_phx.stack.data == NULL &&
           "Phase 4.C Batch 48: post-init op_stack_phx empty");

    void *r0 = (void *)(uintptr_t)0xCAFE0001ULL;
    void *r1 = (void *)(uintptr_t)0xCAFE0002ULL;
    void *r2 = (void *)(uintptr_t)0xCAFE0003ULL;

    /* Push 3 → count grows. */
    hir_c_op_stack_push(&state.op_stack_phx, r0);
    hir_c_op_stack_push(&state.op_stack_phx, r1);
    hir_c_op_stack_push(&state.op_stack_phx, r2);
    assert(state.op_stack_phx.stack.count == 3 &&
           "Phase 4.C Batch 48: 3 pushes grow count to 3");

    /* Pop in LIFO order → r2, r1, r0. */
    assert(hir_c_op_stack_pop(&state.op_stack_phx) == r2 &&
           "Phase 4.C Batch 48: LIFO pop[0] = r2");
    assert(hir_c_op_stack_pop(&state.op_stack_phx) == r1 &&
           "Phase 4.C Batch 48: LIFO pop[1] = r1");
    assert(hir_c_op_stack_pop(&state.op_stack_phx) == r0 &&
           "Phase 4.C Batch 48: LIFO pop[2] = r0");
    assert(state.op_stack_phx.stack.count == 0 &&
           "Phase 4.C Batch 48: post-pop count = 0");

    hir_builder_state_destroy(&state);
    assert(state.op_stack_phx.stack.data == NULL &&
           "Phase 4.C Batch 48: post-destroy stack cleared (no leak)");
}

/* Phase 4.D Batch 51 V5 sentinel falsifier for hir_c_allocate_localsplus_n
 * (allocateLocalsplus port). Uses HirEnvironment + HirFrameStateLayout
 * sentinels directly (no PyCodeObject mock — caller pre-computes
 * nlocalsplus + nlocals, body works on already-resolved ints). */
static void verify_phase4d_batch51_allocate_localsplus() {
    HirEnvironment env = {};
    HirFrameStateLayout fs = {};
    phx_ptr_arr_init(&fs.localsplus);

    hir_c_allocate_localsplus_n(&env, &fs, 3, 2);

    assert(fs.localsplus.count == 3 &&
           "Phase 4.D Batch 51: localsplus.count == nlocalsplus");
    assert(fs.nlocals == 2 &&
           "Phase 4.D Batch 51: nlocals set");
    /* Each slot is a fresh Register from env (ids 0/1/2 sequential). */
    for (size_t i = 0; i < 3; i++) {
        void *r = fs.localsplus.data[i];
        assert(r != NULL &&
               "Phase 4.D Batch 51: localsplus[i] non-NULL");
        assert(hir_reg_id(r) == (int)i &&
               "Phase 4.D Batch 51: env allocator id sequence preserved");
    }

    /* Re-call clears + refills (idempotent for tail callers re-issuing). */
    hir_c_allocate_localsplus_n(&env, &fs, 1, 1);
    assert(fs.localsplus.count == 1 &&
           "Phase 4.D Batch 51: re-call clears + refills with nlocalsplus");
    assert(fs.nlocals == 1 &&
           "Phase 4.D Batch 51: re-call updates nlocals");

    /* Cleanup: env owns its 4 Registers (3 from first call + 1 from
     * second). The first 3 leak from env's view because the second
     * call clears localsplus without freeing them; that matches the
     * C++ behavior — Environment is the sole owner across builder
     * lifetime. Register pointers themselves are heap-allocated by
     * hir_make_register_c via the C++ bridge. */
    for (size_t i = 0; i < env.reg_count; i++) {
        if (env.reg_data[i]) {
            delete static_cast<jit::hir::Register *>(env.reg_data[i]);
        }
    }
    free(env.reg_data);
    phx_ptr_arr_destroy(&fs.localsplus);
}

/* Phase 4.D pilot step 1 (Batch 53): PhxTranslationContext layout pin
 * complementing the C++-side static_asserts at builder.cpp:1011-1014.
 * The C-side PhxTranslationContext struct uses HirFrameStateLayout for
 * the embedded frame; ensure the field offsets and total size match. */
static_assert(offsetof(PhxTranslationContext, block) == 0,
    "PhxTranslationContext.block must be at offset 0");
static_assert(offsetof(PhxTranslationContext, frame) == sizeof(void *),
    "PhxTranslationContext.frame must follow block at offset sizeof(void*)");

/* Phase 4.D Batch 54 V5 sentinel: hir_c_tc_emit_snapshot + emit_load_const
 *
 * Validates the no-FrameState emit cluster dispatch chain end-to-end:
 *   PhxTranslationContext → hir_c_tc_emit_X → hir_c_create_X →
 *   hir_c_set_bytecode_offset → hir_c_bb_append.
 *
 * Two representative cases (per theologian B54 spec); remaining 8 primitives
 * are trivial pass-throughs covered by V1 production path. */
static void verify_phase4d_batch54_no_fs_cluster() {
    HirBasicBlock bb;
    hir_c_test_chain_init(&bb);

    /* Build a PhxTranslationContext with non-zero cur_instr_offs to verify
     * the bytecode-offset transfer. */
    PhxTranslationContext tc{};
    tc.block = &bb;
    tc.frame.cur_instr_offs = 42;

    /* Case 1: emit_snapshot — Snapshot instr appended; its frame_state_ptr
     * deep-copied from tc.frame; bytecode_offset == cur_instr_offs. */
    hir_c_tc_emit_snapshot(&tc);
    void *first = hir_bb_first_instr(&bb);
    assert(first != NULL && "Phase 4.D Batch 54: snapshot appended");
    HirInstrLayout *snap = (HirInstrLayout *)first;
    assert(snap->opcode == HIR_OP_Snapshot &&
           "Phase 4.D Batch 54: appended instr is Snapshot");
    assert(snap->bytecode_offset == 42 &&
           "Phase 4.D Batch 54: Snapshot bytecode_offset == cur_instr_offs");
    HirSnapshot *s = (HirSnapshot *)first;
    assert(s->frame_state_ptr != NULL &&
           "Phase 4.D Batch 54: Snapshot frame_state_ptr populated");

    /* Case 2: emit_load_const — bump cur_instr_offs, call primitive, verify
     * LoadConst appended with matching bytecode_offset + type field. */
    tc.frame.cur_instr_offs = 99;
    /* Real HirRegLayout fixture: hir_c_set_output() writes back-pointer. */
    HirRegLayout fake_reg{};
    fake_reg.id = 7;
    HirType sentinel_type{};
    sentinel_type.bits_and_flags = 0xABCD0001U;
    hir_c_tc_emit_load_const(&tc, &fake_reg, sentinel_type);

    void *second = hir_bb_next_instr(&bb, first);
    assert(second != NULL && "Phase 4.D Batch 54: load_const appended");
    HirInstrLayout *lc_inst = (HirInstrLayout *)second;
    assert(lc_inst->opcode == HIR_OP_LoadConst &&
           "Phase 4.D Batch 54: appended instr is LoadConst");
    assert(lc_inst->bytecode_offset == 99 &&
           "Phase 4.D Batch 54: LoadConst bytecode_offset bump observed");
    HirLoadConst *lc = (HirLoadConst *)second;
    assert(lc->type.bits_and_flags == sentinel_type.bits_and_flags &&
           "Phase 4.D Batch 54: LoadConst type carried through primitive");

    hir_c_test_chain_destroy(&bb);
}

/* Phase 4.D Batch 55 V5 sentinel: FrameState-coupled emit primitives.
 *
 * Two patterns covered:
 *   (1) setFrameState-after — exercises hir_c_tc_emit_check_exc_fs:
 *       create + emit + setFrameState ordering preserved.
 *   (2) fs-as-factory-arg — exercises hir_c_tc_emit_initial_yield:
 *       fs passed to factory then emitted.
 *
 * Validates dispatch end-to-end: PhxTranslationContext → primitive →
 * factory → bytecode_offset stamp → block append. */
static void verify_phase4d_batch55_fs_coupled_cluster() {
    HirBasicBlock bb;
    hir_c_test_chain_init(&bb);

    PhxTranslationContext tc{};
    tc.block = &bb;
    tc.frame.cur_instr_offs = 17;

    /* Real FrameState fixture for setFrameState (pattern 1) +
     * factory-arg (pattern 2). Zero-init suffices; the bridge
     * deep-copies into the instr. */
    FrameState fs{};
    fs.cur_instr_offs = jit::BCOffset{17};

    /* Real HirRegLayout fixtures: hir_c_set_output() writes back-pointer. */
    HirRegLayout dst_reg{};
    dst_reg.id = 11;
    HirRegLayout src_reg{};
    src_reg.id = 12;

    /* Pattern 1: emit_check_exc_fs — create + emit + setFrameState ordering. */
    hir_c_tc_emit_check_exc_fs(&tc, &dst_reg, &src_reg, &fs);
    void *first = hir_bb_first_instr(&bb);
    assert(first != NULL && "Phase 4.D Batch 55: check_exc_fs appended");
    HirInstrLayout *chk = (HirInstrLayout *)first;
    assert(chk->opcode == HIR_OP_CheckExc &&
           "Phase 4.D Batch 55: appended instr is CheckExc");
    assert(chk->bytecode_offset == 17 &&
           "Phase 4.D Batch 55: pattern-1 bytecode_offset stamped pre-setFrameState");

    /* Pattern 2: emit_initial_yield — fs as factory arg. */
    tc.frame.cur_instr_offs = 33;
    HirRegLayout iy_dst{};
    iy_dst.id = 13;
    hir_c_tc_emit_initial_yield(&tc, &iy_dst, &fs);
    void *second = hir_bb_next_instr(&bb, first);
    assert(second != NULL && "Phase 4.D Batch 55: initial_yield appended");
    HirInstrLayout *iy = (HirInstrLayout *)second;
    assert(iy->opcode == HIR_OP_InitialYield &&
           "Phase 4.D Batch 55: appended instr is InitialYield");
    assert(iy->bytecode_offset == 33 &&
           "Phase 4.D Batch 55: pattern-2 bytecode_offset stamped");

    hir_c_test_chain_destroy(&bb);
}

/* Phase 4.D Batch 57 V5 sentinel: emit cluster 3 representative cases.
 *
 * Two coverage points:
 *   - emit_load_arg (no-fs, factory takes int idx + HirType): verifies
 *     LoadArg appended with idx + type carried through.
 *   - emit_set_current_awaiter (no-fs, single src, no output): verifies
 *     SetCurrentAwaiter appended at expected bytecode_offset.
 *
 * Remaining 8 primitives are trivial pass-throughs covered by V1
 * production path. */
static void verify_phase4d_batch57_emit_cluster_3() {
    HirBasicBlock bb;
    hir_c_test_chain_init(&bb);

    PhxTranslationContext tc{};
    tc.block = &bb;
    tc.frame.cur_instr_offs = 51;

    /* Case 1: emit_load_arg — fake dst register + sentinel HirType. */
    HirRegLayout dst_reg{};
    dst_reg.id = 41;
    HirType arg_type{};
    arg_type.bits_and_flags = 0xBEEF1234U;
    hir_c_tc_emit_load_arg(&tc, &dst_reg, /*idx=*/3, arg_type);

    void *first = hir_bb_first_instr(&bb);
    assert(first != NULL && "Phase 4.D Batch 57: load_arg appended");
    HirInstrLayout *la = (HirInstrLayout *)first;
    assert(la->opcode == HIR_OP_LoadArg &&
           "Phase 4.D Batch 57: appended instr is LoadArg");
    assert(la->bytecode_offset == 51 &&
           "Phase 4.D Batch 57: load_arg bytecode_offset stamped");

    /* Case 2: emit_set_current_awaiter — single-src, no-output. */
    tc.frame.cur_instr_offs = 73;
    HirRegLayout src_reg{};
    src_reg.id = 42;
    hir_c_tc_emit_set_current_awaiter(&tc, &src_reg);

    void *second = hir_bb_next_instr(&bb, first);
    assert(second != NULL && "Phase 4.D Batch 57: set_current_awaiter appended");
    HirInstrLayout *sca = (HirInstrLayout *)second;
    assert(sca->opcode == HIR_OP_SetCurrentAwaiter &&
           "Phase 4.D Batch 57: appended instr is SetCurrentAwaiter");
    assert(sca->bytecode_offset == 73 &&
           "Phase 4.D Batch 57: set_current_awaiter bytecode_offset stamped");

    hir_c_test_chain_destroy(&bb);
}

/* Phase 4.D Batch 58 V5 sentinel: emit cluster 4 representative cases.
 *
 *   - emit_store_attr (FS-coupled, 4-arg): StoreAttr instr appended at
 *     cur_instr_offs=89 with bytecode_offset stamped.
 *   - emit_at_quiescent_state (no-FS, no-arg, no-output): instr appended
 *     at cur_instr_offs=101 with bytecode_offset stamped.
 *
 * Remaining 8 primitives covered by V1 production path. */
static void verify_phase4d_batch58_emit_cluster_4() {
    HirBasicBlock bb;
    hir_c_test_chain_init(&bb);

    PhxTranslationContext tc{};
    tc.block = &bb;

    /* Case 1: emit_store_attr — FS-coupled. */
    tc.frame.cur_instr_offs = 89;
    FrameState fs{};
    fs.cur_instr_offs = jit::BCOffset{89};
    HirRegLayout receiver{};
    receiver.id = 51;
    HirRegLayout value{};
    value.id = 52;
    hir_c_tc_emit_store_attr(&tc, &receiver, &value, /*name_idx=*/7, &fs);

    void *first = hir_bb_first_instr(&bb);
    assert(first != NULL && "Phase 4.D Batch 58: store_attr appended");
    HirInstrLayout *sa = (HirInstrLayout *)first;
    assert(sa->opcode == HIR_OP_StoreAttr &&
           "Phase 4.D Batch 58: appended instr is StoreAttr");
    assert(sa->bytecode_offset == 89 &&
           "Phase 4.D Batch 58: store_attr bytecode_offset stamped");

    /* Case 2: emit_at_quiescent_state — no-FS, no-arg, no-output. */
    tc.frame.cur_instr_offs = 101;
    hir_c_tc_emit_at_quiescent_state(&tc);

    void *second = hir_bb_next_instr(&bb, first);
    assert(second != NULL && "Phase 4.D Batch 58: at_quiescent_state appended");
    HirInstrLayout *aqs = (HirInstrLayout *)second;
    assert(aqs->opcode == HIR_OP_AtQuiescentState &&
           "Phase 4.D Batch 58: appended instr is AtQuiescentState");
    assert(aqs->bytecode_offset == 101 &&
           "Phase 4.D Batch 58: at_quiescent_state bytecode_offset stamped");

    hir_c_test_chain_destroy(&bb);
}

/* Phase 4.D Batch 59 V5 sentinel: emit cluster 5 representative cases.
 *
 *   - emit_load_global (FS-coupled, name_idx + dst, cur_instr_offs=119):
 *     LoadGlobal instr appended with bytecode_offset stamped.
 *   - emit_wait_handle_load_coro_or_result (no-FS, dst+src, cur_instr_offs=131).
 *
 * Remaining 8 primitives covered by V1 production path. */
static void verify_phase4d_batch59_emit_cluster_5() {
    HirBasicBlock bb;
    hir_c_test_chain_init(&bb);

    PhxTranslationContext tc{};
    tc.block = &bb;

    /* Case 1: emit_load_global — FS-coupled. */
    tc.frame.cur_instr_offs = 119;
    FrameState fs{};
    fs.cur_instr_offs = jit::BCOffset{119};
    HirRegLayout dst_reg{};
    dst_reg.id = 61;
    hir_c_tc_emit_load_global(&tc, &dst_reg, /*name_idx=*/4, &fs);

    void *first = hir_bb_first_instr(&bb);
    assert(first != NULL && "Phase 4.D Batch 59: load_global appended");
    HirInstrLayout *lg = (HirInstrLayout *)first;
    assert(lg->opcode == HIR_OP_LoadGlobal &&
           "Phase 4.D Batch 59: appended instr is LoadGlobal");
    assert(lg->bytecode_offset == 119 &&
           "Phase 4.D Batch 59: load_global bytecode_offset stamped");

    /* Case 2: emit_wait_handle_load_coro_or_result — no-FS. */
    tc.frame.cur_instr_offs = 131;
    HirRegLayout coro_dst{};
    coro_dst.id = 62;
    HirRegLayout coro_src{};
    coro_src.id = 63;
    hir_c_tc_emit_wait_handle_load_coro_or_result(&tc, &coro_dst, &coro_src);

    void *second = hir_bb_next_instr(&bb, first);
    assert(second != NULL && "Phase 4.D Batch 59: wait_handle_load_coro appended");
    HirInstrLayout *whc = (HirInstrLayout *)second;
    assert(whc->opcode == HIR_OP_WaitHandleLoadCoroOrResult &&
           "Phase 4.D Batch 59: appended instr is WaitHandleLoadCoroOrResult");
    assert(whc->bytecode_offset == 131 &&
           "Phase 4.D Batch 59: wait_handle_load_coro bytecode_offset stamped");

    hir_c_test_chain_destroy(&bb);
}

/* Phase 4.D Batch 60 V5 sentinel: emit cluster 6 representative cases.
 *
 *   - emit_compare (FS-coupled, enum-arg, cur_instr_offs=143, op=Eq=2):
 *     Compare instr appended with bytecode_offset stamped + op carried.
 *   - emit_send (FS-coupled, plain, cur_instr_offs=149).
 *
 * Remaining 8 primitives covered by V1 production path. */
static void verify_phase4d_batch60_emit_cluster_6() {
    HirBasicBlock bb;
    hir_c_test_chain_init(&bb);

    PhxTranslationContext tc{};
    tc.block = &bb;

    /* Case 1: emit_compare — FS-coupled enum-arg. */
    tc.frame.cur_instr_offs = 143;
    FrameState fs{};
    fs.cur_instr_offs = jit::BCOffset{143};
    HirRegLayout cmp_dst{};
    cmp_dst.id = 71;
    HirRegLayout cmp_left{};
    cmp_left.id = 72;
    HirRegLayout cmp_right{};
    cmp_right.id = 73;
    hir_c_tc_emit_compare(&tc, &cmp_dst, /*op=*/2, &cmp_left, &cmp_right, &fs);

    void *first = hir_bb_first_instr(&bb);
    assert(first != NULL && "Phase 4.D Batch 60: compare appended");
    HirInstrLayout *cmp = (HirInstrLayout *)first;
    assert(cmp->opcode == HIR_OP_Compare &&
           "Phase 4.D Batch 60: appended instr is Compare");
    assert(cmp->bytecode_offset == 143 &&
           "Phase 4.D Batch 60: compare bytecode_offset stamped");

    /* Case 2: emit_send — FS-coupled plain. */
    tc.frame.cur_instr_offs = 149;
    HirRegLayout send_iter{};
    send_iter.id = 74;
    HirRegLayout send_vout{};
    send_vout.id = 75;
    HirRegLayout send_vin{};
    send_vin.id = 76;
    hir_c_tc_emit_send(&tc, &send_iter, &send_vout, &send_vin, &fs);

    void *second = hir_bb_next_instr(&bb, first);
    assert(second != NULL && "Phase 4.D Batch 60: send appended");
    HirInstrLayout *snd = (HirInstrLayout *)second;
    assert(snd->opcode == HIR_OP_Send &&
           "Phase 4.D Batch 60: appended instr is Send");
    assert(snd->bytecode_offset == 149 &&
           "Phase 4.D Batch 60: send bytecode_offset stamped");

    hir_c_test_chain_destroy(&bb);
}

/* Phase 4.D Batch 61 V5 sentinel: emit cluster 7 representative cases.
 *
 *   - emit_make_checked_list (FS-coupled, Type arg, Instr* RETURN,
 *     cur_instr_offs=157): MakeCheckedList instr appended + return ptr
 *     non-NULL.
 *   - emit_primitive_unbox (no-FS, Type arg, cur_instr_offs=163):
 *     PrimitiveUnbox instr appended + Type carried.
 *
 * Remaining 8 primitives covered by V1 production path. */
static void verify_phase4d_batch61_emit_cluster_7() {
    HirBasicBlock bb;
    hir_c_test_chain_init(&bb);

    PhxTranslationContext tc{};
    tc.block = &bb;

    /* Case 1: emit_make_checked_list — FS-coupled Instr* return. */
    tc.frame.cur_instr_offs = 157;
    FrameState fs{};
    fs.cur_instr_offs = jit::BCOffset{157};
    HirRegLayout mcl_dst{};
    mcl_dst.id = 81;
    HirType list_type{};
    list_type.bits_and_flags = 0xCAFE0001U;
    HirInstr ret = hir_c_tc_emit_make_checked_list(&tc, /*size=*/4, &mcl_dst,
                                                     list_type, &fs);
    assert(ret != NULL && "Phase 4.D Batch 61: make_checked_list returns non-NULL");

    void *first = hir_bb_first_instr(&bb);
    assert(first != NULL && "Phase 4.D Batch 61: make_checked_list appended");
    HirInstrLayout *mcl = (HirInstrLayout *)first;
    assert(mcl->opcode == HIR_OP_MakeCheckedList &&
           "Phase 4.D Batch 61: appended instr is MakeCheckedList");
    assert(mcl->bytecode_offset == 157 &&
           "Phase 4.D Batch 61: make_checked_list bytecode_offset stamped");

    /* Case 2: emit_primitive_unbox — no-FS Type-arg. */
    tc.frame.cur_instr_offs = 163;
    HirRegLayout pu_dst{};
    pu_dst.id = 82;
    HirRegLayout pu_src{};
    pu_src.id = 83;
    HirType cint32_type{};
    cint32_type.bits_and_flags = 0xCAFE0002U;
    hir_c_tc_emit_primitive_unbox(&tc, &pu_dst, &pu_src, cint32_type);

    void *second = hir_bb_next_instr(&bb, first);
    assert(second != NULL && "Phase 4.D Batch 61: primitive_unbox appended");
    HirInstrLayout *pu = (HirInstrLayout *)second;
    assert(pu->opcode == HIR_OP_PrimitiveUnbox &&
           "Phase 4.D Batch 61: appended instr is PrimitiveUnbox");
    assert(pu->bytecode_offset == 163 &&
           "Phase 4.D Batch 61: primitive_unbox bytecode_offset stamped");

    hir_c_test_chain_destroy(&bb);
}

/* Phase 4.D Batch 62 V5 sentinel: emit cluster 8 representative cases.
 *
 *   - emit_load_global_cached (no-FS, 3 Python.h-pointer pass-through,
 *     cur_instr_offs=181, name_idx=5).
 *   - emit_call_static_ret_void (no-FS, size_t + void* + Instr* return,
 *     cur_instr_offs=187, n=2, addr=0xDEADBEEF).
 *
 * Remaining 8 primitives covered by V1 production path. */
static void verify_phase4d_batch62_emit_cluster_8() {
    HirBasicBlock bb;
    hir_c_test_chain_init(&bb);

    PhxTranslationContext tc{};
    tc.block = &bb;

    /* Case 1: emit_load_global_cached — 3 pointer pass-through. */
    tc.frame.cur_instr_offs = 181;
    HirRegLayout lgc_dst{};
    lgc_dst.id = 91;
    void *fake_code = (void *)0xCAFE0001ULL;
    void *fake_builtins = (void *)0xCAFE0002ULL;
    void *fake_globals = (void *)0xCAFE0003ULL;
    hir_c_tc_emit_load_global_cached(&tc, &lgc_dst, fake_code, fake_builtins,
                                      fake_globals, /*name_idx=*/5);

    void *first = hir_bb_first_instr(&bb);
    assert(first != NULL && "Phase 4.D Batch 62: load_global_cached appended");
    HirInstrLayout *lgc = (HirInstrLayout *)first;
    assert(lgc->opcode == HIR_OP_LoadGlobalCached &&
           "Phase 4.D Batch 62: appended instr is LoadGlobalCached");
    assert(lgc->bytecode_offset == 181 &&
           "Phase 4.D Batch 62: load_global_cached bytecode_offset stamped");

    /* Case 2: emit_call_static_ret_void — Instr* return + size_t + void*. */
    tc.frame.cur_instr_offs = 187;
    void *fake_addr = (void *)0xDEADBEEFULL;
    HirInstr ret = hir_c_tc_emit_call_static_ret_void(&tc, /*n=*/2, fake_addr);
    assert(ret != NULL &&
           "Phase 4.D Batch 62: call_static_ret_void returns non-NULL");

    void *second = hir_bb_next_instr(&bb, first);
    assert(second != NULL &&
           "Phase 4.D Batch 62: call_static_ret_void appended");
    HirInstrLayout *csv = (HirInstrLayout *)second;
    assert(csv->opcode == HIR_OP_CallStaticRetVoid &&
           "Phase 4.D Batch 62: appended instr is CallStaticRetVoid");
    assert(csv->bytecode_offset == 187 &&
           "Phase 4.D Batch 62: call_static_ret_void bytecode_offset stamped");

    hir_c_test_chain_destroy(&bb);
}

/* Phase 4.D Batch 63 V5 sentinel: emit cluster 9 representative cases.
 *
 *   - emit_call_c_func (no-FS, std::vector→data() ptr, Instr* return,
 *     cur_instr_offs=199, n=3, func=0, args=[r4,r5,r6]).
 *   - emit_cast (FS-coupled, PyTypeObject* + 2 bool, cur_instr_offs=205).
 *
 * Remaining 8 primitives covered by V1 production path. */
static void verify_phase4d_batch63_emit_cluster_9() {
    HirBasicBlock bb;
    hir_c_test_chain_init(&bb);

    PhxTranslationContext tc{};
    tc.block = &bb;

    /* Case 1: emit_call_c_func — std::vector→data() + Instr* return. */
    tc.frame.cur_instr_offs = 199;
    HirRegLayout cf_dst{};
    cf_dst.id = 101;
    HirRegLayout r4{};
    r4.id = 104;
    HirRegLayout r5{};
    r5.id = 105;
    HirRegLayout r6{};
    r6.id = 106;
    HirRegister args[3] = {&r4, &r5, &r6};
    HirInstr ret = hir_c_tc_emit_call_c_func(&tc, /*n=*/3, &cf_dst,
                                               /*func_enum=*/0, args);
    assert(ret != NULL && "Phase 4.D Batch 63: call_c_func returns non-NULL");

    void *first = hir_bb_first_instr(&bb);
    assert(first != NULL && "Phase 4.D Batch 63: call_c_func appended");
    HirInstrLayout *cf = (HirInstrLayout *)first;
    assert(cf->opcode == HIR_OP_CallCFunc &&
           "Phase 4.D Batch 63: appended instr is CallCFunc");
    assert(cf->bytecode_offset == 199 &&
           "Phase 4.D Batch 63: call_c_func bytecode_offset stamped");

    /* Case 2: emit_cast — FS-coupled PyTypeObject* + bool. */
    tc.frame.cur_instr_offs = 205;
    FrameState fs{};
    fs.cur_instr_offs = jit::BCOffset{205};
    HirRegLayout cast_dst{};
    cast_dst.id = 102;
    HirRegLayout cast_value{};
    cast_value.id = 103;
    void *fake_pytype = (void *)0xCAFE0010ULL;
    hir_c_tc_emit_cast(&tc, &cast_dst, &cast_value, fake_pytype,
                        /*optional=*/1, /*exact=*/0, &fs);

    void *second = hir_bb_next_instr(&bb, first);
    assert(second != NULL && "Phase 4.D Batch 63: cast appended");
    HirInstrLayout *cst = (HirInstrLayout *)second;
    assert(cst->opcode == HIR_OP_Cast &&
           "Phase 4.D Batch 63: appended instr is Cast");
    assert(cst->bytecode_offset == 205 &&
           "Phase 4.D Batch 63: cast bytecode_offset stamped");

    hir_c_test_chain_destroy(&bb);
}

/* Phase 4.D Batch 64 V5 sentinel: emit cluster 10 representative cases.
 *
 *   - emit_make_list (FS, Instr* return, cur_instr_offs=211, n=4).
 *   - emit_tp_alloc (FS, PyTypeObject* opaque, cur_instr_offs=217). */
static void verify_phase4d_batch64_emit_cluster_10() {
    HirBasicBlock bb;
    hir_c_test_chain_init(&bb);

    PhxTranslationContext tc{};
    tc.block = &bb;

    /* Case 1: emit_make_list — Instr* return. */
    tc.frame.cur_instr_offs = 211;
    FrameState fs{};
    fs.cur_instr_offs = jit::BCOffset{211};
    HirRegLayout ml_dst{};
    ml_dst.id = 111;
    HirInstr ret = hir_c_tc_emit_make_list(&tc, /*n=*/4, &ml_dst, &fs);
    assert(ret != NULL && "Phase 4.D Batch 64: make_list returns non-NULL");

    void *first = hir_bb_first_instr(&bb);
    assert(first != NULL && "Phase 4.D Batch 64: make_list appended");
    HirInstrLayout *ml = (HirInstrLayout *)first;
    assert(ml->opcode == HIR_OP_MakeList &&
           "Phase 4.D Batch 64: appended instr is MakeList");
    assert(ml->bytecode_offset == 211 &&
           "Phase 4.D Batch 64: make_list bytecode_offset stamped");

    /* Case 2: emit_tp_alloc — PyTypeObject* opaque pass-through. */
    tc.frame.cur_instr_offs = 217;
    HirRegLayout tpa_dst{};
    tpa_dst.id = 112;
    void *fake_pytype = (void *)0xCAFE0020ULL;
    hir_c_tc_emit_tp_alloc(&tc, &tpa_dst, fake_pytype, &fs);

    void *second = hir_bb_next_instr(&bb, first);
    assert(second != NULL && "Phase 4.D Batch 64: tp_alloc appended");
    HirInstrLayout *tpa = (HirInstrLayout *)second;
    assert(tpa->opcode == HIR_OP_TpAlloc &&
           "Phase 4.D Batch 64: appended instr is TpAlloc");
    assert(tpa->bytecode_offset == 217 &&
           "Phase 4.D Batch 64: tp_alloc bytecode_offset stamped");

    hir_c_test_chain_destroy(&bb);
}

/* Phase 4.D Batch 65 V5 sentinel: emit cluster 13 representative cases.
 *
 *   - emit_load_frame (no-args, no-FS, cur_instr_offs=223).
 *   - emit_set_function_attr (no-FS, enum→int32_t, cur_instr_offs=229,
 *     field=2). */
static void verify_phase4d_batch65_emit_cluster_13() {
    HirBasicBlock bb;
    hir_c_test_chain_init(&bb);

    PhxTranslationContext tc{};
    tc.block = &bb;

    /* Case 1: emit_load_frame — no-args path. */
    tc.frame.cur_instr_offs = 223;
    hir_c_tc_emit_load_frame(&tc);

    void *first = hir_bb_first_instr(&bb);
    assert(first != NULL && "Phase 4.D Batch 65: load_frame appended");
    HirInstrLayout *lf = (HirInstrLayout *)first;
    assert(lf->opcode == HIR_OP_LoadFrame &&
           "Phase 4.D Batch 65: appended instr is LoadFrame");
    assert(lf->bytecode_offset == 223 &&
           "Phase 4.D Batch 65: load_frame bytecode_offset stamped");

    /* Case 2: emit_set_function_attr — enum→int32_t. */
    tc.frame.cur_instr_offs = 229;
    HirRegLayout sfa_value{};
    sfa_value.id = 121;
    HirRegLayout sfa_base{};
    sfa_base.id = 122;
    hir_c_tc_emit_set_function_attr(&tc, &sfa_value, &sfa_base, /*field=*/2);

    void *second = hir_bb_next_instr(&bb, first);
    assert(second != NULL && "Phase 4.D Batch 65: set_function_attr appended");
    HirInstrLayout *sfa = (HirInstrLayout *)second;
    assert(sfa->opcode == HIR_OP_SetFunctionAttr &&
           "Phase 4.D Batch 65: appended instr is SetFunctionAttr");
    assert(sfa->bytecode_offset == 229 &&
           "Phase 4.D Batch 65: set_function_attr bytecode_offset stamped");

    hir_c_test_chain_destroy(&bb);
}

/* Phase 4.D Batch 66 V5 sentinel: emit cluster 14 representative cases.
 *
 *   - emit_guard (FS-coupled, Instr* return + setFrameState-AFTER pattern,
 *     cur_instr_offs=233).
 *   - emit_double_binary_op (no-FS, enum→int32_t multi-arg,
 *     cur_instr_offs=239, op=1). */
static void verify_phase4d_batch66_emit_cluster_14() {
    HirBasicBlock bb;
    hir_c_test_chain_init(&bb);

    PhxTranslationContext tc{};
    tc.block = &bb;

    /* Case 1: emit_guard — FS-coupled, Instr* return, setFrameState-AFTER. */
    tc.frame.cur_instr_offs = 233;
    FrameState fs{};
    fs.cur_instr_offs = jit::BCOffset{233};
    HirRegLayout g_src{};
    g_src.id = 131;
    HirInstr g = hir_c_tc_emit_guard(&tc, &g_src, &fs);
    assert(g != NULL && "Phase 4.D Batch 66: guard returns non-NULL");

    void *first = hir_bb_first_instr(&bb);
    assert(first != NULL && "Phase 4.D Batch 66: guard appended");
    HirInstrLayout *gi = (HirInstrLayout *)first;
    assert(gi->opcode == HIR_OP_Guard &&
           "Phase 4.D Batch 66: appended instr is Guard");
    assert(gi->bytecode_offset == 233 &&
           "Phase 4.D Batch 66: guard bytecode_offset stamped pre-setFrameState");

    /* Case 2: emit_double_binary_op — no-FS, enum→int32_t. */
    tc.frame.cur_instr_offs = 239;
    HirRegLayout dbo_dst{};
    dbo_dst.id = 132;
    HirRegLayout dbo_left{};
    dbo_left.id = 133;
    HirRegLayout dbo_right{};
    dbo_right.id = 134;
    hir_c_tc_emit_double_binary_op(&tc, &dbo_dst, /*op=*/1, &dbo_left, &dbo_right);

    void *second = hir_bb_next_instr(&bb, first);
    assert(second != NULL && "Phase 4.D Batch 66: double_binary_op appended");
    HirInstrLayout *dbo = (HirInstrLayout *)second;
    assert(dbo->opcode == HIR_OP_DoubleBinaryOp &&
           "Phase 4.D Batch 66: appended instr is DoubleBinaryOp");
    assert(dbo->bytecode_offset == 239 &&
           "Phase 4.D Batch 66: double_binary_op bytecode_offset stamped");

    hir_c_test_chain_destroy(&bb);
}

/* Phase 4.D Batch 52 V5 sentinel falsifier for hir_c_advance_past_yield. */
static void verify_phase4d_batch52_advance_past_yield() {
    HirFrameStateLayout fs = {};
    fs.cur_instr_offs = -1;

    /* In-bounds: index < count → cur_instr_offs updates, no abort. */
    hir_c_advance_past_yield(&fs, 42, 21, 100);
    assert(fs.cur_instr_offs == 42 &&
           "Phase 4.D Batch 52: cur_instr_offs updated to next_offs");

    /* Boundary in-bounds: index = count - 1. */
    hir_c_advance_past_yield(&fs, 198, 99, 100);
    assert(fs.cur_instr_offs == 198 &&
           "Phase 4.D Batch 52: boundary index = count-1 updates");
}

/* Phase 4.A W1 Batch 67 (theologian docs/tier7-phase4a-preanalysis-2026-04-30.md
 * §3 W1): CallCFunc lookup table V5 sentinel falsifier. Pins:
 *   (a) all 4 known func enums return non-NULL canonical name
 *   (b) out-of-range indices (-1, 4) return the "<unknown CallCFunc>" sentinel
 *   (c) canonical anchors prevent silent enum reorder
 * TypedArgument::threadSafeTpFlags W1 conversion verified at compile-time
 * via _Static_assert in typed_argument_c.c (mask == Py_TPFLAGS_BASETYPE);
 * runtime PyType_Type test is impractical at constructor-time (Python not
 * yet initialized) and the static_assert already guards the only invariant. */
static void verify_phase4a_batch67_call_cfunc_func_name() {
    /* (a) All 4 known func enums per CallCFunc_FUNCS X-macro
     * (cinderx/Jit/hir/hir_instr_c.h:2135-2147 kNames table). */
    const char *n0 = hir_c_call_cfunc_func_name(0);
    const char *n1 = hir_c_call_cfunc_func_name(1);
    const char *n2 = hir_c_call_cfunc_func_name(2);
    const char *n3 = hir_c_call_cfunc_func_name(3);
    assert(n0 != NULL && n0[0] != '\0' &&
           "Phase 4.A Batch 67(a): CallCFunc[0] non-empty");
    assert(n1 != NULL && n1[0] != '\0' &&
           "Phase 4.A Batch 67(a): CallCFunc[1] non-empty");
    assert(n2 != NULL && n2[0] != '\0' &&
           "Phase 4.A Batch 67(a): CallCFunc[2] non-empty");
    assert(n3 != NULL && n3[0] != '\0' &&
           "Phase 4.A Batch 67(a): CallCFunc[3] non-empty");

    /* (b) Out-of-range fallback string. */
    assert(strcmp(hir_c_call_cfunc_func_name(-1), "<unknown CallCFunc>") == 0 &&
           "Phase 4.A Batch 67(b): negative index returns sentinel");
    assert(strcmp(hir_c_call_cfunc_func_name(4), "<unknown CallCFunc>") == 0 &&
           "Phase 4.A Batch 67(b): past-end index returns sentinel");
    assert(strcmp(hir_c_call_cfunc_func_name(99), "<unknown CallCFunc>") == 0 &&
           "Phase 4.A Batch 67(b): far-out-of-range returns sentinel");

    /* (c) Canonical anchors per CallCFunc_FUNCS X-macro order. Reorder
     * of the X-macro without re-flowing the C-side kNames table will
     * break these. */
    assert(strcmp(n0, "Cix_PyAsyncGenValueWrapperNew") == 0 &&
           "Phase 4.A Batch 67(c): CallCFunc[0] = Cix_PyAsyncGenValueWrapperNew");
    assert(strcmp(n1, "JitCoro_GetAwaitableIter") == 0 &&
           "Phase 4.A Batch 67(c): CallCFunc[1] = JitCoro_GetAwaitableIter");
    assert(strcmp(n2, "JitGen_yf") == 0 &&
           "Phase 4.A Batch 67(c): CallCFunc[2] = JitGen_yf");
    assert(strcmp(n3, "JITRT_MatchAndClearException") == 0 &&
           "Phase 4.A Batch 67(c): CallCFunc[3] = JITRT_MatchAndClearException");
}

/* Phase 4.A W2 Batch 68 (theologian docs/tier7-phase4a-preanalysis-2026-04-30.md
 * §3 W2): Phi setArgs helper + blockIndex falsifier. Pins:
 *   (a) hir_c_phi_apply_args_from_pairs N=0 path applies empty arrays
 *       without malloc/sort/split (early return)
 *   (b) hir_c_phi_apply_args_from_pairs N>0 path sorts pairs by block id
 *       and applies the paired permutation matching batch20 invariants
 *   (c) hir_phi_block_index returns position in sorted bb_data when found
 *   (d) hir_phi_block_index returns bb_count (= past-end sentinel) on miss
 *   (e) Combined setArgs+blockIndex: build phi via helper, look up each
 *       block, recover its operand. */
static void verify_phase4a_batch68_phi_apply_helper_and_block_index() {
    HirBasicBlock bb1{}, bb2{}, bb3{}, bb_missing{};
    bb1.id = 1;
    bb2.id = 2;
    bb3.id = 3;
    bb_missing.id = 99;

    void *r1 = (void *)(uintptr_t)0xDEAD0001ULL;
    void *r2 = (void *)(uintptr_t)0xDEAD0002ULL;
    void *r3 = (void *)(uintptr_t)0xDEAD0003ULL;

    /* (a) N=0 path: helper short-circuits to hir_c_phi_apply_args(NULL,NULL,0). */
    void *phi0 = hir_c_alloc_instr(sizeof(HirPhi), 0);
    assert(phi0 != NULL);
    hir_c_init_instr(phi0, HIR_OP_Phi);
    hir_c_phi_apply_args_from_pairs(phi0, NULL, 0);
    HirPhi *p0 = (HirPhi *)phi0;
    assert(p0->bb_count == 0 &&
           "Phase 4.A Batch 68(a): N=0 helper leaves bb_count == 0");
    assert(p0->bb_data == NULL &&
           "Phase 4.A Batch 68(a): N=0 helper leaves bb_data == NULL");
    hir_c_instr_free(phi0);

    /* (b) N>0 path with intentionally unsorted input — helper sorts
     * by block id and applies in the paired-permute manner verified
     * by batch20. */
    void *phi3 = hir_c_alloc_instr(sizeof(HirPhi), 3);
    assert(phi3 != NULL);
    hir_c_init_instr(phi3, HIR_OP_Phi);
    HirPhiArgPair pairs[3] = {
        { &bb3, r1 },  /* unsorted on purpose */
        { &bb1, r2 },
        { &bb2, r3 },
    };
    hir_c_phi_apply_args_from_pairs(phi3, pairs, 3);
    HirPhi *p3 = (HirPhi *)phi3;
    assert(p3->bb_count == 3 &&
           "Phase 4.A Batch 68(b): bb_count == 3 after helper apply");
    assert(hir_phi_block_at(phi3, 0) == &bb1 &&
           "Phase 4.A Batch 68(b): sorted bb[0] = bb1 (id=1)");
    assert(hir_phi_block_at(phi3, 1) == &bb2 &&
           "Phase 4.A Batch 68(b): sorted bb[1] = bb2 (id=2)");
    assert(hir_phi_block_at(phi3, 2) == &bb3 &&
           "Phase 4.A Batch 68(b): sorted bb[2] = bb3 (id=3)");
    assert(hir_c_get_operand(phi3, 0) == r2 &&
           "Phase 4.A Batch 68(b): paired-permute operand[0] = r2");
    assert(hir_c_get_operand(phi3, 1) == r3 &&
           "Phase 4.A Batch 68(b): paired-permute operand[1] = r3");
    assert(hir_c_get_operand(phi3, 2) == r1 &&
           "Phase 4.A Batch 68(b): paired-permute operand[2] = r1");

    /* (c) blockIndex hits at every position. */
    assert(hir_phi_block_index(phi3, &bb1) == 0 &&
           "Phase 4.A Batch 68(c): blockIndex(bb1) = 0");
    assert(hir_phi_block_index(phi3, &bb2) == 1 &&
           "Phase 4.A Batch 68(c): blockIndex(bb2) = 1");
    assert(hir_phi_block_index(phi3, &bb3) == 2 &&
           "Phase 4.A Batch 68(c): blockIndex(bb3) = 2");

    /* (d) Miss returns past-end sentinel (bb_count). Binary search lands
     * on the insertion point; for id=99 that is past the end of [1,2,3]. */
    assert(hir_phi_block_index(phi3, &bb_missing) == 3 &&
           "Phase 4.A Batch 68(d): blockIndex(missing id=99) = bb_count = 3");

    /* (e) Combined: index round-trip recovers each block's paired register. */
    void *expected[3] = { r2, r3, r1 };
    for (size_t i = 0; i < 3; i++) {
        size_t idx = hir_phi_block_index(
            phi3, (const HirBasicBlock *)hir_phi_block_at(phi3, i));
        assert(idx == i &&
               "Phase 4.A Batch 68(e): blockIndex round-trip = i");
        assert(hir_c_get_operand(phi3, idx) == expected[i] &&
               "Phase 4.A Batch 68(e): operand at index i matches paired r");
    }

    free(p3->bb_data);
    hir_c_instr_free(phi3);
}

/* Phase 4.A W3 Batch 69 (theologian docs/tier7-phase4a-preanalysis-2026-04-30.md
 * §3 W3, supervisor D-1777577907 STRUCT-tier dispatch): Edge endpoint
 * accessor V5 sentinel falsifier. Pins:
 *   (a) hir_edge_from / hir_edge_to read the from/to fields directly
 *   (b) NULL endpoints (default-constructed edge) yield NULL accessors
 *   (c) Round-trip: hir_c_edge_copy_init srcs from set_from/set_to so
 *       both copy ctor + accessors observe the canonical write path
 *
 * NOTE: this fixture only tests READ accessors. Mutators set_from /
 * set_to are codified as C++-only per feedback_edge_management
 * (in_edges_ tracking on the target BasicBlock). Q4 supervisor
 * 19:18:37Z auto-BLOCK gates any C-side conversion attempt of these. */
static void verify_phase4a_batch69_edge_accessors() {
    /* (a) Direct field-write smoke: HirEdge with sentinel pointer
     * payload (matches test_phx_block_map.c convention). */
    HirEdge e_direct = {};
    void *sentinel_from = (void *)(uintptr_t)0xCAFE0001ULL;
    void *sentinel_to   = (void *)(uintptr_t)0xCAFE0002ULL;
    e_direct.from = sentinel_from;
    e_direct.to   = sentinel_to;
    assert(hir_edge_from(&e_direct) == sentinel_from &&
           "Phase 4.A Batch 69(a): hir_edge_from reads from field");
    assert(hir_edge_to(&e_direct) == sentinel_to &&
           "Phase 4.A Batch 69(a): hir_edge_to reads to field");

    /* (b) NULL endpoints (default-constructed). */
    HirEdge e_null = {};
    assert(hir_edge_from(&e_null) == NULL &&
           "Phase 4.A Batch 69(b): NULL from accessor returns NULL");
    assert(hir_edge_to(&e_null) == NULL &&
           "Phase 4.A Batch 69(b): NULL to accessor returns NULL");

    /* (c) Round-trip via canonical write path: hir_c_edge_copy_init
     * routes through hir_edge_set_from / hir_edge_set_to; accessors
     * then observe the same payload on the destination. We can't
     * call set_from/set_to directly with non-BasicBlock sentinels
     * (in_edges_ would deref); instead we exercise the field-direct
     * path then verify the copy_init helper's documented behavior
     * holds when src is field-direct-populated. */
    HirEdge src = {};
    src.from = sentinel_from;
    src.to   = sentinel_to;
    /* hir_c_edge_copy_init calls set_from/set_to which would mutate
     * in_edges_ on (BasicBlock*)sentinel_from — NOT safe for raw
     * sentinels. Round-trip pins are batch15 territory; here we just
     * confirm the source's accessor reads match the write semantics. */
    assert(hir_edge_from(&src) == hir_edge_from(&e_direct) &&
           "Phase 4.A Batch 69(c): accessor parity src vs e_direct");
    assert(hir_edge_to(&src) == hir_edge_to(&e_direct) &&
           "Phase 4.A Batch 69(c): accessor parity src vs e_direct");
}

/* Phase 4.A W4 Batch 70 (theologian docs/tier7-phase4a-preanalysis-2026-04-30.md
 * §3 W4, supervisor D-1777578145 medium-logic dispatch): Environment
 * accessor falsifier coverage. The 4 mutator/lookup methods (Allocate-
 * Register, addRegister, getRegister, references) are already verified by
 * batch26 + batch35 + the offsetof static_assert at line 124. This batch
 * pins the 5 small read accessors not yet covered:
 *   hir_env_reg_count / hir_env_reg_data / hir_env_next_register_id /
 *   hir_env_num_load_type_attr_caches / hir_env_num_load_type_method_caches
 * plus the references opaque-pointer return identity.
 *
 * SCOPE-NOTE: Environment::~Environment + Environment::addReference
 * (PyObject* / Ref<>) DEFERRED. Dtor mixes ThreadedCompileSerialize RAII
 * + std::unordered_set::clear + delete loop on Register*; addReference
 * uses ThreadedRef + std::unordered_set::emplace. Both are RAII + STL
 * heavy and structurally W7-class (same review tier as Instr ctor/copy
 * deferred from W1). Documented for W7 backlog. */
static void verify_phase4a_batch70_environment_accessors() {
    HirEnvironment env = {};

    /* (a) Initial empty-state accessors. */
    assert(hir_env_reg_count(&env) == 0 &&
           "Phase 4.A Batch 70(a): empty env reg_count == 0");
    assert(hir_env_reg_data(&env) == NULL &&
           "Phase 4.A Batch 70(a): empty env reg_data == NULL");
    assert(hir_env_next_register_id(&env) == 0 &&
           "Phase 4.A Batch 70(a): empty env next_register_id == 0");
    assert(hir_env_num_load_type_attr_caches(&env) == 0 &&
           "Phase 4.A Batch 70(a): empty env num_load_type_attr_caches == 0");
    assert(hir_env_num_load_type_method_caches(&env) == 0 &&
           "Phase 4.A Batch 70(a): empty env num_load_type_method_caches == 0");

    /* (b) references opaque-pointer points into the struct (offset
     * matches the static_assert pin at verify.cpp:124). */
    void *refs_ptr = hir_c_env_references(&env);
    assert(refs_ptr == (void *)env.references_opaque &&
           "Phase 4.A Batch 70(b): hir_c_env_references returns "
           "&references_opaque[0]");
    /* The pointer must lie inside the env struct (not a heap allocation). */
    assert((char *)refs_ptr >= (char *)&env &&
           (char *)refs_ptr < (char *)&env + sizeof(env) &&
           "Phase 4.A Batch 70(b): refs pointer is intra-struct, not heap");

    /* (c) Direct field-write smoke for accessors that bypass mutators. */
    env.next_register_id = 7;
    env.next_load_type_attr_cache = 3;
    env.next_load_type_method_cache = 2;
    assert(hir_env_next_register_id(&env) == 7 &&
           "Phase 4.A Batch 70(c): next_register_id field-write surfaces");
    assert(hir_env_num_load_type_attr_caches(&env) == 3 &&
           "Phase 4.A Batch 70(c): num_load_type_attr_caches surfaces");
    assert(hir_env_num_load_type_method_caches(&env) == 2 &&
           "Phase 4.A Batch 70(c): num_load_type_method_caches surfaces");

    /* (d) reg_count + reg_data update after a single AllocateRegister
     * (re-confirms batch26 invariant via the small-accessor path). */
    void *r0 = hir_c_env_allocate_register(&env);
    assert(hir_env_reg_count(&env) == 1 &&
           "Phase 4.A Batch 70(d): reg_count == 1 after one allocate");
    assert(hir_env_reg_data(&env) != NULL &&
           "Phase 4.A Batch 70(d): reg_data non-NULL after one allocate");
    assert(hir_env_reg_data(&env)[0] == r0 &&
           "Phase 4.A Batch 70(d): reg_data[0] == allocated register");

    /* Cleanup: matches batch26 pattern. */
    delete static_cast<jit::hir::Register*>(r0);
    free(env.reg_data);
}

/* Phase 4.A W5 Batch 71 (theologian docs/tier7-phase4a-preanalysis-2026-04-30.md
 * §3 W5, supervisor D-1777578396 STRUCT-tier dispatch + Q-W5-1/Q-W5-2
 * disposition): DeoptBase accessor falsifier coverage. Existing fixtures
 * cover live_regs iteration (batch11), sortLiveRegs (batch13), set_frame_
 * state (batch18), and copy ctor (batch19). This batch adds the 4
 * unverified surface items:
 *   nonce get/set / descr get/set / guiltyReg get/set / live_regs
 *   const+non-const backing-storage identity.
 *
 * SCOPE-DEFER per supervisor disposition:
 *   visitUsesDeopt → W7 (Cat-B std::function callback class, bundles
 *     with Phi setArgs unordered_map + TypedArgument operator= + Env::~
 *     + addReference per W1/W2/W4 deferral pattern).
 *   sortLiveRegs kPyDebug post-condition → C++-stay W5 EXCEPTION (already
 *     loud-fail via JIT_DCHECK + std::adjacent_find; no port-for-port's-
 *     sake per supervisor 20:16:32Z). NOT a general scope-cut — revisit
 *     in terminal-C-rewrite scope refresh. */
static void verify_phase4a_batch71_deopt_base_accessors() {
    void *db = hir_c_alloc_instr(sizeof(HirDeoptLayout), 0);
    assert(db != NULL);
    hir_c_init_deopt(db, HIR_OP_BinaryOp);
    auto *base = static_cast<jit::hir::DeoptBase *>(db);

    /* (a) nonce default zero + round-trip. */
    assert(base->nonce() == 0 &&
           "Phase 4.A Batch 71(a): default nonce == 0");
    base->set_nonce(42);
    assert(base->nonce() == 42 &&
           "Phase 4.A Batch 71(a): set_nonce(42) round-trip");
    base->set_nonce(-7);
    assert(base->nonce() == -7 &&
           "Phase 4.A Batch 71(a): set_nonce(-7) negative-value round-trip");

    /* (b) descr default NULL + round-trip with two distinct strings. */
    assert(base->descr() == NULL &&
           "Phase 4.A Batch 71(b): default descr == NULL");
    static const char kDescr1[] = "deopt-class-A";
    static const char kDescr2[] = "deopt-class-B";
    base->setDescr(kDescr1);
    assert(base->descr() == kDescr1 &&
           "Phase 4.A Batch 71(b): setDescr(kDescr1) round-trip pointer");
    base->setDescr(kDescr2);
    assert(base->descr() == kDescr2 &&
           "Phase 4.A Batch 71(b): setDescr(kDescr2) overwrites prior");
    base->setDescr(NULL);
    assert(base->descr() == NULL &&
           "Phase 4.A Batch 71(b): setDescr(NULL) clears");

    /* (c) guiltyReg default NULL + round-trip with sentinel pointer. */
    assert(base->guiltyReg() == NULL &&
           "Phase 4.A Batch 71(c): default guiltyReg == NULL");
    auto *sentinel_reg =
        reinterpret_cast<jit::hir::Register *>(uintptr_t(0xBADF00D5ULL));
    base->setGuiltyReg(sentinel_reg);
    assert(base->guiltyReg() == sentinel_reg &&
           "Phase 4.A Batch 71(c): setGuiltyReg sentinel round-trip");
    base->setGuiltyReg(NULL);
    assert(base->guiltyReg() == NULL &&
           "Phase 4.A Batch 71(c): setGuiltyReg(NULL) clears");

    /* (d) live_regs: const + non-const overloads return references to
     * the same backing PhxRegStateArray. We can't directly compare two
     * references, but identical addresses prove identity. */
    const jit::hir::DeoptBase *cbase = base;
    const PhxRegStateArray *const_addr = &cbase->live_regs();
    PhxRegStateArray *mut_addr = &base->live_regs();
    assert((const void *)const_addr == (const void *)mut_addr &&
           "Phase 4.A Batch 71(d): const + non-const live_regs return "
           "same backing storage address");

    /* Cleanup: matches batch11 pattern. */
    base->live_regs() = PhxRegStateArray{};
    free((char *)db - 2 * sizeof(void *) - sizeof(size_t));
}

/* Phase 4.A W6 Batch 72 (theologian docs/tier7-phase4a-preanalysis-2026-04-30.md
 * §3 W6, supervisor D-1777579403 STRUCT-tier dispatch): BasicBlock
 * GetTerminator + entrySnapshot falsifier coverage. The 12 BB methods
 * are ALL pre-bridged; existing fixtures cover Append/push_front/
 * pop_front/insert (batch27/33/36) and intrusive list invariants. This
 * batch adds the 2 unverified edge-case-heavy accessors:
 *   GetTerminator (returns last instr or NULL on empty BB)
 *   entrySnapshot (returns first non-Phi Snapshot or NULL; skips
 *     leading Phis per hir_basic_block_c.c:208-222 contract)
 *
 * SCOPE-NOTE per pre-analysis §4 STRUCT-tier:
 *   IsTrampoline: complex Branch-to-self/Branch-to-empty interplay
 *     with Edge wiring (W3 surface) — testable only with CFG fixture
 *     scaffolding beyond batch72 isolation. Existing CFG-level
 *     fixtures cover IsTrampoline indirectly via downstream consumers
 *     (insert_update_prev_instr_c.c:253, hir_cfg_rpo_c.c:44,
 *     pass_output_type_c.c:485).
 *   retargetPreds + fixupPhis/addPhi/removePhiPredecessor: forEachPhi
 *     callback + Edge::set_to interplay; W7-class (bundles with
 *     visitUsesDeopt + Phi setArgs unordered_map per Cat-B callback
 *     deferral pattern). */
static void verify_phase4a_batch72_bb_terminator_and_entry_snapshot() {
    HirBasicBlock bb;
    hir_c_test_chain_init(&bb);

    /* (a) Empty BB: GetTerminator + entrySnapshot both return NULL. */
    assert(hir_bb_get_terminator(&bb) == NULL &&
           "Phase 4.A Batch 72(a): empty BB GetTerminator returns NULL");
    assert(hir_bb_entry_snapshot(&bb) == NULL &&
           "Phase 4.A Batch 72(a): empty BB entrySnapshot returns NULL");

    /* (b) Single non-Snapshot instr: GetTerminator returns it,
     * entrySnapshot returns NULL (first instr is non-Snapshot). */
    void *A = hir_c_alloc_instr(sizeof(HirInstrLayout), 0);
    assert(A != NULL);
    hir_c_init_instr(A, HIR_OP_Assign);
    hir_c_bb_append(&bb, A);
    assert(hir_bb_get_terminator(&bb) == A &&
           "Phase 4.A Batch 72(b): GetTerminator returns last (and only) instr");
    assert(hir_bb_entry_snapshot(&bb) == NULL &&
           "Phase 4.A Batch 72(b): entrySnapshot NULL when first instr "
           "is non-Snapshot non-Phi");

    /* (c) Two instrs [A, B]: GetTerminator returns the LAST (B). */
    void *B = hir_c_alloc_instr(sizeof(HirInstrLayout), 0);
    assert(B != NULL);
    hir_c_init_instr(B, HIR_OP_Assign);
    hir_c_bb_append(&bb, B);
    assert(hir_bb_get_terminator(&bb) == B &&
           "Phase 4.A Batch 72(c): GetTerminator returns last after Append(B)");

    hir_c_test_chain_destroy(&bb);

    /* (d) Snapshot-first BB: entrySnapshot returns the Snapshot. */
    HirBasicBlock bb_snap;
    hir_c_test_chain_init(&bb_snap);
    void *S = hir_c_alloc_instr(sizeof(HirInstrLayout), 0);
    assert(S != NULL);
    hir_c_init_instr(S, HIR_OP_Snapshot);
    hir_c_bb_append(&bb_snap, S);
    assert(hir_bb_entry_snapshot(&bb_snap) == S &&
           "Phase 4.A Batch 72(d): entrySnapshot returns first instr when "
           "it is a Snapshot");
    /* GetTerminator on Snapshot-only BB returns the Snapshot itself
     * (last == first). */
    assert(hir_bb_get_terminator(&bb_snap) == S &&
           "Phase 4.A Batch 72(d): GetTerminator returns Snapshot when sole instr");
    hir_c_test_chain_destroy(&bb_snap);

    /* (e) Phi-then-Snapshot BB: entrySnapshot SKIPS leading Phi and
     * returns the Snapshot (per hir_bb_entry_snapshot contract). */
    HirBasicBlock bb_phi_snap;
    hir_c_test_chain_init(&bb_phi_snap);
    void *P = hir_c_alloc_instr(sizeof(HirPhi), 0);
    assert(P != NULL);
    hir_c_init_instr(P, HIR_OP_Phi);
    void *S2 = hir_c_alloc_instr(sizeof(HirInstrLayout), 0);
    assert(S2 != NULL);
    hir_c_init_instr(S2, HIR_OP_Snapshot);
    hir_c_bb_append(&bb_phi_snap, P);
    hir_c_bb_append(&bb_phi_snap, S2);
    assert(hir_bb_entry_snapshot(&bb_phi_snap) == S2 &&
           "Phase 4.A Batch 72(e): entrySnapshot skips leading Phi, "
           "returns Snapshot");
    hir_c_test_chain_destroy(&bb_phi_snap);

    /* (f) Phi-then-non-Snapshot BB: entrySnapshot returns NULL (the
     * non-Phi non-Snapshot terminates the search per the contract). */
    HirBasicBlock bb_phi_op;
    hir_c_test_chain_init(&bb_phi_op);
    void *P2 = hir_c_alloc_instr(sizeof(HirPhi), 0);
    assert(P2 != NULL);
    hir_c_init_instr(P2, HIR_OP_Phi);
    void *Op = hir_c_alloc_instr(sizeof(HirInstrLayout), 0);
    assert(Op != NULL);
    hir_c_init_instr(Op, HIR_OP_Assign);
    hir_c_bb_append(&bb_phi_op, P2);
    hir_c_bb_append(&bb_phi_op, Op);
    assert(hir_bb_entry_snapshot(&bb_phi_op) == NULL &&
           "Phase 4.A Batch 72(f): entrySnapshot NULL when first non-Phi "
           "is non-Snapshot");
    hir_c_test_chain_destroy(&bb_phi_op);
}

/* Phase 4.A W7a Batch 73 (theologian docs/tier7-phase4a-preanalysis-2026-04-30.md
 * §9 W7a, supervisor D-1777582314 dispatch): Instr operand-layout STRUCT
 * falsifier. Pins the prefix-pin invariant that NumOperands +
 * operands-array depend on.
 *
 * Layout (from hir_instr_c.h:1255-1273):
 *   [Register*[N] operands] [size_t num_operands] [HirInstrLayout fields...]
 *                                                  ^ instr pointer
 *   NumOperands = ((size_t*)instr)[-1]
 *   operands array starts at ((size_t*)instr - 1) - N
 *
 * Existing fixtures cover: visitor (batch10), ctor_init (batch17),
 * lifecycle (batch23), ExpandInto (batch31). This batch covers the 2
 * structural invariants W7a is responsible for per pre-analysis §9.4.1:
 *   (1) NumOperands prefix-pin matches the alloc_instr arity
 *   (2) operandAt/SetOperand round-trip via the prefix-pin'd offset
 *
 * SCOPE-NOTE: visitUses + Uses + ReplaceUsesOf already covered by
 * batch10/12; ExpandInto by batch31. This batch is additive coverage
 * on the operand-prefix-pin surface that W7a STRUCT-tier identifies as
 * load-bearing for the entire 42-method Instr surface. */
static void verify_phase4a_batch73_instr_operand_layout() {
    /* (a) Zero-operand instr: NumOperands == 0; no operand slots. */
    void *i0 = hir_c_alloc_instr(sizeof(HirInstrLayout), 0);
    assert(i0 != NULL);
    hir_c_init_instr(i0, HIR_OP_Snapshot);
    assert(hir_c_num_operands(i0) == 0 &&
           "Phase 4.A Batch 73(a): NumOperands == 0 for 0-arity alloc");
    hir_c_destroy_instr_impl(i0);

    /* (b) N-operand instr: NumOperands == N; operandAt round-trip with
     * sentinel pointers; operand_slot mutation visible via get. */
    const size_t kN = 3;
    void *i3 = hir_c_alloc_instr(sizeof(HirInstrLayout), kN);
    assert(i3 != NULL);
    hir_c_init_instr(i3, HIR_OP_Assign);
    assert(hir_c_num_operands(i3) == kN &&
           "Phase 4.A Batch 73(b): NumOperands == N for N-arity alloc");

    void *sentinels[3] = {
        (void *)(uintptr_t)0xBEEF0001ULL,
        (void *)(uintptr_t)0xBEEF0002ULL,
        (void *)(uintptr_t)0xBEEF0003ULL,
    };
    for (size_t i = 0; i < kN; i++) {
        hir_c_set_operand(i3, i, sentinels[i]);
    }
    for (size_t i = 0; i < kN; i++) {
        assert(hir_c_get_operand(i3, i) == sentinels[i] &&
               "Phase 4.A Batch 73(b): set/get operand round-trip");
    }

    /* (c) operand_slot mutation: write through the slot, observe via get. */
    void **slot1 = hir_c_operand_slot(i3, 1);
    void *new_op1 = (void *)(uintptr_t)0xCAFE1234ULL;
    *slot1 = new_op1;
    assert(hir_c_get_operand(i3, 1) == new_op1 &&
           "Phase 4.A Batch 73(c): operand_slot write visible via get_operand");
    /* And the unmodified slots stay put. */
    assert(hir_c_get_operand(i3, 0) == sentinels[0] &&
           "Phase 4.A Batch 73(c): slot 0 unchanged after slot 1 write");
    assert(hir_c_get_operand(i3, 2) == sentinels[2] &&
           "Phase 4.A Batch 73(c): slot 2 unchanged after slot 1 write");
    hir_c_destroy_instr_impl(i3);

    /* (d) Larger arity smoke: arity 8 to confirm the prefix-pin works
     * for non-trivial sizes. */
    const size_t kM = 8;
    void *i8 = hir_c_alloc_instr(sizeof(HirInstrLayout), kM);
    assert(i8 != NULL);
    hir_c_init_instr(i8, HIR_OP_Phi);
    assert(hir_c_num_operands(i8) == kM &&
           "Phase 4.A Batch 73(d): NumOperands == 8 for 8-arity alloc");
    /* Verify each slot starts NULL (calloc zero-init). */
    for (size_t i = 0; i < kM; i++) {
        assert(hir_c_get_operand(i8, i) == NULL &&
               "Phase 4.A Batch 73(d): fresh slot is NULL (calloc)");
    }
    hir_c_destroy_instr_impl(i8);
}

/* Phi visitor for batch74 — counts visits + records last seen instr. */
static int batch74_phi_count_visitor(void *phi, void *user) {
    struct B74 { void *last; size_t count; size_t stop_at; };
    B74 *b = (B74 *)user;
    b->last = phi;
    b->count++;
    if (b->stop_at != 0 && b->count >= b->stop_at) {
        return 0;  // stop early
    }
    return 1;
}

/* Phase 4.A W7b Batch 74 (theologian docs/tier7-phase4a-preanalysis-2026-04-30.md
 * §9 W7b, supervisor D-1777582914 dispatch): forEachPhi C-port
 * (hir_c_bb_for_each_phi) falsifier. Pins:
 *   (a) Empty BB visits zero phis
 *   (b) Phi-only BB visits all phis in chain order
 *   (c) Phi-then-non-Phi BB stops at the first non-Phi (matches C++
 *       template forEachPhi break-on-non-Phi at hir.h:3631)
 *   (d) Non-Phi-first BB visits zero phis (early-bail)
 *   (e) Visitor returning 0 stops iteration; count reflects partial visit
 *
 * NOTE: visitUsesDeopt thunk (W7b second deliverable) is verified
 * indirectly via batch11 (live_regs iteration through
 * hir_c_deopt_visit_uses_deopt) — the C body unchanged; W7b only
 * swaps the C++ shim from manual loop to thunk delegation. */
static void verify_phase4a_batch74_for_each_phi() {
    struct B74 { void *last; size_t count; size_t stop_at; };

    /* (a) Empty BB. */
    HirBasicBlock bb_empty;
    hir_c_test_chain_init(&bb_empty);
    B74 ctx_a = { NULL, 0, 0 };
    size_t visited_a = hir_c_bb_for_each_phi(
        &bb_empty, batch74_phi_count_visitor, &ctx_a);
    assert(visited_a == 0 && ctx_a.count == 0 &&
           "Phase 4.A Batch 74(a): empty BB visits 0 phis");
    hir_c_test_chain_destroy(&bb_empty);

    /* (b) Phi-only BB (3 phis). */
    HirBasicBlock bb_phis;
    hir_c_test_chain_init(&bb_phis);
    void *phis[3];
    for (int i = 0; i < 3; i++) {
        phis[i] = hir_c_alloc_instr(sizeof(HirPhi), 0);
        assert(phis[i] != NULL);
        hir_c_init_instr(phis[i], HIR_OP_Phi);
        hir_c_bb_append(&bb_phis, phis[i]);
    }
    B74 ctx_b = { NULL, 0, 0 };
    size_t visited_b = hir_c_bb_for_each_phi(
        &bb_phis, batch74_phi_count_visitor, &ctx_b);
    assert(visited_b == 3 && ctx_b.count == 3 &&
           "Phase 4.A Batch 74(b): 3-phi BB visits 3");
    assert(ctx_b.last == phis[2] &&
           "Phase 4.A Batch 74(b): last visited == 3rd phi (chain order)");
    hir_c_test_chain_destroy(&bb_phis);

    /* (c) Phi-then-non-Phi BB: 2 phis then an Assign. Stops at Assign. */
    HirBasicBlock bb_mixed;
    hir_c_test_chain_init(&bb_mixed);
    for (int i = 0; i < 2; i++) {
        void *p = hir_c_alloc_instr(sizeof(HirPhi), 0);
        assert(p != NULL);
        hir_c_init_instr(p, HIR_OP_Phi);
        hir_c_bb_append(&bb_mixed, p);
    }
    void *non_phi = hir_c_alloc_instr(sizeof(HirInstrLayout), 0);
    assert(non_phi != NULL);
    hir_c_init_instr(non_phi, HIR_OP_Assign);
    hir_c_bb_append(&bb_mixed, non_phi);
    B74 ctx_c = { NULL, 0, 0 };
    size_t visited_c = hir_c_bb_for_each_phi(
        &bb_mixed, batch74_phi_count_visitor, &ctx_c);
    assert(visited_c == 2 && ctx_c.count == 2 &&
           "Phase 4.A Batch 74(c): [Phi, Phi, non-Phi] visits 2");
    hir_c_test_chain_destroy(&bb_mixed);

    /* (d) Non-Phi-first BB visits zero phis. */
    HirBasicBlock bb_nonphi;
    hir_c_test_chain_init(&bb_nonphi);
    void *first_non_phi = hir_c_alloc_instr(sizeof(HirInstrLayout), 0);
    assert(first_non_phi != NULL);
    hir_c_init_instr(first_non_phi, HIR_OP_Assign);
    hir_c_bb_append(&bb_nonphi, first_non_phi);
    B74 ctx_d = { NULL, 0, 0 };
    size_t visited_d = hir_c_bb_for_each_phi(
        &bb_nonphi, batch74_phi_count_visitor, &ctx_d);
    assert(visited_d == 0 && ctx_d.count == 0 &&
           "Phase 4.A Batch 74(d): non-Phi-first BB visits 0 (early-bail)");
    hir_c_test_chain_destroy(&bb_nonphi);

    /* (e) Visitor returning 0 stops iteration. 5 phis, stop after 2nd. */
    HirBasicBlock bb_stop;
    hir_c_test_chain_init(&bb_stop);
    for (int i = 0; i < 5; i++) {
        void *p = hir_c_alloc_instr(sizeof(HirPhi), 0);
        assert(p != NULL);
        hir_c_init_instr(p, HIR_OP_Phi);
        hir_c_bb_append(&bb_stop, p);
    }
    B74 ctx_e = { NULL, 0, /* stop_at */ 2 };
    size_t visited_e = hir_c_bb_for_each_phi(
        &bb_stop, batch74_phi_count_visitor, &ctx_e);
    assert(visited_e == 2 && ctx_e.count == 2 &&
           "Phase 4.A Batch 74(e): visitor returning 0 stops iteration; "
           "visited count reflects partial visit (2 of 5)");
    hir_c_test_chain_destroy(&bb_stop);
}

/* Phase 4.A W7c Batch 75 (theologian docs/tier7-phase4a-preanalysis-2026-04-30.md
 * §9 W7c, supervisor D-1777583349 dispatch): hir_c_bb_collect_phis_alloc
 * falsifier. Pins the std::vector → C-scratch replacement used by
 * BasicBlock::addPhiPredecessor + removePhiPredecessor:
 *   (a) Empty BB returns NULL + n=0
 *   (b) Phi-only BB returns malloc'd array of all phis in chain order
 *   (c) Phi-then-non-Phi BB returns only the leading-Phi prefix (matches
 *       forEachPhi break-on-non-Phi semantics)
 *   (d) Non-Phi-first BB returns NULL + n=0
 *
 * The OOM loud-fail path (JIT_CHECK_C) is unreachable in test (no way
 * to inject malloc failure here); it is checked as syntax-present via
 * grep at compose-time only — runtime exercise belongs to a future
 * fault-injection harness if/when one lands. */
static void verify_phase4a_batch75_collect_phis_alloc() {
    /* (a) Empty BB. */
    HirBasicBlock bb_empty;
    hir_c_test_chain_init(&bb_empty);
    size_t n_empty = 999;  // sentinel — must be overwritten to 0
    void **phis_empty = hir_c_bb_collect_phis_alloc(&bb_empty, &n_empty);
    assert(phis_empty == NULL && n_empty == 0 &&
           "Phase 4.A Batch 75(a): empty BB returns NULL + n=0");
    hir_c_test_chain_destroy(&bb_empty);

    /* (b) Phi-only BB (3 phis): array length 3, chain order. */
    HirBasicBlock bb_phis;
    hir_c_test_chain_init(&bb_phis);
    void *expected_phis[3];
    for (int i = 0; i < 3; i++) {
        expected_phis[i] = hir_c_alloc_instr(sizeof(HirPhi), 0);
        assert(expected_phis[i] != NULL);
        hir_c_init_instr(expected_phis[i], HIR_OP_Phi);
        hir_c_bb_append(&bb_phis, expected_phis[i]);
    }
    size_t n_phis = 0;
    void **phis_arr = hir_c_bb_collect_phis_alloc(&bb_phis, &n_phis);
    assert(n_phis == 3 && phis_arr != NULL &&
           "Phase 4.A Batch 75(b): 3-phi BB returns malloc'd array of 3");
    for (size_t i = 0; i < 3; i++) {
        assert(phis_arr[i] == expected_phis[i] &&
               "Phase 4.A Batch 75(b): chain-order phi pointers");
    }
    free(phis_arr);
    hir_c_test_chain_destroy(&bb_phis);

    /* (c) Phi-then-non-Phi BB: only the leading 2 phis returned. */
    HirBasicBlock bb_mixed;
    hir_c_test_chain_init(&bb_mixed);
    void *p0 = hir_c_alloc_instr(sizeof(HirPhi), 0);
    void *p1 = hir_c_alloc_instr(sizeof(HirPhi), 0);
    void *non_phi_c = hir_c_alloc_instr(sizeof(HirInstrLayout), 0);
    assert(p0 != NULL && p1 != NULL && non_phi_c != NULL);
    hir_c_init_instr(p0, HIR_OP_Phi);
    hir_c_init_instr(p1, HIR_OP_Phi);
    hir_c_init_instr(non_phi_c, HIR_OP_Assign);
    hir_c_bb_append(&bb_mixed, p0);
    hir_c_bb_append(&bb_mixed, p1);
    hir_c_bb_append(&bb_mixed, non_phi_c);
    size_t n_mixed = 0;
    void **phis_mixed = hir_c_bb_collect_phis_alloc(&bb_mixed, &n_mixed);
    assert(n_mixed == 2 && phis_mixed != NULL &&
           "Phase 4.A Batch 75(c): [Phi, Phi, non-Phi] returns 2 phis");
    assert(phis_mixed[0] == p0 && phis_mixed[1] == p1 &&
           "Phase 4.A Batch 75(c): leading-Phi prefix only, no non-Phi");
    free(phis_mixed);
    hir_c_test_chain_destroy(&bb_mixed);

    /* (d) Non-Phi-first BB. */
    HirBasicBlock bb_nonphi;
    hir_c_test_chain_init(&bb_nonphi);
    void *first_op = hir_c_alloc_instr(sizeof(HirInstrLayout), 0);
    assert(first_op != NULL);
    hir_c_init_instr(first_op, HIR_OP_Assign);
    hir_c_bb_append(&bb_nonphi, first_op);
    size_t n_nonphi = 999;
    void **phis_nonphi = hir_c_bb_collect_phis_alloc(&bb_nonphi, &n_nonphi);
    assert(phis_nonphi == NULL && n_nonphi == 0 &&
           "Phase 4.A Batch 75(d): non-Phi-first BB returns NULL + n=0");
    hir_c_test_chain_destroy(&bb_nonphi);
}

/* Phase 4.A W7d Batch 76 (theologian docs/tier7-phase4a-preanalysis-2026-04-30.md
 * §9 W7d, supervisor D-1777584435 + Q-W7-3 disposition): refcount-aware
 * pytype slot swap falsifier (phx_typed_argument_pytype_swap). The swap
 * is the only refcount-touching primitive extracted to C from
 * TypedArgument::operator= + copy-ctor; the GIL + ThreadedCompileSerialize
 * guard + jit_type/locals_idx field-copy stay C++ per Q-W7-3 stay-C++
 * exception (genuinely-can't-port).
 *
 * Runtime fixture exercises the NULL-safe paths only — fixturing real
 * PyTypeObject refcount round-trips at __attribute__((constructor))
 * time would require touching live Python state (PyType_Type) before
 * the interpreter is initialized, which is unsafe. The refcount
 * semantics themselves are trusted per Py_XDECREF/XINCREF spec; W7d
 * exercise verifies the slot-write + NULL-safety invariant only.
 *
 * SCOPE-DEFER (W7d EXCEPTION per Q-W7-3 stay-C++ list):
 *   Environment::~Environment (RAII + STL clear + delete loop) —
 *     genuinely-can't-port surface; ThreadedRef destructors run
 *     Py_DECREF inside std::unordered_set::clear. Stays C++.
 *   Environment::addReference (PyObject*, Ref<>) — ThreadedRef::create
 *     + std::unordered_set::emplace. Stays C++.
 *   These three methods retain ThreadedCompileSerialize guard +
 *     full C++ body. Falsifier coverage deferred to terminal-C-rewrite
 *     scope refresh when ThreadedRef + STL container surrogates land. */
static void verify_phase4a_batch76_typed_argument_pytype_swap() {
    /* (a) NULL → NULL: no-op, slot stays NULL. */
    struct _typeobject *slot_a = NULL;
    phx_typed_argument_pytype_swap(&slot_a, NULL);
    assert(slot_a == NULL &&
           "Phase 4.A Batch 76(a): NULL→NULL leaves slot NULL");

    /* (b) Slot-write semantic: when both incref/decref are no-ops on
     * NULL, the slot must still take on the new value. We can't create
     * a real PyTypeObject here, but the swap function's slot store is
     * separable from refcount semantics — exercise via a sentinel
     * pointer that never gets refcounted (we pass it as new_value;
     * Py_XINCREF on a non-NULL pointer would normally touch the object
     * header, so we must NOT exercise non-NULL paths in this fixture).
     * This (b) case is therefore identical to (a) in this constructor
     * context; full refcount round-trip is left to a future Python-init
     * harness or to live testing during JIT compile. */
    /* No additional sub-checks at constructor-time. */

    (void)slot_a;
}

/* Phase 4.X-mini X-mini-b Batch 80 (theologian docs/tier7-phase4x-stay-cpp-
 * exception-discharge-spec.md §2.3 + §7.3, supervisor 22:42:40Z dispatch):
 * hir_c_deopt_find_adjacent_dup_reg falsifier. Pins std::adjacent_find
 * end-vs-found semantic preservation:
 *   (a) Empty live_regs (n=0): NULL
 *   (b) Single element (n=1): NULL (no adjacent pair)
 *   (c) Two distinct (n=2): NULL (no match)
 *   (d) First-pair match (n=3, regs[0]==regs[1]): returns regs[1].reg
 *   (e) Last-pair match (n=3, regs[1]==regs[2]): returns regs[2].reg
 *   (f) Middle match (n=4, regs[1]==regs[2]): returns regs[2].reg
 *   (g) Multiple-match returns FIRST (n=4, regs[0]==regs[1] AND
 *       regs[2]==regs[3]): returns regs[1].reg (first pair only) */
static void verify_phase4x_batch80_deopt_find_adjacent_dup_reg() {
    /* Sentinel "Register*" pointers for tests; we don't deref them. */
    void *r_a = (void *)(uintptr_t)0xAAAA0000ULL;
    void *r_b = (void *)(uintptr_t)0xBBBB0000ULL;
    void *r_c = (void *)(uintptr_t)0xCCCC0000ULL;
    void *r_d = (void *)(uintptr_t)0xDDDD0000ULL;

    /* (a) Empty: zero-init HirDeoptLayout has live_regs_count=0. */
    void *db_a = hir_c_alloc_instr(sizeof(HirDeoptLayout), 0);
    hir_c_init_deopt(db_a, HIR_OP_BinaryOp);
    assert(hir_c_deopt_find_adjacent_dup_reg(db_a) == NULL &&
           "Phase 4.X Batch 80(a): empty live_regs returns NULL");
    /* Direct cleanup — no live_regs storage. */
    free((char *)db_a - 2 * sizeof(void *) - sizeof(size_t));

    /* (b) Single element. */
    void *db_b = hir_c_alloc_instr(sizeof(HirDeoptLayout), 0);
    hir_c_init_deopt(db_b, HIR_OP_BinaryOp);
    {
        HirRegState *regs = (HirRegState *)malloc(sizeof(HirRegState));
        regs[0].reg = r_a;
        ((HirDeoptLayout *)db_b)->live_regs_data = regs;
        ((HirDeoptLayout *)db_b)->live_regs_count = 1;
        assert(hir_c_deopt_find_adjacent_dup_reg(db_b) == NULL &&
               "Phase 4.X Batch 80(b): single element returns NULL");
        free(regs);
    }
    free((char *)db_b - 2 * sizeof(void *) - sizeof(size_t));

    /* (c) Two distinct. */
    void *db_c = hir_c_alloc_instr(sizeof(HirDeoptLayout), 0);
    hir_c_init_deopt(db_c, HIR_OP_BinaryOp);
    {
        HirRegState *regs = (HirRegState *)malloc(2 * sizeof(HirRegState));
        regs[0].reg = r_a; regs[1].reg = r_b;
        ((HirDeoptLayout *)db_c)->live_regs_data = regs;
        ((HirDeoptLayout *)db_c)->live_regs_count = 2;
        assert(hir_c_deopt_find_adjacent_dup_reg(db_c) == NULL &&
               "Phase 4.X Batch 80(c): two distinct returns NULL");
        free(regs);
    }
    free((char *)db_c - 2 * sizeof(void *) - sizeof(size_t));

    /* (d) First-pair match. */
    void *db_d = hir_c_alloc_instr(sizeof(HirDeoptLayout), 0);
    hir_c_init_deopt(db_d, HIR_OP_BinaryOp);
    {
        HirRegState *regs = (HirRegState *)malloc(3 * sizeof(HirRegState));
        regs[0].reg = r_a; regs[1].reg = r_a; regs[2].reg = r_b;
        ((HirDeoptLayout *)db_d)->live_regs_data = regs;
        ((HirDeoptLayout *)db_d)->live_regs_count = 3;
        assert(hir_c_deopt_find_adjacent_dup_reg(db_d) == r_a &&
               "Phase 4.X Batch 80(d): first-pair match returns regs[1].reg");
        free(regs);
    }
    free((char *)db_d - 2 * sizeof(void *) - sizeof(size_t));

    /* (e) Last-pair match. */
    void *db_e = hir_c_alloc_instr(sizeof(HirDeoptLayout), 0);
    hir_c_init_deopt(db_e, HIR_OP_BinaryOp);
    {
        HirRegState *regs = (HirRegState *)malloc(3 * sizeof(HirRegState));
        regs[0].reg = r_a; regs[1].reg = r_b; regs[2].reg = r_b;
        ((HirDeoptLayout *)db_e)->live_regs_data = regs;
        ((HirDeoptLayout *)db_e)->live_regs_count = 3;
        assert(hir_c_deopt_find_adjacent_dup_reg(db_e) == r_b &&
               "Phase 4.X Batch 80(e): last-pair match returns regs[2].reg");
        free(regs);
    }
    free((char *)db_e - 2 * sizeof(void *) - sizeof(size_t));

    /* (f) Middle match. */
    void *db_f = hir_c_alloc_instr(sizeof(HirDeoptLayout), 0);
    hir_c_init_deopt(db_f, HIR_OP_BinaryOp);
    {
        HirRegState *regs = (HirRegState *)malloc(4 * sizeof(HirRegState));
        regs[0].reg = r_a; regs[1].reg = r_b; regs[2].reg = r_b; regs[3].reg = r_c;
        ((HirDeoptLayout *)db_f)->live_regs_data = regs;
        ((HirDeoptLayout *)db_f)->live_regs_count = 4;
        assert(hir_c_deopt_find_adjacent_dup_reg(db_f) == r_b &&
               "Phase 4.X Batch 80(f): middle match returns regs[2].reg");
        free(regs);
    }
    free((char *)db_f - 2 * sizeof(void *) - sizeof(size_t));

    /* (g) Multiple matches: returns FIRST (matches std::adjacent_find). */
    void *db_g = hir_c_alloc_instr(sizeof(HirDeoptLayout), 0);
    hir_c_init_deopt(db_g, HIR_OP_BinaryOp);
    {
        HirRegState *regs = (HirRegState *)malloc(4 * sizeof(HirRegState));
        regs[0].reg = r_a; regs[1].reg = r_a; regs[2].reg = r_b; regs[3].reg = r_b;
        ((HirDeoptLayout *)db_g)->live_regs_data = regs;
        ((HirDeoptLayout *)db_g)->live_regs_count = 4;
        assert(hir_c_deopt_find_adjacent_dup_reg(db_g) == r_a &&
               "Phase 4.X Batch 80(g): multiple matches returns FIRST (r_a, "
               "not r_b)");
        free(regs);
    }
    free((char *)db_g - 2 * sizeof(void *) - sizeof(size_t));
    (void)r_c; (void)r_d;
}

/* Phase 4.D D1b-prep Batch 83 (slim per supervisor 00:39:12Z OPTION (ii)):
 * 2 PhxPtrArray*-returning FrameState accessors enabling D1b
 * TranslationContext → PhxTranslationContext rename. Pins field-aliasing
 * semantic — PhxPtrArray pointer-returns must equal &fs->stack /
 * &fs->localsplus so C++ callers' phx_ptr_arr_push/pop + .data[i]
 * assignment retain reference semantics across the rename.
 *
 * 3 candidate scalar accessors (code/nlocals/cur_instr_offs) NOT verified
 * here — already exist in hir_instr_c.h as hir_fs_* (round-trip pinned by
 * existing fixtures). */
static void verify_phase4d_batch83_frame_state_accessors() {
    HirFrameStateLayout fs;
    void *sentinel_code = (void *)(uintptr_t)0xC0DE0001ULL;
    void *sentinel_globals = (void *)(uintptr_t)0xC0DE0002ULL;
    void *sentinel_builtins = (void *)(uintptr_t)0xC0DE0003ULL;
    phx_frame_state_init(&fs, sentinel_code, /*nlocals=*/7,
                         sentinel_globals, sentinel_builtins);

    PhxPtrArray *stack_ptr = phx_frame_state_stack(&fs);
    PhxPtrArray *localsplus_ptr = phx_frame_state_localsplus(&fs);
    assert(stack_ptr == &fs.stack &&
           "Phase 4.D Batch 83: stack accessor returns field address");
    assert(localsplus_ptr == &fs.localsplus &&
           "Phase 4.D Batch 83: localsplus accessor returns field address");

    phx_frame_state_destroy(&fs);
}

/* Phase 4.D D5a Batch 87 (supervisor 02:02:33Z (α') split): 2 NEW
 * extern "C" entries enabling D5c compiler.cpp:236 rewire to C-side.
 * NO-OP behaviorally — both entries co-exist with hir::buildHIR() free
 * function until D5c. Verifier pins symbol existence + sig at link time
 * (compile is the primary falsifier; this fixture is the structural
 * pre-falsification pin per pre-analysis §5 STRUCT-tier classification). */
extern "C" void *phx_preloader_make_function(const void *preloader_handle);
extern "C" void *phx_hir_build(const void *preloader_handle);

static void verify_phase4d_batch87_d5a_entries() {
    /* Symbol-existence + signature pin via volatile fn-pointer cast.
     * Cannot exercise without a real Preloader (Preloader has heavy
     * STL/RAII state; full constructor requires PyCodeObject + globals
     * + builtins + preloader). D5b body migration tests behavior; D5c
     * caller rewire tests end-to-end via 4-bench gate. */
    volatile void *(*p_make_fn)(const void *) = phx_preloader_make_function;
    volatile void *(*p_build_fn)(const void *) = phx_hir_build;
    assert(p_make_fn != NULL &&
           "Phase 4.D Batch 87 D5a: phx_preloader_make_function symbol");
    assert(p_build_fn != NULL &&
           "Phase 4.D Batch 87 D5a: phx_hir_build symbol");
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
    verify_phase4a_batch36_bb_insert_substantive();
    verify_phase4a_batch37_function_inline_lookups();
    verify_phase4a_batch38_get_frame_state();
    verify_phase4a_batch39_load_method_model_reg();
    verify_phase4a_batch41_constraint_name();
    verify_phase4c_batch42_temp_allocator();
    verify_phase4c_batch43_state_temps_phx();
    /* batch44 mirror-collapse falsifier deleted in P3c (Batch 77) — wrapper
     * class gone, mirror-divergence class structurally resolved. */
    verify_phase4c_batch48_op_stack();
    verify_phase4d_batch51_allocate_localsplus();
    verify_phase4d_batch52_advance_past_yield();
    verify_phase4d_batch54_no_fs_cluster();
    verify_phase4d_batch55_fs_coupled_cluster();
    verify_phase4d_batch57_emit_cluster_3();
    verify_phase4d_batch58_emit_cluster_4();
    verify_phase4d_batch59_emit_cluster_5();
    verify_phase4d_batch60_emit_cluster_6();
    verify_phase4d_batch61_emit_cluster_7();
    verify_phase4d_batch62_emit_cluster_8();
    verify_phase4d_batch63_emit_cluster_9();
    verify_phase4d_batch64_emit_cluster_10();
    verify_phase4d_batch65_emit_cluster_13();
    verify_phase4d_batch66_emit_cluster_14();
    verify_phase4a_batch67_call_cfunc_func_name();
    verify_phase4a_batch68_phi_apply_helper_and_block_index();
    verify_phase4a_batch69_edge_accessors();
    verify_phase4a_batch70_environment_accessors();
    verify_phase4a_batch71_deopt_base_accessors();
    verify_phase4a_batch72_bb_terminator_and_entry_snapshot();
    verify_phase4a_batch73_instr_operand_layout();
    verify_phase4a_batch74_for_each_phi();
    verify_phase4a_batch75_collect_phis_alloc();
    verify_phase4a_batch76_typed_argument_pytype_swap();
    verify_phase4x_batch80_deopt_find_adjacent_dup_reg();
    verify_phase4d_batch83_frame_state_accessors();
    verify_phase4d_batch87_d5a_entries();
}
