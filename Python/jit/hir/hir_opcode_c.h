/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C-compatible HIR opcode enum — Phase B replacement for
 * the C++ enum class Opcode in hir_ops.h.
 *
 * Uses the same FOREACH_OPCODE macro to generate identical
 * enum values, ensuring C and C++ opcode values match.
 */
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- FOREACH_OPCODE macro ----
 * Exact copy of hir_ops.h so this header is self-contained
 * and includable from pure C files without pulling C++ headers.
 * MUST be kept in sync with hir_ops.h — verified by startup
 * name check in hir_instr_info_verify.cpp. */
#ifndef FOREACH_OPCODE
#define FOREACH_OPCODE(V)              \
  V(Assign)                            \
  V(BatchDecref)                       \
  V(BeginInlinedFunction)              \
  V(BinaryOp)                          \
  V(BitCast)                           \
  V(Branch)                            \
  V(BuildSlice)                        \
  V(BuildString)                       \
  V(BuildInterpolation)                \
  V(BuildTemplate)                     \
  V(CallCFunc)                         \
  V(CallEx)                            \
  V(CallIntrinsic)                     \
  V(CallInd)                           \
  V(CallMethod)                        \
  V(CallStatic)                        \
  V(CallStaticRetVoid)                 \
  V(Cast)                              \
  V(CheckSequenceBounds)               \
  V(CheckErrOccurred)                  \
  V(CheckExc)                          \
  V(CheckNeg)                          \
  V(CheckVar)                          \
  V(CheckFreevar)                      \
  V(CheckField)                        \
  V(CIntToCBool)                       \
  V(Compare)                           \
  V(CompareBool)                       \
  V(ConvertValue)                      \
  V(CopyDictWithoutKeys)               \
  V(CondBranch)                        \
  V(CondBranchIterNotDone)             \
  V(CondBranchCheckType)               \
  V(Decref)                            \
  V(DeleteAttr)                        \
  V(DeleteSubscr)                      \
  V(Deopt)                             \
  V(DeoptPatchpoint)                   \
  V(DictMerge)                         \
  V(DictSubscr)                        \
  V(DictUpdate)                        \
  V(DoubleBinaryOp)                    \
  V(EagerImportName)                   \
  V(EndInlinedFunction)                \
  V(FillTypeAttrCache)                 \
  V(FillTypeMethodCache)               \
  V(FloatBinaryOp)                     \
  V(FloatCompare)                      \
  V(FormatValue)                       \
  V(FormatWithSpec)                    \
  V(GetAIter)                          \
  V(GetANext)                          \
  V(GetIter)                           \
  V(GetLength)                         \
  V(GetSecondOutput)                   \
  V(GetTuple)                          \
  V(Guard)                             \
  V(GuardIs)                           \
  V(GuardType)                         \
  V(HintType)                          \
  V(ImportFrom)                        \
  V(ImportName)                        \
  V(InitFrameCellVars)                 \
  V(InPlaceOp)                         \
  V(Incref)                            \
  V(IndexUnbox)                        \
  V(InitialYield)                      \
  V(IntBinaryOp)                       \
  V(PrimitiveBoxBool)                  \
  V(PrimitiveBox)                      \
  V(PrimitiveCompare)                  \
  V(IntConvert)                        \
  V(PrimitiveUnaryOp)                  \
  V(PrimitiveUnbox)                    \
  V(InvokeIterNext)                    \
  V(IsInstance)                        \
  V(InvokeStaticFunction)              \
  V(IsNegativeAndErrOccurred)          \
  V(IsTruthy)                          \
  V(ListAppend)                        \
  V(ListExtend)                        \
  V(LoadArrayItem)                     \
  V(LoadFieldAddress)                  \
  V(LoadArg)                           \
  V(LoadAttr)                          \
  V(LoadAttrCached)                    \
  V(LoadAttrSpecial)                   \
  V(LoadAttrSuper)                     \
  V(LoadCellItem)                      \
  V(LoadConst)                         \
  V(LoadCurrentFunc)                   \
  V(LoadFrame)                         \
  V(LoadEvalBreaker)                   \
  V(AtQuiescentState)                  \
  V(LoadField)                         \
  V(LoadFunctionIndirect)              \
  V(LoadGlobalCached)                  \
  V(LoadGlobal)                        \
  V(LoadMethod)                        \
  V(LoadMethodCached)                  \
  V(LoadModuleAttrCached)              \
  V(LoadModuleMethodCached)            \
  V(LoadMethodSuper)                   \
  V(LoadSpecial)                       \
  V(LoadSplitDictItem)                 \
  V(LoadTupleItem)                     \
  V(LoadTypeAttrCacheEntryType)        \
  V(LoadTypeAttrCacheEntryValue)       \
  V(LoadTypeMethodCacheEntryType)      \
  V(LoadTypeMethodCacheEntryValue)     \
  V(LoadVarObjectSize)                 \
  V(LongCompare)                       \
  V(LongBinaryOp)                      \
  V(LongInPlaceOp)                     \
  V(MakeCheckedDict)                   \
  V(MakeCheckedList)                   \
  V(MakeCell)                          \
  V(MakeDict)                          \
  V(MakeFunction)                      \
  V(MakeList)                          \
  V(MakeTuple)                         \
  V(MakeSet)                           \
  V(MakeTupleFromList)                 \
  V(MatchClass)                        \
  V(MatchKeys)                         \
  V(MergeSetUnpack)                    \
  V(Phi)                               \
  V(Raise)                             \
  V(RaiseStatic)                       \
  V(RaiseAwaitableError)               \
  V(RefineType)                        \
  V(Return)                            \
  V(RunPeriodicTasks)                  \
  V(Send)                              \
  V(SetCellItem)                       \
  V(SetCurrentAwaiter)                 \
  V(SetDictItem)                       \
  V(SetFunctionAttr)                   \
  V(SetSetItem)                        \
  V(SetUpdate)                         \
  V(Snapshot)                          \
  V(StealCellItem)                     \
  V(SwapCellItem)                      \
  V(StoreArrayItem)                    \
  V(StoreAttr)                         \
  V(StoreAttrCached)                   \
  V(StoreField)                        \
  V(StoreSubscr)                       \
  V(TpAlloc)                           \
  V(UnaryOp)                           \
  V(UnicodeCompare)                    \
  V(UnicodeConcat)                     \
  V(UnicodeRepeat)                     \
  V(UnicodeSubscr)                     \
  V(UnpackExToTuple)                   \
  V(Unreachable)                       \
  V(UpdatePrevInstr)                   \
  V(UseType)                           \
  V(VectorCall)                        \
  V(WaitHandleLoadCoroOrResult)        \
  V(WaitHandleLoadWaiter)              \
  V(WaitHandleRelease)                 \
  V(XDecref)                           \
  V(XIncref)                           \
  V(YieldAndYieldFrom)                 \
  V(YieldFrom)                         \
  V(YieldFromHandleStopAsyncIteration) \
  V(YieldValue)
#endif /* FOREACH_OPCODE */

/* ---- C enum ---- */

typedef enum {
#define HIR_DECLARE_OP(opname) HIR_OP_##opname,
    FOREACH_OPCODE(HIR_DECLARE_OP)
#undef HIR_DECLARE_OP
    HIR_OP_COUNT
} HirOpcode;

/* ---- Opcode name lookup ---- */

/* Get the string name of an opcode. Returns a static string. */
const char* hir_opcode_name(HirOpcode op);

/* Parse an opcode name. Returns HIR_OP_COUNT on failure (sentinel). */
HirOpcode hir_opcode_from_name(const char *name);

#ifdef __cplusplus
} /* extern "C" */
#endif
