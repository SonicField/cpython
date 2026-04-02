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
 * Duplicated from hir_ops.h so this header is self-contained
 * and includable from pure C files without pulling C++ headers.
 * MUST be kept in sync with hir_ops.h. */
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
  V(CheckedDictContains)               \
  V(CheckedDictGet)                    \
  V(CheckedDictGetItem)                \
  V(CheckedDictSetItem)                \
  V(CheckedSetGet)                     \
  V(ClearError)                        \
  V(Compare)                           \
  V(CompareBool)                       \
  V(CondBranch)                        \
  V(CondBranchCheckType)               \
  V(CondBranchIterNotDone)             \
  V(CopyDictWithoutKeys)               \
  V(Decref)                            \
  V(DeleteSubscr)                      \
  V(Deopt)                             \
  V(DeoptPatchpoint)                   \
  V(DictMerge)                         \
  V(DictSubscr)                        \
  V(DictUpdate)                        \
  V(EndInlinedFunction)                \
  V(FillTypeAttrCache)                 \
  V(FillTypeMethodCache)               \
  V(FormatValue)                       \
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
  V(InPlaceOp)                         \
  V(Incref)                            \
  V(InitFunction)                      \
  V(InitListTuple)                     \
  V(InitialYield)                      \
  V(InvokeIterNext)                    \
  V(InvokeMethod)                      \
  V(InvokeMethodStatic)                \
  V(InvokeStaticFunction)              \
  V(IsErrStopAsyncIteration)           \
  V(IsInstance)                        \
  V(IsNegativeAndErrOccurred)          \
  V(IsTruthy)                          \
  V(ListAppend)                        \
  V(ListExtend)                        \
  V(LoadArrayItem)                     \
  V(LoadArg)                           \
  V(LoadAttr)                          \
  V(LoadAttrCached)                    \
  V(LoadAttrSpecial)                   \
  V(LoadAttrSuper)                     \
  V(LoadCellItem)                      \
  V(LoadConst)                         \
  V(LoadCurrentFunc)                   \
  V(LoadEvalBreaker)                   \
  V(LoadField)                         \
  V(LoadFieldAddress)                  \
  V(LoadFunctionIndirect)              \
  V(LoadGlobal)                        \
  V(LoadGlobalCached)                  \
  V(LoadMethod)                        \
  V(LoadMethodCached)                  \
  V(LoadMethodSuper)                   \
  V(LoadModuleMethodCached)            \
  V(LoadSplitDictItem)                 \
  V(LoadTupleItem)                     \
  V(LoadTypeAttrCacheItem)             \
  V(LoadTypeMethodCacheEntryType)      \
  V(LoadTypeMethodCacheEntryValue)     \
  V(LongBinaryOp)                      \
  V(LongCompare)                       \
  V(LongRichcompare)                   \
  V(MakeCell)                          \
  V(MakeCheckedDict)                   \
  V(MakeCheckedList)                   \
  V(MakeDict)                          \
  V(MakeFunction)                      \
  V(MakeList)                          \
  V(MakeListTuple)                     \
  V(MakeSet)                           \
  V(MakeTupleFromList)                 \
  V(MatchClass)                        \
  V(MatchKeys)                         \
  V(MatchMapping)                      \
  V(MatchSequence)                     \
  V(MergeSetUnpack)                    \
  V(Phi)                               \
  V(PrimitiveBox)                      \
  V(PrimitiveCompare)                  \
  V(PrimitiveDecref)                   \
  V(PrimitiveIncref)                   \
  V(PrimitiveUnaryOp)                  \
  V(PrimitiveUnbox)                    \
  V(RaiseAwaitableError)               \
  V(RaiseStatic)                       \
  V(Raise)                             \
  V(RefineType)                        \
  V(RepeatList)                        \
  V(RepeatTuple)                       \
  V(Return)                            \
  V(RunPeriodicTasks)                  \
  V(SetCellItem)                       \
  V(SetCurrentAwaiter)                 \
  V(SetDictItem)                       \
  V(SetFunctionAttr)                   \
  V(SetSetItem)                        \
  V(Snapshot)                          \
  V(StealCellItem)                     \
  V(StoreArrayItem)                    \
  V(StoreAttr)                         \
  V(StoreAttrCached)                   \
  V(StoreField)                        \
  V(StoreGlobal)                       \
  V(StoreSubscr)                       \
  V(UnaryOp)                           \
  V(UnpackExToTuple)                   \
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
