/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C implementation of HIR instruction memory effects.
 * Replaces instr_effects.cpp.
 */
#include "cinderx/Jit/hir/instr_effects_c.h"
#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/hir/alias_class_c.h"
#include "cinderx/Jit/hir/hir_type_c.h"
#include "cinderx/Jit/jit_log_c.h"
#include "cinderx/Jit/hir/hir_c_api.h"

static HirMemoryEffects common_effects(const void *instr, uint64_t may_store) {
    return (HirMemoryEffects){0, AEmpty, 0, may_store};
}

static HirMemoryEffects borrow_from(const void *instr, uint64_t support) {
    return (HirMemoryEffects){1, support, 0, AEmpty};
}

static uint64_t steal_all_mask(const void *instr) {
    size_t n = hir_c_num_operands(instr);
    if (n >= 64) return ~(uint64_t)0;
    return ((uint64_t)1 << n) - 1;
}

HirMemoryEffects hir_memory_effects(const void *instr) {
    int32_t op = hir_c_opcode(instr);
    switch (op) {
    case HIR_OP_Assign:
    case HIR_OP_BitCast:
    case HIR_OP_BuildSlice:
    case HIR_OP_BuildString:
    case HIR_OP_BuildInterpolation:
    case HIR_OP_BuildTemplate:
    case HIR_OP_Cast:
    case HIR_OP_CIntToCBool:
    case HIR_OP_Deopt:
    case HIR_OP_DeoptPatchpoint:
    case HIR_OP_DoubleBinaryOp:
    case HIR_OP_FloatCompare:
    case HIR_OP_GetSecondOutput:
    case HIR_OP_HintType:
    case HIR_OP_IndexUnbox:
    case HIR_OP_IntBinaryOp:
    case HIR_OP_IntConvert:
    case HIR_OP_IsNegativeAndErrOccurred:
    case HIR_OP_LoadEvalBreaker:
    case HIR_OP_LoadVarObjectSize:
    case HIR_OP_LongCompare:
    case HIR_OP_MakeCell:
    case HIR_OP_MakeCheckedDict:
    case HIR_OP_MakeDict:
    case HIR_OP_MakeSet:
    case HIR_OP_MakeTupleFromList:
    case HIR_OP_PrimitiveCompare:
    case HIR_OP_PrimitiveUnaryOp:
    case HIR_OP_PrimitiveUnbox:
    case HIR_OP_RefineType:
    case HIR_OP_Snapshot:
    case HIR_OP_TpAlloc:
    case HIR_OP_UnicodeCompare:
    case HIR_OP_UnicodeConcat:
    case HIR_OP_UnicodeRepeat:
    case HIR_OP_UnicodeSubscr:
    case HIR_OP_Unreachable:
    case HIR_OP_UseType:
    case HIR_OP_WaitHandleLoadCoroOrResult:
    case HIR_OP_WaitHandleLoadWaiter:
        return common_effects(instr, AEmpty);

    case HIR_OP_PrimitiveBoxBool:
        return borrow_from(instr, AEmpty);

    case HIR_OP_PrimitiveBox:
        return common_effects(instr, AEmpty);

    case HIR_OP_BeginInlinedFunction:
    case HIR_OP_EndInlinedFunction:
    case HIR_OP_UpdatePrevInstr:
    case HIR_OP_SetCurrentAwaiter:
    case HIR_OP_WaitHandleRelease:
    case HIR_OP_LoadFrame:
        return common_effects(instr, AOther);

    case HIR_OP_CheckErrOccurred:
    case HIR_OP_CheckExc:
    case HIR_OP_CheckField:
    case HIR_OP_CheckFreevar:
    case HIR_OP_CheckNeg:
    case HIR_OP_CheckSequenceBounds:
    case HIR_OP_CheckVar:
    case HIR_OP_Guard:
    case HIR_OP_GuardType:
        return common_effects(instr, AEmpty);

    case HIR_OP_BinaryOp:
    case HIR_OP_CallEx:
    case HIR_OP_CallInd:
    case HIR_OP_CallIntrinsic:
    case HIR_OP_CallMethod:
    case HIR_OP_CallStatic:
    case HIR_OP_CallStaticRetVoid:
    case HIR_OP_Compare:
    case HIR_OP_CompareBool:
    case HIR_OP_ConvertValue:
    case HIR_OP_CopyDictWithoutKeys:
    case HIR_OP_DeleteAttr:
    case HIR_OP_DeleteSubscr:
    case HIR_OP_DictMerge:
    case HIR_OP_DictUpdate:
    case HIR_OP_DictSubscr:
    case HIR_OP_EagerImportName:
    case HIR_OP_FillTypeAttrCache:
    case HIR_OP_FillTypeMethodCache:
    case HIR_OP_FloatBinaryOp:
    case HIR_OP_FormatValue:
    case HIR_OP_FormatWithSpec:
    case HIR_OP_GetAIter:
    case HIR_OP_GetANext:
    case HIR_OP_GetIter:
    case HIR_OP_GetLength:
    case HIR_OP_ImportFrom:
    case HIR_OP_ImportName:
    case HIR_OP_InPlaceOp:
    case HIR_OP_InitFrameCellVars:
    case HIR_OP_InvokeIterNext:
    case HIR_OP_InvokeStaticFunction:
    case HIR_OP_IsInstance:
    case HIR_OP_IsTruthy:
    case HIR_OP_LoadAttr:
    case HIR_OP_LoadAttrCached:
    case HIR_OP_LoadAttrSpecial:
    case HIR_OP_LoadAttrSuper:
    case HIR_OP_LoadGlobal:
    case HIR_OP_LoadMethod:
    case HIR_OP_LoadMethodCached:
    case HIR_OP_LoadMethodSuper:
    case HIR_OP_LoadModuleAttrCached:
    case HIR_OP_LoadModuleMethodCached:
    case HIR_OP_LoadSpecial:
    case HIR_OP_LongBinaryOp:
    case HIR_OP_LongInPlaceOp:
    case HIR_OP_MatchClass:
    case HIR_OP_MatchKeys:
    case HIR_OP_Send:
    case HIR_OP_UnaryOp:
    case HIR_OP_UnpackExToTuple:
    case HIR_OP_VectorCall:
        return common_effects(instr, AManagedHeapAny);

    case HIR_OP_SetCellItem:
        return (HirMemoryEffects){1, AEmpty, (uint64_t)1 << 1, ACellItem};

    case HIR_OP_SwapCellItem:
        return (HirMemoryEffects){0, AEmpty, (uint64_t)1 << 1, ACellItem};

    case HIR_OP_StealCellItem:
        return common_effects(instr, AEmpty);

    case HIR_OP_MergeSetUnpack:
    case HIR_OP_RunPeriodicTasks:
    case HIR_OP_SetDictItem:
    case HIR_OP_SetSetItem:
    case HIR_OP_SetUpdate:
    case HIR_OP_StoreAttr:
    case HIR_OP_StoreAttrCached:
    case HIR_OP_StoreSubscr:
        return (HirMemoryEffects){1, AEmpty, 0, AManagedHeapAny};

    case HIR_OP_AtQuiescentState:
        return (HirMemoryEffects){0, AEmpty, 0, AManagedHeapAny};

    case HIR_OP_ListAppend:
    case HIR_OP_ListExtend:
        return (HirMemoryEffects){1, AEmpty, 0, AListItem};

    case HIR_OP_Incref:
    case HIR_OP_XIncref:
        return (HirMemoryEffects){0, AEmpty, 0, AOther};

    case HIR_OP_BatchDecref:
        return (HirMemoryEffects){0, AEmpty, (uint64_t)1 << 0, AManagedHeapAny};

    case HIR_OP_Decref:
    case HIR_OP_XDecref: {
        void *operand = hir_c_get_operand(instr, 0);
        HirType op_hir = hir_register_type(operand);
        if (hir_type_has_known_destructor(&op_hir)) {
            return (HirMemoryEffects){0, AEmpty, 0, AOther};
        } else {
            return (HirMemoryEffects){0, AEmpty, (uint64_t)1 << 0, AManagedHeapAny};
        }
    }

    case HIR_OP_MakeFunction:
        return common_effects(instr, AOther);

    case HIR_OP_MakeCheckedList:
    case HIR_OP_MakeList:
        return (HirMemoryEffects){0, AEmpty, steal_all_mask(instr), AListItem};

    case HIR_OP_MakeTuple:
        return common_effects(instr, ATupleItem);

    case HIR_OP_StoreField:
        return (HirMemoryEffects){0, AEmpty, (uint64_t)1 << 1, AInObjectAttr};

    case HIR_OP_LoadArg:
    case HIR_OP_LoadCurrentFunc:
        return borrow_from(instr, AFuncArgs);

    case HIR_OP_GuardIs:
    case HIR_OP_LoadConst:
        return borrow_from(instr, AEmpty);

    case HIR_OP_LoadCellItem:
        return common_effects(instr, AEmpty);

    case HIR_OP_LoadField: {
        const HirLoadField *lf = (const HirLoadField *)instr;
        if (lf->borrowed) {
            return borrow_from(instr, AInObjectAttr);
        }
        return common_effects(instr, AEmpty);
    }

    case HIR_OP_LoadFieldAddress:
        return common_effects(instr, AEmpty);

    case HIR_OP_LoadFunctionIndirect:
    case HIR_OP_LoadGlobalCached:
        return borrow_from(instr, AGlobal);

    case HIR_OP_LoadTupleItem:
        return borrow_from(instr, ATupleItem);

    case HIR_OP_LoadArrayItem:
        return borrow_from(instr, AArrayItem | AListItem);

    case HIR_OP_StoreArrayItem:
        return (HirMemoryEffects){0, AEmpty, (uint64_t)1 << 2, AArrayItem | AListItem};

    case HIR_OP_LoadSplitDictItem:
        return borrow_from(instr, ADictItem);

    case HIR_OP_LoadTypeAttrCacheEntryType:
    case HIR_OP_LoadTypeAttrCacheEntryValue:
        return borrow_from(instr, ATypeAttrCache);

    case HIR_OP_LoadTypeMethodCacheEntryValue:
        return common_effects(instr, AEmpty);

    case HIR_OP_LoadTypeMethodCacheEntryType:
        return borrow_from(instr, ATypeMethodCache);

    case HIR_OP_Return:
        return (HirMemoryEffects){0, AEmpty, (uint64_t)1 << 0, AManagedHeapAny};

    case HIR_OP_SetFunctionAttr:
        return (HirMemoryEffects){0, AEmpty, (uint64_t)1 << 0, AFuncAttr};

    case HIR_OP_Raise:
        return (HirMemoryEffects){0, AEmpty, steal_all_mask(instr), AEmpty};

    case HIR_OP_RaiseAwaitableError:
    case HIR_OP_RaiseStatic:
        return common_effects(instr, AManagedHeapAny);

    case HIR_OP_InitialYield:
        return (HirMemoryEffects){1, AFuncArgs, 0, AAny};

    case HIR_OP_YieldValue:
        return (HirMemoryEffects){1, AFuncArgs, (uint64_t)1 << 0, AAny};

    case HIR_OP_YieldFrom:
#if PY_VERSION_HEX >= 0x030C0000
        return (HirMemoryEffects){1, AFuncArgs, (uint64_t)1 << 0, AAny};
#else
        /* fallthrough to YieldFromHandleStopAsyncIteration */
#endif
    case HIR_OP_YieldFromHandleStopAsyncIteration:
        return common_effects(instr, AAny);

    case HIR_OP_YieldAndYieldFrom:
        return (HirMemoryEffects){0, AEmpty, (uint64_t)1 << 0, AAny};

    case HIR_OP_CallCFunc:
        return common_effects(instr, AManagedHeapAny);

    case HIR_OP_Branch:
    case HIR_OP_CondBranch:
    case HIR_OP_CondBranchCheckType:
    case HIR_OP_CondBranchIterNotDone:
    case HIR_OP_Phi:
        JIT_ABORT("Opcode %d doesn't have well-defined memory effects", op);

    case HIR_OP_GetTuple:
        return common_effects(instr, AAny);
    }

    JIT_ABORT("Bad opcode %d", op);
}

int hir_has_arbitrary_execution(const void *instr) {
    int32_t op = hir_c_opcode(instr);
    switch (op) {
    case HIR_OP_CheckErrOccurred:
    case HIR_OP_CheckExc:
    case HIR_OP_CheckField:
    case HIR_OP_CheckFreevar:
    case HIR_OP_CheckNeg:
    case HIR_OP_CheckSequenceBounds:
    case HIR_OP_CheckVar:
    case HIR_OP_CIntToCBool:
    case HIR_OP_Deopt:
    case HIR_OP_Guard:
    case HIR_OP_GuardType:
    case HIR_OP_Raise:
    case HIR_OP_RaiseAwaitableError:
    case HIR_OP_RaiseStatic:
    case HIR_OP_Return:
    case HIR_OP_Assign:
    case HIR_OP_AtQuiescentState:
    case HIR_OP_BeginInlinedFunction:
    case HIR_OP_BitCast:
    case HIR_OP_Branch:
    case HIR_OP_BuildSlice:
    case HIR_OP_BuildString:
    case HIR_OP_BuildInterpolation:
    case HIR_OP_BuildTemplate:
    case HIR_OP_Cast:
    case HIR_OP_CondBranch:
    case HIR_OP_CondBranchCheckType:
    case HIR_OP_CondBranchIterNotDone:
    case HIR_OP_DeoptPatchpoint:
    case HIR_OP_DoubleBinaryOp:
    case HIR_OP_EndInlinedFunction:
    case HIR_OP_FloatCompare:
    case HIR_OP_GetSecondOutput:
    case HIR_OP_GuardIs:
    case HIR_OP_HintType:
    case HIR_OP_Incref:
    case HIR_OP_IndexUnbox:
    case HIR_OP_InitFrameCellVars:
    case HIR_OP_IntBinaryOp:
    case HIR_OP_IntConvert:
    case HIR_OP_IsNegativeAndErrOccurred:
    case HIR_OP_ListAppend:
    case HIR_OP_ListExtend:
    case HIR_OP_LoadArg:
    case HIR_OP_LoadArrayItem:
    case HIR_OP_LoadCellItem:
    case HIR_OP_LoadConst:
    case HIR_OP_LoadCurrentFunc:
    case HIR_OP_LoadFrame:
    case HIR_OP_LoadEvalBreaker:
    case HIR_OP_LoadField:
    case HIR_OP_LoadFieldAddress:
    case HIR_OP_LoadFunctionIndirect:
    case HIR_OP_LoadGlobalCached:
    case HIR_OP_LoadSplitDictItem:
    case HIR_OP_LoadTupleItem:
    case HIR_OP_LoadTypeAttrCacheEntryType:
    case HIR_OP_LoadTypeAttrCacheEntryValue:
    case HIR_OP_LoadTypeMethodCacheEntryType:
    case HIR_OP_LoadTypeMethodCacheEntryValue:
    case HIR_OP_LoadVarObjectSize:
    case HIR_OP_LongCompare:
    case HIR_OP_MakeCell:
    case HIR_OP_MakeCheckedDict:
    case HIR_OP_MakeCheckedList:
    case HIR_OP_MakeDict:
    case HIR_OP_MakeList:
    case HIR_OP_MakeSet:
    case HIR_OP_MakeTuple:
    case HIR_OP_MakeTupleFromList:
    case HIR_OP_Phi:
    case HIR_OP_PrimitiveBox:
    case HIR_OP_PrimitiveBoxBool:
    case HIR_OP_PrimitiveCompare:
    case HIR_OP_PrimitiveUnaryOp:
    case HIR_OP_PrimitiveUnbox:
    case HIR_OP_RefineType:
    case HIR_OP_SetCellItem:
    case HIR_OP_SetFunctionAttr:
    case HIR_OP_Snapshot:
    case HIR_OP_StealCellItem:
    case HIR_OP_SwapCellItem:
    case HIR_OP_StoreArrayItem:
    case HIR_OP_StoreField:
    case HIR_OP_TpAlloc:
    case HIR_OP_UnicodeCompare:
    case HIR_OP_UnicodeConcat:
    case HIR_OP_UnicodeRepeat:
    case HIR_OP_UnicodeSubscr:
    case HIR_OP_Unreachable:
    case HIR_OP_UpdatePrevInstr:
    case HIR_OP_UseType:
    case HIR_OP_WaitHandleLoadCoroOrResult:
    case HIR_OP_WaitHandleLoadWaiter:
    case HIR_OP_WaitHandleRelease:
    case HIR_OP_XIncref:
        return 0;

    case HIR_OP_BatchDecref:
    case HIR_OP_BinaryOp:
    case HIR_OP_CallEx:
    case HIR_OP_CallInd:
    case HIR_OP_CallIntrinsic:
    case HIR_OP_CallMethod:
    case HIR_OP_CallStatic:
    case HIR_OP_CallStaticRetVoid:
    case HIR_OP_Compare:
    case HIR_OP_CompareBool:
    case HIR_OP_ConvertValue:
    case HIR_OP_CopyDictWithoutKeys:
    case HIR_OP_Decref:
    case HIR_OP_DeleteAttr:
    case HIR_OP_DeleteSubscr:
    case HIR_OP_DictMerge:
    case HIR_OP_DictSubscr:
    case HIR_OP_DictUpdate:
    case HIR_OP_EagerImportName:
    case HIR_OP_FillTypeAttrCache:
    case HIR_OP_FillTypeMethodCache:
    case HIR_OP_FloatBinaryOp:
    case HIR_OP_FormatValue:
    case HIR_OP_FormatWithSpec:
    case HIR_OP_GetAIter:
    case HIR_OP_GetANext:
    case HIR_OP_GetIter:
    case HIR_OP_GetLength:
    case HIR_OP_GetTuple:
    case HIR_OP_ImportFrom:
    case HIR_OP_ImportName:
    case HIR_OP_InitialYield:
    case HIR_OP_InPlaceOp:
    case HIR_OP_InvokeIterNext:
    case HIR_OP_InvokeStaticFunction:
    case HIR_OP_IsInstance:
    case HIR_OP_IsTruthy:
    case HIR_OP_LoadAttr:
    case HIR_OP_LoadAttrCached:
    case HIR_OP_LoadAttrSpecial:
    case HIR_OP_LoadAttrSuper:
    case HIR_OP_LoadGlobal:
    case HIR_OP_LoadMethod:
    case HIR_OP_LoadMethodCached:
    case HIR_OP_LoadMethodSuper:
    case HIR_OP_LoadModuleAttrCached:
    case HIR_OP_LoadModuleMethodCached:
    case HIR_OP_LoadSpecial:
    case HIR_OP_LongBinaryOp:
    case HIR_OP_LongInPlaceOp:
    case HIR_OP_MakeFunction:
    case HIR_OP_MergeSetUnpack:
    case HIR_OP_MatchClass:
    case HIR_OP_MatchKeys:
    case HIR_OP_RunPeriodicTasks:
    case HIR_OP_Send:
    case HIR_OP_SetCurrentAwaiter:
    case HIR_OP_SetDictItem:
    case HIR_OP_SetSetItem:
    case HIR_OP_SetUpdate:
    case HIR_OP_StoreAttr:
    case HIR_OP_StoreAttrCached:
    case HIR_OP_StoreSubscr:
    case HIR_OP_UnaryOp:
    case HIR_OP_UnpackExToTuple:
    case HIR_OP_VectorCall:
    case HIR_OP_XDecref:
    case HIR_OP_YieldAndYieldFrom:
    case HIR_OP_YieldFrom:
    case HIR_OP_YieldFromHandleStopAsyncIteration:
    case HIR_OP_YieldValue:
        return 1;

    case HIR_OP_CallCFunc: {
        const HirCallCFunc *cc = (const HirCallCFunc *)instr;
        switch (cc->func) {
#if PY_VERSION_HEX >= 0x030C0000
        case 0: /* kJitCoro_GetAwaitableIter */
            return 1;
        case 1: /* kCix_PyAsyncGenValueWrapperNew */
            return 0;
        case 2: /* kJitGen_yf */
            return 0;
#else
        case 0: /* kCix_PyCoro_GetAwaitableIter */
            return 1;
        case 1: /* kCix_PyAsyncGenValueWrapperNew */
            return 0;
        case 2: /* kCix_PyGen_yf */
            return 0;
#endif
        }
        JIT_ABORT("Bad CallCFunc function %d", cc->func);
    }
    }

    JIT_ABORT("Bad opcode %d", op);
}
