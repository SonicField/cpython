/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C port of outputType() from pass.cpp.
 * Pure function: reads instruction opcode + fields → returns HirType.
 *
 * CONSTANT FAMILIES — always use named constants, never raw integers:
 *   HIR_OP_*   Opcode constants   (hir_instr_c.h)
 *   HIR_BOP_*  BinaryOpKind       (hir_instr_c.h)
 *   HIR_TYPE_* HirType constants  (hir_type_c.h)
 */

#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/hir/hir_type_c.h"
#include "cinderx/Jit/hir/hir_opcode_c.h"
#include "Python.h"

/* Forward declarations */
extern HirType hir_register_type(void *reg);
extern HirType hir_type_intersect(HirType a, HirType b);
extern HirType hir_type_union(HirType a, HirType b);
extern HirType hir_type_subtract(HirType a, HirType b);

typedef HirType (*HirGetOpTypeFn)(size_t idx, void *ctx);

/* returnType C stub — delegates to C++ for now.
 * TODO: port builtinFunctionReturnType table to C. */
extern HirType hir_return_type_c(void *callable_reg);

HirType hir_output_type_c(const void *instr,
                           HirGetOpTypeFn get_op_type, void *ctx) {
    int op = hir_c_opcode(instr);
    switch (op) {
    /* ---- Call opcodes: returnType on func operand ---- */
    case HIR_OP_CallEx:
    case HIR_OP_VectorCall:
    case HIR_OP_CallMethod:
        return hir_return_type_c(hir_c_get_operand(instr, 0));

    /* ---- Compare ---- */
    case HIR_OP_Compare: {
        int32_t cmp_op = hir_c_compare_op(instr);
        if (cmp_op == HIR_CMP_In || cmp_op == HIR_CMP_NotIn)
            { HirType _t = HIR_TYPE_BOOL; return _t; }
        { HirType _t = HIR_TYPE_OBJECT; return _t; }
    }

    /* ---- InPlaceOp ---- */
    case HIR_OP_InPlaceOp: {
        HirType lhs_t = get_op_type(0, ctx);
        HirType rhs_t = get_op_type(1, ctx);
        HirType t_long = HIR_TYPE_LONGEXACT;
        if (hir_type_is_subtype(lhs_t, t_long) && hir_type_is_subtype(rhs_t, t_long)) {
            int32_t iop = hir_c_inplace_op_kind(instr);
            switch (iop) {
            case HIR_IOP_Add: case HIR_IOP_And: case HIR_IOP_FloorDivide:
            case HIR_IOP_LShift: case HIR_IOP_Modulo: case HIR_IOP_Multiply:
            case HIR_IOP_Or: case HIR_IOP_RShift: case HIR_IOP_Subtract:
            case HIR_IOP_Xor:
                { HirType _t = HIR_TYPE_LONGEXACT; return _t; }
            case HIR_IOP_MatrixMultiply:
                { HirType _t = HIR_TYPE_OBJECT; return _t; }
            case HIR_IOP_Power: {
                HirType t_float = HIR_TYPE_FLOATEXACT;
                return hir_type_union(t_long, t_float);
            }
            case HIR_IOP_TrueDivide:
                { HirType _t = HIR_TYPE_FLOATEXACT; return _t; }
            }
        }
        { HirType _t = HIR_TYPE_OBJECT; return _t; }
    }

    /* ---- BinaryOp ---- */
    case HIR_OP_BinaryOp: {
        HirType lhs_t = get_op_type(0, ctx);
        HirType rhs_t = get_op_type(1, ctx);
        HirType t_long = HIR_TYPE_LONGEXACT;
        if (hir_type_is_subtype(lhs_t, t_long) && hir_type_is_subtype(rhs_t, t_long)) {
            int32_t bop = hir_c_binary_op_kind(instr);
            switch (bop) {
            case HIR_BOP_Add: case HIR_BOP_And: case HIR_BOP_FloorDivide:
            case HIR_BOP_FloorDivideUnsigned: case HIR_BOP_LShift:
            case HIR_BOP_Modulo: case HIR_BOP_ModuloUnsigned:
            case HIR_BOP_Multiply: case HIR_BOP_Or: case HIR_BOP_PowerUnsigned:
            case HIR_BOP_RShift: case HIR_BOP_RShiftUnsigned:
            case HIR_BOP_Subtract: case HIR_BOP_Xor:
                { HirType _t = HIR_TYPE_LONGEXACT; return _t; }
            case HIR_BOP_Power: {
                HirType t_float = HIR_TYPE_FLOATEXACT;
                return hir_type_union(t_long, t_float);
            }
            case HIR_BOP_TrueDivide:
                { HirType _t = HIR_TYPE_FLOATEXACT; return _t; }
            case HIR_BOP_Subscript:
            case HIR_BOP_MatrixMultiply:
                { HirType _t = HIR_TYPE_OBJECT; return _t; }
            }
        }
        { HirType _t = HIR_TYPE_OBJECT; return _t; }
    }

    /* ---- Bulk TObject returns ---- */
    case HIR_OP_BuildInterpolation: case HIR_OP_BuildTemplate:
    case HIR_OP_CallIntrinsic: case HIR_OP_ConvertValue:
    case HIR_OP_DictSubscr: case HIR_OP_EagerImportName:
    case HIR_OP_FillTypeAttrCache: case HIR_OP_FillTypeMethodCache:
    case HIR_OP_GetAIter: case HIR_OP_GetANext:
    case HIR_OP_GetIter: case HIR_OP_ImportFrom: case HIR_OP_ImportName:
    case HIR_OP_InvokeIterNext: case HIR_OP_LoadAttr:
    case HIR_OP_LoadAttrCached: case HIR_OP_LoadAttrSpecial:
    case HIR_OP_LoadAttrSuper: case HIR_OP_LoadGlobal:
    case HIR_OP_LoadMethod: case HIR_OP_LoadMethodCached:
    case HIR_OP_LoadMethodSuper: case HIR_OP_LoadModuleAttrCached:
    case HIR_OP_LoadModuleMethodCached: case HIR_OP_LoadSpecial:
    case HIR_OP_LoadTupleItem: case HIR_OP_MatchKeys:
    case HIR_OP_Send: case HIR_OP_WaitHandleLoadCoroOrResult:
    case HIR_OP_YieldAndYieldFrom: case HIR_OP_YieldFrom:
    case HIR_OP_YieldFromHandleStopAsyncIteration: case HIR_OP_YieldValue:
        { HirType _t = HIR_TYPE_OBJECT; return _t; }

    case HIR_OP_BuildString:
        { HirType _t = HIR_TYPE_UNICODEEXACT; return _t; }

    case HIR_OP_GetLength:
        { HirType _t = HIR_TYPE_LONGEXACT; return _t; }

    case HIR_OP_CopyDictWithoutKeys:
        { HirType _t = HIR_TYPE_DICTEXACT; return _t; }

    /* ---- UnaryOp ---- */
    case HIR_OP_UnaryOp: {
        int32_t uop = hir_c_unary_op_kind(instr);
        if (uop == HIR_UOP_Not) { HirType _t = HIR_TYPE_BOOL; return _t; }
        { HirType _t = HIR_TYPE_OBJECT; return _t; }
    }

    /* ---- Bulk TOptObject returns ---- */
    case HIR_OP_CallCFunc: case HIR_OP_LoadCellItem:
    case HIR_OP_LoadGlobalCached: case HIR_OP_MatchClass:
    case HIR_OP_StealCellItem: case HIR_OP_SwapCellItem:
    case HIR_OP_WaitHandleLoadWaiter:
    case HIR_OP_LoadSplitDictItem:
        { HirType _t = HIR_TYPE_OPTOBJECT; return _t; }

    case HIR_OP_GetSecondOutput:
        return hir_c_get_second_output_type(instr);

    case HIR_OP_FormatValue: case HIR_OP_FormatWithSpec:
        { HirType _t = HIR_TYPE_UNICODE; return _t; }

    case HIR_OP_LoadVarObjectSize:
        { HirType _t = HIR_TYPE_CINT64; return _t; }

    case HIR_OP_InvokeStaticFunction:
        return hir_c_invoke_static_ret_type(instr);

    case HIR_OP_LoadArrayItem:
        return hir_c_load_array_item_type(instr);

    case HIR_OP_LoadField:
        return hir_c_load_field_type(instr);

    case HIR_OP_LoadFieldAddress:
        { HirType _t = HIR_TYPE_CPTR; return _t; }

    case HIR_OP_CallStatic:
        return hir_c_call_static_ret_type(instr);

    case HIR_OP_CallInd:
        return ((const HirCallInd *)instr)->ret_type;

    case HIR_OP_IntConvert:
        return ((const HirIntConvert *)instr)->type;

    /* ---- IntBinaryOp ---- */
    case HIR_OP_IntBinaryOp: {
        int32_t ibop = ((const HirIntBinaryOp *)instr)->op;
        if (ibop == HIR_BOP_Power || ibop == HIR_BOP_PowerUnsigned)
            { HirType _t = HIR_TYPE_CDOUBLE; return _t; }
        HirType op0_t = get_op_type(0, ctx);
        return hir_type_unspecialized(&op0_t);
    }

    case HIR_OP_DoubleBinaryOp:
        { HirType _t = HIR_TYPE_CDOUBLE; return _t; }

    case HIR_OP_PrimitiveCompare:
        { HirType _t = HIR_TYPE_CBOOL; return _t; }

    /* ---- PrimitiveUnaryOp ---- */
    case HIR_OP_PrimitiveUnaryOp: {
        int32_t puop = hir_c_primitive_unary_op_kind(instr);
        if (puop == 2 /* kNotInt */) { HirType _t = HIR_TYPE_CBOOL; return _t; }
        HirType op0_t = get_op_type(0, ctx);
        return hir_type_unspecialized(&op0_t);
    }

    case HIR_OP_BuildSlice:
        { HirType _t = HIR_TYPE_OBJECT; return _t; }

    case HIR_OP_GetTuple:
        { HirType _t = HIR_TYPE_TUPLEEXACT; return _t; }

    case HIR_OP_InitialYield:
        { HirType _t = HIR_TYPE_NONETYPE; return _t; }

    case HIR_OP_LoadArg:
        return ((const HirLoadArg *)instr)->type;

    case HIR_OP_LoadCurrentFunc:
        { HirType _t = HIR_TYPE_FUNC; return _t; }

    case HIR_OP_LoadEvalBreaker:
#if PY_VERSION_HEX >= 0x030D0000
        { HirType _t = HIR_TYPE_CINT64; return _t; }
#else
        { HirType _t = HIR_TYPE_CINT32; return _t; }
#endif

    case HIR_OP_MakeCell:
        { HirType _t = HIR_TYPE_OBJECT; return _t; }

    case HIR_OP_MakeDict:
        { HirType _t = HIR_TYPE_DICTEXACT; return _t; }

    case HIR_OP_MakeCheckedDict:
        return ((const HirMakeCheckedDict *)instr)->type;

    case HIR_OP_MakeCheckedList:
        return ((const HirMakeCheckedList *)instr)->type;

    case HIR_OP_MakeFunction:
        { HirType _t = HIR_TYPE_FUNC; return _t; }

    case HIR_OP_MakeSet:
        { HirType _t = HIR_TYPE_SETEXACT; return _t; }

    /* ---- LongBinaryOp ---- */
    case HIR_OP_LongBinaryOp: {
        int32_t lbop = ((const HirLongBinaryOp *)instr)->op;
        if (lbop == HIR_BOP_TrueDivide) { HirType _t = HIR_TYPE_FLOATEXACT; return _t; }
        if (lbop == HIR_BOP_Power) {
            HirType t_long = HIR_TYPE_LONGEXACT;
            HirType t_float = HIR_TYPE_FLOATEXACT;
            return hir_type_union(t_float, t_long);
        }
        { HirType _t = HIR_TYPE_LONGEXACT; return _t; }
    }

    /* ---- LongInPlaceOp ---- */
    case HIR_OP_LongInPlaceOp: {
        int32_t liop = ((const HirLongInPlaceOp *)instr)->op;
        if (liop == HIR_IOP_TrueDivide) { HirType _t = HIR_TYPE_FLOATEXACT; return _t; }
        if (liop == HIR_IOP_Power) {
            HirType t_long = HIR_TYPE_LONGEXACT;
            HirType t_float = HIR_TYPE_FLOATEXACT;
            return hir_type_union(t_float, t_long);
        }
        { HirType _t = HIR_TYPE_LONGEXACT; return _t; }
    }

    case HIR_OP_FloatBinaryOp:
        { HirType _t = HIR_TYPE_FLOATEXACT; return _t; }

    case HIR_OP_FloatCompare: case HIR_OP_LongCompare:
    case HIR_OP_UnicodeCompare:
        { HirType _t = HIR_TYPE_BOOL; return _t; }

    case HIR_OP_DictUpdate: case HIR_OP_DictMerge:
    case HIR_OP_RunPeriodicTasks:
        { HirType _t = HIR_TYPE_CINT32; return _t; }

    case HIR_OP_ListExtend:
        { HirType _t = HIR_TYPE_NONETYPE; return _t; }

    case HIR_OP_ListAppend: case HIR_OP_MergeSetUnpack:
    case HIR_OP_SetSetItem: case HIR_OP_SetUpdate:
    case HIR_OP_SetDictItem:
        { HirType _t = HIR_TYPE_CINT32; return _t; }

    case HIR_OP_IsNegativeAndErrOccurred:
        { HirType _t = HIR_TYPE_CINT64; return _t; }

    /* ---- Check opcodes: input minus TNullptr ---- */
    case HIR_OP_CheckExc: case HIR_OP_CheckField:
    case HIR_OP_CheckFreevar: case HIR_OP_CheckNeg:
    case HIR_OP_CheckVar: {
        HirType op0_t = get_op_type(0, ctx);
        HirType t_nullptr = HIR_TYPE_NULLPTR;
        return hir_type_subtract(op0_t, t_nullptr);
    }

    /* ---- GuardIs: input & fromObject(target) ---- */
    case HIR_OP_GuardIs: {
        PyObject *target = ((const HirGuardIs *)instr)->target;
        HirType target_type = hir_type_from_object(target);
        HirType op0_t = get_op_type(0, ctx);
        return hir_type_intersect(op0_t, target_type);
    }

    /* ---- Cast: fromType(pytype) | optional ---- */
    case HIR_OP_Cast: {
        const HirCast *cast = (const HirCast *)instr;
        HirType to_type = hir_type_from_pytype((PyTypeObject *)cast->pytype,
                                                 cast->exact ? 1 : 0);
        if (cast->optional) {
            HirType t_none = HIR_TYPE_NONETYPE;
            to_type = hir_type_union(to_type, t_none);
        }
        return to_type;
    }

    /* ---- TpAlloc: fromTypeExact(pytype) ---- */
    case HIR_OP_TpAlloc: {
        const HirTpAlloc *tp = (const HirTpAlloc *)instr;
        return hir_type_from_pytype((PyTypeObject *)tp->pytype, 1);
    }

    /* ---- RefineType: input & type ---- */
    case HIR_OP_RefineType: {
        HirType rt = hir_c_refine_type_type(instr);
        HirType op0_t = get_op_type(0, ctx);
        return hir_type_intersect(op0_t, rt);
    }

    /* ---- GuardType: input & target ---- */
    case HIR_OP_GuardType: {
        HirType gt = hir_c_guard_type_target(instr);
        HirType op0_t = get_op_type(0, ctx);
        return hir_type_intersect(op0_t, gt);
    }

    case HIR_OP_UnicodeConcat: case HIR_OP_UnicodeRepeat:
    case HIR_OP_UnicodeSubscr:
        { HirType _t = HIR_TYPE_UNICODEEXACT; return _t; }

    /* ---- Cache entry types ---- */
    case HIR_OP_LoadTypeAttrCacheEntryType:
    case HIR_OP_LoadTypeMethodCacheEntryType:
        { HirType _t = HIR_TYPE_TYPE; return _t; }

    case HIR_OP_LoadTypeAttrCacheEntryValue:
    case HIR_OP_LoadTypeMethodCacheEntryValue:
        { HirType _t = HIR_TYPE_OBJECT; return _t; }

    /* ---- Assign: same as input ---- */
    case HIR_OP_Assign:
        return get_op_type(0, ctx);

    case HIR_OP_BitCast:
        return hir_c_bitcast_type(instr);

    case HIR_OP_LoadConst:
        return hir_c_load_const_type(instr);

    case HIR_OP_MakeList:
        { HirType _t = HIR_TYPE_LISTEXACT; return _t; }

    case HIR_OP_MakeTuple: case HIR_OP_MakeTupleFromList:
    case HIR_OP_UnpackExToTuple:
        { HirType _t = HIR_TYPE_TUPLEEXACT; return _t; }

    /* ---- Phi: union of all inputs ---- */
    case HIR_OP_Phi: {
        HirType ty = HIR_TYPE_BOTTOM;
        size_t n = hir_c_num_operands(instr);
        for (size_t i = 0; i < n; i++) {
            HirType op_t = get_op_type(i, ctx);
            ty = hir_type_union(ty, op_t);
        }
        return ty;
    }

    case HIR_OP_CheckSequenceBounds:
        { HirType _t = HIR_TYPE_CINT64; return _t; }

    case HIR_OP_IsInstance: case HIR_OP_CompareBool:
    case HIR_OP_CIntToCBool: case HIR_OP_IsTruthy:
        { HirType _t = HIR_TYPE_CBOOL; return _t; }

    case HIR_OP_LoadFunctionIndirect:
        { HirType _t = HIR_TYPE_OBJECT; return _t; }

    case HIR_OP_PrimitiveBoxBool:
        { HirType _t = HIR_TYPE_BOOL; return _t; }

    /* ---- PrimitiveBox ---- */
    case HIR_OP_PrimitiveBox: {
        HirType val_t = get_op_type(0, ctx);
        HirType t_cdouble = HIR_TYPE_CDOUBLE;
        if (hir_type_is_subtype(val_t, t_cdouble))
            { HirType _t = HIR_TYPE_FLOATEXACT; return _t; }
        { HirType _t = HIR_TYPE_LONGEXACT; return _t; }
    }

    case HIR_OP_PrimitiveUnbox:
        return ((const HirPrimitiveUnbox *)instr)->type;

    case HIR_OP_IndexUnbox:
        { HirType _t = HIR_TYPE_CINT64; return _t; }

    default:
        { HirType _t = HIR_TYPE_BOTTOM; return _t; }
    }
}
