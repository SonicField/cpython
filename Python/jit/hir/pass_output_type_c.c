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
#include "cinderx/Jit/hir/hir_basic_block_c.h"

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
        if (puop == HIR_PUOP_NotInt) { HirType _t = HIR_TYPE_CBOOL; return _t; }
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

/* ==== reflowTypes C port ==== */

static HirType get_op_type_from_instr(size_t idx, void *ctx) {
    const void *instr = ctx;
    void *operand = hir_c_get_operand(instr, idx);
    return hir_register_type(operand);
}

extern size_t hir_cfg_get_rpo_from(void *func, void *start, void **out, size_t capacity);

void hir_reflow_types_c(void *func, void *start_block) {
    void *env = hir_func_env(func);
    size_t n_regs = hir_env_reg_count(env);
    void **regs = hir_env_reg_data(env);

    /* Reset all register types to TBottom */
    HirType t_bottom = HIR_TYPE_BOTTOM;
    for (size_t i = 0; i < n_regs; i++) {
        if (regs[i]) {
            hir_reg_set_type(regs[i], t_bottom);
        }
    }

    /* Get RPO traversal from start_block */
    void *rpo_blocks[4096];
    size_t n_blocks = hir_cfg_get_rpo_from(func, start_block, rpo_blocks, 4096);

    /* Fixed-point iteration: flow types forward */
    int changed = 1;
    while (changed) {
        changed = 0;
        for (size_t b = 0; b < n_blocks; b++) {
            void *block = rpo_blocks[b];
            void *instr = hir_bb_first_instr(block);
            while (instr) {
                void *dst = hir_c_output(instr);
                if (dst != NULL) {
                    HirType new_ty = hir_output_type_c(instr,
                        get_op_type_from_instr, (void *)instr);
                    HirType old_ty = hir_register_type(dst);
                    if (memcmp(&new_ty, &old_ty, sizeof(HirType)) != 0) {
                        hir_reg_set_type(dst, new_ty);
                        changed = 1;
                    }
                }
                instr = hir_bb_next_instr(block, instr);
            }
        }
    }
}

/* ==== simplifyRedundantCondBranches C port ==== */

extern size_t hir_c_num_edges(const void *instr);
extern void *hir_c_successor(const void *instr, size_t idx);
extern void *hir_cfg_blocks_first_ptr(void *cfg);
extern void *hir_cfg_blocks_next_ptr(void *cfg, void *block);
extern void hir_instr_unlink(void *instr);
extern void hir_instr_destroy(void *instr);
extern void *hir_c_create_branch_cpp(void *target_block);
extern void hir_c_set_bytecode_offset(void *instr, int32_t off);

void hir_simplify_redundant_cond_branches_c(void *cfg) {
    /* Collect blocks with redundant cond branches (both edges to same target) */
    void *blocks_to_fix[1024];
    size_t n_fix = 0;

    void *block = hir_cfg_blocks_first_ptr(cfg);
    while (block) {
        if (!hir_bb_empty(block)) {
            void *term = hir_bb_get_terminator(block);
            size_t n_edges = hir_c_num_edges(term);
            if (n_edges >= 2 && hir_c_successor(term, 0) == hir_c_successor(term, 1)) {
                int op = hir_c_opcode(term);
                if (op == HIR_OP_CondBranch ||
                    op == HIR_OP_CondBranchIterNotDone ||
                    op == HIR_OP_CondBranchCheckType) {
                    if (n_fix < 1024) {
                        blocks_to_fix[n_fix++] = block;
                    }
                }
            }
        }
        block = hir_cfg_blocks_next_ptr(cfg, block);
    }

    /* Replace redundant cond branches with unconditional branches */
    for (size_t i = 0; i < n_fix; i++) {
        block = blocks_to_fix[i];
        void *term = hir_bb_get_terminator(block);
        int32_t bc_off = ((const HirInstrLayout *)term)->bytecode_offset;
        void *target = hir_c_successor(term, 0);

        hir_instr_unlink(term);

        void *br = hir_c_create_branch_cpp(target);
        hir_c_set_bytecode_offset(br, bc_off);
        hir_bb_append_instr(block, br);

        hir_instr_destroy(term);
    }
}

/* ==== chaseAssignOperand C port ==== */
void *hir_chase_assign_operand(void *value) {
    extern void *hir_reg_instr(void *reg);
    while (1) {
        void *def = hir_reg_instr(value);
        if (def == NULL || hir_c_opcode(def) != HIR_OP_Assign)
            break;
        value = hir_c_get_operand(def, 0);
    }
    return value;
}

/* ==== removeTrampolineBlocks C port ==== */
extern void hir_bb_set_successor_null(void *block, size_t idx);
extern void hir_bb_destroy(void *block);

int hir_remove_trampoline_blocks_c(void *cfg) {
    void *trampolines[1024];
    size_t n_tramp = 0;

    HirCFG *c = (HirCFG *)cfg;
    void *block = hir_cfg_blocks_first_ptr(cfg);
    while (block) {
        void *next = hir_cfg_blocks_next_ptr(cfg, block);
        if (hir_bb_is_trampoline((HirBasicBlock *)block)) {
            void *term = hir_bb_get_terminator((HirBasicBlock *)block);
            void *succ = hir_c_successor(term, 0);

            if (block == c->entry_block) {
                if (hir_bb_in_edges_count((HirBasicBlock *)succ) > 1) {
                    block = next;
                    continue;
                }
                c->entry_block = succ;
            }

            hir_bb_retarget_preds((HirBasicBlock *)block, (HirBasicBlock *)succ);
            hir_bb_set_successor_null(block, 0);

            if (n_tramp < 1024) {
                trampolines[n_tramp++] = block;
            }
        }
        block = next;
    }

    for (size_t i = 0; i < n_tramp; i++) {
        hir_cfg_remove_block(c, (HirBasicBlock *)trampolines[i]);
        hir_bb_destroy(trampolines[i]);
    }

    hir_simplify_redundant_cond_branches_c(cfg);
    return n_tramp > 0 ? 1 : 0;
}

/* ==== removeUnreachableBlocks C port ==== */
extern void hir_bb_remove_phi_predecessor(void *block, void *pred);

int hir_remove_unreachable_blocks_c(void *func) {
    HirCFG *cfg = hir_func_cfg_ptr(func);

    /* DFS to find reachable blocks */
    void *visited[4096];
    size_t n_visited = 0;
    void *stack[4096];
    size_t stack_top = 0;

    stack[stack_top++] = cfg->entry_block;
    while (stack_top > 0) {
        void *block = stack[--stack_top];
        /* Check if already visited */
        int found = 0;
        for (size_t i = 0; i < n_visited; i++) {
            if (visited[i] == block) { found = 1; break; }
        }
        if (found) continue;
        if (n_visited < 4096) visited[n_visited++] = block;

        void *term = hir_bb_get_terminator((HirBasicBlock *)block);
        size_t n_edges = hir_c_num_edges(term);
        for (size_t i = 0; i < n_edges; i++) {
            void *succ = hir_c_successor(term, i);
            int succ_visited = 0;
            for (size_t j = 0; j < n_visited; j++) {
                if (visited[j] == succ) { succ_visited = 1; break; }
            }
            if (!succ_visited && stack_top < 4096) {
                stack[stack_top++] = succ;
            }
        }
    }

    /* Collect unreachable blocks */
    void *unreachable[1024];
    size_t n_unreach = 0;

    void *block = hir_cfg_blocks_first_ptr(cfg);
    while (block) {
        void *next = hir_cfg_blocks_next_ptr(cfg, block);
        int reachable = 0;
        for (size_t i = 0; i < n_visited; i++) {
            if (visited[i] == block) { reachable = 1; break; }
        }
        if (!reachable) {
            void *term = hir_bb_get_terminator((HirBasicBlock *)block);
            if (term) {
                size_t n_edges = hir_c_num_edges(term);
                for (size_t i = 0; i < n_edges; i++) {
                    hir_bb_remove_phi_predecessor(hir_c_successor(term, i), block);
                }
            }
            hir_cfg_remove_block(cfg, (HirBasicBlock *)block);
            hir_bb_clear((HirBasicBlock *)block);
            if (n_unreach < 1024) unreachable[n_unreach++] = block;
        }
        block = next;
    }

    for (size_t i = 0; i < n_unreach; i++) {
        hir_bb_destroy(unreachable[i]);
    }

    return n_unreach > 0 ? 1 : 0;
}

/* ==== splitCriticalEdges C port ==== */
extern void hir_bb_fixup_phis(void *block, void *old_pred, void *new_pred);
extern void *hir_c_set_successor_cpp(void *instr, size_t idx, void *block);
extern void *hir_cfg_alloc_block(void *func);

void hir_cfg_split_critical_edges_c(void *func) {
    void *cfg = hir_func_cfg_ptr(func);

    /* Collect critical edges: (block, edge_idx) pairs */
    struct CritEdge { void *block; size_t edge_idx; };
    struct CritEdge edges[4096];
    size_t n_edges = 0;

    void *block = hir_cfg_blocks_first_ptr(cfg);
    while (block) {
        void *term = hir_bb_get_terminator((HirBasicBlock *)block);
        size_t n = hir_c_num_edges(term);
        if (n >= 2) {
            for (size_t i = 0; i < n; i++) {
                void *succ = hir_c_successor(term, i);
                if (hir_bb_in_edges_count((HirBasicBlock *)succ) > 1) {
                    if (n_edges < 4096) {
                        edges[n_edges].block = block;
                        edges[n_edges].edge_idx = i;
                        n_edges++;
                    }
                }
            }
        }
        block = hir_cfg_blocks_next_ptr(cfg, block);
    }

    /* Split each critical edge */
    for (size_t i = 0; i < n_edges; i++) {
        void *from = edges[i].block;
        void *term = hir_bb_get_terminator((HirBasicBlock *)from);
        void *to = hir_c_successor(term, edges[i].edge_idx);

        void *split_bb = hir_cfg_alloc_block(func);

        /* Append Branch(to) to split_bb */
        int32_t bc_off = ((const HirInstrLayout *)term)->bytecode_offset;
        void *br = hir_c_create_branch_cpp(to);
        hir_c_set_bytecode_offset(br, bc_off);
        hir_bb_append_instr((HirBasicBlock *)split_bb, br);

        /* Redirect edge from→split_bb */
        hir_c_set_successor_cpp(term, edges[i].edge_idx, split_bb);

        /* Fix up phis in 'to' block */
        hir_bb_fixup_phis(to, from, split_bb);
    }
}

/* ==== getBlockById C port ==== */
void *hir_cfg_get_block_by_id(void *cfg, int id) {
    void *block = hir_cfg_blocks_first_ptr(cfg);
    while (block) {
        if (((HirBasicBlock *)block)->id == id) return block;
        block = hir_cfg_blocks_next_ptr(cfg, block);
    }
    return NULL;
}
