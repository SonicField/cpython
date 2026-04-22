/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C emit methods for HIRBuilder — incremental port of builder.cpp.
 * Phase 3D: hardest-first per Alex directive 2026-04-21.
 *
 * POC: emitLoadConst validates FrameState C lifecycle end-to-end.
 */

#include "cinderx/Jit/hir/hir_c_api.h"
#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/hir/hir_basic_block_c.h"
#include "cinderx/Jit/hir/phx_frame_state.h"
#include "cinderx/Jit/hir/hir_type_c.h"
#include "cinderx/Jit/hir/annotation_index_c.h"  /* HirAnnotationIndex (emitTypeAnnotationGuards) */
#include "cinderx/Jit/jit_config_c.h"
#include "cinderx/Common/jit_log_c.h"            /* JIT_CHECK_C (item #17) */
#include "Python.h"
#include "internal/pycore_moduleobject.h"  /* PyModuleObject (LOAD_ATTR_MODULE) */
#include "internal/pycore_dict.h"          /* PyDictKeysObject (LOAD_ATTR_MODULE) */
#include "opcode.h"
#include "cinderx/Common/opcode_stubs.h"  /* YIELD_FROM stub for 3.12 (not in Include/opcode.h) */

/* PhxCallKind values — must match enum PhxCallKind in builder.h. C body
 * dispatches on these instead of opcode constants to avoid pulling in
 * cinder_opcode.h which would shadow Include/opcode.h's BINARY_OP_ADD_INT
 * define + break #ifdef in BINARY_OP specialization (W26 first-attempt
 * regression root cause). C++ stub maps opcode → kind. */
#define PHX_CALL_KIND_VECTOR_CALL     0
#define PHX_CALL_KIND_CALL_EX         1
#define PHX_CALL_KIND_CALL_METHOD     2
#define PHX_CALL_KIND_INVOKE_FUNCTION 3
#define PHX_CALL_KIND_INVOKE_NATIVE   4
#define PHX_CALL_KIND_INVOKE_METHOD   5

/* ---- PhxTranslationContext ---- */

typedef struct {
    void *block;               /* BasicBlock* — current block */
    HirFrameStateLayout frame; /* FrameState (owned value) */
} PhxTranslationContext;

static void phx_tc_init(PhxTranslationContext *tc, void *block,
                         const HirFrameStateLayout *frame) {
    tc->block = block;
    phx_frame_state_copy(&tc->frame, frame);
}

static void phx_tc_destroy(PhxTranslationContext *tc) {
    phx_frame_state_destroy(&tc->frame);
}

/* Emit a pre-created instruction into the current block.
 * Sets bytecode offset from frame's cur_instr_offs. */
static void *phx_tc_emit(PhxTranslationContext *tc, void *instr) {
    hir_c_set_bytecode_offset(instr, (int32_t)tc->frame.cur_instr_offs);
    hir_bb_append_instr(tc->block, instr);
    return instr;
}

/* ---- POC: emitLoadConst ---- */

extern HirType hir_type_from_object(PyObject *obj);
/* hir_assign_create canonical decl in hir_c_api.h. */

/* Move stack registers about to be overwritten by dst. */
static void phx_move_overwritten_stack_regs(
        PhxTranslationContext *tc, void *func, void *dst) {
    void *tmp = NULL;
    PhxPtrArray *stack = &tc->frame.stack;
    for (size_t i = 0; i < stack->count; i++) {
        if (stack->data[i] == dst) {
            if (tmp == NULL) {
                tmp = hir_func_alloc_register(func);
                void *assign = hir_assign_create(tmp, dst);
                phx_tc_emit(tc, assign);
            }
            stack->data[i] = tmp;
        }
    }
}

void hir_builder_emit_load_const_c(
        PhxTranslationContext *tc,
        void *func,
        PyCodeObject *code,
        int oparg) {
    void *reg = hir_func_alloc_register(func);

    PyObject *const_val = PyTuple_GET_ITEM(code->co_consts, oparg);
    HirType type = hir_type_from_object(const_val);
    void *instr = hir_c_create_load_const(reg, type);
    phx_tc_emit(tc, instr);

    phx_ptr_arr_push(&tc->frame.stack, reg);
}

/* POC #2: emitStoreFast — exercises stack pop + localsplus read + assign */
void hir_builder_emit_store_fast_c(
        PhxTranslationContext *tc,
        void *func,
        int oparg) {
    void *src = phx_ptr_arr_pop(&tc->frame.stack);
    void *dst = tc->frame.localsplus.data[oparg];
    phx_move_overwritten_stack_regs(tc, func, dst);
    void *assign = hir_assign_create(dst, src);
    phx_tc_emit(tc, assign);
}

/* emitLoadFast — localsplus read + optional CheckVar + stack push */

static PyObject *get_varname(PyCodeObject *code, int idx) {
    return PyTuple_GET_ITEM(code->co_localsplusnames, idx);
}

void hir_builder_emit_load_fast_c(
        PhxTranslationContext *tc,
        void *func,
        PyCodeObject *code,
        int opcode,
        int oparg) {
    void *var = tc->frame.localsplus.data[oparg];
    if (opcode == LOAD_FAST_CHECK) {
        void *chk = hir_c_create_check_var_reg(var, var, get_varname(code, oparg), &tc->frame);
        phx_tc_emit(tc, chk);
    }
    phx_ptr_arr_push(&tc->frame.stack, var);
    if (opcode == LOAD_FAST_AND_CLEAR) {
        phx_move_overwritten_stack_regs(tc, func, var);
        HirType t_nullptr = HIR_TYPE_NULLPTR;
        void *lc = hir_c_create_load_const(var, t_nullptr);
        phx_tc_emit(tc, lc);
    }
}

/* emitLoadFastLoadFast — 2 localsplus reads + 2 stack pushes */
void hir_builder_emit_load_fast_load_fast_c(
        PhxTranslationContext *tc,
        int oparg) {
    int var_idx1 = oparg >> 4;
    int var_idx2 = oparg & 0xf;
    void *var1 = tc->frame.localsplus.data[var_idx1];
    phx_ptr_arr_push(&tc->frame.stack, var1);
    void *var2 = tc->frame.localsplus.data[var_idx2];
    phx_ptr_arr_push(&tc->frame.stack, var2);
}

/* emitStoreFastStoreFast — 2 stack pops + 2 localsplus writes */
void hir_builder_emit_store_fast_store_fast_c(
        PhxTranslationContext *tc,
        void *func,
        int oparg) {
    int var_idx1 = oparg >> 4;
    int var_idx2 = oparg & 0xf;
    void *src1 = phx_ptr_arr_pop(&tc->frame.stack);
    void *dst1 = tc->frame.localsplus.data[var_idx1];
    phx_move_overwritten_stack_regs(tc, func, dst1);
    phx_tc_emit(tc, hir_assign_create(dst1, src1));

    void *src2 = phx_ptr_arr_pop(&tc->frame.stack);
    void *dst2 = tc->frame.localsplus.data[var_idx2];
    phx_move_overwritten_stack_regs(tc, func, dst2);
    phx_tc_emit(tc, hir_assign_create(dst2, src2));
}

/* emitStoreFastLoadFast — 1 store + 1 load */
void hir_builder_emit_store_fast_load_fast_c(
        PhxTranslationContext *tc,
        void *func,
        int oparg) {
    int store_idx = oparg >> 4;
    int load_idx = oparg & 0xf;
    void *src = phx_ptr_arr_pop(&tc->frame.stack);
    void *dst = tc->frame.localsplus.data[store_idx];
    phx_move_overwritten_stack_regs(tc, func, dst);
    phx_tc_emit(tc, hir_assign_create(dst, src));

    void *var = tc->frame.localsplus.data[load_idx];
    phx_ptr_arr_push(&tc->frame.stack, var);
}

/* emitCopy — duplicate stack item at depth */
void hir_builder_emit_copy_c(PhxTranslationContext *tc, int item_idx) {
    void *item = tc->frame.stack.data[tc->frame.stack.count - item_idx];
    phx_ptr_arr_push(&tc->frame.stack, item);
}

/* emitSwap — swap top with item at depth */
void hir_builder_emit_swap_c(PhxTranslationContext *tc, int item_idx) {
    void *item = tc->frame.stack.data[tc->frame.stack.count - item_idx];
    void *top = tc->frame.stack.data[tc->frame.stack.count - 1];
    tc->frame.stack.data[tc->frame.stack.count - 1] = item;
    tc->frame.stack.data[tc->frame.stack.count - item_idx] = top;
}

/* emitPushNull — allocate reg, load nullptr, push to stack */
void hir_builder_emit_push_null_c(
        PhxTranslationContext *tc,
        void *func) {
    void *reg = hir_func_alloc_register(func);
    HirType t_nullptr = HIR_TYPE_NULLPTR;
    void *instr = hir_c_create_load_const(reg, t_nullptr);
    phx_tc_emit(tc, instr);
    phx_ptr_arr_push(&tc->frame.stack, reg);
}

/* emitEndAsyncFor — pop block_stack + pop stack (block-transition validation) */
void hir_builder_emit_end_async_for_c(PhxTranslationContext *tc) {
    phx_block_stack_pop(&tc->frame);
    phx_ptr_arr_pop(&tc->frame.stack);
}

/* emitGetLen — exercises phx_frame_state_copy (FrameState deep copy validation) */

void hir_builder_emit_get_len_c(PhxTranslationContext *tc, void *func) {
    HirFrameStateLayout state_copy;
    phx_frame_state_copy(&state_copy, &tc->frame);

    void *obj = tc->frame.stack.data[tc->frame.stack.count - 1];
    void *result = hir_func_alloc_register(func);
    void *instr = hir_c_create_get_length_reg(result, obj, &state_copy);
    phx_tc_emit(tc, instr);
    phx_ptr_arr_push(&tc->frame.stack, result);

    phx_frame_state_destroy(&state_copy);
}

/* emitIsOp — 2 pops, PrimitiveCompare + PrimitiveBoxBool, 1 push */

void hir_builder_emit_is_op_c(PhxTranslationContext *tc, void *func, int oparg) {
    void *right = phx_ptr_arr_pop(&tc->frame.stack);
    void *left = phx_ptr_arr_pop(&tc->frame.stack);
    void *unboxed = hir_func_alloc_register(func);
    void *result = hir_func_alloc_register(func);
    int32_t op = (oparg == 0) ? 2 : 3; /* HIR_PCMP_Equal=2, HIR_PCMP_NotEqual=3 */
    phx_tc_emit(tc, hir_c_create_primitive_compare(unboxed, op, left, right));
    phx_tc_emit(tc, hir_c_create_primitive_box_bool(result, unboxed));
    phx_ptr_arr_push(&tc->frame.stack, result);
}

/* emitContainsOp — 2 pops, Compare, 1 push */

/* emitDeleteAttr — pop receiver, emit DeleteAttr */

void hir_builder_emit_delete_attr_c(PhxTranslationContext *tc, int oparg) {
    void *receiver = phx_ptr_arr_pop(&tc->frame.stack);
    phx_tc_emit(tc, hir_c_create_delete_attr_reg(receiver, oparg, &tc->frame));
}

/* emitUnaryOp — pop, map opcode→kind, emit UnaryOp, push */

static int32_t get_unary_op_kind_c(int opcode) {
    switch (opcode) {
        case UNARY_NOT: return 0;       /* kNot */
        case UNARY_NEGATIVE: return 1;  /* kNegate */
        case UNARY_INVERT: return 3;    /* kInvert */
        default: return -1;
    }
}

void hir_builder_emit_unary_op_c(PhxTranslationContext *tc, void *func, int opcode) {
    void *operand = phx_ptr_arr_pop(&tc->frame.stack);
    void *result = hir_func_alloc_register(func);
    int32_t op_kind = get_unary_op_kind_c(opcode);
    phx_tc_emit(tc, hir_c_create_unary_op_reg(result, op_kind, operand, &tc->frame));
    phx_ptr_arr_push(&tc->frame.stack, result);
}

/* emitUnaryNot — LoadConst(False), PrimitiveCompare(Equal), PrimitiveBoxBool */
void hir_builder_emit_unary_not_c(PhxTranslationContext *tc, void *func) {
    void *operand = phx_ptr_arr_pop(&tc->frame.stack);
    void *const_false = hir_func_alloc_register(func);
    void *is_false = hir_func_alloc_register(func);
    void *result = hir_func_alloc_register(func);
    HirType t_false = hir_type_from_object(Py_False);
    phx_tc_emit(tc, hir_c_create_load_const(const_false, t_false));
    phx_tc_emit(tc, hir_c_create_primitive_compare(is_false, 2, const_false, operand)); /* kEqual=2 */
    phx_tc_emit(tc, hir_c_create_primitive_box_bool(result, is_false));
    phx_ptr_arr_push(&tc->frame.stack, result);
}

void hir_builder_emit_contains_op_c(PhxTranslationContext *tc, void *func, int oparg) {
    void *right = phx_ptr_arr_pop(&tc->frame.stack);
    void *left = phx_ptr_arr_pop(&tc->frame.stack);
    void *result = hir_func_alloc_register(func);
    int32_t op = (oparg == 0) ? 6 : 7; /* CompareOp::kIn=6, kNotIn=7 */
    phx_tc_emit(tc, hir_c_create_compare_reg(result, op, left, right, &tc->frame));
    phx_ptr_arr_push(&tc->frame.stack, result);
}

/* emitStoreLocal — pop, co_consts index lookup, localsplus write, Assign */
void hir_builder_emit_store_local_c(
        PhxTranslationContext *tc,
        void *func,
        PyCodeObject *code,
        int oparg) {
    void *src = phx_ptr_arr_pop(&tc->frame.stack);
    PyObject *index_and_descr = PyTuple_GET_ITEM(code->co_consts, oparg);
    int index = (int)PyLong_AsLong(PyTuple_GET_ITEM(index_and_descr, 0));
    void *dst = tc->frame.localsplus.data[index];
    phx_move_overwritten_stack_regs(tc, func, dst);
    phx_tc_emit(tc, hir_assign_create(dst, src));
}

/* emitConvertPrimitive — pop, IntConvert, push */
extern HirType hir_prim_type_to_type(int prim_type);

void hir_builder_emit_convert_primitive_c(
        PhxTranslationContext *tc,
        void *func,
        int oparg) {
    void *val = phx_ptr_arr_pop(&tc->frame.stack);
    void *out = hir_func_alloc_register(func);
    HirType to_type = hir_prim_type_to_type(oparg >> 4);
    phx_tc_emit(tc, hir_c_create_int_convert_reg(out, val, to_type));
    phx_ptr_arr_push(&tc->frame.stack, out);
}

/* emitStoreDeref — pop value, StealCellItem + SetCellItem */

void hir_builder_emit_store_deref_c(
        PhxTranslationContext *tc,
        void *func,
        int oparg) {
    void *old = hir_func_alloc_register(func);
    void *dst = tc->frame.localsplus.data[oparg];
    void *src = phx_ptr_arr_pop(&tc->frame.stack);
    phx_tc_emit(tc, hir_c_create_steal_cell_item_reg(old, dst));
    phx_tc_emit(tc, hir_c_create_set_cell_item_reg(dst, src, old));
}

/* emitStoreAttr — pop receiver + value, emit StoreAttr */

void hir_builder_emit_store_attr_c(PhxTranslationContext *tc, int oparg) {
    void *receiver = phx_ptr_arr_pop(&tc->frame.stack);
    void *value = phx_ptr_arr_pop(&tc->frame.stack);
    phx_tc_emit(tc, hir_c_create_store_attr_reg(receiver, value, oparg, &tc->frame));
}

/* emitLoadType — pop instance, LoadField ob_type, push */

void hir_builder_emit_load_type_c(PhxTranslationContext *tc, void *func) {
    void *instance = phx_ptr_arr_pop(&tc->frame.stack);
    void *type_reg = hir_func_alloc_register(func);
    HirType t_type = HIR_TYPE_TYPE;
    phx_tc_emit(tc, hir_c_create_load_field_reg(type_reg, instance, "ob_type",
        offsetof(PyObject, ob_type), t_type, 0));
    phx_ptr_arr_push(&tc->frame.stack, type_reg);
}

/* emitCopyDictWithoutKeys — peek keys+subject, emit, replace top */

void hir_builder_emit_copy_dict_without_keys_c(PhxTranslationContext *tc, void *func) {
    PhxPtrArray *stack = &tc->frame.stack;
    void *keys = stack->data[stack->count - 1];
    void *subject = stack->data[stack->count - 2];
    void *rest = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_copy_dict_without_keys_reg(rest, subject, keys, &tc->frame));
    stack->data[stack->count - 1] = rest;
}

/* emitImportFrom — peek name, emit ImportFrom, push */

void hir_builder_emit_import_from_c(PhxTranslationContext *tc, void *func, int oparg) {
    void *name = tc->frame.stack.data[tc->frame.stack.count - 1];
    void *res = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_import_from_reg(res, name, oparg, &tc->frame));
    phx_ptr_arr_push(&tc->frame.stack, res);
}

/* emitLoadDeref — localsplus cell read, LoadCellItem + CheckVar/CheckFreevar */

void hir_builder_emit_load_deref_c(
        PhxTranslationContext *tc,
        void *func,
        PyCodeObject *code,
        int oparg) {
    int idx = oparg;
    void *src = tc->frame.localsplus.data[idx];
    void *dst = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_load_cell_item_reg(dst, src));
    PyObject *name = get_varname(code, idx);
#if PY_VERSION_HEX < 0x030C0000
    phx_tc_emit(tc, hir_c_create_check_var_reg(dst, dst, name, &tc->frame));
#else
    if (idx < PyCode_GetFirstFree(code)) {
        phx_tc_emit(tc, hir_c_create_check_var_reg(dst, dst, name, &tc->frame));
    } else {
        phx_tc_emit(tc, hir_c_create_check_freevar_reg(dst, dst, name, &tc->frame));
    }
#endif
    phx_ptr_arr_push(&tc->frame.stack, dst);
}

/* emitDictUpdate — pop update, peek dict at depth, emit DictUpdate */

void hir_builder_emit_dict_update_c(PhxTranslationContext *tc, void *func, int oparg) {
    void *update = phx_ptr_arr_pop(&tc->frame.stack);
    void *dict = tc->frame.stack.data[tc->frame.stack.count - oparg];
    void *dst = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_dict_update_reg(dst, dict, update, &tc->frame));
}

/* emitConvertValue — pop value, emit ConvertValue, push */

void hir_builder_emit_convert_value_c(PhxTranslationContext *tc, void *func, int oparg) {
    void *value = phx_ptr_arr_pop(&tc->frame.stack);
    void *out = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_convert_value_reg(out, value, oparg, &tc->frame));
    phx_ptr_arr_push(&tc->frame.stack, out);
}

/* emitStoreSubscr — snapshot, pop sub+container+value, guard types, emit StoreSubscr */

void hir_builder_emit_store_subscr_c(
        PhxTranslationContext *tc,
        int specialized_opcode) {
    PhxPtrArray *stack = &tc->frame.stack;

    if (jit_get_config()->specialized_opcodes) {
        phx_tc_emit(tc, hir_c_create_snapshot(&tc->frame));
    }

    void *sub = phx_ptr_arr_pop(stack);
    void *container = phx_ptr_arr_pop(stack);
    void *value = phx_ptr_arr_pop(stack);

    if (jit_get_config()->specialized_opcodes) {
        switch (specialized_opcode) {
            case STORE_SUBSCR_DICT: {
                HirType t_dict = hir_type_from_pytype(&PyDict_Type, 1);
                phx_tc_emit(tc, hir_c_create_guard_type_reg(container, t_dict, container));
                break;
            }
            case STORE_SUBSCR_LIST_INT: {
                HirType t_list = hir_type_from_pytype(&PyList_Type, 1);
                HirType t_long = hir_type_from_pytype(&PyLong_Type, 1);
                phx_tc_emit(tc, hir_c_create_guard_type_reg(container, t_list, container));
                phx_tc_emit(tc, hir_c_create_guard_type_reg(sub, t_long, sub));
                break;
            }
            default:
                break;
        }
    }

    phx_tc_emit(tc, hir_c_create_store_subscr_reg(container, sub, value, &tc->frame));
}

/* emitFormatWithSpec — pop fmt_spec+value, emit FormatWithSpec, push */

void hir_builder_emit_format_with_spec_c(PhxTranslationContext *tc, void *func) {
    void *fmt_spec = phx_ptr_arr_pop(&tc->frame.stack);
    void *value = phx_ptr_arr_pop(&tc->frame.stack);
    void *out = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_format_with_spec_reg(out, value, fmt_spec, &tc->frame));
    phx_ptr_arr_push(&tc->frame.stack, out);
}

/* emitMapAdd — pop value+key, peek map at depth, emit SetDictItem */

void hir_builder_emit_map_add_c(PhxTranslationContext *tc, void *func, int oparg) {
    void *value = phx_ptr_arr_pop(&tc->frame.stack);
    void *key = phx_ptr_arr_pop(&tc->frame.stack);
    void *map = tc->frame.stack.data[tc->frame.stack.count - oparg];
    void *dst = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_set_dict_item_reg(dst, map, key, value, &tc->frame));
}

/* emitSetAdd — pop value, peek set at depth, emit SetSetItem */

void hir_builder_emit_set_add_c(PhxTranslationContext *tc, void *func, int oparg) {
    void *item = phx_ptr_arr_pop(&tc->frame.stack);
    void *set = tc->frame.stack.data[tc->frame.stack.count - oparg];
    void *dst = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_set_set_item_reg(dst, set, item, &tc->frame));
}

/* emitSetUpdate — pop iterable, peek set at depth, emit SetUpdate */

void hir_builder_emit_set_update_c(PhxTranslationContext *tc, void *func, int oparg) {
    void *iterable = phx_ptr_arr_pop(&tc->frame.stack);
    void *set = tc->frame.stack.data[tc->frame.stack.count - oparg];
    void *dst = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_set_update_reg(dst, set, iterable, &tc->frame));
}

/* Generic variadic deopt emit — C equivalent of emitVariadicDeopt */
extern void hir_deopt_set_frame_state(void *deopt_instr, const void *frame_state);

static void phx_emit_variadic_deopt(PhxTranslationContext *tc, void *func,
                                     int opcode, size_t num_operands) {
    void *out = hir_func_alloc_register(func);
    void *instr = hir_c_alloc_instr(sizeof(HirDeoptLayout), num_operands);
    hir_c_init_deopt(instr, opcode);
    hir_c_set_output(instr, out);
    for (size_t i = num_operands; i > 0; i--) {
        void *operand = phx_ptr_arr_pop(&tc->frame.stack);
        hir_c_set_operand(instr, i - 1, operand);
    }
    hir_deopt_set_frame_state(instr, &tc->frame);
    phx_ptr_arr_push(&tc->frame.stack, out);
    phx_tc_emit(tc, instr);
}

/* emitBuildSlice — variadic BuildSlice */
void hir_builder_emit_build_slice_c(PhxTranslationContext *tc, void *func, int oparg) {
    phx_emit_variadic_deopt(tc, func, HIR_OP_BuildSlice, (size_t)oparg);
}

/* emitBuildString — variadic BuildString */
void hir_builder_emit_build_string_c(PhxTranslationContext *tc, void *func, int oparg) {
    phx_emit_variadic_deopt(tc, func, HIR_OP_BuildString, (size_t)oparg);
}

/* emitBinarySlice — BuildSlice(2) + pop container + BinaryOp(kSubscript) */

void hir_builder_emit_binary_slice_c(PhxTranslationContext *tc, void *func) {
    phx_emit_variadic_deopt(tc, func, HIR_OP_BuildSlice, 2);
    void *slice = phx_ptr_arr_pop(&tc->frame.stack);
    void *container = phx_ptr_arr_pop(&tc->frame.stack);
    void *result = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_binary_op_reg(result, HIR_BOP_Subscript, container, slice, &tc->frame));
    phx_ptr_arr_push(&tc->frame.stack, result);
}

/* emitStoreSlice — BuildSlice(2) + pop slice+container+values + StoreSubscr */
void hir_builder_emit_store_slice_c(PhxTranslationContext *tc, void *func) {
    phx_emit_variadic_deopt(tc, func, HIR_OP_BuildSlice, 2);
    void *slice = phx_ptr_arr_pop(&tc->frame.stack);
    void *container = phx_ptr_arr_pop(&tc->frame.stack);
    void *values = phx_ptr_arr_pop(&tc->frame.stack);
    phx_tc_emit(tc, hir_c_create_store_subscr_reg(container, slice, values, &tc->frame));
}

/* emitAsyncForHeaderYieldFrom — yield from async for header */
extern void *hir_builder_get_block_at_off(void *builder, int byte_offset);

void hir_builder_emit_async_for_header_yield_from_c(
        PhxTranslationContext *tc, void *func, void *builder,
        PyCodeObject *code, int next_instr_offset) {
    void *send_value = phx_ptr_arr_pop(&tc->frame.stack);
    void *awaitable = tc->frame.stack.data[tc->frame.stack.count - 1];
    void *out = hir_func_alloc_register(func);
    if (code->co_flags & CO_COROUTINE) {
        phx_tc_emit(tc, hir_c_create_set_current_awaiter_reg(awaitable));
    }
    phx_tc_emit(tc, hir_c_create_yield_from_handle_stop_async_reg(out, send_value, awaitable, &tc->frame));
    phx_ptr_arr_pop(&tc->frame.stack);
    phx_ptr_arr_push(&tc->frame.stack, out);

    void *yf_cont_block = hir_builder_get_block_at_off(builder, next_instr_offset);
    PhxExecBlock *top = (PhxExecBlock *)tc->frame.block_stack_data;
    int handler_off = top[tc->frame.block_stack_count - 1].handler_off;
    void *yf_done_block = hir_builder_get_block_at_off(builder, handler_off);
    phx_tc_emit(tc, hir_c_create_cond_branch_iter_not_done_cpp(out, yf_cont_block, yf_done_block));
}

/* emitStoreField — pop receiver+value, fieldInfo lookup, StoreField */
extern void hir_builder_preloader_field_info(void *builder, PyObject *descr,
                                              intptr_t *offset_out, HirType *type_out,
                                              PyObject **name_out);

void hir_builder_emit_store_field_c(PhxTranslationContext *tc, void *func, void *builder,
                                     PyCodeObject *code, int oparg) {
    PyObject *descr = PyTuple_GET_ITEM(code->co_consts, oparg);
    intptr_t offset;
    HirType type;
    PyObject *name;
    hir_builder_preloader_field_info(builder, descr, &offset, &type, &name);
    const char *field_name = PyUnicode_AsUTF8(name);
    if (field_name == NULL) {
        PyErr_Clear();
        field_name = "";
    }

    void *receiver = phx_ptr_arr_pop(&tc->frame.stack);
    void *value = phx_ptr_arr_pop(&tc->frame.stack);
    void *previous = hir_func_alloc_register(func);

    HirType t_prim = (HirType)HIR_TYPE_PRIMITIVE;
    if ((hir_type_bits(&type) & ~hir_type_bits(&t_prim)) == 0 && hir_type_bits(&type) != 0) {
        void *converted = hir_func_alloc_register(func);
        HirType t_nullptr = (HirType)HIR_TYPE_NULLPTR;
        phx_tc_emit(tc, hir_c_create_load_const(previous, t_nullptr));
        phx_tc_emit(tc, hir_c_create_int_convert_reg(converted, value, type));
        value = converted;
    } else {
        phx_tc_emit(tc, hir_c_create_load_field_reg(previous, receiver, field_name, offset, type, 0));
    }
    phx_tc_emit(tc, hir_c_create_store_field_reg(receiver, field_name, offset, value, type, previous));
}

/* emitMakeFunction — pop code (+qualname pre-3.11), MakeFunction + SetFunctionAttr */
#ifndef MAKE_FUNCTION_DEFAULTS
#define MAKE_FUNCTION_DEFAULTS    0x01
#define MAKE_FUNCTION_KWDEFAULTS  0x02
#define MAKE_FUNCTION_ANNOTATIONS 0x04
#define MAKE_FUNCTION_CLOSURE     0x08
#endif
#ifndef FUNC_ATTR_CLOSURE
#define FUNC_ATTR_CLOSURE     0
#define FUNC_ATTR_ANNOTATIONS 1
#define FUNC_ATTR_KWDEFAULTS  2
#define FUNC_ATTR_DEFAULTS    3
#endif

void hir_builder_emit_make_function_c(PhxTranslationContext *tc, void *func, int oparg) {
    void *func_reg = hir_func_alloc_register(func);
    void *qualname;
#if PY_VERSION_HEX < 0x030B0000
    qualname = phx_ptr_arr_pop(&tc->frame.stack);
#else
    qualname = hir_func_alloc_register(func);
    HirType t_nullptr = (HirType)HIR_TYPE_NULLPTR;
    phx_tc_emit(tc, hir_c_create_load_const(qualname, t_nullptr));
#endif
    void *codeobj = phx_ptr_arr_pop(&tc->frame.stack);
    phx_tc_emit(tc, hir_c_create_make_function_reg(func_reg, codeobj, qualname, &tc->frame));

    if (oparg & MAKE_FUNCTION_CLOSURE) {
        void *closure = phx_ptr_arr_pop(&tc->frame.stack);
        phx_tc_emit(tc, hir_c_create_set_function_attr_reg(closure, func_reg, FUNC_ATTR_CLOSURE));
    }
    if (oparg & MAKE_FUNCTION_ANNOTATIONS) {
        void *annotations = phx_ptr_arr_pop(&tc->frame.stack);
        phx_tc_emit(tc, hir_c_create_set_function_attr_reg(annotations, func_reg, FUNC_ATTR_ANNOTATIONS));
    }
    if (oparg & MAKE_FUNCTION_KWDEFAULTS) {
        void *kwdefaults = phx_ptr_arr_pop(&tc->frame.stack);
        phx_tc_emit(tc, hir_c_create_set_function_attr_reg(kwdefaults, func_reg, FUNC_ATTR_KWDEFAULTS));
    }
    if (oparg & MAKE_FUNCTION_DEFAULTS) {
        void *defaults = phx_ptr_arr_pop(&tc->frame.stack);
        phx_tc_emit(tc, hir_c_create_set_function_attr_reg(defaults, func_reg, FUNC_ATTR_DEFAULTS));
    }
    phx_ptr_arr_push(&tc->frame.stack, func_reg);
}

/* emitLoadMethod — pop receiver, LoadMethod + GetSecondOutput, push 2 */
extern HirType hir_type_union(HirType a, HirType b);

void hir_builder_emit_load_method_c(PhxTranslationContext *tc, void *func, int name_idx) {
    void *receiver = phx_ptr_arr_pop(&tc->frame.stack);
    void *result = hir_func_alloc_register(func);
    void *method_instance = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_load_method_reg(result, receiver, name_idx, &tc->frame));
    HirType t_opt_object = hir_type_union((HirType)HIR_TYPE_OBJECT, (HirType)HIR_TYPE_NULLPTR);
    phx_tc_emit(tc, hir_c_create_get_second_output_reg(method_instance, t_opt_object, result));
    phx_ptr_arr_push(&tc->frame.stack, result);
    phx_ptr_arr_push(&tc->frame.stack, method_instance);
}

/* emitLoadAttrSlot — LOAD_ATTR_SLOT specialization */
extern void hir_builder_get_attr_cache(void *builder, int instr_idx,
                                        uint32_t *version_out, uint16_t *index_out);
extern PyTypeObject *hir_find_type_by_version_tag(uint32_t version);

int hir_builder_emit_load_attr_slot_c(
        PhxTranslationContext *tc, void *func, void *builder,
        void *receiver, PyCodeObject *code, int name_idx, int instr_idx) {
    uint32_t type_version;
    uint16_t slot_offset;
    hir_builder_get_attr_cache(builder, instr_idx, &type_version, &slot_offset);

    PyTypeObject *slot_type = hir_find_type_by_version_tag(type_version);
    if (slot_type == NULL ||
        (slot_type->tp_subclasses != NULL &&
         PyDict_GET_SIZE((PyObject *)slot_type->tp_subclasses) > 0)) {
        return 0;
    }

    hir_func_add_reference(func, (PyObject *)slot_type);
    HirType type = hir_type_from_pytype(slot_type, 1);
    phx_tc_emit(tc, hir_c_create_guard_type_reg(receiver, type, receiver));

    PyObject *attr_name = PyTuple_GET_ITEM(code->co_names, name_idx);

    void *attr = hir_func_alloc_register(func);
    HirType t_opt_object = hir_type_union((HirType)HIR_TYPE_OBJECT, (HirType)HIR_TYPE_NULLPTR);
    phx_tc_emit(tc, hir_c_create_load_field_reg(attr, receiver, "slot",
        (intptr_t)slot_offset, t_opt_object, 0));

    void *result = hir_func_alloc_register(func);
    void *cf = hir_c_create_check_field_reg(result, attr, attr_name, &tc->frame);
    hir_c_set_guilty_reg(cf, receiver);
    phx_tc_emit(tc, cf);

    phx_ptr_arr_push(&tc->frame.stack, result);
    return 1;
}

/* emitLoadAttrModule — LOAD_ATTR_MODULE specialization */
extern void *jit_rt_load_module_dict_entry_addr(void);

int hir_builder_emit_load_attr_module_c(
        PhxTranslationContext *tc, void *func, void *builder,
        void *receiver, PyCodeObject *code, int name_idx, int instr_idx) {
    HirType t_module = hir_type_from_pytype(&PyModule_Type, 1);
    phx_tc_emit(tc, hir_c_create_guard_type_reg(receiver, t_module, receiver));

    uint32_t dict_version;
    uint16_t index;
    hir_builder_get_attr_cache(builder, instr_idx, &dict_version, &index);

    if (dict_version == 0) {
        return 0; /* fallback to generic LoadAttr */
    }

    HirType t_object   = (HirType)HIR_TYPE_OBJECT;
    HirType t_cptr     = (HirType)HIR_TYPE_CPTR;
    HirType t_cuint32  = (HirType)HIR_TYPE_CUINT32;
    HirType t_cint64   = (HirType)HIR_TYPE_CINT64;
    HirType t_optobj   = hir_type_union((HirType)HIR_TYPE_OBJECT, (HirType)HIR_TYPE_NULLPTR);

    void *dict = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_load_field_reg(dict, receiver, "md_dict",
        (intptr_t)offsetof(PyModuleObject, md_dict), t_object, 0));

    void *keys = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_load_field_reg(keys, dict, "ma_keys",
        (intptr_t)offsetof(PyDictObject, ma_keys), t_cptr, 0));

    void *loaded_version = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_load_field_reg(loaded_version, keys, "dk_version",
        (intptr_t)offsetof(PyDictKeysObject, dk_version), t_cuint32, 0));

    void *expected_version = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_load_const(expected_version,
        hir_type_from_cuint((uint64_t)dict_version, t_cuint32)));

    void *version_match = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_primitive_compare(version_match,
        HIR_PCMP_Equal, loaded_version, expected_version));

    void *guard = hir_c_create_guard(version_match);
    hir_deopt_set_frame_state(guard, &tc->frame);
    phx_tc_emit(tc, guard);

    void *index_reg = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_load_const(index_reg,
        hir_type_from_cint((int64_t)index, t_cint64)));

    void *result = hir_func_alloc_register(func);
    void *call = hir_c_create_call_static_reg(2, result,
        jit_rt_load_module_dict_entry_addr(), t_optobj);
    hir_c_set_operand(call, 0, keys);
    hir_c_set_operand(call, 1, index_reg);
    phx_tc_emit(tc, call);

    PyObject *attr_name = PyTuple_GET_ITEM(code->co_names, name_idx);
    void *cf = hir_c_create_check_field_reg(result, result, attr_name, &tc->frame);
    hir_c_set_guilty_reg(cf, receiver);
    phx_tc_emit(tc, cf);

    phx_ptr_arr_push(&tc->frame.stack, result);
    return 1;
}

/* emitLoadAttrInstanceValue — LOAD_ATTR_INSTANCE_VALUE specialization */
int hir_builder_emit_load_attr_instance_value_c(
        PhxTranslationContext *tc, void *func, void *builder,
        void *receiver, PyCodeObject *code, int name_idx, int instr_idx) {
    uint32_t type_version;
    uint16_t index;
    hir_builder_get_attr_cache(builder, instr_idx, &type_version, &index);

    PyTypeObject *slot_type = hir_find_type_by_version_tag(type_version);
    if (slot_type == NULL ||
        (slot_type->tp_subclasses != NULL &&
         PyDict_GET_SIZE((PyObject *)slot_type->tp_subclasses) > 0) ||
        !PyType_HasFeature(slot_type, Py_TPFLAGS_MANAGED_DICT)) {
        /* Optional fallback: emit GuardType only if type found and no subclasses */
        if (slot_type != NULL &&
            (slot_type->tp_subclasses == NULL ||
             PyDict_GET_SIZE((PyObject *)slot_type->tp_subclasses) == 0)) {
            HirType t_only = hir_type_from_pytype(slot_type, 1);
            phx_tc_emit(tc, hir_c_create_guard_type_reg(receiver, t_only, receiver));
        }
        return 0;
    }

    PyHeapTypeObject *ht = (PyHeapTypeObject *)slot_type;
    if (ht->ht_cached_keys == NULL) {
        HirType t_only = hir_type_from_pytype(slot_type, 1);
        phx_tc_emit(tc, hir_c_create_guard_type_reg(receiver, t_only, receiver));
        return 0;
    }

    hir_func_add_reference(func, (PyObject *)slot_type);
    HirType t_type = hir_type_from_pytype(slot_type, 1);
    phx_tc_emit(tc, hir_c_create_guard_type_reg(receiver, t_type, receiver));

    PyObject *attr_name = PyTuple_GET_ITEM(code->co_names, name_idx);

    /* Load PyDictOrValues from managed-dict slot at offset -3 * sizeof(PyObject*).
     * borrowed=true: dorv may be a tagged pointer (low bit = inline values) which is
     * not a valid PyObject*. */
    HirType t_optdict = hir_type_union((HirType)HIR_TYPE_DICT, (HirType)HIR_TYPE_NULLPTR);
    void *dorv = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_load_field_reg(dorv, receiver, "__dict__",
        (intptr_t)(-3 * (intptr_t)sizeof(PyObject *)), t_optdict, 1));

    /* CheckField dorv != NULL */
    void *checked_dorv = hir_func_alloc_register(func);
    void *cf1 = hir_c_create_check_field_reg(checked_dorv, dorv, attr_name, &tc->frame);
    hir_c_set_guilty_reg(cf1, receiver);
    phx_tc_emit(tc, cf1);

    /* dorv low bit: 1 = inline values, 0 = regular dict (deopt the regular case) */
    HirType t_cuint64 = (HirType)HIR_TYPE_CUINT64;
    HirType t_optobj  = hir_type_union((HirType)HIR_TYPE_OBJECT, (HirType)HIR_TYPE_NULLPTR);
    void *one = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_load_const(one, hir_type_from_cuint(1, t_cuint64)));

    void *dorv_int = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_bit_cast(dorv_int, checked_dorv, t_cuint64));

    void *is_values = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_int_binary_op(is_values, /*kAnd*/1, dorv_int, one));

    void *guard = hir_c_create_guard(is_values);
    hir_deopt_set_frame_state(guard, &tc->frame);
    hir_c_set_guilty_reg(guard, receiver);
    hir_c_set_descr(guard, "dict values check");
    phx_tc_emit(tc, guard);

    /* values_ptr = dorv + 1 (tag stripped → pointer to values array) */
    void *values_ptr = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_int_binary_op(values_ptr, /*kAdd*/0, dorv_int, one));

    void *values_obj = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_bit_cast(values_obj, values_ptr, t_optobj));

    /* Load attribute at cached index */
    void *attr = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_load_field_reg(attr, values_obj, "attr",
        (intptr_t)((intptr_t)index * (intptr_t)sizeof(PyObject *)), t_optobj, 0));

    void *result = hir_func_alloc_register(func);
    void *cf2 = hir_c_create_check_field_reg(result, attr, attr_name, &tc->frame);
    hir_c_set_guilty_reg(cf2, receiver);
    phx_tc_emit(tc, cf2);

    phx_ptr_arr_push(&tc->frame.stack, result);
    return 1;
}

/* emitTypeAnnotationGuards — walk function args, emit GuardType for any
 * arg with a PyType-checkable annotation. C port closes G2 deletion-gate
 * item (Preloader.annotations bridge) per supervisor 22:32:58Z + theologian
 * 04:18:21Z. */
extern void *hir_builder_preloader_annotations(void *builder);
extern int hir_builder_preloader_num_args(void *builder);

void hir_builder_emit_type_annotation_guards_c(
        PhxTranslationContext *tc, void *func, void *builder) {
    HirAnnotationIndex *index =
        (HirAnnotationIndex *)hir_builder_preloader_annotations(builder);
    if (index == NULL) {
        return;
    }

    PyCodeObject *code = (PyCodeObject *)tc->frame.code;
    int num_args = hir_builder_preloader_num_args(builder);
    int first = 1;

    for (int arg_idx = 0; arg_idx < num_args; arg_idx++) {
        PyObject *annotation = hir_annotation_index_find(
            index, get_varname(code, arg_idx));

        /* Skip args without an annotation OR annotation isn't a plain PyType
         * (matches C++ — unions / complex types skipped, not yet supported). */
        if (annotation == NULL || !PyType_Check(annotation)) {
            continue;
        }

        /* Snapshot ONLY ON FIRST guard (first-flag pattern from C++).
         * cur_instr_offs = 0 ensures deopt restarts at instruction 0
         * (no bytecode has been compiled yet at this point). */
        if (first) {
            first = 0;
            tc->frame.cur_instr_offs = 0;
            phx_tc_emit(tc, hir_c_create_snapshot(&tc->frame));
        }

        /* Guard the arg register against the annotated exact type.
         * arg comes from localsplus (allocated by allocateLocalsplus). */
        void *arg = tc->frame.localsplus.data[arg_idx];
        JIT_CHECK_C(arg != NULL, "No register for argument %d", arg_idx);
        HirType type = hir_type_from_pytype((PyTypeObject *)annotation, /*is_exact=*/1);
        phx_tc_emit(tc, hir_c_create_guard_type_reg(arg, type, arg));
    }
}

/* emitResume — RESUME opcode handler. Inserts an eval-breaker check + periodic
 * tasks block when oparg < 2 (oparg 2/3 are no-op resumes, e.g. coroutine
 * sends/throws which don't need periodic-task points). Mirrors C++
 * HIRBuilder::emitResume @ builder.cpp:3175.
 *
 * Bridge delegation: insertRunPeriodicActivites stays C++ (private member,
 * accessed via friend bridge hir_builder_insert_run_periodic_activities_c).
 * That helper carries Py_GIL_DISABLED #ifdef and emit-helper chain — no
 * value in re-porting it now.
 *
 * Snapshot-on-succ-block trick: instead of allocating a temp PhxTranslationContext
 * (which would need phx_frame_state_copy + destroy), we capture the original
 * tc->block, switch tc->block to succ_block, emit snapshot, then call the
 * helper. Equivalent because:
 *   - tc->frame == succ.frame at this point (succ was just constructed from
 *     a frame copy, no mutations between — emitSnapshot doesn't mutate frame)
 *   - Snapshot lands in succ_block (correct destination)
 *   - Final tc->block == succ_block (matches C++ tc.block = succ.block) */
extern void hir_builder_insert_run_periodic_activities_c(
    void *builder, void *func,
    void *check_block, void *succ_block, void *frame_state);

void hir_builder_emit_resume_c(
        PhxTranslationContext *tc, void *func, void *builder, int oparg) {
    if (oparg >= 2) {
        return;
    }
    void *check_block = tc->block;
    void *succ_block = hir_cfg_alloc_block(func);
    tc->block = succ_block;
    phx_tc_emit(tc, hir_c_create_snapshot(&tc->frame));
    hir_builder_insert_run_periodic_activities_c(
        builder, func, check_block, succ_block, &tc->frame);
    /* tc->block already == succ_block (final state matches C++ tc.block = succ.block) */
}

/* emitKwNames — KW_NAMES opcode handler. Saves the keyword-names tuple
 * (from co_consts[oparg]) into the HIRBuilder kwnames_ slot for the next
 * CALL/CALL_KW to consume as part of the call's operands. Mirrors C++
 * HIRBuilder::emitKwNames @ builder.cpp:3201.
 *
 * kwnames_ is a NON-STACK temp register on HIRBuilder (one slot, consumed
 * by the next CALL opcode). emitKwNames asserts kwnames_ is empty before
 * overwriting (matches C++ JIT_CHECK invariant). Bridge access to kwnames_
 * via getter/setter (hir_builder_get_kwnames / hir_builder_set_kwnames) so
 * future emitCall conversion can also access this state.
 *
 * AllocateNonStack equivalent: hir_func_alloc_register(func) — same path
 * (TempAllocator::AllocateNonStack just calls env_->AllocateRegister at
 * builder.cpp:298). emitLoadConst equivalent: hir_c_create_load_const(reg, type)
 * + phx_tc_emit. Type::fromObject equivalent: hir_type_from_object(obj). */
extern void *hir_builder_get_kwnames(void *builder);
extern void hir_builder_set_kwnames(void *builder, void *reg);

void hir_builder_emit_kw_names_c(
        PhxTranslationContext *tc, void *func, void *builder,
        PyCodeObject *code, int oparg) {
    Py_ssize_t consts_len = PyTuple_Size(code->co_consts);
    JIT_CHECK_C(oparg < consts_len,
                "KW_NAMES index %d is greater than co_consts length %zd",
                oparg, consts_len);
    JIT_CHECK_C(hir_builder_get_kwnames(builder) == NULL,
                "Trying to save KW_NAMES(%d) but previous kwnames_ value wasn't "
                "consumed by a CALL* opcode yet",
                oparg);

    void *kwnames_reg = hir_func_alloc_register(func);
    hir_builder_set_kwnames(builder, kwnames_reg);
    PyObject *names_tuple = PyTuple_GET_ITEM(code->co_consts, oparg);
    HirType type = hir_type_from_object(names_tuple);
    phx_tc_emit(tc, hir_c_create_load_const(kwnames_reg, type));
}

/* emitLoadIterableArg — LOAD_ITERABLE_ARG opcode handler. Pops an iterable
 * from the stack, ensures it's a tuple (via type-check + GetTuple fallback),
 * then loads the element at index `oparg` from the tuple onto the stack
 * along with the tuple itself. Mirrors C++ HIRBuilder::emitLoadIterableArg
 * @ builder.cpp:3313.
 *
 * Multi-block CFG (when iterable->type() != TTupleExact):
 *   tc.block (current) --CondBranchCheckType(TTuple)--> tuple_path / non_tuple_path
 *   tuple_path:     Snapshot, Assign(tuple_reg, iterable), Branch(merge)
 *   non_tuple_path: Snapshot, GetTuple(tuple_reg, iterable), Branch(merge)
 *   merge_block:    Snapshot, ... (continues with primitive box + subscript)
 *
 * Stack-temp allocation via hir_builder_temps_alloc_stack (NEW BRIDGE):
 * preserves cache_ side-effect required by GetOrAllocateStack at
 * builder.cpp:2842 (stack-layout computation). Cannot use bare
 * hir_func_alloc_register (no cache update). */
/* hir_register_type canonical decl in hir_c_api.h. */
extern HirType hir_type_from_cint(int64_t value, HirType cint_type);
extern void *hir_builder_temps_alloc_stack(void *builder);

void hir_builder_emit_load_iterable_arg_c(
        PhxTranslationContext *tc, void *func, void *builder, int oparg) {
    void *iterable = phx_ptr_arr_pop(&tc->frame.stack);
    void *tuple_reg = NULL;

    HirType iter_type = hir_register_type(iterable);
    HirType t_tuple_exact = HIR_TYPE_TUPLEEXACT;
    if (!hir_type_equal(&iter_type, &t_tuple_exact)) {
        /* Multi-block path: type-check at runtime, then converge. */
        void *tuple_path_block = hir_cfg_alloc_block(func);
        void *non_tuple_path_block = hir_cfg_alloc_block(func);
        void *merge_block = hir_cfg_alloc_block(func);

        /* Emit Snapshot into tuple_path_block. */
        void *saved_block = tc->block;
        tc->block = tuple_path_block;
        phx_tc_emit(tc, hir_c_create_snapshot(&tc->frame));

        /* Emit Snapshot into non_tuple_path_block. */
        tc->block = non_tuple_path_block;
        phx_tc_emit(tc, hir_c_create_snapshot(&tc->frame));

        /* Restore current block to emit the cond-branch terminator. */
        tc->block = saved_block;
        HirType t_tuple = HIR_TYPE_TUPLE;
        phx_tc_emit(tc, hir_c_create_cond_branch_check_type_cpp(
            iterable, t_tuple, tuple_path_block, non_tuple_path_block));

        /* Move to merge block + emit Snapshot. */
        tc->block = merge_block;
        phx_tc_emit(tc, hir_c_create_snapshot(&tc->frame));

        /* Allocate the tuple register (with cache_ side-effect). */
        tuple_reg = hir_builder_temps_alloc_stack(builder);

        /* tuple_path: Assign(tuple_reg, iterable); Branch(merge). */
        tc->block = tuple_path_block;
        phx_tc_emit(tc, hir_assign_create(tuple_reg, iterable));
        phx_tc_emit(tc, hir_c_create_branch_cpp(merge_block));

        /* non_tuple_path: GetTuple(tuple_reg, iterable); Branch(merge). */
        tc->block = non_tuple_path_block;
        phx_tc_emit(tc, hir_c_create_get_tuple_reg(
            tuple_reg, iterable, &tc->frame));
        phx_tc_emit(tc, hir_c_create_branch_cpp(merge_block));

        /* Continue on merge_block. */
        tc->block = merge_block;
    } else {
        tuple_reg = iterable;
    }

    /* Load the element at index oparg from the tuple. */
    void *tmp = hir_builder_temps_alloc_stack(builder);
    void *tup_idx = hir_builder_temps_alloc_stack(builder);
    void *element = hir_builder_temps_alloc_stack(builder);

    HirType t_cint64 = HIR_TYPE_CINT64;
    HirType const_type = hir_type_from_cint((int64_t)oparg, t_cint64);
    phx_tc_emit(tc, hir_c_create_load_const(tmp, const_type));
    phx_tc_emit(tc, hir_c_create_primitive_box_reg(
        tup_idx, tmp, t_cint64, &tc->frame));
    phx_tc_emit(tc, hir_c_create_binary_op_reg(
        element, HIR_BOP_Subscript, tuple_reg, tup_idx, &tc->frame));

    phx_ptr_arr_push(&tc->frame.stack, element);
    phx_ptr_arr_push(&tc->frame.stack, tuple_reg);
}

/* emitLoadCommonConstant — LOAD_COMMON_CONSTANT opcode handler. Allocates a
 * stack temp, emits LoadConst with the type returned by the JIT context's
 * common-constant table, pushes onto stack. Mirrors C++
 * HIRBuilder::emitLoadCommonConstant @ builder.cpp:5545.
 *
 * Bridge: hir_get_context_type_for_common_constant wraps the C++ context
 * accessor (jit::getContext()->typeForCommonConstant). Free-function bridge
 * — no friend decl needed (no HIRBuilder access). */
extern HirType hir_get_context_type_for_common_constant(int i);

void hir_builder_emit_load_common_constant_c(
        PhxTranslationContext *tc, void *builder, int oparg) {
    void *out = hir_builder_temps_alloc_stack(builder);
    HirType type = hir_get_context_type_for_common_constant(oparg);
    phx_tc_emit(tc, hir_c_create_load_const(out, type));
    phx_ptr_arr_push(&tc->frame.stack, out);
}

/* emitLoadAttr generic — non-specialized LoadAttr2 fallback */

void hir_builder_emit_load_attr_generic_c(PhxTranslationContext *tc, void *func,
                                           void *receiver, int name_idx) {
    void *result = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_load_attr_reg2(result, receiver, name_idx, &tc->frame));
    phx_ptr_arr_push(&tc->frame.stack, result);
}

/* emitBinaryOp — specialized guards + oparg dispatch + BinaryOp/InPlaceOp */

static int32_t get_binary_op_kind_from_oparg_c(int oparg) {
    switch (oparg) {
        case NB_ADD:              return HIR_BOP_Add;
        case NB_AND:              return HIR_BOP_And;
        case NB_FLOOR_DIVIDE:     return HIR_BOP_FloorDivide;
        case NB_LSHIFT:           return HIR_BOP_LShift;
        case NB_MATRIX_MULTIPLY:  return HIR_BOP_MatrixMultiply;
        case NB_MULTIPLY:         return HIR_BOP_Multiply;
        case NB_REMAINDER:        return HIR_BOP_Modulo;
        case NB_OR:               return HIR_BOP_Or;
        case NB_POWER:            return HIR_BOP_Power;
        case NB_RSHIFT:           return HIR_BOP_RShift;
        case NB_SUBTRACT:         return HIR_BOP_Subtract;
        case NB_TRUE_DIVIDE:      return HIR_BOP_TrueDivide;
        case NB_XOR:              return HIR_BOP_Xor;
        default:                  return -1;
    }
}

static int32_t get_inplace_op_kind_from_oparg_c(int oparg) {
    switch (oparg) {
        case NB_INPLACE_ADD:            return HIR_IOP_Add;
        case NB_INPLACE_AND:            return HIR_IOP_And;
        case NB_INPLACE_FLOOR_DIVIDE:   return HIR_IOP_FloorDivide;
        case NB_INPLACE_LSHIFT:         return HIR_IOP_LShift;
        case NB_INPLACE_MATRIX_MULTIPLY: return HIR_IOP_MatrixMultiply;
        case NB_INPLACE_MULTIPLY:       return HIR_IOP_Multiply;
        case NB_INPLACE_REMAINDER:      return HIR_IOP_Modulo;
        case NB_INPLACE_OR:             return HIR_IOP_Or;
        case NB_INPLACE_POWER:          return HIR_IOP_Power;
        case NB_INPLACE_RSHIFT:         return HIR_IOP_RShift;
        case NB_INPLACE_SUBTRACT:       return HIR_IOP_Subtract;
        case NB_INPLACE_TRUE_DIVIDE:    return HIR_IOP_TrueDivide;
        case NB_INPLACE_XOR:            return HIR_IOP_Xor;
        default:                        return -1;
    }
}

/* Returns 1 if handled in C, 0 if B2 exception path needs C++ callback */
int hir_builder_emit_binary_op_c(
        PhxTranslationContext *tc,
        void *func,
        int opcode,
        int oparg,
        int specialized_opcode) {
    PhxPtrArray *stack = &tc->frame.stack;

    if (jit_get_config()->specialized_opcodes) {
        phx_tc_emit(tc, hir_c_create_snapshot(&tc->frame));
    }

    void *right = phx_ptr_arr_pop(stack);
    void *left = phx_ptr_arr_pop(stack);
    void *result = hir_func_alloc_register(func);

    if (jit_get_config()->specialized_opcodes) {
        switch (specialized_opcode) {
#ifdef BINARY_OP_ADD_INT
            case BINARY_OP_ADD_INT:
            case BINARY_OP_MULTIPLY_INT:
            case BINARY_OP_SUBTRACT_INT: {
                HirType t_long = hir_type_from_pytype(&PyLong_Type, 1);
                phx_tc_emit(tc, hir_c_create_guard_type_reg(left, t_long, left));
                phx_tc_emit(tc, hir_c_create_guard_type_reg(right, t_long, right));
                break;
            }
            case BINARY_OP_ADD_FLOAT:
            case BINARY_OP_MULTIPLY_FLOAT:
            case BINARY_OP_SUBTRACT_FLOAT: {
                HirType t_float = hir_type_from_pytype(&PyFloat_Type, 1);
                phx_tc_emit(tc, hir_c_create_guard_type_reg(left, t_float, left));
                phx_tc_emit(tc, hir_c_create_guard_type_reg(right, t_float, right));
                break;
            }
            case BINARY_OP_ADD_UNICODE: {
                HirType t_unicode = hir_type_from_pytype(&PyUnicode_Type, 1);
                phx_tc_emit(tc, hir_c_create_guard_type_reg(left, t_unicode, left));
                phx_tc_emit(tc, hir_c_create_guard_type_reg(right, t_unicode, right));
                break;
            }
#endif
#ifdef BINARY_SUBSCR_DICT
            case BINARY_SUBSCR_DICT: {
                HirType t_dict = hir_type_from_pytype(&PyDict_Type, 1);
                phx_tc_emit(tc, hir_c_create_guard_type_reg(left, t_dict, left));
                break;
            }
            case BINARY_SUBSCR_LIST_INT: {
                HirType t_list = hir_type_from_pytype(&PyList_Type, 1);
                HirType t_long = hir_type_from_pytype(&PyLong_Type, 1);
                phx_tc_emit(tc, hir_c_create_guard_type_reg(left, t_list, left));
                phx_tc_emit(tc, hir_c_create_guard_type_reg(right, t_long, right));
                break;
            }
            case BINARY_SUBSCR_TUPLE_INT: {
                HirType t_tuple = hir_type_from_pytype(&PyTuple_Type, 1);
                HirType t_long = hir_type_from_pytype(&PyLong_Type, 1);
                phx_tc_emit(tc, hir_c_create_guard_type_reg(left, t_tuple, left));
                phx_tc_emit(tc, hir_c_create_guard_type_reg(right, t_long, right));
                break;
            }
#endif
            default:
                break;
        }
    }

    int32_t op_kind;
    if (opcode == BINARY_OP) {
        op_kind = get_binary_op_kind_from_oparg_c(oparg);
        if (op_kind < 0) {
            int32_t inplace_kind = get_inplace_op_kind_from_oparg_c(oparg);
            if (inplace_kind >= 0) {
                phx_tc_emit(tc, hir_c_create_in_place_op_reg(result, inplace_kind, left, right, &tc->frame));
                phx_ptr_arr_push(stack, result);
                return 1;
            }
            return 0;
        }
    } else {
        /* Legacy BINARY_* opcodes — map opcode to op_kind */
        switch (opcode) {
#ifdef BINARY_ADD
            case BINARY_ADD:            op_kind = HIR_BOP_Add; break;
            case BINARY_AND:            op_kind = HIR_BOP_And; break;
            case BINARY_FLOOR_DIVIDE:   op_kind = HIR_BOP_FloorDivide; break;
            case BINARY_LSHIFT:         op_kind = HIR_BOP_LShift; break;
            case BINARY_MATRIX_MULTIPLY: op_kind = HIR_BOP_MatrixMultiply; break;
            case BINARY_MODULO:         op_kind = HIR_BOP_Modulo; break;
            case BINARY_MULTIPLY:       op_kind = HIR_BOP_Multiply; break;
            case BINARY_OR:             op_kind = HIR_BOP_Or; break;
            case BINARY_POWER:          op_kind = HIR_BOP_Power; break;
            case BINARY_RSHIFT:         op_kind = HIR_BOP_RShift; break;
            case BINARY_SUBTRACT:       op_kind = HIR_BOP_Subtract; break;
            case BINARY_TRUE_DIVIDE:    op_kind = HIR_BOP_TrueDivide; break;
            case BINARY_XOR:            op_kind = HIR_BOP_Xor; break;
#endif
            case BINARY_SUBSCR:         op_kind = HIR_BOP_Subscript; break;
            default:                    return 0;
        }
    }

    phx_tc_emit(tc, hir_c_create_binary_op_reg(result, op_kind, left, right, &tc->frame));
    phx_ptr_arr_push(stack, result);
    return 1;
}

/* emitMatchKeys — match keys, branch on None result */

void hir_builder_emit_match_keys_c(PhxTranslationContext *tc, void *func) {
    PhxPtrArray *stack = &tc->frame.stack;
    void *keys = stack->data[stack->count - 1];
    void *subject = stack->data[stack->count - 2];

    void *values_or_none = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_match_keys_reg(values_or_none, subject, keys, &tc->frame));
    phx_ptr_arr_push(stack, values_or_none);

    void *none = hir_func_alloc_register(func);
    HirType t_none = hir_type_from_object(Py_None);
    phx_tc_emit(tc, hir_c_create_load_const(none, t_none));
    void *is_none = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_primitive_compare(is_none, HIR_PCMP_Equal, values_or_none, none));

    void *true_block = hir_cfg_alloc_block(func);
    void *false_block = hir_cfg_alloc_block(func);
    void *done = hir_cfg_alloc_block(func);

    phx_tc_emit(tc, hir_c_create_cond_branch_cpp(is_none, true_block, false_block));

#if PY_VERSION_HEX < 0x030C0000
    void *if_success = hir_func_alloc_register(func);
#endif

    tc->block = true_block;
    HirType t_nonetype = (HirType)HIR_TYPE_NONETYPE;
    phx_tc_emit(tc, hir_c_create_refine_type_reg(values_or_none, t_nonetype, values_or_none));
#if PY_VERSION_HEX < 0x030C0000
    HirType t_false = hir_type_from_object(Py_False);
    phx_tc_emit(tc, hir_c_create_load_const(if_success, t_false));
#endif
    phx_tc_emit(tc, hir_c_create_branch_cpp(done));

    tc->block = false_block;
    HirType t_tuple = hir_type_from_pytype(&PyTuple_Type, 1);
    phx_tc_emit(tc, hir_c_create_refine_type_reg(values_or_none, t_tuple, values_or_none));
#if PY_VERSION_HEX < 0x030C0000
    HirType t_true = hir_type_from_object(Py_True);
    phx_tc_emit(tc, hir_c_create_load_const(if_success, t_true));
#endif
    phx_tc_emit(tc, hir_c_create_branch_cpp(done));

#if PY_VERSION_HEX < 0x030C0000
    phx_ptr_arr_push(stack, if_success);
#endif
    tc->block = done;
}

/* emitLoadField — load field from receiver using preloader fieldInfo */
extern int hir_type_could_be(const HirType *a, const HirType *b);

void hir_builder_emit_load_field_c(PhxTranslationContext *tc, void *func, void *builder,
                                    PyCodeObject *code, int oparg) {
    PyObject *descr = PyTuple_GET_ITEM(code->co_consts, oparg);
    intptr_t offset;
    HirType type;
    PyObject *name;
    hir_builder_preloader_field_info(builder, descr, &offset, &type, &name);

    void *receiver = phx_ptr_arr_pop(&tc->frame.stack);
    void *result = hir_func_alloc_register(func);
    const char *field_name = PyUnicode_AsUTF8(name);
    if (field_name == NULL) {
        PyErr_Clear();
        field_name = "";
    }
    phx_tc_emit(tc, hir_c_create_load_field_reg(result, receiver, field_name, offset, type, 0));
    HirType h_null = (HirType)HIR_TYPE_NULLPTR;
    if (hir_type_could_be(&type, &h_null)) {
        void *cf = hir_c_create_check_field_reg(result, result, name, &tc->frame);
        hir_c_set_guilty_reg(cf, receiver);
        phx_tc_emit(tc, cf);
    }
    phx_ptr_arr_push(&tc->frame.stack, result);
}

/* emitSetupAsyncWith — pop top, setup finally, push top back */
extern void hir_builder_emit_setup_finally_c(PhxTranslationContext *tc, int handler_off);
void hir_builder_emit_setup_async_with_c(PhxTranslationContext *tc, int handler_off) {
    void *top = phx_ptr_arr_pop(&tc->frame.stack);
    hir_builder_emit_setup_finally_c(tc, handler_off);
    phx_ptr_arr_push(&tc->frame.stack, top);
}

/* emitLoadBuildClass — load __build_class__ from builtins */

void hir_builder_emit_load_build_class_c(PhxTranslationContext *tc, void *func) {
    void *result = hir_func_alloc_register(func);
    void *builtins_reg = hir_func_alloc_register(func);
    void *key_reg = hir_func_alloc_register(func);

    HirType t_builtins = hir_type_from_object((PyObject *)tc->frame.builtins);
    phx_tc_emit(tc, hir_c_create_load_const(builtins_reg, t_builtins));

    void *builtins_dict = hir_func_alloc_register(func);
    HirType t_dict = hir_type_from_pytype(&PyDict_Type, 1);
    void *guard = hir_c_create_guard_type_reg(builtins_dict, t_dict, builtins_reg);
    hir_deopt_set_frame_state(guard, &tc->frame);
    phx_tc_emit(tc, guard);

    static PyObject *build_class_str = NULL;
    if (build_class_str == NULL) {
        build_class_str = PyUnicode_InternFromString("__build_class__");
    }
    HirType t_key = hir_type_from_object(build_class_str);
    phx_tc_emit(tc, hir_c_create_load_const(key_reg, t_key));

    phx_tc_emit(tc, hir_c_create_dict_subscr_reg(result, builtins_dict, key_reg, &tc->frame));
    phx_ptr_arr_push(&tc->frame.stack, result);
}

/* emitStoreGlobal — store to global variable via dict */

void hir_builder_emit_store_global_c(PhxTranslationContext *tc, void *func,
                                      PyCodeObject *code, int oparg) {
    void *globals_reg = hir_func_alloc_register(func);
    void *key_reg = hir_func_alloc_register(func);

    HirType t_globals = hir_type_from_object((PyObject *)tc->frame.globals);
    phx_tc_emit(tc, hir_c_create_load_const(globals_reg, t_globals));

    void *globals_dict = hir_func_alloc_register(func);
    HirType t_dict = hir_type_from_pytype(&PyDict_Type, 1);
    void *guard = hir_c_create_guard_type_reg(globals_dict, t_dict, globals_reg);
    hir_deopt_set_frame_state(guard, &tc->frame);
    phx_tc_emit(tc, guard);

    PyObject *name = PyTuple_GET_ITEM(code->co_names, oparg);
    HirType t_key = hir_type_from_object(name);
    phx_tc_emit(tc, hir_c_create_load_const(key_reg, t_key));

    void *value = phx_ptr_arr_pop(&tc->frame.stack);
    void *result = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_set_dict_item_reg(result, globals_dict, key_reg, value, &tc->frame));
}

/* emitLoadGlobal — load global variable with optional fast path */
extern PyObject *hir_builder_preloader_global(void *builder, int name_idx);
extern void *hir_builder_preloader_globals(void *builder);
extern void *hir_builder_preloader_builtins(void *builder);

void hir_builder_emit_load_global_c(
        PhxTranslationContext *tc,
        void *func,
        void *builder,
        PyCodeObject *code,
        int opcode,
        int oparg) {
    int name_idx;
#if PY_VERSION_HEX >= 0x030B0000
    name_idx = oparg >> 1;
#else
    name_idx = oparg;
#endif

#if PY_VERSION_HEX >= 0x030B0000 && PY_VERSION_HEX < 0x030E0000
    if (oparg & 1) {
        hir_builder_emit_push_null_c(tc, func);
    }
#endif

    void *result = hir_func_alloc_register(func);

    int fast_path_used = 0;
    if (jit_get_config()->stable_frame) {
        PyObject *value = hir_builder_preloader_global(builder, name_idx);
        if (value != NULL) {
            void *builtins = hir_builder_preloader_builtins(builder);
            void *globals = hir_builder_preloader_globals(builder);
            phx_tc_emit(tc, hir_c_create_load_global_cached_reg(
                result, code, builtins, globals, name_idx));
            void *guard = hir_c_create_guard_is_reg(result, value, result);
            PyObject *name = PyTuple_GET_ITEM(code->co_names, name_idx);
            const char *name_str = PyUnicode_AsUTF8(name);
            hir_c_set_descr(guard, name_str);
            phx_tc_emit(tc, guard);
            fast_path_used = 1;
        }
    }

    if (!fast_path_used) {
        phx_tc_emit(tc, hir_c_create_load_global_reg(result, name_idx, &tc->frame));
    }

    phx_ptr_arr_push(&tc->frame.stack, result);

#if PY_VERSION_HEX >= 0x030E0000
    if (oparg & 1) {
        hir_builder_emit_push_null_c(tc, func);
    }
#endif
}

/* emitLoadSmallInt — 3.14+ small int constant */
void hir_builder_emit_load_small_int_c(PhxTranslationContext *tc, void *func, int oparg) {
#if PY_VERSION_HEX >= 0x030E0000
    void *tmp = hir_func_alloc_register(func);
    PyObject *obj = (PyObject *)&_PyLong_SMALL_INTS[_PY_NSMALLNEGINTS + oparg];
    HirType type = hir_type_from_object(obj);
    phx_tc_emit(tc, hir_c_create_load_const(tmp, type));
    phx_ptr_arr_push(&tc->frame.stack, tmp);
#else
    (void)tc; (void)func; (void)oparg;
#endif
}

/* emitBuildInterpolation — 3.14+ string interpolation */

void hir_builder_emit_build_interpolation_c(PhxTranslationContext *tc, void *func, int oparg) {
#if PY_VERSION_HEX >= 0x030E0000
    int conversion = oparg >> 2;
    void *format;
    if (oparg & 1) {
        format = phx_ptr_arr_pop(&tc->frame.stack);
    } else {
        PyObject *empty = &_Py_STR(empty);
        format = hir_func_alloc_register(func);
        HirType type = hir_type_from_object(empty);
        phx_tc_emit(tc, hir_c_create_load_const(format, type));
    }
    void *str = phx_ptr_arr_pop(&tc->frame.stack);
    void *value = phx_ptr_arr_pop(&tc->frame.stack);
    void *out = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_build_interpolation_reg(out, value, str, format, conversion, &tc->frame));
    phx_ptr_arr_push(&tc->frame.stack, out);
#else
    (void)tc; (void)func; (void)oparg;
#endif
}

/* emitLoadAssertionError — load AssertionError as constant */

void hir_builder_emit_load_assertion_error_c(PhxTranslationContext *tc, void *func) {
    void *result = hir_func_alloc_register(func);
    PyObject *ref = hir_func_add_reference(func, PyExc_AssertionError);
    HirType type = hir_type_from_object(ref);
    phx_tc_emit(tc, hir_c_create_load_const(result, type));
    phx_ptr_arr_push(&tc->frame.stack, result);
}

/* emitRefineType — refine stack top to preloaded type */
extern HirType hir_builder_preloader_type(void *builder, PyObject *descr);

void hir_builder_emit_refine_type_c(PhxTranslationContext *tc, void *builder,
                                     PyCodeObject *code, int oparg) {
    PyObject *descr = PyTuple_GET_ITEM(code->co_consts, oparg);
    HirType type = hir_builder_preloader_type(builder, descr);
    void *dst = tc->frame.stack.data[tc->frame.stack.count - 1];
    phx_tc_emit(tc, hir_c_create_refine_type_reg(dst, type, dst));
}

/* emitLoadClass — load preloaded class as constant */
extern void *hir_builder_preloader_py_type(void *builder, PyObject *descr);
void hir_builder_emit_load_class_c(PhxTranslationContext *tc, void *func, void *builder,
                                    PyCodeObject *code, int oparg) {
    PyObject *descr = PyTuple_GET_ITEM(code->co_consts, oparg);
    void *pytype = hir_builder_preloader_py_type(builder, descr);
    void *tmp = hir_func_alloc_register(func);
    HirType type = hir_type_from_object((PyObject *)pytype);
    phx_tc_emit(tc, hir_c_create_load_const(tmp, type));
    phx_ptr_arr_push(&tc->frame.stack, tmp);
}

/* emitCast — cast value to preloaded type */
extern void *hir_builder_preloader_preloaded_type(void *builder, PyObject *descr, int *optional_out, int *exact_out);

void hir_builder_emit_cast_c(PhxTranslationContext *tc, void *func, void *builder,
                              PyCodeObject *code, int oparg) {
    PyObject *descr = PyTuple_GET_ITEM(code->co_consts, oparg);
    int optional, exact;
    void *pytype = hir_builder_preloader_preloaded_type(builder, descr, &optional, &exact);
    void *value = phx_ptr_arr_pop(&tc->frame.stack);
    void *result = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_cast_reg(result, value, pytype, optional, exact, &tc->frame));
    phx_ptr_arr_push(&tc->frame.stack, result);
}

/* emitTpAlloc — allocate object from type (via preloader) */

void hir_builder_emit_tp_alloc_c(PhxTranslationContext *tc, void *func, void *builder,
                                  PyCodeObject *code, int oparg) {
    PyObject *descr = PyTuple_GET_ITEM(code->co_consts, oparg);
    void *pytype = hir_builder_preloader_py_type(builder, descr);
    void *result = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_tp_alloc_reg(result, pytype, &tc->frame));
    phx_ptr_arr_push(&tc->frame.stack, result);
}

/* emitImportName — pop fromlist+level, emit ImportName or EagerImportName */

void hir_builder_emit_import_name_c(PhxTranslationContext *tc, void *func, int opcode, int oparg) {
    void *fromlist = phx_ptr_arr_pop(&tc->frame.stack);
    void *level = phx_ptr_arr_pop(&tc->frame.stack);
    void *res = hir_func_alloc_register(func);
#ifdef EAGER_IMPORT_NAME
    if (opcode == EAGER_IMPORT_NAME) {
        phx_tc_emit(tc, hir_c_create_eager_import_name_reg(res, oparg, fromlist, level, &tc->frame));
    } else
#endif
    {
        int name_idx = oparg;
#if PY_VERSION_HEX >= 0x030F0000
        name_idx = oparg >> 2;
#endif
        phx_tc_emit(tc, hir_c_create_import_name_reg(res, name_idx, fromlist, level, &tc->frame));
    }
    phx_ptr_arr_push(&tc->frame.stack, res);
}

/* emitCallEx — pop kwargs/args/func, emit CallEx */

void hir_builder_emit_call_ex_c(PhxTranslationContext *tc, void *func, int oparg, uint32_t flags) {
    void *dst = hir_func_alloc_register(func);
    PhxPtrArray *stack = &tc->frame.stack;
    int has_kwargs = (PY_VERSION_HEX >= 0x030E0000) || (oparg & 0x1);
    void *kwargs;
    if (has_kwargs) {
        kwargs = phx_ptr_arr_pop(stack);
        flags |= 0x1; /* CallFlags::KwArgs = 1 */
    } else {
        kwargs = hir_func_alloc_register(func);
        HirType t_nullptr = (HirType)HIR_TYPE_NULLPTR;
        phx_tc_emit(tc, hir_c_create_load_const(kwargs, t_nullptr));
    }
    void *pargs = phx_ptr_arr_pop(stack);
    void *func_r;
#if PY_VERSION_HEX >= 0x030E0000
    phx_ptr_arr_pop(stack); /* unused */
    func_r = phx_ptr_arr_pop(stack);
#elif PY_VERSION_HEX >= 0x030C0000
    func_r = phx_ptr_arr_pop(stack);
    phx_ptr_arr_pop(stack); /* unused */
#else
    func_r = phx_ptr_arr_pop(stack);
#endif
    phx_tc_emit(tc, hir_c_create_call_ex_reg(dst, func_r, pargs, kwargs, flags, &tc->frame));
    phx_ptr_arr_push(stack, dst);
}

/* emitGetIter — pop iterable, GetIter, push iter; FOR_ITER specialisation guard */
extern PyTypeObject *jit_g_range_iterator_type;
extern PyTypeObject *jit_g_list_iterator_type;
extern PyTypeObject *jit_g_tuple_iterator_type;

void hir_builder_emit_get_iter_c(
        PhxTranslationContext *tc, void *func,
        int next_specialized_opcode, int next_instr_baseoff) {
    void *iterable = phx_ptr_arr_pop(&tc->frame.stack);
    void *result = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_get_iter_reg(result, iterable, &tc->frame));

    if (jit_get_config()->specialized_opcodes) {
        /* The Snapshot/GuardType FrameState must reflect interpreter state AT
         * FOR_ITER (not GET_ITER): if the type guard fails, the interpreter
         * must resume at FOR_ITER with the iterator on the stack. */
        phx_ptr_arr_push(&tc->frame.stack, result);
        int32_t saved_offs = tc->frame.cur_instr_offs;
        tc->frame.cur_instr_offs = next_instr_baseoff;
        phx_tc_emit(tc, hir_c_create_snapshot(&tc->frame));
        if (next_specialized_opcode == FOR_ITER_RANGE &&
                jit_g_range_iterator_type != NULL) {
            HirType t = hir_type_from_pytype(jit_g_range_iterator_type, 1);
            phx_tc_emit(tc, hir_c_create_guard_type_fs_reg(result, t, result, &tc->frame));
        } else if (next_specialized_opcode == FOR_ITER_LIST &&
                jit_g_list_iterator_type != NULL) {
            HirType t = hir_type_from_pytype(jit_g_list_iterator_type, 1);
            phx_tc_emit(tc, hir_c_create_guard_type_fs_reg(result, t, result, &tc->frame));
        } else if (next_specialized_opcode == FOR_ITER_TUPLE &&
                jit_g_tuple_iterator_type != NULL) {
            HirType t = hir_type_from_pytype(jit_g_tuple_iterator_type, 1);
            phx_tc_emit(tc, hir_c_create_guard_type_fs_reg(result, t, result, &tc->frame));
        }
        tc->frame.cur_instr_offs = saved_offs;
        /* result already pushed above — do NOT push again */
    } else {
        phx_ptr_arr_push(&tc->frame.stack, result);
    }

#if PY_VERSION_HEX >= 0x030F0000
    hir_builder_emit_push_null_c(tc, func);
#endif
}

/* emitForIter — peek iterator, InvokeIterNext, CondBranchIterNotDone */
extern void *hir_builder_get_block_at_off(void *builder, int byte_offset);

void hir_builder_emit_for_iter_c(
        PhxTranslationContext *tc,
        void *func,
        void *builder,
        int jump_target,
        int next_instr_offset) {
#if PY_VERSION_HEX >= 0x030F0000
    void *iterator = tc->frame.stack.data[tc->frame.stack.count - 2];
#else
    void *iterator = tc->frame.stack.data[tc->frame.stack.count - 1];
#endif
    void *next_val = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_invoke_iter_next_reg(next_val, iterator, &tc->frame));
    phx_ptr_arr_push(&tc->frame.stack, next_val);
    void *footer = hir_builder_get_block_at_off(builder, jump_target);
    void *body = hir_builder_get_block_at_off(builder, next_instr_offset);
    phx_tc_emit(tc, hir_c_create_cond_branch_iter_not_done_cpp(next_val, body, footer));
}

/* emitUnpackEx — unpack with star: seq → tuple, load items */

void hir_builder_emit_unpack_ex_c(PhxTranslationContext *tc, void *func, int oparg) {
    int arg_before = oparg & 0xff;
    int arg_after = oparg >> 8;
    void *seq = phx_ptr_arr_pop(&tc->frame.stack);
    void *tuple = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_unpack_ex_to_tuple_reg(tuple, seq, arg_before, arg_after, &tc->frame));
    int total_args = arg_before + arg_after + 1;
    for (int i = total_args - 1; i >= 0; i--) {
        void *item = hir_func_alloc_register(func);
        phx_tc_emit(tc, hir_c_create_load_tuple_item_reg(item, tuple, i));
        phx_ptr_arr_push(&tc->frame.stack, item);
    }
}

/* emitFastLen — type-dispatch LoadField with optional inexact deopt path */

#ifndef FAST_LEN_LIST
#define FAST_LEN_LIST     0
#define FAST_LEN_TUPLE    1
#define FAST_LEN_ARRAY    2
#define FAST_LEN_DICT     3
#define FAST_LEN_SET      4
#define FAST_LEN_STR      5
#define FAST_LEN_INEXACT  0x80
#endif

void hir_builder_emit_fast_len_c(
        PhxTranslationContext *tc,
        void *func,
        int oparg,
        int bc_offset) {
    void *result = hir_func_alloc_register(func);
    int inexact = oparg & FAST_LEN_INEXACT;
    oparg &= ~FAST_LEN_INEXACT;
    intptr_t offset = 0;
    HirType type = {0};
    const char *name = "";

    if (oparg == FAST_LEN_LIST) {
        type = hir_type_from_pytype(&PyList_Type, 1);
        offset = offsetof(PyVarObject, ob_size);
        name = "ob_size";
    } else if (oparg == FAST_LEN_TUPLE) {
        type = hir_type_from_pytype(&PyTuple_Type, 1);
        offset = offsetof(PyVarObject, ob_size);
        name = "ob_size";
    } else if (oparg == FAST_LEN_ARRAY) {
        type = (HirType)HIR_TYPE_ARRAY;
        offset = offsetof(PyVarObject, ob_size);
        name = "ob_size";
    } else if (oparg == FAST_LEN_DICT) {
        type = hir_type_from_pytype(&PyDict_Type, 1);
        offset = offsetof(PyDictObject, ma_used);
        name = "ma_used";
    } else if (oparg == FAST_LEN_SET) {
        type = hir_type_from_pytype(&PySet_Type, 1);
        offset = offsetof(PySetObject, used);
        name = "used";
    } else if (oparg == FAST_LEN_STR) {
        type = hir_type_from_pytype(&PyUnicode_Type, 1);
        offset = offsetof(PyASCIIObject, length);
        name = "length";
    }

    void *collection;
    if (inexact) {
        void *deopt_block = hir_cfg_alloc_block(func);
        PhxTranslationContext deopt_tc;
        deopt_tc.block = deopt_block;
        phx_frame_state_copy(&deopt_tc.frame, &tc->frame);
        deopt_tc.frame.cur_instr_offs = bc_offset;
        phx_tc_emit(&deopt_tc, hir_c_create_snapshot(&deopt_tc.frame));
        phx_tc_emit(&deopt_tc, hir_c_create_deopt());
        phx_frame_state_destroy(&deopt_tc.frame);

        collection = phx_ptr_arr_pop(&tc->frame.stack);
        void *fast_path = hir_cfg_alloc_block(func);
        phx_tc_emit(tc, hir_c_create_cond_branch_check_type_cpp(collection, type, fast_path, deopt_block));
        tc->block = fast_path;
        phx_tc_emit(tc, hir_c_create_refine_type_reg(collection, type, collection));
    } else {
        collection = phx_ptr_arr_pop(&tc->frame.stack);
    }

    HirType t_cint64 = (HirType)HIR_TYPE_CINT64;
    phx_tc_emit(tc, hir_c_create_load_field_reg(result, collection, name, offset, t_cint64, 0));
    phx_ptr_arr_push(&tc->frame.stack, result);
}

/* ---- Tier 2 bridge externs (used by multiple emit methods) ---- */
extern void *hir_builder_get_block_at_off(void *builder, int byte_offset);

/* emitJumpIf — peek var, IsTruthy or direct, conditional branch */
void hir_builder_emit_jump_if_c(
        PhxTranslationContext *tc,
        void *func,
        void *builder,
        int opcode,
        int jump_target,
        int next_instr_offset) {
    void *var = tc->frame.stack.data[tc->frame.stack.count - 1];
    int true_off, false_off;
    int check_truthy = 1;
    switch (opcode) {
#ifdef JUMP_IF_NONZERO_OR_POP
        case JUMP_IF_NONZERO_OR_POP:
            check_truthy = 0;
            /* fallthrough */
#endif
#ifdef JUMP_IF_TRUE_OR_POP
        case JUMP_IF_TRUE_OR_POP:
#endif
            true_off = jump_target;
            false_off = next_instr_offset;
            break;
#ifdef JUMP_IF_ZERO_OR_POP
        case JUMP_IF_ZERO_OR_POP:
            check_truthy = 0;
            /* fallthrough */
#endif
#ifdef JUMP_IF_FALSE_OR_POP
        case JUMP_IF_FALSE_OR_POP:
#endif
            false_off = jump_target;
            true_off = next_instr_offset;
            break;
        default:
            true_off = jump_target;
            false_off = next_instr_offset;
            break;
    }
    void *true_block = hir_builder_get_block_at_off(builder, true_off);
    void *false_block = hir_builder_get_block_at_off(builder, false_off);
    if (check_truthy) {
        void *tval = hir_func_alloc_register(func);
        phx_tc_emit(tc, hir_c_create_is_truthy_reg(tval, var, &tc->frame));
        phx_tc_emit(tc, hir_c_create_cond_branch_cpp(tval, true_block, false_block));
    } else {
        phx_tc_emit(tc, hir_c_create_cond_branch_cpp(var, true_block, false_block));
    }
}

/* emitDictMerge — peek dict+func at depth, pop update, emit DictMerge */

void hir_builder_emit_dict_merge_c(PhxTranslationContext *tc, void *func_reg, int oparg) {
    PhxPtrArray *stack = &tc->frame.stack;
#if PY_VERSION_HEX < 0x030E0000
    void *dict = stack->data[stack->count - oparg - 1];
    void *callable = stack->data[stack->count - oparg - 3];
#else
    void *dict = stack->data[stack->count - 2];
    void *callable = stack->data[stack->count - 5];
#endif
    void *update = phx_ptr_arr_pop(stack);
    void *out = hir_func_alloc_register(func_reg);
    phx_tc_emit(tc, hir_c_create_dict_merge_reg(out, dict, update, callable, &tc->frame));
}

/* emitMakeListTuple — allocate list/tuple, fill operands from stack */

void hir_builder_emit_make_list_tuple_c(PhxTranslationContext *tc, void *func, int opcode, int oparg) {
    size_t num_elems = (size_t)oparg;
    void *dst = hir_func_alloc_register(func);
    void *instr;
    if (opcode == BUILD_TUPLE) {
        instr = hir_c_create_make_tuple_reg(num_elems, dst, &tc->frame);
    } else {
        instr = hir_c_create_make_list_reg(num_elems, dst, &tc->frame);
    }
    for (size_t i = num_elems; i > 0; i--) {
        void *opnd = phx_ptr_arr_pop(&tc->frame.stack);
        hir_c_set_operand(instr, i - 1, opnd);
    }
    phx_tc_emit(tc, instr);
    phx_ptr_arr_push(&tc->frame.stack, dst);
}

/* emitBuildMap — allocate dict, fill key/value pairs from stack */

void hir_builder_emit_build_map_c(PhxTranslationContext *tc, void *func, int oparg) {
    int dict_size = oparg;
    void *dict = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_make_dict_reg(dict, dict_size, &tc->frame));
    PhxPtrArray *stack = &tc->frame.stack;
    for (size_t i = stack->count - dict_size * 2, end = stack->count; i < end; i += 2) {
        void *key = stack->data[i];
        void *value = stack->data[i + 1];
        void *result = hir_func_alloc_register(func);
        phx_tc_emit(tc, hir_c_create_set_dict_item_reg(result, dict, key, value, &tc->frame));
    }
    stack->count -= (dict_size * 2);
    phx_ptr_arr_push(stack, dict);
}

/* emitBuildSet — allocate set, add items from stack */

void hir_builder_emit_build_set_c(PhxTranslationContext *tc, void *func, int oparg) {
    void *set = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_make_set_reg(set, &tc->frame));
    PhxPtrArray *stack = &tc->frame.stack;
    for (int i = oparg; i > 0; i--) {
        void *item = stack->data[stack->count - i];
        void *result = hir_func_alloc_register(func);
        phx_tc_emit(tc, hir_c_create_set_set_item_reg(result, set, item, &tc->frame));
    }
    stack->count -= oparg;
    phx_ptr_arr_push(stack, set);
}

/* emitBuildConstKeyMap — allocate dict, pop keys tuple, fill from stack */

void hir_builder_emit_build_const_key_map_c(PhxTranslationContext *tc, void *func, int oparg) {
    int dict_size = oparg;
    void *dict = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_make_dict_reg(dict, dict_size, &tc->frame));
    PhxPtrArray *stack = &tc->frame.stack;
    void *keys = phx_ptr_arr_pop(stack);
    for (int i = 0; i < dict_size; i++) {
        void *key = hir_func_alloc_register(func);
        phx_tc_emit(tc, hir_c_create_load_tuple_item_reg(key, keys, i));
        void *value = stack->data[stack->count - dict_size + i];
        void *result = hir_func_alloc_register(func);
        phx_tc_emit(tc, hir_c_create_set_dict_item_reg(result, dict, key, value, &tc->frame));
    }
    stack->count -= dict_size;
    phx_ptr_arr_push(stack, dict);
}

/* emitPopJumpIf — pop var, compute true/false blocks, conditional branch */

void hir_builder_emit_pop_jump_if_c(
        PhxTranslationContext *tc,
        void *func,
        void *builder,
        int opcode,
        int jump_target,
        int next_instr_offset) {
    void *var = phx_ptr_arr_pop(&tc->frame.stack);
    int true_off, false_off;
    switch (opcode) {
#ifdef POP_JUMP_IF_ZERO
        case POP_JUMP_IF_ZERO:
#endif
        case POP_JUMP_IF_FALSE:
            true_off = next_instr_offset;
            false_off = jump_target;
            break;
#ifdef POP_JUMP_IF_NONZERO
        case POP_JUMP_IF_NONZERO:
#endif
        case POP_JUMP_IF_TRUE:
            true_off = jump_target;
            false_off = next_instr_offset;
            break;
        default:
            true_off = next_instr_offset;
            false_off = jump_target;
            break;
    }
    void *true_block = hir_builder_get_block_at_off(builder, true_off);
    void *false_block = hir_builder_get_block_at_off(builder, false_off);

    if (opcode == POP_JUMP_IF_FALSE || opcode == POP_JUMP_IF_TRUE) {
        void *is_true = hir_func_alloc_register(func);
#if PY_VERSION_HEX >= 0x030E0000
        void *const_true = hir_func_alloc_register(func);
        HirType t_true = hir_type_from_object(Py_True);
        phx_tc_emit(tc, hir_c_create_load_const(const_true, t_true));
        phx_tc_emit(tc, hir_c_create_primitive_compare(is_true, HIR_PCMP_Equal, var, const_true));
#else
        phx_tc_emit(tc, hir_c_create_is_truthy_reg(is_true, var, &tc->frame));
#endif
        phx_tc_emit(tc, hir_c_create_cond_branch_cpp(is_true, true_block, false_block));
    } else {
        phx_tc_emit(tc, hir_c_create_cond_branch_cpp(var, true_block, false_block));
    }
}

/* emitPopJumpIfNone — pop var, compare to None, conditional branch */
void hir_builder_emit_pop_jump_if_none_c(
        PhxTranslationContext *tc,
        void *func,
        void *builder,
        int opcode,
        int jump_target,
        int next_instr_offset) {
    void *var = phx_ptr_arr_pop(&tc->frame.stack);
    void *true_block = hir_builder_get_block_at_off(builder, jump_target);
    void *false_block = hir_builder_get_block_at_off(builder, next_instr_offset);

    void *none = hir_func_alloc_register(func);
    HirType t_none = hir_type_from_object(Py_None);
    phx_tc_emit(tc, hir_c_create_load_const(none, t_none));
    void *is_true = hir_func_alloc_register(func);
    int32_t op = (opcode == POP_JUMP_IF_NONE) ? HIR_PCMP_Equal : HIR_PCMP_NotEqual;
    phx_tc_emit(tc, hir_c_create_primitive_compare(is_true, op, var, none));
    phx_tc_emit(tc, hir_c_create_cond_branch_cpp(is_true, true_block, false_block));
}

/* emitSetupFinally — push block_stack entry */
void hir_builder_emit_setup_finally_c(PhxTranslationContext *tc, int handler_off) {
    int stack_level = (int)tc->frame.stack.count;
    phx_block_stack_push(&tc->frame, SETUP_FINALLY, handler_off, stack_level);
}

/* emitCallIntrinsic — pop 1-2 args, emit CallIntrinsic, push result */

void hir_builder_emit_call_intrinsic_c(PhxTranslationContext *tc, void *func,
                                        int opcode, int oparg) {
    void *value = phx_ptr_arr_pop(&tc->frame.stack);
    void *res = hir_func_alloc_register(func);
    void *args[2];
    int num_operands = 1;
#if PY_VERSION_HEX >= 0x030C0000
    if (opcode == CALL_INTRINSIC_2) {
        void *value2 = phx_ptr_arr_pop(&tc->frame.stack);
        args[0] = value2;
        args[1] = value;
        num_operands = 2;
    } else {
        args[0] = value;
    }
#else
    args[0] = value;
#endif
    phx_tc_emit(tc, hir_c_create_call_intrinsic_reg2(num_operands, res, oparg, (HirRegister *)args));
    phx_ptr_arr_push(&tc->frame.stack, res);
}

/* emitSetFunctionAttribute — pop func+value, map oparg→FunctionAttr, emit */

/* FunctionAttr enum values (must match C++ enum class FunctionAttr in hir.h) */
#define FUNC_ATTR_CLOSURE     0
#define FUNC_ATTR_ANNOTATIONS 1
#define FUNC_ATTR_KWDEFAULTS  2
#define FUNC_ATTR_DEFAULTS    3
#define FUNC_ATTR_ANNOTATE    4

/* MAKE_FUNCTION_* constants from py-portability.h */
#ifndef MAKE_FUNCTION_DEFAULTS
#define MAKE_FUNCTION_DEFAULTS    0x01
#define MAKE_FUNCTION_KWDEFAULTS  0x02
#define MAKE_FUNCTION_ANNOTATIONS 0x04
#define MAKE_FUNCTION_CLOSURE     0x08
#endif
#ifndef MAKE_FUNCTION_ANNOTATE
#define MAKE_FUNCTION_ANNOTATE    0x10
#endif

void hir_builder_emit_set_function_attribute_c(PhxTranslationContext *tc, int oparg) {
    void *func = phx_ptr_arr_pop(&tc->frame.stack);
    void *value = phx_ptr_arr_pop(&tc->frame.stack);
    int32_t attr;
    switch (oparg) {
        case MAKE_FUNCTION_DEFAULTS:    attr = FUNC_ATTR_DEFAULTS; break;
        case MAKE_FUNCTION_KWDEFAULTS:  attr = FUNC_ATTR_KWDEFAULTS; break;
        case MAKE_FUNCTION_ANNOTATIONS: attr = FUNC_ATTR_ANNOTATIONS; break;
        case MAKE_FUNCTION_CLOSURE:     attr = FUNC_ATTR_CLOSURE; break;
        case MAKE_FUNCTION_ANNOTATE:    attr = FUNC_ATTR_ANNOTATE; break;
        default: attr = -1; break;
    }
    phx_tc_emit(tc, hir_c_create_set_function_attr_reg(value, func, attr));
    phx_ptr_arr_push(&tc->frame.stack, func);
}

/* emitGetAIter — pop obj, emit GetAIter, push */

void hir_builder_emit_get_aiter_c(PhxTranslationContext *tc, void *func) {
    void *obj = phx_ptr_arr_pop(&tc->frame.stack);
    void *out = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_get_a_iter_reg(out, obj, &tc->frame));
    phx_ptr_arr_push(&tc->frame.stack, out);
}

/* emitGetANext — peek obj, emit GetANext, push */

void hir_builder_emit_get_anext_c(PhxTranslationContext *tc, void *func) {
    void *obj = tc->frame.stack.data[tc->frame.stack.count - 1];
    void *out = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_get_a_next_reg(out, obj, &tc->frame));
    phx_ptr_arr_push(&tc->frame.stack, out);
}

/* emitBuildTemplate — pop interpolations+strings, emit BuildTemplate, push */

void hir_builder_emit_build_template_c(PhxTranslationContext *tc, void *func) {
    void *interpolations = phx_ptr_arr_pop(&tc->frame.stack);
    void *strings = phx_ptr_arr_pop(&tc->frame.stack);
    void *out = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_build_template_reg(strings, interpolations, out, &tc->frame));
    phx_ptr_arr_push(&tc->frame.stack, out);
}

/* emitFormatValue — pop fmt_spec (if present), pop value, FormatValue, push */

void hir_builder_emit_format_value_c(PhxTranslationContext *tc, void *func, int oparg) {
    int have_fmt_spec = (oparg & 0x04) == 0x04; /* FVS_HAVE_SPEC=4, FVS_MASK=4 */
    void *fmt_spec;
    if (have_fmt_spec) {
        fmt_spec = phx_ptr_arr_pop(&tc->frame.stack);
    } else {
        fmt_spec = hir_func_alloc_register(func);
        HirType t_nullptr = HIR_TYPE_NULLPTR;
        phx_tc_emit(tc, hir_c_create_load_const(fmt_spec, t_nullptr));
    }
    void *value = phx_ptr_arr_pop(&tc->frame.stack);
    void *dst = hir_func_alloc_register(func);
    int which_conversion = oparg & 0x03; /* FVC_MASK=3 */
    phx_tc_emit(tc, hir_c_create_format_value_reg(dst, fmt_spec, value, which_conversion, &tc->frame));
    phx_ptr_arr_push(&tc->frame.stack, dst);
}

/* emitListExtend — pop iterable, peek list at depth, emit ListExtend */

void hir_builder_emit_list_extend_c(PhxTranslationContext *tc, void *func, int oparg) {
    void *iterable = phx_ptr_arr_pop(&tc->frame.stack);
    void *list = tc->frame.stack.data[tc->frame.stack.count - oparg];
    void *none = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_list_extend_reg(none, list, iterable, &tc->frame));
}

/* emitListToTuple — pop list, emit MakeTupleFromList, push tuple */

void hir_builder_emit_list_to_tuple_c(PhxTranslationContext *tc, void *func) {
    void *list = phx_ptr_arr_pop(&tc->frame.stack);
    void *tuple = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_make_tuple_from_list_reg(tuple, list, &tc->frame));
    phx_ptr_arr_push(&tc->frame.stack, tuple);
}

/* emitMakeCell — allocate cell, MakeCell + moveOverwritten + Assign */

void hir_builder_emit_make_cell_c(PhxTranslationContext *tc, void *func, int local_idx) {
    void *local = tc->frame.localsplus.data[local_idx];
    void *cell = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_make_cell_reg(cell, local, &tc->frame));
    phx_move_overwritten_stack_regs(tc, func, local);
    phx_tc_emit(tc, hir_assign_create(local, cell));
}

/* emitRaiseVarargs — emit Raise */

void hir_builder_emit_raise_varargs_c(PhxTranslationContext *tc) {
    phx_tc_emit(tc, hir_c_create_raise_reg(&tc->frame));
}

/* emitListAppend — pop item, peek list at depth, emit ListAppend */

void hir_builder_emit_list_append_c(PhxTranslationContext *tc, void *func, int oparg) {
    void *item = phx_ptr_arr_pop(&tc->frame.stack);
    void *list = tc->frame.stack.data[tc->frame.stack.count - oparg];
    void *dst = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_list_append_reg(dst, list, item, &tc->frame));
}

/* emitLoadLocal — co_consts index lookup, localsplus read, push */
void hir_builder_emit_load_local_c(
        PhxTranslationContext *tc,
        PyCodeObject *code,
        int oparg) {
    PyObject *index_and_descr = PyTuple_GET_ITEM(code->co_consts, oparg);
    int index = (int)PyLong_AsLong(PyTuple_GET_ITEM(index_and_descr, 0));
    void *var = tc->frame.localsplus.data[index];
    phx_ptr_arr_push(&tc->frame.stack, var);
}

/* emitToBool — pop, IsTruthy, PrimitiveBoxBool, push */

void hir_builder_emit_to_bool_c(PhxTranslationContext *tc, void *func) {
    void *operand = phx_ptr_arr_pop(&tc->frame.stack);
    void *truthy_result = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_is_truthy_reg(truthy_result, operand, &tc->frame));
    void *coerced_result = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_primitive_box_bool(coerced_result, truthy_result));
    phx_ptr_arr_push(&tc->frame.stack, coerced_result);
}

/* emitInPlaceOp — pop right+left, map opcode→kind, emit InPlaceOp, push */

static int32_t get_inplace_op_kind_from_opcode_c(int opcode) {
    switch (opcode) {
#ifdef INPLACE_ADD
        case INPLACE_ADD:            return HIR_IOP_Add;
        case INPLACE_AND:            return HIR_IOP_And;
        case INPLACE_FLOOR_DIVIDE:   return HIR_IOP_FloorDivide;
        case INPLACE_LSHIFT:         return HIR_IOP_LShift;
        case INPLACE_MATRIX_MULTIPLY: return HIR_IOP_MatrixMultiply;
        case INPLACE_MODULO:         return HIR_IOP_Modulo;
        case INPLACE_MULTIPLY:       return HIR_IOP_Multiply;
        case INPLACE_OR:             return HIR_IOP_Or;
        case INPLACE_POWER:          return HIR_IOP_Power;
        case INPLACE_RSHIFT:         return HIR_IOP_RShift;
        case INPLACE_SUBTRACT:       return HIR_IOP_Subtract;
        case INPLACE_TRUE_DIVIDE:    return HIR_IOP_TrueDivide;
        case INPLACE_XOR:            return HIR_IOP_Xor;
#endif
        default:                     return -1;
    }
}

void hir_builder_emit_in_place_op_c(
        PhxTranslationContext *tc,
        void *func,
        int opcode) {
    void *right = phx_ptr_arr_pop(&tc->frame.stack);
    void *left = phx_ptr_arr_pop(&tc->frame.stack);
    void *result = hir_func_alloc_register(func);
    int32_t op_kind = get_inplace_op_kind_from_opcode_c(opcode);
    phx_tc_emit(tc, hir_c_create_in_place_op_reg(result, op_kind, left, right, &tc->frame));
    phx_ptr_arr_push(&tc->frame.stack, result);
}

/* emitCompareOp — specialized guard types, compare, optional ToBool */

void hir_builder_emit_compare_op_c(
        PhxTranslationContext *tc,
        void *func,
        int oparg,
        int specialized_opcode) {
    int compare_op = oparg;
#if PY_VERSION_HEX >= 0x030E0000
    compare_op >>= 5;
#elif PY_VERSION_HEX >= 0x030B0000
    compare_op >>= 4;
#endif

    PhxPtrArray *stack = &tc->frame.stack;

    if (jit_get_config()->specialized_opcodes) {
        phx_tc_emit(tc, hir_c_create_snapshot(&tc->frame));
    }

    void *right = phx_ptr_arr_pop(stack);
    void *left = phx_ptr_arr_pop(stack);
    void *result = hir_func_alloc_register(func);

    if (jit_get_config()->specialized_opcodes) {
        switch (specialized_opcode) {
            case COMPARE_OP_FLOAT: {
                HirType t_float = hir_type_from_pytype(&PyFloat_Type, 1);
                phx_tc_emit(tc, hir_c_create_guard_type_reg(left, t_float, left));
                phx_tc_emit(tc, hir_c_create_guard_type_reg(right, t_float, right));
                break;
            }
            case COMPARE_OP_INT: {
                HirType t_long = hir_type_from_pytype(&PyLong_Type, 1);
                phx_tc_emit(tc, hir_c_create_guard_type_reg(left, t_long, left));
                phx_tc_emit(tc, hir_c_create_guard_type_reg(right, t_long, right));
                break;
            }
            case COMPARE_OP_STR: {
                HirType t_unicode = hir_type_from_pytype(&PyUnicode_Type, 1);
                phx_tc_emit(tc, hir_c_create_guard_type_reg(left, t_unicode, left));
                phx_tc_emit(tc, hir_c_create_guard_type_reg(right, t_unicode, right));
                break;
            }
            default:
                break;
        }
    }

    phx_tc_emit(tc, hir_c_create_compare_reg(result, (int32_t)compare_op, left, right, &tc->frame));
    phx_ptr_arr_push(stack, result);

#if PY_VERSION_HEX >= 0x030E0000
    if (oparg & 0x10) {
        hir_builder_emit_to_bool_c(tc, func);
    }
#endif
}

/* emitYieldValue — YIELD_VALUE handler. Pops the yielded value, handles the
 * async-generator wrapper case, then dispatches to YieldFrom or YieldValue
 * based on the next bytecode (RESUME with oparg >= 2 indicates yield-from /
 * await per _PyGen_yf semantics). Mirrors C++ HIRBuilder::emitYieldValue
 * @ builder.cpp:5137.
 *
 * The C++ stub at builder.cpp:5137 does the next_bc peek + co_flags read up
 * front (BytecodeInstruction::nextInstr + code_->co_flags require Python.h
 * structures + the BC iterator type) and passes precomputed scalars in.
 *
 * Async-gen wrap path uses the emitChecked<CallCFunc> equivalent helper
 * phx_tc_emit_checked_call_cfunc_1 below. kCix_PyAsyncGenValueWrapperNew = 0
 * per hir.h:992-996 X-macro on PY_VERSION_HEX >= 0x030C0000.
 *
 * NOTE (separate concern, see chat 11:29:12Z): instr_effects_c.c:485-487
 * comment-vs-position mismatch on this enum is filed for follow-up. */

static void phx_tc_emit_checked_call_cfunc_1(
        PhxTranslationContext *tc, void *dst, int32_t func_enum, void *operand) {
    void *operands[1] = { operand };
    phx_tc_emit(tc, hir_c_create_call_cfunc_reg(1, dst, func_enum, (HirRegister *)operands));
    void *chk = hir_c_create_check_exc_reg(dst, dst);
    hir_deopt_set_frame_state(chk, &tc->frame);
    phx_tc_emit(tc, chk);
}

void hir_builder_emit_yield_value_c(
        PhxTranslationContext *tc,
        void *builder,
        int code_flags,
        int next_bc_opcode,
        int next_bc_oparg) {
    void *in = phx_ptr_arr_pop(&tc->frame.stack);
    void *out = hir_builder_temps_alloc_stack(builder);
    if (code_flags & CO_ASYNC_GENERATOR) {
        phx_tc_emit_checked_call_cfunc_1(
            tc, out, /*kCix_PyAsyncGenValueWrapperNew=*/0, in);
        in = out;
        out = hir_builder_temps_alloc_stack(builder);
    }
    if (next_bc_opcode == RESUME && next_bc_oparg >= 2) {
        /* yield-from path: peek (do not pop) the iter at TOS. */
        void *iter = tc->frame.stack.data[tc->frame.stack.count - 1];
        phx_tc_emit(tc, hir_c_create_yield_from_reg(out, in, iter, &tc->frame));
    } else {
        phx_tc_emit(tc, hir_c_create_yield_value_reg(out, in, &tc->frame));
    }
    phx_ptr_arr_push(&tc->frame.stack, out);
}

/* emitLoadMethodOrAttrSuper — LOAD_SUPER_ATTR / LOAD_SUPER_METHOD handler.
 * Multi-block: builds a deopt path (taken if the type-check on the receiver
 * type fails) plus a fast path (RefineType + LoadAttrSuper or
 * LoadMethodSuper). Mirrors C++ HIRBuilder::emitLoadMethodOrAttrSuper @
 * builder.cpp:3882.
 *
 * Theologian audit 11:58:52Z invariants enforced inline:
 *   #1 phx_frame_state_copy MUST run BEFORE the 3 pops so deopt-path frame
 *      preserves PRE-POP stack (3 values present) for interpreter resumption
 *   #3 load_method param to the C++ method is OVERWRITTEN by oparg & 1 in
 *      3.11+; we recompute from oparg directly in this C body
 *   #4 pop order: receiver (TOS), type, global_super (3rd from top)
 *   #5 push order varies by load_method (1 value vs 2 with method_instance)
 *   #8 deopt-path emits (snapshot + deopt) go into deopt_tc.block
 *   #9 fast_path = AllocateBlock; CondBranchCheckType branches on type;
 *      tc.block <- fast_path; emitRefineType in-place narrows
 *
 * Pre-3.11 oparg-tuple branch dropped per 3.12-only project (consistent
 * with emitYieldValue precedent). frame_state_destroy MUST run on both
 * exit paths (theologian pitfall — early return on !load_method needs it). */

void hir_builder_emit_load_method_or_attr_super_c(
        PhxTranslationContext *tc,
        void *func,
        void *builder,
        int oparg,
        int bc_offset) {
    /* (#1) Copy deopt_tc.frame BEFORE pops — preserves pre-pop stack. */
    void *deopt_block = hir_cfg_alloc_block(func);
    PhxTranslationContext deopt_tc;
    deopt_tc.block = deopt_block;
    phx_frame_state_copy(&deopt_tc.frame, &tc->frame);
    deopt_tc.frame.cur_instr_offs = bc_offset;

    /* (#3) 3.11+ oparg packing — load_method recomputed from oparg. */
    int name_idx = oparg >> 2;
    int load_method = oparg & 1;
    int no_args_in_super_call = !(oparg & 2);

    /* (#4) Pop receiver, type, global_super (in that order). */
    void *receiver = phx_ptr_arr_pop(&tc->frame.stack);
    void *type = phx_ptr_arr_pop(&tc->frame.stack);
    void *global_super = phx_ptr_arr_pop(&tc->frame.stack);
    void *result = hir_builder_temps_alloc_stack(builder);

    /* (#8) deopt-path emits go into deopt_tc.block. */
    phx_tc_emit(&deopt_tc, hir_c_create_snapshot(&deopt_tc.frame));
    phx_tc_emit(&deopt_tc, hir_c_create_deopt());

    /* (#9) Fast-path branch + in-place RefineType (corner-case alias). */
    void *fast_path = hir_cfg_alloc_block(func);
    HirType t_type = HIR_TYPE_TYPE;
    phx_tc_emit(tc, hir_c_create_cond_branch_check_type_cpp(
        type, t_type, fast_path, deopt_block));
    tc->block = fast_path;
    phx_tc_emit(tc, hir_c_create_refine_type_reg(type, t_type, type));

    if (!load_method) {
        phx_tc_emit(tc, hir_c_create_load_attr_super_reg(
            result, global_super, type, receiver,
            (int32_t)name_idx, no_args_in_super_call, &tc->frame));
        phx_ptr_arr_push(&tc->frame.stack, result);
        phx_frame_state_destroy(&deopt_tc.frame);
        return;
    }

    /* (#5) load_method=true: alloc method_instance, emit LoadMethodSuper,
     * extract second output. Order matters: LoadMethodSuper FIRST, then
     * GetSecondOutput which depends on LoadMethodSuper's result reg. */
    void *method_instance = hir_builder_temps_alloc_stack(builder);
    phx_tc_emit(tc, hir_c_create_load_method_super_reg(
        result, global_super, type, receiver,
        (int32_t)name_idx, no_args_in_super_call, &tc->frame));
    HirType t_opt_object = HIR_TYPE_OPTOBJECT;
    phx_tc_emit(tc, hir_c_create_get_second_output_reg(
        method_instance, t_opt_object, result));
    phx_ptr_arr_push(&tc->frame.stack, result);
    phx_ptr_arr_push(&tc->frame.stack, method_instance);
    phx_frame_state_destroy(&deopt_tc.frame);
}

/* emitYieldFrom — YIELD_FROM helper invoked from the YIELD_FROM bytecode
 * dispatch (and from emitGetAwaitable). Pop send_value, peek iter at TOS
 * (do not pop yet), optionally emit SetCurrentAwaiter when the enclosing
 * code object has CO_COROUTINE, emit YieldFrom, THEN pop iter and push
 * out. Mirrors C++ HIRBuilder::emitYieldFrom @ builder.cpp:5085.
 *
 * out is allocated by the CALLER (e.g. dispatch site at builder.cpp:2519
 * passes temps_.AllocateStack()) so this C function takes it as a
 * parameter rather than allocating itself.
 *
 * The peek-then-emit-then-pop ordering for iter matters: emitYieldFrom
 * uses iter as an operand, but the YIELD_FROM bytecode semantics consume
 * it from the stack only once the yield resolves. The C++ original
 * matches this; we mirror exactly. */

void hir_builder_emit_yield_from_method_c(
        PhxTranslationContext *tc,
        void *out,
        int code_flags) {
    void *send_value = phx_ptr_arr_pop(&tc->frame.stack);
    void *iter = tc->frame.stack.data[tc->frame.stack.count - 1];
    if (code_flags & CO_COROUTINE) {
        phx_tc_emit(tc, hir_c_create_set_current_awaiter_reg(iter));
    }
    phx_tc_emit(tc, hir_c_create_yield_from_reg(out, send_value, iter, &tc->frame));
    phx_ptr_arr_pop(&tc->frame.stack);  /* now pop iter (after emit captured operand) */
    phx_ptr_arr_push(&tc->frame.stack, out);
}

/* emitInvokeMethodVectorCall — VectorCall + fixStaticReturn dispatch for a
 * static method invoke. Mirrors C++ HIRBuilder::emitInvokeMethodVectorCall
 * @ builder.cpp:3561.
 *
 * out is allocated by the C++ stub (per emitYieldFrom precedent — caller-
 * provides). target.return_type is extracted to HirType in the stub via
 * Type::toHirType. arg_regs comes through as void** + count (decay from
 * std::vector<Register*>::data()).
 *
 * CallFlags::Awaited = 1<<1 = 2 per hir.h:881-886. None = 0.
 *
 * fixStaticReturn invoked via the C-callable wrapper added at builder.cpp
 * (lite bridge spec 12:42:11Z, theologian ACK 12:42:26Z). */
extern void hir_builder_fix_static_return_c(
    void *builder, void *tc, void *ret_val, HirType ret_type);

void hir_builder_emit_invoke_method_vector_call_c(
        PhxTranslationContext *tc,
        void *builder,
        void *out,
        void **arg_regs_data,
        size_t arg_regs_count,
        int is_awaited,
        HirType ret_type) {
    uint32_t flags = is_awaited ? 2u : 0u;  /* CallFlags::Awaited = 1<<1 */
    void *call = hir_c_create_vectorcall_reg(arg_regs_count, out, flags);
    for (size_t i = 0; i < arg_regs_count; i++) {
        hir_c_set_operand(call, i, arg_regs_data[i]);
    }
    hir_deopt_set_frame_state(call, &tc->frame);
    phx_tc_emit(tc, call);
    hir_builder_fix_static_return_c(builder, tc, out, ret_type);
    phx_ptr_arr_push(&tc->frame.stack, out);
}

/* emitInvokeFunction — INVOKE_FUNCTION opcode. Mirrors C++ HIRBuilder::
 * emitInvokeFunction @ builder.cpp:3324.
 *
 * Three paths gated on InvokeTarget fields:
 *   (A) container_is_immutable + is_function + is_statically_typed:
 *       Direct InvokeStaticFunction call. Returns false (already pushed).
 *   (B) container_is_immutable + is_builtin: tryEmitDirectMethodCall fast
 *       path. Returns false if it took.
 *   (C) Fall through: load callable indirect/direct, setup args, VectorCall,
 *       fixStaticReturn. Returns true.
 *
 * Pre-path: PY_VERSION_HEX >= 0x030C0000 conditional for __static__.rand
 * special-case (folded into hir_builder_is_static_rand_and_try_emit_c bridge
 * which does its own version-guarded behavior).
 *
 * 4 NEW bridges:
 *   - hir_builder_invoke_function_target_c (8-field InvokeTarget query)
 *   - hir_builder_try_emit_direct_method_call_for_function_c (path B)
 *   - hir_builder_setup_static_args_for_function_c (function-target variant)
 *   - hir_builder_is_static_rand_and_try_emit_c (PY_VERSION_HEX conditional)
 *
 * CallFlags::Static = 1<<2 = 4 per hir.h:881-886. */
extern void hir_builder_invoke_function_target_c(
    void *builder, PyObject *descr,
    int *out_container_is_immutable,
    int *out_is_function,
    int *out_is_statically_typed,
    int *out_is_builtin,
    void **out_callable,
    void **out_func,
    void **out_indirect_ptr,
    HirType *out_return_type);
extern int hir_builder_try_emit_direct_method_call_for_function_c(
    void *builder, void *tc, PyObject *descr, long nargs);
extern void hir_builder_setup_static_args_for_function_c(
    void *builder, void *tc, PyObject *descr, long nargs, int statically_typed,
    void **out_arg_regs, size_t *out_count);
extern int hir_builder_is_static_rand_and_try_emit_c(
    void *builder, void *tc, PyObject *descr, long nargs);

bool hir_builder_emit_invoke_function_c(
        PhxTranslationContext *tc,
        void *func,
        void *builder,
        PyObject *descr,
        long nargs,
        uint32_t flags) {
    int container_is_immutable, is_function, is_statically_typed, is_builtin;
    void *callable, *target_func, *indirect_ptr;
    HirType return_type;
    hir_builder_invoke_function_target_c(builder, descr,
        &container_is_immutable, &is_function, &is_statically_typed, &is_builtin,
        &callable, &target_func, &indirect_ptr, &return_type);

    /* PY_VERSION_HEX >= 0x030C0000 conditional folded into the bridge. */
    if (hir_builder_is_static_rand_and_try_emit_c(builder, tc, descr, nargs)) {
        return false;
    }

    void *funcreg = hir_builder_temps_alloc_stack(builder);

    if (container_is_immutable) {
        if (is_function && is_statically_typed) {
            /* Path (A): direct InvokeStaticFunction call. */
            void *out = hir_builder_temps_alloc_stack(builder);
            HirType callable_type = hir_type_from_object((PyObject *)callable);
            phx_tc_emit(tc, hir_c_create_load_const(funcreg, callable_type));

            void *call = hir_c_create_invoke_static_function_reg(
                (size_t)(nargs + 1), out, target_func, return_type);
            hir_c_set_operand(call, 0, funcreg);
            for (long i = nargs - 1; i >= 0; i--) {
                void *operand = phx_ptr_arr_pop(&tc->frame.stack);
                hir_c_set_operand(call, (size_t)(i + 1), operand);
            }
            hir_deopt_set_frame_state(call, &tc->frame);
            phx_tc_emit(tc, call);
            phx_ptr_arr_push(&tc->frame.stack, out);
            return false;
        } else if (is_builtin && hir_builder_try_emit_direct_method_call_for_function_c(
                builder, tc, descr, nargs)) {
            /* Path (B): builtin fast-path took. */
            return false;
        }
        /* Couldn't emit x64 call but know the callable; load it directly. */
        HirType callable_type = hir_type_from_object((PyObject *)callable);
        phx_tc_emit(tc, hir_c_create_load_const(funcreg, callable_type));
    } else {
        /* Patchable target: load indirect via deopt-base instr. */
        phx_tc_emit(tc, hir_c_create_load_function_indirect_reg(
            indirect_ptr, descr, funcreg, &tc->frame));
    }

    /* Path (C): VectorCall. */
    void *arg_regs[nargs];
    size_t arg_regs_count = 0;
    hir_builder_setup_static_args_for_function_c(
        builder, tc, descr, nargs, /*statically_invoked=*/0,
        arg_regs, &arg_regs_count);

    void *out = hir_builder_temps_alloc_stack(builder);
    if (container_is_immutable) {
        flags |= 4u;  /* CallFlags::Static = 1<<2 */
    }

    /* Add one for the function argument. */
    void *call = hir_c_create_vectorcall_reg((size_t)(nargs + 1), out, flags);
    for (long i = 0; i < nargs; i++) {
        hir_c_set_operand(call, (size_t)(i + 1), arg_regs[i]);
    }
    hir_c_set_operand(call, 0, funcreg);
    hir_deopt_set_frame_state(call, &tc->frame);
    phx_tc_emit(tc, call);

    hir_builder_fix_static_return_c(builder, tc, out, return_type);
    phx_ptr_arr_push(&tc->frame.stack, out);
    return true;
}

/* emitInvokeMethod — INVOKE_METHOD opcode. Mirrors C++ HIRBuilder::emitInvokeMethod
 * @ builder.cpp:3493.
 *
 * Two paths:
 *   (1) Builtin fast path: tryEmitDirectMethodCall returns 1 → swap top two
 *       stack entries (pop result, pop thunk, push result) and return false.
 *   (2) Statically typed: pop arg_regs via setupStaticArgs, pop entry from
 *       static_method_stack_, emit CallInd with entry as first operand, emit.
 *   (3) Dynamic typed: pop arg_regs, dispatch via emitInvokeMethodVectorCall
 *       (already C-bridged at builder_emit_c.c:2377).
 *
 * Returns false in path (1) — caller should not push (already pushed).
 * Returns true in paths (2)+(3) — bytecode emit fully completed.
 *
 * Theologian L2430 INVOKE_* Phase 2 spec. 4 NEW bridges:
 *   - hir_builder_invoke_method_target_c (InvokeTarget query)
 *   - hir_builder_try_emit_direct_method_call_c (path 1)
 *   - hir_builder_setup_static_args_c (arg_regs)
 *   - hir_builder_static_method_stack_pop_c (entry pop) */
extern void hir_builder_invoke_method_target_c(
    void *builder, PyObject *descr,
    int *out_is_builtin, int *out_is_statically_typed, HirType *out_return_type);
extern int hir_builder_try_emit_direct_method_call_c(
    void *builder, void *tc, PyObject *descr, long nargs);
extern void hir_builder_setup_static_args_c(
    void *builder, void *tc, PyObject *descr, long nargs, int statically_typed,
    void **out_arg_regs, size_t *out_count);
extern void *hir_builder_static_method_stack_pop_c(void *builder);

bool hir_builder_emit_invoke_method_c(
        PhxTranslationContext *tc,
        void *func,
        void *builder,
        PyObject *descr,
        long nargs,
        int is_awaited) {
    int is_builtin, is_statically_typed;
    HirType return_type;
    hir_builder_invoke_method_target_c(
        builder, descr, &is_builtin, &is_statically_typed, &return_type);

    if (is_builtin && hir_builder_try_emit_direct_method_call_c(
            builder, tc, descr, nargs - 1)) {
        void *res = phx_ptr_arr_pop(&tc->frame.stack);
        phx_ptr_arr_pop(&tc->frame.stack);  /* pop the thunk */
        phx_ptr_arr_push(&tc->frame.stack, res);
        return false;
    }

    void *arg_regs[nargs];  /* C99 VLA — sized exactly to nargs */
    size_t arg_regs_count = 0;
    hir_builder_setup_static_args_c(
        builder, tc, descr, nargs, is_statically_typed, arg_regs, &arg_regs_count);

    if (is_statically_typed) {
        /* AllocateNonStack equivalent: hir_func_alloc_register
         * (TempAllocator::AllocateNonStack just calls env->AllocateRegister). */
        void *out = hir_func_alloc_register(func);
        void *entry = hir_builder_static_method_stack_pop_c(builder);
        void *invoke = hir_c_create_call_ind_reg2(
            (size_t)nargs + 1, out, "vtable invoke", return_type);
        hir_c_set_operand(invoke, 0, entry);
        for (size_t i = 0; i < arg_regs_count; i++) {
            hir_c_set_operand(invoke, i + 1, arg_regs[i]);
        }
        hir_deopt_set_frame_state(invoke, &tc->frame);
        phx_tc_emit(tc, invoke);
        phx_ptr_arr_push(&tc->frame.stack, out);
    } else {
        /* Dynamic dispatch via existing emitInvokeMethodVectorCall bridge.
         * out is allocated INSIDE that bridge per its convention. */
        void *out = hir_builder_temps_alloc_stack(builder);
        hir_builder_emit_invoke_method_vector_call_c(
            tc, builder, out, arg_regs, arg_regs_count, is_awaited, return_type);
    }
    return true;
}

/* emitInvokeNative — INVOKE_NATIVE opcode. Mirrors C++ HIRBuilder::emitInvokeNative
 * @ builder.cpp:3405.
 *
 * NativeTarget fields (callable + return_type) extracted via the
 * hir_builder_invoke_native_target_c bridge (preloader_ access lives in
 * C++-only namespace). Theologian L2430 INVOKE_* Phase 2 spec.
 *
 * nargs derived from tuple signature size minus 1 (the last entry is the
 * return type). Operands popped right-to-left from frame.stack and assigned
 * to operand slots i = nargs-1 .. 0 to preserve evaluation order. */
extern void hir_builder_invoke_native_target_c(
    void *builder, PyObject *descr, void **out_callable, HirType *out_return_type);

void hir_builder_emit_invoke_native_c(
        PhxTranslationContext *tc,
        void *builder,
        PyObject *descr,
        PyObject *signature) {
    void *callable = NULL;
    HirType ret_type;
    hir_builder_invoke_native_target_c(builder, descr, &callable, &ret_type);

    Py_ssize_t nargs = PyTuple_GET_SIZE(signature) - 1;

    void *out = hir_builder_temps_alloc_stack(builder);
    void *call = hir_c_create_call_static_reg((size_t)nargs, out, callable, ret_type);
    for (Py_ssize_t i = nargs - 1; i >= 0; i--) {
        void *operand = phx_ptr_arr_pop(&tc->frame.stack);
        hir_c_set_operand(call, (size_t)i, operand);
    }
    phx_tc_emit(tc, call);
    phx_ptr_arr_push(&tc->frame.stack, out);
}

/* emitDispatchEagerCoroResult — eager coroutine result dispatch helper.
 * Branches on whether the awaitable's runtime type is WaitHandle. If yes,
 * loads the wait_handle's coro/result + waiter, releases the wait_handle,
 * and dispatches: coro waiter → YieldAndYieldFrom path; res waiter →
 * direct Assign path. Both converge at the caller-provided post_await_block.
 *
 * Mirrors C++ HIRBuilder::emitDispatchEagerCoroResult @ builder.cpp:5286.
 *
 * Theologian audit (chat 2026-04-22 13:24:20Z) — invariants enforced inline:
 *   (A) STACK PEEK ONLY: stack_top peeked, never popped/pushed; caller
 *       manages stack via `out` reg + post_await_block convergence
 *   (B) FRAME COPY ORIGIN: coro_block + res_block BOTH copy from tc.frame
 *       (NOT has_wh_block.frame) — invariant 8
 *   (C) yield-and-yield-from uses tc.frame (BEFORE-branch state, not coro
 *       block's copy) — invariant 11
 *   (7) WaitHandle emit ORDER: LoadCoroOrResult → LoadWaiter → Release
 *   (10) CO_COROUTINE guards ONLY emitSetCurrentAwaiter (not coro_block as a whole)
 *   (12) Both coro_block + res_block end with branch(post_await_block)
 *
 * 3 separate PhxTranslationContext instances (has_wh_block, coro_block,
 * res_block) — frame_state_destroy on ALL 3 at function end. */

void hir_builder_emit_dispatch_eager_coro_result_c(
        PhxTranslationContext *tc,
        void *func,
        void *builder,
        void *out,
        void *await_block,
        void *post_await_block,
        int code_flags) {
    /* (A,1) Peek stack_top — never pop/push in this method. */
    void *stack_top = tc->frame.stack.data[tc->frame.stack.count - 1];
    void *wait_handle = stack_top;  /* (pitfall) alias only, not a fresh temp */

    /* (4) has_wh_block: copy from tc.frame BEFORE the cond branch. */
    void *has_wh_blk = hir_cfg_alloc_block(func);
    PhxTranslationContext has_wh_tc;
    has_wh_tc.block = has_wh_blk;
    phx_frame_state_copy(&has_wh_tc.frame, &tc->frame);

    /* (5) Branch FROM tc.block on type-check: success → has_wh_blk, fail → await_block. */
    HirType t_wait_handle = HIR_TYPE_WAITHANDLE;
    phx_tc_emit(tc, hir_c_create_cond_branch_check_type_cpp(
        stack_top, t_wait_handle, has_wh_blk, await_block));

    /* (6) Allocate wh_coro_or_result + wh_waiter — both BEFORE WaitHandle emits. */
    void *wh_coro_or_result = hir_builder_temps_alloc_stack(builder);
    void *wh_waiter = hir_builder_temps_alloc_stack(builder);

    /* (7) WaitHandle ORDER: LoadCoroOrResult → LoadWaiter → Release. */
    phx_tc_emit(&has_wh_tc, hir_c_create_wait_handle_load_coro_reg(
        wh_coro_or_result, wait_handle));
    phx_tc_emit(&has_wh_tc, hir_c_create_wait_handle_load_waiter_reg(
        wh_waiter, wait_handle));
    phx_tc_emit(&has_wh_tc, hir_c_create_wait_handle_release_reg(wait_handle));

    /* (B,8) coro_block + res_block: COPY FROM tc.frame (NOT has_wh_tc.frame). */
    void *coro_blk = hir_cfg_alloc_block(func);
    PhxTranslationContext coro_tc;
    coro_tc.block = coro_blk;
    phx_frame_state_copy(&coro_tc.frame, &tc->frame);

    void *res_blk = hir_cfg_alloc_block(func);
    PhxTranslationContext res_tc;
    res_tc.block = res_blk;
    phx_frame_state_copy(&res_tc.frame, &tc->frame);

    /* (9) has_wh_block branches on wh_waiter → coro vs res. */
    phx_tc_emit(&has_wh_tc, hir_c_create_cond_branch_cpp(
        wh_waiter, coro_blk, res_blk));

    /* (10) CO_COROUTINE guards SetCurrentAwaiter ONLY (per-emit, not block-wide). */
    if (code_flags & CO_COROUTINE) {
        phx_tc_emit(&coro_tc, hir_c_create_set_current_awaiter_reg(wh_coro_or_result));
    }

    /* (C,11) yield-and-yield-from uses tc->frame (ORIGINAL, not coro_tc copy). */
    phx_tc_emit(&coro_tc, hir_c_create_yield_and_yield_from_reg(
        out, wh_waiter, wh_coro_or_result, &tc->frame));

    /* (12) Both blocks branch to post_await_block. */
    phx_tc_emit(&coro_tc, hir_c_create_branch_cpp(post_await_block));

    /* (13) res_block: assign out = wh_coro_or_result, then branch. */
    phx_tc_emit(&res_tc, hir_assign_create(out, wh_coro_or_result));
    phx_tc_emit(&res_tc, hir_c_create_branch_cpp(post_await_block));

    /* Cleanup all 3 deopt_tcs (RAII → manual). */
    phx_frame_state_destroy(&has_wh_tc.frame);
    phx_frame_state_destroy(&coro_tc.frame);
    phx_frame_state_destroy(&res_tc.frame);
}

/* emitGetAwaitable — GET_AWAITABLE handler (4 phases, 6+ blocks).
 * Most complex remaining method per theologian pre-audit (chat 2026-04-22
 * 13:52:50Z, 17 invariants).
 *
 * P1: pop iterable, alloc iter, CallCFunc(JitCoro_GetAwaitableIter)
 * P2: error_aenter/aexit dispatch (computed in C++ stub from oparg).
 *     If either set: error_block does LoadField+RaiseAwaitableError;
 *     ok_block does in-place RefineType. Else: CheckExc(iter, iter).
 * P3: coroutine-type assertion. Allocate assert_not_awaited_coro +
 *     done blocks. 3.12+: also allocate check_coro_block.
 *     CondBranchCheckType against modulestate.coroType (3.12+) →
 *     assert vs check, then check_coro vs done against PyCoro_Type.
 * P4: yf check + RaiseStatic.
 * P5: push iter, set tc.block = done_block.
 *
 * NEW LITE BRIDGE (per spec chat 2026-04-22 13:53:something):
 *   cinderx_get_module_state_coro_type_c — returns PyTypeObject* of
 *   cinderx-augmented coroutine type (singleton). Implemented in
 *   builder.cpp file scope (cinderx::getModuleState in C++ scope there).
 *
 * EXISTING bridge re-use: hir_type_from_pytype(type, 1) — second arg
 * is is_exact, value 1 makes this the C-side equivalent of
 * Type::fromTypeExact (verified via grep: type.cpp:67 uses
 * hir_type_from_pytype(type, 1) for fromTypeExact).
 *
 * CallCFunc enum integers (per hir.h:992-996 X-macro on 3.12+):
 *   kCix_PyAsyncGenValueWrapperNew = 0
 *   kJitCoro_GetAwaitableIter      = 1
 *   kJitGen_yf                     = 2
 *   kJITRT_MatchAndClearException  = 3
 * Used inline as comments below per push 51 + push 54 precedent. */
extern void *cinderx_get_module_state_coro_type_c(void);

void hir_builder_emit_get_awaitable_c(
        PhxTranslationContext *tc,
        void *func,
        void *builder,
        int error_aenter,
        int error_aexit) {
    /* P1: pop iterable, alloc iter, CallCFunc(JitCoro_GetAwaitableIter). */
    void *iterable = phx_ptr_arr_pop(&tc->frame.stack);
    void *iter = hir_builder_temps_alloc_stack(builder);
    {
        void *operands[1] = { iterable };
        phx_tc_emit(tc, hir_c_create_call_cfunc_reg(
            1, iter, /*kJitCoro_GetAwaitableIter=*/1, (HirRegister *)operands));
    }

    /* P2: error_aenter/aexit dispatch (3.12+ semantic via stub). */
    if (error_aenter || error_aexit) {
        void *error_block = hir_cfg_alloc_block(func);
        void *ok_block = hir_cfg_alloc_block(func);
        phx_tc_emit(tc, hir_c_create_cond_branch_cpp(iter, ok_block, error_block));

        /* error_block: LoadField(type=ob_type) + RaiseAwaitableError */
        tc->block = error_block;
        void *type_reg = hir_builder_temps_alloc_stack(builder);
        HirType t_type = HIR_TYPE_TYPE;
        phx_tc_emit(tc, hir_c_create_load_field_reg(
            type_reg, iterable, "ob_type",
            (intptr_t)offsetof(PyObject, ob_type), t_type, 0));
        phx_tc_emit(tc, hir_c_create_raise_awaitable_error_reg(
            type_reg, (int32_t)error_aenter, &tc->frame));

        /* ok_block: in-place RefineType(iter, TObject, iter) (SSA-rename). */
        tc->block = ok_block;
        HirType t_object = HIR_TYPE_OBJECT;
        phx_tc_emit(tc, hir_c_create_refine_type_reg(iter, t_object, iter));
    } else {
        /* Single CheckExc(iter, iter) — no block switch. */
        void *chk = hir_c_create_check_exc_reg(iter, iter);
        hir_deopt_set_frame_state(chk, &tc->frame);
        phx_tc_emit(tc, chk);
    }

    /* P3: coroutine-type assertion blocks (always alloc). */
    void *block_assert_not_awaited_coro = hir_cfg_alloc_block(func);
    void *block_done = hir_cfg_alloc_block(func);

    /* P3.5: 3.12+ check against cinderx coroType (extra block + branch). */
    {
        PyTypeObject *cinderx_coro_type =
            (PyTypeObject *)cinderx_get_module_state_coro_type_c();
        HirType t_cinderx_coro = hir_type_from_pytype(cinderx_coro_type, 1);
        void *block_check_coro = hir_cfg_alloc_block(func);
        phx_tc_emit(tc, hir_c_create_cond_branch_check_type_cpp(
            iter, t_cinderx_coro,
            block_assert_not_awaited_coro, block_check_coro));
        tc->block = block_check_coro;
    }

    /* P3.6: always check against PyCoro_Type (exact match). */
    HirType t_pycoro = hir_type_from_pytype(&PyCoro_Type, 1);
    phx_tc_emit(tc, hir_c_create_cond_branch_check_type_cpp(
        iter, t_pycoro,
        block_assert_not_awaited_coro, block_done));

    /* P4: yf check (CallCFunc kJitGen_yf=2). */
    void *yf = hir_builder_temps_alloc_stack(builder);
    tc->block = block_assert_not_awaited_coro;
    {
        void *operands[1] = { iter };
        phx_tc_emit(tc, hir_c_create_call_cfunc_reg(
            1, yf, /*kJitGen_yf=*/2, (HirRegister *)operands));
    }

    /* P4.5: cond branch on yf — coro_already_awaited vs done. */
    void *block_coro_already_awaited = hir_cfg_alloc_block(func);
    phx_tc_emit(tc, hir_c_create_cond_branch_cpp(
        yf, block_coro_already_awaited, block_done));

    /* P4.6: RaiseStatic in coro_already_awaited block. oparg=0 (NOT
     * raising var-args). */
    tc->block = block_coro_already_awaited;
    phx_tc_emit(tc, hir_c_create_raise_static_reg(
        /*reraise=*/0, (void *)PyExc_RuntimeError,
        "coroutine is being awaited already", &tc->frame));

    /* P5: push iter to stack, end at block_done.
     * Push BEFORE block switch — frame.stack is frame-state-level not
     * block-position-level (per theologian invariant 17). */
    phx_ptr_arr_push(&tc->frame.stack, iter);
    tc->block = block_done;
}

/* emitLoadMethodStatic — LOAD_METHOD_STATIC handler (Cinder static-typed
 * method dispatch). Mirrors C++ HIRBuilder::emitLoadMethodStatic @
 * builder.cpp:3600.
 *
 * Approach A1 (theologian chat 2026-04-22 14:14:19Z): zero new bridges.
 * C++ stub extracts InvokeTarget data + computes vtable byte offsets
 * (avoiding classloader.h dependency in this C TU) + post-call handles
 * static_method_stack_.push via out-param. C body emits the LoadField +
 * CallInd + GetSecondOutput chain.
 *
 * 6 phases per pre-audit:
 *   P2: pop self, alloc type, conditional LoadField (or type=self for
 *       classmethod alias)
 *   P3: alloc vtable + func_obj (NonStack via hir_func_alloc_register),
 *       LoadField vtable (tp_cache), LoadField func_obj (vte_state)
 *   P4: alloc entry_func + vtable_load (NonStack), LoadField vtable_load
 *       (vte_load), CallInd(3 ops, vte_load name, TOptObject)
 *   P5: conditional GetSecondOutput(entry_func, TCPtr, func_obj); always
 *       write *out_entry_func = entry_func
 *   P6: push func_obj THEN push self (order matters)
 *
 * out_entry_func is set to the pre-allocated entry_func register
 * unconditionally so the C++ stub can decide whether to push it onto
 * static_method_stack_ based on is_static (gates the stack push only). */

void hir_builder_emit_load_method_static_c(
        PhxTranslationContext *tc,
        void *func,
        void *builder,
        int is_classmethod,
        intptr_t vte_state_offset,
        intptr_t vte_load_offset,
        int is_static,
        void **out_entry_func) {
    HirType t_type = HIR_TYPE_TYPE;
    HirType t_object = HIR_TYPE_OBJECT;
    HirType t_cptr = HIR_TYPE_CPTR;
    HirType t_opt_object = HIR_TYPE_OPTOBJECT;

    /* P2: pop self, alloc type. */
    void *self = phx_ptr_arr_pop(&tc->frame.stack);
    void *type = hir_builder_temps_alloc_stack(builder);
    if (!is_classmethod) {
        phx_tc_emit(tc, hir_c_create_load_field_reg(
            type, self, "ob_type",
            (intptr_t)offsetof(PyObject, ob_type), t_type, 0));
    } else {
        /* Classmethod path: type aliases self. The pre-allocated `type`
         * temp is unused (matches C++ semantics — leaked-but-unused). */
        type = self;
    }

    /* P3: alloc vtable + func_obj (NonStack equivalent). */
    void *vtable = hir_func_alloc_register(func);
    void *func_obj = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_load_field_reg(
        vtable, type, "tp_cache",
        (intptr_t)offsetof(PyTypeObject, tp_cache), t_object, 0));
    phx_tc_emit(tc, hir_c_create_load_field_reg(
        func_obj, vtable, "vte_state", vte_state_offset, t_object, 0));

    /* P4: alloc entry_func + vtable_load (NonStack), CallInd. */
    void *entry_func = hir_func_alloc_register(func);
    void *vtable_load = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_load_field_reg(
        vtable_load, vtable, "vte_load", vte_load_offset, t_cptr, 0));

    void *call = hir_c_create_call_ind_reg2(3, func_obj, "vte_load", t_opt_object);
    hir_c_set_operand(call, 0, vtable_load);
    hir_c_set_operand(call, 1, func_obj);
    hir_c_set_operand(call, 2, self);
    hir_deopt_set_frame_state(call, &tc->frame);
    phx_tc_emit(tc, call);

    /* P5: conditional GetSecondOutput; always write out_entry_func. */
    if (is_static) {
        phx_tc_emit(tc, hir_c_create_get_second_output_reg(
            entry_func, t_cptr, func_obj));
    }
    *out_entry_func = entry_func;

    /* P6: push func_obj THEN self (order matters per C++ line 3660-3661). */
    phx_ptr_arr_push(&tc->frame.stack, func_obj);
    phx_ptr_arr_push(&tc->frame.stack, self);
}

/* emitUnpackSequence — UNPACK_SEQUENCE handler. Most-complex 7-phase
 * multi-block conversion remaining (122-line C++ source). Mirrors C++
 * HIRBuilder::emitUnpackSequence @ builder.cpp:4759.
 *
 * Theologian pre-audit (chat 2026-04-22 14:30:30Z): 7 phases, 22
 * invariants, 5 pitfalls. ZERO new bridges per W25-defer A1 design.
 *
 * 5 BLOCKS allocated upfront (deopt_path, fast_path_1, list_check_path,
 * list_fast_path, tuple_fast_path) + 1 RE-ALLOC (fast_path_2 inside P6
 * — distinct from fast_path_1 per pitfall PB).
 *
 * 3 mutually-exclusive type-narrowing paths (P3):
 *   isA(TTupleExact) — static narrow → Branch(tuple_fast_path)
 *   isA(TListExact)  — static narrow → Branch(list_fast_path)
 *   else             — runtime CondBranchCheckType chain
 *
 * Pitfalls addressed inline:
 *   (PA) deopt_path frame_state_copy BEFORE pop (preserves pre-pop stack
 *        with seq for interpreter resume on deopt)
 *   (PB) fast_path RE-ALLOCATED at P6 (distinct from initial alloc at
 *        P2 — semantically "fast path 1" vs "fast path 2"; named
 *        fast_path_extract_loop in C body for clarity)
 *   (PC) emitGuardType in-place SSA-rename: GuardType(seq, type, seq)
 *   (PD) Py_GIL_DISABLED branches DROPPED for 3.12 (3.13+ would need
 *        re-introduction)
 *   (PE) phx_frame_state_destroy on deopt_path TC at function end
 *
 * specialized_op = -1 means jit_get_config()->specialized_opcodes was
 * disabled (C++ stub passes -1 to skip P0). Other values are opcode
 * integers per opcode.h. */

void hir_builder_emit_unpack_sequence_c(
        PhxTranslationContext *tc,
        void *func,
        void *builder,
        int oparg,
        int bc_offset,
        int specialized_op) {
    /* P0: PEEK seq (do not pop yet — see PA). */
    void *seq = tc->frame.stack.data[tc->frame.stack.count - 1];

    /* P0 specialized-opcode dispatch (skipped if stub passed -1). */
    if (specialized_op >= 0) {
        phx_tc_emit(tc, hir_c_create_snapshot(&tc->frame));
        switch (specialized_op) {
            case UNPACK_SEQUENCE_LIST: {
                HirType t_list = HIR_TYPE_LISTEXACT;
                /* (PC) in-place SSA rename: seq is both input and output. */
                void *guard = hir_c_create_guard_type_reg(seq, t_list, seq);
                hir_deopt_set_frame_state(guard, &tc->frame);
                phx_tc_emit(tc, guard);
                break;
            }
            case UNPACK_SEQUENCE_TUPLE:
            case UNPACK_SEQUENCE_TWO_TUPLE: {
                HirType t_tuple = HIR_TYPE_TUPLEEXACT;
                void *guard = hir_c_create_guard_type_reg(seq, t_tuple, seq);
                hir_deopt_set_frame_state(guard, &tc->frame);
                phx_tc_emit(tc, guard);
                break;
            }
            default:
                break;
        }
    }

    /* P1: deopt_path setup. (PA) frame_state_copy BEFORE pop. */
    void *deopt_block = hir_cfg_alloc_block(func);
    PhxTranslationContext deopt_tc;
    deopt_tc.block = deopt_block;
    phx_frame_state_copy(&deopt_tc.frame, &tc->frame);
    deopt_tc.frame.cur_instr_offs = bc_offset;
    phx_tc_emit(&deopt_tc, hir_c_create_snapshot(&deopt_tc.frame));
    void *deopt_instr = hir_c_create_deopt();
    phx_tc_emit(&deopt_tc, deopt_instr);
    hir_c_set_guilty_reg(deopt_instr, seq);
    hir_c_set_descr(deopt_instr, "UNPACK_SEQUENCE");

    /* P2: block allocations + pop. */
    void *fast_path = hir_cfg_alloc_block(func);
    void *list_check_path = hir_cfg_alloc_block(func);
    void *list_fast_path = hir_cfg_alloc_block(func);
    void *tuple_fast_path = hir_cfg_alloc_block(func);
    void *list_mem = hir_builder_temps_alloc_stack(builder);
    /* Now pop seq from main tc.stack (deopt_tc.frame already preserved
     * the pre-pop state per PA). */
    phx_ptr_arr_pop(&tc->frame.stack);

    /* P3: 3 mutually-exclusive type-narrowing paths. */
    HirType seq_type = hir_register_type(seq);
    HirType t_tuple_exact = HIR_TYPE_TUPLEEXACT;
    HirType t_list_exact = HIR_TYPE_LISTEXACT;
    if (hir_type_is_subtype(seq_type, t_tuple_exact)) {
        phx_tc_emit(tc, hir_c_create_branch_cpp(tuple_fast_path));
    } else if (hir_type_is_subtype(seq_type, t_list_exact)) {
        /* (PD) Py_GIL_DISABLED branch dropped for 3.12 (3.13+ would
         * Branch(deopt_block) instead). */
        phx_tc_emit(tc, hir_c_create_branch_cpp(list_fast_path));
    } else {
        phx_tc_emit(tc, hir_c_create_cond_branch_check_type_cpp(
            seq, t_tuple_exact, tuple_fast_path, list_check_path));
        tc->block = list_check_path;
        /* (PD) same: 3.12 always uses list_fast_path. */
        phx_tc_emit(tc, hir_c_create_cond_branch_check_type_cpp(
            seq, t_list_exact, list_fast_path, deopt_block));
    }

    /* P4: tuple_fast_path emission. */
    tc->block = tuple_fast_path;
    {
        void *offset_reg = hir_builder_temps_alloc_stack(builder);
        HirType t_cint64 = (HirType)HIR_TYPE_CINT64;
        HirType offset_type = hir_type_from_cint(
            (int64_t)offsetof(PyTupleObject, ob_item), t_cint64);
        phx_tc_emit(tc, hir_c_create_load_const(offset_reg, offset_type));
        phx_tc_emit(tc, hir_c_create_load_field_address_reg(
            list_mem, seq, offset_reg));
        phx_tc_emit(tc, hir_c_create_branch_cpp(fast_path));
    }

    /* P5: list_fast_path emission. */
    tc->block = list_fast_path;
    {
        HirType t_cptr = (HirType)HIR_TYPE_CPTR;
        phx_tc_emit(tc, hir_c_create_load_field_reg(
            list_mem, seq, "ob_item",
            (intptr_t)offsetof(PyListObject, ob_item), t_cptr, 0));
        phx_tc_emit(tc, hir_c_create_branch_cpp(fast_path));
    }

    /* P6: fast_path size check. */
    tc->block = fast_path;
    {
        void *seq_size = hir_builder_temps_alloc_stack(builder);
        void *target_size = hir_builder_temps_alloc_stack(builder);
        void *is_equal = hir_builder_temps_alloc_stack(builder);
        HirType t_cint64 = (HirType)HIR_TYPE_CINT64;
        phx_tc_emit(tc, hir_c_create_load_var_object_size_reg(seq_size, seq));
        phx_tc_emit(tc, hir_c_create_load_const(
            target_size, hir_type_from_cint((int64_t)oparg, t_cint64)));
        phx_tc_emit(tc, hir_c_create_primitive_compare(
            is_equal, /*HIR_PCMP_Equal=*/2, seq_size, target_size));
        /* (PB) RE-ALLOC fast_path — distinct block from initial. */
        void *fast_path_extract_loop = hir_cfg_alloc_block(func);
        phx_tc_emit(tc, hir_c_create_cond_branch_cpp(
            is_equal, fast_path_extract_loop, deopt_block));

        /* P7: item extraction loop. */
        tc->block = fast_path_extract_loop;
        void *idx_reg = hir_builder_temps_alloc_stack(builder);
        HirType t_object = HIR_TYPE_OBJECT;
        for (int idx = oparg - 1; idx >= 0; --idx) {
            void *item = hir_builder_temps_alloc_stack(builder);
            phx_tc_emit(tc, hir_c_create_load_const(
                idx_reg, hir_type_from_cint((int64_t)idx, t_cint64)));
            phx_tc_emit(tc, hir_c_create_load_array_item_reg(
                item, list_mem, idx_reg, seq, /*offset=*/0, t_object));
            phx_ptr_arr_push(&tc->frame.stack, item);
        }
    }

    /* (PE) frame_state_destroy on deopt_path TC. */
    phx_frame_state_destroy(&deopt_tc.frame);
}

/* emitInlineExceptionMatch — BINARY_SUBSCR exception-match inline handler.
 * Largest most-complex method in remaining set (185-line C++ source +
 * inline bytecode dispatch loop). Mirrors C++ HIRBuilder::
 * emitInlineExceptionMatch @ builder.cpp:1329.
 *
 * Theologian pre-audit (chat 2026-04-22 14:48:10Z): 5 phases, 27
 * invariants, 8 pitfalls. ZERO new bridges per W25-defer A1. Approach:
 * stub-heavy bytecode-parse — C++ stub iterates info.except_body
 * bytecodes and builds an opcode array (5 fields per entry). C body
 * does framework + dispatch via switch on the array.
 *
 * CRITICAL D-1774910012 INVARIANT (librarian chat L1769 + scribe L1771):
 * After match_tc allocation, push Py_None as prev_exc placeholder
 * BEFORE the dispatch loop; POP_EXCEPT case pops the placeholder. If
 * skipped, closure LOAD_DEREF in except blocks corrupts exc_info chain
 * post-threshold. Failure mode is silent + only-after-1000-calls.
 *
 * 3 separate PhxTranslationContext objects (PB pitfall):
 *   exc_tc copies from tc.frame
 *   match_tc copies from EXC_TC.FRAME (depth-trimmed)
 *   deopt_tc copies from TC.FRAME (original, NOT exc_tc)
 *
 * 8 pitfalls addressed inline. See chat L1782 for full enumeration. */

/* Stub-built opcode array entry (C struct mirrors C++ struct in builder.cpp).
 * Layout MUST match — both sides verify via static_assert on size. */
typedef struct {
    int opcode;
    int oparg;
    int base_offset;
    void *const_obj;          /* PyObject* for LOAD_CONST/RETURN_CONST, else NULL */
    void *jump_target_block;  /* HirBasicBlock for JUMP_BACKWARD*, else NULL */
} OpcodeArrayEntry;

/* hir_c_create_call_static_reg already declared at line ~720 with correct
 * 4-arg signature (size_t n_operands, void *dst, void *addr, HirType
 * ret_type). Per hir_c_api.h:305 canonical decl. NOT redeclaring here. */
extern HirType hir_builder_preloader_return_type(void *builder);
extern void hir_builder_emit_swap_c(PhxTranslationContext *tc, int oparg);
extern void hir_builder_emit_load_fast_c(
    PhxTranslationContext *tc, void *func, PyCodeObject *code,
    int opcode, int oparg);
extern void hir_builder_emit_store_fast_c(
    PhxTranslationContext *tc, void *func, int oparg);
extern int hir_builder_emit_binary_op_c(
    PhxTranslationContext *tc, void *func,
    int opcode, int oparg, int specialized_opcode);

/* Opcode constants — defined in opcode.h (already included by this TU).
 * Listed inline as comments for grep + audit-trail per theologian L1782 PD. */
/* POP_EXCEPT, POP_TOP, SWAP, LOAD_FAST, LOAD_FAST_CHECK, LOAD_FAST_AND_CLEAR,
 * LOAD_CONST, STORE_FAST, BINARY_OP, RETURN_CONST, RETURN_VALUE,
 * JUMP_BACKWARD, JUMP_BACKWARD_NO_INTERRUPT — handled cases. Default → deopt. */

/* Shared helper — emits the exc_match → match_dispatch → deopt sequence
 * shared by emitInlineExceptionMatch and emitCallExceptionHandler.
 *
 * Caller pre-conditions:
 *   - tc points to the OUTER block + frame (will be source for exc_tc/deopt_tc copy)
 *   - exc_match_block already allocated and reachable via CondBranch from caller
 *
 * Caller post-conditions (NOT done here):
 *   - tc->block is NOT updated — caller sets to ok_block + emits P5
 *
 * Helper guarantees:
 *   - PA D-1774910012 invariant: Py_None placeholder pushed before dispatch,
 *     POP_EXCEPT case pops it. SHARED across both callers (single source of truth).
 *   - PB invariants: exc_tc.frame ← tc.frame; match_tc.frame ← exc_tc.frame;
 *     deopt_tc.frame ← tc.frame (NOT exc_tc.frame).
 *   - PE invariant: phx_frame_state_destroy on EXACTLY 3 TCs.
 *
 * deopt_left_repush + deopt_right_repush: optional registers to push onto
 * deopt_tc.frame.stack BEFORE Snapshot. Pass NULL to skip. Used by
 * emitInlineExceptionMatch (re-push left+right for BINARY_SUBSCR offset);
 * emitCallExceptionHandler passes NULL,NULL (call-result already popped). */
static void emit_except_match_body_c(
        PhxTranslationContext *tc,
        void *func,
        void *builder,
        int bc_base_offset,
        int handler_depth,
        void *exc_match_block,
        void *exc_type_obj,
        int except_body_offset,
        HirType return_type,
        void *match_and_clear_fn,
        const OpcodeArrayEntry *opcodes,
        size_t opcode_count,
        void *deopt_left_repush,
        void *deopt_right_repush) {
    HirType t_cint32 = (HirType)HIR_TYPE_CINT32;
    HirType t_nonetype = HIR_TYPE_NONETYPE;

    /* P2: exc_tc setup — copy from tc.frame, depth-trim. */
    PhxTranslationContext exc_tc;
    exc_tc.block = exc_match_block;
    phx_frame_state_copy(&exc_tc.frame, &tc->frame);
    /* Depth trim (P2 inv 7): pop excess items above handler depth. */
    while ((int)exc_tc.frame.stack.count > handler_depth) {
        phx_ptr_arr_pop(&exc_tc.frame.stack);
    }

    /* P2 inv 8-9: exc_type_reg + LoadConst. */
    void *exc_type_reg = hir_func_alloc_register(func);
    HirType exc_type_hir = hir_type_from_object((PyObject *)exc_type_obj);
    phx_tc_emit(&exc_tc, hir_c_create_load_const(exc_type_reg, exc_type_hir));

    /* P2 inv 10-11: match_result + CallStatic JITRT_MatchAndClearException.
     * Function pointer passed from C++ stub via match_and_clear_fn — avoids
     * C-side referencing a C++-mangled symbol from jit_rt.cpp. */
    void *match_result = hir_func_alloc_register(func);
    void *match_call = hir_c_create_call_static_reg(
        1, match_result, match_and_clear_fn, t_cint32);
    hir_c_set_operand(match_call, 0, exc_type_reg);
    phx_tc_emit(&exc_tc, match_call);

    /* P2 inv 12-13: match_block + deopt_block + CondBranch on match_result. */
    void *match_block = hir_cfg_alloc_block(func);
    void *deopt_block = hir_cfg_alloc_block(func);
    phx_tc_emit(&exc_tc, hir_c_create_cond_branch_cpp(
        match_result, match_block, deopt_block));

    /* P3: match_tc setup — copy from EXC_TC.FRAME (depth-trimmed). */
    PhxTranslationContext match_tc;
    match_tc.block = match_block;
    phx_frame_state_copy(&match_tc.frame, &exc_tc.frame);
    match_tc.frame.cur_instr_offs = except_body_offset;

    /* (PA D-1774910012) Push Py_None placeholder BEFORE dispatch loop.
     * POP_EXCEPT case pops this placeholder. If skipped, closure
     * LOAD_DEREF in except blocks corrupts exc_info chain post-threshold.
     * THIS IS THE SHARED-HELPER PAYOFF: invariant lives in EXACTLY ONE
     * place across both callers. */
    void *prev_exc_reg = hir_builder_temps_alloc_stack(builder);
    phx_tc_emit(&match_tc, hir_c_create_load_const(prev_exc_reg, t_nonetype));
    phx_ptr_arr_push(&match_tc.frame.stack, prev_exc_reg);

    /* P3 dispatch loop: switch on opcode array. */
    for (size_t i = 0; i < opcode_count; i++) {
        const OpcodeArrayEntry *e = &opcodes[i];
        switch (e->opcode) {
            case POP_EXCEPT: {
                /* (PA) Pop the prev_exc placeholder. */
                phx_ptr_arr_pop(&match_tc.frame.stack);
                break;
            }
            case POP_TOP: {
                phx_ptr_arr_pop(&match_tc.frame.stack);
                break;
            }
            case SWAP: {
                hir_builder_emit_swap_c(&match_tc, e->oparg);
                break;
            }
            case LOAD_FAST:
            case LOAD_FAST_CHECK:
            case LOAD_FAST_AND_CLEAR: {
                hir_builder_emit_load_fast_c(&match_tc, func,
                    (PyCodeObject *)match_tc.frame.code,
                    e->opcode, e->oparg);
                break;
            }
            case LOAD_CONST: {
                void *reg = hir_builder_temps_alloc_stack(builder);
                HirType t = hir_type_from_object((PyObject *)e->const_obj);
                phx_tc_emit(&match_tc, hir_c_create_load_const(reg, t));
                phx_ptr_arr_push(&match_tc.frame.stack, reg);
                break;
            }
            case STORE_FAST: {
                hir_builder_emit_store_fast_c(&match_tc, func, e->oparg);
                break;
            }
            case BINARY_OP: {
                hir_builder_emit_binary_op_c(
                    &match_tc, func, e->opcode, e->oparg, /*specialized=*/-1);
                break;
            }
            case RETURN_CONST: {
                void *ret_reg = hir_builder_temps_alloc_stack(builder);
                HirType t = hir_type_from_object((PyObject *)e->const_obj);
                phx_tc_emit(&match_tc, hir_c_create_load_const(ret_reg, t));
                phx_tc_emit(&match_tc, hir_c_create_return(ret_reg, t));
                /* Terminator — exit loop. */
                goto match_loop_done;
            }
            case RETURN_VALUE: {
                void *ret_val = phx_ptr_arr_pop(&match_tc.frame.stack);
                phx_tc_emit(&match_tc, hir_c_create_return(ret_val, return_type));
                goto match_loop_done;
            }
            case JUMP_BACKWARD:
            case JUMP_BACKWARD_NO_INTERRUPT: {
                phx_tc_emit(&match_tc, hir_c_create_branch_cpp(e->jump_target_block));
                goto match_loop_done;
            }
            default: {
                /* Unsupported opcode → deopt to interpreter. */
                match_tc.frame.cur_instr_offs = e->base_offset;
                phx_tc_emit(&match_tc, hir_c_create_snapshot(&match_tc.frame));
                phx_tc_emit(&match_tc, hir_c_create_deopt());
                goto match_loop_done;
            }
        }
    }
match_loop_done:;

    /* P4: deopt_tc setup — copy from TC.FRAME (NOT exc_tc per PB.b). */
    PhxTranslationContext deopt_tc;
    deopt_tc.block = deopt_block;
    phx_frame_state_copy(&deopt_tc.frame, &tc->frame);
    /* Optional re-push for callers whose deopt offset expects extra
     * stack items (e.g. inline-match re-pushes left+right for
     * BINARY_SUBSCR offset; call-handler skips since the call-result
     * was already popped on the C++ side). */
    if (deopt_left_repush != NULL) {
        phx_ptr_arr_push(&deopt_tc.frame.stack, deopt_left_repush);
    }
    if (deopt_right_repush != NULL) {
        phx_ptr_arr_push(&deopt_tc.frame.stack, deopt_right_repush);
    }
    deopt_tc.frame.cur_instr_offs = bc_base_offset;
    phx_tc_emit(&deopt_tc, hir_c_create_snapshot(&deopt_tc.frame));
    phx_tc_emit(&deopt_tc, hir_c_create_deopt());

    /* (PE) frame_state_destroy on ALL 3 TCs. */
    phx_frame_state_destroy(&exc_tc.frame);
    phx_frame_state_destroy(&match_tc.frame);
    phx_frame_state_destroy(&deopt_tc.frame);
}

void hir_builder_emit_inline_exception_match_c(
        PhxTranslationContext *tc,
        void *func,
        void *builder,
        int bc_base_offset,
        int handler_depth,
        void *exc_type_obj,           /* SimpleExceptInfo::exc_type */
        int except_body_offset,       /* SimpleExceptInfo::except_body */
        HirType return_type,          /* preloader_.returnType() */
        void *left,
        void *right,
        void *result,
        void *getitem_fn,             /* JITRT_DictGetItem or PyObject_GetItem */
        void *match_and_clear_fn,     /* JITRT_MatchAndClearException */
        const OpcodeArrayEntry *opcodes,
        size_t opcode_count) {
    HirType t_opt_object = HIR_TYPE_OPTOBJECT;
    HirType t_object = HIR_TYPE_OBJECT;

    /* P1: getitem CallStatic dispatch. */
    void *getitem_call = hir_c_create_call_static_reg(
        2, result, getitem_fn, t_opt_object);
    hir_c_set_operand(getitem_call, 0, left);
    hir_c_set_operand(getitem_call, 1, right);
    phx_tc_emit(tc, getitem_call);

    void *ok_block = hir_cfg_alloc_block(func);
    void *exc_match_block = hir_cfg_alloc_block(func);
    phx_tc_emit(tc, hir_c_create_cond_branch_cpp(result, ok_block, exc_match_block));

    /* P2-P4 + PE: shared body. Re-push left+right for BINARY_SUBSCR offset. */
    emit_except_match_body_c(
        tc, func, builder,
        bc_base_offset, handler_depth, exc_match_block,
        exc_type_obj, except_body_offset, return_type,
        match_and_clear_fn, opcodes, opcode_count,
        /*deopt_left_repush=*/left, /*deopt_right_repush=*/right);

    /* P5: ok block — RefineType in-place SSA-rename. */
    tc->block = ok_block;
    phx_tc_emit(tc, hir_c_create_refine_type_reg(result, t_object, result));
}

/* emitCallExceptionHandler — CALL exception-match inline handler.
 * Sibling of emitInlineExceptionMatch (~95% shared). Mirrors C++
 * HIRBuilder::emitCallExceptionHandler @ builder.cpp:1426.
 *
 * Theologian PTE pre-audit (chat 2026-04-22 16:53Z): 26 invariants
 * SHARED with inline-match (P2+P3+P4 phases via shared helper) +
 * 3 UNIQUE (D1: pre-amble setSuppressExceptionDeopt + pop; D2: deopt
 * has ZERO re-pushes; D3: P5 RefineType + push result back). ZERO new
 * bridges per gate verdict. Shared-helper option (a) per STRONG lean:
 * D-1774910012 PA invariant lives in exactly one C body.
 *
 * 3 separate PhxTranslationContext objects (PB pitfall): exc_tc copies
 * from tc.frame (post-pop); match_tc copies from EXC_TC.FRAME; deopt_tc
 * copies from TC.FRAME (NOT exc_tc). All 3 destroyed in helper (PE). */
void hir_builder_emit_call_exception_handler_c(
        PhxTranslationContext *tc,
        void *func,
        void *builder,
        int bc_base_offset,
        int handler_depth,
        void *exc_type_obj,
        int except_body_offset,
        HirType return_type,
        void *result,
        void *match_and_clear_fn,
        const OpcodeArrayEntry *opcodes,
        size_t opcode_count) {
    HirType t_object = HIR_TYPE_OBJECT;

    /* (D1) Pre-amble (setSuppressExceptionDeopt + pop result) is done on
     * the C++ stub side BEFORE this C body — see builder.cpp emitter.
     * The C body's first action is the ok/exc branch on result. */

    void *ok_block = hir_cfg_alloc_block(func);
    void *exc_match_block = hir_cfg_alloc_block(func);
    phx_tc_emit(tc, hir_c_create_cond_branch_cpp(result, ok_block, exc_match_block));

    /* (D2) P2-P4 + PE: shared body. NO deopt re-push (call-result was
     * popped on C++ side; deopt offset = CALL itself, no extra stack). */
    emit_except_match_body_c(
        tc, func, builder,
        bc_base_offset, handler_depth, exc_match_block,
        exc_type_obj, except_body_offset, return_type,
        match_and_clear_fn, opcodes, opcode_count,
        /*deopt_left_repush=*/NULL, /*deopt_right_repush=*/NULL);

    /* (D3) P5: ok block — RefineType + push result back to OUTER tc stack.
     * RefineType is in-place SSA-rename; the result reg is then re-pushed
     * since emitAnyCall's pre-amble pop discarded it from the stack
     * (call_instr->setSuppressExceptionDeopt flow). */
    tc->block = ok_block;
    phx_tc_emit(tc, hir_c_create_refine_type_reg(result, t_object, result));
    phx_ptr_arr_push(&tc->frame.stack, result);
}

/* emitAnyCall full conversion — W26 (theologian L2462+L2466).
 *
 * Mirrors C++ HIRBuilder::emitAnyCall @ builder.cpp:2875 (pre-conversion).
 * Single C body covering all 5 opcode paths (CALL_FUNCTION/CALL_FUNCTION_KW,
 * CALL_FUNCTION_EX, CALL/CALL_KW/CALL_METHOD, INVOKE_FUNCTION/NATIVE/METHOD)
 * plus the await-tail dispatch INLINED (149b7e2d40 PartialConversion bridge
 * REABSORBED — its body is now part of this C body).
 *
 * C++ stub responsibilities: extract opcode/oparg/baseOffset, compute
 * is_awaited via iterator peek + PY_VERSION_HEX, pre-extract const_arg for
 * INVOKE_* paths, pass opaque iterator + bytecode block pointers for
 * await-tail iterator advancement.
 *
 * NEW bridges per theologian L2466:
 *   - hir_builder_emit_call_method_exception_handler_inline_c (combined per (B))
 *   - hir_builder_check_async_with_error_c (await-tail)
 *   - hir_builder_bc_it_advance_and_opcode_c (await-tail)
 *   - hir_builder_bc_it_oparg_c (await-tail)
 *
 * REABSORBED: hir_builder_emit_awaited_call_tail_c bridge declaration + impl
 * + REABSORB-WHEN comment block REMOVED. Body inlined below in await-tail
 * section. REABSORB-WHEN trigger fired post push 81 (Tier 6 INVOKE_* family
 * fully C). */
extern void hir_builder_emit_dispatch_eager_coro_result_c(
    PhxTranslationContext *tc, void *func, void *builder, void *out,
    void *await_block, void *post_await_block, int code_flags);
extern void hir_builder_emit_get_awaitable_c(
    PhxTranslationContext *tc, void *func, void *builder,
    int error_aenter, int error_aexit);
extern void hir_builder_emit_yield_from_method_c(
    PhxTranslationContext *tc, void *out, int code_flags);
extern void hir_builder_emit_call_ex_c(
    PhxTranslationContext *tc, void *func, int oparg, uint32_t flags);
extern void *hir_builder_get_kwnames(void *builder);
extern void hir_builder_set_kwnames(void *builder, void *reg);
extern void hir_builder_emit_call_method_exception_handler_inline_c(
    void *builder, void *tc, void *cfg, int base_offset,
    void *call_instr, void *result_reg);
extern void hir_builder_check_async_with_error_c(
    void *bc_instrs, void *bc_it,
    int *out_error_aenter, int *out_error_aexit);
extern int hir_builder_bc_it_advance_and_opcode_c(void *bc_it);
extern int hir_builder_bc_it_oparg_c(void *bc_it);

/* INVOKE_* delegations (push 81). */
extern void hir_builder_emit_invoke_native_c(
    PhxTranslationContext *tc, void *builder,
    PyObject *descr, PyObject *signature);
extern bool hir_builder_emit_invoke_method_c(
    PhxTranslationContext *tc, void *func, void *builder,
    PyObject *descr, long nargs, int is_awaited);
extern bool hir_builder_emit_invoke_function_c(
    PhxTranslationContext *tc, void *func, void *builder,
    PyObject *descr, long nargs, uint32_t flags);

void hir_builder_emit_any_call_c(
        PhxTranslationContext *tc,
        void *cfg,
        void *func,
        void *builder,
        void *bc_instrs,
        void *bc_it,
        int call_kind,        /* PhxCallKind enum (mapped from opcode by C++ stub) */
        int oparg,
        int base_offset,
        int is_awaited,
        int is_kw_arg,        /* set for CALL_FUNCTION_KW + CALL_KW (kwnames flag) */
        void *code,
        int code_flags,
        PyObject *const_arg) {
    /* CallFlags: None=0, KwArgs=1<<0=1, Awaited=1<<1=2, Static=1<<2=4.
     *
     * call_kind dispatch (set by C++ stub) replaces direct opcode switching
     * here so this C body does not need to import opcode constants. The
     * opcode switch in the C++ stub maps each call-class opcode to one of
     * the PHX_CALL_KIND_* values (defined in builder.h). */
    uint32_t flags = is_awaited ? 2u : 0u;
    int call_used_is_awaited = 1;

    switch (call_kind) {
        case PHX_CALL_KIND_VECTOR_CALL: {
            /* CALL_FUNCTION / CALL_FUNCTION_KW: variadic vector call.
             * Operands include the function arguments plus the function
             * itself. is_kw_arg adds 1 for the kwnames tuple. */
            size_t num_operands = (size_t)(oparg + 1);
            if (is_kw_arg) {
                num_operands++;
                flags |= 1u;  /* CallFlags::KwArgs */
            }
            void *out = hir_builder_temps_alloc_stack(builder);
            void *call = hir_c_create_vectorcall_reg(num_operands, out, flags);
            for (size_t i = num_operands; i > 0; i--) {
                void *operand = phx_ptr_arr_pop(&tc->frame.stack);
                hir_c_set_operand(call, i - 1, operand);
            }
            hir_deopt_set_frame_state(call, &tc->frame);
            phx_tc_emit(tc, call);
            phx_ptr_arr_push(&tc->frame.stack, out);
            break;
        }
        case PHX_CALL_KIND_CALL_EX: {
            hir_builder_emit_call_ex_c(tc, func, oparg, flags);
            break;
        }
        case PHX_CALL_KIND_CALL_METHOD: {
            /* CALL / CALL_KW / CALL_METHOD: dynamic call-method dispatch
             * with kwnames + exception-handler inline. is_kw_arg tracks
             * whether opcode is CALL_KW (extra stack input + flag). */
            size_t num_operands = (size_t)(oparg + 2);
            size_t num_stack_inputs = num_operands;
            void *kwnames_reg = hir_builder_get_kwnames(builder);
            if (kwnames_reg != NULL || is_kw_arg) {
                if (is_kw_arg) {
                    num_stack_inputs++;
                }
                num_operands++;
                flags |= 1u;  /* CallFlags::KwArgs */
            }

            /* Manually set up the instruction instead of using emitVariadic.
             * kwnames_ isn't on the stack, but it has to be part of the operand count. */
            void *out = hir_builder_temps_alloc_stack(builder);
            void *call = hir_c_create_call_method_reg(num_operands, out, flags);
            for (size_t i = num_stack_inputs; i > 0; i--) {
                void *arg = phx_ptr_arr_pop(&tc->frame.stack);
                hir_c_set_operand(call, i - 1, arg);
            }
            if (kwnames_reg != NULL) {
                JIT_CHECK_C(hir_c_get_operand(call, num_operands - 1) == NULL,
                    "Somehow already set the kwnames argument");
                hir_c_set_operand(call, num_operands - 1, kwnames_reg);
                hir_builder_set_kwnames(builder, NULL);
            }
            hir_deopt_set_frame_state(call, &tc->frame);
            phx_tc_emit(tc, call);
            phx_ptr_arr_push(&tc->frame.stack, out);

            /* B2: If this CALL is inside a try block with a simple except pattern,
             * inline the exception handler instead of deopting on exception.
             * Combined per W26 (B) decision: NULL-safe internally — does nothing
             * if no handler matches OR not simple-pattern. */
            hir_builder_emit_call_method_exception_handler_inline_c(
                builder, tc, cfg, base_offset, call, out);
            break;
        }
        case PHX_CALL_KIND_INVOKE_FUNCTION: {
            PyObject *descr = PyTuple_GET_ITEM(const_arg, 0);
            long nargs = PyLong_AsLong(PyTuple_GET_ITEM(const_arg, 1));
            call_used_is_awaited = hir_builder_emit_invoke_function_c(
                tc, func, builder, descr, nargs, flags) ? 1 : 0;
            break;
        }
        case PHX_CALL_KIND_INVOKE_NATIVE: {
            PyObject *native_target_descr = PyTuple_GET_ITEM(const_arg, 0);
            PyObject *signature = PyTuple_GET_ITEM(const_arg, 1);
            hir_builder_emit_invoke_native_c(tc, builder,
                native_target_descr, signature);
            call_used_is_awaited = 0;  /* emitInvokeNative always returns false */
            break;
        }
        case PHX_CALL_KIND_INVOKE_METHOD: {
            PyObject *descr = PyTuple_GET_ITEM(const_arg, 0);
            long nargs = PyLong_AsLong(PyTuple_GET_ITEM(const_arg, 1)) + 2;
            call_used_is_awaited = hir_builder_emit_invoke_method_c(
                tc, func, builder, descr, nargs, is_awaited) ? 1 : 0;
            break;
        }
        default:
            JIT_CHECK_C(0, "Unhandled call kind %d", call_kind);
    }

    if (is_awaited && call_used_is_awaited) {
        /* INLINED await-tail (149b7e2d40 PartialConversion bridge body
         * REABSORBED). Iterator-driven via the new bc_it bridges. */
        int op = hir_builder_bc_it_advance_and_opcode_c(bc_it);
        JIT_CHECK_C(op == GET_AWAITABLE,
            "Awaited function call must be followed by GET_AWAITABLE");

        int error_aenter = 0, error_aexit = 0;
        hir_builder_check_async_with_error_c(bc_instrs, bc_it,
            &error_aenter, &error_aexit);

        op = hir_builder_bc_it_advance_and_opcode_c(bc_it);
        JIT_CHECK_C(op == LOAD_CONST,
            "GET_AWAITABLE must be followed by LOAD_CONST");
        int load_const_oparg = hir_builder_bc_it_oparg_c(bc_it);

        op = hir_builder_bc_it_advance_and_opcode_c(bc_it);
        JIT_CHECK_C(op == YIELD_FROM,
            "GET_AWAITABLE should always be followed by LOAD_CONST+YIELD_FROM");

        /* await-tail body — was the body of hir_builder_emit_awaited_call_tail_c
         * (149b7e2d40 PartialConversion). Now inlined here. */
        void *out = hir_builder_temps_alloc_stack(builder);
        void *await_block = hir_cfg_alloc_block(func);
        void *post_await_block = hir_cfg_alloc_block(func);

        hir_builder_emit_dispatch_eager_coro_result_c(
            tc, func, builder, out, await_block, post_await_block, code_flags);

        tc->block = await_block;

        hir_builder_emit_get_awaitable_c(
            tc, func, builder, error_aenter, error_aexit);

        hir_builder_emit_load_const_c(tc, func, code, load_const_oparg);

        hir_builder_emit_yield_from_method_c(tc, out, code_flags);

        phx_tc_emit(tc, hir_c_create_branch_cpp(post_await_block));

        tc->block = post_await_block;
    }
}



