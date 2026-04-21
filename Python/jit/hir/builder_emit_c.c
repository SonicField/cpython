/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C emit methods for HIRBuilder — incremental port of builder.cpp.
 * Phase 3D: hardest-first per Alex directive 2026-04-21.
 *
 * POC: emitLoadConst validates FrameState C lifecycle end-to-end.
 */

#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/hir/hir_basic_block_c.h"
#include "cinderx/Jit/hir/phx_frame_state.h"
#include "cinderx/Jit/hir/hir_type_c.h"
#include "cinderx/Jit/jit_config_c.h"
#include "Python.h"
#include "opcode.h"

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
extern void *hir_func_alloc_register(void *func);
extern void *hir_assign_create(void *output, void *value);

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
extern void *hir_c_create_check_var_reg(void *dst, void *src, PyObject *name, void *fs);

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
extern void *hir_c_create_get_length_reg(void *dst, void *src, void *fs);

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
extern void *hir_c_create_primitive_compare(void *dst, int32_t op, void *left, void *right);
extern void *hir_c_create_primitive_box_bool(void *dst, void *src);

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
extern void *hir_c_create_compare_reg(void *dst, int32_t op, void *left, void *right, void *fs);

/* emitDeleteAttr — pop receiver, emit DeleteAttr */
extern void *hir_c_create_delete_attr_reg(void *recv, int name_idx, void *fs);

void hir_builder_emit_delete_attr_c(PhxTranslationContext *tc, int oparg) {
    void *receiver = phx_ptr_arr_pop(&tc->frame.stack);
    phx_tc_emit(tc, hir_c_create_delete_attr_reg(receiver, oparg, &tc->frame));
}

/* emitUnaryOp — pop, map opcode→kind, emit UnaryOp, push */
extern void *hir_c_create_unary_op_reg(void *dst, int32_t op, void *operand, void *fs);

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
extern void *hir_c_create_int_convert_reg(void *dst, void *src, HirType type);
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
extern void *hir_c_create_steal_cell_item_reg(void *dst, void *cell);
extern void *hir_c_create_set_cell_item_reg(void *cell, void *value, void *old);

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
extern void *hir_c_create_store_attr_reg(void *recv, void *val, int name_idx, void *fs);

void hir_builder_emit_store_attr_c(PhxTranslationContext *tc, int oparg) {
    void *receiver = phx_ptr_arr_pop(&tc->frame.stack);
    void *value = phx_ptr_arr_pop(&tc->frame.stack);
    phx_tc_emit(tc, hir_c_create_store_attr_reg(receiver, value, oparg, &tc->frame));
}

/* emitLoadType — pop instance, LoadField ob_type, push */
extern void *hir_c_create_load_field_reg(void *dst, void *recv, const char *name,
                                          intptr_t offset, HirType type, int borrowed);

void hir_builder_emit_load_type_c(PhxTranslationContext *tc, void *func) {
    void *instance = phx_ptr_arr_pop(&tc->frame.stack);
    void *type_reg = hir_func_alloc_register(func);
    HirType t_type = HIR_TYPE_TYPE;
    phx_tc_emit(tc, hir_c_create_load_field_reg(type_reg, instance, "ob_type",
        offsetof(PyObject, ob_type), t_type, 0));
    phx_ptr_arr_push(&tc->frame.stack, type_reg);
}

/* emitCopyDictWithoutKeys — peek keys+subject, emit, replace top */
extern void *hir_c_create_copy_dict_without_keys_reg(void *dst, void *subj, void *keys, void *fs);

void hir_builder_emit_copy_dict_without_keys_c(PhxTranslationContext *tc, void *func) {
    PhxPtrArray *stack = &tc->frame.stack;
    void *keys = stack->data[stack->count - 1];
    void *subject = stack->data[stack->count - 2];
    void *rest = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_copy_dict_without_keys_reg(rest, subject, keys, &tc->frame));
    stack->data[stack->count - 1] = rest;
}

/* emitImportFrom — peek name, emit ImportFrom, push */
extern void *hir_c_create_import_from_reg(void *dst, void *name, int name_idx, void *fs);

void hir_builder_emit_import_from_c(PhxTranslationContext *tc, void *func, int oparg) {
    void *name = tc->frame.stack.data[tc->frame.stack.count - 1];
    void *res = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_import_from_reg(res, name, oparg, &tc->frame));
    phx_ptr_arr_push(&tc->frame.stack, res);
}

/* emitLoadDeref — localsplus cell read, LoadCellItem + CheckVar/CheckFreevar */
extern void *hir_c_create_load_cell_item_reg(void *dst, void *src);
extern void *hir_c_create_check_freevar_reg(void *dst, void *src, void *name, void *fs);

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
extern void *hir_c_create_dict_update_reg(void *dst, void *dict, void *update, void *fs);

void hir_builder_emit_dict_update_c(PhxTranslationContext *tc, void *func, int oparg) {
    void *update = phx_ptr_arr_pop(&tc->frame.stack);
    void *dict = tc->frame.stack.data[tc->frame.stack.count - oparg];
    void *dst = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_dict_update_reg(dst, dict, update, &tc->frame));
}

/* emitConvertValue — pop value, emit ConvertValue, push */
extern void *hir_c_create_convert_value_reg(void *dst, void *value, int conversion, void *fs);

void hir_builder_emit_convert_value_c(PhxTranslationContext *tc, void *func, int oparg) {
    void *value = phx_ptr_arr_pop(&tc->frame.stack);
    void *out = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_convert_value_reg(out, value, oparg, &tc->frame));
    phx_ptr_arr_push(&tc->frame.stack, out);
}

/* emitStoreSubscr — snapshot, pop sub+container+value, guard types, emit StoreSubscr */
extern void *hir_c_create_snapshot(void *frame_state);
extern void *hir_c_create_guard_type_reg(void *dst, HirType target, void *src);
extern void *hir_c_create_store_subscr_reg(void *container, void *sub, void *value, void *fs);

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
extern void *hir_c_create_format_with_spec_reg(void *dst, void *value, void *fmt_spec, void *fs);

void hir_builder_emit_format_with_spec_c(PhxTranslationContext *tc, void *func) {
    void *fmt_spec = phx_ptr_arr_pop(&tc->frame.stack);
    void *value = phx_ptr_arr_pop(&tc->frame.stack);
    void *out = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_format_with_spec_reg(out, value, fmt_spec, &tc->frame));
    phx_ptr_arr_push(&tc->frame.stack, out);
}

/* emitMapAdd — pop value+key, peek map at depth, emit SetDictItem */
extern void *hir_c_create_set_dict_item_reg(void *dst, void *dict, void *key, void *value, void *fs);

void hir_builder_emit_map_add_c(PhxTranslationContext *tc, void *func, int oparg) {
    void *value = phx_ptr_arr_pop(&tc->frame.stack);
    void *key = phx_ptr_arr_pop(&tc->frame.stack);
    void *map = tc->frame.stack.data[tc->frame.stack.count - oparg];
    void *dst = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_set_dict_item_reg(dst, map, key, value, &tc->frame));
}

/* emitSetAdd — pop value, peek set at depth, emit SetSetItem */
extern void *hir_c_create_set_set_item_reg(void *dst, void *set, void *item, void *fs);

void hir_builder_emit_set_add_c(PhxTranslationContext *tc, void *func, int oparg) {
    void *item = phx_ptr_arr_pop(&tc->frame.stack);
    void *set = tc->frame.stack.data[tc->frame.stack.count - oparg];
    void *dst = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_set_set_item_reg(dst, set, item, &tc->frame));
}

/* emitSetUpdate — pop iterable, peek set at depth, emit SetUpdate */
extern void *hir_c_create_set_update_reg(void *dst, void *set, void *iter, void *fs);

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
extern void *hir_c_create_binary_op_reg(void *dst, int32_t op, void *left, void *right, void *fs);

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

/* emitStoreField — pop receiver+value, fieldInfo lookup, StoreField */
extern void hir_builder_preloader_field_info(void *builder, PyObject *descr,
                                              intptr_t *offset_out, HirType *type_out,
                                              PyObject **name_out);
extern void *hir_c_create_store_field_reg(void *receiver, const char *name, intptr_t offset, void *value, HirType type, void *previous);
extern void *hir_c_create_int_convert_reg(void *dst, void *src, HirType type);

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
extern void *hir_c_create_make_function_reg(void *dst, void *code, void *qualname, void *fs);
extern void *hir_c_create_set_function_attr_reg(void *value, void *base, int32_t field);
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
extern void *hir_c_create_load_method_reg(void *dst, void *receiver, int32_t name_idx, void *fs);
extern void *hir_c_create_get_second_output_reg(void *dst, HirType type, void *src);
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

/* emitLoadAttr generic — non-specialized LoadAttr2 fallback */
extern void *hir_c_create_load_attr_reg2(void *dst, void *receiver, int32_t name_idx, void *fs);

void hir_builder_emit_load_attr_generic_c(PhxTranslationContext *tc, void *func,
                                           void *receiver, int name_idx) {
    void *result = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_load_attr_reg2(result, receiver, name_idx, &tc->frame));
    phx_ptr_arr_push(&tc->frame.stack, result);
}

/* emitBinaryOp — specialized guards + oparg dispatch + BinaryOp/InPlaceOp */
extern void *hir_c_create_binary_op_reg(void *dst, int32_t op, void *left, void *right, void *fs);
extern void *hir_c_create_in_place_op_reg(void *dst, int32_t op_kind,
                                           void *left, void *right, void *fs);

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
extern void *hir_c_create_match_keys_reg(void *dst, void *subj, void *keys, void *fs);
extern void *hir_cfg_alloc_block(void *func);
extern void *hir_c_create_cond_branch_cpp(void *cond_reg, void *true_bb, void *false_bb);
extern void *hir_c_create_branch_cpp(void *target_block);
extern void *hir_c_create_refine_type_reg(void *dst, HirType type, void *src);

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
extern void *hir_c_create_check_field_reg(void *dst, void *src, void *name, void *fs);
extern void hir_c_set_guilty_reg(void *instr, void *reg);
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
extern void *hir_c_create_dict_subscr_reg(void *dst, void *dict, void *key, void *fs);

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
extern void *hir_c_create_guard_type_fs_reg(void *dst, HirType type, void *src, void *fs);

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
extern void *hir_c_create_load_global_cached_reg(void *dst, void *code, void *builtins, void *globals, int32_t name_idx);
extern void *hir_c_create_guard_is_reg(void *dst, void *target, void *src);
extern void *hir_c_create_load_global_reg(void *dst, int32_t name_idx, void *fs);
extern void hir_c_set_descr(void *instr, const char *descr);

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
extern void *hir_c_create_build_interpolation_reg(void *dst, void *val, void *str, void *fmt, int32_t conv, void *fs);

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
extern PyObject *hir_func_add_reference(void *func, PyObject *obj);

void hir_builder_emit_load_assertion_error_c(PhxTranslationContext *tc, void *func) {
    void *result = hir_func_alloc_register(func);
    PyObject *ref = hir_func_add_reference(func, PyExc_AssertionError);
    HirType type = hir_type_from_object(ref);
    phx_tc_emit(tc, hir_c_create_load_const(result, type));
    phx_ptr_arr_push(&tc->frame.stack, result);
}

/* emitRefineType — refine stack top to preloaded type */
extern HirType hir_builder_preloader_type(void *builder, PyObject *descr);
extern void *hir_c_create_refine_type_reg(void *dst, HirType type, void *src);

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
extern void *hir_c_create_cast_reg(void *dst, void *recv, void *pytype, int optional, int exact, void *fs);
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
extern void *hir_c_create_tp_alloc_reg(void *dst, void *pytype, void *fs);

void hir_builder_emit_tp_alloc_c(PhxTranslationContext *tc, void *func, void *builder,
                                  PyCodeObject *code, int oparg) {
    PyObject *descr = PyTuple_GET_ITEM(code->co_consts, oparg);
    void *pytype = hir_builder_preloader_py_type(builder, descr);
    void *result = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_tp_alloc_reg(result, pytype, &tc->frame));
    phx_ptr_arr_push(&tc->frame.stack, result);
}

/* emitImportName — pop fromlist+level, emit ImportName or EagerImportName */
extern void *hir_c_create_import_name_reg(void *dst, int32_t name_idx, void *fromlist, void *level, void *fs);
extern void *hir_c_create_eager_import_name_reg(void *dst, int32_t name_idx, void *fromlist, void *level, void *fs);

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
extern void *hir_c_create_call_ex_reg(void *dst, void *func_r, void *pargs, void *kwargs, uint32_t flags, void *fs);

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

/* emitForIter — peek iterator, InvokeIterNext, CondBranchIterNotDone */
extern void *hir_builder_get_block_at_off(void *builder, int byte_offset);
extern void *hir_c_create_invoke_iter_next_reg(void *dst, void *iter, void *fs);
extern void *hir_c_create_cond_branch_iter_not_done_cpp(void *val, void *body, void *footer);

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
extern void *hir_c_create_unpack_ex_to_tuple_reg(void *dst, void *seq, int32_t before, int32_t after, void *fs);
extern void *hir_c_create_load_tuple_item_reg(void *dst, void *tuple, int32_t idx);

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
extern void *hir_c_create_deopt(void);
extern void *hir_c_create_branch_cpp(void *target_block);
extern void *hir_c_create_cond_branch_check_type_cpp(void *target, HirType type, void *true_bb, void *false_bb);
extern void *hir_c_create_load_field_reg(void *dst, void *recv, const char *name, intptr_t offset, HirType type, int borrowed);
extern void *hir_cfg_alloc_block(void *func);

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
extern void *hir_c_create_cond_branch_cpp(void *cond_reg, void *true_bb, void *false_bb);
extern void *hir_c_create_is_truthy_reg(void *dst, void *src, void *fs);

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
extern void *hir_c_create_dict_merge_reg(void *dst, void *dict, void *update, void *func, void *fs);

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
extern void *hir_c_create_make_tuple_reg(size_t n, void *dst, void *fs);
extern void *hir_c_create_make_list_reg(size_t n, void *dst, void *fs);

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
extern void *hir_c_create_make_dict_reg(void *dst, int32_t dict_size, void *fs);

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
extern void *hir_c_create_make_set_reg(void *dst, void *fs);

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
extern void *hir_c_create_call_intrinsic_reg2(size_t n, void *dst, int32_t index, void **operands);

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
    phx_tc_emit(tc, hir_c_create_call_intrinsic_reg2(num_operands, res, oparg, args));
    phx_ptr_arr_push(&tc->frame.stack, res);
}

/* emitSetFunctionAttribute — pop func+value, map oparg→FunctionAttr, emit */
extern void *hir_c_create_set_function_attr_reg(void *value, void *base, int32_t field);

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
extern void *hir_c_create_get_a_iter_reg(void *dst, void *src, void *fs);

void hir_builder_emit_get_aiter_c(PhxTranslationContext *tc, void *func) {
    void *obj = phx_ptr_arr_pop(&tc->frame.stack);
    void *out = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_get_a_iter_reg(out, obj, &tc->frame));
    phx_ptr_arr_push(&tc->frame.stack, out);
}

/* emitGetANext — peek obj, emit GetANext, push */
extern void *hir_c_create_get_a_next_reg(void *dst, void *src, void *fs);

void hir_builder_emit_get_anext_c(PhxTranslationContext *tc, void *func) {
    void *obj = tc->frame.stack.data[tc->frame.stack.count - 1];
    void *out = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_get_a_next_reg(out, obj, &tc->frame));
    phx_ptr_arr_push(&tc->frame.stack, out);
}

/* emitBuildTemplate — pop interpolations+strings, emit BuildTemplate, push */
extern void *hir_c_create_build_template_reg(void *strings, void *interps, void *dst, void *fs);

void hir_builder_emit_build_template_c(PhxTranslationContext *tc, void *func) {
    void *interpolations = phx_ptr_arr_pop(&tc->frame.stack);
    void *strings = phx_ptr_arr_pop(&tc->frame.stack);
    void *out = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_build_template_reg(strings, interpolations, out, &tc->frame));
    phx_ptr_arr_push(&tc->frame.stack, out);
}

/* emitFormatValue — pop fmt_spec (if present), pop value, FormatValue, push */
extern void *hir_c_create_format_value_reg(void *dst, void *fmt, void *val, int32_t conv, void *fs);

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
extern void *hir_c_create_list_extend_reg(void *dst, void *list, void *iter, void *fs);

void hir_builder_emit_list_extend_c(PhxTranslationContext *tc, void *func, int oparg) {
    void *iterable = phx_ptr_arr_pop(&tc->frame.stack);
    void *list = tc->frame.stack.data[tc->frame.stack.count - oparg];
    void *none = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_list_extend_reg(none, list, iterable, &tc->frame));
}

/* emitListToTuple — pop list, emit MakeTupleFromList, push tuple */
extern void *hir_c_create_make_tuple_from_list_reg(void *dst, void *list, void *fs);

void hir_builder_emit_list_to_tuple_c(PhxTranslationContext *tc, void *func) {
    void *list = phx_ptr_arr_pop(&tc->frame.stack);
    void *tuple = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_make_tuple_from_list_reg(tuple, list, &tc->frame));
    phx_ptr_arr_push(&tc->frame.stack, tuple);
}

/* emitMakeCell — allocate cell, MakeCell + moveOverwritten + Assign */
extern void *hir_c_create_make_cell_reg(void *dst, void *src, void *fs);

void hir_builder_emit_make_cell_c(PhxTranslationContext *tc, void *func, int local_idx) {
    void *local = tc->frame.localsplus.data[local_idx];
    void *cell = hir_func_alloc_register(func);
    phx_tc_emit(tc, hir_c_create_make_cell_reg(cell, local, &tc->frame));
    phx_move_overwritten_stack_regs(tc, func, local);
    phx_tc_emit(tc, hir_assign_create(local, cell));
}

/* emitRaiseVarargs — emit Raise */
extern void *hir_c_create_raise_reg(void *fs);

void hir_builder_emit_raise_varargs_c(PhxTranslationContext *tc) {
    phx_tc_emit(tc, hir_c_create_raise_reg(&tc->frame));
}

/* emitListAppend — pop item, peek list at depth, emit ListAppend */
extern void *hir_c_create_list_append_reg(void *dst, void *list, void *item, void *fs);

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
extern void *hir_c_create_in_place_op_reg(void *dst, int32_t op_kind,
                                           void *left, void *right, void *fs);

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
