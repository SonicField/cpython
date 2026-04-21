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
extern void *hir_c_create_load_tuple_item_reg(void *dst, void *tuple, int32_t idx);

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
extern void *hir_builder_get_block_at_off(void *builder, int byte_offset);
extern void *hir_c_create_cond_branch_cpp(void *cond_reg, void *true_bb, void *false_bb);
extern void *hir_c_create_is_truthy_reg(void *dst, void *src, void *fs);

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
