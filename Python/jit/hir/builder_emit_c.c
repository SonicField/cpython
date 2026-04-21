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
