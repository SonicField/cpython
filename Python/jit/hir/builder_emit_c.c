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
