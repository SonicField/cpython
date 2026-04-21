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
