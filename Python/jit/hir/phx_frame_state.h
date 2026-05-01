/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * PhxFrameState — C lifecycle for HIR FrameState value type.
 * Provides init/copy/destroy for owned FrameState instances
 * (TranslationContext holds FrameState by value with deep copy).
 */
#pragma once

#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/hir/phx_ptr_array.h"

#include <string.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Block stack element — must match C++ ExecutionBlock layout */
typedef struct {
    int opcode;
    int handler_off;
    int stack_level;
} PhxExecBlock;

static inline void phx_frame_state_init(HirFrameStateLayout *fs,
                                         void *code, int nlocals,
                                         void *globals, void *builtins) {
    memset(fs, 0, sizeof(*fs));
    fs->cur_instr_offs = -1;
    fs->code = code;
    fs->nlocals = nlocals;
    fs->globals = globals;
    fs->builtins = builtins;
    phx_ptr_arr_init(&fs->localsplus);
    phx_ptr_arr_init(&fs->stack);
}

static inline void phx_frame_state_copy(HirFrameStateLayout *dst,
                                         const HirFrameStateLayout *src) {
    dst->cur_instr_offs = src->cur_instr_offs;
    dst->nlocals = src->nlocals;
    dst->_fs_pad0 = src->_fs_pad0;
    dst->code = src->code;
    dst->globals = src->globals;
    dst->builtins = src->builtins;
    dst->parent = src->parent;

    phx_ptr_arr_copy(&dst->localsplus, &src->localsplus);
    phx_ptr_arr_copy(&dst->stack, &src->stack);

    /* Block stack: deep copy */
    dst->block_stack_count = src->block_stack_count;
    dst->block_stack_cap = src->block_stack_cap;
    if (src->block_stack_data && src->block_stack_cap > 0) {
        size_t bs_size = src->block_stack_cap * sizeof(PhxExecBlock);
        dst->block_stack_data = malloc(bs_size);
        memcpy(dst->block_stack_data, src->block_stack_data,
               src->block_stack_count * sizeof(PhxExecBlock));
    } else {
        dst->block_stack_data = NULL;
    }
}

/* Phase 4.D D1b-prep (Batch 83, slim per supervisor 00:39:12Z OPTION (ii)):
 * 2 PhxPtrArray*-returning accessors enabling D1b TranslationContext →
 * PhxTranslationContext rename. Returns the field-pointer for in-place
 * mutation by C++ caller — matches existing C++ tc.frame.stack/localsplus
 * reference semantics needed by phx_ptr_arr_push/pop + .data[i] assignment.
 *
 * 3 candidate scalar accessors (code/nlocals/cur_instr_offs) NOT added —
 * already exist as hir_fs_code/nlocals/cur_instr_offs in hir_instr_c.h.
 * Block_stack ops already exist as phx_block_stack_top/pop/is_empty below. */
static inline PhxPtrArray *phx_frame_state_stack(HirFrameStateLayout *fs) {
    return &fs->stack;
}

static inline PhxPtrArray *phx_frame_state_localsplus(
    HirFrameStateLayout *fs) {
    return &fs->localsplus;
}

static inline void phx_frame_state_destroy(HirFrameStateLayout *fs) {
    phx_ptr_arr_destroy(&fs->localsplus);
    phx_ptr_arr_destroy(&fs->stack);
    if (fs->block_stack_data) {
        free(fs->block_stack_data);
        fs->block_stack_data = NULL;
    }
}

/* ---- Block stack helpers ---- */

static inline void phx_block_stack_push(HirFrameStateLayout *fs,
                                         int opcode, int handler_off, int stack_level) {
    PhxExecBlock *data = (PhxExecBlock *)fs->block_stack_data;
    if (fs->block_stack_count >= fs->block_stack_cap) {
        size_t new_cap = fs->block_stack_cap ? fs->block_stack_cap * 2 : 4;
        data = (PhxExecBlock *)realloc(data, new_cap * sizeof(PhxExecBlock));
        fs->block_stack_data = data;
        fs->block_stack_cap = new_cap;
    }
    data[fs->block_stack_count].opcode = opcode;
    data[fs->block_stack_count].handler_off = handler_off;
    data[fs->block_stack_count].stack_level = stack_level;
    fs->block_stack_count++;
}

static inline PhxExecBlock phx_block_stack_top(const HirFrameStateLayout *fs) {
    PhxExecBlock *data = (PhxExecBlock *)fs->block_stack_data;
    return data[fs->block_stack_count - 1];
}

static inline PhxExecBlock phx_block_stack_pop(HirFrameStateLayout *fs) {
    fs->block_stack_count--;
    PhxExecBlock *data = (PhxExecBlock *)fs->block_stack_data;
    return data[fs->block_stack_count];
}

static inline int phx_block_stack_is_empty(const HirFrameStateLayout *fs) {
    return fs->block_stack_count == 0;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
