/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C dataflow analysis framework — replaces dataflow.h template.
 */
#include "cinderx/Jit/dataflow_c.h"
#include "cinderx/Common/jit_log_c.h"
#include <string.h>
#include <stdlib.h>

void phx_df_block_init(PhxDataFlowBlock *b) {
    memset(b, 0, sizeof(*b));
}

void phx_df_block_destroy(PhxDataFlowBlock *b) {
    phx_bv_destroy(&b->gen);
    phx_bv_destroy(&b->kill);
    phx_bv_destroy(&b->in);
    phx_bv_destroy(&b->out);
    free(b->preds);
    free(b->succs);
}

void phx_df_block_connect(PhxDataFlowBlock *from, PhxDataFlowBlock *to) {
    if (from->n_succs >= from->cap_succs) {
        from->cap_succs = from->cap_succs ? from->cap_succs * 2 : 4;
        from->succs = (PhxDataFlowBlock **)realloc(from->succs, from->cap_succs * sizeof(PhxDataFlowBlock *));
    }
    if (to->n_preds >= to->cap_preds) {
        to->cap_preds = to->cap_preds ? to->cap_preds * 2 : 4;
        to->preds = (PhxDataFlowBlock **)realloc(to->preds, to->cap_preds * sizeof(PhxDataFlowBlock *));
    }
    from->succs[from->n_succs++] = to;
    to->preds[to->n_preds++] = from;
}

void phx_df_init(PhxDataFlowAnalyzer *a, size_t max_obj_id) {
    memset(a, 0, sizeof(*a));
    a->max_obj_id = max_obj_id;
    a->obj_id_to_index = (size_t *)calloc(max_obj_id + 1, sizeof(size_t));
    a->index_to_obj = NULL;
    a->blocks = NULL;
    a->capacity = 0;
}

void phx_df_destroy(PhxDataFlowAnalyzer *a) {
    free(a->obj_id_to_index);
    free(a->index_to_obj);
    free(a->blocks);
}

void phx_df_add_object(PhxDataFlowAnalyzer *a, void *obj, size_t obj_id) {
    size_t idx = a->num_bits;
    a->obj_id_to_index[obj_id] = idx;
    a->num_bits++;
    a->index_to_obj = (void **)realloc(a->index_to_obj, a->num_bits * sizeof(void *));
    a->index_to_obj[idx] = obj;
    for (size_t i = 0; i < a->n_blocks; i++) {
        PhxDataFlowBlock *b = a->blocks[i];
        phx_bv_init(&b->gen, a->num_bits);
        phx_bv_init(&b->kill, a->num_bits);
        phx_bv_init(&b->in, a->num_bits);
        phx_bv_init(&b->out, a->num_bits);
    }
}

void phx_df_add_block(PhxDataFlowAnalyzer *a, PhxDataFlowBlock *b) {
    if (a->n_blocks >= a->capacity) {
        a->capacity = a->capacity ? a->capacity * 2 : 16;
        a->blocks = (PhxDataFlowBlock **)realloc(a->blocks, a->capacity * sizeof(PhxDataFlowBlock *));
    }
    a->blocks[a->n_blocks++] = b;
    phx_bv_init(&b->gen, a->num_bits);
    phx_bv_init(&b->kill, a->num_bits);
    phx_bv_init(&b->in, a->num_bits);
    phx_bv_init(&b->out, a->num_bits);
}

void phx_df_set_entry(PhxDataFlowAnalyzer *a, PhxDataFlowBlock *b) {
    a->entry = b;
}

void phx_df_set_exit(PhxDataFlowAnalyzer *a, PhxDataFlowBlock *b) {
    a->exit_block = b;
}

void phx_df_set_gen_bit(PhxDataFlowAnalyzer *a, PhxDataFlowBlock *b, void *obj, size_t obj_id) {
    (void)obj;
    size_t idx = a->obj_id_to_index[obj_id];
    phx_bv_set_bit(&b->gen, idx, 1);
}

void phx_df_set_kill_bit(PhxDataFlowAnalyzer *a, PhxDataFlowBlock *b, void *obj, size_t obj_id) {
    (void)obj;
    size_t idx = a->obj_id_to_index[obj_id];
    phx_bv_set_bit(&b->kill, idx, 1);
}

int phx_df_get_in_bit(const PhxDataFlowAnalyzer *a, const PhxDataFlowBlock *b, size_t obj_id) {
    size_t idx = a->obj_id_to_index[obj_id];
    return phx_bv_get_bit(&b->in, idx);
}

int phx_df_get_out_bit(const PhxDataFlowAnalyzer *a, const PhxDataFlowBlock *b, size_t obj_id) {
    size_t idx = a->obj_id_to_index[obj_id];
    return phx_bv_get_bit(&b->out, idx);
}

void phx_df_for_each_in(const PhxDataFlowAnalyzer *a, const PhxDataFlowBlock *b,
                         PhxDfPerObjFunc func, void *ctx) {
    for (size_t i = 0; i < a->num_bits; i++) {
        if (phx_bv_get_bit(&b->in, i)) {
            func(a->index_to_obj[i], ctx);
        }
    }
}

void phx_df_for_each_out(const PhxDataFlowAnalyzer *a, const PhxDataFlowBlock *b,
                          PhxDfPerObjFunc func, void *ctx) {
    for (size_t i = 0; i < a->num_bits; i++) {
        if (phx_bv_get_bit(&b->out, i)) {
            func(a->index_to_obj[i], ctx);
        }
    }
}

void phx_df_run(PhxDataFlowAnalyzer *a, int forward) {
    PhxDataFlowBlock *skip = forward ? a->entry : a->exit_block;

    PhxDataFlowBlock **worklist = (PhxDataFlowBlock **)malloc(a->n_blocks * sizeof(PhxDataFlowBlock *));
    size_t wl_head = 0, wl_tail = 0;

    for (size_t i = 0; i < a->n_blocks; i++) {
        if (a->blocks[i] != skip) {
            worklist[wl_tail++] = a->blocks[i];
        }
    }

    PhxBitVector new_in, new_out;
    phx_bv_init(&new_in, a->num_bits);
    phx_bv_init(&new_out, a->num_bits);

    while (wl_head < wl_tail) {
        PhxDataFlowBlock *block = worklist[wl_head++];

        PhxDataFlowBlock **pred_arr = forward ? block->preds : block->succs;
        size_t n_pred = forward ? block->n_preds : block->n_succs;
        PhxDataFlowBlock **succ_arr = forward ? block->succs : block->preds;
        size_t n_succ = forward ? block->n_succs : block->n_preds;
        PhxBitVector *in_bv = forward ? &block->in : &block->out;
        PhxBitVector *out_bv = forward ? &block->out : &block->in;

        phx_bv_reset_all(&new_in);
        for (size_t i = 0; i < n_pred; i++) {
            PhxBitVector *p_out = forward ? &pred_arr[i]->out : &pred_arr[i]->in;
            phx_bv_or_assign(&new_in, p_out);
        }

        int changed = !phx_bv_equal(&new_in, in_bv);
        phx_bv_copy(in_bv, &new_in);

        /* new_out = gen | (in - kill) */
        phx_bv_copy(&new_out, in_bv);
        phx_bv_sub_assign(&new_out, &block->kill);
        phx_bv_or_assign(&new_out, &block->gen);

        changed |= !phx_bv_equal(&new_out, out_bv);
        phx_bv_copy(out_bv, &new_out);

        if (changed) {
            for (size_t i = 0; i < n_succ; i++) {
                worklist[wl_tail++] = succ_arr[i];
                if (wl_tail >= a->n_blocks * 4) {
                    /* Compact worklist if it grows too large */
                    size_t remaining = wl_tail - wl_head;
                    memmove(worklist, worklist + wl_head, remaining * sizeof(PhxDataFlowBlock *));
                    wl_head = 0;
                    wl_tail = remaining;
                }
            }
        }
    }

    phx_bv_destroy(&new_in);
    phx_bv_destroy(&new_out);
    free(worklist);
}
