/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C dataflow analysis framework — replaces dataflow.h template.
 * De-templated: T = Register* (void* in C).
 */
#pragma once

#include "cinderx/Jit/bitvector_c.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PhxDataFlowBlock {
    PhxBitVector gen;
    PhxBitVector kill;
    PhxBitVector in;
    PhxBitVector out;
    struct PhxDataFlowBlock **preds;
    size_t n_preds;
    size_t cap_preds;
    struct PhxDataFlowBlock **succs;
    size_t n_succs;
    size_t cap_succs;
} PhxDataFlowBlock;

typedef struct {
    void **index_to_obj;
    size_t *obj_id_to_index;
    size_t max_obj_id;
    PhxDataFlowBlock **blocks;
    size_t n_blocks;
    size_t capacity;
    size_t num_bits;
    PhxDataFlowBlock *entry;
    PhxDataFlowBlock *exit_block;
} PhxDataFlowAnalyzer;

void phx_df_block_init(PhxDataFlowBlock *b);
void phx_df_block_destroy(PhxDataFlowBlock *b);
void phx_df_block_connect(PhxDataFlowBlock *from, PhxDataFlowBlock *to);

void phx_df_init(PhxDataFlowAnalyzer *a, size_t max_obj_id);
void phx_df_destroy(PhxDataFlowAnalyzer *a);
void phx_df_add_object(PhxDataFlowAnalyzer *a, void *obj, size_t obj_id);
void phx_df_add_block(PhxDataFlowAnalyzer *a, PhxDataFlowBlock *b);
void phx_df_set_entry(PhxDataFlowAnalyzer *a, PhxDataFlowBlock *b);
void phx_df_set_exit(PhxDataFlowAnalyzer *a, PhxDataFlowBlock *b);
void phx_df_set_gen_bit(PhxDataFlowAnalyzer *a, PhxDataFlowBlock *b, void *obj, size_t obj_id);
void phx_df_set_kill_bit(PhxDataFlowAnalyzer *a, PhxDataFlowBlock *b, void *obj, size_t obj_id);
int phx_df_get_in_bit(const PhxDataFlowAnalyzer *a, const PhxDataFlowBlock *b, size_t obj_id);
int phx_df_get_out_bit(const PhxDataFlowAnalyzer *a, const PhxDataFlowBlock *b, size_t obj_id);
void phx_df_run(PhxDataFlowAnalyzer *a, int forward);

/* Extended: meet_and=1 uses AND (intersection) for combining predecessors,
 * init_out_ones=1 initializes all out bitvectors to all-1s before solving.
 * Needed for definite assignment analysis. */
void phx_df_run_ex(PhxDataFlowAnalyzer *a, int forward, int meet_and, int init_out_ones);

typedef void (*PhxDfPerObjFunc)(void *obj, void *ctx);
void phx_df_for_each_in(const PhxDataFlowAnalyzer *a, const PhxDataFlowBlock *b,
                         PhxDfPerObjFunc func, void *ctx);
void phx_df_for_each_out(const PhxDataFlowAnalyzer *a, const PhxDataFlowBlock *b,
                          PhxDfPerObjFunc func, void *ctx);

#ifdef __cplusplus
}
#endif
