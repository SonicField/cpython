/*
 * dce.c -- Dead code elimination for LIR (pure C)
 *
 * Phase 3D conversion: dce.cpp -> dce.c
 *
 * Standard worklist-based DCE: seed with "useful" instructions (branches,
 * terminators, side-effecting ops), propagate liveness through operand
 * def-use chains, then remove anything not marked live.
 */

#include "cinderx/Jit/lir/lir_c_api.h"

#include "Python.h"

#include <string.h>

/* ---- Simple bit set indexed by instruction ID ---- */

typedef struct {
    uint8_t *bits;
    int capacity;  /* in bits */
} BitSet;

static void
bitset_init(BitSet *s, int max_id) {
    int nbytes = (max_id + 8) / 8;
    s->bits = (uint8_t *)PyMem_RawCalloc((size_t)nbytes, 1);
    s->capacity = max_id + 1;
}

static void
bitset_free(BitSet *s) {
    PyMem_RawFree(s->bits);
    s->bits = NULL;
}

static int
bitset_contains(const BitSet *s, int id) {
    if (id < 0 || id >= s->capacity) return 0;
    return (s->bits[id / 8] >> (id % 8)) & 1;
}

/* Returns 1 if newly inserted, 0 if already present. */
static int
bitset_insert(BitSet *s, int id) {
    if (id < 0 || id >= s->capacity) return 0;
    int idx = id / 8;
    uint8_t mask = (uint8_t)(1 << (id % 8));
    if (s->bits[idx] & mask) {
        return 0;
    }
    s->bits[idx] |= mask;
    return 1;
}

/* ---- Simple growable instruction queue ---- */

typedef struct {
    JitLirInstr *items;
    size_t head;
    size_t tail;
    size_t capacity;
} InstrQueue;

static void
instrq_init(InstrQueue *q, size_t cap) {
    q->items = (JitLirInstr *)PyMem_RawMalloc(cap * sizeof(JitLirInstr));
    q->head = 0;
    q->tail = 0;
    q->capacity = cap;
}

static void
instrq_free(InstrQueue *q) {
    PyMem_RawFree(q->items);
    q->items = NULL;
}

static int
instrq_empty(const InstrQueue *q) {
    return q->head == q->tail;
}

static void
instrq_push(InstrQueue *q, JitLirInstr instr) {
    if (q->tail >= q->capacity) {
        size_t new_cap = q->capacity * 2;
        q->items = (JitLirInstr *)PyMem_RawRealloc(
            q->items, new_cap * sizeof(JitLirInstr));
        q->capacity = new_cap;
    }
    q->items[q->tail++] = instr;
}

static JitLirInstr
instrq_pop(InstrQueue *q) {
    return q->items[q->head++];
}

/* ---- DCE helper: does the output operand affect memory? ---- */

static int
operand_affects_memory(JitLirOperand operand) {
    return (jit_lir_operand_is_reg(operand) ||
            jit_lir_operand_is_stack(operand) ||
            jit_lir_operand_is_mem(operand) ||
            jit_lir_operand_is_ind(operand));
}

/* ---- DCE helper: is this instruction in the root live set? ---- */

static int
is_useful(JitLirInstr instr) {
    JitLirOperand output = jit_lir_instr_output(instr);
    int essential = jit_lir_instr_is_essential(instr);
    int flag_fx = jit_lir_instr_flag_effects(instr);

    return (jit_lir_instr_is_any_branch(instr) ||
            jit_lir_instr_is_terminator(instr) ||
            flag_fx != JIT_LIR_FLAG_NONE ||
            essential ||
            (output != NULL && operand_affects_memory(output)));
}

/* ---- Worklist context ---- */

typedef struct {
    BitSet live_set;
    InstrQueue worklist;
} DceCtx;

static void
mark_live(DceCtx *ctx, JitLirInstr instr) {
    int id = jit_lir_instr_id(instr);
    if (bitset_insert(&ctx->live_set, id)) {
        instrq_push(&ctx->worklist, instr);
    }
}

/* ---- Operand def-use tracing ---- */

static void
add_linked_to_worklist(DceCtx *ctx, JitLirOperand operand) {
    if (operand == NULL || !jit_lir_operand_is_linked(operand)) {
        return;
    }
    JitLirInstr linked = jit_lir_operand_get_linked_instr(operand);
    mark_live(ctx, linked);
}

static void
add_all_regs_to_worklist(DceCtx *ctx, JitLirOperand operand) {
    if (jit_lir_operand_is_ind(operand)) {
        JitLirIndirect indirect = jit_lir_operand_get_indirect(operand);
        add_linked_to_worklist(ctx, jit_lir_indirect_base_reg(indirect));
        add_linked_to_worklist(ctx, jit_lir_indirect_index_reg(indirect));
    } else {
        add_linked_to_worklist(ctx, operand);
    }
}

/* Callback for jit_lir_instr_foreach_input */
static void
trace_input_cb(JitLirOperand operand, void *vctx) {
    add_all_regs_to_worklist((DceCtx *)vctx, operand);
}

/* Callback for jit_lir_block_remove_dead_instrs */
static int
is_live_cb(JitLirInstr instr, void *vctx) {
    DceCtx *ctx = (DceCtx *)vctx;
    return bitset_contains(&ctx->live_set, jit_lir_instr_id(instr));
}

/* ---- Main DCE entry point ---- */

void
jit_lir_eliminate_dead_code(JitLirFunc func) {
    /* First pass: find max instruction ID for bit set sizing. */
    size_t num_blocks = jit_lir_func_num_blocks(func);
    int max_id = 0;
    for (size_t bi = 0; bi < num_blocks; bi++) {
        JitLirBlock block = jit_lir_func_get_block(func, bi);
        size_t ni = jit_lir_block_num_instrs(block);
        for (size_t ii = 0; ii < ni; ii++) {
            JitLirInstr instr = jit_lir_block_get_instr_at(block, ii);
            int id = jit_lir_instr_id(instr);
            if (id > max_id) max_id = id;
        }
    }

    DceCtx ctx;
    bitset_init(&ctx.live_set, max_id);
    instrq_init(&ctx.worklist, 256);

    /* Phase 1: Seed the live set with useful instructions. */
    for (size_t bi = 0; bi < num_blocks; bi++) {
        JitLirBlock block = jit_lir_func_get_block(func, bi);
        size_t ni = jit_lir_block_num_instrs(block);
        for (size_t ii = 0; ii < ni; ii++) {
            JitLirInstr instr = jit_lir_block_get_instr_at(block, ii);
            if (is_useful(instr)) {
                mark_live(&ctx, instr);
            }
        }
    }

    /* Phase 2: Propagate liveness through def-use chains. */
    while (!instrq_empty(&ctx.worklist)) {
        JitLirInstr live_instr = instrq_pop(&ctx.worklist);
        jit_lir_instr_foreach_input(live_instr, trace_input_cb, &ctx);
        JitLirOperand output = jit_lir_instr_output(live_instr);
        if (output != NULL) {
            add_all_regs_to_worklist(&ctx, output);
        }
    }

    /* Phase 3: Remove dead instructions. */
    for (size_t bi = 0; bi < num_blocks; bi++) {
        JitLirBlock block = jit_lir_func_get_block(func, bi);
        jit_lir_block_remove_dead_instrs(block, is_live_cb, &ctx);
    }

    instrq_free(&ctx.worklist);
    bitset_free(&ctx.live_set);
}
