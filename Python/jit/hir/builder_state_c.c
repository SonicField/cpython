/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Pure-C port of HIRBuilder Class A state initialization +
 * parseExceptionTable + findExceptionHandler algorithms. Phase 3 Batch 1
 * + Tier 8 pilot Phase A per
 * docs/tier7-phase3-hirbuilder-state-extraction-spec.md +
 * docs/tier8-class-b-cport-migrate-arm-spec.md.
 */

#include "cinderx/Jit/hir/builder_state_c.h"
#include "cinderx/Common/jit_log_c.h"
#include "Python.h"

#include <stdint.h>

void *phx_block_map_lookup_or_panic(const PhxBlockMap *m, int key) {
    void *blk = phx_block_map_lookup(m, key);
    JIT_CHECK_C(blk != NULL, "No block for offset %d", key);
    return blk;
}

void hir_builder_state_init(
        PhxHirBuilderState *state,
        void *code,
        const void *preloader) {
    state->code = code;
    state->preloader = preloader;
    state->current_func = NULL;
    state->func = NULL;
    state->kwnames = NULL;
    phx_exception_table_init(&state->exception_table_phx);
    phx_block_map_init(&state->block_map_phx);
    phx_bc_block_array_init(&state->bc_block_array_phx);
    /* Phase 4.C Pilot 3 step 2: temps_phx env wired by HIRBuilder ctor
     * via state->temps_phx.env = &irfunc->env (B44+ migration). For now
     * leave NULL — TempAllocator wrapper still owns the live state. */
    state->temps_phx.env = NULL;
    phx_ptr_arr_init(&state->temps_phx.cache);
    /* Phase 4.C Pilot 4 step 1: op_stack_phx zero-state; B49 collapses
     * the C++ static_method_stack_ wrapper onto this storage. */
    phx_ptr_arr_init(&state->op_stack_phx.stack);
}

void hir_builder_state_destroy(PhxHirBuilderState *state) {
    phx_exception_table_destroy(&state->exception_table_phx);
    phx_block_map_destroy(&state->block_map_phx);
    phx_bc_block_array_destroy(&state->bc_block_array_phx);
    phx_ptr_arr_destroy(&state->temps_phx.cache);
    phx_ptr_arr_destroy(&state->op_stack_phx.stack);
}

void hir_builder_state_parse_exception_table_c(
        PhxHirBuilderState *state,
        void *builder) {
    (void)builder;  /* Tier 8 pilot Phase A: PhxExceptionTable now lives
                     * in state, not C++-side; builder param retained
                     * for signature stability across Phase 3 Batch 1
                     * + Tier 8 transition. Phase B will drop it. */
    PyCodeObject *code = (PyCodeObject*)state->code;
    PyObject *table_obj = code->co_exceptiontable;
    if (table_obj == NULL || !PyBytes_Check(table_obj)) {
        return;
    }
    const uint8_t *table = (const uint8_t*)PyBytes_AS_STRING(table_obj);
    Py_ssize_t length = PyBytes_GET_SIZE(table_obj);
    Py_ssize_t pos = 0;

    /* CPython 3.12 varint: 6 payload bits per byte, bit 6 = continuation,
     * MSB first. */
    while (pos < length) {
        int vals[4];
        for (int i = 0; i < 4; i++) {
            int val = 0;
            uint8_t b;
            do {
                b = table[pos++];
                val = (val << 6) | (b & 0x3F);
            } while (b & 0x40);
            vals[i] = val;
        }
        int start = vals[0];
        int size = vals[1];
        int target = vals[2];
        int depth_lasti = vals[3];

        /* Convert instruction units to byte offsets. */
        const int scale = (int)sizeof(_Py_CODEUNIT);
        ExceptionTableEntry entry = {
            .start = start * scale,
            .end = (start + size) * scale,
            .target = target * scale,
            .depth = depth_lasti >> 1,
            .lasti = (unsigned char)(depth_lasti & 1),
        };
        phx_exception_table_push(&state->exception_table_phx, &entry);
    }
}

int hir_builder_state_find_exception_handler_c(
        PhxHirBuilderState *state,
        void *builder,
        int off,
        int *out_idx) {
    (void)builder;  /* Tier 8 pilot Phase A: lookup goes via state's
                     * PhxExceptionTable directly; builder param retained
                     * for signature stability. Phase B will drop it. */
    PhxExceptionTable *t = &state->exception_table_phx;
    size_t n = phx_exception_table_size(t);
    for (size_t i = 0; i < n; i++) {
        const ExceptionTableEntry *e = phx_exception_table_at(t, i);
        if (off >= e->start && off < e->end) {
            *out_idx = (int)i;
            return 1;
        }
    }
    return 0;
}
