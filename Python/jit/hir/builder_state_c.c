/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Pure-C port of HIRBuilder Class A state initialization +
 * parseExceptionTable algorithm. Phase 3 Batch 1 per
 * docs/tier7-phase3-hirbuilder-state-extraction-spec.md.
 */

#include "cinderx/Jit/hir/builder_state_c.h"
#include "Python.h"

#include <stdint.h>

void hir_builder_state_init(
        PhxHirBuilderState *state,
        void *code,
        const void *preloader) {
    state->code = code;
    state->preloader = preloader;
    state->current_func = NULL;
    state->func = NULL;
    state->kwnames = NULL;
}

void hir_builder_state_parse_exception_table_c(
        PhxHirBuilderState *state,
        void *builder) {
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
        hir_builder_state_exception_table_push_cpp(
            builder,
            start * scale,
            (start + size) * scale,
            target * scale,
            depth_lasti >> 1,
            depth_lasti & 1);
    }
}

int hir_builder_state_find_exception_handler_c(
        PhxHirBuilderState *state,
        void *builder,
        int off,
        int *out_idx) {
    (void)state;  /* state unused (vector access via _cpp bridge). */
    int n = hir_builder_state_exception_table_size_cpp(builder);
    for (int i = 0; i < n; i++) {
        int start, end, target, depth, lasti;
        hir_builder_state_exception_table_entry_cpp(
            builder, i, &start, &end, &target, &depth, &lasti);
        if (off >= start && off < end) {
            *out_idx = i;
            return 1;
        }
    }
    return 0;
}
