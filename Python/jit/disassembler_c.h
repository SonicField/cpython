/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C-compatible disassembler — Phase 3D replacement for disassembler.cpp.
 * Wraps capstone (C library) directly, outputs to FILE*.
 */
#ifndef JIT_DISASSEMBLER_C_H
#define JIT_DISASSEMBLER_C_H

#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *buf;
    size_t start;
    size_t size;
    size_t addr_len;
    int print_addr;
    int print_instr_bytes;
} JitDisassembler;

void jit_disasm_init(JitDisassembler *d, const char *buf, size_t size);
void jit_disasm_set_print_addr(JitDisassembler *d, int print);
void jit_disasm_set_print_instr_bytes(JitDisassembler *d, int print);
const char *jit_disasm_cursor(const JitDisassembler *d);

/* Disassemble a single instruction to FILE*. */
void jit_disasm_one(JitDisassembler *d, FILE *out);

/* Disassemble the entire buffer to FILE*. */
void jit_disasm_all(JitDisassembler *d, FILE *out);

/* C bridge for symbolizer (implemented in lir_c_api.cpp). */
int jit_symbolize(const void *func, char *buf, size_t buflen);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* JIT_DISASSEMBLER_C_H */
