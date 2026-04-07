/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C-compatible LIR printer — Phase 3D replacement for printer.cpp.
 * Outputs to FILE* using fprintf.
 */
#ifndef JIT_LIR_PRINTER_C_H
#define JIT_LIR_PRINTER_C_H

#include "cinderx/Jit/lir/lir_c_api.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Print a complete LIR function. */
void lir_print_function(FILE *out, void *func);

/* Print a single LIR basic block. */
void lir_print_block(FILE *out, void *block, int show_hir_origin);

/* Print a single LIR instruction. */
void lir_print_instruction(FILE *out, const LirInstruction *instr);

/* Print a single LIR operand. */
void lir_print_operand(FILE *out, const LirOperand *operand);

/* Print a MemoryIndirect operand. */
void lir_print_memind(FILE *out, const LirMemoryIndirect *ind);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* JIT_LIR_PRINTER_C_H */
