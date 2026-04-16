/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C-compatible bytecode instruction representation.
 * Phase 3D: replaces C++ BytecodeInstruction class.
 */
#pragma once

/* Forward-declare CPython types to avoid Python.h in C++ TUs */
#ifndef PyObject_HEAD
typedef struct _object PyObject;
typedef struct _code PyCodeObject;
#endif

#include <stdint.h>
#include <stdbool.h>

#ifndef EXTENDED_OPCODE_FLAG
#define EXTENDED_OPCODE_FLAG 0
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* C equivalent of jit::BytecodeInstruction.
 * All offsets/indices are plain int (replaces BCOffset/BCIndex CRTP wrappers). */
typedef struct {
    PyCodeObject *code;
    int base_offset;
    int opcode_index;      /* mutable, lazily computed. INT_MIN = uncomputed */
    int extended_oparg;    /* mutable, lazily computed */
    int extended_opcode;   /* mutable, lazily computed (bool) */
} JitBytecodeInstr;

/* Initialize a bytecode instruction */
void jit_bc_instr_init(JitBytecodeInstr *bci, PyCodeObject *code, int base_offset);

/* Accessors */
int jit_bc_instr_base_offset(const JitBytecodeInstr *bci);
int jit_bc_instr_opcode_offset(JitBytecodeInstr *bci);
int jit_bc_instr_opcode(JitBytecodeInstr *bci);
int jit_bc_instr_specialized_opcode(JitBytecodeInstr *bci);
int jit_bc_instr_oparg(JitBytecodeInstr *bci);

/* Control flow queries */
int jit_bc_instr_is_branch(JitBytecodeInstr *bci);
int jit_bc_instr_is_return(JitBytecodeInstr *bci);
int jit_bc_instr_is_terminator(JitBytecodeInstr *bci);

/* Navigation */
int jit_bc_instr_get_jump_target(JitBytecodeInstr *bci);
int jit_bc_instr_next_offset(JitBytecodeInstr *bci);

/* Bytecode block iteration */
typedef struct {
    PyCodeObject *code;
    int start;
    int end;
} JitBytecodeBlock;

void jit_bc_block_init(JitBytecodeBlock *block, PyCodeObject *code);
void jit_bc_block_init_range(JitBytecodeBlock *block, PyCodeObject *code,
                              int start, int end);

/* Get first/next instruction in block. Returns 0 if at end. */
int jit_bc_block_first(const JitBytecodeBlock *block, JitBytecodeInstr *out);
int jit_bc_block_next(const JitBytecodeBlock *block, JitBytecodeInstr *cur);

#ifdef __cplusplus
}
#endif
