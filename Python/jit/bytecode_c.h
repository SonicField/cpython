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
int jit_bc_instr_is_absolute_control_flow(JitBytecodeInstr *bci);

/* Navigation */
int jit_bc_instr_get_jump_target(JitBytecodeInstr *bci);
int jit_bc_instr_next_offset(JitBytecodeInstr *bci);

/* Raw word access (requires _Py_CODEUNIT from cpython/code.h) */
#ifdef _Py_OPCODE
_Py_CODEUNIT jit_bc_instr_word(JitBytecodeInstr *bci);

/* Boundary-domain conversions between BCOffset (byte offsets, used by
 * C++ BCOffset.value() and phx_block_map keys per builder.cpp:1235) and
 * instruction indices (codeUnit[] indexing, used by all jit_bc_instr_*
 * accessors and phx_block_map_lookup_or_panic argument). See
 * Python/jit/bytecode.cpp:8-14 for the boundary convention.
 *
 * Use these named helpers at every C/C++ seam that crosses byte ↔ index
 * units, per the boundary-domain rule (W-PROTOCOL-CODIFY, supervisor
 * 2026-04-25 19:01:19Z + theologian 19:01:17Z). Inline arithmetic at the
 * seam is the bug class: BCOffset/InstrIndex Class A + B + C mismatches
 * in build_inline_except_opcode_array_c (W-2B-RECONVERT investigation
 * found three boundary-domain bugs in a single helper across two HIR-diff
 * Phase 0 cycles). */
static inline int phx_bc_offset_to_instr_index(int byte_off) {
    return byte_off / (int)sizeof(_Py_CODEUNIT);
}
static inline int phx_bc_instr_index_to_offset(int instr_idx) {
    return instr_idx * (int)sizeof(_Py_CODEUNIT);
}
#endif

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
