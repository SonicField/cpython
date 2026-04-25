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
 * W-PROTOCOL-CODIFY P5: TYPE-LEVEL DOMAIN ENFORCEMENT.
 * Single-field struct typedefs make BcByteOffset and BcInstrIndex
 * compile-time-distinct so the compiler catches accidental cross-domain
 * assignment. Conversion functions take/return the wrapper types; raw int
 * access is via the .v field. This is the STRUCTURAL form of the
 * boundary-domain rule (supervisor 19:37:39Z STRONG CONCUR per pythia
 * #138 #3, post W-2B-RECONVERT Class A→B→C cascade where naming-only
 * enforcement degraded as theologian's HirType negative control predicted).
 *
 * Paired empirical controls (per librarian 19:18:55Z + 19:43:28Z):
 *   POSITIVE: PhxMem has_base structural flag — JIT_CHECK_C didn't fire
 *     on dual-arch gate; structural prevention worked.
 *   NEGATIVE: HirType named-conversion-only — 14 reinterpret_casts
 *     shipped broken until 2026-04-15 (f59580a0e6); naming alone degrades. */
typedef struct { int v; } BcByteOffset;
typedef struct { int v; } BcInstrIndex;

/* Named entry-point factories — Phase A.5 (W-PROTOCOL-CODIFY) per pythia
 * #139 #1 + supervisor 20:51:44Z. These document the trust boundary
 * where a raw int CROSSES INTO the wrapper-typed domain. Mid-flow
 * propagation between BcByteOffset/BcInstrIndex is type-enforced by P5
 * wrappers; entry-point wrap from raw int is an unchecked-trust contract
 * the caller asserts. Use these factories instead of brace-init so
 * grep/lint can audit entry points and enforce that callers explicitly
 * named the source domain.
 *
 * USAGE: bc_byte_offset_from_int(BCOffset.value())   <-- input is BYTE OFFSET
 *        bc_instr_index_from_int(jit_bc_instr_*_*())  <-- input is INSTRUCTION INDEX */
static inline BcByteOffset bc_byte_offset_from_int(int v) {
    BcByteOffset out = { v };
    return out;
}
static inline BcInstrIndex bc_instr_index_from_int(int v) {
    BcInstrIndex out = { v };
    return out;
}

static inline BcInstrIndex phx_bc_offset_to_instr_index(BcByteOffset off) {
    BcInstrIndex out = { off.v / (int)sizeof(_Py_CODEUNIT) };
    return out;
}
static inline BcByteOffset phx_bc_instr_index_to_offset(BcInstrIndex idx) {
    BcByteOffset out = { idx.v * (int)sizeof(_Py_CODEUNIT) };
    return out;
}

/* P5 expansion: VTABLE byte offset wrapper. NOT bytecode-byte (use
 * BcByteOffset for that); this is offset-into-_PyType_VTable
 * (vt_entries[slot] + member). Used by emitLoadMethodStatic seam
 * (theologian P7 audit 2026-04-25 20:53:51Z found vte_state_offset +
 * vte_load_offset as raw intptr_t = swap-risk Class-A-shape; wrapper
 * prevents CROSS-DOMAIN swap). intptr_t (not int) for vtable size
 * headroom — bytecode size fits int but vtable could exceed.
 *
 * Same-domain ordering protection (distinct VTableStateOffset /
 * VTableLoadOffset typedefs) is OUT-OF-SCOPE per theologian spec
 * (defer until ordering bug observed). */
typedef struct { intptr_t v; } VTableByteOffset;

static inline VTableByteOffset vtable_byte_offset_from_intptr(intptr_t v) {
    VTableByteOffset out = { v };
    return out;
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
