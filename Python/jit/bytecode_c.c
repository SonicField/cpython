/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C implementation of bytecode instruction operations.
 * Phase 3D: replaces C++ BytecodeInstruction class methods.
 */

#include <limits.h>
#include <stdbool.h>

#include "cinderx/python.h"          /* Must be first — defines PyCodeObject */
#include "cinderx/Common/code.h"
#include "cinderx/Common/opcode_stubs.h"
#include "cinderx/Interpreter/cinder_opcode.h"
#include "cinderx/Jit/bytecode_c.h"

void jit_bc_instr_init(JitBytecodeInstr *bci, PyCodeObject *code, int base_offset) {
    bci->code = code;
    bci->base_offset = base_offset;
    bci->opcode_index = INT_MIN;  /* uncomputed sentinel */
    bci->extended_oparg = 0;
    bci->extended_opcode = 0;
}

int jit_bc_instr_base_offset(const JitBytecodeInstr *bci) {
    return bci->base_offset;
}

/* Lazily compute opcode index and extended oparg */
static void calc_opcode_offset_and_oparg(JitBytecodeInstr *bci) {
    if (bci->opcode_index != INT_MIN) {
        return;
    }

    bci->opcode_index = bci->base_offset;
    int end_idx = (int)countIndices(bci->code);
    if (bci->opcode_index >= end_idx) {
        return;
    }

    _Py_CODEUNIT *units = codeUnit(bci->code);

    /* Consume EXTENDED_ARG opcodes */
    while (_Py_OPCODE(units[bci->opcode_index]) == EXTENDED_ARG) {
        bci->extended_oparg = (bci->extended_oparg << 8) |
                              _Py_OPARG(units[bci->opcode_index]);
        bci->opcode_index++;
    }

#if PY_VERSION_HEX >= 0x030E0000
    /* Check for EXTENDED_OPCODE */
    if (_Py_OPCODE(units[bci->opcode_index]) == EXTENDED_OPCODE) {
        bci->opcode_index++;
        bci->extended_oparg = 0;
        bci->extended_opcode = 1;
        /* Consume more EXTENDED_ARGs after EXTENDED_OPCODE */
        while (_Py_OPCODE(units[bci->opcode_index]) == EXTENDED_ARG) {
            bci->extended_oparg = (bci->extended_oparg << 8) |
                                  _Py_OPARG(units[bci->opcode_index]);
            bci->opcode_index++;
        }
    }
#endif

    bci->extended_oparg = (bci->extended_oparg << 8) |
                          _Py_OPARG(units[bci->opcode_index]);
}

int jit_bc_instr_opcode_offset(JitBytecodeInstr *bci) {
    calc_opcode_offset_and_oparg(bci);
    return bci->opcode_index;
}

static _Py_CODEUNIT get_word(JitBytecodeInstr *bci) {
    calc_opcode_offset_and_oparg(bci);
#if PY_VERSION_HEX >= 0x030C0000
    int op = unspecialize(uninstrument(bci->code, bci->opcode_index));
    int arg = _Py_OPARG(codeUnit(bci->code)[bci->opcode_index]);
    return _Py_MAKE_CODEUNIT(op, arg);
#else
    return codeUnit(bci->code)[bci->opcode_index];
#endif
}

int jit_bc_instr_opcode(JitBytecodeInstr *bci) {
    int op = _Py_OPCODE(get_word(bci));
    if (bci->extended_opcode) {
        return EXTENDED_OPCODE_FLAG | op;
    }
    return op;
}

int jit_bc_instr_oparg(JitBytecodeInstr *bci) {
    calc_opcode_offset_and_oparg(bci);
    return bci->extended_oparg;
}

int jit_bc_instr_is_branch(JitBytecodeInstr *bci) {
    switch (jit_bc_instr_opcode(bci)) {
        case FOR_ITER:
        case JUMP_ABSOLUTE:
        case JUMP_BACKWARD:
        case JUMP_BACKWARD_NO_INTERRUPT:
        case JUMP_FORWARD:
        case JUMP_IF_FALSE_OR_POP:
        case JUMP_IF_NONZERO_OR_POP:
        case JUMP_IF_NOT_EXC_MATCH:
        case JUMP_IF_TRUE_OR_POP:
        case JUMP_IF_ZERO_OR_POP:
        case POP_JUMP_IF_FALSE:
        case POP_JUMP_IF_NONE:
        case POP_JUMP_IF_NONZERO:
        case POP_JUMP_IF_NOT_NONE:
        case POP_JUMP_IF_TRUE:
        case POP_JUMP_IF_ZERO:
        case SEND:
        case SETUP_FINALLY:
            return 1;
        default:
            return 0;
    }
}

int jit_bc_instr_is_return(JitBytecodeInstr *bci) {
    switch (jit_bc_instr_opcode(bci)) {
        case RETURN_CONST:
        case RETURN_PRIMITIVE:
        case RETURN_VALUE:
            return 1;
        default:
            return 0;
    }
}

int jit_bc_instr_is_terminator(JitBytecodeInstr *bci) {
    switch (jit_bc_instr_opcode(bci)) {
        case RAISE_VARARGS:
        case RERAISE:
            return 1;
        default:
            return jit_bc_instr_is_branch(bci) || jit_bc_instr_is_return(bci);
    }
}

int jit_bc_instr_next_offset(JitBytecodeInstr *bci) {
    int idx = jit_bc_instr_opcode_offset(bci);
    return idx + inlineCacheSize(bci->code, idx) + 1;
}

/* Block operations */
void jit_bc_block_init(JitBytecodeBlock *block, PyCodeObject *code) {
    block->code = code;
    block->start = 0;
    block->end = (int)countIndices(code);
}

void jit_bc_block_init_range(JitBytecodeBlock *block, PyCodeObject *code,
                              int start, int end) {
    block->code = code;
    block->start = start;
    block->end = end;
}

int jit_bc_block_first(const JitBytecodeBlock *block, JitBytecodeInstr *out) {
    if (block->start >= block->end) return 0;
    jit_bc_instr_init(out, block->code, block->start);
    return 1;
}

int jit_bc_block_next(const JitBytecodeBlock *block, JitBytecodeInstr *cur) {
    int next = jit_bc_instr_next_offset(cur);
    if (next >= block->end) return 0;
    jit_bc_instr_init(cur, block->code, next);
    return 1;
}
