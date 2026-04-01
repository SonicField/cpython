/*
 * type.c -- LIR type utilities (pure C)
 *
 * Phase 3D conversion: type.cpp -> type.c
 * Provides bitSize and type name functions.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* DataType enum values (must match type.h enum class DataType) */
enum {
    JIT_LIR_DT_8BIT = 0,
    JIT_LIR_DT_16BIT,
    JIT_LIR_DT_32BIT,
    JIT_LIR_DT_64BIT,
    JIT_LIR_DT_DOUBLE,
    JIT_LIR_DT_OBJECT,
};

size_t
jit_lir_bit_size(int dt) {
    switch (dt) {
    case JIT_LIR_DT_8BIT:
        return 8;
    case JIT_LIR_DT_16BIT:
        return 16;
    case JIT_LIR_DT_32BIT:
        return 32;
    case JIT_LIR_DT_64BIT:
    case JIT_LIR_DT_DOUBLE:
    case JIT_LIR_DT_OBJECT:
        return 64;
    }
    fprintf(stderr, "JIT: %s:%d -- Unrecognized LIR DataType: %d\n",
            __FILE__, __LINE__, dt);
    abort();
}

const char*
jit_lir_data_type_name(int dt) {
    switch (dt) {
    case JIT_LIR_DT_8BIT:    return "8bit";
    case JIT_LIR_DT_16BIT:   return "16bit";
    case JIT_LIR_DT_32BIT:   return "32bit";
    case JIT_LIR_DT_64BIT:   return "64bit";
    case JIT_LIR_DT_DOUBLE:  return "Double";
    case JIT_LIR_DT_OBJECT:  return "Object";
    }
    return "<unknown DataType>";
}

const char*
jit_lir_operand_type_name(int ty) {
    switch (ty) {
    case 0: return "None";
    case 1: return "Vreg";
    case 2: return "Reg";
    case 3: return "Stack";
    case 4: return "Mem";
    case 5: return "Ind";
    case 6: return "Imm";
    case 7: return "Label";
    }
    return "<unknown OperandType>";
}
