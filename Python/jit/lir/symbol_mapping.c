/*
 * symbol_mapping.c -- CPython symbol name to address mapping (pure C)
 *
 * Phase 3D conversion: symbol_mapping.cpp -> symbol_mapping.c
 * Maps CPython function names to their runtime addresses.
 *
 * Note: PyExc_TypeError is an extern variable (not a function), so its
 * address is not a compile-time constant in C. We use a lazy-initialized
 * table populated on first call.
 */

#include "Python.h"

#include <stdint.h>
#include <string.h>

typedef struct {
    const char *name;
    uint64_t addr;
} JitSymbolEntry;

#define NUM_SYMBOLS 9

static JitSymbolEntry symbol_table[NUM_SYMBOLS];
static int table_initialized = 0;

static void
init_symbol_table(void) {
    int i = 0;
    symbol_table[i].name = "PyErr_Format";
    symbol_table[i].addr = (uint64_t)(uintptr_t)PyErr_Format;
    i++;
    symbol_table[i].name = "PyExc_TypeError";
    symbol_table[i].addr = (uint64_t)(uintptr_t)PyExc_TypeError;
    i++;
    symbol_table[i].name = "PyLong_AsSsize_t";
    symbol_table[i].addr = (uint64_t)(uintptr_t)PyLong_AsSsize_t;
    i++;
    symbol_table[i].name = "PyLong_AsSize_t";
    symbol_table[i].addr = (uint64_t)(uintptr_t)PyLong_AsSize_t;
    i++;
    symbol_table[i].name = "PyLong_FromLong";
    symbol_table[i].addr = (uint64_t)(uintptr_t)PyLong_FromLong;
    i++;
    symbol_table[i].name = "PyLong_FromSize_t";
    symbol_table[i].addr = (uint64_t)(uintptr_t)PyLong_FromSize_t;
    i++;
    symbol_table[i].name = "PyLong_FromSsize_t";
    symbol_table[i].addr = (uint64_t)(uintptr_t)PyLong_FromSsize_t;
    i++;
    symbol_table[i].name = "PyLong_FromUnsignedLong";
    symbol_table[i].addr = (uint64_t)(uintptr_t)PyLong_FromUnsignedLong;
    i++;
    symbol_table[i].name = "PyType_IsSubtype";
    symbol_table[i].addr = (uint64_t)(uintptr_t)PyType_IsSubtype;
    i++;
    table_initialized = 1;
}

const uint64_t*
jit_lir_py_function_from_name(const char *name) {
    if (!table_initialized) {
        init_symbol_table();
    }
    for (int i = 0; i < NUM_SYMBOLS; i++) {
        if (strcmp(symbol_table[i].name, name) == 0) {
            return &symbol_table[i].addr;
        }
    }
    return NULL;
}
