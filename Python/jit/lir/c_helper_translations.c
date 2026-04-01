/*
 * c_helper_translations.c -- C helper to LIR mapping (pure C)
 *
 * Phase 3D conversion: c_helper_translations.cpp -> c_helper_translations.c
 * Maps C helper function addresses to their LIR implementations.
 *
 * The JITRT_Cast address is set from C++ via jit_lir_set_cast_addr()
 * because JITRT_Cast has C++ linkage (name-mangled).
 */

#include "Python.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* JITRT_Cast address, set from C++ at init time */
static uint64_t jitrt_cast_addr = 0;

/* Buffer for the formatted JITRT_Cast LIR string */
static char cast_lir_buf[2048];
static int cast_lir_initialized = 0;

static void
init_cast_lir(void) {
    snprintf(cast_lir_buf, sizeof(cast_lir_buf),
        "Function:\n"
        "BB %%0 - succs: %%2 %%1\n"
        "       %%5:Object = LoadArg 0(0x0):Object\n"
        "       %%6:Object = LoadArg 1(0x1):Object\n"
        "       %%7:Object = Move [%%5:Object + %#zx]:Object\n"
        "       %%8:Object = Equal %%7:Object, %%6:Object\n"
        "                   CondBranch %%8:Object\n"
        "\n"
        "BB %%1 - preds: %%0 - succs: %%2 %%3\n"
        "      %%10:Object = Call PyType_IsSubtype, %%7:Object, %%6:Object\n"
        "                   CondBranch %%10:Object\n"
        "\n"
        "BB %%2 - preds: %%0 %%1 - succs: %%4\n"
        "                   Return %%5:Object\n"
        "\n"
        "BB %%3 - preds: %%1 - succs: %%4\n"
        "      %%13:Object = Move [%%7:Object + %#zx]:Object\n"
        "      %%14:Object = Move [%%6:Object + %#zx]:Object\n"
        "                   Call PyErr_Format, PyExc_TypeError, "
            "\"expected '%%s', got '%%s'\", %%14:Object, %%13:Object\n"
        "      %%16:Object = Move 0(0x0):Object\n"
        "                   Return %%16:Object\n"
        "\n"
        "BB %%4 - preds: %%2 %%3\n",
        offsetof(PyObject, ob_type),
        offsetof(PyTypeObject, tp_name),
        offsetof(PyTypeObject, tp_name));
    cast_lir_initialized = 1;
}

void
jit_lir_set_cast_addr(uint64_t addr) {
    jitrt_cast_addr = addr;
}

const char*
jit_lir_map_c_helper_to_lir(uint64_t addr) {
    if (!cast_lir_initialized) {
        init_cast_lir();
    }
    if (jitrt_cast_addr != 0 && addr == jitrt_cast_addr) {
        return cast_lir_buf;
    }
    return NULL;
}
