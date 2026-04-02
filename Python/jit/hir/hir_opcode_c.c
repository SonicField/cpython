/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C implementation of HIR opcode name lookup.
 */

#include "cinderx/Jit/hir/hir_opcode_c.h"

#include <string.h>

static const char* s_opcode_names[] = {
#define HIR_NAME_OP(opname) #opname,
    FOREACH_OPCODE(HIR_NAME_OP)
#undef HIR_NAME_OP
};

const char* hir_opcode_name(HirOpcode op) {
    if (op >= 0 && op < HIR_OP_COUNT) {
        return s_opcode_names[op];
    }
    return "<unknown>";
}

HirOpcode hir_opcode_from_name(const char *name) {
    for (int i = 0; i < HIR_OP_COUNT; i++) {
        if (strcmp(name, s_opcode_names[i]) == 0) {
            return (HirOpcode)i;
        }
    }
    return HIR_OP_COUNT; /* not found */
}
