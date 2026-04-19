/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Pure C simplify handlers — incremental port of simplify.cpp.
 * Chunk 1: Env-free handlers (return existing register or NULL).
 */

#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/hir/hir_type_c.h"

/* Forward declarations (avoid hir_c_api.h typedef conflicts) */
extern HirType hir_register_type(void *reg);

/* ---- simplifyCheck ----
 * CheckVar, CheckExc, CheckField all check input for null.
 * If input is known to be an Object (non-null), the check is redundant. */
void *simplify_check_c(const void *instr) {
    void *operand = hir_c_get_operand(instr, 0);
    HirType reg_type = hir_register_type(operand);
    HirType t_object = HIR_TYPE_OBJECT;
    if (hir_type_is_subtype(reg_type, t_object)) {
        return operand;
    }
    return NULL;
}
