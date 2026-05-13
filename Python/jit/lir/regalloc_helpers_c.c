/*
 * regalloc_helpers_c.c -- pure-C ports of small predicate/lookup
 * helpers from lir::regalloc.cpp.
 *
 * Phase 5.B c15: phx_should_replace_operand — port of regalloc.cpp:46-50
 * shouldReplaceOperand(OperandBase&). Uses jit_lir_operand_is_vreg
 * (Phase 5.B c15 new accessor) + jit_lir_operand_is_linked (Phase 5.B
 * c9 inlined). LirOperand layout cross-validated by
 * lir_instr_c_verify.cpp (sizeof @:35, type_ offset @:43, is_linked_
 * offset @:41).
 */

#include "cinderx/Jit/lir/regalloc_helpers_c.h"

int phx_should_replace_operand(JitLirOperand op) {
    /* Linked operands are always replaced with new Operand instances.
     * Matches C++ shouldReplaceOperand semantics:
     *   return operand.isVreg() || operand.isLinked(); */
    return jit_lir_operand_is_vreg(op) || jit_lir_operand_is_linked(op);
}
