/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C implementation of funcTypeChecks — validates SSA type constraints.
 * Replaces funcTypeChecks + registerTypeMatches + operandsMustMatch
 * from analysis.cpp.
 */
#include "cinderx/Jit/hir/func_type_checks_c.h"
#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/hir/hir_type_c.h"
#include "cinderx/Common/jit_log_c.h"
#include "cinderx/StaticPython/checked_dict.h"
#include "cinderx/StaticPython/checked_list.h"

#include <stdio.h>

static int is_single_cint(HirType h) {
    return hir_type_is_subtype(h, HIR_TYPE_CINT8) ||
        hir_type_is_subtype(h, HIR_TYPE_CUINT8) ||
        hir_type_is_subtype(h, HIR_TYPE_CINT16) ||
        hir_type_is_subtype(h, HIR_TYPE_CUINT16) ||
        hir_type_is_subtype(h, HIR_TYPE_CINT32) ||
        hir_type_is_subtype(h, HIR_TYPE_CUINT32) ||
        hir_type_is_subtype(h, HIR_TYPE_CINT64) ||
        hir_type_is_subtype(h, HIR_TYPE_CUINT64);
}

static int register_type_matches(HirType op_hir, HirOperandType expected) {
    switch (expected.kind) {
    case HIR_CONSTRAINT_kType:
        return hir_type_is_subtype(op_hir, expected.type);
    case HIR_CONSTRAINT_kTupleExactOrCPtr:
        return hir_type_is_subtype(op_hir, HIR_TYPE_TUPLEEXACT) ||
            hir_type_is_subtype(op_hir, HIR_TYPE_CPTR);
    case HIR_CONSTRAINT_kListOrChkList:
        return hir_type_is_subtype(op_hir, HIR_TYPE_LIST) ||
            (hir_type_has_type_spec(&op_hir) &&
             Ci_CheckedList_TypeCheck(hir_type_type_spec(&op_hir)));
    case HIR_CONSTRAINT_kDictOrChkDict:
        return hir_type_is_subtype(op_hir, HIR_TYPE_DICT) ||
            (hir_type_has_type_spec(&op_hir) &&
             Ci_CheckedDict_TypeCheck(hir_type_type_spec(&op_hir)));
    case HIR_CONSTRAINT_kOptObjectOrCIntOrCBool:
        return hir_type_is_subtype(op_hir, HIR_TYPE_OPTOBJECT) ||
            hir_type_is_subtype(op_hir, HIR_TYPE_CINT) ||
            hir_type_is_subtype(op_hir, HIR_TYPE_CBOOL);
    case HIR_CONSTRAINT_kOptObjectOrCInt:
        return hir_type_is_subtype(op_hir, HIR_TYPE_OPTOBJECT) ||
            hir_type_is_subtype(op_hir, HIR_TYPE_CINT);
    case HIR_CONSTRAINT_kMatchAllAsCInt:
        return is_single_cint(op_hir);
    case HIR_CONSTRAINT_kMatchAllAsPrimitive:
        return is_single_cint(op_hir) ||
            hir_type_is_subtype(op_hir, HIR_TYPE_CBOOL) ||
            hir_type_is_subtype(op_hir, HIR_TYPE_CDOUBLE) ||
            hir_type_is_subtype(op_hir, HIR_TYPE_CPTR);
    }
    JIT_ABORT_C("unknown constraint %d", expected.kind);
}

static int operands_must_match(HirOperandType op_type) {
    switch (op_type.kind) {
    case HIR_CONSTRAINT_kMatchAllAsCInt:
    case HIR_CONSTRAINT_kMatchAllAsPrimitive:
        return 1;
    case HIR_CONSTRAINT_kType:
    case HIR_CONSTRAINT_kTupleExactOrCPtr:
    case HIR_CONSTRAINT_kListOrChkList:
    case HIR_CONSTRAINT_kDictOrChkDict:
    case HIR_CONSTRAINT_kOptObjectOrCInt:
    case HIR_CONSTRAINT_kOptObjectOrCIntOrCBool:
        return 0;
    }
    JIT_ABORT_C("unknown constraint %d", op_type.kind);
}

int hir_func_type_checks(HirFunction func) {
    HirCFG cfg = hir_func_cfg(func);
    const char *fullname = hir_func_fullname(func);

    for (HirBasicBlock block = hir_cfg_blocks_first(cfg);
         block != NULL;
         block = hir_cfg_blocks_next(cfg, block)) {
        int block_id = hir_block_id(block);

        for (HirInstr instr = hir_block_first(block);
             instr != NULL;
             instr = hir_block_next(block, instr)) {
            size_t n_ops = hir_c_num_operands(instr);

            if (n_ops > 1 &&
                operands_must_match(hir_c_get_operand_type(instr, 0))) {
                HirType join = HIR_TYPE_BOTTOM;
                for (size_t i = 0; i < n_ops; i++) {
                    HirRegister op = hir_c_get_operand(instr, i);
                    join = hir_type_union(join, hir_register_type(op));
                }
                HirOperandType expected = hir_c_get_operand_type(instr, 0);
                if (!register_type_matches(join, expected)) {
                    fprintf(stderr,
                        "TYPE MISMATCH in bb %d of '%s'\n"
                        "Instr (opcode %d) expected join of operands to match constraint %d\n",
                        block_id, fullname, hir_c_opcode(instr), expected.kind);
                    return 0;
                }
            } else {
                for (size_t i = 0; i < n_ops; i++) {
                    HirRegister op = hir_c_get_operand(instr, i);
                    HirOperandType expected = hir_c_get_operand_type(instr, i);
                    HirType op_type = hir_register_type(op);
                    if (!register_type_matches(op_type, expected)) {
                        fprintf(stderr,
                            "TYPE MISMATCH in bb %d of '%s'\n"
                            "Instr (opcode %d) expected operand %zu to match constraint %d\n",
                            block_id, fullname, hir_c_opcode(instr), i, expected.kind);
                        return 0;
                    }
                }
            }
        }
    }
    return 1;
}
