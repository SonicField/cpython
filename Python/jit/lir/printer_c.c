/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C implementation of the LIR printer — Phase 3D replacement for
 * printer.cpp. Uses fprintf to FILE*, C API calls for LIR access.
 */

#include "cinderx/Jit/lir/printer_c.h"
#include "cinderx/Jit/lir/lir_c_api.h"
#include "cinderx/Jit/jit_config_c.h"
#include "cinderx/Jit/codegen/code_section.h"
#include "cinderx/Jit/codegen/phylocation.h"

#include <inttypes.h>
#include <string.h>

/* ---- Operand printing ---- */

void
lir_print_operand(FILE *out, const LirOperand *operand) {
    if (!operand) {
        fprintf(out, "<null>");
        return;
    }

    /* Follow linked operand chain */
    if (lir_operand_is_linked(operand)) {
        const LirOperand *def = lir_operand_get_define((LirOperand *)operand);
        if (def) {
            lir_print_operand(out, def);
            return;
        }
    }

    uint8_t ty = lir_operand_type(operand);
    switch (ty) {
    case JIT_LIR_OPTYPE_VREG:
        fprintf(out, "%%%d", lir_instruction_id(
            (const LirInstruction *)lir_operand_instr(operand)));
        break;
    case JIT_LIR_OPTYPE_REG: {
        LirPhyLocation loc = lir_operand_get_phy_register(operand);
        char buf[32];
        phyloc_to_string(loc, buf, sizeof(buf));
        fprintf(out, "%s", buf);
        break;
    }
    case JIT_LIR_OPTYPE_STACK: {
        LirPhyLocation loc = lir_operand_get_stack_slot(operand);
        char buf[32];
        phyloc_to_string(loc, buf, sizeof(buf));
        fprintf(out, "%s", buf);
        break;
    }
    case JIT_LIR_OPTYPE_MEM:
        fprintf(out, "[%" PRIx64 "]",
                (uint64_t)(uintptr_t)lir_operand_get_mem_address(operand));
        break;
    case JIT_LIR_OPTYPE_IND: {
        LirMemoryIndirect *ind = lir_operand_get_indirect(operand);
        if (ind) {
            lir_print_memind(out, ind);
        } else {
            fprintf(out, "[???]");
        }
        break;
    }
    case JIT_LIR_OPTYPE_IMM: {
        uint64_t val = lir_operand_get_constant(operand);
        fprintf(out, "%" PRId64 "(0x%" PRIx64 ")",
                (int64_t)val, val);
        break;
    }
    case JIT_LIR_OPTYPE_LABEL: {
        void *bb = lir_operand_get_basic_block(operand);
        if (bb) {
            fprintf(out, "BB%%%d", jit_lir_block_get_id(bb));
        } else {
            fprintf(out, "BB%%?");
        }
        break;
    }
    case JIT_LIR_OPTYPE_NONE:
        fprintf(out, "<!!!None!!!>");
        break;
    }

    if (ty != JIT_LIR_OPTYPE_LABEL) {
        const char *dt_name = lir_operand_data_type_name(
            lir_operand_data_type(operand));
        fprintf(out, ":%s", dt_name);
    }
}

/* ---- MemoryIndirect printing ---- */

void
lir_print_memind(FILE *out, const LirMemoryIndirect *ind) {
    LirOperand *base = lir_memind_base_reg(ind);
    fprintf(out, "[");
    if (base) {
        lir_print_operand(out, base);
    }

    LirOperand *index = lir_memind_index_reg(ind);
    if (index) {
        fprintf(out, " + ");
        lir_print_operand(out, index);
        uint8_t mult = lir_memind_multiplier(ind);
        if (mult > 0) {
            fprintf(out, " * %d", 1 << mult);
        }
    }

    int32_t offset = lir_memind_offset(ind);
    if (offset > 0) {
        fprintf(out, " + 0x%x", offset);
    } else if (offset < 0) {
        fprintf(out, " - 0x%x", -offset);
    }

    fprintf(out, "]");
}

/* ---- Instruction printing ---- */

void
lir_print_instruction(FILE *out, const LirInstruction *instr) {
    /* Output operand */
    const LirOperand *output = lir_instruction_output((LirInstruction *)instr);
    if (lir_operand_type(output) == JIT_LIR_OPTYPE_NONE) {
        fprintf(out, "%16s   ", "");
    } else {
        /* Print output to a temp buffer for right-alignment */
        char buf[64];
        FILE *tmp = fmemopen(buf, sizeof(buf), "w");
        if (tmp) {
            lir_print_operand(tmp, output);
            fclose(tmp);
            fprintf(out, "%16s = ", buf);
        } else {
            lir_print_operand(out, output);
            fprintf(out, " = ");
        }
    }

    int opcode = lir_instruction_opcode(instr);
    fprintf(out, "%s", lir_instruction_opcode_name(opcode));

    const char *opname = lir_instruction_opcode_name(opcode);
    int is_phi = (strcmp(opname, "Phi") == 0);

    if (is_phi) {
        size_t n = lir_instruction_num_inputs(instr);
        const char *sep = " ";
        for (size_t i = 0; i < n; i += 2) {
            fprintf(out, "%s(", sep);
            lir_print_operand(out, lir_instruction_get_input(instr, i));
            fprintf(out, ", ");
            if (i + 1 < n) {
                lir_print_operand(out, lir_instruction_get_input(instr, i + 1));
            }
            fprintf(out, ")");
            sep = ", ";
        }
    } else {
        const char *sep = " ";
        size_t n = lir_instruction_num_inputs(instr);
        for (size_t i = 0; i < n; i++) {
            fprintf(out, "%s", sep);
            lir_print_operand(out, lir_instruction_get_input(instr, i));
            sep = ", ";
        }
    }
}

/* ---- Block printing ---- */

static int
block_id_cmp(const void *a, const void *b) {
    int id_a = jit_lir_block_get_id(*(void *const *)a);
    int id_b = jit_lir_block_get_id(*(void *const *)b);
    return id_a - id_b;
}

void
lir_print_block(FILE *out, void *block, int show_hir_origin) {
    fprintf(out, "BB %%%d", jit_lir_block_get_id(block));

    /* Predecessors (sorted by id) */
    size_t num_preds = jit_lir_block_num_preds(block);
    if (num_preds > 0) {
        void *preds[256];
        size_t n = num_preds < 256 ? num_preds : 256;
        for (size_t i = 0; i < n; i++) {
            preds[i] = jit_lir_block_get_pred(block, i);
        }
        qsort(preds, n, sizeof(void *), block_id_cmp);
        fprintf(out, " - preds:");
        for (size_t i = 0; i < n; i++) {
            fprintf(out, " %%%d", jit_lir_block_get_id(preds[i]));
        }
    }

    /* Successors (unsorted, preserving order) */
    size_t num_succs = jit_lir_block_num_succs(block);
    if (num_succs > 0) {
        fprintf(out, " - succs:");
        for (size_t i = 0; i < num_succs; i++) {
            void *succ = jit_lir_block_get_succ(block, i);
            fprintf(out, " %%%d", jit_lir_block_get_id(succ));
        }
    }

    /* Section (skip hot to reduce noise) */
    int section = jit_lir_block_get_section(block);
    if (section != 0 /* kHot */) {
        fprintf(out, " - section: %s", jit_code_section_name(section));
    }
    fprintf(out, "\n");

    /* Instructions */
    const void *prev_origin = NULL;
    size_t num_instrs = jit_lir_block_num_instrs(block);
    for (size_t i = 0; i < num_instrs; i++) {
        LirInstruction *instr = (LirInstruction *)jit_lir_block_get_instr_at(
            block, i);
        const void *origin = lir_instruction_get_origin(instr);
        if (show_hir_origin && origin != prev_origin) {
            if (origin) {
                fprintf(out, "\n");
                lir_hir_print_instr(out, origin);
                fprintf(out, "\n");
            }
            prev_origin = origin;
        }
        lir_print_instruction(out, instr);
        fprintf(out, "\n");
    }
}

/* ---- Function printing ---- */

void
lir_print_function(FILE *out, void *func) {
    const JitConfig *cfg = jit_get_config();
    int show_hir = cfg ? cfg->log.lir_origin : 0;

    fprintf(out, "Function:\n");
    size_t num_blocks = jit_lir_func_num_blocks(func);
    for (size_t i = 0; i < num_blocks; i++) {
        void *block = jit_lir_func_get_block(func, i);
        lir_print_block(out, block, show_hir);
        fprintf(out, "\n");
    }
}
