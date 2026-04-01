/*
 * lir_c_api.h -- C-compatible accessor API for LIR types
 *
 * Phase 3D: Provides opaque pointer accessors for LIR Function, BasicBlock,
 * Instruction, and OperandBase so that algorithm files can be written in pure C.
 *
 * C++ callers: include this header and use the C functions directly,
 * or continue using C++ class methods. The C API is additive.
 *
 * C callers: include this header for opaque pointer access to LIR types.
 */

#ifndef JIT_LIR_C_API_H
#define JIT_LIR_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handles for LIR types */
typedef void* JitLirFunc;
typedef void* JitLirBlock;
typedef void* JitLirInstr;
typedef void* JitLirOperand;

/* Code section constants (must match codegen::CodeSection enum) */
#define JIT_LIR_SECTION_HOT  0
#define JIT_LIR_SECTION_COLD 1

/* ---- Function accessors ---- */
size_t jit_lir_func_num_blocks(JitLirFunc func);
JitLirBlock jit_lir_func_get_block(JitLirFunc func, size_t index);
JitLirBlock jit_lir_func_entry_block(JitLirFunc func);

/* ---- BasicBlock accessors ---- */
size_t jit_lir_block_num_preds(JitLirBlock block);
JitLirBlock jit_lir_block_get_pred(JitLirBlock block, size_t index);
size_t jit_lir_block_num_succs(JitLirBlock block);
JitLirBlock jit_lir_block_get_succ(JitLirBlock block, size_t index);
JitLirInstr jit_lir_block_get_last_instr(JitLirBlock block);
JitLirInstr jit_lir_block_get_first_instr(JitLirBlock block);
size_t jit_lir_block_num_instrs(JitLirBlock block);
JitLirBlock jit_lir_block_get_false_succ(JitLirBlock block);
int jit_lir_block_get_section(JitLirBlock block);
void jit_lir_block_set_section(JitLirBlock block, int section);
int jit_lir_block_get_id(JitLirBlock block);
JitLirInstr jit_lir_block_get_instr_at(JitLirBlock block, size_t index);

/* ---- Instruction accessors ---- */
int jit_lir_instr_opcode(JitLirInstr instr);
int jit_lir_instr_is_branch(JitLirInstr instr);
int jit_lir_instr_is_branch_cc(JitLirInstr instr);
int jit_lir_instr_is_any_branch(JitLirInstr instr);
int jit_lir_instr_is_terminator(JitLirInstr instr);
size_t jit_lir_instr_num_inputs(JitLirInstr instr);
JitLirOperand jit_lir_instr_get_input(JitLirInstr instr, size_t index);
JitLirOperand jit_lir_instr_output(JitLirInstr instr);

/* ---- Operand accessors ---- */
int jit_lir_operand_type(JitLirOperand op);
JitLirBlock jit_lir_operand_get_basic_block(JitLirOperand op);

/* Operand type constants (must match lir::OperandType enum) */
#define JIT_LIR_OPTYPE_NONE  0
#define JIT_LIR_OPTYPE_VREG  1
#define JIT_LIR_OPTYPE_REG   2
#define JIT_LIR_OPTYPE_STACK 3
#define JIT_LIR_OPTYPE_MEM   4
#define JIT_LIR_OPTYPE_IND   5
#define JIT_LIR_OPTYPE_IMM   6
#define JIT_LIR_OPTYPE_LABEL 7

/* Opcode constants */
int jit_lir_opcode_guard(void);

/* ---- Extended instruction accessors (Phase 3D Step 10: DCE) ---- */
int jit_lir_instr_id(JitLirInstr instr);
int jit_lir_instr_is_essential(JitLirInstr instr);
int jit_lir_instr_flag_effects(JitLirInstr instr);

/* Iterate all input operands of an instruction. */
void jit_lir_instr_foreach_input(
    JitLirInstr instr,
    void (*cb)(JitLirOperand operand, void *ctx),
    void *ctx);

/* ---- Extended operand accessors ---- */
int jit_lir_operand_is_reg(JitLirOperand op);
int jit_lir_operand_is_stack(JitLirOperand op);
int jit_lir_operand_is_mem(JitLirOperand op);
int jit_lir_operand_is_ind(JitLirOperand op);
int jit_lir_operand_is_linked(JitLirOperand op);

/* Get the defining instruction of a LinkedOperand. */
JitLirInstr jit_lir_operand_get_linked_instr(JitLirOperand op);

/* ---- MemoryIndirect accessors ---- */
typedef void* JitLirIndirect;
JitLirIndirect jit_lir_operand_get_indirect(JitLirOperand op);
JitLirOperand jit_lir_indirect_base_reg(JitLirIndirect ind);
JitLirOperand jit_lir_indirect_index_reg(JitLirIndirect ind);

/* ---- Block instruction removal ---- */
/* Remove all instructions for which is_live(instr, ctx) returns 0.
 * Handles C++ iterator invalidation internally. */
void jit_lir_block_remove_dead_instrs(
    JitLirBlock block,
    int (*is_live)(JitLirInstr instr, void *ctx),
    void *ctx);

/* FlagEffects constants (must match lir::FlagEffects enum) */
#define JIT_LIR_FLAG_NONE       0
#define JIT_LIR_FLAG_SET        1
#define JIT_LIR_FLAG_INVALIDATE 2

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* JIT_LIR_C_API_H */
