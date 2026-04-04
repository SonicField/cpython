/*
 * lir_c_api.h -- C-compatible accessor API for LIR types
 *
 * Phase 3D: Function declarations for C callers of LIR operations.
 * Struct definitions are in lir_types_c.h (single source of truth).
 *
 * Only functions with active .c callers are declared here.
 * Do NOT add speculative wrapper functions — convert the underlying
 * C++ to C instead (Phase 3D directive).
 */

#ifndef JIT_LIR_C_API_H
#define JIT_LIR_C_API_H

#include "cinderx/Jit/lir/lir_types_c.h"

#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handles for LIR types (used by files that don't need struct access) */
typedef void* JitLirFunc;
typedef void* JitLirBlock;
typedef void* JitLirInstr;
typedef void* JitLirOperand;

/* ---- Opaque pointer API (functions with active .c callers) ---- */

/* Function accessors */
size_t jit_lir_func_num_blocks(JitLirFunc func);
JitLirBlock jit_lir_func_get_block(JitLirFunc func, size_t index);
JitLirBlock jit_lir_func_entry_block(JitLirFunc func);

/* BasicBlock accessors */
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

/* Instruction accessors */
int jit_lir_instr_opcode(JitLirInstr instr);
int jit_lir_instr_is_branch(JitLirInstr instr);
int jit_lir_instr_is_branch_cc(JitLirInstr instr);
int jit_lir_instr_is_any_branch(JitLirInstr instr);
int jit_lir_instr_is_terminator(JitLirInstr instr);
JitLirOperand jit_lir_instr_get_input(JitLirInstr instr, size_t index);
JitLirOperand jit_lir_instr_output(JitLirInstr instr);

/* Operand accessors */
JitLirBlock jit_lir_operand_get_basic_block(JitLirOperand op);

/* Opcode constants */
int jit_lir_opcode_guard(void);

/* DCE instruction accessors */
int jit_lir_instr_id(JitLirInstr instr);
int jit_lir_instr_is_essential(JitLirInstr instr);
int jit_lir_instr_flag_effects(JitLirInstr instr);

/* Iterate all input operands of an instruction. */
void jit_lir_instr_foreach_input(
    JitLirInstr instr,
    void (*cb)(JitLirOperand operand, void *ctx),
    void *ctx);

/* Extended operand accessors */
int jit_lir_operand_is_reg(JitLirOperand op);
int jit_lir_operand_is_stack(JitLirOperand op);
int jit_lir_operand_is_mem(JitLirOperand op);
int jit_lir_operand_is_ind(JitLirOperand op);
int jit_lir_operand_is_linked(JitLirOperand op);

/* Get the defining instruction of a LinkedOperand. */
JitLirInstr jit_lir_operand_get_linked_instr(JitLirOperand op);

/* MemoryIndirect accessors */
typedef void* JitLirIndirect;
JitLirIndirect jit_lir_operand_get_indirect(JitLirOperand op);
JitLirOperand jit_lir_indirect_base_reg(JitLirIndirect ind);
JitLirOperand jit_lir_indirect_index_reg(JitLirIndirect ind);

/* Block instruction removal.
 * Remove all instructions for which is_live(instr, ctx) returns 0.
 * Handles C++ iterator invalidation internally. */
void jit_lir_block_remove_dead_instrs(
    JitLirBlock block,
    int (*is_live)(JitLirInstr instr, void *ctx),
    void *ctx);

/* ---- Phase B1: LirOperand C struct operations ---- */
/* These operate directly on LirOperand/LirMemoryIndirect structs.
 * New C code should use these directly rather than the opaque
 * jit_lir_* wrappers above. */
LirOperand *lir_operand_new(LirInstruction *parent);
LirOperand *lir_operand_new_linked(LirInstruction *parent,
                                   LirInstruction *def_instr);
void lir_operand_free(LirOperand *op);

LirMemoryIndirect *lir_memind_new(LirInstruction *parent);
void lir_memind_free(LirMemoryIndirect *mi);

uint8_t lir_operand_type(const LirOperand *op);
uint8_t lir_operand_data_type(const LirOperand *op);
int lir_operand_is_linked(const LirOperand *op);
int lir_operand_is_fp(const LirOperand *op);
int lir_operand_is_last_use(const LirOperand *op);
size_t lir_operand_size_in_bits(const LirOperand *op);
LirInstruction *lir_operand_instr(const LirOperand *op);

uint64_t lir_operand_get_constant(const LirOperand *op);
double lir_operand_get_fp_constant(const LirOperand *op);
LirPhyLocation lir_operand_get_phy_register(const LirOperand *op);
LirPhyLocation lir_operand_get_stack_slot(const LirOperand *op);
void *lir_operand_get_mem_address(const LirOperand *op);
void *lir_operand_get_basic_block(const LirOperand *op);
LirMemoryIndirect *lir_operand_get_indirect(const LirOperand *op);
LirOperand *lir_operand_get_define(LirOperand *op);
uint64_t lir_operand_get_constant_or_address(const LirOperand *op);
LirInstruction *lir_operand_get_linked_instr(const LirOperand *op);

void lir_operand_set_constant(LirOperand *op, uint64_t val, uint8_t dt);
void lir_operand_set_fp_constant(LirOperand *op, double val);
void lir_operand_set_phy_register(LirOperand *op, LirPhyLocation reg);
void lir_operand_set_stack_slot(LirOperand *op, LirPhyLocation slot);
void lir_operand_set_mem_address(LirOperand *op, void *addr);
void lir_operand_set_basic_block(LirOperand *op, void *block);
void lir_operand_set_virtual_register(LirOperand *op);
void lir_operand_set_data_type(LirOperand *op, uint8_t dt);
void lir_operand_set_last_use(LirOperand *op);
void lir_operand_set_none(LirOperand *op);
void lir_operand_set_linked_instr(LirOperand *op, LirInstruction *def);
void lir_operand_set_phy_reg_or_stack(LirOperand *op, LirPhyLocation loc);
LirPhyLocation lir_operand_get_phy_reg_or_stack(const LirOperand *op);
void lir_operand_set_memory_indirect_instr(LirOperand *op,
                                            LirInstruction *base, int32_t offset);
void lir_operand_set_memory_indirect_phy(LirOperand *op,
                                          LirPhyLocation base, int32_t offset);
void lir_operand_set_memory_indirect_phy3(LirOperand *op,
                                           LirPhyLocation base,
                                           LirPhyLocation index_reg,
                                           uint8_t multiplier);
LirOperand *lir_operand_new_with_type(LirInstruction *parent, uint8_t data_type,
                                       uint8_t type, uint64_t data);
LirOperand *lir_operand_new_fp(LirInstruction *parent, uint8_t type, double data);
LirOperand *lir_operand_new_copy(LirInstruction *parent, const LirOperand *src);
LirOperand *lir_memind_base_reg(const LirMemoryIndirect *mi);
LirOperand *lir_memind_index_reg(const LirMemoryIndirect *mi);
uint8_t lir_memind_multiplier(const LirMemoryIndirect *mi);
int32_t lir_memind_offset(const LirMemoryIndirect *mi);

void lir_memind_set(LirMemoryIndirect *mi,
                    LirPhyLocation base, LirPhyLocation index,
                    uint8_t multiplier, int32_t offset);
void lir_memind_set_linked(LirMemoryIndirect *mi,
                           LirInstruction *base, LirInstruction *index,
                           uint8_t multiplier, int32_t offset);

/* Instruction lifecycle and manipulation (lir_instruction.c) */
LirInstruction *lir_instruction_create(LirBasicBlock *bb, int opcode,
                                        const void *origin);
void lir_instruction_free(LirInstruction *inst);
int lir_instruction_id(const LirInstruction *inst);
int lir_instruction_opcode(const LirInstruction *inst);
LirOperand *lir_instruction_output(LirInstruction *inst);
size_t lir_instruction_num_inputs(const LirInstruction *inst);
LirOperand *lir_instruction_get_input(const LirInstruction *inst, size_t index);
LirBasicBlock *lir_instruction_basic_block(const LirInstruction *inst);
void lir_instruction_set_opcode(LirInstruction *inst, int opcode);
void lir_instruction_set_id(LirInstruction *inst, int id);
void lir_instruction_set_basic_block(LirInstruction *inst, LirBasicBlock *bb);
void lir_instruction_set_input(LirInstruction *inst, size_t i,
                                LirOperand *input);
LirOperand *lir_instruction_append_input(LirInstruction *inst,
                                          LirOperand *operand);
LirOperand *lir_instruction_release_input(LirInstruction *inst, size_t index);
LirOperand *lir_instruction_remove_input(LirInstruction *inst, size_t index);
LirOperand *lir_instruction_prepend_input(LirInstruction *inst,
                                           LirOperand *operand);
void lir_instruction_set_num_inputs(LirInstruction *inst, size_t n);
int lir_instruction_get_num_outputs(const LirInstruction *inst);
LirOperand *lir_instruction_get_operand_by_predecessor(
    const LirInstruction *inst, const LirBasicBlock *pred);
LirOperand *lir_instruction_alloc_imm_input(LirInstruction *inst,
                                             uint64_t val, int dt);
LirOperand *lir_instruction_alloc_fp_imm_input(LirInstruction *inst,
                                                double val);
LirOperand *lir_instruction_alloc_linked_input(LirInstruction *inst,
                                                LirInstruction *def_instr);
LirOperand *lir_instruction_alloc_phyreg_input(LirInstruction *inst,
                                                LirPhyLocation loc);
LirOperand *lir_instruction_alloc_stack_input(LirInstruction *inst,
                                               LirPhyLocation loc);
LirOperand *lir_instruction_alloc_label_input(LirInstruction *inst,
                                               LirBasicBlock *block);
LirOperand *lir_instruction_alloc_addr_input(LirInstruction *inst,
                                              void *addr);
void lir_instruction_foreach_input(const LirInstruction *inst,
                                    void (*cb)(LirOperand *, void *),
                                    void *ctx);

/* Allocate instruction before position in block (C equivalent of
 * BasicBlock::allocateInstrBefore with no variadic args) */
LirInstruction *lir_block_alloc_instr_before(LirBasicBlock *bb,
                                              LirInstruction *before,
                                              int opcode);

/* LIR inliner (C wrapper around C++ LIRInliner::inlineCalls) */
int lir_inliner_inline_calls(void *func);

/* Opcode query functions (take opcode int, not LirInstruction*) */
int lir_instruction_is_compare(int opcode);
int lir_instruction_is_branch_cc(int opcode);
int lir_instruction_is_any_branch(int opcode);
int lir_instruction_is_terminator(int opcode);
int lir_instruction_is_any_yield(int opcode);

/* Opcode manipulation */
int lir_instruction_negate_branch_cc(int opcode);
int lir_instruction_flip_branch_cc_direction(int opcode);
int lir_instruction_flip_comparison_direction(int opcode);
int lir_instruction_compare_to_branch_cc(int opcode);

/* BasicBlock lifecycle (block_impl.c) */
LirBasicBlock *lir_block_create(void *function, int id);
void lir_block_free(LirBasicBlock *bb);
void lir_block_add_successor(LirBasicBlock *bb, LirBasicBlock *succ);
LirInstruction *lir_block_alloc_instr(LirBasicBlock *bb, int opcode,
                                       const void *origin);

/* Function lifecycle (function_impl.c) */
LirFunction *lir_function_create(const void *hir_func);
void lir_function_free(LirFunction *func);
LirBasicBlock *lir_function_alloc_block(LirFunction *func);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* JIT_LIR_C_API_H */
