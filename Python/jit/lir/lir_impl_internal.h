/*
 * lir_impl_internal.h -- Forward declarations for Phase B impl files
 *
 * Each _impl.c file includes this to get cross-file lifecycle function
 * declarations. NOT for external use.
 */

#ifndef JIT_LIR_IMPL_INTERNAL_H
#define JIT_LIR_IMPL_INTERNAL_H

#include "cinderx/Jit/lir/lir_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* From type.c */
size_t jit_lir_bit_size(int dt);

/* From operand_impl.c */
LirOperand *lir_operand_new(LirInstruction *parent);
LirOperand *lir_operand_new_linked(LirInstruction *parent,
                                   LirInstruction *def_instr);
void lir_operand_free(LirOperand *op);
void lir_operand_set_constant(LirOperand *op, uint64_t val, uint8_t dt);
void lir_operand_set_fp_constant(LirOperand *op, double val);
void lir_operand_set_phy_register(LirOperand *op, LirPhyLocation reg);
void lir_operand_set_stack_slot(LirOperand *op, LirPhyLocation slot);
void lir_operand_set_mem_address(LirOperand *op, void *addr);
void lir_operand_set_basic_block(LirOperand *op, void *block);
void lir_operand_set_linked_instr(LirOperand *op, LirInstruction *def);

/* From lir_instruction.c */
void lir_instruction_destroy(LirInstruction *inst);
void lir_instruction_free(LirInstruction *inst);
void lir_instruction_ensure_input_capacity(LirInstruction *inst, size_t needed);
LirOperand *lir_instruction_alloc_imm_input(LirInstruction *inst,
                                             uint64_t val, int dt);
LirOperand *lir_instruction_alloc_fp_imm_input(LirInstruction *inst, double val);
LirOperand *lir_instruction_alloc_linked_input(LirInstruction *inst,
                                                LirInstruction *def_instr);
LirOperand *lir_instruction_alloc_phyreg_input(LirInstruction *inst,
                                                LirPhyLocation loc);
LirOperand *lir_instruction_alloc_stack_input(LirInstruction *inst,
                                               LirPhyLocation loc);
LirOperand *lir_instruction_alloc_addr_input(LirInstruction *inst, void *addr);
LirOperand *lir_instruction_alloc_label_input(LirInstruction *inst,
                                               LirBasicBlock *block);
LirOperand *lir_instruction_alloc_phyreg_or_stack_input(LirInstruction *inst,
                                                         LirPhyLocation loc);
void lir_instruction_set_input(LirInstruction *inst, size_t i,
                                LirOperand *input);
LirOperand *lir_instruction_remove_input(LirInstruction *inst, size_t index);
LirOperand *lir_instruction_release_input(LirInstruction *inst, size_t index);
LirOperand *lir_instruction_append_input(LirInstruction *inst,
                                          LirOperand *operand);
LirOperand *lir_instruction_prepend_input(LirInstruction *inst,
                                           LirOperand *operand);
LirOperand *lir_instruction_get_operand_by_predecessor(
    const LirInstruction *inst, const LirBasicBlock *pred);

/* From block_impl.c */
LirBasicBlock *lir_block_new(void *function, int id);
void lir_block_destroy(LirBasicBlock *bb);
void lir_block_free(LirBasicBlock *bb);
void lir_block_set_successor(LirBasicBlock *bb, size_t index,
                              LirBasicBlock *new_succ);
void lir_block_fixup_phis(LirBasicBlock *bb,
                          LirBasicBlock *old_pred, LirBasicBlock *new_pred);
void lir_block_append_instr(LirBasicBlock *bb, LirInstruction *instr);
void lir_block_insert_instr_before(LirBasicBlock *bb, LirInstruction *before,
                                   LirInstruction *instr);
LirInstruction *lir_block_remove_instr(LirBasicBlock *bb, LirInstruction *instr);
LirBasicBlock *lir_block_insert_between(LirBasicBlock *bb,
                                         LirBasicBlock *succ_block);
LirBasicBlock *lir_block_split_before(LirBasicBlock *bb,
                                       LirInstruction *instr);

/* From function_impl.c */
void lir_function_destroy(LirFunction *func);
LirBasicBlock *lir_function_alloc_block(LirFunction *func);
LirBasicBlock *lir_function_alloc_block_after(LirFunction *func,
                                               LirBasicBlock *after);
int lir_function_allocate_id(LirFunction *func);
void lir_function_ensure_block_capacity(LirFunction *func, size_t needed);
void lir_function_sort_blocks(LirFunction *func);

/* From lir_c_api.h (blocksorter) */
JitLirBlock *jit_lir_sort_blocks_rpo(
    JitLirBlock *blocks, size_t count, size_t *out_count);

/* From lir_memind */
void lir_memind_free(LirMemoryIndirect *mi);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* JIT_LIR_IMPL_INTERNAL_H */
