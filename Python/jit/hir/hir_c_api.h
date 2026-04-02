/*
 * hir_c_api.h -- C-compatible accessor API for HIR types
 *
 * Phase 3D: Provides opaque pointer accessors for HIR Function, CFG,
 * BasicBlock, Instr, and Register so that optimization passes can be
 * written in pure C.
 *
 * Only functions with active .c callers are declared here.
 * Do NOT add speculative wrapper functions — convert the underlying
 * C++ to C instead (Phase 3D directive).
 */

#ifndef JIT_HIR_C_API_H
#define JIT_HIR_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Opaque handle types ---- */
typedef void* HirFunction;
typedef void* HirCFG;
typedef void* HirBasicBlock;
typedef void* HirInstr;
typedef void* HirRegister;

/* ---- AliasClass constants ----
 * AEmpty is the identity element — no memory effects. */
#define HIR_ACLS_EMPTY 0

/* ---- Function / CFG accessors ---- */

/* Get the CFG from a function. */
HirCFG hir_func_cfg(HirFunction func);

/* Get blocks in reverse postorder. Caller provides output array.
 * Returns number of blocks written (clamped to capacity). */
size_t hir_cfg_get_rpo(HirCFG cfg, HirBasicBlock *out, size_t capacity);

/* Linked-list iteration over all blocks in CFG order.
 * hir_cfg_blocks_first returns the first block (or NULL if empty).
 * hir_cfg_blocks_next returns the next block (or NULL at end). */
HirBasicBlock hir_cfg_blocks_first(HirCFG cfg);
HirBasicBlock hir_cfg_blocks_next(HirCFG cfg, HirBasicBlock block);

/* ---- BasicBlock accessors ---- */

int hir_block_empty(HirBasicBlock block);

/* Instruction iteration within a block.
 * hir_block_first returns first instr (or NULL if empty).
 * hir_block_next returns next instr (or NULL at end).
 * Safe to call hir_instr_unlink on current instr if you advance first. */
HirInstr hir_block_first(HirBasicBlock block);
HirInstr hir_block_next(HirBasicBlock block, HirInstr instr);

/* Get the terminator (last instruction) of the block. */
HirInstr hir_block_terminator(HirBasicBlock block);

/* Append an instruction to the end of the block. Returns the instr. */
HirInstr hir_block_append(HirBasicBlock block, HirInstr instr);

/* Remove and return the first instruction. */
HirInstr hir_block_pop_front(HirBasicBlock block);

/* Number of incoming edges. */
size_t hir_block_in_edges_count(HirBasicBlock block);

/* Update Phi instructions: replace old_pred with new_pred. */
void hir_block_fixup_phis(HirBasicBlock block,
                          HirBasicBlock old_pred,
                          HirBasicBlock new_pred);

/* ---- Instruction predicates ---- */

int hir_instr_is_terminator(HirInstr instr);
int hir_instr_is_snapshot(HirInstr instr);
int hir_instr_is_phi(HirInstr instr);
int hir_instr_is_assign(HirInstr instr);
int hir_instr_is_primitive_box(HirInstr instr);
int hir_instr_is_branch(HirInstr instr);

/* Returns 1 if the instruction is a DeoptBase subclass, 0 otherwise. */
int hir_instr_has_deopt_base(HirInstr instr);

/* ---- Instruction accessors ---- */

/* Get the output register (may be NULL for side-effect-only instrs). */
HirRegister hir_instr_output(HirInstr instr);

/* Control flow edges. */
size_t hir_instr_num_edges(HirInstr instr);
HirBasicBlock hir_instr_successor(HirInstr instr, size_t index);

/* ---- Instruction mutation ---- */

/* Remove instruction from its block (does not free). */
void hir_instr_unlink(HirInstr instr);

/* Insert instr immediately before 'before'. */
void hir_instr_insert_before(HirInstr instr, HirInstr before);

/* Copy bytecode offset from src to dst. */
void hir_instr_copy_bytecode_offset(HirInstr dst, HirInstr src);

/* Free an unlinked instruction. Caller must unlink first. */
void hir_instr_delete(HirInstr instr);

/* ---- Operand use visitation ----
 *
 * Visits each operand register of the instruction. The callback receives
 * a pointer to the register slot (HirRegister*) so it can mutate in place
 * (for copy propagation / register replacement).
 *
 * Callback returns 1 to continue, 0 to stop early.
 * ctx is passed through to the callback unchanged. */
void hir_instr_visit_uses(HirInstr instr,
                          int (*callback)(HirRegister *reg_slot, void *ctx),
                          void *ctx);

/* ---- Branch-specific ---- */

/* Get the target block of a Branch instruction. */
HirBasicBlock hir_branch_target(HirInstr branch);

/* ---- Register accessors ---- */

/* Get the instruction that defines this register. */
HirInstr hir_reg_instr(HirRegister reg);

/* Follow Assign chains to the original value. */
HirRegister hir_chase_assign(HirRegister reg);

/* ---- Phi-specific ---- */

/* If phi is trivial (all inputs are the same value), returns that value.
 * Returns NULL if the phi is non-trivial. */
HirRegister hir_phi_is_trivial(HirInstr phi);

/* ---- Factory functions ---- */

/* Create a LoadConst instruction producing TBottom (unreachable value). */
HirInstr hir_load_const_bottom_create(HirRegister output);

/* Create an Assign instruction (copy register). */
HirInstr hir_assign_create(HirRegister output, HirRegister value);

/* ---- Memory effects ---- */

/* Returns the may_store AliasClass bitmask. 0 means no stores. */
int hir_memory_effects_may_store(HirInstr instr);

/* ---- CFG / pass utilities ---- */

/* Remove trampoline blocks (single unconditional jumps). Returns 1 if changed. */
int hir_remove_trampoline_blocks(HirCFG cfg);

/* Remove unreachable blocks from function. Returns 1 if changed. */
int hir_remove_unreachable_blocks(HirFunction func);

/* Remove unreachable instructions. */
void hir_remove_unreachable_instructions(HirFunction func);

/* Re-derive all register types from instructions. */
void hir_reflow_types(HirFunction func);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* JIT_HIR_C_API_H */
