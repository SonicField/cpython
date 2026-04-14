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

#include "cinderx/Jit/hir/hir_type_c.h"

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

/* Get the function's full qualified name (C string, valid until function freed). */
const char *hir_func_fullname(HirFunction func);

/* Allocate a new Register from the function's Environment.
 * Returns an opaque HirRegister handle. */
HirRegister hir_func_alloc_register(HirFunction func);

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

/* Append instruction to block and set its bytecode offset.
 * Convenience for factory functions (alloc + init + append). */
HirInstr hir_block_append_at(HirBasicBlock block, HirInstr instr, int32_t bc_off);

/* Remove and return the first instruction. */
HirInstr hir_block_pop_front(HirBasicBlock block);

/* Number of incoming edges. */
size_t hir_block_in_edges_count(HirBasicBlock block);

/* Update Phi instructions: replace old_pred with new_pred. */
void hir_block_fixup_phis(HirBasicBlock block,
                          HirBasicBlock old_pred,
                          HirBasicBlock new_pred);

/* ---- Instruction predicates ----
 * Most predicates moved to hir_instr_c.h as hir_c_is_* inline functions
 * (direct opcode field read, no C++ bridge). Only complex predicates
 * that need C++ internals remain here. */

/* ---- Instruction accessors ---- */
/* hir_instr_output moved to hir_instr_c.h as hir_c_output */

/* Control flow edges. */
size_t hir_instr_num_edges(HirInstr instr);
HirBasicBlock hir_instr_successor(HirInstr instr, size_t index);

/* ---- Instruction mutation ---- */

/* Remove instruction from its block (does not free). */
void hir_instr_unlink(HirInstr instr);

/* Insert instr immediately before 'before'. */
void hir_instr_insert_before(HirInstr instr, HirInstr before);

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

/* Visit DeoptBase/Snapshot extension uses (frame_state, live_regs, guilty_reg).
 * Called by hir_c_visit_uses after operand array iteration.
 * For Snapshot: visits frame_state only.
 * For DeoptBase: visits frame_state + live_regs + guilty_reg.
 * Uses void** to match hir_instr_c.h's hir_c_visit_uses signature. */
int hir_c_visit_deopt_extension(void *instr,
                                int (*callback)(void **reg_slot, void *ctx),
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

/* ---- Instruction query/mutation (T2-D) ---- */

/* Get the Compare operation kind (CompareOp enum as int).
 * Kept for assertion wrappers — C consumers also have hir_c_compare_op. */
int hir_instr_compare_op(HirInstr instr);

/* Check if instruction is replayable (can be safely re-executed). */
int hir_instr_is_replayable(HirInstr instr);

/* Check if instruction uses the given register as an input. */
int hir_instr_uses_reg(HirInstr instr, HirRegister reg);

/* Replace instruction with another (inserts before, unlinks original). */
void hir_instr_replace_with(HirInstr old_instr, HirInstr new_instr);

/* Get the last instruction in a block. Returns NULL if empty. */
HirInstr hir_block_back(HirBasicBlock block);

/* Get an instruction's block. */
HirBasicBlock hir_instr_block(HirInstr instr);

/* Get an instruction's operand by index. */
HirRegister hir_instr_get_operand(HirInstr instr, size_t i);

/* ---- Factory functions (T2-D) ---- */

/* Create a CompareBool instruction. */
HirInstr hir_compare_bool_create(
    HirRegister output, int compare_op,
    HirRegister left, HirRegister right,
    HirInstr frame_state_source);

/* ---- Frame state ---- */

/* Get the dominating FrameState for an instruction (opaque, for passing
 * to factory functions). Returns NULL if none. */
void *hir_get_frame_state(HirInstr instr);

/* ---- Analysis utilities (T2-D Tier 1) ---- */

/* Check if instruction is a passthrough (Assign, BitCast, etc.). */
int hir_is_passthrough(HirInstr instr);

/* Check if an operand constraint requires exact type match (primitives). */
int hir_operands_must_match(HirInstr instr, size_t operand_idx);

/* Check if a register's type satisfies the expected operand constraint. */
int hir_register_type_matches_operand(HirInstr instr, size_t operand_idx, HirRegister reg);

/* Check if a given type satisfies the expected operand constraint.
 * Like hir_register_type_matches_operand but takes an explicit type
 * instead of reading from the register. */
int hir_type_matches_operand(HirInstr instr, size_t operand_idx,
                             const HirType *type);

/* ---- Instruction count ---- */

/* Number of operands. */
size_t hir_instr_num_operands(HirInstr instr);

/* ---- Register type ---- */

/* Get the type of a register (returned by value as HirType). */
HirType hir_register_type(HirRegister reg);

/* GuardType predicate moved to hir_instr_c.h as hir_c_is_guard_type */

/* ---- RegUses (opaque handle) ---- */
typedef void* HirRegUses;

/* Collect direct register uses for all registers in a function.
 * Caller must destroy with hir_reg_uses_destroy. */
HirRegUses hir_collect_reg_uses(HirFunction func);

/* Returns 1 if the register has any uses. */
int hir_reg_uses_contains(HirRegUses uses, HirRegister reg);

/* Number of instructions that use this register. */
size_t hir_reg_uses_count(HirRegUses uses, HirRegister reg);

/* Get the i-th instruction that uses this register. */
HirInstr hir_reg_uses_get(HirRegUses uses, HirRegister reg, size_t idx);

/* Free the RegUses handle. */
void hir_reg_uses_destroy(HirRegUses uses);

/* ---- outputType with override ---- */

/* Compute the output type of an instruction, but override one operand's
 * type at override_idx with override_type. Used by guard removal to
 * compute what the output type would be if a guard were relaxed. */
HirType hir_output_type_with_override(HirInstr instr,
                                      size_t override_idx,
                                      const HirType *override_type);

/* hir_instr_opname moved to hir_instr_c.h as hir_instr_info_name(hir_c_opcode()) */

/* ---- Type to string ---- */

/* Write a human-readable type string into buf (NUL-terminated).
 * Returns the number of chars written (excluding NUL), or the
 * number that would have been written if buf were large enough.
 * safe=1 uses toStringSafe (no GIL needed). */
size_t hir_type_to_string(const HirType *type, char *buf, size_t bufsz,
                          int safe);

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
