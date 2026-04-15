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

/* Create a LoadField instruction (Instr + string name + offset + type).
 * Allocates output register from func's env. Returns the instruction. */
HirInstr hir_c_create_load_field(HirFunction func, HirRegister receiver,
                                  const char *name, intptr_t offset,
                                  HirType type, int borrowed);

/* Create a GuardIs instruction (DeoptBase + PyObject* target).
 * Allocates output register from func's env. Returns the instruction. */
HirInstr hir_c_create_guard_is(HirFunction func, void *target,
                                HirRegister src);

/* Create a CheckNeg instruction (DeoptBase, no custom fields).
 * Allocates output register from func's env. Returns the instruction. */
HirInstr hir_c_create_check_neg(HirFunction func, HirRegister src,
                                 void *frame_state);

/* Create a PrimitiveBox instruction (DeoptBase + HirType).
 * Allocates output register from func's env. Returns the instruction. */
HirInstr hir_c_create_primitive_box(HirFunction func, HirRegister src,
                                     HirType type, void *frame_state);

/* Create a CheckSequenceBounds instruction (DeoptBase, 2 operands).
 * Allocates output register from func's env. Returns the instruction. */
HirInstr hir_c_create_check_seq_bounds(HirFunction func,
                                        HirRegister seq, HirRegister idx,
                                        void *frame_state);

/* Create a FloatBinaryOp instruction (DeoptBase + enum op).
 * Allocates output register from func's env. Returns the instruction. */
HirInstr hir_c_create_float_binary_op(HirFunction func, int32_t op_kind,
                                       HirRegister left, HirRegister right,
                                       void *frame_state);

/* Create a LongBinaryOp instruction (DeoptBase + enum op).
 * Allocates output register from func's env. Returns the instruction. */
HirInstr hir_c_create_long_binary_op(HirFunction func, int32_t op_kind,
                                      HirRegister left, HirRegister right,
                                      void *frame_state);

/* Create an IsNegativeAndErrOccurred instruction (DeoptBase, no custom fields).
 * Allocates output register from func's env. Returns the instruction. */
HirInstr hir_c_create_is_neg_and_err(HirFunction func, HirRegister src,
                                      void *frame_state);

/* DeoptBaseWithNameIdx factories (1-op + name_idx + FrameState) */
HirInstr hir_c_create_load_module_method_cached(HirFunction func,
    HirRegister receiver, int name_idx, void *frame_state);
HirInstr hir_c_create_load_method_cached(HirFunction func,
    HirRegister receiver, int name_idx, void *frame_state);
HirInstr hir_c_create_load_module_attr_cached(HirFunction func,
    HirRegister receiver, int name_idx, void *frame_state);
HirInstr hir_c_create_load_attr_cached(HirFunction func,
    HirRegister receiver, int name_idx, void *frame_state);
HirInstr hir_c_create_store_attr_cached(HirFunction func,
    HirRegister obj, HirRegister value, int name_idx, void *frame_state);

/* Tier 3: CheckField + LoadAttr + LoadArrayItem */
HirInstr hir_c_create_check_field(HirFunction func, HirRegister src,
    void *name, void *frame_state);
HirInstr hir_c_create_load_attr(HirFunction func, HirRegister receiver,
    int name_idx, void *frame_state, int already_optimized);
HirInstr hir_c_create_load_array_item(HirFunction func,
    HirRegister arr, HirRegister idx, HirRegister container,
    intptr_t offset, HirType type);

/* Set the guilty register on a DeoptBase instruction. */
void hir_c_set_guilty_reg(HirInstr instr, HirRegister reg);

/* Set the deopt description string on a DeoptBase instruction. */
void hir_c_set_descr(HirInstr instr, const char *descr);

/* Create a Guard instruction (DeoptBase, 1 operand, no output, no FrameState). */
HirInstr hir_c_create_guard(HirRegister src);

/* DeoptBaseWithNameIdx + cache_id factories */
HirInstr hir_c_create_fill_type_attr_cache(HirFunction func,
    HirRegister receiver, int name_idx, int cache_id, void *frame_state);
HirInstr hir_c_create_fill_type_method_cache(HirFunction func,
    HirRegister receiver, int name_idx, int cache_id, void *frame_state);

/* Simple DeoptBase factories (no custom fields beyond operands + FrameState) */
HirInstr hir_c_create_dict_subscr(HirFunction func, HirRegister lhs,
                                   HirRegister rhs, void *frame_state);
HirInstr hir_c_create_unicode_subscr(HirFunction func, HirRegister lhs,
                                      HirRegister rhs, void *frame_state);
HirInstr hir_c_create_unicode_repeat(HirFunction func, HirRegister lhs,
                                      HirRegister rhs, void *frame_state);
HirInstr hir_c_create_unicode_concat(HirFunction func, HirRegister lhs,
                                      HirRegister rhs, void *frame_state);
HirInstr hir_c_create_get_length(HirFunction func, HirRegister src,
                                  void *frame_state);
HirInstr hir_c_create_list_append(HirFunction func, HirRegister list,
                                   HirRegister item, void *frame_state);
HirInstr hir_c_create_is_instance(HirFunction func, HirRegister obj,
                                   HirRegister type, void *frame_state);
HirInstr hir_c_create_long_in_place_op(HirFunction func, int32_t op_kind,
                                        HirRegister left, HirRegister right,
                                        void *frame_state);

/* ---- Tier 5: Variable-arity + infrastructure factories ---- */

/* Create a VectorCall instruction (variadic DeoptBase).
 * Caller must wire operands via hir_c_set_operand after creation. */
HirInstr hir_c_create_vectorcall(HirFunction func, size_t n_operands,
                                  uint32_t flags, void *frame_state);

/* Create a CallStatic instruction (variadic, non-DeoptBase).
 * Caller must wire operands via hir_c_set_operand after creation. */
HirInstr hir_c_create_call_static(HirFunction func, size_t n_operands,
                                   void *addr, HirType ret_type);

/* Create a CallStatic instruction with caller-provided register. */
HirInstr hir_c_create_call_static_reg(size_t n_operands, HirRegister dst,
                                       void *addr, HirType ret_type);

/* Create a DeoptPatchpoint instruction (0 operands, DeoptBase). */
HirInstr hir_c_create_deopt_patchpoint(void *patcher);

/* Create a Snapshot instruction (0 operands, from FrameState). */
HirInstr hir_c_create_snapshot(void *frame_state);

/* Set suppress_exception_deopt flag on a DeoptBase instruction. */
void hir_c_set_suppress_exc_deopt(HirInstr instr, int val);

/* Set the output type on an instruction's output register. */
void hir_c_set_output_type(HirInstr instr, HirType type);

/* Create a Branch instruction (0 operands, 1 edge).
 * Uses C++ Edge::set_to for proper in_edges management. */
HirInstr hir_c_create_branch_cpp(void *target_block);

/* ---- Builder-style factories (caller provides dst register) ---- */

/* Create a LoadField instruction with caller-provided register.
 * No FrameState — builder sets it separately if needed. */
HirInstr hir_c_create_load_field_reg(HirRegister dst, HirRegister receiver,
                                      const char *name, intptr_t offset,
                                      HirType type, int borrowed);

/* Create a GuardType instruction with caller-provided register.
 * No FrameState — builder sets it separately if needed. */
HirInstr hir_c_create_guard_type_reg(HirRegister dst, HirType target,
                                      HirRegister src);

/* Create a RefineType instruction with caller-provided register. */
HirInstr hir_c_create_refine_type_reg(HirRegister dst, HirType type,
                                       HirRegister src);

/* Create a CheckExc instruction with caller-provided registers.
 * No FrameState — builder sets it separately. */
HirInstr hir_c_create_check_exc_reg(HirRegister dst, HirRegister src);

/* Create a Deopt instruction (0 operands, no output, DeoptBase). */
HirInstr hir_c_create_deopt(void);

/* Create a Return instruction (1 operand, no output). */
HirInstr hir_c_create_return(HirRegister src, HirType type);

/* Create a CondBranch instruction (1 operand, 2 edges).
 * Uses C++ Edge::set_to for proper in_edges management. */
HirInstr hir_c_create_cond_branch_cpp(void *cond_reg,
                                       void *true_block,
                                       void *false_block);

/* Create a VectorCall instruction with caller-provided register.
 * No FrameState — builder sets it separately. */
HirInstr hir_c_create_vectorcall_reg(size_t n_operands, HirRegister dst,
                                      uint32_t flags);

/* Create a CondBranchCheckType instruction (1 operand, 2 edges, Type field).
 * Uses C++ Edge::set_to for proper in_edges management. */
HirInstr hir_c_create_cond_branch_check_type_cpp(
    HirRegister target, HirType type,
    void *true_block, void *false_block);

/* ---- Builder-style DeoptBase factories (reg + FrameState) ---- */

HirInstr hir_c_create_check_seq_bounds_reg(HirRegister dst, HirRegister seq,
                                            HirRegister idx, void *frame_state);
HirInstr hir_c_create_check_field_reg(HirRegister dst, HirRegister src,
                                       void *name, void *frame_state);
HirInstr hir_c_create_binary_op_reg(HirRegister dst, int32_t op_kind,
                                     HirRegister left, HirRegister right,
                                     void *frame_state);
HirInstr hir_c_create_guard_is_reg(HirRegister dst, void *target,
                                    HirRegister src);
HirInstr hir_c_create_get_second_output_reg(HirRegister dst, HirType type,
                                             HirRegister src);
HirInstr hir_c_create_set_dict_item_reg(HirRegister dst, HirRegister dict,
                                         HirRegister key, HirRegister value,
                                         void *frame_state);
HirInstr hir_c_create_load_tuple_item_reg(HirRegister dst, HirRegister tuple,
                                           int32_t idx);
HirInstr hir_c_create_is_truthy_reg(HirRegister dst, HirRegister src,
                                     void *frame_state);
HirInstr hir_c_create_load_field_address_reg(HirRegister dst, HirRegister object,
                                              HirRegister offset);
HirInstr hir_c_create_yield_value_reg(HirRegister dst, HirRegister src,
                                       void *frame_state);
HirInstr hir_c_create_yield_from_reg(HirRegister dst, HirRegister send_value,
                                      HirRegister iter, void *frame_state);
HirInstr hir_c_create_check_var_reg(HirRegister dst, HirRegister src,
                                     void *name, void *frame_state);
HirInstr hir_c_create_set_current_awaiter_reg(HirRegister src);
HirInstr hir_c_create_decref_reg(HirRegister src);
HirInstr hir_c_create_make_cell_reg(HirRegister dst, HirRegister src,
                                     void *frame_state);
HirInstr hir_c_create_initial_yield_reg(HirRegister dst, void *frame_state);
HirInstr hir_c_create_load_arg_reg(HirRegister dst, int32_t idx, HirType type);
/* 1-op HasOutput DeoptBase factories (dst, src, frame) */
HirInstr hir_c_create_get_a_iter_reg(HirRegister dst, HirRegister src, void *fs);
HirInstr hir_c_create_get_a_next_reg(HirRegister dst, HirRegister src, void *fs);
HirInstr hir_c_create_get_tuple_reg(HirRegister dst, HirRegister src, void *fs);
HirInstr hir_c_create_is_neg_and_err_reg(HirRegister dst, HirRegister src, void *fs);
/* 0-op or 1-op no-frame factories */
HirInstr hir_c_create_load_cell_item_reg(HirRegister dst, HirRegister src);
HirInstr hir_c_create_load_current_func_reg(HirRegister dst);
HirInstr hir_c_create_load_eval_breaker_reg(HirRegister dst);
HirInstr hir_c_create_load_frame_reg(void);
HirInstr hir_c_create_load_var_object_size_reg(HirRegister dst, HirRegister src);
HirInstr hir_c_create_check_err_occurred_reg(void *frame_state);

/* Batch: simple DEFINE_SIMPLE_INSTR factories */
HirInstr hir_c_create_raise_reg(void *frame_state);
HirInstr hir_c_create_wait_handle_release_reg(HirRegister src);
HirInstr hir_c_create_make_set_reg(HirRegister dst, void *frame_state);
HirInstr hir_c_create_delete_attr_reg(HirRegister receiver, int32_t name_idx, void *fs);
HirInstr hir_c_create_delete_subscr_reg(HirRegister container, HirRegister sub, void *fs);
HirInstr hir_c_create_store_attr_reg(HirRegister receiver, HirRegister value, int32_t idx, void *fs);
HirInstr hir_c_create_swap_cell_item_reg(HirRegister dst, HirRegister cell, HirRegister value);
HirInstr hir_c_create_steal_cell_item_reg(HirRegister dst, HirRegister cell);
HirInstr hir_c_create_set_cell_item_reg(HirRegister cell, HirRegister value, HirRegister old);
HirInstr hir_c_create_at_quiescent_state_reg(void);
HirInstr hir_c_create_run_periodic_tasks_reg(HirRegister dst, void *fs);

/* Batch 2: more DEFINE_SIMPLE_INSTR factories */
HirInstr hir_c_create_wait_handle_load_waiter_reg(HirRegister dst, HirRegister src);
HirInstr hir_c_create_wait_handle_load_coro_reg(HirRegister dst, HirRegister src);
HirInstr hir_c_create_set_update_reg(HirRegister dst, HirRegister set, HirRegister iter, void *fs);
HirInstr hir_c_create_dict_update_reg(HirRegister dst, HirRegister dict, HirRegister update, void *fs);
HirInstr hir_c_create_list_extend_reg(HirRegister dst, HirRegister list, HirRegister iter, void *fs);
HirInstr hir_c_create_copy_dict_without_keys_reg(HirRegister dst, HirRegister subj, HirRegister keys, void *fs);
HirInstr hir_c_create_make_tuple_from_list_reg(HirRegister dst, HirRegister list, void *fs);
HirInstr hir_c_create_list_append_reg(HirRegister dst, HirRegister list, HirRegister item, void *fs);
HirInstr hir_c_create_check_freevar_reg(HirRegister dst, HirRegister src, void *name, void *fs);
HirInstr hir_c_create_load_global_reg(HirRegister dst, int32_t name_idx, void *fs);

/* Batch 3 */
HirInstr hir_c_create_dict_merge_reg(HirRegister dst, HirRegister dict, HirRegister update, HirRegister func, void *fs);
HirInstr hir_c_create_dict_subscr_reg(HirRegister dst, HirRegister dict, HirRegister key, void *fs);
HirInstr hir_c_create_send_reg(HirRegister iter, HirRegister vout, HirRegister vin, void *fs);
HirInstr hir_c_create_convert_value_reg(HirRegister dst, HirRegister value, int32_t conversion, void *fs);
HirInstr hir_c_create_unary_op_reg(HirRegister dst, int32_t op_kind, HirRegister operand, void *fs);
HirInstr hir_c_create_import_from_reg(HirRegister dst, HirRegister name, int32_t name_idx, void *fs);
HirInstr hir_c_create_invoke_iter_next_reg(HirRegister dst, HirRegister iter, void *fs);
HirInstr hir_c_create_primitive_unbox_reg(HirRegister dst, HirRegister src, HirType type);
/* Batch 4 */
HirInstr hir_c_create_make_tuple_reg(size_t n, HirRegister dst, void *fs);
HirInstr hir_c_create_make_list_reg(size_t n, HirRegister dst, void *fs);
HirInstr hir_c_create_tp_alloc_reg(HirRegister dst, void *pytype, void *fs);
HirInstr hir_c_create_unpack_ex_to_tuple_reg(HirRegister dst, HirRegister seq, int32_t before, int32_t after, void *fs);
HirInstr hir_c_create_load_method_reg(HirRegister dst, HirRegister receiver, int32_t name_idx, void *fs);
HirInstr hir_c_create_load_special_reg(HirRegister dst, HirRegister self, int32_t oparg, void *fs);
HirInstr hir_c_create_match_keys_reg(HirRegister dst, HirRegister subj, HirRegister keys, void *fs);
HirInstr hir_c_create_raise_awaitable_error_reg(HirRegister type, int32_t is_aenter, void *fs);
HirInstr hir_c_create_format_value_reg(HirRegister dst, HirRegister fmt, HirRegister val, int32_t conv, void *fs);
/* Batch 5 */
HirInstr hir_c_create_eager_import_name_reg(HirRegister dst, int32_t name_idx, HirRegister fromlist, HirRegister level, void *fs);
HirInstr hir_c_create_make_checked_dict_reg(HirRegister dst, int32_t size, HirType type, void *fs);
HirInstr hir_c_create_make_checked_list_reg(int32_t size, HirRegister dst, HirType type, void *fs);
HirInstr hir_c_create_make_function_reg(HirRegister dst, HirRegister code, HirRegister qualname, void *fs);
HirInstr hir_c_create_build_template_reg(HirRegister strings, HirRegister interps, HirRegister dst, void *fs);
HirInstr hir_c_create_build_interpolation_reg(HirRegister dst, HirRegister val, HirRegister str, HirRegister fmt, int32_t conv, void *fs);
HirInstr hir_c_create_load_attr_reg2(HirRegister dst, HirRegister receiver, int32_t name_idx, void *fs);
HirInstr hir_c_create_init_frame_cell_vars_reg(HirRegister func, int32_t nfree);

HirInstr hir_c_create_cond_branch_iter_not_done_cpp(
    HirRegister src, void *body_block, void *done_block);
HirInstr hir_c_create_int_convert_reg(HirRegister dst, HirRegister src,
                                       HirType type);
HirInstr hir_c_create_get_iter_reg(HirRegister dst, HirRegister src,
                                    void *frame_state);
HirInstr hir_c_create_store_subscr_reg(HirRegister container, HirRegister sub,
                                        HirRegister value, void *frame_state);
HirInstr hir_c_create_set_set_item_reg(HirRegister dst, HirRegister set,
                                        HirRegister item, void *frame_state);
HirInstr hir_c_create_in_place_op_reg(HirRegister dst, int32_t op_kind,
                                       HirRegister left, HirRegister right,
                                       void *frame_state);
HirInstr hir_c_create_compare_reg(HirRegister dst, int32_t op,
                                   HirRegister left, HirRegister right,
                                   void *frame_state);
HirInstr hir_c_create_format_with_spec_reg(HirRegister dst, HirRegister value,
                                            HirRegister fmt_spec, void *frame_state);
HirInstr hir_c_create_make_dict_reg(HirRegister dst, int32_t dict_size,
                                     void *frame_state);
HirInstr hir_c_create_set_function_attr_reg(HirRegister value, HirRegister base,
                                             int32_t field);
HirInstr hir_c_create_check_neg_reg(HirRegister dst, HirRegister src,
                                     void *frame_state);
HirInstr hir_c_create_get_length_reg(HirRegister dst, HirRegister src,
                                      void *frame_state);
HirInstr hir_c_create_primitive_box_reg(HirRegister dst, HirRegister src,
                                         HirType type, void *frame_state);
HirInstr hir_c_create_load_array_item_reg(HirRegister dst, HirRegister arr,
                                           HirRegister idx, HirRegister container,
                                           intptr_t offset, HirType type);

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

/* Create a BinaryOp instruction (DeoptBase + enum op).
 * Allocates output register from func's env. Returns the instruction. */
HirInstr hir_c_create_binary_op(HirFunction func, int32_t op_kind,
                                HirRegister left, HirRegister right,
                                void *frame_state);

/* Create a GuardType instruction (DeoptBase + HirType target).
 * Allocates output register from func's env. Returns the instruction. */
HirInstr hir_c_create_guard_type(HirFunction func, HirType target,
                                 HirRegister src, void *frame_state);

/* Create a CheckExc instruction (DeoptBase, no custom fields).
 * Allocates output register from func's env. Returns the instruction. */
HirInstr hir_c_create_check_exc(HirFunction func, HirRegister src,
                                void *frame_state);

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
