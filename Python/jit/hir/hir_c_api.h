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
#include "cinderx/Jit/hir/hir_instr_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Opaque handle types ----
 *
 * W25b canonical: HirFunction, HirInstr, HirRegister are typed pointers.
 * Pre-W25b these were typedef void * and C99 6.3.2.3 silently masked
 * cross-handle drift. Post-W25b:
 *
 *   - C consumers see distinct opaque struct-pointer types
 *     (struct HirFunctionOpaque *, etc.). Cross-handle drift on .c-callable
 *     bridges is caught at compile time.
 *   - C++ consumers see direct pointers to the underlying jit::hir classes.
 *     This preserves existing C++ caller patterns and implicit conversions
 *     between Instr / Register / Function pointers and the handle typedefs,
 *     while still keeping cross-handle drift caught (C++ also disallows
 *     implicit conversion between unrelated pointer types).
 *
 * The C-visible struct tags are forward-declared only and never defined;
 * the C++ typedefs alias to the canonical jit::hir classes directly so that
 * existing C++ code does not need per-call casts.
 *
 * See docs/w25b-typedef-promotion.md. */
#ifdef __cplusplus
/* C++ side: keep void* for backward compatibility with the existing
 * cast-heavy hir_c_api.cpp implementation. C++ already has type-safety on
 * the canonical jit::hir::Instr / Register / Function classes — the W25b
 * drift class is C99-§6.3.2.3-specific (implicit void* <-> object pointer
 * conversion in C). C++ does not have the equivalent silent-conversion
 * surface, so the .c-side struct-typed handles are sufficient. */
typedef void *HirFunction;
typedef void *HirInstr;
typedef void *HirRegister;
#else
struct HirFunctionOpaque;
struct HirInstrOpaque;
struct HirRegisterOpaque;
typedef struct HirFunctionOpaque *HirFunction;
typedef struct HirInstrOpaque *HirInstr;
typedef struct HirRegisterOpaque *HirRegister;
#endif

/* W25 canonical: HirBasicBlock + HirCFG are forward-declared structs.
 * Full layout in hir_basic_block_c.h; API consumers see only the opaque
 * pointer. Forward decl satisfies API-only TUs; full def satisfies field-
 * access TUs. Both can include both headers without typedef collision. */
struct HirBasicBlock;
struct HirCFG;

/* ---- AliasClass constants ----
 * AEmpty is the identity element — no memory effects. */
#define HIR_ACLS_EMPTY 0

/* ---- Function / CFG accessors ---- */

/* Get the CFG from a function. */
struct HirCFG *hir_func_cfg(HirFunction func);

/* Get the function's full qualified name (C string, valid until function freed). */
const char *hir_func_fullname(HirFunction func);

/* Allocate a new Register from the function's Environment.
 * Returns an opaque HirRegister handle. */
HirRegister hir_func_alloc_register(HirFunction func);

/* Register a PyObject* reference to keep alive during compilation.
 * Returns the same object (for chaining). */
PyObject* hir_func_add_reference(HirFunction func, PyObject* obj);

/* Allocate a type method cache slot, returns cache_id */
int hir_func_env_allocate_type_method_cache(HirFunction func);

/* Allocate a type attr cache slot, returns cache_id */
int hir_func_env_allocate_type_attr_cache(HirFunction func);

/* Allocate a TypeAttrDeoptPatcher for use with DeoptPatchpoint */
void *hir_func_allocate_type_attr_deopt_patcher(
    HirFunction func, void *type, void *attr_name, void *method);

/* Get the name object from a FrameState's code->co_names at name_idx */
PyObject *hir_frame_state_get_name(void *frame_state, int name_idx);

/* Allocate a GlobalDeoptPatcher (watches global rebinding) */
void *hir_func_allocate_global_deopt_patcher(
    HirFunction func, void *globals, void *key_name, void *expected);

/* Allocate a TypeDeoptPatcher (watches any type modification) */
void *hir_func_allocate_type_deopt_patcher(HirFunction func, void *type);

/* Allocate a SplitDictDeoptPatcher for use with DeoptPatchpoint */
void *hir_func_allocate_split_dict_deopt_patcher(
    HirFunction func, void *type, void *attr_name, void *keys);

/* Ensure a preloader exists for a Python function (find+create if needed) */
void hir_preloader_ensure(void *py_func);

/* Get blocks in reverse postorder. Caller provides output array.
 * Returns number of blocks written (clamped to capacity). */
size_t hir_cfg_get_rpo(struct HirCFG *cfg, struct HirBasicBlock **out, size_t capacity);

/* Linked-list iteration over all blocks in CFG order.
 * hir_cfg_blocks_first returns the first block (or NULL if empty).
 * hir_cfg_blocks_next returns the next block (or NULL at end). */
struct HirBasicBlock *hir_cfg_blocks_first(struct HirCFG *cfg);
struct HirBasicBlock *hir_cfg_blocks_next(struct HirCFG *cfg, struct HirBasicBlock *block);

/* ---- BasicBlock accessors ---- */

int hir_block_empty(struct HirBasicBlock *block);
int hir_block_id(struct HirBasicBlock *block);

/* Instruction iteration within a block.
 * hir_block_first returns first instr (or NULL if empty).
 * hir_block_next returns next instr (or NULL at end).
 * Safe to call hir_instr_unlink on current instr if you advance first. */
HirInstr hir_block_first(struct HirBasicBlock *block);
HirInstr hir_block_next(struct HirBasicBlock *block, HirInstr instr);

/* Get the terminator (last instruction) of the block. */
HirInstr hir_block_terminator(struct HirBasicBlock *block);

/* Append an instruction to the end of the block. Returns the instr. */
HirInstr hir_block_append(struct HirBasicBlock *block, HirInstr instr);

/* Remove and return the first instruction. */
HirInstr hir_block_pop_front(struct HirBasicBlock *block);

/* Number of incoming edges. */
size_t hir_block_in_edges_count(struct HirBasicBlock *block);

/* Update Phi instructions: replace old_pred with new_pred. */
void hir_block_fixup_phis(struct HirBasicBlock *block,
                          struct HirBasicBlock *old_pred,
                          struct HirBasicBlock *new_pred);

/* ---- Instruction predicates ----
 * Most predicates moved to hir_instr_c.h as hir_c_is_* inline functions
 * (direct opcode field read, no C++ bridge). Only complex predicates
 * that need C++ internals remain here. */

/* ---- Instruction accessors ---- */
/* hir_instr_output moved to hir_instr_c.h as hir_c_output */

/* Control flow edges. */
size_t hir_instr_num_edges(HirInstr instr);
struct HirBasicBlock *hir_instr_successor(HirInstr instr, size_t index);

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

/* Visit DeoptBase/Snapshot extension uses (frame_state, live_regs, guilty_reg).
 * Called by hir_c_visit_uses after operand array iteration.
 * For Snapshot: visits frame_state only.
 * For DeoptBase: visits frame_state + live_regs + guilty_reg.
 * Uses void** to match hir_instr_c.h's hir_c_visit_uses signature. */
int hir_c_visit_deopt_extension(void *instr,
                                int (*callback)(void **reg_slot, void *ctx),
                                void *ctx);

/* ---- OperandType ---- */

/* Get the operand type for operand i of the instruction.
 * Dispatches via T2-C3 function pointer table (per-opcode). */
HirOperandType hir_c_get_operand_type(HirInstr instr, size_t i);

/* ---- Branch-specific ---- */

/* Get the target block of a Branch instruction. */
struct HirBasicBlock *hir_branch_target(HirInstr branch);

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

/* Allocate a new unlinked BasicBlock (not added to CFG block list) */
void *hir_cfg_allocate_unlinked_block(void *cfg);

/* Update Phi instructions in block to replace old_pred with new_pred */
void hir_bb_fixup_phis(void *block, void *old_pred, void *new_pred);

/* W25 Step 1.5 (chat 13:38:34Z): redundant `hir_instr_unlink` decl removed
 * — same signature already declared above at line ~134 ("Instruction
 * mutation" section). Kept the canonical decl in the mutation section. */

/* Destroy (free) an unlinked instruction. */
void hir_instr_destroy(HirInstr instr);

/* Set the true_bb on a CondBranch instruction (for emitCondSlowPath). */
void hir_c_set_true_bb(HirInstr branch, void *new_true_block);

/* Create a Phi with caller-provided output register (for emitCondSlowPath). */
void *hir_phi_create_2way_with_output(void *output_reg,
    void *bb1, void *reg1, void *bb2, void *reg2);

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

/* CRTP Phase 2: FrameState-accepting factory variants */
HirInstr hir_c_create_vectorcall_fs_reg(size_t n_operands, HirRegister dst,
                                         uint32_t flags, void *frame_state);
HirInstr hir_c_create_guard_type_fs_reg(HirRegister dst, HirType type,
                                         HirRegister src, void *frame_state);

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
HirInstr hir_c_create_xdecref_reg(HirRegister src);
HirInstr hir_c_create_incref_reg(HirRegister src);
HirInstr hir_c_create_assign_reg(HirRegister dst, HirRegister src);
HirInstr hir_c_create_primitive_box_bool_reg(HirRegister dst, HirRegister src);
HirInstr hir_c_create_check_sequence_bounds_reg(HirRegister dst,
    HirRegister seq, HirRegister idx, void *frame_state);
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
HirInstr hir_c_create_load_method_cached_reg(HirRegister dst, HirRegister receiver, int32_t name_idx, void *fs);
HirInstr hir_c_create_load_attr_cached_reg(HirRegister dst, HirRegister receiver, int32_t name_idx, void *fs);
HirInstr hir_c_create_store_attr_cached_reg(HirRegister receiver, HirRegister value, int32_t name_idx, void *fs);
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
/* Batch 6 */
HirInstr hir_c_create_store_field_reg(HirRegister receiver, const char *name, intptr_t offset, HirRegister value, HirType type, HirRegister previous);
HirInstr hir_c_create_yield_and_yield_from_reg(HirRegister dst, HirRegister waiter, HirRegister coro, void *fs);
HirInstr hir_c_create_yield_from_handle_stop_async_reg(HirRegister dst, HirRegister send, HirRegister awaitable, void *fs);
HirInstr hir_c_create_call_ex_reg(HirRegister dst, HirRegister func, HirRegister pargs, HirRegister kwargs, uint32_t flags, void *fs);
HirInstr hir_c_create_import_name_reg(HirRegister dst, int32_t name_idx, HirRegister fromlist, HirRegister level, void *fs);
/* Batch 7: variadic call types */
HirInstr hir_c_create_call_method_reg(size_t n_operands, HirRegister dst, uint32_t flags);
HirInstr hir_c_create_call_static_ret_void_reg(size_t n_operands, void *addr);
HirInstr hir_c_create_invoke_static_function_reg(size_t n_operands, HirRegister dst, void *func, HirType ret_type);
/* Batch 8: remaining complex types */
HirInstr hir_c_create_load_global_cached_reg(HirRegister dst, void *code, void *builtins, void *globals, int32_t name_idx);
HirInstr hir_c_create_load_function_indirect_reg(void *indirect_ptr, void *descr, HirRegister dst, void *fs);
HirInstr hir_c_create_store_array_item_reg(HirRegister arr, HirRegister idx, HirRegister value, HirRegister container, HirType elem_type);
HirInstr hir_c_create_cast_reg(HirRegister dst, HirRegister receiver, void *pytype, int optional, int exact, void *fs);
HirInstr hir_c_create_raise_static_reg(int32_t reraise, void *exc_type, const char *fmt, void *fs);
HirInstr hir_c_create_call_intrinsic_reg2(size_t n_operands, HirRegister dst, int32_t index, HirRegister *operands);
HirInstr hir_c_create_load_attr_special_reg(HirRegister dst, HirRegister receiver, void *id, const char *fmt, void *fs);
HirInstr hir_c_create_call_cfunc_reg(size_t n_operands, HirRegister dst, int32_t func_enum, HirRegister *operands);
HirInstr hir_c_create_call_ind_reg2(size_t n_operands, HirRegister dst, const char *name, HirType ret_type);
HirInstr hir_c_create_load_method_super_reg(HirRegister dst, HirRegister global_super, HirRegister type, HirRegister receiver, int32_t name_idx, int no_args, void *fs);
HirInstr hir_c_create_load_attr_super_reg(HirRegister dst, HirRegister global_super, HirRegister type, HirRegister receiver, int32_t name_idx, int no_args, void *fs);
HirInstr hir_c_create_match_class_reg2(HirRegister dst, HirRegister subject, HirRegister type, HirRegister nargs, HirRegister names);

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
HirInstr hir_c_create_unicode_concat_reg(HirRegister dst, HirRegister lhs,
                                          HirRegister rhs, void *frame_state);
HirInstr hir_c_create_unicode_repeat_reg(HirRegister dst, HirRegister lhs,
                                          HirRegister rhs, void *frame_state);
HirInstr hir_c_create_unicode_subscr_reg(HirRegister dst, HirRegister lhs,
                                          HirRegister rhs, void *frame_state);

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
HirInstr hir_block_back(struct HirBasicBlock *block);

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

/* Get the FrameState from a Snapshot instruction. Returns NULL if none. */
void *hir_snapshot_get_frame_state(void *snapshot);

/* Copy a FrameState onto a DeoptBase instruction. */
void hir_deopt_set_frame_state(void *deopt_instr, const void *frame_state);

/* ---- Analysis utilities (T2-D Tier 1) ---- */

/* Check if instruction is a passthrough (Assign, BitCast, etc.). */
int hir_is_passthrough(HirInstr instr);

/* Check if an operand constraint requires exact type match (primitives). */
int hir_operands_must_match(HirInstr instr, size_t operand_idx);

/* Check if a given type satisfies the expected operand constraint.
 * Takes an explicit type
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

/* ---- outputType ---- */

/* Compute the output type of an instruction based on its opcode and operands. */
HirType hir_output_type(HirInstr instr);

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
int hir_remove_trampoline_blocks(struct HirCFG *cfg);

/* W25 Step B-77a hir_cfg_split_critical_edges_c promotion (Class C1 → C2):
 * impl in pass_output_type_c.c, callers in cfg.cpp + refcount_insertion.cpp. */
void hir_cfg_split_critical_edges_c(void *func);

/* Remove unreachable blocks from function. Returns 1 if changed. */
int hir_remove_unreachable_blocks(HirFunction func);

/* Remove unreachable instructions. */
void hir_remove_unreachable_instructions(HirFunction func);

/* Re-derive all register types from instructions. */
void hir_reflow_types(HirFunction func);

/* ---- W25 Step B-3.5 promotions (Class C1 → canonical) ----
 * These functions exist in hir_c_api.cpp but were never exposed via
 * canonical headers — §1b TUs declared them via local extern, contributing
 * to the signature-drift surface W25 closes. Promoted here to stop the
 * §1b drift class. Signatures preserved (void* for HBB/CFG to keep
 * call-site source compat — struct-typification is a future cleanup. */
size_t hir_cfg_get_rpo_from(HirFunction func, void *start, void **out, size_t capacity);
void *hir_cfg_blocks_first_ptr(void *cfg);
void *hir_cfg_blocks_next_ptr(void *cfg, void *block);
int hir_instr_is_deopt_base(void *instr);
void hir_instr_replace_uses_of(void *instr, void *old_reg, void *new_reg);
void *hir_c_cond_branch_true_bb(void *instr);
void *hir_c_cond_branch_false_bb(void *instr);

/* ==== H2-C: Instr lifecycle and list manipulation (extern C wrappers) ==== */

/* C++ destruction helpers for hir_c_destroy_instr_impl(). */
void hir_c_destroy_frame_state(void *frame_state);
void hir_c_destroy_edge(void *edge_ptr);
void hir_c_destroy_profiled_types(void *types_ptr);

/* Destroy an instruction and free its slab-allocated memory.
 * Pure C dispatch — handles per-type cleanup explicitly (H2-C). */
void hir_c_destroy_instr(HirInstr instr);

/* Insert 'new_instr' before 'before' in the instruction list. */
void hir_c_insert_before(HirInstr new_instr, HirInstr before);

/* Insert 'new_instr' after 'after' in the instruction list. */
void hir_c_insert_after(HirInstr new_instr, HirInstr after);

/* W25 Step 1.5 (chat 13:38:34Z): the 1-arg `hir_c_unlink(HirInstr)` decl
 * that lived here was DEAD CODE — zero callers in tree per pre-work grep
 * (hir.cpp:642 calls the 2-arg static inline `hir_c_unlink(void*, const
 * HirBasicBlock*)` from hir_basic_block_c.h instead). Removed to resolve
 * the same-name-different-signature conflict that blocked the prelude
 * Layer B attempt. The 2-arg static inline in hir_basic_block_c.h is now
 * the only `hir_c_unlink`. */

/* Replace 'old_instr' with 'new_instr' in the instruction list.
 * old_instr is unlinked but NOT destroyed. */
void hir_c_replace_with(HirInstr old_instr, HirInstr new_instr);

/* Set the i-th successor block of a terminator instruction. */
void hir_c_set_successor_cpp(HirInstr instr, size_t i, void *block);

/* Replace all uses of 'orig' register with 'replacement' in this instruction. */
void hir_c_replace_uses_of(HirInstr instr, HirRegister orig, HirRegister replacement);

/* Copy frame state from src instruction (DeoptBase) onto dst instruction (DeoptBase).
 * Both must be DeoptBase subclasses. Deep copies the FrameState. */
void hir_c_copy_frame_state(HirInstr dst, HirInstr src);

/* ---- Block manipulation bridges ---- */

/* Allocate a new basic block in the function's CFG.
 * Lifetime: compilation-scoped (valid until function is destroyed). */
void *hir_cfg_alloc_block(HirFunction func);

/* Split the current block after the given instruction, creating a new
 * tail block. Returns the tail block.
 * Lifetime: compilation-scoped. */
void *hir_cfg_split_after(HirFunction func, void *instr);

/* Create a 2-way Phi instruction merging results from two predecessor blocks.
 * Lifetime: compilation-scoped. */
void *hir_phi_create_2way(HirFunction func, void *bb1, void *reg1,
                           void *bb2, void *reg2);

/* ---- Runtime function address bridges ---- */
/* Returns function pointer for JITRT_InvokeIterNext (C++ linkage).
 * Lifetime: static (function pointer, never changes). */
void *jit_rt_invoke_iter_next_addr(void);

/* Returns function pointer for JITRT_LoadModuleDictEntry. Used by the
 * LOAD_ATTR_MODULE specialization to invoke a CallStatic against the
 * 2-arg runtime helper (PyDictKeysObject* keys, Py_ssize_t index)
 * → PyObject*. Lifetime: static. */
void *jit_rt_load_module_dict_entry_addr(void);

/* ---- Context/Preload bridges ---- */

/* Check if a PyMethodDef is a known builtin. Returns the builtin name
 * or NULL if not found.
 * Lifetime: thread-local static (valid until next call from same thread). */
const char *jit_builtins_find(void *method_def);

/* Note: hir_c_num_edges, hir_c_successor, hir_c_is_terminator,
 * hir_c_set_bytecode_offset, hir_c_copy_bytecode_offset are already
 * defined as static inline in hir_instr_c.h — no extern C needed. */

/* ---- Instr mutation API (additions — insert_before/unlink/replace_with already above) ---- */

/* Insert instr after 'after' in its block's instruction list. */
void hir_instr_insert_after(void *instr, void *after);

/* Insert N instructions after original, copy bytecodeOffset, unlink original. */
void hir_instr_expand_into(void *original, void **expansion, size_t count);

/* ---- Builder bridges (Tier 2 emit conversion) ---- */

/* Read _PyAttrCache inline cache from bytecode at instr_idx.
 * Returns version (uint32) and index (uint16). */
void hir_builder_get_attr_cache(void *builder, int instr_idx,
                                 uint32_t *version_out, uint16_t *index_out);

/* Find a PyTypeObject by its tp_version_tag. Returns NULL if not found. */
PyTypeObject *hir_find_type_by_version_tag(uint32_t version);

/* ---- Preloader bridges (Tier 4 emit conversion) ---- */

/* Get the Preloader's annotations (HirAnnotationIndex*). The returned
 * pointer is OWNED by the Preloader (its lifetime ends with the
 * Preloader's destructor at preload.h:92). The caller MUST NOT call
 * hir_annotation_index_destroy on it. May return NULL if the function
 * has no annotations — caller must guard. */
void *hir_builder_preloader_annotations(void *builder);

/* Get the Preloader's numArgs. */
int hir_builder_preloader_num_args(void *builder);

/* Get the Preloader's return type as HirType. */
HirType hir_builder_preloader_return_type(void *builder);

/* Get the Preloader's type(PyObject* descr) as HirType. */
HirType hir_builder_preloader_type(void *builder, PyObject *descr);

/* Get the Preloader's pyType(PyObject* descr) as PyTypeObject*. */
void *hir_builder_preloader_py_type(void *builder, PyObject *descr);

/* Get the Preloader's global(name_idx) — returns PyObject* or NULL. */
PyObject *hir_builder_preloader_global(void *builder, int name_idx);

/* Get the Preloader's globals() as PyDictObject*. */
void *hir_builder_preloader_globals(void *builder);

/* Get the Preloader's builtins() as PyDictObject*. */
void *hir_builder_preloader_builtins(void *builder);

/* Get the Preloader's fieldInfo (offset, type, name). */
void hir_builder_preloader_field_info(void *builder, PyObject *descr,
                                       intptr_t *offset_out, HirType *type_out,
                                       PyObject **name_out);

/* Get the Preloader's preloadedType fields (type, optional, exact). */
void *hir_builder_preloader_preloaded_type(void *builder, PyObject *descr,
                                            int *optional_out, int *exact_out);

/* ---- Phase 4.D pilot step 2 (Batch 54): TranslationContext emit primitives ----
 *
 * Per-method C primitives for the no-FrameState emit cluster. Each takes
 * `void *tc` (PhxTranslationContext layout pinned via static_assert in
 * hir_instr_c_verify.cpp) and dispatches via the B53 hir_c_tc_emit_c
 * primitive. C++ TranslationContext methods become 1-line shims.
 *
 * These live in hir_c_api.h (rather than hir_instr_c.h per theologian's
 * spec) because each body calls extern hir_c_create_X factories declared
 * in this header — putting the primitives in hir_instr_c.h would require
 * either circular include or duplicating the W25b opaque-handle
 * typedefs. hir_c_api.h is included by all consumers (builder.cpp +
 * builder_emit_c.c) so the architectural intent is preserved. */

static inline void hir_c_tc_emit_snapshot(void *tc) {
    /* PhxTranslationContext.frame at offset sizeof(void *) — see
     * static_assert in hir_instr_c_verify.cpp (Batch 53). */
    void *frame_ptr = (void *)((char *)tc + sizeof(void *));
    hir_c_tc_emit_c(tc, hir_c_create_snapshot(frame_ptr));
}

static inline void hir_c_tc_emit_load_const(void *tc, HirRegister dst,
                                              HirType type) {
    hir_c_tc_emit_c(tc, hir_c_create_load_const(dst, type));
}

static inline void hir_c_tc_emit_guard_type(void *tc, HirRegister dst,
                                              HirType target,
                                              HirRegister src) {
    hir_c_tc_emit_c(tc, hir_c_create_guard_type_reg(dst, target, src));
}

static inline void hir_c_tc_emit_refine_type(void *tc, HirRegister dst,
                                               HirType type, HirRegister src) {
    hir_c_tc_emit_c(tc, hir_c_create_refine_type_reg(dst, type, src));
}

static inline void hir_c_tc_emit_check_exc(void *tc, HirRegister dst,
                                             HirRegister src) {
    hir_c_tc_emit_c(tc, hir_c_create_check_exc_reg(dst, src));
}

static inline HirInstr hir_c_tc_emit_branch(void *tc, void *target_block) {
    HirInstr i = hir_c_create_branch_cpp(target_block);
    hir_c_tc_emit_c(tc, i);
    return i;
}

static inline HirInstr hir_c_tc_emit_cond_branch(void *tc, HirRegister cond,
                                                   void *true_bb,
                                                   void *false_bb) {
    HirInstr i = hir_c_create_cond_branch_cpp(cond, true_bb, false_bb);
    hir_c_tc_emit_c(tc, i);
    return i;
}

static inline HirInstr hir_c_tc_emit_deopt(void *tc) {
    HirInstr i = hir_c_create_deopt();
    hir_c_tc_emit_c(tc, i);
    return i;
}

static inline void hir_c_tc_emit_return(void *tc, HirRegister src,
                                          HirType type) {
    hir_c_tc_emit_c(tc, hir_c_create_return(src, type));
}

static inline void hir_c_tc_emit_assign(void *tc, HirRegister dst,
                                          HirRegister src) {
    hir_c_tc_emit_c(tc, hir_assign_create(dst, src));
}

/* ---- Phase 4.D pilot step 3 (Batch 55): FrameState-coupled emit cluster ----
 *
 * Two patterns:
 *   (1) setFrameState-after: factory takes no FrameState, instr is emitted,
 *       then setFrameState applies via hir_deopt_set_frame_state. Used by
 *       emitGuardType / emitCheckExc when FrameState is supplied.
 *   (2) fs-as-factory-arg: factory takes void *frame_state; emit follows.
 *       Used by emitYieldValue, emitMakeCell, emitInitialYield, etc.
 *
 * Order preserves C++ semantics: emit before setFrameState (pattern 1) so
 * bytecode_offset is stamped via hir_c_tc_emit_c first, then FrameState
 * metadata attaches. */

/* Pattern 1: emit + setFrameState-after */
static inline void hir_c_tc_emit_guard_type_fs(void *tc, HirRegister dst,
                                                 HirType target,
                                                 HirRegister src,
                                                 const void *fs) {
    HirInstr instr = hir_c_create_guard_type_reg(dst, target, src);
    hir_c_tc_emit_c(tc, instr);
    hir_deopt_set_frame_state(instr, fs);
}

static inline void hir_c_tc_emit_check_exc_fs(void *tc, HirRegister dst,
                                                HirRegister src,
                                                const void *fs) {
    HirInstr instr = hir_c_create_check_exc_reg(dst, src);
    hir_c_tc_emit_c(tc, instr);
    hir_deopt_set_frame_state(instr, fs);
}

/* Pattern 2: fs-as-factory-arg */
static inline void hir_c_tc_emit_yield_value(void *tc, HirRegister dst,
                                               HirRegister src, void *fs) {
    hir_c_tc_emit_c(tc, hir_c_create_yield_value_reg(dst, src, fs));
}

static inline void hir_c_tc_emit_make_cell(void *tc, HirRegister dst,
                                             HirRegister src, void *fs) {
    hir_c_tc_emit_c(tc, hir_c_create_make_cell_reg(dst, src, fs));
}

static inline void hir_c_tc_emit_initial_yield(void *tc, HirRegister dst,
                                                 void *fs) {
    hir_c_tc_emit_c(tc, hir_c_create_initial_yield_reg(dst, fs));
}

static inline void hir_c_tc_emit_yield_from(void *tc, HirRegister dst,
                                              HirRegister send_value,
                                              HirRegister iter, void *fs) {
    hir_c_tc_emit_c(tc, hir_c_create_yield_from_reg(dst, send_value, iter, fs));
}

static inline void hir_c_tc_emit_check_var(void *tc, HirRegister dst,
                                             HirRegister src, void *name,
                                             void *fs) {
    hir_c_tc_emit_c(tc, hir_c_create_check_var_reg(dst, src, name, fs));
}

static inline void hir_c_tc_emit_get_iter(void *tc, HirRegister dst,
                                            HirRegister src, void *fs) {
    hir_c_tc_emit_c(tc, hir_c_create_get_iter_reg(dst, src, fs));
}

/* ---- Phase 4.D pilot step 5 (Batch 57): emit cluster 3 (10 methods) ---- */

static inline void hir_c_tc_emit_binary_op(void *tc, HirRegister dst,
                                             int32_t op_kind, HirRegister left,
                                             HirRegister right, void *fs) {
    hir_c_tc_emit_c(tc,
        hir_c_create_binary_op_reg(dst, op_kind, left, right, fs));
}

static inline void hir_c_tc_emit_set_dict_item(void *tc, HirRegister dst,
                                                 HirRegister dict, HirRegister key,
                                                 HirRegister value, void *fs) {
    hir_c_tc_emit_c(tc,
        hir_c_create_set_dict_item_reg(dst, dict, key, value, fs));
}

static inline void hir_c_tc_emit_load_tuple_item(void *tc, HirRegister dst,
                                                   HirRegister tuple,
                                                   int32_t idx) {
    hir_c_tc_emit_c(tc, hir_c_create_load_tuple_item_reg(dst, tuple, idx));
}

static inline void hir_c_tc_emit_load_field_address(void *tc, HirRegister dst,
                                                      HirRegister object,
                                                      HirRegister offset) {
    hir_c_tc_emit_c(tc,
        hir_c_create_load_field_address_reg(dst, object, offset));
}

static inline void hir_c_tc_emit_set_current_awaiter(void *tc, HirRegister src) {
    hir_c_tc_emit_c(tc, hir_c_create_set_current_awaiter_reg(src));
}

/* emitDecref/emitXDecref: caller chooses based on register nullability
 * (TObject vs TOptObject); two primitives so the type-branch stays in
 * the C++ shim without extending the C-side surface with type queries. */
static inline void hir_c_tc_emit_decref(void *tc, HirRegister src) {
    hir_c_tc_emit_c(tc, hir_c_create_decref_reg(src));
}

static inline void hir_c_tc_emit_xdecref(void *tc, HirRegister src) {
    hir_c_tc_emit_c(tc, hir_c_create_xdecref_reg(src));
}

static inline void hir_c_tc_emit_load_arg(void *tc, HirRegister dst,
                                            int32_t idx, HirType type) {
    hir_c_tc_emit_c(tc, hir_c_create_load_arg_reg(dst, idx, type));
}

static inline void hir_c_tc_emit_cond_branch_iter_not_done(void *tc,
                                                             HirRegister src,
                                                             void *body_block,
                                                             void *done_block) {
    hir_c_tc_emit_c(tc,
        hir_c_create_cond_branch_iter_not_done_cpp(src, body_block, done_block));
}

static inline void hir_c_tc_emit_int_convert(void *tc, HirRegister dst,
                                               HirRegister src, HirType type) {
    hir_c_tc_emit_c(tc, hir_c_create_int_convert_reg(dst, src, type));
}

static inline void hir_c_tc_emit_wait_handle_release(void *tc, HirRegister src) {
    hir_c_tc_emit_c(tc, hir_c_create_wait_handle_release_reg(src));
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* JIT_HIR_C_API_H */
