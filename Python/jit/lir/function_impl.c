/*
 * function_impl.c -- C implementation of LIR Function operations
 *
 * Phase B1: Pure C functions operating on LirFunction struct.
 * Coexists with function.cpp — copyFrom (deep copy for inlining)
 * remains in C++ until 5.A3 commit 5 cuts over.
 *
 * Phase 5.A3 commit 3 adds copy_indirect + copy_operand C ports
 * (parallel; C++ Function::copyFrom still runs in production until
 * commit 5). Substrate provided by hir/phx_int_ptr_map.h (commit 1)
 * and hir/phx_ptr_int_map.h (commit 2).
 */

#include "cinderx/Jit/lir/lir_impl_internal.h"

#include "cinderx/Common/jit_log_c.h"
#include "cinderx/Jit/hir/phx_int_ptr_map.h"
#include "cinderx/Jit/hir/phx_ptr_int_map.h"

#include "Python.h"

#include <assert.h>

#define INITIAL_BLOCK_CAPACITY 16

/* ---- Lifecycle ---- */

LirFunction *
lir_function_new(const void *hir_func) {
    LirFunction *func = (LirFunction *)PyMem_RawCalloc(
        1, sizeof(LirFunction));
    func->hir_func_ = hir_func;
    func->blocks_capacity_ = INITIAL_BLOCK_CAPACITY;
    func->blocks_ = (LirBasicBlock **)PyMem_RawCalloc(
        func->blocks_capacity_, sizeof(LirBasicBlock *));
    func->num_blocks_ = 0;
    func->next_id_ = 0;
    return func;
}

void
lir_function_destroy(LirFunction *func) {
    if (func == NULL) return;
    /* Free all owned blocks (which recursively free instructions) */
    for (size_t i = 0; i < func->num_blocks_; i++) {
        lir_block_free(func->blocks_[i]);
    }
    PyMem_RawFree(func->blocks_);
    /* Does NOT free func itself — caller handles that */
}

void
lir_function_free(LirFunction *func) {
    if (func == NULL) return;
    lir_function_destroy(func);
    PyMem_RawFree(func);
}

/* ---- ID allocation ---- */

int
lir_function_allocate_id(LirFunction *func) {
    return func->next_id_++;
}

void
lir_function_set_next_id(LirFunction *func, int id) {
    func->next_id_ = id;
}

/* ---- Block management ---- */

void
lir_function_ensure_block_capacity(LirFunction *func, size_t needed) {
    if (needed <= func->blocks_capacity_) return;
    size_t new_cap = func->blocks_capacity_ == 0 ? 8 : func->blocks_capacity_ * 2;
    while (new_cap < needed) new_cap *= 2;
    if (func->blocks_ == NULL) {
        func->blocks_ = (LirBasicBlock **)PyMem_RawMalloc(
            new_cap * sizeof(LirBasicBlock *));
    } else {
        func->blocks_ = (LirBasicBlock **)PyMem_RawRealloc(
            func->blocks_, new_cap * sizeof(LirBasicBlock *));
    }
    func->blocks_capacity_ = new_cap;
}

static void
ensure_block_capacity(LirFunction *func) {
    lir_function_ensure_block_capacity(func, func->num_blocks_ + 1);
}

LirBasicBlock *
lir_function_alloc_block(LirFunction *func) {
    int id = lir_function_allocate_id(func);
    LirBasicBlock *bb = lir_block_new(func, id);
    ensure_block_capacity(func);
    func->blocks_[func->num_blocks_++] = bb;
    return bb;
}

LirBasicBlock *
lir_function_alloc_block_after(LirFunction *func, LirBasicBlock *after) {
    int id = lir_function_allocate_id(func);
    LirBasicBlock *bb = lir_block_new(func, id);
    ensure_block_capacity(func);

    /* Find the position of 'after' and insert the new block after it */
    size_t pos = func->num_blocks_; /* default: append */
    for (size_t i = 0; i < func->num_blocks_; i++) {
        if (func->blocks_[i] == after) {
            pos = i + 1;
            break;
        }
    }

    /* Shift blocks right to make room */
    for (size_t i = func->num_blocks_; i > pos; i--) {
        func->blocks_[i] = func->blocks_[i - 1];
    }
    func->blocks_[pos] = bb;
    func->num_blocks_++;
    return bb;
}

/* ---- Accessors ---- */

size_t
lir_function_num_blocks(const LirFunction *func) {
    return func->num_blocks_;
}

LirBasicBlock *
lir_function_get_block(const LirFunction *func, size_t index) {
    assert(index < func->num_blocks_);
    return func->blocks_[index];
}

LirBasicBlock *
lir_function_entry_block(const LirFunction *func) {
    if (func->num_blocks_ == 0) return NULL;
    return func->blocks_[0];
}

const void *
lir_function_hir_func(const LirFunction *func) {
    return func->hir_func_;
}

/* ---- Block sorting ---- */

/* ---- Phase 5.A3 commit 3: deep-copy helpers (parallel C ports) ----
 *
 * Mirror function.cpp anonymous-namespace helpers copyIndirect +
 * copyOperand. These are wired by commit 5 (lir_function_copy_from_impl)
 * which replaces the C++ Function::copyFrom body. Until commit 5 lands,
 * these helpers are unreachable from production paths — the C++ code
 * still runs.
 *
 * I-1d (linked+linked memory-indirect) loud-fail rationale: testkeeper
 * 14:00:33Z enumerated all 33 production setMemoryIndirect callsites +
 * 30 OutInd construction sites in generator.cpp; ZERO produce the
 * (Instruction*, Instruction*) variant pair. The case is structurally
 * representable but data-flow unreachable. Per supervisor 14:01:23Z
 * disposition + theologian 14:01:06Z spec amendment, we elide the
 * bridge and trip JIT_CHECK_C if a future codegen change introduces
 * the pattern. */

void
lir_copy_indirect(PhxPtrIntMap *instr_refs,
                  LirOperand *dest_op,
                  LirMemoryIndirect *source_op) {
    LirOperand *base = source_op->base_reg_;
    LirOperand *index = source_op->index_reg_;
    int base_linked = (base != NULL) && base->is_linked_;
    int index_present = (index != NULL);
    int index_linked = index_present && index->is_linked_;

    if (base_linked && !index_present) {
        /* I-1a: linked base, no index. */
        lir_operand_set_memory_indirect_instr(
            dest_op, lir_operand_instr(dest_op), source_op->offset_);
    } else if (!base_linked && !index_present) {
        /* I-1b: phy base, no index. */
        lir_operand_set_memory_indirect_phy(
            dest_op, lir_operand_get_phy_register(base), source_op->offset_);
    } else if (!base_linked && index_present && !index_linked) {
        /* I-1c: phy base + phy index + multiplier.
         * NOTE: existing _phy3 bridge does not propagate offset_; per
         * spec §3.1 + testkeeper enumeration this case is data-flow
         * unreachable from copyIndirect (no production OutInd uses
         * non-trivial index_reg). Substrate gap is documented but not
         * fixed in this commit. */
        lir_operand_set_memory_indirect_phy3(
            dest_op,
            lir_operand_get_phy_register(base),
            lir_operand_get_phy_register(index),
            source_op->multiplier_);
    } else {
        /* Catch-all for the three (base_linked, index_present, index_linked)
         * combinations not enumerated by I-1a/b/c:
         *   - I-1d: linked base + linked index (no bridge exists)
         *   - linked base + phy index (no spec invariant; no bridge)
         *   - phy base + linked index (no spec invariant; no bridge)
         * All three are data-flow-unreachable per testkeeper 14:00:33Z
         * enumeration of 33 setMemoryIndirect callsites + 30 OutInd
         * construction sites in generator.cpp. A future codegen change
         * that introduces any of these patterns will trip here,
         * signalling the need to add the corresponding bridge. */
        JIT_CHECK_C(0,
                    "lir_copy_indirect: unhandled (base_linked=%d, "
                    "index_present=%d, index_linked=%d) — "
                    "data-flow-unreachable per Phase 5.A3 Q-D",
                    base_linked, index_present, index_linked);
    }

    /* Populate instr_refs for any linked side (mirrors function.cpp:42-54).
     * Reads back the just-installed memory-indirect from dest_op so
     * instr_refs keys match the freshly-allocated LinkedOperand pointers
     * inside the copy, not the source LinkedOperand pointers. */
    LirMemoryIndirect *dest_mi = lir_operand_get_indirect(dest_op);
    if (base_linked) {
        LirOperand *base_linked_src = base; /* aliases LinkedOperand* */
        int src_id = lir_operand_get_linked_instr(base_linked_src)->id_;
        phx_ptr_int_map_insert(
            instr_refs, dest_mi->base_reg_, src_id);
    }
    if (index_linked) {
        /* Unreachable today — only base may be linked per testkeeper
         * 14:00:33Z. Kept for completeness; falls under the I-1d
         * loud-fail above and will not be reached. */
        LirOperand *index_linked_src = index;
        int src_id = lir_operand_get_linked_instr(index_linked_src)->id_;
        phx_ptr_int_map_insert(
            instr_refs, dest_mi->index_reg_, src_id);
    }
}

void
lir_copy_operand(PhxIntPtrMap *block_index_map,
                 PhxPtrIntMap *instr_refs,
                 LirOperand *operand,
                 LirOperand *operand_copy) {
    switch (operand->type_) {
    case JIT_LIR_OPTYPE_REG:
        lir_operand_set_phy_register(
            operand_copy, lir_operand_get_phy_register(operand));
        lir_operand_set_data_type(operand_copy, operand->data_type_);
        break;
    case JIT_LIR_OPTYPE_STACK:
        lir_operand_set_stack_slot(
            operand_copy, lir_operand_get_stack_slot(operand));
        lir_operand_set_data_type(operand_copy, operand->data_type_);
        break;
    case JIT_LIR_OPTYPE_MEM:
        lir_operand_set_mem_address(
            operand_copy, lir_operand_get_mem_address(operand));
        break;
    case JIT_LIR_OPTYPE_IMM:
        lir_operand_set_constant(
            operand_copy, operand->value_.constant, operand->data_type_);
        break;
    case JIT_LIR_OPTYPE_LABEL: {
        LirBasicBlock *src_block = (LirBasicBlock *)operand->value_.block;
        void *mapped = phx_int_ptr_map_get_strict(
            block_index_map, src_block->id_);
        lir_operand_set_basic_block(operand_copy, mapped);
        break;
    }
    case JIT_LIR_OPTYPE_IND:
        lir_copy_indirect(
            instr_refs, operand_copy, lir_operand_get_indirect(operand));
        break;
    case JIT_LIR_OPTYPE_NONE:
    case JIT_LIR_OPTYPE_VREG:
        /* No-op (matches C++ source). */
        break;
    default:
        JIT_CHECK_C(0,
                    "lir_copy_operand: unknown operand type %u",
                    operand->type_);
    }
}

void
lir_copy_input(PhxIntPtrMap *block_index_map,
               PhxPtrIntMap *instr_refs,
               LirOperand *input,
               LirInstruction *instr_copy) {
    if (input->is_linked_) {
        /* I-8 linked path: allocate placeholder linked input + record
         * source-instruction id in instr_refs for later resolution by
         * lir_connect_linked_operands. */
        LirOperand *linked_opnd =
            lir_instruction_alloc_linked_input(instr_copy, NULL);
        int src_id = lir_operand_get_linked_instr(input)->id_;
        phx_ptr_int_map_insert(instr_refs, linked_opnd, src_id);
    } else {
        /* I-8 immediate path: allocate immediate input + structural
         * field copy via lir_copy_operand. setDataType called AFTER
         * (matches C++ source order at function.cpp:104). */
        LirOperand *input_copy =
            lir_instruction_alloc_imm_input(instr_copy, 0, JIT_LIR_DT_64BIT);
        lir_copy_operand(block_index_map, instr_refs, input, input_copy);
        lir_operand_set_data_type(input_copy, input->data_type_);
    }
}

void
lir_connect_linked_operands(PhxIntPtrMap *output_index_map,
                            PhxPtrIntMap *instr_refs) {
    /* I-10 raw-slot iteration: skip empty slots (key==NULL).
     * I-11 per-pair effect: setLinkedInstr to the destination
     * instruction looked up by source instruction id; loud-fail via
     * phx_int_ptr_map_get_strict if id is absent in output_index_map. */
    size_t cap = phx_ptr_int_map_capacity(instr_refs);
    for (size_t i = 0; i < cap; i++) {
        LirOperand *operand =
            (LirOperand *)phx_ptr_int_map_at_key(instr_refs, i);
        if (operand == NULL) continue;
        int src_instr_id = phx_ptr_int_map_at_value(instr_refs, i);
        LirInstruction *def =
            (LirInstruction *)phx_int_ptr_map_get_strict(
                output_index_map, src_instr_id);
        lir_operand_set_linked_instr(operand, def);
    }
}

void
lir_deep_copy_basic_blocks(LirBasicBlock *const *src_blocks, size_t src_count,
                           PhxIntPtrMap *block_index_map,
                           const void *origin) {
    /* I-12 stack-local maps: lifecycle bound to this function's frame;
     * both init/destroy must pair. */
    PhxIntPtrMap output_index_map;
    PhxPtrIntMap instr_refs;
    phx_int_ptr_map_init(&output_index_map);
    phx_ptr_int_map_init(&instr_refs);

    for (size_t bi = 0; bi < src_count; bi++) {
        LirBasicBlock *bb = src_blocks[bi];
        LirBasicBlock *bb_copy =
            (LirBasicBlock *)phx_int_ptr_map_get_strict(
                block_index_map, bb->id_);
        /* I-13 successor mapping. */
        for (size_t si = 0; si < bb->num_succs_; si++) {
            LirBasicBlock *succ = bb->successors_[si];
            LirBasicBlock *succ_copy =
                (LirBasicBlock *)phx_int_ptr_map_get_strict(
                    block_index_map, succ->id_);
            lir_block_add_successor(bb_copy, succ_copy);
        }
        /* I-13 per-instruction processing: iterate intrusive list,
         * allocate copy via the commit-0 bridge, append, register in
         * output_index_map keyed by source id, copy output operand,
         * copy each input operand. */
        for (LirInstruction *instr = bb->instr_head_;
             instr != NULL;
             instr = instr->next_) {
            LirInstruction *instr_copy =
                lir_instruction_new_copy(bb_copy, instr, origin);
            lir_block_append_instr(bb_copy, instr_copy);
            phx_int_ptr_map_insert(&output_index_map, instr->id_, instr_copy);
            lir_copy_operand(block_index_map, &instr_refs,
                             &instr->output_, &instr_copy->output_);
            for (size_t ii = 0; ii < instr->num_inputs_; ii++) {
                lir_copy_input(block_index_map, &instr_refs,
                               instr->inputs_[ii], instr_copy);
            }
        }
    }
    /* I-14 link resolution: AFTER all output instructions are in
     * output_index_map. */
    lir_connect_linked_operands(&output_index_map, &instr_refs);

    phx_ptr_int_map_destroy(&instr_refs);
    phx_int_ptr_map_destroy(&output_index_map);
}

int
lir_function_copy_from_impl(LirFunction *caller,
                            const LirFunction *callee,
                            LirBasicBlock *prev_bb,
                            LirBasicBlock *next_bb,
                            const void *origin,
                            int *out_begin, int *out_end) {
    /* I-15 precondition: prev_bb must have exactly one successor and
     * that successor must be next_bb. */
    JIT_CHECK_C(
        prev_bb->num_succs_ == 1 && prev_bb->successors_[0] == next_bb,
        "lir_function_copy_from_impl: prev_bb should only have 1 "
        "successor which should be next_bb (num_succs=%zu)",
        prev_bb->num_succs_);

    PhxIntPtrMap block_index_map;
    phx_int_ptr_map_init(&block_index_map);
    size_t src_count = callee->num_blocks_;

    /* I-16 per-src-bb new-bb allocation + block_index_map population +
     * blocks_ tail-shift insertion. The tail-shift preserves the
     * "last block is exit block" invariant by inserting bb_copy
     * immediately before the current last entry on each iteration. */
    for (size_t i = 0; i < callee->num_blocks_; i++) {
        LirBasicBlock *bb = callee->blocks_[i];
        int new_id = lir_function_allocate_id(caller);
        LirBasicBlock *bb_copy = lir_block_new(caller, new_id);
        phx_int_ptr_map_insert(&block_index_map, bb->id_, bb_copy);
        lir_function_ensure_block_capacity(caller, caller->num_blocks_ + 1);
        caller->blocks_[caller->num_blocks_] =
            caller->blocks_[caller->num_blocks_ - 1];
        caller->blocks_[caller->num_blocks_ - 1] = bb_copy;
        caller->num_blocks_++;
    }

    /* I-17 deep_copy AFTER all bb_copy allocations are in
     * block_index_map (so successor / label resolution can see all
     * source-id → dest-bb mappings). */
    lir_deep_copy_basic_blocks(
        callee->blocks_, src_count, &block_index_map, origin);

    /* I-18 post-copy stitching. */
    int end = (int)caller->num_blocks_ - 1;
    int start = end - (int)src_count;
    lir_block_set_successor(prev_bb, 0, caller->blocks_[start]);
    JIT_CHECK_C(
        caller->blocks_[end - 1]->num_succs_ == 0,
        "lir_function_copy_from_impl: last block of inlined function "
        "should have no successors (num_succs=%zu)",
        caller->blocks_[end - 1]->num_succs_);
    lir_block_add_successor(caller->blocks_[end - 1], next_bb);

    /* I-19 return CopyResult{start, end} via out-params. */
    *out_begin = start;
    *out_end = end;

    phx_int_ptr_map_destroy(&block_index_map);
    return 0;
}

void
lir_function_sort_blocks(LirFunction *func) {
    size_t out_count = 0;
    JitLirBlock *sorted = jit_lir_sort_blocks_rpo(
        (JitLirBlock *)func->blocks_, func->num_blocks_, &out_count);
    if (sorted != NULL) {
        /* Replace the block array with the sorted result */
        if (out_count > func->blocks_capacity_) {
            func->blocks_ = (LirBasicBlock **)PyMem_RawRealloc(
                func->blocks_, out_count * sizeof(LirBasicBlock *));
            func->blocks_capacity_ = out_count;
        }
        for (size_t i = 0; i < out_count; i++) {
            func->blocks_[i] = (LirBasicBlock *)sorted[i];
        }
        func->num_blocks_ = out_count;
        PyMem_RawFree(sorted);
    }
}
