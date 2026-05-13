/*
 * lir_block_builder_c.h -- C-callable wrappers for jit::lir::BasicBlockBuilder.
 *
 * Phase 5.B c17: FIRST wrapper batch of BBB-bridge construction
 * (foundation: c16 5338c5ef24 LirBasicBlockBuilder struct + verifier).
 *
 * Wrappers OPERATE ON OPAQUE LirBasicBlockBuilder* (cast to C++
 * BasicBlockBuilder* on the C++ side); blob interior is NOT accessed
 * by C-side per pass-through-only constraint
 * (docs/methodology/pre-port-audit-checklist.md addendum 072c3f08f3).
 */

#ifndef JIT_LIR_BLOCK_BUILDER_C_H
#define JIT_LIR_BLOCK_BUILDER_C_H

#include "cinderx/Jit/lir/lir_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* setCurrentInstr(const hir::Instr*) — set current HIR instruction
 * marker on the BBB. C++ side: bbb->setCurrentInstr(inst). */
void lir_bbb_set_current_instr(LirBasicBlockBuilder *bbb,
                                const void *hir_instr);

/* allocateBlock() -> BasicBlock* — call func_->allocateBasicBlock()
 * via the BBB. C++ side: bbb->allocateBlock(). */
JitLirBlock lir_bbb_allocate_block(LirBasicBlockBuilder *bbb);

/* switchBlock(BasicBlock*) — set cur_bb_ to the given block. C++
 * side: bbb->switchBlock(block). */
void lir_bbb_switch_block(LirBasicBlockBuilder *bbb, JitLirBlock block);

/* Phase 5.B c18: BBB wrappers batch 2. */

/* appendBlock(BasicBlock*) — add successor edge from cur_bb_ + switch
 * to block. C++ side: bbb->appendBlock(block). */
void lir_bbb_append_block(LirBasicBlockBuilder *bbb, JitLirBlock block);

/* getDefInstr(const hir::Register*) — find the LIR instruction that
 * defined a HIR register (chase env_->copy_propagation_map). C++ side:
 * bbb->getDefInstr(reg). */
JitLirInstr lir_bbb_get_def_instr(LirBasicBlockBuilder *bbb,
                                   const void *hir_reg);

/* createInstrInput(Instruction*, hir::Register*) — allocate a linked
 * input on instr pointing at the def-instruction of reg. C++ side:
 * bbb->createInstrInput(instr, reg). */
void lir_bbb_create_instr_input(LirBasicBlockBuilder *bbb,
                                 JitLirInstr instr, void *hir_reg);

#ifdef __cplusplus
}
#endif

#endif /* JIT_LIR_BLOCK_BUILDER_C_H */
