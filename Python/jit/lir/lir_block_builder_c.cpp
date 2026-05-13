/*
 * lir_block_builder_c.cpp -- extern "C" wrapper bodies for
 * jit::lir::BasicBlockBuilder, FIRST batch of c17 BBB-bridge construction
 * per supervisor 02:18:39Z auth.
 *
 * All wrappers cast LirBasicBlockBuilder* to C++ BasicBlockBuilder*
 * (sizeof + offsetof + blob-size cross-validated by
 * lir_block_builder_c_verify.cpp via 9 static_asserts) and call C++
 * methods. Blob interior is NOT accessed from C side per
 * docs/methodology/pre-port-audit-checklist.md addendum 072c3f08f3
 * pass-through-only constraint.
 */

#include "cinderx/Jit/lir/lir_block_builder_c.h"
#include "cinderx/Jit/lir/block_builder.h"
#include "cinderx/Jit/hir/hir.h"

using jit::lir::BasicBlock;
using jit::lir::BasicBlockBuilder;

extern "C" void
lir_bbb_set_current_instr(LirBasicBlockBuilder *bbb, const void *hir_instr) {
    reinterpret_cast<BasicBlockBuilder *>(bbb)->setCurrentInstr(
        reinterpret_cast<const jit::hir::Instr *>(hir_instr));
}

extern "C" JitLirBlock
lir_bbb_allocate_block(LirBasicBlockBuilder *bbb) {
    return reinterpret_cast<BasicBlockBuilder *>(bbb)->allocateBlock();
}

extern "C" void
lir_bbb_switch_block(LirBasicBlockBuilder *bbb, JitLirBlock block) {
    reinterpret_cast<BasicBlockBuilder *>(bbb)->switchBlock(
        reinterpret_cast<BasicBlock *>(block));
}

/* Phase 5.B c18: BBB wrappers batch 2. */

extern "C" void
lir_bbb_append_block(LirBasicBlockBuilder *bbb, JitLirBlock block) {
    reinterpret_cast<BasicBlockBuilder *>(bbb)->appendBlock(
        reinterpret_cast<BasicBlock *>(block));
}

extern "C" JitLirInstr
lir_bbb_get_def_instr(LirBasicBlockBuilder *bbb, const void *hir_reg) {
    return reinterpret_cast<BasicBlockBuilder *>(bbb)->getDefInstr(
        reinterpret_cast<const jit::hir::Register *>(hir_reg));
}

extern "C" void
lir_bbb_create_instr_input(LirBasicBlockBuilder *bbb,
                            JitLirInstr instr, void *hir_reg) {
    reinterpret_cast<BasicBlockBuilder *>(bbb)->createInstrInput(
        reinterpret_cast<jit::lir::Instruction *>(instr),
        reinterpret_cast<jit::hir::Register *>(hir_reg));
}

/* Phase 5.B c20: BBB wrappers batch 3. */

extern "C" size_t
lir_bbb_make_deopt_metadata(LirBasicBlockBuilder *bbb) {
    return reinterpret_cast<BasicBlockBuilder *>(bbb)->makeDeoptMetadata();
}
