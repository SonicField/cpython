/*
 * lir_c_api.cpp -- C accessor implementations for LIR types
 *
 * Phase 3D: Thin C++ wrappers that delegate to LIR class methods.
 * Each function casts the opaque void* handle to the correct C++ type.
 */

#include "cinderx/Jit/lir/lir_c_api.h"

#include <iterator>

#include "cinderx/Jit/codegen/code_section.h"
#include "cinderx/Jit/lir/block.h"
#include "cinderx/Jit/lir/function.h"
#include "cinderx/Jit/lir/instruction.h"
#include "cinderx/Jit/lir/operand.h"

using jit::codegen::CodeSection;
using jit::lir::BasicBlock;
using jit::lir::Function;
using jit::lir::Instruction;
using jit::lir::OperandBase;

/* ---- Function accessors ---- */

extern "C" size_t
jit_lir_func_num_blocks(JitLirFunc func) {
  return static_cast<Function*>(func)->basicblocks().size();
}

extern "C" JitLirBlock
jit_lir_func_get_block(JitLirFunc func, size_t index) {
  return static_cast<Function*>(func)->basicblocks()[index];
}

extern "C" JitLirBlock
jit_lir_func_entry_block(JitLirFunc func) {
  return static_cast<Function*>(func)->entryBlock();
}

/* ---- BasicBlock accessors ---- */

extern "C" size_t
jit_lir_block_num_preds(JitLirBlock block) {
  return static_cast<BasicBlock*>(block)->predecessors().size();
}

extern "C" JitLirBlock
jit_lir_block_get_pred(JitLirBlock block, size_t index) {
  return static_cast<BasicBlock*>(block)->predecessors()[index];
}

extern "C" size_t
jit_lir_block_num_succs(JitLirBlock block) {
  return static_cast<BasicBlock*>(block)->successors().size();
}

extern "C" JitLirBlock
jit_lir_block_get_succ(JitLirBlock block, size_t index) {
  return static_cast<BasicBlock*>(block)->successors()[index];
}

extern "C" JitLirInstr
jit_lir_block_get_last_instr(JitLirBlock block) {
  return const_cast<Instruction*>(
      static_cast<BasicBlock*>(block)->getLastInstr());
}

extern "C" JitLirInstr
jit_lir_block_get_first_instr(JitLirBlock block) {
  return const_cast<Instruction*>(
      static_cast<BasicBlock*>(block)->getFirstInstr());
}

extern "C" size_t
jit_lir_block_num_instrs(JitLirBlock block) {
  return static_cast<BasicBlock*>(block)->getNumInstrs();
}

extern "C" JitLirBlock
jit_lir_block_get_false_succ(JitLirBlock block) {
  return static_cast<BasicBlock*>(block)->getFalseSuccessor();
}

extern "C" int
jit_lir_block_get_section(JitLirBlock block) {
  return static_cast<int>(static_cast<BasicBlock*>(block)->section());
}

extern "C" void
jit_lir_block_set_section(JitLirBlock block, int section) {
  static_cast<BasicBlock*>(block)->setSection(
      static_cast<CodeSection>(section));
}

extern "C" int
jit_lir_block_get_id(JitLirBlock block) {
  return static_cast<BasicBlock*>(block)->id();
}

extern "C" JitLirInstr
jit_lir_block_get_instr_at(JitLirBlock block, size_t index) {
  auto& instrs = static_cast<BasicBlock*>(block)->instructions();
  auto it = instrs.begin();
  std::advance(it, index);
  return it->get();
}

/* ---- Instruction accessors ---- */

extern "C" int
jit_lir_instr_opcode(JitLirInstr instr) {
  return static_cast<int>(static_cast<Instruction*>(instr)->opcode());
}

extern "C" int
jit_lir_instr_is_branch(JitLirInstr instr) {
  return static_cast<Instruction*>(instr)->isBranch();
}

extern "C" int
jit_lir_instr_is_branch_cc(JitLirInstr instr) {
  return static_cast<Instruction*>(instr)->isBranchCC();
}

extern "C" int
jit_lir_instr_is_any_branch(JitLirInstr instr) {
  return static_cast<Instruction*>(instr)->isAnyBranch();
}

extern "C" int
jit_lir_instr_is_terminator(JitLirInstr instr) {
  return static_cast<Instruction*>(instr)->isTerminator();
}

extern "C" size_t
jit_lir_instr_num_inputs(JitLirInstr instr) {
  return static_cast<Instruction*>(instr)->getNumInputs();
}

extern "C" JitLirOperand
jit_lir_instr_get_input(JitLirInstr instr, size_t index) {
  return static_cast<Instruction*>(instr)->getInput(index);
}

extern "C" JitLirOperand
jit_lir_instr_output(JitLirInstr instr) {
  return static_cast<Instruction*>(instr)->output();
}

/* ---- Operand accessors ---- */

extern "C" int
jit_lir_operand_type(JitLirOperand op) {
  return static_cast<int>(static_cast<OperandBase*>(op)->type());
}

extern "C" JitLirBlock
jit_lir_operand_get_basic_block(JitLirOperand op) {
  return static_cast<OperandBase*>(op)->getBasicBlock();
}

/* ---- Opcode constants ---- */

extern "C" int
jit_lir_opcode_guard(void) {
  return static_cast<int>(Instruction::kGuard);
}
