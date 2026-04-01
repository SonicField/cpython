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

/* ---- Extended instruction accessors (Phase 3D Step 10: DCE) ---- */

extern "C" int
jit_lir_instr_id(JitLirInstr instr) {
  return static_cast<Instruction*>(instr)->id();
}

extern "C" int
jit_lir_instr_is_essential(JitLirInstr instr) {
  auto* i = static_cast<Instruction*>(instr);
  return jit::lir::InstrProperty::getProperties(i).is_essential ? 1 : 0;
}

extern "C" int
jit_lir_instr_flag_effects(JitLirInstr instr) {
  auto* i = static_cast<Instruction*>(instr);
  return static_cast<int>(
      jit::lir::InstrProperty::getProperties(i).flag_effects);
}

extern "C" void
jit_lir_instr_foreach_input(
    JitLirInstr instr,
    void (*cb)(JitLirOperand operand, void* ctx),
    void* ctx) {
  auto* i = static_cast<Instruction*>(instr);
  for (size_t idx = 0; idx < i->getNumInputs(); idx++) {
    cb(i->getInput(idx), ctx);
  }
}

/* ---- Extended operand accessors ---- */

extern "C" int
jit_lir_operand_is_reg(JitLirOperand op) {
  return static_cast<OperandBase*>(op)->isReg() ? 1 : 0;
}

extern "C" int
jit_lir_operand_is_stack(JitLirOperand op) {
  return static_cast<OperandBase*>(op)->isStack() ? 1 : 0;
}

extern "C" int
jit_lir_operand_is_mem(JitLirOperand op) {
  return static_cast<OperandBase*>(op)->isMem() ? 1 : 0;
}

extern "C" int
jit_lir_operand_is_ind(JitLirOperand op) {
  return static_cast<OperandBase*>(op)->isInd() ? 1 : 0;
}

extern "C" int
jit_lir_operand_is_linked(JitLirOperand op) {
  return static_cast<OperandBase*>(op)->isLinked() ? 1 : 0;
}

extern "C" JitLirInstr
jit_lir_operand_get_linked_instr(JitLirOperand op) {
  auto* linked = static_cast<jit::lir::LinkedOperand*>(
      static_cast<OperandBase*>(op));
  return linked->getLinkedInstr();
}

extern "C" JitLirIndirect
jit_lir_operand_get_indirect(JitLirOperand op) {
  return static_cast<OperandBase*>(op)->getMemoryIndirect();
}

extern "C" JitLirOperand
jit_lir_indirect_base_reg(JitLirIndirect ind) {
  return static_cast<jit::lir::MemoryIndirect*>(ind)->getBaseRegOperand();
}

extern "C" JitLirOperand
jit_lir_indirect_index_reg(JitLirIndirect ind) {
  return static_cast<jit::lir::MemoryIndirect*>(ind)->getIndexRegOperand();
}

/* ---- Block instruction removal ---- */

extern "C" void
jit_lir_block_remove_dead_instrs(
    JitLirBlock block,
    int (*is_live)(JitLirInstr instr, void* ctx),
    void* ctx) {
  auto* bb = static_cast<BasicBlock*>(block);
  for (auto it = bb->instructions().begin();
       it != bb->instructions().end();) {
    auto to_remove = it;
    ++it;
    if (!is_live(to_remove->get(), ctx)) {
      bb->removeInstr(to_remove);
    }
  }
}

/* ---- Phase A: Extended operand getters ---- */

extern "C" int
jit_lir_operand_data_type(JitLirOperand op) {
  return static_cast<int>(static_cast<OperandBase*>(op)->dataType());
}

extern "C" int
jit_lir_operand_is_fp(JitLirOperand op) {
  return static_cast<OperandBase*>(op)->isFp() ? 1 : 0;
}

extern "C" int
jit_lir_operand_is_last_use(JitLirOperand op) {
  return static_cast<OperandBase*>(op)->isLastUse() ? 1 : 0;
}

extern "C" uint64_t
jit_lir_operand_get_constant(JitLirOperand op) {
  return static_cast<OperandBase*>(op)->getConstant();
}

extern "C" double
jit_lir_operand_get_fp_constant(JitLirOperand op) {
  return static_cast<OperandBase*>(op)->getFPConstant();
}

extern "C" int
jit_lir_operand_get_phy_register(JitLirOperand op) {
  return static_cast<int>(static_cast<OperandBase*>(op)->getPhyRegister());
}

extern "C" int
jit_lir_operand_get_stack_slot(JitLirOperand op) {
  return static_cast<int>(
      static_cast<OperandBase*>(op)->getStackSlot().loc);
}

extern "C" void*
jit_lir_operand_get_mem_address(JitLirOperand op) {
  return static_cast<OperandBase*>(op)->getMemoryAddress();
}

extern "C" JitLirOperand
jit_lir_operand_get_define(JitLirOperand op) {
  return static_cast<OperandBase*>(op)->getDefine();
}

/* ---- Phase A: Extended instruction getters ---- */

extern "C" JitLirBlock
jit_lir_instr_basic_block(JitLirInstr instr) {
  return static_cast<Instruction*>(instr)->basicblock();
}

extern "C" const void*
jit_lir_instr_origin(JitLirInstr instr) {
  return static_cast<Instruction*>(instr)->origin();
}

extern "C" int
jit_lir_instr_is_compare(JitLirInstr instr) {
  return static_cast<Instruction*>(instr)->isCompare() ? 1 : 0;
}

extern "C" int
jit_lir_instr_is_any_yield(JitLirInstr instr) {
  return static_cast<Instruction*>(instr)->isAnyYield() ? 1 : 0;
}

extern "C" int
jit_lir_instr_inputs_live_across(JitLirInstr instr) {
  auto& props = jit::lir::InstrProperty::getProperties(
      static_cast<Instruction*>(instr));
  return props.inputs_live_across ? 1 : 0;
}

extern "C" int
jit_lir_instr_output_phy_use(JitLirInstr instr) {
  return static_cast<Instruction*>(instr)->getOutputPhyRegUse() ? 1 : 0;
}

extern "C" int
jit_lir_instr_input_phy_use(JitLirInstr instr, size_t index) {
  return static_cast<Instruction*>(instr)->getInputPhyRegUse(index) ? 1 : 0;
}

/* ---- Phase A: MemoryIndirect getters ---- */

extern "C" int
jit_lir_indirect_multiplier(JitLirIndirect ind) {
  return static_cast<jit::lir::MemoryIndirect*>(ind)->getMultipiler();
}

extern "C" int32_t
jit_lir_indirect_offset(JitLirIndirect ind) {
  return static_cast<jit::lir::MemoryIndirect*>(ind)->getOffset();
}

/* ---- Phase A: Branch CC statics ---- */

extern "C" int
jit_lir_negate_branch_cc(int opcode) {
  return static_cast<int>(
      Instruction::negateBranchCC(static_cast<Instruction::Opcode>(opcode)));
}

extern "C" int
jit_lir_flip_branch_cc_direction(int opcode) {
  return static_cast<int>(Instruction::flipBranchCCDirection(
      static_cast<Instruction::Opcode>(opcode)));
}

extern "C" int
jit_lir_compare_to_branch_cc(int opcode) {
  return static_cast<int>(Instruction::compareToBranchCC(
      static_cast<Instruction::Opcode>(opcode)));
}

/* ---- Phase A: Operand setters ---- */

extern "C" void
jit_lir_operand_set_constant(JitLirOperand op, uint64_t val, int dt) {
  static_cast<jit::lir::Operand*>(
      static_cast<OperandBase*>(op))
      ->setConstant(val, static_cast<jit::lir::DataType>(dt));
}

extern "C" void
jit_lir_operand_set_fp_constant(JitLirOperand op, double val) {
  static_cast<jit::lir::Operand*>(
      static_cast<OperandBase*>(op))->setFPConstant(val);
}

extern "C" void
jit_lir_operand_set_phy_register(JitLirOperand op, int loc) {
  static_cast<jit::lir::Operand*>(
      static_cast<OperandBase*>(op))
      ->setPhyRegister(jit::codegen::PhyLocation(loc));
}

extern "C" void
jit_lir_operand_set_stack_slot(JitLirOperand op, int loc) {
  static_cast<jit::lir::Operand*>(
      static_cast<OperandBase*>(op))
      ->setStackSlot(jit::codegen::PhyLocation(loc));
}

extern "C" void
jit_lir_operand_set_virtual_register(JitLirOperand op) {
  static_cast<jit::lir::Operand*>(
      static_cast<OperandBase*>(op))->setVirtualRegister();
}

extern "C" void
jit_lir_operand_set_data_type(JitLirOperand op, int dt) {
  static_cast<jit::lir::Operand*>(
      static_cast<OperandBase*>(op))
      ->setDataType(static_cast<jit::lir::DataType>(dt));
}

extern "C" void
jit_lir_operand_set_basic_block(JitLirOperand op, JitLirBlock block) {
  static_cast<jit::lir::Operand*>(
      static_cast<OperandBase*>(op))
      ->setBasicBlock(static_cast<BasicBlock*>(block));
}

extern "C" void
jit_lir_operand_set_mem_address(JitLirOperand op, void* addr) {
  static_cast<jit::lir::Operand*>(
      static_cast<OperandBase*>(op))->setMemoryAddress(addr);
}

extern "C" void
jit_lir_operand_set_last_use(JitLirOperand op) {
  static_cast<OperandBase*>(op)->setLastUse();
}

/* ---- Phase A: Instruction mutation ---- */

extern "C" void
jit_lir_instr_set_opcode(JitLirInstr instr, int opcode) {
  static_cast<Instruction*>(instr)->setOpcode(
      static_cast<Instruction::Opcode>(opcode));
}

/* ---- Phase A: Instruction operand allocation ---- */

extern "C" JitLirOperand
jit_lir_instr_alloc_imm_input(JitLirInstr instr, uint64_t val, int dt) {
  auto* inst = static_cast<Instruction*>(instr);
  auto* op = inst->allocateImmediateInput(val, static_cast<jit::lir::DataType>(dt));
  return op;
}

extern "C" JitLirOperand
jit_lir_instr_alloc_linked_input(JitLirInstr instr, JitLirInstr def_instr) {
  auto* inst = static_cast<Instruction*>(instr);
  auto* op = inst->allocateLinkedInput(static_cast<Instruction*>(def_instr));
  return op;
}

extern "C" JitLirOperand
jit_lir_instr_alloc_phyreg_input(JitLirInstr instr, int loc) {
  auto* inst = static_cast<Instruction*>(instr);
  auto* op = inst->allocatePhyRegisterInput(jit::codegen::PhyLocation(loc));
  return op;
}

extern "C" JitLirOperand
jit_lir_instr_alloc_label_input(JitLirInstr instr, JitLirBlock block) {
  auto* inst = static_cast<Instruction*>(instr);
  auto* op = inst->allocateLabelInput(static_cast<BasicBlock*>(block));
  return op;
}

extern "C" JitLirOperand
jit_lir_instr_alloc_addr_input(JitLirInstr instr, void* addr) {
  auto* inst = static_cast<Instruction*>(instr);
  auto* op = inst->allocateAddressInput(addr);
  return op;
}

/* ---- Phase A: Block/Function allocation ---- */

extern "C" JitLirBlock
jit_lir_func_alloc_block(JitLirFunc func) {
  return static_cast<Function*>(func)->allocateBasicBlock();
}

extern "C" JitLirInstr
jit_lir_block_alloc_instr(JitLirBlock block, int opcode,
                          const void* hir_origin) {
  auto* bb = static_cast<BasicBlock*>(block);
  auto* inst = bb->allocateInstr(
      static_cast<Instruction::Opcode>(opcode),
      static_cast<const jit::hir::Instr*>(hir_origin));
  return inst;
}

extern "C" void
jit_lir_block_add_successor(JitLirBlock block, JitLirBlock succ) {
  static_cast<BasicBlock*>(block)->addSuccessor(
      static_cast<BasicBlock*>(succ));
}

/* ---- Phase A: Missing setter/mutation implementations ---- */

extern "C" void
jit_lir_instr_set_num_inputs(JitLirInstr instr, size_t n) {
  static_cast<Instruction*>(instr)->setNumInputs(n);
}

extern "C" const char*
jit_lir_instr_opname(JitLirInstr instr) {
  auto sv = static_cast<Instruction*>(instr)->opname();
  return sv.data();
}

extern "C" JitLirOperand
jit_lir_instr_alloc_fp_imm_input(JitLirInstr instr, double val) {
  return static_cast<Instruction*>(instr)->allocateFPImmediateInput(val);
}

extern "C" JitLirOperand
jit_lir_instr_alloc_stack_input(JitLirInstr instr, int loc) {
  return static_cast<Instruction*>(instr)->allocateStackInput(
      jit::codegen::PhyLocation(loc));
}
