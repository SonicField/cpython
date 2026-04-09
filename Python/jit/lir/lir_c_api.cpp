/*
 * lir_c_api.cpp -- C accessor implementations for LIR types
 *
 * Phase 3D: Thin C++ wrappers that delegate to LIR class methods.
 * Each function casts the opaque void* handle to the correct C++ type.
 *
 * Only functions with active .c callers are implemented here.
 * Do NOT add speculative wrapper functions — convert the underlying
 * C++ to C instead (Phase 3D directive).
 */

#include "cinderx/Jit/lir/lir_c_api.h"
#include "cinderx/module_state.h"

#include <iterator>
#include <sstream>

#include "cinderx/Jit/codegen/code_section.h"
#include "cinderx/Jit/codegen/environ.h"
#include "cinderx/Jit/config.h"
#include "cinderx/Jit/jit_config_c.h"
#include "cinderx/Jit/gen_data_footer.h"
#include "cinderx/Jit/lir/block.h"
#include "cinderx/Jit/lir/function.h"
#include "cinderx/Jit/lir/instruction.h"
#include "cinderx/Jit/lir/operand.h"
#include "cinderx/Jit/lir/printer.h"
#include "cinderx/Jit/threaded_compile.h"
#include "cinderx/Jit/code_runtime.h"
#include "cinderx/Jit/code_patcher.h"
#include "cinderx/Jit/hir/function.h"
#include "cinderx/Jit/hir/printer.h"
#include "cinderx/Jit/hir/type.h"
#include "cinderx/Jit/symbolizer.h"
#include "cinderx/Jit/jit_rt.h"

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
  auto* f = static_cast<Function*>(func);
  return f->num_blocks_ > 0 ? f->blocks_[0] : nullptr;
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
      static_cast<BasicBlock*>(block)->instr_tail_);
}

extern "C" JitLirInstr
jit_lir_block_get_first_instr(JitLirBlock block) {
  return const_cast<Instruction*>(
      static_cast<BasicBlock*>(block)->instr_head_);
}

extern "C" size_t
jit_lir_block_num_instrs(JitLirBlock block) {
  return static_cast<BasicBlock*>(block)->num_instrs_;
}

extern "C" JitLirBlock
jit_lir_block_get_false_succ(JitLirBlock block) {
  return static_cast<BasicBlock*>(block)->successors_[1];
}

extern "C" int
jit_lir_block_get_section(JitLirBlock block) {
  return static_cast<int>(static_cast<BasicBlock*>(block)->section_);
}

extern "C" void
jit_lir_block_set_section(JitLirBlock block, int section) {
  static_cast<BasicBlock*>(block)->setSection(
      static_cast<CodeSection>(section));
}

extern "C" int
jit_lir_block_get_id(JitLirBlock block) {
  return static_cast<BasicBlock*>(block)->id_;
}

extern "C" JitLirInstr
jit_lir_block_get_instr_at(JitLirBlock block, size_t index) {
  auto instrs = static_cast<BasicBlock*>(block)->instructions();
  Instruction* cur = instrs.front();
  for (size_t i = 0; i < index && cur; i++) {
    cur = cur->next_;
  }
  return cur;
}

/* ---- Instruction accessors ---- */

extern "C" int
jit_lir_instr_opcode(JitLirInstr instr) {
  return static_cast<int>(static_cast<Instruction*>(instr)->opcode_);
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

extern "C" JitLirOperand
jit_lir_instr_get_input(JitLirInstr instr, size_t index) {
  return static_cast<Instruction*>(instr)->inputs_[index];
}

extern "C" JitLirOperand
jit_lir_instr_output(JitLirInstr instr) {
  return &static_cast<Instruction*>(instr)->output_;
}

/* ---- Operand accessors ---- */

extern "C" JitLirBlock
jit_lir_operand_get_basic_block(JitLirOperand op) {
  return static_cast<OperandBase*>(op)->getBasicBlock();
}

/* ---- Opcode constants ---- */

extern "C" int
jit_lir_opcode_guard(void) {
  return static_cast<int>(Instruction::kGuard);
}

/* ---- DCE instruction accessors ---- */

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
  Instruction* instr = bb->instr_head_;
  while (instr) {
    Instruction* next = instr->next_;
    if (!is_live(instr, ctx)) {
      delete bb->removeInstr(instr);
    }
    instr = next;
  }
}

/* ---- Environ accessors for C callers ---- */

extern "C" LirPhyLocation
jit_environ_get_arg_location(void* env_ptr, size_t index) {
  auto* env = static_cast<jit::codegen::Environ*>(env_ptr);
  auto loc = env->arg_locations[index];
  return LirPhyLocation{loc.loc, loc.bitSize};
}

extern "C" LirPhyLocation
jit_environ_get_return_reg(int index) {
  auto loc = jit::codegen::RETURN_REGS[index];
  return LirPhyLocation{loc.loc, loc.bitSize};
}

extern "C" int
jit_environ_get_max_arg_buffer(void* env_ptr) {
  return static_cast<jit::codegen::Environ*>(env_ptr)->max_arg_buffer_size;
}

extern "C" void
jit_environ_update_max_arg_buffer(void* env_ptr, int size) {
  auto* env = static_cast<jit::codegen::Environ*>(env_ptr);
  if (size > env->max_arg_buffer_size) {
    env->max_arg_buffer_size = size;
  }
}

/* ---- Architecture register constants ---- */

static LirPhyLocation phyloc_from(jit::codegen::PhyLocation loc) {
  return LirPhyLocation{loc.loc, loc.bitSize};
}

extern "C" size_t jit_arch_num_arg_regs(void) {
  return jit::codegen::ARGUMENT_REGS.size();
}

extern "C" size_t jit_arch_num_fp_arg_regs(void) {
  return jit::codegen::FP_ARGUMENT_REGS.size();
}

extern "C" LirPhyLocation jit_arch_arg_reg(size_t index) {
  return phyloc_from(jit::codegen::ARGUMENT_REGS[index]);
}

extern "C" LirPhyLocation jit_arch_fp_arg_reg(size_t index) {
  return phyloc_from(jit::codegen::FP_ARGUMENT_REGS[index]);
}

extern "C" LirPhyLocation jit_arch_scratch_0_loc(void) {
  return phyloc_from(jit::codegen::arch::reg_scratch_0_loc);
}

extern "C" LirPhyLocation jit_arch_stack_pointer_loc(void) {
  return phyloc_from(jit::codegen::arch::reg_stack_pointer_loc);
}

extern "C" LirPhyLocation jit_arch_general_return_loc(void) {
  return phyloc_from(jit::codegen::arch::reg_general_return_loc);
}

extern "C" LirPhyLocation jit_arch_double_return_loc(void) {
  return phyloc_from(jit::codegen::arch::reg_double_return_loc);
}

extern "C" int
jit_is_frame_mode_lightweight(void) {
  return jit_get_config()->frame_mode == JIT_FRAME_LIGHTWEIGHT ? 1 : 0;
}

extern "C" void*
jit_environ_get_phx_builder(void* env_ptr) {
  auto* env = static_cast<jit::codegen::Environ*>(env_ptr);
  return env->as->impl();
}

extern "C" int
jit_environ_is_generator(void* env_ptr) {
#if defined(CINDER_AARCH64)
  return static_cast<jit::codegen::Environ*>(env_ptr)->is_generator ? 1 : 0;
#else
  (void)env_ptr;
  return 0;
#endif
}

extern "C" int
jit_environ_saved_ip_fp_offset(void* env_ptr) {
#if defined(CINDER_AARCH64)
  return static_cast<jit::codegen::Environ*>(env_ptr)->saved_ip_fp_offset;
#else
  (void)env_ptr;
  return 0;
#endif
}

extern "C" void
jit_environ_add_pending_debug_loc(void* env_ptr, PhxLabel label,
                                   const void* origin) {
  auto* env = static_cast<jit::codegen::Environ*>(env_ptr);
  asmjit::Label asmjit_label(label);
  env->pending_debug_locs.emplace_back(
      asmjit_label,
      static_cast<const jit::hir::Instr*>(origin));
}

extern "C" int
jit_gen_data_footer_saved_ip_offset(void) {
#if defined(__aarch64__)
  return (int)offsetof(jit::GenDataFooter, savedIP);
#else
  return -1;  // savedIP only exists on ARM64
#endif
}

extern "C" void*
jit_environ_get_code_rt(void* env_ptr) {
  return static_cast<jit::codegen::Environ*>(env_ptr)->code_rt;
}

extern "C" void
jit_environ_add_deopt_exit(void* env_ptr, size_t index, PhxLabel label,
                            const LirInstruction* instr) {
  auto* env = static_cast<jit::codegen::Environ*>(env_ptr);
  asmjit::Label asmjit_label(label);
  env->deopt_exits.emplace_back(
      index, asmjit_label,
      reinterpret_cast<const jit::lir::Instruction*>(instr));
}

extern "C" void
jit_fill_live_value_locations(void* code_rt_ptr, size_t deopt_idx,
                               const LirInstruction* instr_ptr,
                               size_t begin_input, size_t end_input) {
  auto* code_rt = static_cast<jit::CodeRuntime*>(code_rt_ptr);
  auto* instr = reinterpret_cast<const jit::lir::Instruction*>(instr_ptr);
  jit::ThreadedCompileSerialize guard;
  auto& deopt_meta = code_rt->getDeoptMetadata(deopt_idx);
  for (size_t i = begin_input; i < end_input; i++) {
    auto loc = instr->inputs_[i]->getPhyRegOrStackSlot();
    deopt_meta.live_values[i - begin_input].location = loc;
  }
}

extern "C" void
jit_jump_patcher_stored_bytes(void* patcher_ptr,
                               const uint8_t** data, size_t* size) {
  auto* patcher = static_cast<jit::JumpPatcher*>(patcher_ptr);
  auto bytes = patcher->storedBytes();
  *data = bytes.data();
  *size = bytes.size();
}

extern "C" void
jit_environ_add_pending_deopt_patcher(void* env_ptr, void* patcher_ptr,
                                       PhxLabel patchpoint, PhxLabel deopt_exit) {
  auto* env = static_cast<jit::codegen::Environ*>(env_ptr);
  auto* patcher = static_cast<jit::JumpPatcher*>(patcher_ptr);
  env->pending_deopt_patchers.emplace_back(
      patcher, asmjit::Label(patchpoint), asmjit::Label(deopt_exit));
}

extern "C" PhxLabel
jit_environ_get_block_label(void* env_ptr, const LirBasicBlock* block) {
  auto* env = static_cast<jit::codegen::Environ*>(env_ptr);
  auto* bb = reinterpret_cast<const jit::lir::BasicBlock*>(block);
  auto it = env->block_label_map.find(const_cast<jit::lir::BasicBlock*>(bb));
  JIT_CHECK(it != env->block_label_map.end(), "Block label not found");
  return it->second;
}

extern "C" int
jit_environ_shadow_frames_and_spill_size(void* env_ptr) {
  return static_cast<jit::codegen::Environ*>(env_ptr)->shadow_frames_and_spill_size;
}

extern "C" void*
jit_environ_get_gen_resume_entry_label(void* env_ptr) {
  auto* env = static_cast<jit::codegen::Environ*>(env_ptr);
  return &env->gen_resume_entry_label;
}

extern "C" int
jit_environ_get_frame_mode(void* env_ptr) {
  return static_cast<int>(
      static_cast<jit::codegen::Environ*>(env_ptr)->frame_mode);
}

/* ---- HIR Function accessors ---- */

extern "C" void*
jit_hir_func_get_code(const void* hir_func) {
  auto* func = static_cast<const jit::hir::Function*>(hir_func);
  return func->code.get();
}

extern "C" int
jit_hir_func_returns_double(const void* hir_func) {
  auto* func = static_cast<const jit::hir::Function*>(hir_func);
  return (func->return_type <= jit::hir::TCDouble) ? 1 : 0;
}

extern "C" int
jit_hir_func_is_gen(const void* hir_func) {
  auto* func = static_cast<const jit::hir::Function*>(hir_func);
  int gen_flags = CO_GENERATOR | CO_COROUTINE | CO_ASYNC_GENERATOR |
                  CO_ITERABLE_COROUTINE;
  return (func->code->co_flags & gen_flags) ? 1 : 0;
}

extern "C" int
jit_hir_func_get_frame_mode(const void* hir_func) {
  return static_cast<int>(
      static_cast<const jit::hir::Function*>(hir_func)->frameMode);
}

extern "C" void*
jit_hir_func_get_reifier(const void* hir_func) {
  auto* func = static_cast<const jit::hir::Function*>(hir_func);
  return func->reifier.get();
}

/* ---- CodeRuntime accessors ---- */

extern "C" void*
jit_code_rt_get_reifier(void* code_rt_ptr) {
  auto* code_rt = static_cast<jit::CodeRuntime*>(code_rt_ptr);
  return code_rt->reifier();
}

extern "C" void*
jit_module_state_get_frame_reifier(void) {
  return cinderx::getModuleState()->frameReifier();
}

/* ---- JITRT function address getters ---- */

extern "C" void*
jit_rt_get_unlink_frame_addr(void) {
  return reinterpret_cast<void*>(JITRT_UnlinkFrame);
}

extern "C" void*
jit_rt_get_alloc_link_frame_debug_addr(void) {
  return reinterpret_cast<void*>(JITRT_AllocateAndLinkInterpreterFrame_Debug);
}

extern "C" void*
jit_rt_get_alloc_link_frame_release_addr(void) {
  return reinterpret_cast<void*>(JITRT_AllocateAndLinkInterpreterFrame_Release);
}

extern "C" void*
jit_rt_get_alloc_link_gen_frame_addr(void) {
  return reinterpret_cast<void*>(JITRT_AllocateAndLinkGenAndInterpreterFrame);
}

/* ---- LIR printer support (Phase 3D: printer.cpp → printer_c.c) ---- */

extern "C" const char*
lir_instruction_opcode_name(int opcode) {
  auto& info = jit::lir::InstrProperty::getProperties(
      static_cast<jit::lir::Instruction::Opcode>(opcode));
  return info.name.data();
}

extern "C" const char*
lir_operand_data_type_name(int dt) {
  using DT = jit::lir::DataType;
  switch (static_cast<DT>(dt)) {
    case DT::kObject:  return "Object";
    case DT::k8bit:    return "8bit";
    case DT::k16bit:   return "16bit";
    case DT::k32bit:   return "32bit";
    case DT::k64bit:   return "64bit";
    case DT::kDouble:  return "Double";
  }
  return "?";
}

extern "C" void
lir_hir_print_instr(FILE *out, const void *hir_instr) {
  if (!hir_instr) return;
  auto* instr = static_cast<const jit::hir::Instr*>(hir_instr);
  jit::hir::HIRPrinter printer;
  printer.setFullSnapshots(true);
  printer.setLinePrefix("# ");
  std::ostringstream ss;
  printer.Print(ss, *instr);
  fprintf(out, "%s", ss.str().c_str());
}

extern "C" const void*
lir_instruction_get_origin(const LirInstruction *inst) {
  auto* i = reinterpret_cast<const jit::lir::Instruction*>(inst);
  return static_cast<const void*>(i->origin());
}

/* ---- Symbolizer bridge (disassembler_c.c) ---- */

extern "C" int
jit_symbolize(const void *func, char *buf, size_t buflen) {
  auto name = jit::symbolize(func);
  if (!name.has_value() || name->empty()) return 0;
  std::strncpy(buf, name->c_str(), buflen - 1);
  buf[buflen - 1] = '\0';
  return 1;
}
