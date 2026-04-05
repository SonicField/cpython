// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cinderx/Jit/codegen/autogen.h"
#include "cinderx/Jit/codegen/autogen_translate_c.h"
#include "cinderx/Jit/gen_data_footer.h"

#include "cinderx/Common/util.h"
#include "cinderx/Jit/code_patcher.h"
#include "cinderx/Jit/codegen/arch.h"
#include "cinderx/Jit/codegen/gen_asm_utils.h"
#include "cinderx/Jit/frame.h"
#include "cinderx/Jit/generators_rt.h"
#include "cinderx/Jit/jit_rt.h"
#include "cinderx/Jit/lir/instruction.h"
#include "cinderx/Jit/lir/printer.h"

#include "jit/phoenix_asm/x86_64.h"
#include "jit/phoenix_asm/arm64.h"

#include <type_traits>
#include <vector>

using namespace asmjit;
using namespace jit::lir;
using namespace jit::codegen;

namespace jit::codegen::autogen {

#define ANY "*"

namespace {
// Add a pattern to an existing trie tree. If the trie tree is nullptr, create a
// new one.
std::unique_ptr<PatternNode> addPattern(
    std::unique_ptr<PatternNode> patterns,
    const std::string& s,
    PatternNode::func_t func) {
  JIT_DCHECK(!s.empty(), "pattern string should not be empty.");

  if (patterns == nullptr) {
    patterns = std::make_unique<PatternNode>();
  }

  PatternNode* cur = patterns.get();
  for (auto& c : s) {
    auto iter = cur->next.find(c);
    if (iter == cur->next.end()) {
      cur = cur->next.emplace(c, std::make_unique<PatternNode>())
                .first->second.get();
      continue;
    }
    cur = iter->second.get();
  }

  JIT_DCHECK(cur->func == nullptr, "Found duplicated pattern.");
  cur->func = func;

  return patterns;
}

// Find the function associated to the pattern given in s.
PatternNode::func_t findByPattern(
    const PatternNode* patterns,
    const std::string& s) {
  auto cur = patterns;
  if (s.empty()) {
    // handle the special case of matching '*' with an empty string
    auto iter = cur->next.find('*');
    if (iter != cur->next.end()) {
      cur = iter->second.get();
      return cur->func;
    }
  }
  for (auto& c : s) {
    auto iter = cur->next.find(c);
    if (iter != cur->next.end()) {
      cur = iter->second.get();
      continue;
    }

    iter = cur->next.find('?');
    if (iter != cur->next.end()) {
      cur = iter->second.get();
      continue;
    }

    iter = cur->next.find('*');
    if (iter != cur->next.end()) {
      cur = iter->second.get();
      break;
    }

    return nullptr;
  }

  return cur->func;
}

} // namespace

// this function generates operand patterns from the inputs and outputs
// of a given instruction instr and calls the correspoinding code generation
// functions.
void AutoTranslator::translateInstr(Environ* env, const Instruction* instr)
    const {
  auto opcode = instr->opcode_;
  if (opcode == Instruction::kBind) {
    return;
  }

  // Try C dispatch first — handles converted opcodes directly.
  // Falls through to C++ trie only for unconverted opcodes (Yield*, ASM on x86).
  if (autogen_c_dispatch(env, reinterpret_cast<const LirInstruction*>(instr))) {
    return;
  }

  auto& instr_map = map_get(instr_rule_map_, opcode);

  std::string pattern;
  pattern.reserve(instr->num_inputs_ + instr->getNumOutputs());

  if (instr->getNumOutputs()) {
    auto operand = (&instr->output_);

    switch (operand->type()) {
      case OperandBase::kReg:
        pattern += (operand->isVecD() ? "X" : "R");
        break;
      case OperandBase::kStack:
      case OperandBase::kMem:
      case OperandBase::kInd:
        pattern += "M";
        break;
      default:
        JIT_ABORT("Output operand has to be of type register or memory");
    }
  }

  instr->foreachInputOperand([&](const OperandBase* operand) {
    switch (operand->type()) {
      case OperandBase::kReg:
        pattern += (operand->isVecD() ? "x" : "r");
        break;
      case OperandBase::kStack:
      case OperandBase::kMem:
      case OperandBase::kInd:
        pattern += "m";
        break;
      case OperandBase::kImm:
        pattern += "i";
        break;
      case OperandBase::kLabel:
        pattern += "b";
        break;
      default:
        JIT_ABORT(
            "Illegal input type {} for instruction {}",
            operand->type(),
            *instr);
    }
  });

  auto func = findByPattern(instr_map.get(), pattern);
  JIT_CHECK(
      func != nullptr,
      "No pattern found for opcode {}: {}",
      InstrProperty::getProperties(instr).name,
      pattern);
  func(env, instr);
}

namespace {

void fillLiveValueLocations(
    CodeRuntime* code_runtime,
    std::size_t deopt_idx,
    const Instruction* instr,
    size_t begin_input,
    size_t end_input) {
  ThreadedCompileSerialize guard;

  DeoptMetadata& deopt_meta = code_runtime->getDeoptMetadata(deopt_idx);
  for (size_t i = begin_input; i < end_input; i++) {
    auto loc = instr->inputs_[i]->getPhyRegOrStackSlot();
    deopt_meta.live_values[i - begin_input].location = loc;
  }
}


// Store meta-data about this yield in a generator suspend data pointed to by
// suspend_data_r. Data includes things like the address to resume execution at,
// and owned entries in the suspended spill data needed for GC operations etc.
void emitStoreGenYieldPoint(
    arch::Builder* as,
    Environ* env,
    const Instruction* yield,
    asmjit::Label resume_label,
    arch::Gp suspend_data_r,
    arch::Gp scratch_r) {
  bool is_yield_from = yield->isYieldFrom() ||
      yield->isYieldFromSkipInitialSend() ||
      yield->isYieldFromHandleStopAsyncIteration();

  auto calc_spill_offset = [&](size_t live_input_n) {
    PhyLocation mem = yield->inputs_[live_input_n]->getStackSlot();
    return mem.loc / kPointerSize;
  };

  size_t input_n = yield->num_inputs_ - 1;
  size_t deopt_idx = yield->inputs_[input_n]->getConstant();

  size_t live_regs_input = input_n - 1;
  int num_live_regs = yield->inputs_[live_regs_input]->getConstant();
  fillLiveValueLocations(
      env->code_rt,
      deopt_idx,
      yield,
      live_regs_input - num_live_regs,
      live_regs_input);

  auto yield_from_offset =
      is_yield_from ? calc_spill_offset(2) : kInvalidYieldFromOffset;
  GenYieldPoint* gen_yield_point = env->code_rt->addGenYieldPoint(
      GenYieldPoint{deopt_idx, yield_from_offset});

  env->unresolved_gen_entry_labels.emplace(gen_yield_point, resume_label);
  if (yield->origin_) {
    env->pending_debug_locs.emplace_back(resume_label, yield->origin_);
  }

  PhxBuilder* pb = as->impl();

#if defined(CINDER_X86_64)
  phx_x86_mov_ri(pb, scratch_r, reinterpret_cast<uint64_t>(gen_yield_point));
  auto yieldPointOffset = offsetof(GenDataFooter, yieldPoint);
  phx_x86_mov_mr(pb, phx_qword_ptr(suspend_data_r, yieldPointOffset), scratch_r);
#elif defined(CINDER_AARCH64)
  phx_a64_mov_ri(pb, scratch_r, reinterpret_cast<uint64_t>(gen_yield_point));
  auto yieldPointOffset = offsetof(GenDataFooter, yieldPoint);
  phx_a64_str(pb,
      scratch_r,
      arch::ptr_resolve(
          as, suspend_data_r, yieldPointOffset, arch::reg_scratch_0));
#else
  (void)gen_yield_point;
  CINDER_UNSUPPORTED
#endif
}

void emitLoadResumedYieldInputs(
    arch::Builder* as,
    const Instruction* instr,
    PhyLocation sent_in_source_loc,
    arch::Gp tstate_reg) {
  PhxBuilder* pb = as->impl();

#if defined(CINDER_X86_64)
  PhyLocation tstate = instr->inputs_[0]->getStackSlot();
  phx_x86_mov_mr(pb, phx_qword_ptr(x86::rbp, tstate.loc), tstate_reg);

  const lir::Operand* target = (&instr->output_);

  if (target->isStack()) {
    phx_x86_mov_mr(pb,
        phx_qword_ptr(x86::rbp, target->getStackSlot().loc),
        x86::gpq(sent_in_source_loc.loc));
    return;
  }

  if (target->isReg()) {
    PhyLocation target_loc = target->getPhyRegister();
    if (target_loc != sent_in_source_loc) {
      phx_x86_mov_rr(pb, x86::gpq(target_loc.loc), x86::gpq(sent_in_source_loc.loc));
    }
    return;
  }

  JIT_CHECK(
      target->isNone(),
      "Have an output that isn't a register or a stack slot, {}",
      target->type());
#elif defined(CINDER_AARCH64)
  PhyLocation tstate = instr->inputs_[0]->getStackSlot();
  phx_a64_str(pb,
      tstate_reg,
      arch::ptr_resolve(as, arch::fp, tstate.loc, arch::reg_scratch_0));

  const lir::Operand* target = (&instr->output_);

  if (target->isStack()) {
    phx_a64_str(pb,
        a64::x(sent_in_source_loc.loc),
        arch::ptr_resolve(
            as, arch::fp, target->getStackSlot().loc, arch::reg_scratch_0));
    return;
  }

  if (target->isReg()) {
    PhyLocation target_loc = target->getPhyRegister();
    if (target_loc != sent_in_source_loc) {
      phx_a64_mov_rr(pb, a64::x(target_loc.loc), a64::x(sent_in_source_loc.loc));
    }
    return;
  }

  JIT_CHECK(
      target->isNone(),
      "Have an output that isn't a register or a stack slot, {}",
      target->type());
#else
  CINDER_UNSUPPORTED
#endif
}

void translateYieldInitial(Environ* env, const Instruction* instr) {
  arch::Builder* as = env->as;
  PhxBuilder* pb = as->impl();

#if defined(CINDER_X86_64)
#if PY_VERSION_HEX < 0x030C0000
  // Load tstate into RDI for call to JITRT_MakeGenObject*.

  // Consider avoiding reloading the tstate in from memory if it was already in
  // a register before spilling. Still needs to be in memory though so it can be
  // recovered after calling JITRT_MakeGenObject* which will trash it.
  PhyLocation tstate = instr->inputs_[0]->getStackSlot();
  phx_x86_mov_rm(pb, x86::rdi, phx_qword_ptr(x86::rbp, tstate.loc));

  // Make a generator object to be returned by the epilogue.
  phx_x86_lea(pb, x86::rsi, x86::ptr(env->gen_resume_entry_label));
  JIT_CHECK(
      env->shadow_frames_and_spill_size % kPointerSize == 0,
      "Bad spill alignment");
  phx_x86_mov_ri(pb, x86::rdx, env->shadow_frames_and_spill_size / kPointerSize);
  phx_x86_mov_ri(pb, x86::rcx, reinterpret_cast<uint64_t>(env->code_rt));
  JIT_CHECK(instr->origin_->IsInitialYield(), "expected InitialYield");
  PyCodeObject* code = static_cast<const hir::InitialYield*>(instr->origin_)
                           ->frameState()
                           ->code;
  phx_x86_mov_ri(pb, x86::r8, reinterpret_cast<uint64_t>(code));
  if (code->co_flags & CO_COROUTINE) {
    emitCall(*env, reinterpret_cast<uint64_t>(JITRT_MakeGenObjectCoro), instr);
  } else if (code->co_flags & CO_ASYNC_GENERATOR) {
    emitCall(
        *env, reinterpret_cast<uint64_t>(JITRT_MakeGenObjectAsyncGen), instr);
  } else {
    emitCall(*env, reinterpret_cast<uint64_t>(JITRT_MakeGenObject), instr);
  }
  // Resulting generator is now in RAX for filling in below and epilogue return.
  const auto gen_reg = x86::rax;

  // Exit early if return from JITRT_MakeGenObject was nullptr.
  phx_x86_test_rr(pb, gen_reg, gen_reg);
  phx_x86_jz(pb, env->hard_exit_label);

  // Set RDI to gen->gi_jit_data for use in emitStoreGenYieldPoint() and data
  // copy using 'movsq' below.
  auto gi_jit_data_offset = offsetof(PyGenObject, gi_jit_data);
  phx_x86_mov_rm(pb, x86::rdi, phx_qword_ptr(gen_reg, gi_jit_data_offset));

  // Arbitrary scratch register for use in emitStoreGenYieldPoint().
  auto scratch_r = x86::r9;
  asmjit::Label resume_label = Label(phx_builder_new_label(pb));
  emitStoreGenYieldPoint(as, env, instr, resume_label, x86::rdi, scratch_r);

  // Store variables spilled by this point to generator.
  int spill_bytes = env->initial_yield_spill_size_;
  JIT_CHECK(spill_bytes % kPointerSize == 0, "Bad spill alignment");

  // Point rsi at the bottom word of the current spill space.
  phx_x86_lea(pb, x86::rsi, phx_ptr(x86::rbp, -spill_bytes));
  // Point rdi at the bottom word of the generator's spill space.
  phx_x86_sub_ri(pb, x86::rdi, spill_bytes);
  phx_x86_mov_ri(pb, x86::rcx, spill_bytes / kPointerSize);
  // TODO(phoenix-asm): rep movsq not yet in phoenix-asm C API
  as->rep().movsq();

  // Jump to bottom half of epilogue
  phx_x86_jmp_label(pb, env->hard_exit_label);

  // Resumed execution in this generator begins here
  phx_builder_bind(pb, resume_label);

  // Sent in value is in RSI, and tstate is in RCX from resume entry-point args
  emitLoadResumedYieldInputs(as, instr, RSI, x86::rcx);
#else
  // Load tstate into RDI for call to
  // JITRT_UnlinkGenFrameAndReturnGenDataFooter.

  // Consider avoiding reloading the tstate in from memory if it was already in
  // a register before spilling. Still needs to be in memory though so it can be
  // recovered after calling JITRT_MakeGenObject* which will trash it.
  PhyLocation tstate = instr->inputs_[0]->getStackSlot();
  phx_x86_mov_rm(pb, x86::rdi, phx_qword_ptr(x86::rbp, tstate.loc));

  emitCall(
      *env,
      reinterpret_cast<uint64_t>(JITRT_UnlinkGenFrameAndReturnGenDataFooter),
      instr);
  // This will return pointers to a generator in RAX and JIT data in RDX.

  // Arbitrary scratch register for use in emitStoreGenYieldPoint(). Any
  // caller-saved register not used in this scope will do because we're on the
  // exit path now.
  auto scratch_r = x86::r9;
  asmjit::Label resume_label = Label(phx_builder_new_label(pb));
  emitStoreGenYieldPoint(as, env, instr, resume_label, x86::rdx, scratch_r);

  // Jump to epilogue
  phx_x86_jmp_label(pb, env->exit_for_yield_label);

  // Resumed execution in this generator begins here
  phx_builder_bind(pb, resume_label);

  // Sent in value is in RSI, and tstate is in RCX from resume entry-point args
  emitLoadResumedYieldInputs(as, instr, RSI, x86::rcx);
#endif
#elif defined(CINDER_AARCH64)
#if PY_VERSION_HEX < 0x030C0000
  CINDER_UNSUPPORTED
#else
  // Load tstate into X0 for call to
  // JITRT_UnlinkGenFrameAndReturnGenDataFooter.

  // Consider avoiding reloading the tstate in from memory if it was already in
  // a register before spilling. Still needs to be in memory though so it can be
  // recovered after calling JITRT_MakeGenObject* which will trash it.
  PhyLocation tstate = instr->inputs_[0]->getStackSlot();
  phx_a64_ldr(pb,
      a64::x0,
      arch::ptr_resolve(as, arch::fp, tstate.loc, arch::reg_scratch_0));

  emitCall(
      *env,
      reinterpret_cast<uint64_t>(JITRT_UnlinkGenFrameAndReturnGenDataFooter),
      instr);
  // This will return pointers to a generator in X0 and JIT data in X1.

  // Arbitrary scratch register for use in emitStoreGenYieldPoint(). Any
  // caller-saved register not used in this scope will do because we're on the
  // exit path now.
  auto scratch_r = arch::reg_scratch_0;
  asmjit::Label resume_label = Label(phx_builder_new_label(pb));
  emitStoreGenYieldPoint(as, env, instr, resume_label, a64::x1, scratch_r);

  // Jump to epilogue
  phx_a64_b(pb, env->exit_for_yield_label);

  // Resumed execution in this generator begins here
  phx_builder_bind(pb, resume_label);

  // Sent in value is in X1, and tstate is in X3 from resume entry-point args
  emitLoadResumedYieldInputs(as, instr, X1, a64::x3);
#endif
#else
  CINDER_UNSUPPORTED
#endif
}

void translateYieldValue(Environ* env, const Instruction* instr) {
  arch::Builder* as = env->as;
  PhxBuilder* pb = as->impl();

#if defined(CINDER_X86_64)
  // Make sure tstate is in RDI for use in epilogue.
  PhyLocation tstate = instr->inputs_[0]->getStackSlot();
  phx_x86_mov_rm(pb, x86::rdi, phx_qword_ptr(x86::rbp, tstate.loc));

  // Value to send goes to RAX so it can be yielded (returned) by epilogue.
  if (instr->inputs_[1]->isImm()) {
    phx_x86_mov_ri(pb, x86::rax, instr->inputs_[1]->getConstant());
  } else {
    PhyLocation value_out = instr->inputs_[1]->getStackSlot();
    phx_x86_mov_rm(pb, x86::rax, phx_qword_ptr(x86::rbp, value_out.loc));
  }

  // Arbitrary scratch register for use in emitStoreGenYieldPoint()
  auto scratch_r = x86::r9;
  auto resume_label = Label(phx_builder_new_label(pb));
  emitStoreGenYieldPoint(as, env, instr, resume_label, x86::rbp, scratch_r);

  // Jump to epilogue
  phx_x86_jmp_label(pb, env->exit_for_yield_label);

  // Resumed execution in this generator begins here
  phx_builder_bind(pb, resume_label);

  // Sent in value is in RSI, and tstate is in RCX from resume entry-point args
  emitLoadResumedYieldInputs(as, instr, RSI, x86::rcx);
#elif defined(CINDER_AARCH64)
  // Make sure tstate is in x2 for use in epilogue.
  PhyLocation tstate = instr->inputs_[0]->getStackSlot();
  phx_a64_ldr(pb,
      a64::x2,
      arch::ptr_resolve(as, arch::fp, tstate.loc, arch::reg_scratch_0));

  // Value to send goes to x0 so it can be yielded (returned) by epilogue.
  if (instr->inputs_[1]->isImm()) {
    phx_a64_mov_ri(pb, a64::x0, instr->inputs_[1]->getConstant());
  } else {
    PhyLocation value_out = instr->inputs_[1]->getStackSlot();
    phx_a64_ldr(pb,
        a64::x0,
        arch::ptr_resolve(as, arch::fp, value_out.loc, arch::reg_scratch_0));
  }

  // Arbitrary scratch register for use in emitStoreGenYieldPoint()
  auto scratch_r = arch::reg_scratch_0;
  auto resume_label = Label(phx_builder_new_label(pb));
  emitStoreGenYieldPoint(as, env, instr, resume_label, arch::fp, scratch_r);

  // Jump to epilogue
  phx_a64_b(pb, env->exit_for_yield_label);

  // Resumed execution in this generator begins here
  phx_builder_bind(pb, resume_label);

  // Sent in value is in x1, and tstate is in x3 from resume entry-point args
  emitLoadResumedYieldInputs(as, instr, X1, a64::x3);
#else
  CINDER_UNSUPPORTED
#endif
}

void translateYieldFrom(Environ* env, const Instruction* instr) {
  arch::Builder* as = env->as;
  PhxBuilder* pb = as->impl();

#if defined(CINDER_X86_64)
  bool skip_initial_send = instr->isYieldFromSkipInitialSend();

  // Make sure tstate is in RDI for use in epilogue and here.
  PhyLocation tstate = instr->inputs_[0]->getStackSlot();
  auto tstate_phys_reg = x86::rdi;
  phx_x86_mov_rm(pb, tstate_phys_reg, phx_qword_ptr(x86::rbp, tstate.loc));

  // If we're skipping the initial send the send value is actually the first
  // value to yield and so needs to go into RAX to be returned. Otherwise,
  // put initial send value in RSI, the same location future send values will
  // be on resume.
  PhyLocation send_value = instr->inputs_[1]->getStackSlot();
  const auto send_value_phys_reg = skip_initial_send ? RAX : RSI;
  phx_x86_mov_rm(pb,
      x86::gpq(send_value_phys_reg.loc), phx_qword_ptr(x86::rbp, send_value.loc));

  asmjit::Label yield_label = Label(phx_builder_new_label(pb));
  if (skip_initial_send) {
    phx_x86_jmp_label(pb, yield_label);
  } else {
    // Setup call to JITRT_GenSend

    // Put tstate and the current generator into RCX and RDI respectively, and
    // set finish_yield_from (RDX) to 0. This register setup matches that when
    // `resume_label` is reached from the resume entry.
    auto gen_offs = offsetof(GenDataFooter, gen);
    phx_x86_mov_rr(pb, x86::rcx, tstate_phys_reg);
    phx_x86_mov_rm(pb, x86::rdi, phx_qword_ptr(x86::rbp, gen_offs));
    phx_x86_xor_rr(pb, x86::rdx, x86::rdx);
  }

  // Resumed execution begins here
  auto resume_label = Label(phx_builder_new_label(pb));
  phx_builder_bind(pb, resume_label);

  // Save tstate from resume to callee-saved reigster.
  phx_x86_mov_rr(pb, x86::rbx, x86::rcx);

  // 'send_value', and 'finish_yield_from' will already be in RSI and RCX
  // respectively, either from code above on initial start or from resume entry
  // point args.

  // Load sub-iterator into RDI
  PhyLocation iter_slot = instr->inputs_[2]->getStackSlot();
  phx_x86_mov_rm(pb, x86::rdi, phx_qword_ptr(x86::rbp, iter_slot.loc));

  uint64_t func = reinterpret_cast<uint64_t>(
      instr->isYieldFromHandleStopAsyncIteration()
          ? JITRT_GenSendHandleStopAsyncIteration
          : JITRT_GenSend);
  emitCall(*env, func, instr);
  // Yielded or final result value now in RAX. If the result was nullptr then
  // done will be set so we'll correctly jump to the following CheckExc.
  const auto yf_result_phys_reg = RAX;
  const auto done_r = x86::rdx;

  // Restore tstate from callee-saved register.
  phx_x86_mov_rr(pb, tstate_phys_reg, x86::rbx);

  // If not done, jump to epilogue which will yield/return the value from
  // JITRT_GenSend in RAX.
  phx_x86_test_rr(pb, done_r, done_r);
  asmjit::Label done_label = Label(phx_builder_new_label(pb));
  phx_x86_jnz(pb, done_label);

  phx_builder_bind(pb, yield_label);
  // Arbitrary scratch register for use in emitStoreGenYieldPoint()
  auto scratch_r = x86::r9;
  emitStoreGenYieldPoint(as, env, instr, resume_label, x86::rbp, scratch_r);
  phx_x86_jmp_label(pb, env->exit_for_yield_label);

  phx_builder_bind(pb, done_label);
  emitLoadResumedYieldInputs(as, instr, yf_result_phys_reg, tstate_phys_reg);
#elif defined(CINDER_AARCH64)
  bool skip_initial_send = instr->isYieldFromSkipInitialSend();

  // Make sure tstate is in X0 for use in epilogue and here.
  PhyLocation tstate = instr->inputs_[0]->getStackSlot();
  auto tstate_phys_reg = a64::x0;
  phx_a64_ldr(pb,
      tstate_phys_reg,
      arch::ptr_resolve(as, arch::fp, tstate.loc, arch::reg_scratch_0));

  // If we're skipping the initial send the send value is actually the first
  // value to yield and so needs to go into X0 to be returned. Otherwise,
  // put initial send value in X1, the same location future send values will
  // be on resume.
  PhyLocation send_value = instr->inputs_[1]->getStackSlot();
  const auto send_value_phys_reg = skip_initial_send ? X0 : X1;
  phx_a64_ldr(pb,
      a64::x(send_value_phys_reg.loc),
      arch::ptr_resolve(as, arch::fp, send_value.loc, arch::reg_scratch_0));

  asmjit::Label yield_label = Label(phx_builder_new_label(pb));
  if (skip_initial_send) {
    phx_a64_b(pb, yield_label);
  } else {
    // Setup call to JITRT_GenSend

    // Put tstate and the current generator into X3 and X0 respectively, and
    // set finish_yield_from (X2) to 0. This register setup matches that when
    // `resume_label` is reached from the resume entry.
    auto gen_offs = offsetof(GenDataFooter, gen);
    phx_a64_mov_rr(pb, a64::x3, tstate_phys_reg);
    phx_a64_ldr(pb, a64::x0, arch::ptr_offset(arch::fp, gen_offs));
    phx_a64_mov_rr(pb, a64::x2, a64::xzr);
  }

  // Resumed execution begins here
  auto resume_label = Label(phx_builder_new_label(pb));
  phx_builder_bind(pb, resume_label);

  // Save tstate from resume to callee-saved reigster.
  phx_a64_mov_rr(pb, a64::x19, a64::x3);

  // 'send_value', and 'finish_yield_from' will already be in X1 and X3
  // respectively, either from code above on initial start or from resume entry
  // point args.

  // Load sub-iterator into X0
  PhyLocation iter_slot = instr->inputs_[2]->getStackSlot();
  phx_a64_ldr(pb,
      a64::x0,
      arch::ptr_resolve(as, arch::fp, iter_slot.loc, arch::reg_scratch_0));

  uint64_t func = reinterpret_cast<uint64_t>(
      instr->isYieldFromHandleStopAsyncIteration()
          ? JITRT_GenSendHandleStopAsyncIteration
          : JITRT_GenSend);
  emitCall(*env, func, instr);
  // Yielded or final result value now in X0. If the result was nullptr then
  // done will be set so we'll correctly jump to the following CheckExc.
  const auto yf_result_phys_reg = X0;
  const auto done_r = a64::x2;

  // Restore tstate from callee-saved register.
  phx_a64_mov_rr(pb, tstate_phys_reg, a64::x19);

  // If not done, jump to epilogue which will yield/return the value from
  // JITRT_GenSend in X0.
  asmjit::Label done_label = Label(phx_builder_new_label(pb));
  phx_a64_cbnz(pb, done_r, done_label);

  phx_builder_bind(pb, yield_label);
  // Arbitrary scratch register for use in emitStoreGenYieldPoint()
  auto scratch_r = arch::reg_scratch_0;
  emitStoreGenYieldPoint(as, env, instr, resume_label, arch::fp, scratch_r);
  phx_a64_b(pb, env->exit_for_yield_label);

  phx_builder_bind(pb, done_label);
  emitLoadResumedYieldInputs(as, instr, yf_result_phys_reg, tstate_phys_reg);
#else
  CINDER_UNSUPPORTED
#endif
}

// ***********************************************************************
// The following templates and macros implement the auto generation table.
// The generator table defines a hash table, whose key is instruction type,
// and value is another hash table mapping instruction operand pattern and
// a function carrying out certain Actions for the instruction with the
// operand pattern.
// The list of Actions are encoded in the template class RuleActions as its
// template arguments. Currently, there are two types of Actions:
//   * AsmAction - generate an asm instruction
//   * CallAction - call a user defined instruction
// The Action classes are also templates, whose argument lists encode the
// parameters for the Action. For example, an AsmAction's argument list has
// the assembly instruction mnemonic and its operands.
// ***********************************************************************
template <int N>
const OperandBase* LIROperandMapper(const Instruction* instr) {
  auto num_outputs = instr->getNumOutputs();
  if (N < num_outputs) {
    return (&instr->output_);
  } else {
    return instr->inputs_[N - num_outputs];
  }
}

template <int N>
int LIROperandSizeMapper(const Instruction* instr) {
  auto size_type = InstrProperty::getProperties(instr).opnd_size_type;
  switch (size_type) {
    case kDefault:
      return LIROperandMapper<N>(instr)->sizeInBits();
    case kAlways64:
      return 64;
    case kOut:
      return LIROperandMapper<0>(instr)->sizeInBits();
  }

  JIT_ABORT("Unknown size type");
}

template <int N>
struct ImmOperand {
  using asmjit_type = const asmjit::Imm&;

  static asmjit::Imm GetAsmOperand(Environ*, const Instruction* instr) {
    return asmjit::Imm(LIROperandMapper<N>(instr)->getConstant());
  }
};

template <typename T>
struct ImmOperandNegate {
  using asmjit_type = const asmjit::Imm&;

  static asmjit::Imm GetAsmOperand(Environ* env, const Instruction* instr) {
    return asmjit::Imm(
        -T::GetAsmOperand(env, instr).template valueAs<int64_t>());
  }
};

template <typename T>
struct ImmOperandInvert {
  using asmjit_type = const asmjit::Imm&;

  static asmjit::Imm GetAsmOperand(Environ* env, const Instruction* instr) {
    return asmjit::Imm(
        ~T::GetAsmOperand(env, instr).template valueAs<uint64_t>());
  }
};

template <int N, int Size = -1>
struct RegOperand {
  using asmjit_type = const arch::Gp&;
  static arch::Gp GetAsmOperand(Environ*, const Instruction* instr) {
    static_assert(
        Size == -1 || Size == 8 || Size == 16 || Size == 32 || Size == 64,
        "Invalid Size");

#if defined(CINDER_X86_64)
    int size = Size == -1 ? LIROperandSizeMapper<N>(instr) : Size;

    PhyLocation reg = LIROperandMapper<N>(instr)->getPhyRegister();
    switch (size) {
      case 8:
        return asmjit::x86::gpb(reg.loc);
      case 16:
        return asmjit::x86::gpw(reg.loc);
      case 32:
        return asmjit::x86::gpd(reg.loc);
      case 64:
        return asmjit::x86::gpq(reg.loc);
    }
#elif defined(CINDER_AARCH64)
    int size = Size == -1 ? LIROperandSizeMapper<N>(instr) : Size;

    PhyLocation reg = LIROperandMapper<N>(instr)->getPhyRegister();
    switch (size) {
      case 8:
      case 16:
        JIT_ABORT("Currently unsupported size.");
      case 32:
        return asmjit::a64::w(reg.loc);
      case 64:
        return asmjit::a64::x(reg.loc);
    }
#else
    CINDER_UNSUPPORTED
#endif

    JIT_ABORT("Incorrect operand size.");
  }
};

template <int N>
struct VecDOperand {
  using asmjit_type = const arch::VecD&;
  static arch::VecD GetAsmOperand(Environ*, const Instruction* instr) {
#if defined(CINDER_X86_64)
    return asmjit::x86::xmm(
        LIROperandMapper<N>(instr)->getPhyRegister().loc - VECD_REG_BASE);
#elif defined(CINDER_AARCH64)
    return asmjit::a64::d(
        LIROperandMapper<N>(instr)->getPhyRegister().loc - VECD_REG_BASE);
#else
    CINDER_UNSUPPORTED
    return arch::VecD();
#endif
  }
};

#define OP(v)                                       \
  typename std::conditional_t<                      \
      pattern[v] == 'i',                            \
      ImmOperand<v>,                                \
      std::conditional_t<                           \
          (pattern[v] == 'x' || pattern[v] == 'X'), \
          VecDOperand<v>,                           \
          RegOperand<v>>>

#define REG_OP(v, size) RegOperand<v, size>

arch::Mem AsmIndirectOperandBuilder(const OperandBase* operand) {
  JIT_DCHECK(operand->isInd(), "operand should be an indirect reference");

#if defined(CINDER_X86_64)
  auto indirect = operand->getMemoryIndirect();

  OperandBase* base = indirect->getBaseRegOperand();
  OperandBase* index = indirect->getIndexRegOperand();

  if (index == nullptr) {
    return asmjit::x86::ptr(
        x86::gpq(base->getPhyRegister().loc), indirect->getOffset());
  } else {
    return asmjit::x86::ptr(
        x86::gpq(base->getPhyRegister().loc),
        x86::gpq(index->getPhyRegister().loc),
        indirect->getMultipiler(),
        indirect->getOffset());
  }
#elif defined(CINDER_AARCH64)
  JIT_ABORT("Unreachable.");
#else
  CINDER_UNSUPPORTED
  return arch::Mem();
#endif
}

template <int N>
struct MemOperand {
  using asmjit_type = const arch::Mem&;
  static arch::Mem GetAsmOperand(Environ*, const Instruction* instr) {
#if defined(CINDER_X86_64)
    const OperandBase* operand = LIROperandMapper<N>(instr);
    auto size = LIROperandSizeMapper<N>(instr) / 8;

    asmjit::x86::Mem memptr;
    if (operand->isStack()) {
      memptr = asmjit::x86::ptr(asmjit::x86::rbp, operand->getStackSlot().loc);
    } else if (operand->isMem()) {
      memptr = asmjit::x86::ptr(
          reinterpret_cast<uint64_t>(operand->getMemoryAddress()));
    } else if (operand->isInd()) {
      memptr = AsmIndirectOperandBuilder(operand);
    } else {
      JIT_ABORT("Unsupported operand type.");
    }

    memptr.setSize(size);
    return memptr;
#elif defined(CINDER_AARCH64)
    const OperandBase* operand = LIROperandMapper<N>(instr);
    if (!operand->isStack()) {
      JIT_ABORT("Unreachable.");
    }

    int32_t loc = operand->getStackSlot().loc;
    JIT_CHECK(loc >= -256 && loc < 256, "Stack slot out of range");

    return arch::ptr_offset(arch::fp, loc);
#else
    CINDER_UNSUPPORTED
    return arch::Mem();
#endif
  }
};

#define MEM(m) MemOperand<m>
#define STK(v) MemOperand<v>

template <int N>
struct LabelOperand {
  using asmjit_type = const asmjit::Label&;
  static asmjit::Label GetAsmOperand(Environ* env, const Instruction* instr) {
    auto block = LIROperandMapper<N>(instr)->getBasicBlock();
    return map_get(env->block_label_map, block);
  }
};

#define LBL(v) LabelOperand<v>

template <typename... Args>
struct OperandList;

template <typename FuncType, FuncType func, typename OpndList>
struct AsmAction;

template <typename FuncType, FuncType func, typename... OpndTypes>
struct AsmAction<FuncType, func, OperandList<OpndTypes...>> {
  static void eval(Environ* env, const Instruction* instr) {
    static_cast<void>(instr);
    (env->as->*func)(OpndTypes::GetAsmOperand(env, instr)...);
  }
};

template <typename... Args>
struct AsminstructionType {
  using type = asmjit::Error (arch::EmitterExplicitT<arch::Builder>::*)(
      typename Args::asmjit_type...);
};

template <void (*func)(Environ*, const Instruction*)>
struct CallAction {
  static void eval(Environ* env, const Instruction* instr) {
    func(env, instr);
  }
};

template <typename... Actions>
struct RuleActions;

template <typename AAction, typename... Actions>
struct RuleActions<AAction, Actions...> {
  static void eval(Environ* env, const Instruction* instr) {
    AAction::eval(env, instr);
    RuleActions<Actions...>::eval(env, instr);
  }
};

template <>
struct RuleActions<> {
  static void eval(Environ*, const Instruction*) {}
};

struct AddDebugEntryAction {
  static void eval(Environ* env, const Instruction* instr) {
    PhxBuilder* pb = env->as->impl();
    asmjit::Label label = Label(phx_builder_new_label(pb));
    phx_builder_bind(pb, label);
    if (instr->origin_) {
      env->pending_debug_locs.emplace_back(label, instr->origin_);
    }
  }
};

} // namespace

#define ASM(instr, args...)                    \
  AsmAction<                                   \
      typename AsminstructionType<args>::type, \
      &arch::Builder::instr,                   \
      OperandList<args>>

// Can't be named CALL as that conflicts with the opcode.
#define CALL_C(func) CallAction<func>

#define ADDDEBUGENTRY() AddDebugEntryAction

#define BEGIN_RULE_TABLE void AutoTranslator::initTable() {
#define END_RULE_TABLE }

#define BEGIN_RULES(__t)                                \
  {                                                     \
    auto& __rules = instr_rule_map_                     \
                        .emplace(                       \
                            std::piecewise_construct,   \
                            std::forward_as_tuple(__t), \
                            std::forward_as_tuple())    \
                        .first->second;

#define END_RULES }
#define GEN(s, actions...)                                  \
  {                                                         \
    UNUSED constexpr char pattern[] = s;                    \
    using rule_actions = RuleActions<actions>;              \
    auto gen = [](Environ* env, const Instruction* instr) { \
      rule_actions::eval(env, instr);                       \
    };                                                      \
    __rules = addPattern(std::move(__rules), s, gen);       \
  }

// ***********************************************************************
// Definition of Auto Generation Table
// The table consisting of multiple rules, and the rules for the same LIR
// instruction are grouped by BEGIN_RULES(LIR instruction type) and
// END_RULES.
// GEN defines a rule for a certain operand pattern of the LIR instruction,
// and maps it to a list of actions:
//   GEN(<operand pattern>, action1, action2, ...)
//
// The operand pattern is defined by a string, and each character in the string
// correpsonds to an operand of the instruction. The character can be one
// of the following:
//   * 'R' - general purpose register operand output
//   * 'r' - general purpose register operand input
//   * 'X' - floating-point register operand output
//   * 'x' - floating-point register operand input
//   * 'i' - immediate operand input
//   * 'M' - memory stack operand output
//   * 'm' - memory stack operand input
// Wildcards "?" and "*" can also be used in patterns, where "?" represents any
// one of the types listed above and "*" represents one or more above types.
// Please note that while "?" can appear anywhere in a pattern, "*" can only be
// used at the end of a pattern.
// The actions can be ASM and CALL_C, meaning generating an assembly instruction
// and call a user-defined function, respectively. The first argument of ASM
// action is the mnemonic of the instruction to be generated, and the following
// arguments are the operands to the instruction. Currently, we have four types
// of assembly instruction operands:
//   * OP  - either an immediate operand or register oeprand
//   * STK - a memory stack location [RBP - ?]
//   * LBL - a label to a basic block
//   * MEM - a memory operand. The size of the memory operand will be set to the
//           size of the LIR instruction operand specified by the first argument
//           of MEM.
// The assembly instruction operands are constructed from one or more LIR
// instruction operands. To specify the LIR operands, we use indices
// of the pattern string. For example:
//   GEN("Rri", ASM(mov, OP(0), MEM(0, 1, 2)))
// means generating a mov instruction, whose first operand is a
// register/immediate operand, constructed from the only output of the LIR
// instruction, and the second operand is memory operand, constructed from the
// register input and the immediate input of the LIR instruction. The size of
// the memory operand is set to the size of the output of the LIR instruction.
// ***********************************************************************

#if defined(CINDER_X86_64)
// clang-format off
BEGIN_RULE_TABLE

BEGIN_RULES(Instruction::kLea)
  GEN("Rm", ASM(lea, OP(0), MEM(1)))
END_RULES

BEGIN_RULES(Instruction::kCall)
  GEN("Ri", ASM(call, OP(1)), ADDDEBUGENTRY())
  GEN("Rr", ASM(call, OP(1)), ADDDEBUGENTRY())
  GEN("i", ASM(call, OP(0)), ADDDEBUGENTRY())
  GEN("r", ASM(call, OP(0)), ADDDEBUGENTRY())
  GEN("m", ASM(call, STK(0)), ADDDEBUGENTRY())
END_RULES

BEGIN_RULES(Instruction::kMove)
  GEN("Rr", ASM(mov, OP(0), OP(1)))
  GEN("Ri", ASM(mov, OP(0), OP(1)))
  GEN("Rm", ASM(mov, OP(0), MEM(1)))
  GEN("Mr", ASM(mov, MEM(0), OP(1)))
  GEN("Mi", ASM(mov, MEM(0), OP(1)))
  GEN("Xx", ASM(movsd, OP(0), OP(1)))
  GEN("Xm", ASM(movsd, OP(0), MEM(1)))
  GEN("Mx", ASM(movsd, MEM(0), OP(1)))
  GEN("Xr", ASM(movq, OP(0), OP(1)))
  GEN("Rx", ASM(movq, OP(0), OP(1)))
END_RULES

// Atomic move with relaxed ordering.
// On x86-64, relaxed loads/stores are plain mov.
// This corresponds to the C++/C memory_order_relaxed.
BEGIN_RULES(Instruction::kMoveRelaxed)
  GEN("Rm", ASM(mov, OP(0), MEM(1)))
  GEN("Mr", ASM(mov, MEM(0), OP(1)))
  GEN("Mi", ASM(mov, MEM(0), OP(1)))
END_RULES


// Guard + DeoptPatchpoint handled by C dispatch — rules deleted.

BEGIN_RULES(Instruction::kNegate)
  GEN("r", ASM(neg, OP(0)))
  GEN("Ri", ASM(mov, OP(0), ImmOperandNegate<OP(1)>))
  GEN("Rr", ASM(mov, OP(0), OP(1)), ASM(neg, OP(0)))
  GEN("Rm", ASM(mov, OP(0), STK(1)), ASM(neg, OP(0)))
END_RULES

BEGIN_RULES(Instruction::kInvert)
  GEN("Ri", ASM(mov, OP(0), ImmOperandInvert<OP(1)>))
  GEN("Rr", ASM(mov, OP(0), OP(1)), ASM(not_, OP(0)))
  GEN("Rm", ASM(mov, OP(0), STK(1)), ASM(not_, OP(0)))
END_RULES

BEGIN_RULES(Instruction::kMovZX)
  GEN("Rr", ASM(movzx, OP(0), OP(1)))
  GEN("Rm", ASM(movzx, OP(0), STK(1)))
END_RULES

BEGIN_RULES(Instruction::kMovSX)
  GEN("Rr", ASM(movsx, OP(0), OP(1)))
  GEN("Rm", ASM(movsx, OP(0), STK(1)))
END_RULES

BEGIN_RULES(Instruction::kMovSXD)
  GEN("Rr", ASM(movsxd, OP(0), OP(1)))
  GEN("Rm", ASM(movsxd, OP(0), STK(1)))
END_RULES

BEGIN_RULES(Instruction::kUnreachable)
  GEN(ANY, ASM(ud2))
END_RULES

#define DEF_BINARY_OP_RULES(name, instr) \
  BEGIN_RULES(Instruction::name) \
    GEN("ri", ASM(instr, OP(0), OP(1))) \
    GEN("rr", ASM(instr, OP(0), OP(1))) \
    GEN("rm", ASM(instr, OP(0), STK(1))) \
    /* rewriteBinaryOpInstrs() makes it safe to write the output before reading
     * all inputs without inputs_live_across being set for most binary ops; see
     * postalloc.cpp for details. */ \
    GEN("Rri", ASM(mov, OP(0), OP(1)), ASM(instr, OP(0), OP(2))) \
    GEN("Rrr", ASM(mov, OP(0), OP(1)), ASM(instr, OP(0), OP(2))) \
    GEN("Rrm", ASM(mov, OP(0), OP(1)), ASM(instr, OP(0), STK(2))) \
  END_RULES

DEF_BINARY_OP_RULES(kAdd, add)
DEF_BINARY_OP_RULES(kSub, sub)
DEF_BINARY_OP_RULES(kAnd, and_)
DEF_BINARY_OP_RULES(kOr, or_)
DEF_BINARY_OP_RULES(kXor, xor_)
DEF_BINARY_OP_RULES(kMul, imul)

BEGIN_RULES(Instruction::kDiv)
  GEN("rrr", ASM(idiv, OP(0), OP(1), OP(2)) )
  GEN("rrm", ASM(idiv, OP(0), OP(1), STK(2)) )
  GEN("rr", ASM(idiv, OP(0), OP(1)) )
  GEN("rm", ASM(idiv, OP(0), STK(1)) )
END_RULES

BEGIN_RULES(Instruction::kDivUn)
  GEN("rrr", ASM(div, OP(0), OP(1), OP(2)) )
  GEN("rrm", ASM(div, OP(0), OP(1), STK(2)) )
  GEN("rr", ASM(div, OP(0), OP(1)) )
  GEN("rm", ASM(div, OP(0), STK(1)) )
END_RULES

#undef DEF_BINARY_OP_RULES

BEGIN_RULES(Instruction::kFadd)
  /* rewriteBinaryOpInstrs() makes it safe to write the output before reading
   * all inputs without inputs_live_across being set for Fadd; see
   * postalloc.cpp for details. */
  GEN("Xxx", ASM(movsd, OP(0), OP(1)), ASM(addsd, OP(0), OP(2)))
  GEN("xx", ASM(addsd, OP(0), OP(1)))
END_RULES

BEGIN_RULES(Instruction::kFsub)
  GEN("Xxx", ASM(movsd, OP(0), OP(1)), ASM(subsd, OP(0), OP(2)))
  GEN("xx", ASM(subsd, OP(0), OP(1)))
END_RULES

BEGIN_RULES(Instruction::kFmul)
  /* rewriteBinaryOpInstrs() makes it safe to write the output before reading
   * all inputs without inputs_live_across being set for Fmul; see
   * postalloc.cpp for details. */
  GEN("Xxx", ASM(movsd, OP(0), OP(1)), ASM(mulsd, OP(0), OP(2)))
  GEN("xx", ASM(mulsd, OP(0), OP(1)))
END_RULES

BEGIN_RULES(Instruction::kFdiv)
  GEN("Xxx", ASM(movsd, OP(0), OP(1)), ASM(divsd, OP(0), OP(2)))
  GEN("xx", ASM(divsd, OP(0), OP(1)))
END_RULES

BEGIN_RULES(Instruction::kPush)
  GEN("r", ASM(push, OP(0)))
  GEN("m", ASM(push, STK(0)))
  GEN("i", ASM(push, OP(0)))
END_RULES

BEGIN_RULES(Instruction::kPop)
  GEN("R", ASM(pop, OP(0)))
  GEN("M", ASM(pop, STK(0)))
END_RULES

BEGIN_RULES(Instruction::kCdq)
  GEN("Rr", ASM(cdq, OP(0), OP(1)))
END_RULES

BEGIN_RULES(Instruction::kCwd)
  GEN("Rr", ASM(cwd, OP(0), OP(1)))
END_RULES

BEGIN_RULES(Instruction::kCqo)
  GEN("Rr", ASM(cqo, OP(0), OP(1)))
END_RULES

BEGIN_RULES(Instruction::kExchange)
  GEN("Rr", ASM(xchg, OP(0), OP(1)))
  GEN("Xx", ASM(pxor, OP(0), OP(1)),
            ASM(pxor, OP(1), OP(0)),
            ASM(pxor, OP(0), OP(1)))
END_RULES

BEGIN_RULES(Instruction::kCmp)
  GEN("rr", ASM(cmp, OP(0), OP(1)))
  GEN("ri", ASM(cmp, OP(0), OP(1)))
  GEN("xx", ASM(comisd, OP(0), OP(1)))
END_RULES

BEGIN_RULES(Instruction::kTest)
  GEN("rr", ASM(test, OP(0), OP(1)))
END_RULES

BEGIN_RULES(Instruction::kTest32)
  GEN("rr", ASM(test, REG_OP(0, 32), REG_OP(1, 32)))
END_RULES

BEGIN_RULES(Instruction::kBranch)
  GEN("b", ASM(jmp, LBL(0)))
END_RULES

BEGIN_RULES(Instruction::kBranchZ)
  GEN("b", ASM(jz, LBL(0)))
END_RULES

BEGIN_RULES(Instruction::kBranchNZ)
  GEN("b", ASM(jnz, LBL(0)))
END_RULES

BEGIN_RULES(Instruction::kBranchA)
  GEN("b", ASM(ja, LBL(0)))
END_RULES

BEGIN_RULES(Instruction::kBranchB)
  GEN("b", ASM(jb, LBL(0)))
END_RULES

BEGIN_RULES(Instruction::kBranchAE)
  GEN("b", ASM(jae, LBL(0)))
END_RULES

BEGIN_RULES(Instruction::kBranchBE)
  GEN("b", ASM(jbe, LBL(0)))
END_RULES

BEGIN_RULES(Instruction::kBranchG)
  GEN("b", ASM(jg, LBL(0)))
END_RULES

BEGIN_RULES(Instruction::kBranchL)
  GEN("b", ASM(jl, LBL(0)))
END_RULES

BEGIN_RULES(Instruction::kBranchGE)
  GEN("b", ASM(jge, LBL(0)))
END_RULES

BEGIN_RULES(Instruction::kBranchLE)
  GEN("b", ASM(jle, LBL(0)))
END_RULES

BEGIN_RULES(Instruction::kBranchC)
  GEN("b", ASM(jc, LBL(0)))
END_RULES

BEGIN_RULES(Instruction::kBranchNC)
  GEN("b", ASM(jnc, LBL(0)))
END_RULES

BEGIN_RULES(Instruction::kBranchO)
  GEN("b", ASM(jo, LBL(0)))
END_RULES

BEGIN_RULES(Instruction::kBranchNO)
  GEN("b", ASM(jno, LBL(0)))
END_RULES

BEGIN_RULES(Instruction::kBranchS)
  GEN("b", ASM(js, LBL(0)))
END_RULES

BEGIN_RULES(Instruction::kBranchNS)
  GEN("b", ASM(jns, LBL(0)))
END_RULES

BEGIN_RULES(Instruction::kBranchE)
  GEN("b", ASM(je, LBL(0)))
END_RULES

BEGIN_RULES(Instruction::kBranchNE)
  GEN("b", ASM(jne, LBL(0)))
END_RULES

// Compare ops handled by C dispatch — rules deleted.

BEGIN_RULES(Instruction::kInc)
  GEN("r", ASM(inc, OP(0)))
  GEN("m", ASM(inc, STK(0)))
END_RULES

BEGIN_RULES(Instruction::kDec)
  GEN("r", ASM(dec, OP(0)))
  GEN("m", ASM(dec, STK(0)))
END_RULES

BEGIN_RULES(Instruction::kBitTest)
  GEN("ri", ASM(bt, OP(0), OP(1)))
END_RULES

BEGIN_RULES(Instruction::kYieldInitial)
  GEN(ANY, CALL_C(translateYieldInitial))
END_RULES

#if PY_VERSION_HEX < 0x030C0000
BEGIN_RULES(Instruction::kYieldFrom)
  GEN(ANY, CALL_C(translateYieldFrom))
END_RULES
#else
// In 3.12+ YieldFrom is a pseudo-op which is YieldValue plus enough
// information to know which live value contains the target iterator. See
// emitStoreGenYieldPoint() for where this is captured. The target iterator is
// used for things like the result of reading gi_yieldfrom.
BEGIN_RULES(Instruction::kYieldFrom)
  GEN(ANY, CALL_C(translateYieldValue))
END_RULES
#endif

BEGIN_RULES(Instruction::kYieldFromSkipInitialSend)
  GEN(ANY, CALL_C(translateYieldFrom))
END_RULES

BEGIN_RULES(Instruction::kYieldFromHandleStopAsyncIteration)
  GEN(ANY, CALL_C(translateYieldFrom))
END_RULES

BEGIN_RULES(Instruction::kYieldValue)
  GEN(ANY, CALL_C(translateYieldValue))
END_RULES

BEGIN_RULES(Instruction::kSelect)
  GEN("Rrri", ASM(mov, OP(0), OP(3)),
              ASM(test, OP(1), OP(1)),
              ASM(cmovnz, OP(0), OP(2)))
END_RULES

// IntToBool handled by C dispatch — rules deleted.

END_RULE_TABLE
// clang-format on
#elif defined(CINDER_AARCH64)

// ARM64 translate* functions have been moved to autogen_translate_c.c (pure C).
// Only yield rule table entries remain here (yield functions are cross-arch,
// defined above with #if CINDER_X86_64 / #elif CINDER_AARCH64 branches).

namespace {
using AT = AutoTranslator;

// Dead ARM64 translate* functions deleted — replaced by autogen_translate_c.c
// Dead ARM64 rule table entries deleted — C dispatch handles all non-yield opcodes

// clang-format off
BEGIN_RULE_TABLE

BEGIN_RULES(Instruction::kYieldInitial)
  GEN(ANY, CALL_C(translateYieldInitial))
END_RULES

#if PY_VERSION_HEX < 0x030C0000
BEGIN_RULES(Instruction::kYieldFrom)
  GEN(ANY, CALL_C(translateYieldFrom))
END_RULES
#else
// In 3.12+ YieldFrom is a pseudo-op which is YieldValue plus enough
// information to know which live value contains the target iterator. See
// emitStoreGenYieldPoint() for where this is captured. The target iterator is
// used for things like the result of reading gi_yieldfrom.
BEGIN_RULES(Instruction::kYieldFrom)
  GEN(ANY, CALL_C(translateYieldValue))
END_RULES
#endif

BEGIN_RULES(Instruction::kYieldFromSkipInitialSend)
  GEN(ANY, CALL_C(translateYieldFrom))
END_RULES

BEGIN_RULES(Instruction::kYieldFromHandleStopAsyncIteration)
  GEN(ANY, CALL_C(translateYieldFrom))
END_RULES

BEGIN_RULES(Instruction::kYieldValue)
  GEN(ANY, CALL_C(translateYieldValue))
END_RULES

// Select and IntToBool handled by C dispatch — rules deleted.

END_RULE_TABLE
// clang-format on
#else

BEGIN_RULE_TABLE
END_RULE_TABLE

#endif

} // namespace jit::codegen::autogen
