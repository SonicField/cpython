// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cinderx/Jit/codegen/gen_asm_utils.h"

#include "cinderx/Jit/codegen/arch.h"
#include "cinderx/Common/util.h"
#include "cinderx/Jit/codegen/environ.h"
#include "cinderx/Jit/gen_data_footer.h"

#include "jit/phoenix_asm/x86_64.h"
#include "jit/phoenix_asm/arm64.h"

namespace jit::codegen {

namespace {
void recordDebugEntry(Environ& env, const jit::lir::Instruction* instr) {
  if (instr->origin() == nullptr) {
    return;
  }
  asmjit::Label addr = env.as->newLabel();
  phx_builder_bind(env.as->impl(), addr);
  env.pending_debug_locs.emplace_back(addr, instr->origin());
}
} // namespace

void emitCall(
    Environ& env,
    asmjit::Label label,
    const jit::lir::Instruction* instr) {
#if defined(CINDER_X86_64)
  phx_x86_call_label(env.as->impl(), label);
#elif defined(CINDER_AARCH64)
  // Save return address to stack before bl, matching x86 call semantics.
  // Slot at [FP - (stack_frame_size - 8)] = [SP + 8], within the extra
  // kStackAlign bytes reserved by computeFrameInfo.
  {
    asmjit::Label after_call = env.as->newLabel();
    phx_a64_adr(env.as->impl(), arch::reg_scratch_0, after_call);
    if (env.is_generator) {
      phx_a64_str(env.as->impl(), arch::reg_scratch_0,
                  asmjit::arm::Mem(asmjit::a64::x29,
                                   offsetof(jit::GenDataFooter, savedIP)));
    } else {
      phx_a64_str(
          env.as->impl(),
          arch::reg_scratch_0,
          arch::ptr_resolve(
              env.as, arch::fp, env.saved_ip_fp_offset, arch::reg_scratch_1));
    }
    phx_a64_bl(env.as->impl(), label);
    phx_builder_bind(env.as->impl(), after_call);
  }
#else
  CINDER_UNSUPPORTED
#endif
  recordDebugEntry(env, instr);
}

void emitCall(Environ& env, uint64_t func, const jit::lir::Instruction* instr) {
#if defined(CINDER_X86_64)
  phx_x86_mov_ri(env.as->impl(), PHX_R11, (int64_t)func);
  phx_x86_call_r(env.as->impl(), PHX_R11);
#elif defined(CINDER_AARCH64)
  // Note that we could do better than this if asmjit knew how to handle arm64
  // relocations for relative calls. That work is done in
  // https://github.com/asmjit/asmjit/issues/499, but as of writing is not yet
  // available.
  phx_a64_mov_ri(env.as->impl(), arch::reg_scratch_br, (uint64_t)(uintptr_t)func);
  // Save return address to stack before blr, matching x86 call semantics.
  {
    asmjit::Label after_call = env.as->newLabel();
    phx_a64_adr(env.as->impl(), arch::reg_scratch_0, after_call);
    if (env.is_generator) {
      phx_a64_str(env.as->impl(), arch::reg_scratch_0,
                  asmjit::arm::Mem(asmjit::a64::x29,
                                   offsetof(jit::GenDataFooter, savedIP)));
    } else {
      phx_a64_str(
          env.as->impl(),
          arch::reg_scratch_0,
          arch::ptr_resolve(
              env.as, arch::fp, env.saved_ip_fp_offset, arch::reg_scratch_1));
    }
    phx_a64_blr(env.as->impl(), arch::reg_scratch_br);
    phx_builder_bind(env.as->impl(), after_call);
  }
#else
  CINDER_UNSUPPORTED
#endif
  recordDebugEntry(env, instr);
}

} // namespace jit::codegen
