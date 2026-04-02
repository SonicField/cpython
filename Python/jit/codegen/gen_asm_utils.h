// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/python.h"

#include "cinderx/Jit/debug_info.h"
#include "cinderx/Jit/gen_data_footer.h"
#include "cinderx/Jit/lir/instruction.h"

#ifdef PHOENIX_ASM
#include "jit/phoenix_asm/asmjit_compat.h"
#else
#include <asmjit/asmjit.h>
#endif

#ifdef PHOENIX_ASM
/* ---- C API (implemented in gen_asm_utils.c) ---- */
#include "cinderx/Jit/codegen/gen_asm_utils_c.h"
#endif

#include "cinderx/Jit/codegen/environ.h"

namespace jit::codegen {

#ifdef PHOENIX_ASM

// Inline C++ wrappers that bridge Environ to the C PhxEmitCallCtx API.
// The C implementation handles instruction emission; the wrapper handles
// the C++ debug_locs bookkeeping.

inline void emitCall(
    Environ& env,
    asmjit::Label label,
    const jit::lir::Instruction* instr) {
  PhxEmitCallCtx ctx;
  ctx.builder = env.as->impl();
#if defined(CINDER_AARCH64)
  ctx.is_generator = env.is_generator ? 1 : 0;
  ctx.saved_ip_fp_offset = env.saved_ip_fp_offset;
  ctx.gen_footer_saved_ip_offset =
      static_cast<int>(offsetof(jit::GenDataFooter, savedIP));
#else
  ctx.is_generator = 0;
  ctx.saved_ip_fp_offset = 0;
  ctx.gen_footer_saved_ip_offset = 0;
#endif

  PhxLabel debug_label;
  phx_emit_call_label(
      &ctx, label, instr->origin() != nullptr ? 1 : 0, &debug_label);

  if (debug_label.id != UINT32_MAX && instr->origin() != nullptr) {
    env.pending_debug_locs.emplace_back(
        asmjit::Label(debug_label), instr->origin());
  }
}

inline void emitCall(
    Environ& env,
    uint64_t func,
    const jit::lir::Instruction* instr) {
  PhxEmitCallCtx ctx;
  ctx.builder = env.as->impl();
#if defined(CINDER_AARCH64)
  ctx.is_generator = env.is_generator ? 1 : 0;
  ctx.saved_ip_fp_offset = env.saved_ip_fp_offset;
  ctx.gen_footer_saved_ip_offset =
      static_cast<int>(offsetof(jit::GenDataFooter, savedIP));
#else
  ctx.is_generator = 0;
  ctx.saved_ip_fp_offset = 0;
  ctx.gen_footer_saved_ip_offset = 0;
#endif

  PhxLabel debug_label;
  phx_emit_call_func(
      &ctx, func, instr->origin() != nullptr ? 1 : 0, &debug_label);

  if (debug_label.id != UINT32_MAX && instr->origin() != nullptr) {
    env.pending_debug_locs.emplace_back(
        asmjit::Label(debug_label), instr->origin());
  }
}

#else /* !PHOENIX_ASM -- raw asmjit path (ARM64 without phoenix-asm) */

namespace {
inline void recordDebugEntry(Environ& env, const jit::lir::Instruction* instr) {
  if (instr->origin() == nullptr) {
    return;
  }
  asmjit::Label addr = env.as->newLabel();
  env.as->bind(addr);
  env.pending_debug_locs.emplace_back(addr, instr->origin());
}
} // namespace

inline void emitCall(
    Environ& env,
    asmjit::Label label,
    const jit::lir::Instruction* instr) {
#if defined(CINDER_X86_64)
  env.as->call(label);
#elif defined(CINDER_AARCH64)
  {
    asmjit::Label after_call = env.as->newLabel();
    env.as->adr(arch::reg_scratch_0, after_call);
    if (env.is_generator) {
      env.as->str(arch::reg_scratch_0,
                  asmjit::arm::Mem(asmjit::a64::x29,
                                   offsetof(jit::GenDataFooter, savedIP)));
    } else {
      env.as->str(
          arch::reg_scratch_0,
          arch::ptr_resolve(
              env.as, arch::fp, env.saved_ip_fp_offset, arch::reg_scratch_1));
    }
    env.as->bl(label);
    env.as->bind(after_call);
  }
#endif
  recordDebugEntry(env, instr);
}

inline void emitCall(
    Environ& env,
    uint64_t func,
    const jit::lir::Instruction* instr) {
#if defined(CINDER_X86_64)
  env.as->call(func);
#elif defined(CINDER_AARCH64)
  env.as->mov(arch::reg_scratch_br, func);
  {
    asmjit::Label after_call = env.as->newLabel();
    env.as->adr(arch::reg_scratch_0, after_call);
    if (env.is_generator) {
      env.as->str(arch::reg_scratch_0,
                  asmjit::arm::Mem(asmjit::a64::x29,
                                   offsetof(jit::GenDataFooter, savedIP)));
    } else {
      env.as->str(
          arch::reg_scratch_0,
          arch::ptr_resolve(
              env.as, arch::fp, env.saved_ip_fp_offset, arch::reg_scratch_1));
    }
    env.as->blr(arch::reg_scratch_br);
    env.as->bind(after_call);
  }
#endif
  recordDebugEntry(env, instr);
}

#endif /* PHOENIX_ASM */

} // namespace jit::codegen
