// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cinderx/Jit/codegen/register_preserver.h"

#include "jit/phoenix_asm/x86_64.h"
#include "jit/phoenix_asm/arm64.h"

namespace jit::codegen {

#if defined(CINDER_X86_64)
// Stack alignment requirement for x86-64
constexpr size_t kConstStackAlignmentRequirement = 16;
#elif defined(CINDER_AARCH64)
// Stack alignment requirement for aarch64
constexpr int32_t kConstStackAlignmentRequirement = 16;
#else
CINDER_UNSUPPORTED
#endif

RegisterPreserver::RegisterPreserver(
    arch::Builder* as,
    const std::vector<std::pair<const arch::Reg&, const arch::Reg&>>& save_regs)
    : as_(as), save_regs_(save_regs), align_stack_(false) {}

void RegisterPreserver::preserve() {
#if defined(CINDER_X86_64)
  size_t rsp_offset = 0;
  for (const auto& pair : save_regs_) {
    if (pair.first.isGpq()) {
      phx_x86_push_r(as_->impl(), pair.first);
    } else if (pair.first.isXmm()) {
      phx_x86_sub_ri(as_->impl(), PHX_RSP, pair.first.size());
      phx_x86_movdqu_mr(
          as_->impl(), phx_dqword_ptr(PHX_RSP, 0), pair.first);
    } else {
      JIT_ABORT("unsupported saved register type");
    }
    rsp_offset += pair.first.size();
  }
  align_stack_ = rsp_offset % kConstStackAlignmentRequirement;
  if (align_stack_) {
    phx_x86_push_r(as_->impl(), PHX_RAX);
  }
#elif defined(CINDER_AARCH64)
  for (const auto& [idx, kind] : registerGroups()) {
    switch (kind) {
      case RegisterGroup::kGpPair:
        phx_a64_stp_pre(
            as_->impl(),
            save_regs_[idx].first,
            save_regs_[idx + 1].first,
            PHX_SP, -kConstStackAlignmentRequirement);
        break;
      case RegisterGroup::kVecDPair:
        phx_a64_stp_pre(
            as_->impl(),
            save_regs_[idx].first,
            save_regs_[idx + 1].first,
            PHX_SP, -kConstStackAlignmentRequirement);
        break;
      case RegisterGroup::kGp:
        as_->str(
            static_cast<const asmjit::a64::Gp&>(save_regs_[idx].first),
            asmjit::a64::ptr_pre(
                asmjit::a64::sp, -kConstStackAlignmentRequirement));
        break;
      case RegisterGroup::kVecD:
        as_->str(
            static_cast<const asmjit::a64::VecD&>(save_regs_[idx].first),
            asmjit::a64::ptr_pre(
                asmjit::a64::sp, -kConstStackAlignmentRequirement));
        break;
    }
  }
#else
  CINDER_UNSUPPORTED
#endif
}

void RegisterPreserver::remap() {
#if defined(CINDER_X86_64)
  for (const auto& pair : save_regs_) {
    if (pair.first != pair.second) {
      if (pair.first.isGpq()) {
        JIT_DCHECK(pair.second.isGpq(), "can't mix and match register types");
        phx_x86_mov_rr(as_->impl(), pair.second, pair.first);
      } else if (pair.first.isXmm()) {
        JIT_DCHECK(pair.second.isXmm(), "can't mix and match register types");
        phx_x86_movsd_rr(as_->impl(), pair.second, pair.first);
      }
    }
  }
#elif defined(CINDER_AARCH64)
  for (const auto& pair : save_regs_) {
    if (pair.first != pair.second) {
      if (pair.first.isGpX()) {
        JIT_DCHECK(pair.second.isGpX(), "can't mix and match register types");
        phx_a64_mov_rr(as_->impl(), pair.second, pair.first);
      } else if (pair.first.isVecD()) {
        JIT_DCHECK(pair.second.isVecD(), "can't mix and match register types");
        phx_a64_fmov(as_->impl(), pair.second, pair.first);
      }
    }
  }
#else
  CINDER_UNSUPPORTED
#endif
}

void RegisterPreserver::restore() {
#if defined(CINDER_X86_64)
  if (align_stack_) {
    phx_x86_add_ri(as_->impl(), PHX_RSP, 8);
  }
  for (auto iter = save_regs_.rbegin(); iter != save_regs_.rend(); ++iter) {
    if (iter->second.isGpq()) {
      phx_x86_pop_r(as_->impl(), iter->second);
    } else if (iter->second.isXmm()) {
      phx_x86_movdqu_rm(
          as_->impl(), iter->second,
          phx_dqword_ptr(PHX_RSP, 0));
      phx_x86_add_ri(as_->impl(), PHX_RSP, 16);
    } else {
      JIT_ABORT("unsupported saved register type");
    }
  }
#elif defined(CINDER_AARCH64)
  auto groups = registerGroups();
  for (auto it = groups.rbegin(); it != groups.rend(); ++it) {
    auto [idx, kind] = *it;

    switch (kind) {
      case RegisterGroup::kGpPair:
        phx_a64_ldp_post(
            as_->impl(),
            save_regs_[idx].second,
            save_regs_[idx + 1].second,
            PHX_SP, kConstStackAlignmentRequirement);
        break;
      case RegisterGroup::kVecDPair:
        phx_a64_ldp_post(
            as_->impl(),
            save_regs_[idx].second,
            save_regs_[idx + 1].second,
            PHX_SP, kConstStackAlignmentRequirement);
        break;
      case RegisterGroup::kGp:
        as_->ldr(
            static_cast<const asmjit::a64::Gp&>(save_regs_[idx].second),
            asmjit::a64::ptr_post(
                asmjit::a64::sp, kConstStackAlignmentRequirement));
        break;
      case RegisterGroup::kVecD:
        as_->ldr(
            static_cast<const asmjit::a64::VecD&>(save_regs_[idx].second),
            asmjit::a64::ptr_post(
                asmjit::a64::sp, kConstStackAlignmentRequirement));
        break;
    }
  }
#else
  CINDER_UNSUPPORTED
#endif
}

#if defined(CINDER_AARCH64)
std::vector<std::pair<size_t, RegisterPreserver::RegisterGroup>>
RegisterPreserver::registerGroups() const {
  std::vector<std::pair<size_t, RegisterPreserver::RegisterGroup>> groups;
  for (size_t idx = 0; idx < save_regs_.size();) {
    if ((idx + 1 < save_regs_.size()) && save_regs_[idx].first.isGpX() &&
        save_regs_[idx + 1].first.isGpX()) {
      groups.emplace_back(idx, RegisterGroup::kGpPair);
      idx += 2;
    } else if (
        (idx + 1 < save_regs_.size()) && save_regs_[idx].first.isVecD() &&
        save_regs_[idx + 1].first.isVecD()) {
      groups.emplace_back(idx, RegisterGroup::kVecDPair);
      idx += 2;
    } else if (save_regs_[idx].first.isGpX()) {
      groups.emplace_back(idx, RegisterGroup::kGp);
      idx += 1;
    } else {
      groups.emplace_back(idx, RegisterGroup::kVecD);
      idx += 1;
    }
  }
  return groups;
}
#endif

} // namespace jit::codegen
