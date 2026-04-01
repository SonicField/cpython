// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/Jit/codegen/arch.h"
#include "cinderx/Jit/codegen/register_preserver_c.h"

#include <vector>

namespace jit::codegen {

// C++ wrapper around the C PhxRegPreserver API.
// Converts vector<pair<Reg&, Reg&>> to PhxRegPair array, then forwards
// preserve/remap/restore to the C implementation in register_preserver.c.
class RegisterPreserver {
 public:
  static constexpr int kMaxRegs = 32;

  RegisterPreserver(
      arch::Builder* as,
      const std::vector<std::pair<const arch::Reg&, const arch::Reg&>>&
          save_regs)
      : num_regs_(static_cast<int>(save_regs.size())) {
    for (int i = 0; i < num_regs_ && i < kMaxRegs; i++) {
      pairs_[i].src = save_regs[i].first;
      pairs_[i].dst = save_regs[i].second;
    }
    phx_reg_preserver_init(&rp_, as->impl(), pairs_, num_regs_);
  }

  void preserve() {
    phx_reg_preserver_preserve(&rp_);
  }

  void restore() {
    phx_reg_preserver_restore(&rp_);
  }

  void remap() {
    phx_reg_preserver_remap(&rp_);
  }

 private:
  PhxRegPair pairs_[kMaxRegs];
  PhxRegPreserver rp_;
  int num_regs_;
};

} // namespace jit::codegen
