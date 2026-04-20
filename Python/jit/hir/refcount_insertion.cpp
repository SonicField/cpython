// Copyright (c) Meta Platforms, Inc. and affiliates.

extern "C" void hir_cfg_split_critical_edges_c(void *func);

#include "cinderx/Jit/hir/refcount_insertion.h"
#include "cinderx/Jit/hir/refcount_env_c.h"
#include "cinderx/Jit/hir/refcount_pass_c.h"
#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/hir/hir_type_c.h"
#include "cinderx/Jit/jit_config_c.h"

#include "cinderx/Common/log.h"
#include "cinderx/Jit/hir/dead_code_elimination.h"
#include "cinderx/Jit/hir/frame_state.h"
#include "cinderx/Jit/hir/phi_elimination.h"
#include "cinderx/Jit/hir/printer.h"

#include <fmt/ostream.h>

#include <vector>

// This file implements our reference count insertion pass. If this is your
// first time here, I recommend reading refcount_insertion.md first.

namespace jit::hir {

namespace {

// bindGuards and optimizeLongDecrefRuns remain as C++ post-processing.
// The refcount analysis+mutation pass is now in C (refcount_pass_c.c).
// necessary.
void bindGuards(Function& irfunc) {
  std::vector<Instr*> snapshots;
  for (auto& block : irfunc.cfg.blocks) {
    FrameState* fs = nullptr;
    for (auto& instr : block) {
      if (instr.IsSnapshot()) {
        auto& snapshot = static_cast<const Snapshot&>(instr);
        fs = snapshot.frameState();
        snapshots.emplace_back(&instr);
      } else if (
          instr.IsGuard() || instr.IsGuardIs() || instr.IsGuardType() ||
          instr.IsDeopt() || instr.IsDeoptPatchpoint()) {
        JIT_DCHECK(
            fs != nullptr,
            "No dominating snapshot for '{}' in function:\n{}",
            instr,
            irfunc);
        auto& guard = static_cast<DeoptBase&>(instr);
        guard.setFrameState(*fs);
      } else if (!instr.isReplayable()) {
        fs = nullptr;
      }
    }
  }
  for (auto& snapshot : snapshots) {
    snapshot->unlink();
    Instr::Destroy(snapshot);
  }
  DeadCodeElimination{}.Run(irfunc);
}

void optimizeLongDecrefRuns(Function& irfunc) {
  constexpr int kMinimumNumberOfDecrefsToOptimize = 4;

  auto get_number_of_decrefs = [](auto block, auto cur_iter) {
    int result = 0;
    while (cur_iter != block->end()) {
      if (!cur_iter->IsDecref()) {
        break;
      }
      result++;
      ++cur_iter;
    }
    return result;
  };

  for (auto& block : irfunc.cfg.GetRPOTraversal()) {
    auto cur_iter = block->begin();

    while (cur_iter != block->end()) {
      if (!cur_iter->IsDecref()) {
        ++cur_iter;
        continue;
      }

      int num = get_number_of_decrefs(block, cur_iter);
      if (num < kMinimumNumberOfDecrefsToOptimize) {
        std::advance(cur_iter, num);
        continue;
      }

      auto batch_decref = static_cast<Instr*>(hir_c_create_batch_decref(num));
      batch_decref->copyBytecodeOffset(*cur_iter);
      batch_decref->InsertBefore(*cur_iter);

      constexpr size_t kDecrefOperandIndex = 0;
      for (int i = 0; i < num; i++) {
        JIT_CHECK(
            cur_iter->IsDecref(),
            "An unexpected non-decref instruction in a decref run.");

        batch_decref->SetOperand(i, cur_iter->GetOperand(kDecrefOperandIndex));
        auto old_instr = cur_iter++;
        old_instr->unlink();
        Instr::Destroy(&(*old_instr));
      }
    }
  }
}

} // namespace

void RefcountInsertion::Run(Function& func) {
  PhiElimination{}.Run(func);
  bindGuards(func);
  hir_cfg_split_critical_edges_c(&func);

  PhxRefcountEnv *c_env = phx_rc_env_create(static_cast<void*>(&func));
  phx_rc_run(c_env);
  phx_rc_env_destroy(c_env);

  removeTrampolineBlocks(&func.cfg);
  optimizeLongDecrefRuns(func);
}

} // namespace jit::hir
