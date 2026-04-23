// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cinderx/Jit/hir/ssa.h"

#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/hir/ssaify_c.h"
#include "cinderx/Common/log.h"
#include "cinderx/Jit/hir/hir.h"
#include "cinderx/Jit/hir/phi_elimination.h"
#include "cinderx/Jit/hir/type.h"

#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

extern "C" void hir_reflow_types_c(void *func, void *start_block);

namespace jit::hir {


void SSAify::Run(Function& irfunc) {
  hir_ssaify_run_c(static_cast<void*>(&irfunc));
}

// This implements the algorithm outlined in "Simple and Efficient Construction
// of Static Single Assignment Form"
// https://pp.info.uni-karlsruhe.de/uploads/publikationen/braun13cc.pdf
void SSAify::Run(Function& irfunc, BasicBlock* start) {
  env_ = &irfunc.env;

  auto blocks = CFG::GetRPOTraversal(start);
  auto ssa_basic_blocks = initSSABasicBlocks(blocks);
  phi_uses_.clear();

  for (auto& block : blocks) {
    auto ssablock = ssa_basic_blocks.at(block);

    for (auto& instr : *block) {
      JIT_CHECK(!instr.IsPhi(), "SSAify does not support Phis in its input");
      instr.visitUses([&](Register*& reg) {
        JIT_CHECK(
            reg != nullptr, "Instructions should not have nullptr operands.");
        reg = getDefine(ssablock, reg);
        return true;
      });

      auto out_reg = instr.output();

      if (out_reg != nullptr) {
        auto new_reg = env_->AllocateRegister();
        instr.setOutput(new_reg);
        ssablock->local_defs[out_reg] = new_reg;
      }
    }

    for (auto& succ : ssablock->succs) {
      succ->unsealed_preds--;
      if (succ->unsealed_preds > 0) {
        continue;
      }
      fixIncompletePhis(succ);
    }
  }

  // realize phi functions
  for (auto& bb : ssa_basic_blocks) {
    auto block = bb.first;
    auto ssablock = bb.second;

    // Collect and sort to stabilize IR ordering.
    std::vector<Phi*> phis;
    for (auto& pair : ssablock->phi_nodes) {
      phis.push_back(pair.second);
    }
    std::sort(phis.begin(), phis.end(), [](const Phi* a, const Phi* b) -> bool {
      // Sort using > instead of the typical < because we're effectively
      // reversing by looping push_front below.
      return a->output()->id() > b->output()->id();
    });
    for (auto& phi : phis) {
      block->push_front(phi);
    }

    delete ssablock;
  }

  hir_reflow_types_c(&irfunc, start);
}

Register* SSAify::getDefine(SSABasicBlock* ssablock, Register* reg) {
  auto iter = ssablock->local_defs.find(reg);
  if (iter != ssablock->local_defs.end()) {
    // If defined locally, just return
    return iter->second;
  }

  if (ssablock->preds.size() == 0) {
    // If we made it back to the entry block and didn't find a definition, use
    // a Nullptr from LoadConst. Place it after the initialization of the args
    // which explicitly come first.
    if (null_reg_ == nullptr) {
      auto it = ssablock->block->begin();
      while (
          it != ssablock->block->end() &&
          (it->IsLoadArg() || it->IsLoadCurrentFunc() || it->IsLoadFrame())) {
        ++it;
      }
      null_reg_ = env_->AllocateRegister();
      auto loadnull = static_cast<Instr*>(hir_c_create_load_const(null_reg_, Type::toHirType(TNullptr)));
      loadnull->copyBytecodeOffset(*it);
      loadnull->InsertBefore(*it);
    }
    ssablock->local_defs.emplace(reg, null_reg_);
    return null_reg_;
  }

  if (ssablock->unsealed_preds > 0) {
    // If we haven't visited all our predecessors, they can't provide
    // definitions for us to look up. We'll place an incomplete phi that will
    // be resolved once we've visited all predecessors.
    auto phi_output = env_->AllocateRegister();
    ssablock->incomplete_phis.emplace_back(reg, phi_output);
    ssablock->local_defs.emplace(reg, phi_output);
    return phi_output;
  }

  if (ssablock->preds.size() == 1) {
    // If we only have a single predecessor, use its value
    auto new_reg = getDefine(*ssablock->preds.begin(), reg);
    ssablock->local_defs.emplace(reg, new_reg);
    return new_reg;
  }

  // We have multiple predecessors and may need to create a phi.
  auto new_reg = env_->AllocateRegister();
  // Adding a phi may loop back to our block if there is a loop in the CFG.  We
  // update our local_defs before adding the phi to terminate the recursion
  // rather than looping infinitely.
  ssablock->local_defs.emplace(reg, new_reg);
  maybeAddPhi(ssablock, reg, new_reg);

  return ssablock->local_defs.at(reg);
}

void SSAify::maybeAddPhi(
    SSABasicBlock* ssa_block,
    Register* reg,
    Register* out) {
  std::unordered_map<BasicBlock*, Register*> pred_defs;
  for (auto& pred : ssa_block->preds) {
    auto pred_reg = getDefine(pred, reg);
    pred_defs.emplace(pred->block, pred_reg);
  }
  auto bc_off = ssa_block->block->begin()->bytecodeOffset();
  auto phi = Phi::create(out, pred_defs);
  phi->setBytecodeOffset(bc_off);
  ssa_block->phi_nodes.emplace(out, phi);
  for (auto& def_pair : pred_defs) {
    phi_uses_[def_pair.second].emplace(phi, ssa_block);
  }
}

Register* SSAify::getCommonPredValue(
    const Register* out_reg,
    const std::unordered_map<BasicBlock*, Register*>& defs) {
  Register* other_reg = nullptr;

  for (auto& def_pair : defs) {
    auto def = def_pair.second;

    if (def == out_reg) {
      continue;
    }

    if (other_reg != nullptr && def != other_reg) {
      return nullptr;
    }

    other_reg = def;
  }

  return other_reg;
}

void SSAify::fixIncompletePhis(SSABasicBlock* ssa_block) {
  for (auto& pi : ssa_block->incomplete_phis) {
    maybeAddPhi(ssa_block, pi.first, pi.second);
  }
}

std::unordered_map<BasicBlock*, SSABasicBlock*> SSAify::initSSABasicBlocks(
    std::vector<BasicBlock*>& blocks) {
  std::unordered_map<BasicBlock*, SSABasicBlock*> ssa_basic_blocks;

  auto get_or_create_ssa_block =
      [&ssa_basic_blocks](BasicBlock* block) -> SSABasicBlock* {
    auto iter = ssa_basic_blocks.find(block);
    if (iter == ssa_basic_blocks.end()) {
      auto ssablock = new SSABasicBlock(block);
      ssa_basic_blocks.emplace(block, ssablock);
      return ssablock;
    }
    return iter->second;
  };

  for (auto& block : blocks) {
    auto ssablock = get_or_create_ssa_block(block);
    for (auto& edge : block->out_edges()) {
      auto succ = edge->to();
      auto succ_ssa_block = get_or_create_ssa_block(succ);
      auto p = succ_ssa_block->preds.insert(ssablock);
      if (p.second) {
        // It's possible that we have multiple outgoing edges to the same
        // successor. Since we only care about the number of unsealed
        // predecessor *nodes*, only update if this is the first time we're
        // processing this predecessor.
        succ_ssa_block->unsealed_preds++;
        ssablock->succs.insert(succ_ssa_block);
      }
    }
  }

  return ssa_basic_blocks;
}

} // namespace jit::hir
