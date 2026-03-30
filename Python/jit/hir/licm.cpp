// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cinderx/Jit/hir/licm.h"

#include "cinderx/Jit/hir/analysis.h"
#include "cinderx/Jit/hir/hir.h"

#include <queue>
#include <unordered_set>
#include <vector>

namespace jit::hir {

namespace {

// A natural loop identified by a back edge in the CFG.
struct LoopInfo {
  BasicBlock* header;  // The loop header (target of back edge)
  std::unordered_set<BasicBlock*> body;  // All blocks in the loop body
  BasicBlock* preheader;  // Single-entry predecessor of header (may be null)
};

// Check if block A dominates block B using the dominator analysis.
bool dominates(
    DominatorAnalysis& doms,
    const BasicBlock* a,
    const BasicBlock* b) {
  const BasicBlock* cur = b;
  while (cur != nullptr) {
    if (cur == a) {
      return true;
    }
    cur = doms.immediateDominator(cur);
  }
  return false;
}

// Find all natural loops in the function using back-edge detection.
std::vector<LoopInfo> findLoops(
    Function& irfunc,
    DominatorAnalysis& doms) {
  std::vector<LoopInfo> loops;
  auto rpo = irfunc.cfg.GetRPOTraversal();

  for (BasicBlock* block : rpo) {
    for (const Edge* edge : block->out_edges()) {
      BasicBlock* target = edge->to();
      // A back edge is an edge where the target dominates the source.
      if (target != nullptr && dominates(doms, target, block)) {
        // Found back edge: block -> target. Compute natural loop body.
        LoopInfo loop;
        loop.header = target;
        loop.body.insert(target);
        loop.preheader = nullptr;

        // BFS backwards from the back-edge source to find all blocks
        // that can reach the source without going through the header.
        if (block != target) {
          std::queue<BasicBlock*> worklist;
          loop.body.insert(block);
          worklist.push(block);

          while (!worklist.empty()) {
            BasicBlock* cur = worklist.front();
            worklist.pop();
            for (const Edge* pred_edge : cur->in_edges()) {
              BasicBlock* pred = pred_edge->from();
              if (pred != nullptr && loop.body.find(pred) == loop.body.end()) {
                loop.body.insert(pred);
                worklist.push(pred);
              }
            }
          }
        }

        // Find preheader: a predecessor of the header that is NOT in the
        // loop body. If there is exactly one such predecessor, it is the
        // preheader. Otherwise, we skip this loop (creating preheaders
        // would require CFG modification).
        BasicBlock* preheader_candidate = nullptr;
        int non_loop_preds = 0;
        for (const Edge* pred_edge : target->in_edges()) {
          BasicBlock* pred = pred_edge->from();
          if (pred != nullptr && loop.body.find(pred) == loop.body.end()) {
            preheader_candidate = pred;
            non_loop_preds++;
          }
        }
        if (non_loop_preds == 1) {
          loop.preheader = preheader_candidate;
        }

        if (loop.preheader != nullptr) {
          loops.push_back(std::move(loop));
        }
      }
    }
  }

  return loops;
}

// Check if an instruction is a guard that can be hoisted.
// Only GuardType and GuardIs are candidates.
bool isHoistableGuard(const Instr& instr) {
  return instr.IsGuardType() || instr.IsGuardIs();
}

// Check if a register is defined outside the loop.
bool isDefinedOutsideLoop(
    Register* reg,
    const std::unordered_set<BasicBlock*>& loop_body) {
  if (reg == nullptr) {
    return true;
  }
  Instr* def = reg->instr();
  if (def == nullptr) {
    return true;  // Function argument or constant — always outside loop
  }
  BasicBlock* def_block = def->block();
  return def_block == nullptr || loop_body.count(def_block) == 0;
}

// Check if all uses of an instruction are defined outside the loop.
// For DeoptBase instructions (GuardType, GuardIs), this also checks the
// FrameState and live_regs — registers referenced by deoptimisation
// metadata must also be defined outside the loop, otherwise deopt after
// hoisting would reference uninitialised values and segfault.
bool allUsesOutsideLoop(
    Instr& instr,
    const std::unordered_set<BasicBlock*>& loop_body) {
  bool all_outside = true;
  instr.visitUses([&](Register*& reg) -> bool {
    if (!isDefinedOutsideLoop(reg, loop_body)) {
      all_outside = false;
      return false;  // Stop visiting
    }
    return true;
  });
  return all_outside;
}

// Hoist loop-invariant guards from a single loop to its preheader.
int hoistInvariantGuards(LoopInfo& loop) {
  int hoisted = 0;

  // Collect instructions to hoist (can not modify while iterating).
  std::vector<Instr*> to_hoist;

  for (BasicBlock* block : loop.body) {
    if (block == loop.preheader) {
      continue;  // Don't scan the preheader itself
    }
    for (auto it = block->begin(); it != block->end(); ++it) {
      Instr& instr = *it;
      if (!isHoistableGuard(instr)) {
        continue;
      }
      if (instr.IsPhi()) {
        continue;  // Never hoist phi nodes
      }
      if (allUsesOutsideLoop(instr, loop.body)) {
        to_hoist.push_back(&instr);
      }
    }
  }

  // Move each hoistable instruction to the preheader.
  // Insert before the terminator (Branch to loop header).
  for (Instr* instr : to_hoist) {
    // Unlink from current block
    BasicBlock* old_block = instr->block();
    instr->unlink();

    // Insert into preheader before the terminator
    auto term_it = loop.preheader->end();
    --term_it;  // Point to terminator
    loop.preheader->insert(instr, term_it);

    hoisted++;
    JIT_LOG("LICM: hoisted {} from bb{} to preheader bb{}",
            instr->opname(), old_block->id, loop.preheader->id);
  }

  return hoisted;
}

} // namespace

void LICM::Run(Function& irfunc) {
  DominatorAnalysis doms(irfunc);
  auto loops = findLoops(irfunc, doms);

  if (loops.empty()) {
    return;
  }

  int total_hoisted = 0;
  for (auto& loop : loops) {
    total_hoisted += hoistInvariantGuards(loop);
  }

  if (total_hoisted > 0) {
    JIT_LOG("LICM: hoisted {} instructions across {} loops in {}",
            total_hoisted, loops.size(), irfunc.fullname);
  }
}

} // namespace jit::hir
