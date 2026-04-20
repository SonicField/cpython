// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cinderx/Jit/hir/pass.h"

#include "cinderx/Jit/hir/hir_c_api.h"
#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/hir/hir_type_c.h"
#include "cinderx/Jit/context.h"
#include "cinderx/Jit/hir/analysis.h"
#include "cinderx/Jit/hir/printer.h"

extern "C" void hir_simplify_redundant_cond_branches_c(void *cfg);
extern "C" void hir_reflow_types_c(void *func, void *start_block);
extern "C" int hir_remove_trampoline_blocks_c(void *cfg);
extern "C" int hir_remove_unreachable_blocks_c(void *func);

namespace jit::hir {

static inline HirType to_hir(Type t) {
  return Type::toHirType(t);
}



RegUses collectDirectRegUses(Function& func) {
  RegUses uses;
  for (auto& block : func.cfg.blocks) {
    for (Instr& instr : block) {
      for (size_t i = 0; i < instr.NumOperands(); ++i) {
        uses[instr.GetOperand(i)].insert(&instr);
      }
    }
  }
  return uses;
}


void reflowTypes(Function& func) {
  hir_reflow_types_c(&func, func.cfg.entry_block);
}

void reflowTypes(Function& func, BasicBlock* start) {
  hir_reflow_types_c(&func, start);
}

bool removeTrampolineBlocks(CFG* cfg) {
  return hir_remove_trampoline_blocks_c(cfg) != 0;
}

bool removeUnreachableBlocks(Function& func) {
  return hir_remove_unreachable_blocks_c(&func) != 0;
}

bool removeUnreachableInstructions(Function& func) {
  auto cfg = &func.cfg;

  bool modified = false;
  std::vector<BasicBlock*> blocks = cfg->GetPostOrderTraversal();
  DominatorAnalysis dom(func);
  RegUses reg_uses = collectDirectRegUses(func);
  auto remove_reg_uses = [&reg_uses](Instr* instr) {
    for (auto op : instr->GetOperands()) {
      auto instrs = reg_uses.find(op);
      if (instrs != reg_uses.end()) {
        instrs->second.erase(instr);
      }
    }
  };
  for (BasicBlock* block : blocks) {
    auto it = block->begin();
    while (it != block->end()) {
      Instr& instr = *it;
      ++it;
      if ((instr.output() == nullptr || !instr.output()->isA(TBottom)) &&
          !instr.IsUnreachable()) {
        continue;
      }
      // 1) Any instruction dominated by a definition of a Bottom value is
      // unreachable, so we delete any such instructions and replace them
      // with a special marker instruction (Unreachable)
      // 2) Any instruction post dominated by Unreachable must deopt if it can
      // deopt, else it is unreachable itself.

      modified = true;
      // Find the last instruction between [block.begin, current instruction]
      // that can deopt. Place the Unreachable marker right after that
      // instruction. If we can't find any instruction that can deopt, the
      // Unreachable marker is placed at the beginning of the block.
      do {
        auto prev_it = std::prev(it);
        Instr& prev_instr = *prev_it;
        if (prev_instr.asDeoptBase() != nullptr) {
          break;
        }
        it = prev_it;
      } while (it != block->begin());

      if (it != block->begin() && std::prev(it)->IsGuardType()) {
        // Everything after this GuardType is unreachable, but only as long as
        // the GuardType fails at runtime. Indicate that the guard is required
        // for correctness with a UseType. This prevents GuardTypeElimination
        // from removing it.
        Instr& guard_type = *std::prev(it);
        block->insert(
            static_cast<Instr*>(hir_c_create_use_type(guard_type.output(), to_hir(guard_type.output()->type()))),
            it);
      }

      block->insert(static_cast<Instr*>(hir_c_create_unreachable()), it);
      // Clean up dangling phi references
      if (Instr* old_term = block->GetTerminator()) {
        for (std::size_t i = 0, n = old_term->numEdges(); i < n; ++i) {
          auto bb = old_term->successor(i);
          for (auto& potential_phi : *bb) {
            if (potential_phi.IsPhi()) {
              remove_reg_uses(&potential_phi);
            }
          }

          old_term->successor(i)->removePhiPredecessor(block);
        }
      }
      // Remove all instructions after the Unreachable
      while (it != block->end()) {
        Instr& instrToDelete = *it;
        ++it;
        instrToDelete.unlink();
        remove_reg_uses(&instrToDelete);
        Instr::Destroy(&instrToDelete);
      }
    }
    if (block->begin()->IsUnreachable()) {
      std::vector<Instr*> interesting_branches;
      // If one edge of a conditional branch leads to an Unreachable, it can be
      // replaced with a Branch to the other target. If a Branch leads to an
      // Unreachable, it is replaced with an Unreachable.
      for (const Edge* edge : block->in_edges()) {
        BasicBlock* predecessor = edge->from();
        interesting_branches.emplace_back(predecessor->GetTerminator());
      }
      for (Instr* branch : interesting_branches) {
        if (branch->IsBranch()) {
          branch->ReplaceWith(*static_cast<Instr*>(hir_c_create_unreachable()));
        } else if (branch->IsCondBranch() || branch->IsCondBranchIterNotDone() ||
                   branch->IsCondBranchCheckType()) {
          auto cond_branch = static_cast<CondBranchBase*>(branch);
          BasicBlock* target;
          if (cond_branch->false_bb() == block) {
            target = cond_branch->true_bb();
          } else {
            JIT_CHECK(
                cond_branch->true_bb() == block,
                "true branch must be unreachable");
            target = cond_branch->false_bb();
          }

          if (branch->IsCondBranchCheckType()) {
            // Before replacing a CondBranchCheckType with a Branch to the
            // reachable block, insert a RefineType to preserve the type
            // information implied by following that path.
            auto check_type_branch = static_cast<CondBranchCheckType*>(branch);
            Register* refined_value = func.env.AllocateRegister();
            Type check_type = check_type_branch->type();
            if (target == cond_branch->false_bb()) {
              check_type = TTop - check_type_branch->type();
            }

            Register* operand = check_type_branch->GetOperand(0);
            static_cast<Instr*>(hir_c_create_refine_type_reg(refined_value, to_hir(check_type), operand))
                ->InsertBefore(*cond_branch);
            auto uses = reg_uses.find(operand);
            if (uses == reg_uses.end()) {
              break;
            }
            std::unordered_set<Instr*>& instrs_using_reg = uses->second;
            const std::unordered_set<const BasicBlock*>& dom_set =
                dom.getBlocksDominatedBy(target);
            for (Instr* instr : instrs_using_reg) {
              if (dom_set.contains(instr->block())) {
                instr->ReplaceUsesOf(operand, refined_value);
              }
            }
          }
          cond_branch->ReplaceWith(*static_cast<Instr*>(hir_c_create_branch_cpp(target)));
        } else {
          JIT_ABORT("Unexpected branch instruction {}", *branch);
        }
        remove_reg_uses(branch);
        Instr::Destroy(branch);
      }
    }
  }
  if (modified) {
    removeUnreachableBlocks(func);
    reflowTypes(func);
  }
  return modified;
}


} // namespace jit::hir
