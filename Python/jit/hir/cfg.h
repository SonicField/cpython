// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/Jit/hir/hir.h"
#include "cinderx/Jit/hir/hir_basic_block_c.h"
#include "cinderx/Jit/hir/hir_cfg_rpo_c.h"

#include <algorithm>
#include <unordered_set>
#include <vector>

struct HirCFGLayoutVerifier;

namespace jit::hir {

class CFG {
 public:
  CFG() = default;
  ~CFG();

  // Allocate a new basic block and insert it into this CFG
  BasicBlock* AllocateBlock();

  // Allocate a block without linking it into the CFG
  BasicBlock* AllocateUnlinkedBlock();

  // Insert a block into the CFG. The CFG takes ownership and will free it
  // upon destruction of the CFG.
  void InsertBlock(BasicBlock* block);

  // Remove block from the CFG
  void RemoveBlock(BasicBlock* block);

  // Split a block after instr. Once split, the block will contain all
  // instructions up to and including instr. A newly allocated block is returned
  // that contains all instructions following instr.
  BasicBlock* splitAfter(Instr& target);

  // Split any critical edges by inserting trampoline blocks.
  void splitCriticalEdges();

  // Return the RPO traversal of the basic blocks in the CFG starting from
  // entry_block.
  std::vector<BasicBlock*> GetRPOTraversal() const;

  // Return the post order traversal of the basic blocks in the CFG starting
  // from entry_block. Used in backward data-flow analysis like unreachable
  // instructions
  std::vector<BasicBlock*> GetPostOrderTraversal() const;

  // Return the BasicBlock in the CFG with the specified id, or nullptr if none
  // exist
  const BasicBlock* getBlockById(int id) const;

  // Return the RPO traversal of the reachable basic blocks in the CFG starting
  // from the given block.
  static std::vector<BasicBlock*> GetRPOTraversal(BasicBlock* start);

  // Returns the post order traversal of the reachable basic blocks in the CFG
  // starting from the given block. Used in backward data-flow analysis like
  // unreachable instructions
  static std::vector<BasicBlock*> GetPostOrderTraversal(BasicBlock* start);

  // Entry point into the CFG; may be null
  BasicBlock* entry_block{nullptr};

  // List of all blocks in the CFG
  IntrusiveList<BasicBlock, &BasicBlock::cfg_node> blocks;

 private:
  DISALLOW_COPY_AND_ASSIGN(CFG);
  friend struct ::HirCFGLayoutVerifier;

  int next_block_id_{0};
};

// ---- Batch 2-F: bodies relocated from cfg.cpp (header-inline). ----
// Trivial delegations stay 1-3 lines; the recursive postorder helper +
// splitAfter + splitCriticalEdges are unchanged from the .cpp version.

// Recursive postorder helper. static-inline to avoid the
// header-anonymous-namespace per-TU duplicate-symbol bug.
static inline void postorder_traverse(
    BasicBlock* block,
    std::vector<BasicBlock*>* traversal,
    std::unordered_set<BasicBlock*>* visited) {
  JIT_CHECK(block != nullptr, "visiting null block!");
  visited->emplace(block);

  Instr* instr = block->GetTerminator();
  switch (instr->opcode()) {
    case Opcode::kCondBranch:
    case Opcode::kCondBranchIterNotDone:
    case Opcode::kCondBranchCheckType: {
      auto cbr = static_cast<CondBranchBase*>(instr);
      if (!visited->contains(cbr->false_bb())) {
        postorder_traverse(cbr->false_bb(), traversal, visited);
      }
      if (!visited->contains(cbr->true_bb())) {
        postorder_traverse(cbr->true_bb(), traversal, visited);
      }
      break;
    }
    case Opcode::kBranch: {
      auto br = static_cast<Branch*>(instr);
      if (!visited->contains(br->target())) {
        postorder_traverse(br->target(), traversal, visited);
      }
      break;
    }
    case Opcode::kDeopt:
    case Opcode::kRaise:
    case Opcode::kRaiseAwaitableError:
    case Opcode::kRaiseStatic:
    case Opcode::kUnreachable:
    case Opcode::kReturn: {
      break;
    }
    default: {
      JIT_ABORT(
          "Block {} has invalid terminator {}", block->id, instr->opname());
    }
  }

  traversal->emplace_back(block);
}

inline CFG::~CFG() {
  hir_cfg_destroy_c(reinterpret_cast<HirCFG*>(this));
}

inline BasicBlock* CFG::AllocateBlock() {
  auto block = AllocateUnlinkedBlock();
  blocks.PushBack(*block);
  return block;
}

inline BasicBlock* CFG::AllocateUnlinkedBlock() {
  int id = next_block_id_;
  auto block = new BasicBlock(id);
  next_block_id_++;
  return block;
}

inline void CFG::InsertBlock(BasicBlock* block) {
  hir_cfg_insert_block(
      reinterpret_cast<HirCFG*>(this),
      reinterpret_cast<HirBasicBlock*>(block));
}

inline void CFG::RemoveBlock(BasicBlock* block) {
  hir_cfg_remove_block(
      reinterpret_cast<HirCFG*>(this),
      reinterpret_cast<HirBasicBlock*>(block));
}

inline BasicBlock* CFG::splitAfter(Instr& target) {
  auto block = target.block();
  auto tail = AllocateBlock();
  for (auto it = std::next(block->iterator_to(target)); it != block->end();) {
    auto& instr = *it;
    ++it;
    instr.unlink();
    tail->Append(&instr);
  }

  for (auto edge : tail->out_edges()) {
    edge->to()->fixupPhis(block, tail);
  }
  return tail;
}

inline void CFG::splitCriticalEdges() {
  // C++ callers don't have Function*, so keep minimal C++ here.
  // The C port (hir_cfg_split_critical_edges_c) needs Function* for
  // hir_cfg_alloc_block. When all callers pass Function*, this can
  // delegate to C. For now, keep the C++ implementation.
  std::vector<Edge*> critical_edges;

  for (auto& block : blocks) {
    auto term = block.GetTerminator();
    JIT_DCHECK(term != nullptr, "Invalid block");
    auto num_edges = term->numEdges();
    if (num_edges < 2) {
      continue;
    }
    for (std::size_t i = 0; i < num_edges; ++i) {
      auto edge = term->edge(i);
      if (edge->to()->in_edges().size() > 1) {
        critical_edges.emplace_back(edge);
      }
    }
  }

  for (auto edge : critical_edges) {
    auto from = edge->from();
    auto to = edge->to();
    auto split_bb = AllocateBlock();
    auto term = edge->from()->GetTerminator();
    split_bb->appendWithOff<Branch>(term->bytecodeOffset(), to);
    edge->set_to(split_bb);
    to->fixupPhis(from, split_bb);
  }
}

inline std::vector<BasicBlock*> CFG::GetRPOTraversal() const {
  void* blocks[4096];
  size_t n = hir_cfg_get_rpo_c(
      const_cast<CFG*>(this), blocks, 4096);
  return std::vector<BasicBlock*>(
      reinterpret_cast<BasicBlock**>(blocks),
      reinterpret_cast<BasicBlock**>(blocks) + n);
}

inline std::vector<BasicBlock*> CFG::GetRPOTraversal(BasicBlock* start) {
  auto traversal = GetPostOrderTraversal(start);
  std::reverse(traversal.begin(), traversal.end());
  return traversal;
}

inline std::vector<BasicBlock*> CFG::GetPostOrderTraversal() const {
  void* blocks[4096];
  size_t n = hir_cfg_get_rpo_c(
      const_cast<CFG*>(this), blocks, 4096);
  std::vector<BasicBlock*> result(
      reinterpret_cast<BasicBlock**>(blocks),
      reinterpret_cast<BasicBlock**>(blocks) + n);
  std::reverse(result.begin(), result.end());
  return result;
}

inline std::vector<BasicBlock*> CFG::GetPostOrderTraversal(BasicBlock* start) {
  std::vector<BasicBlock*> traversal;
  if (start == nullptr) {
    return traversal;
  }
  std::unordered_set<BasicBlock*> visited;
  postorder_traverse(start, &traversal, &visited);
  return traversal;
}

inline const BasicBlock* CFG::getBlockById(int id) const {
  for (auto& block : blocks) {
    if (block.id == id) {
      return &block;
    }
  }
  return nullptr;
}

} // namespace jit::hir
