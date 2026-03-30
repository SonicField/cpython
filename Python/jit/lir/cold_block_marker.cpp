// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "cinderx/Jit/lir/cold_block_marker.h"

#include "cinderx/Jit/codegen/code_section.h"
#include "cinderx/Jit/lir/block.h"
#include "cinderx/Jit/lir/function.h"
#include "cinderx/Jit/lir/instruction.h"

#include <queue>
#include <vector>

namespace jit::lir {

namespace {

// H1: A block is a guard failure target if:
//   - It has exactly one predecessor
//   - That predecessor's last instruction is a Guard
//   - This block is the guard's failure target (false successor)
bool isGuardFailureTarget(const BasicBlock* block) {
  const auto& preds = block->predecessors();
  if (preds.size() != 1) {
    return false;
  }

  const BasicBlock* pred = preds[0];
  const Instruction* last = pred->getLastInstr();
  if (last == nullptr || last->opcode() != Instruction::kGuard) {
    return false;
  }

  // Guard's false successor is the failure path.
  // Guard has two successors: true (continue) and false (deopt/failure).
  return pred->getFalseSuccessor() == block;
}

// H2: A block is a deopt stub if it contains only a single Guard instruction
// (which encodes the deopt exit inline) or is otherwise a minimal block
// that terminates in a branch to an already-cold block.
bool isDeoptStub(const BasicBlock* block) {
  if (block->getNumInstrs() == 0) {
    return false;
  }

  // Single-instruction blocks with a Guard are deopt stubs.
  if (block->getNumInstrs() == 1) {
    const Instruction* instr = block->getFirstInstr();
    if (instr->opcode() == Instruction::kGuard) {
      return true;
    }
  }

  return false;
}

} // namespace

void markColdBlocks(Function* func) {
  auto& blocks = func->basicblocks();
  if (blocks.empty()) {
    return;
  }

  // Never mark the entry block as cold.
  BasicBlock* entry = func->entryBlock();

  // Phase 1: Seed cold blocks using H1 and H2 heuristics.
  std::queue<BasicBlock*> worklist;

  for (BasicBlock* block : blocks) {
    if (block == entry) {
      continue;
    }

    if (isGuardFailureTarget(block) || isDeoptStub(block)) {
      block->setSection(codegen::CodeSection::kCold);
      // Add successors to worklist for transitive closure.
      for (BasicBlock* succ : block->successors()) {
        worklist.push(succ);
      }
    }
  }

  // Phase 2: Transitive closure (H3).
  // A block is cold if ALL its predecessors are cold.
  // This is a monotonic forward dataflow — we only transition hot→cold.
  // O(V+E) since each block is enqueued at most once.
  while (!worklist.empty()) {
    BasicBlock* block = worklist.front();
    worklist.pop();

    // Skip entry block and already-cold blocks.
    if (block == entry ||
        block->section() == codegen::CodeSection::kCold) {
      continue;
    }

    // Check if ALL predecessors are cold.
    bool all_preds_cold = true;
    for (const BasicBlock* pred : block->predecessors()) {
      if (pred->section() != codegen::CodeSection::kCold) {
        all_preds_cold = false;
        break;
      }
    }

    if (all_preds_cold && !block->predecessors().empty()) {
      block->setSection(codegen::CodeSection::kCold);
      for (BasicBlock* succ : block->successors()) {
        if (succ->section() != codegen::CodeSection::kCold) {
          worklist.push(succ);
        }
      }
    }
  }
}

} // namespace jit::lir
