// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/Jit/hir/function.h"
#include "cinderx/Jit/hir/hir.h"

#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace jit::hir {

// An abstract compiler pass over an HIR function.
class Pass {
 public:
  explicit Pass(std::string_view name) : name_{name} {}
  virtual ~Pass() = default;

  virtual void Run(Function& irfunc) = 0;

  constexpr std::string_view name() const {
    return name_;
  }

 protected:
  std::string name_;
};

// General utilities that a compiler pass might want to make use of.

using RegUses = std::unordered_map<Register*, std::unordered_set<Instr*>>;

// Recursively chase a list of assignments and get the original register value.
// If there are no assignments then just get the register back.
Register* chaseAssignOperand(Register* value);

// Collect direct operand uses of all Registers in the given func, excluding
// uses in FrameState or other metadata.
RegUses collectDirectRegUses(Function& func);

// Re-derive all Register types in the given function. Meant to be called after
// SSAify and any optimizations that could refine the output type of an
// instruction.
void reflowTypes(Function& func);
void reflowTypes(Function& func, BasicBlock* start);

// Remove any blocks that consist of a single jump to another block.
bool removeTrampolineBlocks(CFG* cfg);

// Remove blocks that aren't reachable from the entry, whether or not they're
// empty. Return true if it changed the graph and false otherwise.
bool removeUnreachableBlocks(Function& func);

// Remove instructions that aren't reachable from the entry. Return true if it
// changed the graph and false otherwise.
bool removeUnreachableInstructions(Function& func);

// Replace cond branches where both sides go to the same block with a direct
// branch.
void simplifyRedundantCondBranches(CFG* cfg);

} // namespace jit::hir
