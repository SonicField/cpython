// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Phase 3D: Reduced — getCodeObjLoc, getCodeObjID, getUnitCallStack inlined
// to debug_info.h. Remaining: buildActivationMap (HIR traversal),
// resolvePending (asmjit::CodeHolder), getCallerID/addUnitCallStack
// (hir::FrameState).

#include "cinderx/Jit/debug_info.h"

#include "cinderx/Jit/hir/function.h"
#include "cinderx/Jit/hir/hir.h"

#include <deque>

namespace jit {

namespace {

struct Activation {
  Activation(PyCodeObject* c, const jit::hir::FrameState* cfs)
      : code_obj(c), caller_frame_state(cfs) {}
  PyCodeObject* code_obj;
  const jit::hir::FrameState* caller_frame_state;
};

using ActivationMap = std::unordered_map<const jit::hir::Instr*, Activation>;

struct WorkItem {
  WorkItem(const jit::hir::BasicBlock* b, const Activation& c)
      : block(b), activation(c) {}
  const jit::hir::BasicBlock* block;
  Activation activation;
};

ActivationMap buildActivationMap(const jit::hir::Function& func) {
  JIT_CHECK(func.code, "func has no code object");
  ActivationMap amap;
  std::deque<WorkItem> workq{
      WorkItem{func.cfg.entry_block, Activation{func.code, nullptr}}};
  std::unordered_set<const jit::hir::BasicBlock*> processed;
  while (!workq.empty()) {
    WorkItem item = std::move(workq.front());
    workq.pop_front();

    if (processed.contains(item.block)) {
      continue;
    }

    for (const auto& instr : *item.block) {
      switch (instr.opcode()) {
        case jit::hir::Opcode::kBeginInlinedFunction: {
          auto bif = static_cast<const jit::hir::BeginInlinedFunction*>(&instr);
          item.activation = Activation{bif->code(), bif->callerFrameState()};
          amap.emplace(&instr, item.activation);
          break;
        }
        case jit::hir::Opcode::kEndInlinedFunction: {
          const jit::hir::FrameState* caller =
              item.activation.caller_frame_state;
          item.activation = Activation{caller->code, caller->parent};
          amap.emplace(&instr, item.activation);
          break;
        }
        default: {
          amap.emplace(&instr, item.activation);
          break;
        }
      }
    }

    processed.insert(item.block);

    for (const auto& edge : item.block->out_edges()) {
      workq.emplace_back(edge->to(), item.activation);
    }
  }

  return amap;
}

} // namespace

void DebugInfo::resolvePending(
    const std::vector<PendingDebugLoc>& pending,
    const jit::hir::Function& func,
    const asmjit::CodeHolder& code) {
  ActivationMap amap = buildActivationMap(func);
  JIT_CHECK(code.hasBaseAddress(), "code not generated");
  uint64_t base = code.baseAddress();
  for (const PendingDebugLoc& item : pending) {
    auto it = amap.find(item.instr);
    JIT_CHECK(it != amap.end(), "instr doesn't belong to func");
    uintptr_t addr = base + code.labelOffsetFromBase(item.label);
    const auto& [code_obj, caller_frame_state] = it->second;
    addUnitCallStack(
        addr, code_obj, item.instr->bytecodeOffset(), caller_frame_state);
  }
}

void DebugInfo::addUnitCallStack(
    uintptr_t addr,
    PyCodeObject* code,
    BCOffset bc_off,
    const jit::hir::FrameState* caller_frame_state) {
  uint16_t caller_id = getCallerID(caller_frame_state);
  uint16_t code_obj_id = getCodeObjID(code);
  addr_locs_.emplace(addr, LocNode{code_obj_id, caller_id, bc_off});
}

uint16_t DebugInfo::getCallerID(const jit::hir::FrameState* caller) {
  if (caller == nullptr) {
    return kNoCallerID;
  }
  LocNode node{
      getCodeObjID(caller->code),
      getCallerID(caller->parent),
      caller->instrOffset()};
  for (uint16_t i = 0; i < inlined_calls_.size(); i++) {
    if (inlined_calls_[i] == node) {
      return i;
    }
  }
  JIT_CHECK(inlined_calls_.size() < kMaxInlined, "too many inlined functions");
  inlined_calls_.emplace_back(node);
  return inlined_calls_.size() - 1;
}

} // namespace jit
