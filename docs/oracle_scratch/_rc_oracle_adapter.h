// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// W3 R4 ORACLE ADAPTER — re-creates deleted analysis.h C++ API as a thin
// wrapper around the HirLivenessState C API. Used by docs/oracle_scratch/
// rc_oracle.cpp (a snapshot of cc4a18e7e5:Python/jit/hir/refcount_insertion.cpp)
// to compile against current jit/hir/ infrastructure.
//
// SCOPE: oracle scratch lib ONLY. Pinned to push 35 HEAD d81e5806c3 per
// theologian 23:34:14Z risk-mitigation. Not for production use.
//
// PHASE 0 AUDIT findings:
//   - cc4a18e7e5:analysis.h had `class LivenessAnalysis` (line 108) +
//     `extern const RegisterSet kEmptyRegSet` (line 20).
//   - HEAD analysis.h has only `using RegisterSet = std::unordered_set<Register*>`.
//   - liveness_c.h provides HirLivenessState C API that this adapter wraps.
//
// BRIDGE SPEC TEMPLATE (LITE form):
//   Bridge: _rc_oracle_adapter.h (LivenessAnalysis adapter class + kEmptyRegSet)
//   Purpose: expose deleted-from-HEAD LivenessAnalysis C++ class as wrapper
//            around HirLivenessState C API
//   C++ source: cc4a18e7e5:Python/jit/hir/analysis.h (deleted at HEAD)
//   PRIOR DECISIONS:
//     - D-1776820568: W3 MEDIUM priority, oracle-pinned to d81e5806c3
//     - 14-bug R3b session: RC_DIFF found bugs #1-#14 by C-vs-C++ runtime diff
//     - R4 deletion timing: refcount_insertion.cpp deleted before corpus
//       methodology adopted; W3 closes the historical gap
//   INVARIANTS PRESERVED:
//     1. LivenessAnalysis API surface: Run() + GetLastUses() + GetIn(block) —
//        the 3 methods refcount_insertion.cpp actually uses (verified via grep)
//     2. LastUses iteration order: refcount_insertion uses iteration order to
//        decide WHERE Decref instructions are inserted; adapter delegates to
//        HirLivenessState which preserves the underlying analysis order
//     3. kEmptyRegSet sentinel: empty-set value used as fallback in
//        ?: expressions; defined in _rc_oracle_adapter.cpp as static const
//        empty RegisterSet (operator== / contains / size all OK out of box)
//     4. RegisterSet semantics: identical typedef (std::unordered_set<Register*>)
//        — adapter just passes Register* pointers through, no marshalling
//     5. Oracle pin: this header is in docs/oracle_scratch/, included only by
//        rc_oracle.cpp; CMakeLists.txt (Step 4) git-checkouts d81e5806c3 for
//        the underlying jit/hir headers
//   Falsifier (Step 5 deliverable): inject synthetic refcount divergence into
//   C path, scripts/rc_diff_oracle.sh produces non-empty output. If empty
//   under injection, oracle is non-functional.

#pragma once

#include "cinderx/Jit/hir/analysis.h"   // RegisterSet (HEAD)
#include "cinderx/Jit/hir/liveness_c.h" // HirLivenessState C API
#include "cinderx/Jit/hir/hir.h"        // Function, BasicBlock, Instr, Register

#include <ostream>
#include <unordered_map>
#include <unordered_set>

namespace jit::hir {

// Sentinel: empty RegisterSet used as fallback in refcount_insertion.cpp's
// ?: expressions. Defined in _rc_oracle_adapter.cpp.
extern const RegisterSet kEmptyRegSet;

// cc4a18e7e5:analysis.h had operator<< for RegisterSet (used by
// fmt::streamed in rc_oracle.cpp's TRACE). HEAD's analysis.h dropped it.
// Adapter restores the formatter so rc_oracle.cpp compiles unmodified.
// Format is debug-only; not load-bearing for the post-pass HIR diff.
std::ostream &operator<<(std::ostream &os, const RegisterSet &set);

// LivenessAnalysis adapter. Constructor takes Function& (matches cc4a18e7e5
// API). Run() builds the underlying HirLivenessState. GetLastUses() and
// GetIn(block) project the C state into the C++ container types
// refcount_insertion.cpp expects.
class LivenessAnalysis {
 public:
  explicit LivenessAnalysis(const Function& irfunc) : func_(irfunc) {}

  ~LivenessAnalysis() {
    if (state_ != nullptr) {
      hir_liveness_destroy(state_);
    }
  }

  // Non-copyable (state_ is unique-owned).
  LivenessAnalysis(const LivenessAnalysis&) = delete;
  LivenessAnalysis& operator=(const LivenessAnalysis&) = delete;

  void Run() {
    // const_cast: HirFunction is an opaque void* alias; the C API only reads.
    state_ = hir_liveness_create(
        reinterpret_cast<HirFunction>(const_cast<Function*>(&func_)));
  }

  using LastUses =
      std::unordered_map<const Instr*, std::unordered_set<Register*>>;

  // Iterate every Instr in the function; for each, collect dying registers
  // via hir_liveness_get_dying_regs. Result map matches cc4a18e7e5 typedef.
  LastUses GetLastUses() {
    LastUses result;
    for (auto& block : func_.cfg.blocks) {
      for (auto& instr : block) {
        constexpr size_t kCap = 64;  // refcount_insertion's deepest fn ≤32 dying
        void* dying[kCap];
        size_t n = hir_liveness_get_dying_regs(
            state_,
            reinterpret_cast<HirInstr>(const_cast<Instr*>(&instr)),
            dying, kCap);
        if (n == 0) continue;
        auto& set = result[&instr];
        for (size_t i = 0; i < n; i++) {
          set.insert(reinterpret_cast<Register*>(dying[i]));
        }
      }
    }
    return result;
  }

  // Live-in set for a basic block. Builds via foreach callback (no per-reg
  // iteration on caller side).
  RegisterSet GetIn(const BasicBlock* block) {
    RegisterSet result;
    auto cb = [](void* reg, void* ctx) {
      static_cast<RegisterSet*>(ctx)->insert(static_cast<Register*>(reg));
    };
    hir_liveness_foreach_live_in(state_, block, cb, &result);
    return result;
  }

 private:
  const Function& func_;
  HirLivenessState* state_{nullptr};
};

} // namespace jit::hir
