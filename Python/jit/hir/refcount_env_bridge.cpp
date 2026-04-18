/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C++ bridge for refcount_env_c functions that need C++ type constants.
 */

#include "cinderx/Jit/hir/refcount_env_c.h"
#include "cinderx/Jit/hir/refcount_structs_c.h"
#include "cinderx/Jit/hir/hir.h"
#include "cinderx/Jit/hir/type.h"
#include "cinderx/Jit/hir/analysis.h"
#include "cinderx/Jit/deopt.h"

using namespace jit::hir;

extern "C" {

int phx_rc_is_uncounted(void *reg) {
  auto* r = static_cast<Register*>(reg);
  HirType h_reg = Type::toHirType(r->type());
  HirType h_mortal = Type::toHirType(TMortalObject);
  return !hir_type_could_be(&h_reg, &h_mortal);
}

int phx_rc_reg_is_object(void *reg) {
  auto* r = static_cast<Register*>(reg);
  return r->type() <= TObject ? 1 : 0;
}

int phx_rc_condbranch_check_type_is_wait_handle(void *instr) {
  auto* cond = static_cast<CondBranchCheckType*>(static_cast<Instr*>(instr));
  return cond->type() == TWaitHandle ? 1 : 0;
}

int phx_rc_is_passthrough(void *instr) {
  return isPassthrough(*static_cast<Instr*>(instr)) ? 1 : 0;
}

int phx_rc_is_guard_is(void *instr) {
  return static_cast<Instr*>(instr)->IsGuardIs() ? 1 : 0;
}

void phx_rc_fill_deopt_live_regs(const PhxStateMap *live_regs, void *instr_ptr) {
  auto& instr = *static_cast<Instr*>(instr_ptr);
  auto deopt = instr.asDeoptBase();
  if (deopt == nullptr) {
    return;
  }

  JIT_CHECK(deopt->live_regs().empty(), "Instruction should have no live regs");

  for (size_t idx = 0; idx < live_regs->capacity; idx++) {
    if (!live_regs->keys[idx]) continue;
    const PhxRegState& rstate = live_regs->values[idx];
    auto ref_kind = static_cast<RefKind>(rstate.kind);

    for (int i = 0, n = (int)rstate.n_copies; i < n; ++i) {
      auto* reg = static_cast<Register*>(rstate.copies[i]);
      HirType h_regtype = Type::toHirType(reg->type());
      HirType h_cptr = Type::toHirType(TCPtr);
      if (hir_type_could_be(&h_regtype, &h_cptr)) {
        continue;
      }
      deopt->emplaceLiveReg(reg, ref_kind, jit::deoptValueKind(reg->type()));
      if (ref_kind == RefKind::kOwned) {
        ref_kind = RefKind::kBorrowed;
      }
    }
  }
  deopt->sortLiveRegs();
}

int phx_rc_merge_verify(const PhxRegState *c_dst, const PhxRegState *c_from,
                        const PhxRegState *c_result) {
  RefKind dst_kind = static_cast<RefKind>(c_dst->kind);
  RefKind from_kind = static_cast<RefKind>(c_from->kind);
  RefKind result_kind = static_cast<RefKind>(c_result->kind);

  /* Apply C++ merge logic directly */
  RefKind expected_kind;
  if (dst_kind == from_kind) {
    expected_kind = dst_kind;
  } else if (dst_kind == RefKind::kUncounted) {
    expected_kind = from_kind;
  } else if (from_kind == RefKind::kUncounted) {
    expected_kind = dst_kind;
  } else {
    expected_kind = RefKind::kOwned;
  }

  if (result_kind != expected_kind) {
    JIT_LOG(
        "phx_rs_merge DIVERGENCE: dst_kind=%d from_kind=%d "
        "expected=%d got=%d",
        (int)dst_kind, (int)from_kind,
        (int)expected_kind, (int)result_kind);
    return 0;
  }
  return 1;
}

void *phx_rc_model_reg(void *reg) {
  return modelReg(static_cast<Register*>(reg));
}

static LivenessAnalysis* g_cpp_liveness = nullptr;
static Function* g_cpp_liveness_func = nullptr;

int phx_rc_liveness_is_live_in(void *func_ptr, void *block, void *reg) {
  auto* func = static_cast<Function*>(func_ptr);
  if (g_cpp_liveness_func != func) {
    delete g_cpp_liveness;
    g_cpp_liveness = new LivenessAnalysis(*func);
    g_cpp_liveness->Run();
    g_cpp_liveness_func = func;
  }
  auto live_in = g_cpp_liveness->GetIn(static_cast<BasicBlock*>(block));
  return live_in.count(static_cast<Register*>(reg)) ? 1 : 0;
}

} /* extern "C" */
