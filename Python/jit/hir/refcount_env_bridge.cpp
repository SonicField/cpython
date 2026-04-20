/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C++ bridge for refcount_env_c functions that need C++ type constants.
 */

#include "cinderx/Jit/hir/refcount_env_c.h"
#include "cinderx/Jit/hir/refcount_structs_c.h"
#include "cinderx/Jit/hir/hir.h"
#include "cinderx/Jit/hir/function.h"
#include "cinderx/Jit/hir/type.h"
#include "cinderx/Jit/deopt.h"

using namespace jit::hir;

extern "C" HirType hir_register_type(void *reg);
extern "C" size_t hir_cfg_get_rpo(void *cfg, void **out, size_t capacity);

extern "C" {

int phx_rc_is_uncounted(void *reg) {
  HirType h_reg = hir_register_type(reg);
  HirType h_mortal = HIR_TYPE_SIMPLE(0x000ffffffffULL, HIR_LIFETIME_MORTAL);
  return !hir_type_could_be(&h_reg, &h_mortal);
}

int phx_rc_reg_is_object(void *reg) {
  HirType h_reg = hir_register_type(reg);
  HirType t_object = HIR_TYPE_OBJECT;
  return hir_type_is_subtype(h_reg, t_object) ? 1 : 0;
}

int phx_rc_condbranch_check_type_is_wait_handle(void *instr) {
  auto* cond = static_cast<CondBranchCheckType*>(static_cast<Instr*>(instr));
  return cond->type() == TWaitHandle ? 1 : 0;
}

int phx_rc_is_passthrough(void *instr) {
  extern int hir_is_passthrough_c(const void *instr);
  return hir_is_passthrough_c(instr);
}

int phx_rc_is_guard_is(void *instr) {
  return hir_c_opcode(instr) == HIR_OP_GuardIs ? 1 : 0;
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
  extern void *hir_model_reg_c(void *reg);
  return hir_model_reg_c(reg);
}

size_t phx_rc_get_rpo(void *func_ptr, void **out, size_t capacity) {
  auto& func = *static_cast<jit::hir::Function*>(func_ptr);
  auto rpo = func.cfg.GetRPOTraversal();
  size_t n = rpo.size() < capacity ? rpo.size() : capacity;
  for (size_t i = 0; i < n; i++) out[i] = rpo[i];
  return rpo.size();
}

} /* extern "C" */
