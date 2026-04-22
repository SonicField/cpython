/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C++ bridge for refcount_env_c functions that need C++ type constants.
 */

#include "cinderx/Jit/hir/refcount_env_c.h"
#include "cinderx/Jit/hir/refcount_structs_c.h"
#include "cinderx/Jit/hir/hir_c_api.h"
#include "cinderx/Jit/hir/hir_cfg_rpo_c.h"
#include "cinderx/Jit/hir/hir.h"
#include "cinderx/Jit/hir/function.h"
#include "cinderx/Jit/hir/type.h"
#include "cinderx/Jit/deopt.h"

using namespace jit::hir;

extern "C" HirType hir_register_type(void *reg);

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
  HirType check_type = ((const HirCondBranchCheckType *)instr)->type;
  HirType t_wh = HIR_TYPE_WAITHANDLE;
  return (memcmp(&check_type, &t_wh, sizeof(HirType)) == 0) ? 1 : 0;
}

int phx_rc_is_passthrough(void *instr) {
  extern int hir_is_passthrough_c(const void *instr);
  return hir_is_passthrough_c(instr);
}

int phx_rc_is_guard_is(void *instr) {
  return hir_c_opcode(instr) == HIR_OP_GuardIs ? 1 : 0;
}

void phx_rc_fill_deopt_live_regs(const PhxStateMap *live_regs, void *instr_ptr) {
  extern void hir_deopt_emplace_live_reg(void *instr, void *reg, int ref_kind, int value_kind);
  extern void hir_deopt_sort_live_regs(void *instr);
  extern int hir_deopt_value_kind(void *reg);

  if (!hir_instr_is_deopt_base(instr_ptr)) return;

  HirType h_cptr = HIR_TYPE_CPTR;

  for (size_t idx = 0; idx < live_regs->capacity; idx++) {
    if (!live_regs->keys[idx]) continue;
    const PhxRegState& rstate = live_regs->values[idx];
    int ref_kind = rstate.kind;

    for (int i = 0, n = (int)rstate.n_copies; i < n; ++i) {
      void* reg = rstate.copies[i];
      HirType h_regtype = hir_register_type(reg);
      if (hir_type_could_be(&h_regtype, &h_cptr)) {
        continue;
      }
      hir_deopt_emplace_live_reg(instr_ptr, reg, ref_kind, hir_deopt_value_kind(reg));
      if (ref_kind == 0) { /* PHX_REF_OWNED = 0 */
        ref_kind = 1; /* PHX_REF_BORROWED = 1 */
      }
    }
  }
  hir_deopt_sort_live_regs(instr_ptr);
}


void *phx_rc_model_reg(void *reg) {
  extern void *hir_model_reg_c(void *reg);
  return hir_model_reg_c(reg);
}

size_t phx_rc_get_rpo(void *func_ptr, void **out, size_t capacity) {
  void *cfg = hir_func_cfg_ptr(func_ptr);
  size_t c_count = hir_cfg_get_rpo_c(cfg, out, capacity);

#ifdef JIT_DCHECK_C
  auto& func = *static_cast<jit::hir::Function*>(func_ptr);
  auto cpp_rpo = func.cfg.GetRPOTraversal();
  JIT_DCHECK(c_count == cpp_rpo.size(),
      "RPO count mismatch: C={} C++={}", c_count, cpp_rpo.size());
  for (size_t i = 0; i < c_count && i < cpp_rpo.size(); i++) {
    JIT_DCHECK(out[i] == cpp_rpo[i],
        "RPO block mismatch at index {}: C={} C++={}", i, out[i], (void*)cpp_rpo[i]);
  }
#endif

  return c_count;
}

} /* extern "C" */
