// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/Jit/hir/pass.h"
#include "cinderx/Jit/hir/refcount_env_c.h"
#include "cinderx/Jit/hir/refcount_pass_c.h"

extern "C" void hir_bind_guards_c(void *func);
extern "C" void hir_optimize_long_decref_runs_c(void *func);
extern "C" int hir_remove_trampoline_blocks_c(void *cfg);
extern "C" void hir_phi_elimination_run(void *func);

namespace jit::hir {

// Inserts incref/decref instructions.
class RefcountInsertion : public Pass {
 public:
  RefcountInsertion() : Pass("RefcountInsertion") {}

  void Run(Function& irfunc) override;

  static std::unique_ptr<RefcountInsertion> Factory() {
    return std::make_unique<RefcountInsertion>();
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(RefcountInsertion);
};

inline void RefcountInsertion::Run(Function& func) {
  hir_phi_elimination_run(&func);
  hir_bind_guards_c(&func);
  hir_cfg_split_critical_edges_c(&func);

  PhxRefcountEnv *c_env = phx_rc_env_create(static_cast<void*>(&func));
  phx_rc_run(c_env);
  phx_rc_env_destroy(c_env);

  hir_remove_trampoline_blocks_c(&func.cfg);
  hir_optimize_long_decref_runs_c(&func);
}

} // namespace jit::hir
