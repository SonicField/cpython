/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Differential verification for C RPO traversal vs C++ GetRPOTraversal.
 */

#include "cinderx/Jit/hir/function.h"
#include "cinderx/Jit/hir/cfg.h"
#include "cinderx/Jit/hir/hir_cfg_rpo_c.h"
#include "cinderx/Common/log.h"

using namespace jit::hir;

extern "C" {

int hir_cfg_rpo_verify(void *func_handle) {
  Function* func = static_cast<Function*>(func_handle);
  std::vector<BasicBlock*> cpp_rpo = func->cfg.GetRPOTraversal();

  size_t n = cpp_rpo.size();
  void** c_rpo = new void*[n + 1];
  size_t c_count = hir_cfg_get_rpo_c(static_cast<void*>(&func->cfg), c_rpo, n + 1);

  int ok = 1;
  if (c_count != n) {
    JIT_LOG("RPO count mismatch: C={} C++={} in {}", c_count, n, func->fullname);
    ok = 0;
  } else {
    for (size_t i = 0; i < n; i++) {
      if (c_rpo[i] != static_cast<void*>(cpp_rpo[i])) {
        JIT_LOG("RPO order mismatch at index {}: C bb {} vs C++ bb {} in {}",
                i,
                static_cast<BasicBlock*>(c_rpo[i])->id,
                cpp_rpo[i]->id,
                func->fullname);
        ok = 0;
        break;
      }
    }
  }

  delete[] c_rpo;
  return ok;
}

} /* extern "C" */
