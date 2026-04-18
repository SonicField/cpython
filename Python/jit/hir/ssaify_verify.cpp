/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * SSAify self-consistency invariants (R4 prerequisite).
 * 5 structural checks that survive the compiler.cpp flip.
 */

#include "cinderx/Jit/hir/ssaify_c.h"
#include "cinderx/Jit/hir/hir_basic_block_c.h"
#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Common/log.h"

extern "C" void *hir_func_cfg(void *);
extern "C" void *hir_reg_instr(void *);

extern "C" {

int hir_ssaify_verify(void *func_handle) {
  HirCFG *cfg = (HirCFG *)hir_func_cfg(func_handle);
  HirBasicBlock *entry = (HirBasicBlock *)cfg->entry_block;

  /* Invariant 5: No phis in entry block */
  void *first = hir_bb_first_instr(entry);
  if (first && hir_c_is_phi(first)) {
    JIT_LOG("SSAify verify FAIL: entry block has phi");
    return 0;
  }

  for (HirBasicBlock *bb = hir_cfg_first_block(cfg); bb;
       bb = hir_cfg_next_block(cfg, bb)) {
    size_t num_preds = hir_bb_in_edges_count(bb);

    for (void *instr = hir_bb_first_instr(bb); instr;
         instr = hir_bb_next_instr(bb, instr)) {
      /* Invariant 1: Every output register has exactly one definition */
      void *output = hir_c_output(instr);
      if (output) {
        void *def_instr = hir_reg_instr(output);
        if (def_instr != instr) {
          JIT_LOG("SSAify verify FAIL: register has wrong defining instr");
          return 0;
        }
      }

      if (hir_c_is_phi(instr)) {
        size_t num_ops = hir_c_num_operands(instr);

        /* Invariant 3: Phi operand count == predecessor count */
        if (num_ops != num_preds) {
          JIT_LOG(
              "SSAify verify FAIL: phi has {} operands but block has {} preds",
              num_ops, num_preds);
          return 0;
        }

        /* Invariant 4: Each phi operand from a distinct predecessor */
        for (size_t i = 0; i < num_preds; i++) {
          const HirEdge *edge_i = hir_bb_in_edge(bb, i);
          HirBasicBlock *pred_i = (HirBasicBlock *)edge_i->from;
          for (size_t j = i + 1; j < num_preds; j++) {
            const HirEdge *edge_j = hir_bb_in_edge(bb, j);
            HirBasicBlock *pred_j = (HirBasicBlock *)edge_j->from;
            if (pred_i == pred_j) {
              JIT_LOG("SSAify verify FAIL: duplicate predecessor in phi");
              return 0;
            }
          }
        }
      }
    }
  }

  return 1;
}

} /* extern "C" */
