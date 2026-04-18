/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * SSAify self-consistency invariants (R4 prerequisite).
 * 5 structural checks that survive the compiler.cpp flip.
 */

#include "cinderx/Jit/hir/ssaify_c.h"
#include "cinderx/Jit/hir/hir_basic_block_c.h"
#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/hir/dominator_c.h"
#include "cinderx/Common/log.h"

extern "C" void *hir_func_cfg(void *);
extern "C" void *hir_reg_instr(void *);

struct DomCtx {
  PhxDominatorState *dom;
  void *use_instr;
  int ok;
};

static int check_use_dominated(void **reg_slot, void *ctx_raw) {
  DomCtx *ctx = (DomCtx *)ctx_raw;
  void *reg = *reg_slot;
  if (!reg) return 1;
  void *def_instr = hir_reg_instr(reg);
  if (!def_instr) return 1;
  HirBasicBlock *def_block = (HirBasicBlock *)hir_c_block(def_instr);
  HirBasicBlock *use_block = (HirBasicBlock *)hir_c_block(ctx->use_instr);
  if (!def_block || !use_block) return 1;
  int def_id = hir_bb_id(def_block);
  int use_id = hir_bb_id(use_block);
  if (!phx_dom_dominates(ctx->dom, def_id, use_id)) {
    JIT_LOG("SSAify verify FAIL: use not dominated by definition");
    ctx->ok = 0;
    return 0;
  }
  return 1;
}

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

  /* Build dominator tree for invariant 2 */
  PhxDominatorState *dom = phx_dom_create(func_handle);

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
          phx_dom_destroy(dom);
          return 0;
        }
      }

      /* Invariant 2: Every use dominated by its definition */
      if (!hir_c_is_phi(instr)) {
        DomCtx ctx;
        ctx.dom = dom;
        ctx.use_instr = instr;
        ctx.ok = 1;
        hir_c_visit_uses(instr, check_use_dominated, &ctx);
        if (!ctx.ok) {
          phx_dom_destroy(dom);
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
          phx_dom_destroy(dom);
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
              phx_dom_destroy(dom);
              return 0;
            }
          }
        }
      }
    }
  }

  phx_dom_destroy(dom);
  return 1;
}

} /* extern "C" */
