/*
 * hir_c_api.cpp -- Thin C wrapper implementations for HIR types.
 *
 * Each function casts the opaque void* handle to the C++ type and
 * calls the corresponding method. No logic lives here — this is
 * purely a translation layer.
 *
 * Only functions with active .c callers are implemented here.
 * Do NOT add speculative wrapper functions — convert the underlying
 * C++ to C instead (Phase 3D directive).
 */

#include "cinderx/Jit/hir/hir_c_api.h"
#include "cinderx/Jit/hir/hir_type_c.h"

#include "cinderx/Jit/hir/type.h"
#include "cinderx/Jit/hir/hir.h"
#include "cinderx/Jit/hir/cfg.h"
#include "cinderx/Jit/hir/function.h"
#include "cinderx/Jit/hir/instr_effects.h"
#include "cinderx/Jit/hir/pass.h"

#include <vector>

using namespace jit::hir;

/* Verify HirType is layout-compatible with C++ Type */
static_assert(sizeof(HirType) == sizeof(Type),
              "HirType size must match C++ Type");
static_assert(sizeof(HirType) == 16,
              "HirType must be 16 bytes");
static_assert(offsetof(HirType, pytype) == 8,
              "HirType spec union must be at offset 8");

/* ---- Type constants for hir_type_c.c ---- */

extern "C" {
const uint64_t _hir_type_kObject = Type::kObject;
const uint64_t _hir_type_kPrimitive = Type::kPrimitive;

int _hir_type_is_builtin_pytype(PyTypeObject *type) {
    /* A builtin type is one where fromType() produces an unspecialized
     * result (the bits already encode it uniquely). This matches the
     * pyTypeToType().contains() check in type.cpp:578. */
    Type t = Type::fromType(type);
    return !t.hasTypeSpec() ? 1 : 0;
}
} /* extern "C" */

/* ---- Cast helpers ---- */

static inline Function* as_func(HirFunction f) {
  return static_cast<Function*>(f);
}

static inline CFG* as_cfg(HirCFG c) {
  return static_cast<CFG*>(c);
}

static inline BasicBlock* as_block(HirBasicBlock b) {
  return static_cast<BasicBlock*>(b);
}

static inline Instr* as_instr(HirInstr i) {
  return static_cast<Instr*>(i);
}

static inline Register* as_reg(HirRegister r) {
  return static_cast<Register*>(r);
}

/* ---- Function / CFG ---- */

extern "C" {

HirCFG hir_func_cfg(HirFunction func) {
  return &as_func(func)->cfg;
}

size_t hir_cfg_get_rpo(HirCFG cfg, HirBasicBlock *out, size_t capacity) {
  auto rpo = as_cfg(cfg)->GetRPOTraversal();
  size_t count = rpo.size() < capacity ? rpo.size() : capacity;
  for (size_t i = 0; i < count; i++) {
    out[i] = rpo[i];
  }
  return count;
}

HirBasicBlock hir_cfg_blocks_first(HirCFG cfg) {
  auto& blocks = as_cfg(cfg)->blocks;
  if (blocks.begin() == blocks.end()) {
    return nullptr;
  }
  return &*blocks.begin();
}

HirBasicBlock hir_cfg_blocks_next(HirCFG cfg, HirBasicBlock block) {
  auto& blocks = as_cfg(cfg)->blocks;
  auto it = blocks.iterator_to(*as_block(block));
  ++it;
  if (it == blocks.end()) {
    return nullptr;
  }
  return &*it;
}

/* ---- BasicBlock ---- */

int hir_block_empty(HirBasicBlock block) {
  return as_block(block)->empty() ? 1 : 0;
}

HirInstr hir_block_first(HirBasicBlock block) {
  if (as_block(block)->empty()) {
    return nullptr;
  }
  return &as_block(block)->front();
}

HirInstr hir_block_next(HirBasicBlock block, HirInstr instr) {
  auto it = as_block(block)->iterator_to(*as_instr(instr));
  ++it;
  if (it == as_block(block)->end()) {
    return nullptr;
  }
  return &*it;
}

HirInstr hir_block_terminator(HirBasicBlock block) {
  return as_block(block)->GetTerminator();
}

HirInstr hir_block_append(HirBasicBlock block, HirInstr instr) {
  return as_block(block)->Append(as_instr(instr));
}

HirInstr hir_block_pop_front(HirBasicBlock block) {
  return as_block(block)->pop_front();
}

size_t hir_block_in_edges_count(HirBasicBlock block) {
  return as_block(block)->in_edges().size();
}

void hir_block_fixup_phis(HirBasicBlock block,
                          HirBasicBlock old_pred,
                          HirBasicBlock new_pred) {
  as_block(block)->fixupPhis(as_block(old_pred), as_block(new_pred));
}

/* ---- Instruction predicates ---- */

int hir_instr_is_terminator(HirInstr instr) {
  return as_instr(instr)->IsTerminator() ? 1 : 0;
}

int hir_instr_is_snapshot(HirInstr instr) {
  return as_instr(instr)->IsSnapshot() ? 1 : 0;
}

int hir_instr_is_phi(HirInstr instr) {
  return as_instr(instr)->IsPhi() ? 1 : 0;
}

int hir_instr_is_assign(HirInstr instr) {
  return as_instr(instr)->IsAssign() ? 1 : 0;
}

int hir_instr_is_primitive_box(HirInstr instr) {
  return as_instr(instr)->IsPrimitiveBox() ? 1 : 0;
}

int hir_instr_is_branch(HirInstr instr) {
  return as_instr(instr)->IsBranch() ? 1 : 0;
}

int hir_instr_has_deopt_base(HirInstr instr) {
  return as_instr(instr)->asDeoptBase() != nullptr ? 1 : 0;
}

/* ---- Instruction accessors ---- */

HirRegister hir_instr_output(HirInstr instr) {
  return as_instr(instr)->output();
}

size_t hir_instr_num_edges(HirInstr instr) {
  return as_instr(instr)->numEdges();
}

HirBasicBlock hir_instr_successor(HirInstr instr, size_t index) {
  return as_instr(instr)->successor(index);
}

/* ---- Instruction mutation ---- */

void hir_instr_unlink(HirInstr instr) {
  as_instr(instr)->unlink();
}

void hir_instr_insert_before(HirInstr instr, HirInstr before) {
  as_instr(instr)->InsertBefore(*as_instr(before));
}

void hir_instr_copy_bytecode_offset(HirInstr dst, HirInstr src) {
  as_instr(dst)->copyBytecodeOffset(*as_instr(src));
}

void hir_instr_delete(HirInstr instr) {
  delete as_instr(instr);
}

/* ---- Operand use visitation ---- */

void hir_instr_visit_uses(HirInstr instr,
                          int (*callback)(HirRegister *reg_slot, void *ctx),
                          void *ctx) {
  as_instr(instr)->visitUses([callback, ctx](Register*& reg) -> bool {
    /* The callback receives a pointer to the register pointer.
     * It can read *reg_slot to get the current register, or
     * write *reg_slot to replace the operand (copy propagation). */
    HirRegister reg_as_handle = static_cast<HirRegister>(reg);
    int result = callback(&reg_as_handle, ctx);
    /* If the callback changed the handle, write it back. */
    reg = static_cast<Register*>(reg_as_handle);
    return result != 0;
  });
}

/* ---- Branch-specific ---- */

HirBasicBlock hir_branch_target(HirInstr branch) {
  auto* br = dynamic_cast<Branch*>(as_instr(branch));
  if (br == nullptr) {
    return nullptr;
  }
  return br->target();
}

/* ---- Register accessors ---- */

HirInstr hir_reg_instr(HirRegister reg) {
  return as_reg(reg)->instr();
}

HirRegister hir_chase_assign(HirRegister reg) {
  return chaseAssignOperand(as_reg(reg));
}

/* ---- Phi-specific ---- */

HirRegister hir_phi_is_trivial(HirInstr phi) {
  auto* phi_instr = dynamic_cast<Phi*>(as_instr(phi));
  if (phi_instr == nullptr) {
    return nullptr;
  }
  return phi_instr->isTrivial();
}

/* ---- Factory functions ---- */

HirInstr hir_load_const_bottom_create(HirRegister output) {
  return LoadConst::create(as_reg(output), TBottom);
}

HirInstr hir_assign_create(HirRegister output, HirRegister value) {
  return Assign::create(as_reg(output), as_reg(value));
}

/* ---- Memory effects ---- */

int hir_memory_effects_may_store(HirInstr instr) {
  auto effects = memoryEffects(*as_instr(instr));
  return static_cast<int>(effects.may_store.bits());
}

/* ---- CFG / pass utilities ---- */

int hir_remove_trampoline_blocks(HirCFG cfg) {
  return removeTrampolineBlocks(as_cfg(cfg)) ? 1 : 0;
}

int hir_remove_unreachable_blocks(HirFunction func) {
  return removeUnreachableBlocks(*as_func(func)) ? 1 : 0;
}

void hir_remove_unreachable_instructions(HirFunction func) {
  removeUnreachableInstructions(*as_func(func));
}

void hir_reflow_types(HirFunction func) {
  reflowTypes(*as_func(func));
}

} /* extern "C" */
