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
#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/hir/hir_type_c.h"
#include "cinderx/Jit/hir/analysis.h"
#include "cinderx/Jit/bytecode_offsets.h"

#include "cinderx/Jit/hir/type.h"
#include "cinderx/Jit/hir/hir.h"
#include "cinderx/Jit/hir/cfg.h"
#include "cinderx/Jit/hir/function.h"
#include "cinderx/Jit/hir/instr_effects.h"
#include "cinderx/Jit/hir/pass.h"

#include <cstring>
#include <vector>

using namespace jit::hir;

/* Verify HirType is layout-compatible with C++ Type */
static_assert(sizeof(HirType) == sizeof(Type),
              "HirType size must match C++ Type");
static_assert(sizeof(HirType) == 16,
              "HirType must be 16 bytes");
static_assert(offsetof(HirType, pytype) == 8,
              "HirType spec union must be at offset 8");

/* ---- Cross-boundary layout verification ---- */
/* Runs during JIT init to catch C/C++ bit-packing divergence. */

static void verify_hir_type_layout() {
    /* Verify Type::toHirType conversion produces correct C HirType values */
    HirType c_type = Type::toHirType(TLong);
    assert(hir_type_bits(&c_type) == Type::kLong &&
           "C/C++ HirType bits_ divergence");
    assert(hir_type_lifetime(&c_type) == Type::kLifetimeTop &&
           "C/C++ HirType lifetime_ divergence");
    assert(hir_type_spec_kind(&c_type) == 0 /* kSpecTop */ &&
           "C/C++ HirType spec_kind_ divergence");

    /* Verify kSpecObject */
    HirType c_obj = Type::toHirType(Type::fromObject(Py_None));
    assert(hir_type_has_object_spec(&c_obj) &&
           "C/C++ HirType object spec divergence");
    assert(c_obj.pyobject == Py_None &&
           "C/C++ HirType object spec value divergence");

    /* Verify kSpecDouble */
    HirType c_dbl = Type::toHirType(Type::fromCDouble(3.14));
    assert(hir_type_has_double_spec(&c_dbl) &&
           "C/C++ HirType kSpecDouble divergence");
    assert(c_dbl.double_val == 3.14 &&
           "C/C++ HirType double value divergence");

    /* Verify kSpecInt */
    HirType c_int = Type::toHirType(Type::fromCInt(42, TCInt64));
    assert(hir_type_has_int_spec(&c_int) &&
           "C/C++ HirType kSpecInt divergence");
    assert(c_int.int_val == 42 &&
           "C/C++ HirType int value divergence");

    /* Verify TFloatExact round-trip (base type constant, kSpecTop) */
    HirType c_flt = Type::toHirType(TFloatExact);
    assert(hir_type_bits(&c_flt) == Type::kFloatExact &&
           "C/C++ HirType TFloatExact bits divergence");
    assert(hir_type_spec_kind(&c_flt) == 0 /* kSpecTop */ &&
           "C/C++ HirType TFloatExact spec divergence");

    /* Verify memcpy round-trip matches toHirType */
    HirType c_memcpy;
    Type t_flt = TFloatExact;
    memcpy(&c_memcpy, &t_flt, sizeof(c_memcpy));
    assert(c_memcpy.bits_and_flags == c_flt.bits_and_flags &&
           "memcpy vs toHirType bits_and_flags DIFFER — layout mismatch!");

    /* Critical test: toHirType → fromHirType round-trip.
     * If this fails, C factory types will be corrupted when read by C++. */
    HirType written = Type::toHirType(TFloatExact);
    Type via_from = Type::fromHirType(written);
    assert(via_from == TFloatExact &&
           "fromHirType round-trip fails for TFloatExact!");

    /* Also test TCDouble (non-specialized, just bits + lifetime) */
    HirType c_cdbl = Type::toHirType(TCDouble);
    Type cdbl_rt = Type::fromHirType(c_cdbl);
    assert(cdbl_rt == TCDouble &&
           "fromHirType round-trip fails for TCDouble!");
}

/* Run layout verification at program startup (before main) */
__attribute__((constructor))
static void hir_type_layout_check() {
    verify_hir_type_layout();
}

/* ---- Type constants for hir_type_c.c ---- */

extern "C" {
/* extern needed: C++ const has internal linkage by default,
 * but hir_type_c.c references these from a different TU. */
extern const uint64_t _hir_type_kObject = Type::kObject;
extern const uint64_t _hir_type_kPrimitive = Type::kPrimitive;

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

const char *hir_func_fullname(HirFunction func) {
  return as_func(func)->fullname.c_str();
}

HirRegister hir_func_alloc_register(HirFunction func) {
  return as_func(func)->env.AllocateRegister();
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

HirInstr hir_block_append_at(HirBasicBlock block, HirInstr instr, int32_t bc_off) {
  as_instr(instr)->setBytecodeOffset(jit::BCOffset{bc_off});
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

/* Instruction predicates moved to hir_instr_c.h as hir_c_is_* inline functions
 * (direct opcode field read, no C++ bridge needed). */

/* ---- Instruction accessors ---- */

/* hir_instr_output deleted — use hir_c_output from hir_instr_c.h */

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

/* hir_instr_copy_bytecode_offset moved to hir_instr_c.h as hir_c_copy_bytecode_offset */

void hir_instr_delete(HirInstr instr) {
  Instr::Destroy(as_instr(instr));
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

int hir_c_visit_deopt_extension(void *instr_ptr,
                                int (*callback)(void **reg_slot, void *ctx),
                                void *ctx) {
  auto wrap = [callback, ctx](Register*& reg) -> bool {
    void *reg_as_handle = static_cast<void *>(reg);
    int result = callback(&reg_as_handle, ctx);
    reg = static_cast<Register*>(reg_as_handle);
    return result != 0;
  };

  auto* i = as_instr(instr_ptr);
  if (i->opcode() == Opcode::kSnapshot) {
    auto* snap = static_cast<Snapshot*>(i);
    return snap->visitUses(wrap) ? 1 : 0;
  }

  auto* db = i->asDeoptBase();
  if (db) {
    return db->visitUsesDeopt(wrap) ? 1 : 0;
  }

  return 1;
}

/* ---- Branch-specific ---- */

HirBasicBlock hir_branch_target(HirInstr branch) {
  auto* instr = as_instr(branch);
  if (!instr->IsBranch()) {
    return nullptr;
  }
  return static_cast<Branch*>(instr)->target();
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
  auto* instr = as_instr(phi);
  if (!instr->IsPhi()) {
    return nullptr;
  }
  return static_cast<Phi*>(instr)->isTrivial();
}

/* ---- Factory functions ---- */

HirInstr hir_load_const_bottom_create(HirRegister output) {
  HirType bottom = {0, 0};  /* TBottom = zero bits */
  return hir_c_create_load_const(output, bottom);
}

HirInstr hir_assign_create(HirRegister output, HirRegister value) {
  HirAssign *a = (HirAssign *)hir_c_alloc_instr(sizeof(HirAssign), 1);
  hir_c_init_instr(a, HIR_OP_Assign);
  hir_c_set_output(a, output);
  hir_c_set_operand(a, 0, value);
  return a;
}

/* ---- Type queries (bridge) ---- */

/* hir_type_as_object() — moved to hir_type_c.c (pure C) */

int hir_type_is_exact(const HirType *t) {
  Type type = Type::fromHirType(*t);
  return type.isExact() ? 1 : 0;
}

int hir_type_has_known_destructor(const HirType *t) {
  Type type = Type::fromHirType(*t);
  return type.runtimePyTypeDestructor().has_value() ? 1 : 0;
}

PyTypeObject *hir_type_runtime_py_type(const HirType *t) {
  Type type = Type::fromHirType(*t);
  return type.runtimePyType();
}

/* ---- Analysis utilities (T2-D Tier 1) ---- */

int hir_is_passthrough(HirInstr instr) {
  return isPassthrough(*as_instr(instr)) ? 1 : 0;
}

int hir_operands_must_match(HirInstr instr, size_t operand_idx) {
  OperandType op_type = as_instr(instr)->GetOperandType(operand_idx);
  return operandsMustMatch(op_type) ? 1 : 0;
}

int hir_register_type_matches_operand(HirInstr instr, size_t operand_idx, HirRegister reg) {
  OperandType expected = as_instr(instr)->GetOperandType(operand_idx);
  Type reg_type = as_reg(reg)->type();
  return registerTypeMatches(reg_type, expected) ? 1 : 0;
}

int hir_type_matches_operand(HirInstr instr, size_t operand_idx,
                             const HirType *type) {
  OperandType expected = as_instr(instr)->GetOperandType(operand_idx);
  Type cpp_type = Type::fromHirType(*type);
  return registerTypeMatches(cpp_type, expected) ? 1 : 0;
}

/* ---- Instruction count ---- */

size_t hir_instr_num_operands(HirInstr instr) {
  return as_instr(instr)->NumOperands();
}

/* ---- Register type ---- */

HirType hir_register_type(HirRegister reg) {
  Type cpp_type = as_reg(reg)->type();
  HirType result;
  memcpy(&result, &cpp_type, sizeof(HirType));
  return result;
}

/* hir_instr_is_guard_type deleted — use hir_c_is_guard_type from hir_instr_c.h */

/* ---- RegUses (opaque handle) ---- */

HirRegUses hir_collect_reg_uses(HirFunction func) {
  auto* uses = new RegUses(collectDirectRegUses(*as_func(func)));
  return static_cast<HirRegUses>(uses);
}

int hir_reg_uses_contains(HirRegUses uses, HirRegister reg) {
  auto* map = static_cast<RegUses*>(uses);
  return map->find(as_reg(reg)) != map->end() ? 1 : 0;
}

size_t hir_reg_uses_count(HirRegUses uses, HirRegister reg) {
  auto* map = static_cast<RegUses*>(uses);
  auto it = map->find(as_reg(reg));
  if (it == map->end()) return 0;
  return it->second.size();
}

HirInstr hir_reg_uses_get(HirRegUses uses, HirRegister reg, size_t idx) {
  auto* map = static_cast<RegUses*>(uses);
  auto it = map->find(as_reg(reg));
  if (it == map->end()) return nullptr;
  auto& set = it->second;
  if (idx >= set.size()) return nullptr;
  auto sit = set.begin();
  for (size_t i = 0; i < idx; i++) ++sit;
  return *sit;
}

void hir_reg_uses_destroy(HirRegUses uses) {
  delete static_cast<RegUses*>(uses);
}

/* ---- outputType with override ---- */

HirType hir_output_type_with_override(HirInstr instr,
                                      size_t override_idx,
                                      const HirType *override_type) {
  Type cpp_override = Type::fromHirType(*override_type);
  Instr* i = as_instr(instr);
  Type result = outputType(*i, [&](std::size_t ind) -> Type {
    if (ind == override_idx) {
      return cpp_override;
    }
    return i->GetOperand(ind)->type();
  });
  HirType c_result;
  memcpy(&c_result, &result, sizeof(HirType));
  return c_result;
}

/* hir_instr_opname deleted — use hir_instr_info_name(hir_c_opcode(instr)) */

/* ---- Type to string ---- */

size_t hir_type_to_string(const HirType *type, char *buf, size_t bufsz,
                          int safe) {
  Type cpp_type = Type::fromHirType(*type);
  std::string s = safe ? cpp_type.toStringSafe() : cpp_type.toString();
  size_t len = s.size();
  if (buf && bufsz > 0) {
    size_t copy = len < bufsz ? len : bufsz - 1;
    memcpy(buf, s.data(), copy);
    buf[copy] = '\0';
  }
  return len;
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

/* T2-D predicates (is_condbranch, is_istruthy, is_compare, is_vectorcall,
 * opcode) deleted — moved to hir_instr_c.h as hir_c_* inline functions. */

/* ---- Instruction query/mutation (T2-D) ---- */

int hir_instr_compare_op(HirInstr instr) {
  auto* i = as_instr(instr);
  if (i->IsCompare()) {
    return static_cast<int>(static_cast<const Compare*>(i)->op());
  }
  if (i->IsCompareBool()) {
    return static_cast<int>(static_cast<const CompareBool*>(i)->op());
  }
  return -1;
}

int hir_instr_is_replayable(HirInstr instr) {
  return as_instr(instr)->isReplayable() ? 1 : 0;
}

int hir_instr_uses_reg(HirInstr instr, HirRegister reg) {
  return as_instr(instr)->Uses(as_reg(reg)) ? 1 : 0;
}

void hir_instr_replace_with(HirInstr old_instr, HirInstr new_instr) {
  as_instr(old_instr)->ReplaceWith(*as_instr(new_instr));
}

HirInstr hir_block_back(HirBasicBlock block) {
  auto* bb = as_block(block);
  if (bb->empty()) return nullptr;
  return &bb->back();
}

/* hir_instr_block deleted — use hir_c_block from hir_instr_c.h */

HirRegister hir_instr_get_operand(HirInstr instr, size_t i) {
  return as_instr(instr)->GetOperand(i);
}

/* ---- Factory functions (T2-D) ---- */

HirInstr hir_compare_bool_create(
    HirRegister output, int compare_op,
    HirRegister left, HirRegister right,
    HirInstr frame_state_source) {
  auto* fs_instr = as_instr(frame_state_source);
  const FrameState* fs = get_frame_state(*fs_instr);
  if (fs == nullptr) return nullptr;
  HirCompareBool *c = (HirCompareBool *)hir_c_alloc_instr(sizeof(HirCompareBool), 2);
  hir_c_init_deopt(c, HIR_OP_CompareBool);
  c->op = compare_op;
  hir_c_set_output(c, output);
  hir_c_set_operand(c, 0, left);
  hir_c_set_operand(c, 1, right);
  as_instr(c)->asDeoptBase()->setFrameState(*fs);
  return c;
}

HirInstr hir_c_create_binary_op(HirFunction func, int32_t op_kind,
                                HirRegister left, HirRegister right,
                                void *frame_state) {
  if (!frame_state) return nullptr;
  HirRegister dst = hir_func_alloc_register(func);
  HirBinaryOp *b = (HirBinaryOp *)hir_c_alloc_instr(sizeof(HirBinaryOp), 2);
  hir_c_init_deopt(b, HIR_OP_BinaryOp);
  b->op = op_kind;
  hir_c_set_output(b, dst);
  hir_c_set_operand(b, 0, left);
  hir_c_set_operand(b, 1, right);
  as_instr(b)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return b;
}

HirInstr hir_c_create_guard_type(HirFunction func, HirType target,
                                 HirRegister src, void *frame_state) {
  HirRegister dst = hir_func_alloc_register(func);
  HirGuardType *g = (HirGuardType *)hir_c_alloc_instr(sizeof(HirGuardType), 1);
  hir_c_init_deopt(g, HIR_OP_GuardType);
  g->target = target;
  hir_c_set_output(g, dst);
  hir_c_set_operand(g, 0, src);
  if (frame_state)
    as_instr(g)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return g;
}

HirInstr hir_c_create_check_exc(HirFunction func, HirRegister src,
                                void *frame_state) {
  if (!frame_state) return nullptr;
  HirRegister dst = hir_func_alloc_register(func);
  HirCheckExc *c = (HirCheckExc *)hir_c_alloc_instr(sizeof(HirCheckExc), 1);
  hir_c_init_deopt(c, HIR_OP_CheckExc);
  hir_c_set_output(c, dst);
  hir_c_set_operand(c, 0, src);
  as_instr(c)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return c;
}

/* ---- LoadField / GuardIs / CheckNeg / PrimitiveBox / CheckSequenceBounds ---- */

HirInstr hir_c_create_load_field(HirFunction func, HirRegister receiver,
                                  const char *name, intptr_t offset,
                                  HirType type, int borrowed) {
  HirRegister dst = hir_func_alloc_register(func);
  HirLoadField *l = (HirLoadField *)hir_c_alloc_instr(sizeof(HirLoadField), 1);
  hir_c_init_instr(l, HIR_OP_LoadField);
  new (&l->name_storage) std::string(name);
  l->offset = (size_t)offset;
  l->type = type;
  l->borrowed = (uint8_t)(borrowed != 0);
  hir_c_set_output(l, dst);
  hir_c_set_operand(l, 0, receiver);
  return l;
}

HirInstr hir_c_create_guard_is(HirFunction func, void *target,
                                HirRegister src) {
  HirRegister dst = hir_func_alloc_register(func);
  HirGuardIs *g = (HirGuardIs *)hir_c_alloc_instr(sizeof(HirGuardIs), 1);
  hir_c_init_deopt(g, HIR_OP_GuardIs);
  g->target = target;
  hir_c_set_output(g, dst);
  hir_c_set_operand(g, 0, src);
  return g;
}

HirInstr hir_c_create_check_neg(HirFunction func, HirRegister src,
                                 void *frame_state) {
  if (!frame_state) return nullptr;
  HirRegister dst = hir_func_alloc_register(func);
  HirCheckNeg *c = (HirCheckNeg *)hir_c_alloc_instr(sizeof(HirCheckNeg), 1);
  hir_c_init_deopt(c, HIR_OP_CheckNeg);
  hir_c_set_output(c, dst);
  hir_c_set_operand(c, 0, src);
  as_instr(c)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return (HirInstr)c;
}

HirInstr hir_c_create_primitive_box(HirFunction func, HirRegister src,
                                     HirType type, void *frame_state) {
  if (!frame_state) return nullptr;
  HirRegister dst = hir_func_alloc_register(func);
  HirPrimitiveBox *p = (HirPrimitiveBox *)hir_c_alloc_instr(sizeof(HirPrimitiveBox), 1);
  hir_c_init_deopt(p, HIR_OP_PrimitiveBox);
  p->type = type;
  hir_c_set_output(p, dst);
  hir_c_set_operand(p, 0, src);
  as_instr(p)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return p;
}

HirInstr hir_c_create_check_seq_bounds(HirFunction func,
                                        HirRegister seq, HirRegister idx,
                                        void *frame_state) {
  if (!frame_state) return nullptr;
  HirRegister dst = hir_func_alloc_register(func);
  HirDeoptLayout *c = (HirDeoptLayout *)hir_c_alloc_instr(sizeof(HirDeoptLayout), 2);
  hir_c_init_deopt(c, HIR_OP_CheckSequenceBounds);
  hir_c_set_output(c, dst);
  hir_c_set_operand(c, 0, seq);
  hir_c_set_operand(c, 1, idx);
  as_instr(c)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return c;
}

/* ---- DeoptBaseWithNameIdx factories ---- */

#define DEOPT_NAMEIDX1_FACTORY(name, CppType, opcode) \
HirInstr hir_c_create_##name(HirFunction func, \
    HirRegister receiver, int name_idx, void *frame_state) { \
  if (!frame_state) return nullptr; \
  HirRegister dst = hir_func_alloc_register(func); \
  HirLoadAttrCached *i = (HirLoadAttrCached *)hir_c_alloc_instr(sizeof(HirLoadAttrCached), 1); \
  hir_c_init_deopt(i, opcode); \
  i->name_idx = name_idx; \
  hir_c_set_output(i, dst); \
  hir_c_set_operand(i, 0, receiver); \
  as_instr(i)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state)); \
  return i; \
}

DEOPT_NAMEIDX1_FACTORY(load_module_method_cached, LoadModuleMethodCached, HIR_OP_LoadModuleMethodCached)
DEOPT_NAMEIDX1_FACTORY(load_method_cached, LoadMethodCached, HIR_OP_LoadMethodCached)
DEOPT_NAMEIDX1_FACTORY(load_module_attr_cached, LoadModuleAttrCached, HIR_OP_LoadModuleAttrCached)
DEOPT_NAMEIDX1_FACTORY(load_attr_cached, LoadAttrCached, HIR_OP_LoadAttrCached)
#undef DEOPT_NAMEIDX1_FACTORY

HirInstr hir_c_create_store_attr_cached(HirFunction func,
    HirRegister obj, HirRegister value, int name_idx, void *frame_state) {
  if (!frame_state) return nullptr;
  HirStoreAttrCached *s = (HirStoreAttrCached *)hir_c_alloc_instr(sizeof(HirStoreAttrCached), 2);
  hir_c_init_deopt(s, HIR_OP_StoreAttrCached);
  s->name_idx = name_idx;
  hir_c_set_operand(s, 0, obj);
  hir_c_set_operand(s, 1, value);
  as_instr(s)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return s;
}

/* ---- Tier 3: CheckField + LoadAttr + LoadArrayItem + setGuiltyReg ---- */

HirInstr hir_c_create_check_field(HirFunction func, HirRegister src,
    void *name, void *frame_state) {
  if (!frame_state) return nullptr;
  HirRegister dst = hir_func_alloc_register(func);
  HirCheckField *c = (HirCheckField *)hir_c_alloc_instr(sizeof(HirCheckField), 1);
  hir_c_init_deopt(c, HIR_OP_CheckField);
  c->name = name;
  hir_c_set_output(c, dst);
  hir_c_set_operand(c, 0, src);
  as_instr(c)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return c;
}

HirInstr hir_c_create_load_attr(HirFunction func, HirRegister receiver,
    int name_idx, void *frame_state, int already_optimized) {
  if (!frame_state) return nullptr;
  HirRegister dst = hir_func_alloc_register(func);
  HirLoadAttr *la = (HirLoadAttr *)hir_c_alloc_instr(sizeof(HirLoadAttr), 1);
  hir_c_init_deopt(la, HIR_OP_LoadAttr);
  la->name_idx = name_idx;
  la->already_optimized = (uint8_t)(already_optimized != 0);
  hir_c_set_output(la, dst);
  hir_c_set_operand(la, 0, receiver);
  as_instr(la)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return la;
}

HirInstr hir_c_create_load_array_item(HirFunction func,
    HirRegister arr, HirRegister idx, HirRegister container,
    intptr_t offset, HirType type) {
  HirRegister dst = hir_func_alloc_register(func);
  HirLoadArrayItem *l = (HirLoadArrayItem *)hir_c_alloc_instr(sizeof(HirLoadArrayItem), 3);
  hir_c_init_instr(l, HIR_OP_LoadArrayItem);
  l->offset = (intptr_t)offset;
  l->type = type;
  hir_c_set_output(l, dst);
  hir_c_set_operand(l, 0, arr);
  hir_c_set_operand(l, 1, idx);
  hir_c_set_operand(l, 2, container);
  return l;
}

void hir_c_set_guilty_reg(HirInstr instr, HirRegister reg) {
  static_cast<DeoptBase*>(as_instr(instr))->setGuiltyReg(as_reg(reg));
}

void hir_c_set_descr(HirInstr instr, const char *descr) {
  static_cast<DeoptBase*>(as_instr(instr))->setDescr(std::string(descr));
}

HirInstr hir_c_create_guard(HirRegister src) {
  HirDeoptLayout *g = (HirDeoptLayout *)hir_c_alloc_instr(sizeof(HirDeoptLayout), 1);
  hir_c_init_deopt(g, HIR_OP_Guard);
  hir_c_set_operand(g, 0, src);
  return g;
}

/* ---- Tier 5: Variable-arity + infrastructure factories ---- */

HirInstr hir_c_create_vectorcall(HirFunction func, size_t n_operands,
                                  uint32_t flags, void *frame_state) {
  HirRegister dst = hir_func_alloc_register(func);
  HirVectorCall *v = (HirVectorCall *)hir_c_alloc_instr(sizeof(HirVectorCall), n_operands);
  hir_c_init_deopt(v, HIR_OP_VectorCall);
  v->flags = flags;
  hir_c_set_output(v, dst);
  if (frame_state)
    as_instr(v)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return v;
}

HirInstr hir_c_create_call_static(HirFunction func, size_t n_operands,
                                   void *addr, HirType ret_type) {
  HirRegister dst = hir_func_alloc_register(func);
  HirCallStatic *c = (HirCallStatic *)hir_c_alloc_instr(sizeof(HirCallStatic), n_operands);
  hir_c_init_instr(c, HIR_OP_CallStatic);
  c->addr = addr;
  c->ret_type = ret_type;
  hir_c_set_output(c, dst);
  return c;
}

HirInstr hir_c_create_call_static_reg(size_t n_operands, HirRegister dst,
                                       void *addr, HirType ret_type) {
  HirCallStatic *c = (HirCallStatic *)hir_c_alloc_instr(sizeof(HirCallStatic), n_operands);
  hir_c_init_instr(c, HIR_OP_CallStatic);
  c->addr = addr;
  c->ret_type = ret_type;
  hir_c_set_output(c, dst);
  return c;
}

HirInstr hir_c_create_deopt_patchpoint(void *patcher) {
  HirDeoptPatchpoint *d = (HirDeoptPatchpoint *)hir_c_alloc_instr(sizeof(HirDeoptPatchpoint), 0);
  hir_c_init_deopt(d, HIR_OP_DeoptPatchpoint);
  d->patcher = patcher;
  return d;
}

HirInstr hir_c_create_snapshot(void *frame_state) {
  if (!frame_state) return nullptr;
  HirSnapshot *s = (HirSnapshot *)hir_c_alloc_instr(sizeof(HirSnapshot), 0);
  hir_c_init_instr(s, HIR_OP_Snapshot);
  /* Snapshot stores FrameState as a unique_ptr — use C++ bridge */
  auto fs_copy = std::make_unique<FrameState>(*static_cast<const FrameState*>(frame_state));
  s->frame_state_ptr = fs_copy.release();
  return s;
}

void hir_c_set_suppress_exc_deopt(HirInstr instr, int val) {
  static_cast<DeoptBase*>(as_instr(instr))
      ->setSuppressExceptionDeopt(val != 0);
}

void hir_c_set_output_type(HirInstr instr, HirType type) {
  Type cpp_type = Type::fromHirType(type);
  as_instr(instr)->output()->set_type(cpp_type);
}

/* ---- FillTypeAttrCache / FillTypeMethodCache ---- */

HirInstr hir_c_create_fill_type_attr_cache(HirFunction func,
    HirRegister receiver, int name_idx, int cache_id, void *frame_state) {
  if (!frame_state) return nullptr;
  HirRegister dst = hir_func_alloc_register(func);
  HirFillTypeAttrCache *f = (HirFillTypeAttrCache *)hir_c_alloc_instr(sizeof(HirFillTypeAttrCache), 1);
  hir_c_init_deopt(f, HIR_OP_FillTypeAttrCache);
  f->name_idx = name_idx;
  f->cache_id = cache_id;
  hir_c_set_output(f, dst);
  hir_c_set_operand(f, 0, receiver);
  as_instr(f)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return f;
}

HirInstr hir_c_create_fill_type_method_cache(HirFunction func,
    HirRegister receiver, int name_idx, int cache_id, void *frame_state) {
  if (!frame_state) return nullptr;
  HirRegister dst = hir_func_alloc_register(func);
  HirFillTypeMethodCache *f = (HirFillTypeMethodCache *)hir_c_alloc_instr(sizeof(HirFillTypeMethodCache), 1);
  hir_c_init_deopt(f, HIR_OP_FillTypeMethodCache);
  f->name_idx = name_idx;
  f->cache_id = cache_id;
  hir_c_set_output(f, dst);
  hir_c_set_operand(f, 0, receiver);
  as_instr(f)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return f;
}

/* ---- Simple DeoptBase factories (2-operand + FrameState, no custom fields) ---- */

#define DEOPT2_FACTORY(name, opcode) \
HirInstr hir_c_create_##name(HirFunction func, HirRegister lhs, \
                              HirRegister rhs, void *frame_state) { \
  if (!frame_state) return nullptr; \
  HirRegister dst = hir_func_alloc_register(func); \
  HirDeoptLayout *i = (HirDeoptLayout *)hir_c_alloc_instr(sizeof(HirDeoptLayout), 2); \
  hir_c_init_deopt(i, opcode); \
  hir_c_set_output(i, dst); \
  hir_c_set_operand(i, 0, lhs); \
  hir_c_set_operand(i, 1, rhs); \
  as_instr(i)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state)); \
  return i; \
}

DEOPT2_FACTORY(dict_subscr, HIR_OP_DictSubscr)
DEOPT2_FACTORY(unicode_subscr, HIR_OP_UnicodeSubscr)
DEOPT2_FACTORY(unicode_repeat, HIR_OP_UnicodeRepeat)
DEOPT2_FACTORY(unicode_concat, HIR_OP_UnicodeConcat)
DEOPT2_FACTORY(list_append, HIR_OP_ListAppend)
DEOPT2_FACTORY(is_instance, HIR_OP_IsInstance)
#undef DEOPT2_FACTORY

HirInstr hir_c_create_get_length(HirFunction func, HirRegister src,
                                  void *frame_state) {
  if (!frame_state) return nullptr;
  HirRegister dst = hir_func_alloc_register(func);
  HirGetLength *g = (HirGetLength *)hir_c_alloc_instr(sizeof(HirGetLength), 1);
  hir_c_init_deopt(g, HIR_OP_GetLength);
  hir_c_set_output(g, dst);
  hir_c_set_operand(g, 0, src);
  as_instr(g)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return g;
}

HirInstr hir_c_create_long_in_place_op(HirFunction func, int32_t op_kind,
                                        HirRegister left, HirRegister right,
                                        void *frame_state) {
  if (!frame_state) return nullptr;
  HirRegister dst = hir_func_alloc_register(func);
  HirLongInPlaceOp *l = (HirLongInPlaceOp *)hir_c_alloc_instr(sizeof(HirLongInPlaceOp), 2);
  hir_c_init_deopt(l, HIR_OP_LongInPlaceOp);
  l->op = op_kind;
  hir_c_set_output(l, dst);
  hir_c_set_operand(l, 0, left);
  hir_c_set_operand(l, 1, right);
  as_instr(l)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return l;
}

/* ---- FloatBinaryOp / LongBinaryOp / IsNegativeAndErrOccurred factories ---- */

HirInstr hir_c_create_float_binary_op(HirFunction func, int32_t op_kind,
                                       HirRegister left, HirRegister right,
                                       void *frame_state) {
  if (!frame_state) return nullptr;
  HirRegister dst = hir_func_alloc_register(func);
  HirFloatBinaryOp *b = (HirFloatBinaryOp *)hir_c_alloc_instr(sizeof(HirFloatBinaryOp), 2);
  hir_c_init_deopt(b, HIR_OP_FloatBinaryOp);
  b->op = op_kind;
  hir_c_set_output(b, dst);
  hir_c_set_operand(b, 0, left);
  hir_c_set_operand(b, 1, right);
  as_instr(b)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return b;
}

HirInstr hir_c_create_long_binary_op(HirFunction func, int32_t op_kind,
                                      HirRegister left, HirRegister right,
                                      void *frame_state) {
  if (!frame_state) return nullptr;
  HirRegister dst = hir_func_alloc_register(func);
  HirLongBinaryOp *b = (HirLongBinaryOp *)hir_c_alloc_instr(sizeof(HirLongBinaryOp), 2);
  hir_c_init_deopt(b, HIR_OP_LongBinaryOp);
  b->op = op_kind;
  hir_c_set_output(b, dst);
  hir_c_set_operand(b, 0, left);
  hir_c_set_operand(b, 1, right);
  as_instr(b)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return b;
}

HirInstr hir_c_create_is_neg_and_err(HirFunction func, HirRegister src,
                                      void *frame_state) {
  if (!frame_state) return nullptr;
  auto* f = static_cast<Function*>(func);
  auto* dst = f->env.AllocateRegister();
  HirIsNegativeAndErrOccurred *i = (HirIsNegativeAndErrOccurred *)hir_c_alloc_instr(sizeof(HirIsNegativeAndErrOccurred), 1);
  hir_c_init_deopt(i, HIR_OP_IsNegativeAndErrOccurred);
  hir_c_set_output(i, dst);
  hir_c_set_operand(i, 0, src);
  as_instr(i)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return i;
}

/* ---- Branch/CondBranch factories (C++ bridge for Edge management) ---- */

HirInstr hir_c_create_branch_cpp(void *target_block) {
  auto* bb = static_cast<BasicBlock*>(target_block);
  return Branch::create(bb);
}

HirInstr hir_c_create_check_seq_bounds_reg(HirRegister dst, HirRegister seq,
                                            HirRegister idx, void *frame_state) {
  HirDeoptLayout *c = (HirDeoptLayout *)hir_c_alloc_instr(sizeof(HirDeoptLayout), 2);
  hir_c_init_deopt(c, HIR_OP_CheckSequenceBounds);
  hir_c_set_output(c, dst);
  hir_c_set_operand(c, 0, seq);
  hir_c_set_operand(c, 1, idx);
  if (frame_state)
    as_instr(c)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return c;
}

HirInstr hir_c_create_check_field_reg(HirRegister dst, HirRegister src,
                                       void *name, void *frame_state) {
  HirCheckField *c = (HirCheckField *)hir_c_alloc_instr(sizeof(HirCheckField), 1);
  hir_c_init_deopt(c, HIR_OP_CheckField);
  c->name = name;
  hir_c_set_output(c, dst);
  hir_c_set_operand(c, 0, src);
  if (frame_state)
    as_instr(c)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return c;
}

HirInstr hir_c_create_binary_op_reg(HirRegister dst, int32_t op_kind,
                                     HirRegister left, HirRegister right,
                                     void *frame_state) {
  HirBinaryOp *b = (HirBinaryOp *)hir_c_alloc_instr(sizeof(HirBinaryOp), 2);
  hir_c_init_deopt(b, HIR_OP_BinaryOp);
  b->op = op_kind;
  hir_c_set_output(b, dst);
  hir_c_set_operand(b, 0, left);
  hir_c_set_operand(b, 1, right);
  as_instr(b)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return b;
}

HirInstr hir_c_create_guard_is_reg(HirRegister dst, void *target,
                                    HirRegister src) {
  HirGuardIs *g = (HirGuardIs *)hir_c_alloc_instr(sizeof(HirGuardIs), 1);
  hir_c_init_deopt(g, HIR_OP_GuardIs);
  g->target = target;
  hir_c_set_output(g, dst);
  hir_c_set_operand(g, 0, src);
  return g;
}

HirInstr hir_c_create_set_current_awaiter_reg(HirRegister src) {
  HirSetCurrentAwaiter *i = (HirSetCurrentAwaiter *)hir_c_alloc_instr(sizeof(HirSetCurrentAwaiter), 1);
  hir_c_init_instr(i, HIR_OP_SetCurrentAwaiter);
  hir_c_set_operand(i, 0, src);
  return i;
}

HirInstr hir_c_create_decref_reg(HirRegister src) {
  HirDecref *d = (HirDecref *)hir_c_alloc_instr(sizeof(HirDecref), 1);
  hir_c_init_instr(d, HIR_OP_Decref);
  hir_c_set_operand(d, 0, src);
  return d;
}

HirInstr hir_c_create_make_cell_reg(HirRegister dst, HirRegister src,
                                     void *frame_state) {
  HirMakeCell *c = (HirMakeCell *)hir_c_alloc_instr(sizeof(HirMakeCell), 1);
  hir_c_init_deopt(c, HIR_OP_MakeCell);
  hir_c_set_output(c, dst);
  hir_c_set_operand(c, 0, src);
  as_instr(c)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return c;
}

HirInstr hir_c_create_initial_yield_reg(HirRegister dst, void *frame_state) {
  HirInitialYield *y = (HirInitialYield *)hir_c_alloc_instr(sizeof(HirInitialYield), 0);
  hir_c_init_deopt(y, HIR_OP_InitialYield);
  hir_c_set_output(y, dst);
  as_instr(y)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return y;
}

HirInstr hir_c_create_load_arg_reg(HirRegister dst, int32_t idx, HirType type) {
  HirLoadArg *la = (HirLoadArg *)hir_c_alloc_instr(sizeof(HirLoadArg), 0);
  hir_c_init_instr(la, HIR_OP_LoadArg);
  la->arg_idx = (uint32_t)idx;
  la->type = type;
  hir_c_set_output(la, dst);
  return la;
}

HirInstr hir_c_create_store_subscr_reg(HirRegister container, HirRegister sub,
                                        HirRegister value, void *frame_state) {
  HirStoreSubscr *s = (HirStoreSubscr *)hir_c_alloc_instr(sizeof(HirStoreSubscr), 3);
  hir_c_init_deopt(s, HIR_OP_StoreSubscr);
  hir_c_set_operand(s, 0, container);
  hir_c_set_operand(s, 1, sub);
  hir_c_set_operand(s, 2, value);
  as_instr(s)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return s;
}

HirInstr hir_c_create_set_set_item_reg(HirRegister dst, HirRegister set,
                                        HirRegister item, void *frame_state) {
  HirSetSetItem *s = (HirSetSetItem *)hir_c_alloc_instr(sizeof(HirSetSetItem), 2);
  hir_c_init_deopt(s, HIR_OP_SetSetItem);
  hir_c_set_output(s, dst);
  hir_c_set_operand(s, 0, set);
  hir_c_set_operand(s, 1, item);
  as_instr(s)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return s;
}

HirInstr hir_c_create_in_place_op_reg(HirRegister dst, int32_t op_kind,
                                       HirRegister left, HirRegister right,
                                       void *frame_state) {
  HirInPlaceOp *ip = (HirInPlaceOp *)hir_c_alloc_instr(sizeof(HirInPlaceOp), 2);
  hir_c_init_deopt(ip, HIR_OP_InPlaceOp);
  ip->op = op_kind;
  hir_c_set_output(ip, dst);
  hir_c_set_operand(ip, 0, left);
  hir_c_set_operand(ip, 1, right);
  as_instr(ip)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return ip;
}

HirInstr hir_c_create_compare_reg(HirRegister dst, int32_t op,
                                   HirRegister left, HirRegister right,
                                   void *frame_state) {
  HirCompare *c = (HirCompare *)hir_c_alloc_instr(sizeof(HirCompare), 2);
  hir_c_init_deopt(c, HIR_OP_Compare);
  c->op = op;
  hir_c_set_output(c, dst);
  hir_c_set_operand(c, 0, left);
  hir_c_set_operand(c, 1, right);
  as_instr(c)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return c;
}

HirInstr hir_c_create_format_with_spec_reg(HirRegister dst, HirRegister value,
                                            HirRegister fmt_spec, void *frame_state) {
  /* FormatWithSpec: DeoptBase, HasOutput, Operands<2>, no custom fields */
  HirDeoptLayout *f = (HirDeoptLayout *)hir_c_alloc_instr(sizeof(HirDeoptLayout), 2);
  hir_c_init_deopt(f, HIR_OP_FormatWithSpec);
  hir_c_set_output(f, dst);
  hir_c_set_operand(f, 0, value);
  hir_c_set_operand(f, 1, fmt_spec);
  as_instr(f)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return f;
}

HirInstr hir_c_create_make_dict_reg(HirRegister dst, int32_t dict_size,
                                     void *frame_state) {
  HirMakeDict *d = (HirMakeDict *)hir_c_alloc_instr(sizeof(HirMakeDict), 0);
  hir_c_init_deopt(d, HIR_OP_MakeDict);
  d->capacity = (size_t)dict_size;
  hir_c_set_output(d, dst);
  as_instr(d)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return d;
}

HirInstr hir_c_create_get_a_iter_reg(HirRegister dst, HirRegister src, void *fs) {
  HirGetAIter *g = (HirGetAIter *)hir_c_alloc_instr(sizeof(HirGetAIter), 1);
  hir_c_init_deopt(g, HIR_OP_GetAIter);
  hir_c_set_output(g, dst);
  hir_c_set_operand(g, 0, src);
  as_instr(g)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return g;
}
HirInstr hir_c_create_get_a_next_reg(HirRegister dst, HirRegister src, void *fs) {
  HirGetANext *g = (HirGetANext *)hir_c_alloc_instr(sizeof(HirGetANext), 1);
  hir_c_init_deopt(g, HIR_OP_GetANext);
  hir_c_set_output(g, dst);
  hir_c_set_operand(g, 0, src);
  as_instr(g)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return g;
}
HirInstr hir_c_create_get_tuple_reg(HirRegister dst, HirRegister src, void *fs) {
  HirGetTuple *g = (HirGetTuple *)hir_c_alloc_instr(sizeof(HirGetTuple), 1);
  hir_c_init_deopt(g, HIR_OP_GetTuple);
  hir_c_set_output(g, dst);
  hir_c_set_operand(g, 0, src);
  as_instr(g)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return g;
}
HirInstr hir_c_create_is_neg_and_err_reg(HirRegister dst, HirRegister src, void *fs) {
  HirIsNegativeAndErrOccurred *i = (HirIsNegativeAndErrOccurred *)hir_c_alloc_instr(sizeof(HirIsNegativeAndErrOccurred), 1);
  hir_c_init_deopt(i, HIR_OP_IsNegativeAndErrOccurred);
  hir_c_set_output(i, dst);
  hir_c_set_operand(i, 0, src);
  as_instr(i)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return i;
}
HirInstr hir_c_create_load_cell_item_reg(HirRegister dst, HirRegister src) {
  HirLoadCellItem *i = (HirLoadCellItem *)hir_c_alloc_instr(sizeof(HirLoadCellItem), 1);
  hir_c_init_instr(i, HIR_OP_LoadCellItem);
  hir_c_set_output(i, dst);
  hir_c_set_operand(i, 0, src);
  return i;
}
HirInstr hir_c_create_load_current_func_reg(HirRegister dst) {
  HirLoadCurrentFunc *i = (HirLoadCurrentFunc *)hir_c_alloc_instr(sizeof(HirLoadCurrentFunc), 0);
  hir_c_init_instr(i, HIR_OP_LoadCurrentFunc);
  hir_c_set_output(i, dst);
  return i;
}
HirInstr hir_c_create_load_eval_breaker_reg(HirRegister dst) {
  HirLoadEvalBreaker *i = (HirLoadEvalBreaker *)hir_c_alloc_instr(sizeof(HirLoadEvalBreaker), 0);
  hir_c_init_instr(i, HIR_OP_LoadEvalBreaker);
  hir_c_set_output(i, dst);
  return i;
}
HirInstr hir_c_create_load_frame_reg(void) {
  HirLoadFrame *i = (HirLoadFrame *)hir_c_alloc_instr(sizeof(HirLoadFrame), 0);
  hir_c_init_instr(i, HIR_OP_LoadFrame);
  return i;
}
HirInstr hir_c_create_load_var_object_size_reg(HirRegister dst, HirRegister src) {
  HirLoadVarObjectSize *i = (HirLoadVarObjectSize *)hir_c_alloc_instr(sizeof(HirLoadVarObjectSize), 1);
  hir_c_init_instr(i, HIR_OP_LoadVarObjectSize);
  hir_c_set_output(i, dst);
  hir_c_set_operand(i, 0, src);
  return i;
}
HirInstr hir_c_create_check_err_occurred_reg(void *frame_state) {
  HirCheckErrOccurred *c = (HirCheckErrOccurred *)hir_c_alloc_instr(sizeof(HirCheckErrOccurred), 0);
  hir_c_init_deopt(c, HIR_OP_CheckErrOccurred);
  as_instr(c)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return c;
}

HirInstr hir_c_create_raise_reg(void *fs) {
  HirRaise *r = (HirRaise *)hir_c_alloc_instr(sizeof(HirRaise), 0);
  hir_c_init_deopt(r, HIR_OP_Raise);
  as_instr(r)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return r;
}
HirInstr hir_c_create_wait_handle_release_reg(HirRegister src) {
  HirWaitHandleRelease *w = (HirWaitHandleRelease *)hir_c_alloc_instr(sizeof(HirWaitHandleRelease), 1);
  hir_c_init_instr(w, HIR_OP_WaitHandleRelease);
  hir_c_set_operand(w, 0, src);
  return w;
}
HirInstr hir_c_create_make_set_reg(HirRegister dst, void *fs) {
  HirMakeSet *m = (HirMakeSet *)hir_c_alloc_instr(sizeof(HirMakeSet), 0);
  hir_c_init_deopt(m, HIR_OP_MakeSet);
  hir_c_set_output(m, dst);
  as_instr(m)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return m;
}
HirInstr hir_c_create_delete_attr_reg(HirRegister receiver, int32_t idx, void *fs) {
  HirDeleteAttr *d = (HirDeleteAttr *)hir_c_alloc_instr(sizeof(HirDeleteAttr), 1);
  hir_c_init_deopt(d, HIR_OP_DeleteAttr);
  d->name_idx = idx;
  hir_c_set_operand(d, 0, receiver);
  as_instr(d)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return d;
}
HirInstr hir_c_create_delete_subscr_reg(HirRegister container, HirRegister sub, void *fs) {
  HirDeleteSubscr *d = (HirDeleteSubscr *)hir_c_alloc_instr(sizeof(HirDeleteSubscr), 2);
  hir_c_init_deopt(d, HIR_OP_DeleteSubscr);
  hir_c_set_operand(d, 0, container);
  hir_c_set_operand(d, 1, sub);
  as_instr(d)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return d;
}
HirInstr hir_c_create_store_attr_reg(HirRegister receiver, HirRegister value, int32_t idx, void *fs) {
  HirStoreAttr *s = (HirStoreAttr *)hir_c_alloc_instr(sizeof(HirStoreAttr), 2);
  hir_c_init_deopt(s, HIR_OP_StoreAttr);
  s->name_idx = idx;
  hir_c_set_operand(s, 0, receiver);
  hir_c_set_operand(s, 1, value);
  as_instr(s)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return s;
}
HirInstr hir_c_create_swap_cell_item_reg(HirRegister dst, HirRegister cell, HirRegister value) {
  HirSwapCellItem *i = (HirSwapCellItem *)hir_c_alloc_instr(sizeof(HirSwapCellItem), 2);
  hir_c_init_instr(i, HIR_OP_SwapCellItem);
  hir_c_set_output(i, dst);
  hir_c_set_operand(i, 0, cell);
  hir_c_set_operand(i, 1, value);
  return i;
}
HirInstr hir_c_create_steal_cell_item_reg(HirRegister dst, HirRegister cell) {
  HirStealCellItem *i = (HirStealCellItem *)hir_c_alloc_instr(sizeof(HirStealCellItem), 1);
  hir_c_init_instr(i, HIR_OP_StealCellItem);
  hir_c_set_output(i, dst);
  hir_c_set_operand(i, 0, cell);
  return i;
}
HirInstr hir_c_create_set_cell_item_reg(HirRegister cell, HirRegister value, HirRegister old) {
  HirSetCellItem *s = (HirSetCellItem *)hir_c_alloc_instr(sizeof(HirSetCellItem), 3);
  hir_c_init_instr(s, HIR_OP_SetCellItem);
  hir_c_set_operand(s, 0, cell);
  hir_c_set_operand(s, 1, value);
  hir_c_set_operand(s, 2, old);
  return s;
}
HirInstr hir_c_create_at_quiescent_state_reg(void) {
  HirAtQuiescentState *a = (HirAtQuiescentState *)hir_c_alloc_instr(sizeof(HirAtQuiescentState), 0);
  hir_c_init_instr(a, HIR_OP_AtQuiescentState);
  return a;
}
HirInstr hir_c_create_run_periodic_tasks_reg(HirRegister dst, void *fs) {
  HirRunPeriodicTasks *r = (HirRunPeriodicTasks *)hir_c_alloc_instr(sizeof(HirRunPeriodicTasks), 0);
  hir_c_init_deopt(r, HIR_OP_RunPeriodicTasks);
  hir_c_set_output(r, dst);
  as_instr(r)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return r;
}

HirInstr hir_c_create_wait_handle_load_waiter_reg(HirRegister dst, HirRegister src) {
  HirWaitHandleLoadWaiter *i = (HirWaitHandleLoadWaiter *)hir_c_alloc_instr(sizeof(HirWaitHandleLoadWaiter), 1);
  hir_c_init_instr(i, HIR_OP_WaitHandleLoadWaiter);
  hir_c_set_output(i, dst);
  hir_c_set_operand(i, 0, src);
  return i;
}
HirInstr hir_c_create_wait_handle_load_coro_reg(HirRegister dst, HirRegister src) {
  HirWaitHandleLoadCoroOrResult *i = (HirWaitHandleLoadCoroOrResult *)hir_c_alloc_instr(sizeof(HirWaitHandleLoadCoroOrResult), 1);
  hir_c_init_instr(i, HIR_OP_WaitHandleLoadCoroOrResult);
  hir_c_set_output(i, dst);
  hir_c_set_operand(i, 0, src);
  return i;
}
HirInstr hir_c_create_set_update_reg(HirRegister dst, HirRegister set, HirRegister iter, void *fs) {
  HirSetUpdate *s = (HirSetUpdate *)hir_c_alloc_instr(sizeof(HirSetUpdate), 2);
  hir_c_init_deopt(s, HIR_OP_SetUpdate);
  hir_c_set_output(s, dst);
  hir_c_set_operand(s, 0, set);
  hir_c_set_operand(s, 1, iter);
  as_instr(s)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return s;
}
HirInstr hir_c_create_dict_update_reg(HirRegister dst, HirRegister dict, HirRegister update, void *fs) {
  HirDictUpdate *d = (HirDictUpdate *)hir_c_alloc_instr(sizeof(HirDictUpdate), 2);
  hir_c_init_deopt(d, HIR_OP_DictUpdate);
  hir_c_set_output(d, dst);
  hir_c_set_operand(d, 0, dict);
  hir_c_set_operand(d, 1, update);
  as_instr(d)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return d;
}
HirInstr hir_c_create_list_extend_reg(HirRegister dst, HirRegister list, HirRegister iter, void *fs) {
  HirListExtend *l = (HirListExtend *)hir_c_alloc_instr(sizeof(HirListExtend), 2);
  hir_c_init_deopt(l, HIR_OP_ListExtend);
  hir_c_set_output(l, dst);
  hir_c_set_operand(l, 0, list);
  hir_c_set_operand(l, 1, iter);
  as_instr(l)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return l;
}
HirInstr hir_c_create_copy_dict_without_keys_reg(HirRegister dst, HirRegister subj, HirRegister keys, void *fs) {
  HirCopyDictWithoutKeys *c = (HirCopyDictWithoutKeys *)hir_c_alloc_instr(sizeof(HirCopyDictWithoutKeys), 2);
  hir_c_init_deopt(c, HIR_OP_CopyDictWithoutKeys);
  hir_c_set_output(c, dst);
  hir_c_set_operand(c, 0, subj);
  hir_c_set_operand(c, 1, keys);
  as_instr(c)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return c;
}
HirInstr hir_c_create_make_tuple_from_list_reg(HirRegister dst, HirRegister list, void *fs) {
  HirMakeTupleFromList *m = (HirMakeTupleFromList *)hir_c_alloc_instr(sizeof(HirMakeTupleFromList), 1);
  hir_c_init_deopt(m, HIR_OP_MakeTupleFromList);
  hir_c_set_output(m, dst);
  hir_c_set_operand(m, 0, list);
  as_instr(m)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return m;
}
HirInstr hir_c_create_list_append_reg(HirRegister dst, HirRegister list, HirRegister item, void *fs) {
  HirListAppend *l = (HirListAppend *)hir_c_alloc_instr(sizeof(HirListAppend), 2);
  hir_c_init_deopt(l, HIR_OP_ListAppend);
  hir_c_set_output(l, dst);
  hir_c_set_operand(l, 0, list);
  hir_c_set_operand(l, 1, item);
  as_instr(l)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return l;
}
HirInstr hir_c_create_check_freevar_reg(HirRegister dst, HirRegister src, void *name, void *fs) {
  HirCheckFreevar *c = (HirCheckFreevar *)hir_c_alloc_instr(sizeof(HirCheckFreevar), 1);
  hir_c_init_deopt(c, HIR_OP_CheckFreevar);
  c->name = name;
  hir_c_set_output(c, dst);
  hir_c_set_operand(c, 0, src);
  as_instr(c)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return c;
}
HirInstr hir_c_create_load_global_reg(HirRegister dst, int32_t name_idx, void *fs) {
  HirLoadGlobal *lg = (HirLoadGlobal *)hir_c_alloc_instr(sizeof(HirLoadGlobal), 0);
  hir_c_init_deopt(lg, HIR_OP_LoadGlobal);
  lg->name_idx = name_idx;
  hir_c_set_output(lg, dst);
  as_instr(lg)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return lg;
}

HirInstr hir_c_create_dict_merge_reg(HirRegister dst, HirRegister dict, HirRegister update, HirRegister func, void *fs) {
  HirDictMerge *d = (HirDictMerge *)hir_c_alloc_instr(sizeof(HirDictMerge), 3);
  hir_c_init_deopt(d, HIR_OP_DictMerge);
  hir_c_set_output(d, dst);
  hir_c_set_operand(d, 0, dict);
  hir_c_set_operand(d, 1, update);
  hir_c_set_operand(d, 2, func);
  as_instr(d)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return d;
}
HirInstr hir_c_create_dict_subscr_reg(HirRegister dst, HirRegister dict, HirRegister key, void *fs) {
  HirDictSubscr *d = (HirDictSubscr *)hir_c_alloc_instr(sizeof(HirDictSubscr), 2);
  hir_c_init_deopt(d, HIR_OP_DictSubscr);
  hir_c_set_output(d, dst);
  hir_c_set_operand(d, 0, dict);
  hir_c_set_operand(d, 1, key);
  as_instr(d)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return d;
}
HirInstr hir_c_create_send_reg(HirRegister iter, HirRegister vout, HirRegister vin, void *fs) {
  /* Keep C++ — pure C crashes with _cinderx module loaded (bytecode_offset
   * fix was necessary but not sufficient, second cause TBD). */
  return Send::create(as_reg(iter), as_reg(vout), as_reg(vin), *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_convert_value_reg(HirRegister dst, HirRegister value, int32_t conversion, void *fs) {
  HirConvertValue *c = (HirConvertValue *)hir_c_alloc_instr(sizeof(HirConvertValue), 1);
  hir_c_init_deopt(c, HIR_OP_ConvertValue);
  c->converter_idx = conversion;
  hir_c_set_output(c, dst);
  hir_c_set_operand(c, 0, value);
  as_instr(c)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return c;
}
HirInstr hir_c_create_unary_op_reg(HirRegister dst, int32_t op_kind, HirRegister operand, void *fs) {
  HirUnaryOp *u = (HirUnaryOp *)hir_c_alloc_instr(sizeof(HirUnaryOp), 1);
  hir_c_init_deopt(u, HIR_OP_UnaryOp);
  u->op = op_kind;
  hir_c_set_output(u, dst);
  hir_c_set_operand(u, 0, operand);
  as_instr(u)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return u;
}
HirInstr hir_c_create_import_from_reg(HirRegister dst, HirRegister name, int32_t name_idx, void *fs) {
  HirImportFrom *i = (HirImportFrom *)hir_c_alloc_instr(sizeof(HirImportFrom), 1);
  hir_c_init_deopt(i, HIR_OP_ImportFrom);
  i->name_idx = name_idx;
  hir_c_set_output(i, dst);
  hir_c_set_operand(i, 0, name);
  as_instr(i)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return i;
}
HirInstr hir_c_create_invoke_iter_next_reg(HirRegister dst, HirRegister iter, void *fs) {
  HirInvokeIterNext *i = (HirInvokeIterNext *)hir_c_alloc_instr(sizeof(HirInvokeIterNext), 1);
  hir_c_init_deopt(i, HIR_OP_InvokeIterNext);
  hir_c_set_output(i, dst);
  hir_c_set_operand(i, 0, iter);
  as_instr(i)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return i;
}
HirInstr hir_c_create_primitive_unbox_reg(HirRegister dst, HirRegister src, HirType type) {
  HirPrimitiveUnbox *pu = (HirPrimitiveUnbox *)hir_c_alloc_instr(sizeof(HirPrimitiveUnbox), 1);
  hir_c_init_instr(pu, HIR_OP_PrimitiveUnbox);
  pu->type = type;
  hir_c_set_output(pu, dst);
  hir_c_set_operand(pu, 0, src);
  return pu;
}

HirInstr hir_c_create_make_tuple_reg(size_t n, HirRegister dst, void *fs) {
  HirDeoptLayout *m = (HirDeoptLayout *)hir_c_alloc_instr(sizeof(HirDeoptLayout), n);
  hir_c_init_deopt(m, HIR_OP_MakeTuple);
  hir_c_set_output(m, dst);
  as_instr(m)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return m;
}
HirInstr hir_c_create_make_list_reg(size_t n, HirRegister dst, void *fs) {
  HirDeoptLayout *m = (HirDeoptLayout *)hir_c_alloc_instr(sizeof(HirDeoptLayout), n);
  hir_c_init_deopt(m, HIR_OP_MakeList);
  hir_c_set_output(m, dst);
  as_instr(m)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return m;
}
HirInstr hir_c_create_tp_alloc_reg(HirRegister dst, void *pytype, void *fs) {
  HirTpAlloc *t = (HirTpAlloc *)hir_c_alloc_instr(sizeof(HirTpAlloc), 0);
  hir_c_init_deopt(t, HIR_OP_TpAlloc);
  t->pytype = pytype;
  hir_c_set_output(t, dst);
  as_instr(t)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return t;
}
HirInstr hir_c_create_unpack_ex_to_tuple_reg(HirRegister dst, HirRegister seq, int32_t before, int32_t after, void *fs) {
  HirUnpackExToTuple *u = (HirUnpackExToTuple *)hir_c_alloc_instr(sizeof(HirUnpackExToTuple), 1);
  hir_c_init_deopt(u, HIR_OP_UnpackExToTuple);
  u->before = before;
  u->after = after;
  hir_c_set_output(u, dst);
  hir_c_set_operand(u, 0, seq);
  as_instr(u)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return u;
}
HirInstr hir_c_create_load_method_reg(HirRegister dst, HirRegister receiver, int32_t name_idx, void *fs) {
  HirLoadMethod *m = (HirLoadMethod *)hir_c_alloc_instr(sizeof(HirLoadMethod), 1);
  hir_c_init_deopt(m, HIR_OP_LoadMethod);
  m->name_idx = name_idx;
  hir_c_set_output(m, dst);
  hir_c_set_operand(m, 0, receiver);
  as_instr(m)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return m;
}
HirInstr hir_c_create_load_special_reg(HirRegister dst, HirRegister self, int32_t oparg, void *fs) {
  HirLoadSpecial *l = (HirLoadSpecial *)hir_c_alloc_instr(sizeof(HirLoadSpecial), 1);
  hir_c_init_deopt(l, HIR_OP_LoadSpecial);
  l->special_idx = oparg;
  hir_c_set_output(l, dst);
  hir_c_set_operand(l, 0, self);
  as_instr(l)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return l;
}
HirInstr hir_c_create_match_keys_reg(HirRegister dst, HirRegister subj, HirRegister keys, void *fs) {
  HirMatchKeys *m = (HirMatchKeys *)hir_c_alloc_instr(sizeof(HirMatchKeys), 2);
  hir_c_init_deopt(m, HIR_OP_MatchKeys);
  hir_c_set_output(m, dst);
  hir_c_set_operand(m, 0, subj);
  hir_c_set_operand(m, 1, keys);
  as_instr(m)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return m;
}
HirInstr hir_c_create_raise_awaitable_error_reg(HirRegister type, int32_t is_aenter, void *fs) {
  HirRaiseAwaitableError *r = (HirRaiseAwaitableError *)hir_c_alloc_instr(sizeof(HirRaiseAwaitableError), 1);
  hir_c_init_deopt(r, HIR_OP_RaiseAwaitableError);
  r->is_aenter = (uint8_t)is_aenter;
  hir_c_set_operand(r, 0, type);
  as_instr(r)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return r;
}
HirInstr hir_c_create_format_value_reg(HirRegister dst, HirRegister fmt, HirRegister val, int32_t conv, void *fs) {
  /* FormatValue: DeoptBase + int conversion_, HasOutput, Operands<2> */
  HirBuildInterpolation *f = (HirBuildInterpolation *)hir_c_alloc_instr(sizeof(HirBuildInterpolation), 2);
  hir_c_init_deopt(f, HIR_OP_FormatValue);
  f->conversion = conv;
  hir_c_set_output(f, dst);
  hir_c_set_operand(f, 0, fmt);
  hir_c_set_operand(f, 1, val);
  as_instr(f)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return f;
}

HirInstr hir_c_create_eager_import_name_reg(HirRegister dst, int32_t name_idx, HirRegister fromlist, HirRegister level, void *fs) {
  HirImportFrom *e = (HirImportFrom *)hir_c_alloc_instr(sizeof(HirImportFrom), 2);
  hir_c_init_deopt(e, HIR_OP_EagerImportName);
  e->name_idx = name_idx;
  hir_c_set_output(e, dst);
  hir_c_set_operand(e, 0, fromlist);
  hir_c_set_operand(e, 1, level);
  as_instr(e)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return e;
}
HirInstr hir_c_create_make_checked_dict_reg(HirRegister dst, int32_t size, HirType type, void *fs) {
  HirMakeCheckedDict *d = (HirMakeCheckedDict *)hir_c_alloc_instr(sizeof(HirMakeCheckedDict), 0);
  hir_c_init_deopt(d, HIR_OP_MakeCheckedDict);
  d->capacity = (size_t)size;
  d->type = type;
  hir_c_set_output(d, dst);
  as_instr(d)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return d;
}
HirInstr hir_c_create_make_checked_list_reg(int32_t size, HirRegister dst, HirType type, void *fs) {
  HirMakeCheckedList *l = (HirMakeCheckedList *)hir_c_alloc_instr(sizeof(HirMakeCheckedList), 0);
  hir_c_init_deopt(l, HIR_OP_MakeCheckedList);
  l->type = type;
  hir_c_set_output(l, dst);
  as_instr(l)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return l;
}
HirInstr hir_c_create_make_function_reg(HirRegister dst, HirRegister code, HirRegister qualname, void *fs) {
  HirMakeFunction *m = (HirMakeFunction *)hir_c_alloc_instr(sizeof(HirMakeFunction), 2);
  hir_c_init_deopt(m, HIR_OP_MakeFunction);
  hir_c_set_output(m, dst);
  hir_c_set_operand(m, 0, code);
  hir_c_set_operand(m, 1, qualname);
  as_instr(m)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return m;
}
HirInstr hir_c_create_build_template_reg(HirRegister strings, HirRegister interps, HirRegister dst, void *fs) {
  /* BuildTemplate: DeoptBase, HasOutput, Operands<2>, no custom fields */
  HirDeoptLayout *b = (HirDeoptLayout *)hir_c_alloc_instr(sizeof(HirDeoptLayout), 2);
  hir_c_init_deopt(b, HIR_OP_BuildTemplate);
  hir_c_set_output(b, dst);
  hir_c_set_operand(b, 0, strings);
  hir_c_set_operand(b, 1, interps);
  as_instr(b)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return b;
}
HirInstr hir_c_create_build_interpolation_reg(HirRegister dst, HirRegister val, HirRegister str, HirRegister fmt, int32_t conv, void *fs) {
  HirBuildInterpolation *b = (HirBuildInterpolation *)hir_c_alloc_instr(sizeof(HirBuildInterpolation), 3);
  hir_c_init_deopt(b, HIR_OP_BuildInterpolation);
  b->conversion = conv;
  hir_c_set_output(b, dst);
  hir_c_set_operand(b, 0, val);
  hir_c_set_operand(b, 1, str);
  hir_c_set_operand(b, 2, fmt);
  as_instr(b)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return b;
}
HirInstr hir_c_create_load_attr_reg2(HirRegister dst, HirRegister receiver, int32_t name_idx, void *fs) {
  HirLoadAttr *la = (HirLoadAttr *)hir_c_alloc_instr(sizeof(HirLoadAttr), 1);
  hir_c_init_deopt(la, HIR_OP_LoadAttr);
  la->name_idx = name_idx;
  la->already_optimized = 0;
  hir_c_set_output(la, dst);
  hir_c_set_operand(la, 0, receiver);
  as_instr(la)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return la;
}
HirInstr hir_c_create_init_frame_cell_vars_reg(HirRegister func, int32_t nfree) {
  HirInitFrameCellVars *i = (HirInitFrameCellVars *)hir_c_alloc_instr(sizeof(HirInitFrameCellVars), 1);
  hir_c_init_instr(i, HIR_OP_InitFrameCellVars);
  i->cells = nfree;
  hir_c_set_operand(i, 0, func);
  return i;
}

HirInstr hir_c_create_store_field_reg(HirRegister receiver, const char *name, intptr_t offset, HirRegister value, HirType type, HirRegister previous) {
  HirStoreField *s = (HirStoreField *)hir_c_alloc_instr(sizeof(HirStoreField), 3);
  hir_c_init_instr(s, HIR_OP_StoreField);
  /* Placement new for std::string name_ (stored as opaque bytes) */
  new (&s->name_storage) std::string(name);
  s->offset = offset;
  s->type = type;
  hir_c_set_operand(s, 0, receiver);
  hir_c_set_operand(s, 1, value);
  hir_c_set_operand(s, 2, previous);
  return s;
}
HirInstr hir_c_create_yield_and_yield_from_reg(HirRegister dst, HirRegister waiter, HirRegister coro, void *fs) {
  HirYieldAndYieldFrom *y = (HirYieldAndYieldFrom *)hir_c_alloc_instr(sizeof(HirYieldAndYieldFrom), 2);
  hir_c_init_deopt(y, HIR_OP_YieldAndYieldFrom);
  hir_c_set_output(y, dst);
  hir_c_set_operand(y, 0, waiter);
  hir_c_set_operand(y, 1, coro);
  as_instr(y)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return y;
}
HirInstr hir_c_create_yield_from_handle_stop_async_reg(HirRegister dst, HirRegister send, HirRegister awaitable, void *fs) {
  HirYieldFromHandleStopAsyncIteration *y = (HirYieldFromHandleStopAsyncIteration *)hir_c_alloc_instr(sizeof(HirYieldFromHandleStopAsyncIteration), 2);
  hir_c_init_deopt(y, HIR_OP_YieldFromHandleStopAsyncIteration);
  hir_c_set_output(y, dst);
  hir_c_set_operand(y, 0, send);
  hir_c_set_operand(y, 1, awaitable);
  as_instr(y)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return y;
}
HirInstr hir_c_create_call_ex_reg(HirRegister dst, HirRegister func, HirRegister pargs, HirRegister kwargs, uint32_t flags, void *fs) {
  HirCallEx *c = (HirCallEx *)hir_c_alloc_instr(sizeof(HirCallEx), 3);
  hir_c_init_deopt(c, HIR_OP_CallEx);
  c->flags = flags;
  hir_c_set_output(c, dst);
  hir_c_set_operand(c, 0, func);
  hir_c_set_operand(c, 1, pargs);
  hir_c_set_operand(c, 2, kwargs);
  as_instr(c)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return c;
}
HirInstr hir_c_create_import_name_reg(HirRegister dst, int32_t name_idx, HirRegister fromlist, HirRegister level, void *fs) {
  /* ImportName: DeoptBaseWithNameIdx, HasOutput, Operands<2> */
  HirImportFrom *i = (HirImportFrom *)hir_c_alloc_instr(sizeof(HirImportFrom), 2);
  hir_c_init_deopt(i, HIR_OP_ImportName);
  i->name_idx = name_idx;
  hir_c_set_output(i, dst);
  hir_c_set_operand(i, 0, fromlist);
  hir_c_set_operand(i, 1, level);
  as_instr(i)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return i;
}

HirInstr hir_c_create_call_method_reg(size_t n_operands, HirRegister dst, uint32_t flags) {
  HirCallMethod *m = (HirCallMethod *)hir_c_alloc_instr(sizeof(HirCallMethod), n_operands);
  hir_c_init_deopt(m, HIR_OP_CallMethod);
  m->flags = flags;
  hir_c_set_output(m, dst);
  return m;
}
HirInstr hir_c_create_call_static_ret_void_reg(size_t n_operands, void *addr) {
  HirCallStaticRetVoid *c = (HirCallStaticRetVoid *)hir_c_alloc_instr(sizeof(HirCallStaticRetVoid), n_operands);
  hir_c_init_instr(c, HIR_OP_CallStaticRetVoid);
  c->addr = addr;
  return c;
}
HirInstr hir_c_create_invoke_static_function_reg(size_t n_operands, HirRegister dst, void *func, HirType ret_type) {
  HirInvokeStaticFunction *i = (HirInvokeStaticFunction *)hir_c_alloc_instr(sizeof(HirInvokeStaticFunction), n_operands);
  hir_c_init_deopt(i, HIR_OP_InvokeStaticFunction);
  i->func = func;
  i->ret_type = ret_type;
  hir_c_set_output(i, dst);
  return i;
}

HirInstr hir_c_create_load_global_cached_reg(HirRegister dst, void *code, void *builtins, void *globals, int32_t name_idx) {
  HirLoadGlobalCached *lg = (HirLoadGlobalCached *)hir_c_alloc_instr(sizeof(HirLoadGlobalCached), 0);
  hir_c_init_instr(lg, HIR_OP_LoadGlobalCached);
  lg->code = code;
  lg->builtins = builtins;
  lg->globals = globals;
  lg->name_idx = name_idx;
  hir_c_set_output(lg, dst);
  return lg;
}
HirInstr hir_c_create_load_function_indirect_reg(void *indirect_ptr, void *descr, HirRegister dst, void *fs) {
  HirLoadFunctionIndirect *l = (HirLoadFunctionIndirect *)hir_c_alloc_instr(sizeof(HirLoadFunctionIndirect), 0);
  hir_c_init_deopt(l, HIR_OP_LoadFunctionIndirect);
  l->funcptr = indirect_ptr;
  l->descr = descr;
  hir_c_set_output(l, dst);
  as_instr(l)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return l;
}
HirInstr hir_c_create_store_array_item_reg(HirRegister arr, HirRegister idx, HirRegister value, HirRegister container, HirType elem_type) {
  HirStoreArrayItem *s = (HirStoreArrayItem *)hir_c_alloc_instr(sizeof(HirStoreArrayItem), 4);
  hir_c_init_instr(s, HIR_OP_StoreArrayItem);
  s->type = elem_type;
  hir_c_set_operand(s, 0, arr);
  hir_c_set_operand(s, 1, idx);
  hir_c_set_operand(s, 2, value);
  hir_c_set_operand(s, 3, container);
  return s;
}

HirInstr hir_c_create_cast_reg(HirRegister dst, HirRegister receiver, void *pytype, int optional, int exact, void *fs) {
  HirCast *c = (HirCast *)hir_c_alloc_instr(sizeof(HirCast), 1);
  hir_c_init_deopt(c, HIR_OP_Cast);
  c->pytype = pytype;
  c->optional = (uint8_t)(optional != 0);
  c->exact = (uint8_t)(exact != 0);
  hir_c_set_output(c, dst);
  hir_c_set_operand(c, 0, receiver);
  as_instr(c)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return c;
}
HirInstr hir_c_create_raise_static_reg(int32_t reraise, void *exc_type, const char *fmt, void *fs) {
  HirRaiseStatic *r = (HirRaiseStatic *)hir_c_alloc_instr(sizeof(HirRaiseStatic), 0);
  hir_c_init_deopt(r, HIR_OP_RaiseStatic);
  r->fmt = fmt;
  r->exc_type = exc_type;
  as_instr(r)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return r;
}

HirInstr hir_c_create_call_cfunc_reg(size_t n_operands, HirRegister dst, int32_t func_enum, HirRegister *operands) {
  HirCallCFunc *c = (HirCallCFunc *)hir_c_alloc_instr(sizeof(HirCallCFunc), n_operands);
  hir_c_init_instr(c, HIR_OP_CallCFunc);
  c->func = func_enum;
  hir_c_set_output(c, dst);
  for (size_t i = 0; i < n_operands; i++)
    hir_c_set_operand(c, i, operands[i]);
  return c;
}

HirInstr hir_c_create_call_ind_reg2(size_t n_operands, HirRegister dst, const char *name, HirType ret_type) {
  HirCallInd *c = (HirCallInd *)hir_c_alloc_instr(sizeof(HirCallInd), n_operands);
  hir_c_init_deopt(c, HIR_OP_CallInd);
  c->name = name;
  c->ret_type = ret_type;
  hir_c_set_output(c, dst);
  return c;
}

HirInstr hir_c_create_load_method_super_reg(HirRegister dst, HirRegister global_super, HirRegister type, HirRegister receiver, int32_t name_idx, int no_args, void *fs) {
  HirLoadMethodSuper *m = (HirLoadMethodSuper *)hir_c_alloc_instr(sizeof(HirLoadMethodSuper), 3);
  hir_c_init_deopt(m, HIR_OP_LoadMethodSuper);
  m->name_idx = name_idx;
  m->no_args_in_super_call = (uint8_t)(no_args != 0);
  hir_c_set_output(m, dst);
  hir_c_set_operand(m, 0, global_super);
  hir_c_set_operand(m, 1, type);
  hir_c_set_operand(m, 2, receiver);
  as_instr(m)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return m;
}
HirInstr hir_c_create_load_attr_super_reg(HirRegister dst, HirRegister global_super, HirRegister type, HirRegister receiver, int32_t name_idx, int no_args, void *fs) {
  HirLoadAttrSuper *a = (HirLoadAttrSuper *)hir_c_alloc_instr(sizeof(HirLoadAttrSuper), 3);
  hir_c_init_deopt(a, HIR_OP_LoadAttrSuper);
  a->name_idx = name_idx;
  a->no_args_in_super_call = (uint8_t)(no_args != 0);
  hir_c_set_output(a, dst);
  hir_c_set_operand(a, 0, global_super);
  hir_c_set_operand(a, 1, type);
  hir_c_set_operand(a, 2, receiver);
  as_instr(a)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return a;
}

HirInstr hir_c_create_match_class_reg2(HirRegister dst, HirRegister subject, HirRegister type, HirRegister nargs, HirRegister names) {
  HirMatchClass *m = (HirMatchClass *)hir_c_alloc_instr(sizeof(HirMatchClass), 4);
  hir_c_init_instr(m, HIR_OP_MatchClass);
  hir_c_set_output(m, dst);
  hir_c_set_operand(m, 0, subject);
  hir_c_set_operand(m, 1, type);
  hir_c_set_operand(m, 2, nargs);
  hir_c_set_operand(m, 3, names);
  return m;
}

HirInstr hir_c_create_load_attr_special_reg(HirRegister dst, HirRegister receiver, void *id, const char *fmt, void *fs) {
  HirLoadAttrSpecial *l = (HirLoadAttrSpecial *)hir_c_alloc_instr(sizeof(HirLoadAttrSpecial), 1);
  hir_c_init_deopt(l, HIR_OP_LoadAttrSpecial);
  l->id = id;
  l->failure_fmt = fmt;
  hir_c_set_output(l, dst);
  hir_c_set_operand(l, 0, receiver);
  as_instr(l)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(fs));
  return l;
}

HirInstr hir_c_create_call_intrinsic_reg2(size_t n_operands, HirRegister dst, int32_t index, HirRegister *operands) {
  HirCallIntrinsic *c = (HirCallIntrinsic *)hir_c_alloc_instr(sizeof(HirCallIntrinsic), n_operands);
  hir_c_init_instr(c, HIR_OP_CallIntrinsic);
  c->index = (size_t)index;
  hir_c_set_output(c, dst);
  for (size_t i = 0; i < n_operands; i++)
    hir_c_set_operand(c, i, operands[i]);
  return c;
}

HirInstr hir_c_create_cond_branch_iter_not_done_cpp(
    HirRegister src, void *body_block, void *done_block) {
  return CondBranchIterNotDone::create(
      as_reg(src), static_cast<BasicBlock*>(body_block),
      static_cast<BasicBlock*>(done_block));
}

HirInstr hir_c_create_int_convert_reg(HirRegister dst, HirRegister src,
                                       HirType type) {
  HirIntConvert *i = (HirIntConvert *)hir_c_alloc_instr(sizeof(HirIntConvert), 1);
  hir_c_init_instr(i, HIR_OP_IntConvert);
  i->type = type;
  hir_c_set_output(i, dst);
  hir_c_set_operand(i, 0, src);
  return i;
}

HirInstr hir_c_create_get_iter_reg(HirRegister dst, HirRegister src,
                                    void *frame_state) {
  HirDeoptLayout *g = (HirDeoptLayout *)hir_c_alloc_instr(sizeof(HirDeoptLayout), 1);
  hir_c_init_deopt(g, HIR_OP_GetIter);
  hir_c_set_output(g, dst);
  hir_c_set_operand(g, 0, src);
  as_instr(g)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return g;
}

HirInstr hir_c_create_load_field_address_reg(HirRegister dst, HirRegister object,
                                              HirRegister offset) {
  HirInstrLayout *l = (HirInstrLayout *)hir_c_alloc_instr(sizeof(HirInstrLayout), 2);
  hir_c_init_instr(l, HIR_OP_LoadFieldAddress);
  hir_c_set_output(l, dst);
  hir_c_set_operand(l, 0, object);
  hir_c_set_operand(l, 1, offset);
  return l;
}

HirInstr hir_c_create_yield_value_reg(HirRegister dst, HirRegister src,
                                       void *frame_state) {
  HirYieldValue *y = (HirYieldValue *)hir_c_alloc_instr(sizeof(HirYieldValue), 1);
  hir_c_init_deopt(y, HIR_OP_YieldValue);
  hir_c_set_output(y, dst);
  hir_c_set_operand(y, 0, src);
  as_instr(y)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return y;
}

HirInstr hir_c_create_yield_from_reg(HirRegister dst, HirRegister send_value,
                                      HirRegister iter, void *frame_state) {
  HirDeoptLayout *y = (HirDeoptLayout *)hir_c_alloc_instr(sizeof(HirDeoptLayout), 2);
  hir_c_init_deopt(y, HIR_OP_YieldFrom);
  hir_c_set_output(y, dst);
  hir_c_set_operand(y, 0, send_value);
  hir_c_set_operand(y, 1, iter);
  as_instr(y)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return y;
}

HirInstr hir_c_create_check_var_reg(HirRegister dst, HirRegister src,
                                     void *name, void *frame_state) {
  HirCheckVar *c = (HirCheckVar *)hir_c_alloc_instr(sizeof(HirCheckVar), 1);
  hir_c_init_deopt(c, HIR_OP_CheckVar);
  c->name = name;
  hir_c_set_output(c, dst);
  hir_c_set_operand(c, 0, src);
  as_instr(c)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return c;
}

HirInstr hir_c_create_set_dict_item_reg(HirRegister dst, HirRegister dict,
                                         HirRegister key, HirRegister value,
                                         void *frame_state) {
  HirDeoptLayout *s = (HirDeoptLayout *)hir_c_alloc_instr(sizeof(HirDeoptLayout), 3);
  hir_c_init_deopt(s, HIR_OP_SetDictItem);
  hir_c_set_output(s, dst);
  hir_c_set_operand(s, 0, dict);
  hir_c_set_operand(s, 1, key);
  hir_c_set_operand(s, 2, value);
  as_instr(s)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return s;
}

HirInstr hir_c_create_load_tuple_item_reg(HirRegister dst, HirRegister tuple,
                                           int32_t idx) {
  HirLoadTupleItem *l = (HirLoadTupleItem *)hir_c_alloc_instr(sizeof(HirLoadTupleItem), 1);
  hir_c_init_instr(l, HIR_OP_LoadTupleItem);
  l->idx = (size_t)idx;
  hir_c_set_output(l, dst);
  hir_c_set_operand(l, 0, tuple);
  return l;
}

HirInstr hir_c_create_is_truthy_reg(HirRegister dst, HirRegister src,
                                     void *frame_state) {
  HirIsTruthy *t = (HirIsTruthy *)hir_c_alloc_instr(sizeof(HirIsTruthy), 1);
  hir_c_init_deopt(t, HIR_OP_IsTruthy);
  hir_c_set_output(t, dst);
  hir_c_set_operand(t, 0, src);
  as_instr(t)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return t;
}

HirInstr hir_c_create_get_second_output_reg(HirRegister dst, HirType type,
                                             HirRegister src) {
  HirGetSecondOutput *g = (HirGetSecondOutput *)hir_c_alloc_instr(sizeof(HirGetSecondOutput), 1);
  hir_c_init_instr(g, HIR_OP_GetSecondOutput);
  g->type = type;
  hir_c_set_output(g, dst);
  hir_c_set_operand(g, 0, src);
  return g;
}

HirInstr hir_c_create_set_function_attr_reg(HirRegister value, HirRegister base,
                                             int32_t field) {
  HirSetFunctionAttr *s = (HirSetFunctionAttr *)hir_c_alloc_instr(sizeof(HirSetFunctionAttr), 2);
  hir_c_init_instr(s, HIR_OP_SetFunctionAttr);
  s->field = field;
  hir_c_set_operand(s, 0, value);
  hir_c_set_operand(s, 1, base);
  return s;
}

HirInstr hir_c_create_check_neg_reg(HirRegister dst, HirRegister src,
                                     void *frame_state) {
  HirCheckNeg *c = (HirCheckNeg *)hir_c_alloc_instr(sizeof(HirCheckNeg), 1);
  hir_c_init_deopt(c, HIR_OP_CheckNeg);
  hir_c_set_output(c, dst);
  hir_c_set_operand(c, 0, src);
  if (frame_state)
    as_instr(c)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return (HirInstr)c;
}

HirInstr hir_c_create_get_length_reg(HirRegister dst, HirRegister src,
                                      void *frame_state) {
  HirGetLength *g = (HirGetLength *)hir_c_alloc_instr(sizeof(HirGetLength), 1);
  hir_c_init_deopt(g, HIR_OP_GetLength);
  hir_c_set_output(g, dst);
  hir_c_set_operand(g, 0, src);
  if (frame_state)
    as_instr(g)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return (HirInstr)g;
}

HirInstr hir_c_create_primitive_box_reg(HirRegister dst, HirRegister src,
                                         HirType type, void *frame_state) {
  HirPrimitiveBox *p = (HirPrimitiveBox *)hir_c_alloc_instr(sizeof(HirPrimitiveBox), 1);
  hir_c_init_deopt(p, HIR_OP_PrimitiveBox);
  p->type = type;
  hir_c_set_output(p, dst);
  hir_c_set_operand(p, 0, src);
  as_instr(p)->asDeoptBase()->setFrameState(*static_cast<const FrameState*>(frame_state));
  return p;
}

HirInstr hir_c_create_load_array_item_reg(HirRegister dst, HirRegister arr,
                                           HirRegister idx, HirRegister container,
                                           intptr_t offset, HirType type) {
  HirLoadArrayItem *l = (HirLoadArrayItem *)hir_c_alloc_instr(sizeof(HirLoadArrayItem), 3);
  hir_c_init_instr(l, HIR_OP_LoadArrayItem);
  l->offset = (intptr_t)offset;
  l->type = type;
  hir_c_set_output(l, dst);
  hir_c_set_operand(l, 0, arr);
  hir_c_set_operand(l, 1, idx);
  hir_c_set_operand(l, 2, container);
  return l;
}

HirInstr hir_c_create_vectorcall_reg(size_t n_operands, HirRegister dst,
                                      uint32_t flags) {
  HirVectorCall *v = (HirVectorCall *)hir_c_alloc_instr(sizeof(HirVectorCall), n_operands);
  hir_c_init_deopt(v, HIR_OP_VectorCall);
  v->flags = flags;
  hir_c_set_output(v, dst);
  return v;
}

HirInstr hir_c_create_cond_branch_check_type_cpp(
    HirRegister target, HirType type,
    void *true_block, void *false_block) {
  Type cpp_type = Type::fromHirType(type);
  auto* true_bb = static_cast<BasicBlock*>(true_block);
  auto* false_bb = static_cast<BasicBlock*>(false_block);
  return CondBranchCheckType::create(as_reg(target), cpp_type, true_bb, false_bb);
}

HirInstr hir_c_create_cond_branch_cpp(void *cond_reg,
                                       void *true_block,
                                       void *false_block) {
  auto* true_bb = static_cast<BasicBlock*>(true_block);
  auto* false_bb = static_cast<BasicBlock*>(false_block);
  return CondBranch::create(as_reg(cond_reg), true_bb, false_bb);
}

/* ---- Builder-style factories (caller provides dst register) ---- */

HirInstr hir_c_create_load_field_reg(HirRegister dst, HirRegister receiver,
                                      const char *name, intptr_t offset,
                                      HirType type, int borrowed) {
  HirLoadField *l = (HirLoadField *)hir_c_alloc_instr(sizeof(HirLoadField), 1);
  hir_c_init_instr(l, HIR_OP_LoadField);
  new (&l->name_storage) std::string(name);
  l->offset = (size_t)offset;
  l->type = type;
  l->borrowed = (uint8_t)(borrowed != 0);
  hir_c_set_output(l, dst);
  hir_c_set_operand(l, 0, receiver);
  return l;
}

HirInstr hir_c_create_guard_type_reg(HirRegister dst, HirType target,
                                      HirRegister src) {
  HirGuardType *g = (HirGuardType *)hir_c_alloc_instr(sizeof(HirGuardType), 1);
  hir_c_init_deopt(g, HIR_OP_GuardType);
  g->target = target;
  hir_c_set_output(g, dst);
  hir_c_set_operand(g, 0, src);
  return g;
}

HirInstr hir_c_create_refine_type_reg(HirRegister dst, HirType type,
                                       HirRegister src) {
  HirRefineType *r = (HirRefineType *)hir_c_alloc_instr(sizeof(HirRefineType), 1);
  hir_c_init_instr(r, HIR_OP_RefineType);
  r->type = type;
  hir_c_set_output(r, dst);
  hir_c_set_operand(r, 0, src);
  return r;
}

HirInstr hir_c_create_check_exc_reg(HirRegister dst, HirRegister src) {
  HirCheckExc *c = (HirCheckExc *)hir_c_alloc_instr(sizeof(HirCheckExc), 1);
  hir_c_init_deopt(c, HIR_OP_CheckExc);
  hir_c_set_output(c, dst);
  hir_c_set_operand(c, 0, src);
  return c;
}

HirInstr hir_c_create_deopt(void) {
  HirDeopt *d = (HirDeopt *)hir_c_alloc_instr(sizeof(HirDeopt), 0);
  hir_c_init_deopt(d, HIR_OP_Deopt);
  return d;
}

HirInstr hir_c_create_return(HirRegister src, HirType type) {
  HirReturn *r = (HirReturn *)hir_c_alloc_instr(sizeof(HirReturn), 1);
  hir_c_init_instr(r, HIR_OP_Return);
  r->type = type;
  hir_c_set_operand(r, 0, src);
  return r;
}

/* ---- Frame state ---- */

void *hir_get_frame_state(HirInstr instr) {
  return const_cast<FrameState*>(get_frame_state(*as_instr(instr)));
}

} /* extern "C" */
