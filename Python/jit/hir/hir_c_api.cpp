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
  return LoadConst::create(as_reg(output), TBottom);
}

HirInstr hir_assign_create(HirRegister output, HirRegister value) {
  return Assign::create(as_reg(output), as_reg(value));
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
  return CompareBool::create(
      as_reg(output),
      static_cast<CompareOp>(compare_op),
      as_reg(left), as_reg(right),
      *fs);
}

HirInstr hir_c_create_binary_op(HirFunction func, int32_t op_kind,
                                HirRegister left, HirRegister right,
                                void *frame_state) {
  if (!frame_state) return nullptr;
  auto* f = static_cast<Function*>(func);
  auto* dst = f->env.AllocateRegister();
  return BinaryOp::create(
      dst, static_cast<BinaryOpKind>(op_kind),
      as_reg(left), as_reg(right),
      *static_cast<const FrameState*>(frame_state));
}

HirInstr hir_c_create_guard_type(HirFunction func, HirType target,
                                 HirRegister src, void *frame_state) {
  auto* f = static_cast<Function*>(func);
  auto* dst = f->env.AllocateRegister();
  Type cpp_target = Type::fromHirType(target);
  if (frame_state) {
    return GuardType::create(
        dst, cpp_target, as_reg(src),
        *static_cast<const FrameState*>(frame_state));
  }
  return GuardType::create(dst, cpp_target, as_reg(src));
}

HirInstr hir_c_create_check_exc(HirFunction func, HirRegister src,
                                void *frame_state) {
  if (!frame_state) return nullptr;
  auto* f = static_cast<Function*>(func);
  auto* dst = f->env.AllocateRegister();
  return CheckExc::create(
      dst, as_reg(src),
      *static_cast<const FrameState*>(frame_state));
}

/* ---- LoadField / GuardIs / CheckNeg / PrimitiveBox / CheckSequenceBounds ---- */

HirInstr hir_c_create_load_field(HirFunction func, HirRegister receiver,
                                  const char *name, intptr_t offset,
                                  HirType type, int borrowed) {
  auto* f = static_cast<Function*>(func);
  auto* dst = f->env.AllocateRegister();
  Type cpp_type = Type::fromHirType(type);
  return LoadField::create(
      dst, as_reg(receiver), std::string(name),
      static_cast<std::size_t>(offset), cpp_type,
      borrowed != 0);
}

HirInstr hir_c_create_guard_is(HirFunction func, void *target,
                                HirRegister src) {
  auto* f = static_cast<Function*>(func);
  auto* dst = f->env.AllocateRegister();
  return GuardIs::create(
      dst, static_cast<PyObject*>(target), as_reg(src));
}

HirInstr hir_c_create_check_neg(HirFunction func, HirRegister src,
                                 void *frame_state) {
  if (!frame_state) return nullptr;
  auto* f = static_cast<Function*>(func);
  auto* dst = f->env.AllocateRegister();
  return CheckNeg::create(
      dst, as_reg(src),
      *static_cast<const FrameState*>(frame_state));
}

HirInstr hir_c_create_primitive_box(HirFunction func, HirRegister src,
                                     HirType type, void *frame_state) {
  if (!frame_state) return nullptr;
  auto* f = static_cast<Function*>(func);
  auto* dst = f->env.AllocateRegister();
  Type cpp_type = Type::fromHirType(type);
  return PrimitiveBox::create(
      dst, as_reg(src), cpp_type,
      *static_cast<const FrameState*>(frame_state));
}

HirInstr hir_c_create_check_seq_bounds(HirFunction func,
                                        HirRegister seq, HirRegister idx,
                                        void *frame_state) {
  if (!frame_state) return nullptr;
  auto* f = static_cast<Function*>(func);
  auto* dst = f->env.AllocateRegister();
  return CheckSequenceBounds::create(
      dst, as_reg(seq), as_reg(idx),
      *static_cast<const FrameState*>(frame_state));
}

/* ---- DeoptBaseWithNameIdx factories ---- */

#define DEOPT_NAMEIDX1_FACTORY(name, CppType) \
HirInstr hir_c_create_##name(HirFunction func, \
    HirRegister receiver, int name_idx, void *frame_state) { \
  if (!frame_state) return nullptr; \
  auto* f = static_cast<Function*>(func); \
  auto* dst = f->env.AllocateRegister(); \
  return CppType::create( \
      dst, as_reg(receiver), name_idx, \
      *static_cast<const FrameState*>(frame_state)); \
}

DEOPT_NAMEIDX1_FACTORY(load_module_method_cached, LoadModuleMethodCached)
DEOPT_NAMEIDX1_FACTORY(load_method_cached, LoadMethodCached)
DEOPT_NAMEIDX1_FACTORY(load_module_attr_cached, LoadModuleAttrCached)
DEOPT_NAMEIDX1_FACTORY(load_attr_cached, LoadAttrCached)
#undef DEOPT_NAMEIDX1_FACTORY

HirInstr hir_c_create_store_attr_cached(HirFunction func,
    HirRegister obj, HirRegister value, int name_idx, void *frame_state) {
  if (!frame_state) return nullptr;
  auto* f = static_cast<Function*>(func);
  return StoreAttrCached::create(
      as_reg(obj), as_reg(value), name_idx,
      *static_cast<const FrameState*>(frame_state));
}

/* ---- Tier 3: CheckField + LoadAttr + LoadArrayItem + setGuiltyReg ---- */

HirInstr hir_c_create_check_field(HirFunction func, HirRegister src,
    void *name, void *frame_state) {
  if (!frame_state) return nullptr;
  auto* f = static_cast<Function*>(func);
  auto* dst = f->env.AllocateRegister();
  return CheckField::create(
      dst, as_reg(src),
      BorrowedRef<>(static_cast<PyObject*>(name)),
      *static_cast<const FrameState*>(frame_state));
}

HirInstr hir_c_create_load_attr(HirFunction func, HirRegister receiver,
    int name_idx, void *frame_state, int already_optimized) {
  if (!frame_state) return nullptr;
  auto* f = static_cast<Function*>(func);
  auto* dst = f->env.AllocateRegister();
  return LoadAttr::create(
      dst, as_reg(receiver), name_idx,
      *static_cast<const FrameState*>(frame_state),
      already_optimized != 0);
}

HirInstr hir_c_create_load_array_item(HirFunction func,
    HirRegister arr, HirRegister idx, HirRegister container,
    intptr_t offset, HirType type) {
  auto* f = static_cast<Function*>(func);
  auto* dst = f->env.AllocateRegister();
  Type cpp_type = Type::fromHirType(type);
  return LoadArrayItem::create(
      dst, as_reg(arr), as_reg(idx), as_reg(container),
      static_cast<ssize_t>(offset), cpp_type);
}

void hir_c_set_guilty_reg(HirInstr instr, HirRegister reg) {
  static_cast<DeoptBase*>(as_instr(instr))->setGuiltyReg(as_reg(reg));
}

void hir_c_set_descr(HirInstr instr, const char *descr) {
  static_cast<DeoptBase*>(as_instr(instr))->setDescr(std::string(descr));
}

HirInstr hir_c_create_guard(HirRegister src) {
  return Guard::create(as_reg(src));
}

/* ---- Tier 5: Variable-arity + infrastructure factories ---- */

HirInstr hir_c_create_vectorcall(HirFunction func, size_t n_operands,
                                  uint32_t flags, void *frame_state) {
  auto* f = static_cast<Function*>(func);
  auto* dst = f->env.AllocateRegister();
  if (frame_state) {
    return VectorCall::create(
        n_operands, dst, static_cast<CallFlags>(flags),
        *static_cast<const FrameState*>(frame_state));
  }
  return VectorCall::create(
      n_operands, dst, static_cast<CallFlags>(flags));
}

HirInstr hir_c_create_call_static(HirFunction func, size_t n_operands,
                                   void *addr, HirType ret_type) {
  auto* f = static_cast<Function*>(func);
  auto* dst = f->env.AllocateRegister();
  Type cpp_type = Type::fromHirType(ret_type);
  return CallStatic::create(n_operands, dst, addr, cpp_type);
}

HirInstr hir_c_create_call_static_reg(size_t n_operands, HirRegister dst,
                                       void *addr, HirType ret_type) {
  Type cpp_type = Type::fromHirType(ret_type);
  return CallStatic::create(n_operands, as_reg(dst), addr, cpp_type);
}

HirInstr hir_c_create_deopt_patchpoint(void *patcher) {
  return DeoptPatchpoint::create(
      static_cast<jit::JumpPatcher*>(patcher));
}

HirInstr hir_c_create_snapshot(void *frame_state) {
  if (!frame_state) return nullptr;
  return Snapshot::create(
      *static_cast<const FrameState*>(frame_state));
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
  auto* f = static_cast<Function*>(func);
  auto* dst = f->env.AllocateRegister();
  return FillTypeAttrCache::create(
      dst, as_reg(receiver), name_idx, cache_id,
      *static_cast<const FrameState*>(frame_state));
}

HirInstr hir_c_create_fill_type_method_cache(HirFunction func,
    HirRegister receiver, int name_idx, int cache_id, void *frame_state) {
  if (!frame_state) return nullptr;
  auto* f = static_cast<Function*>(func);
  auto* dst = f->env.AllocateRegister();
  return FillTypeMethodCache::create(
      dst, as_reg(receiver), name_idx, cache_id,
      *static_cast<const FrameState*>(frame_state));
}

/* ---- Simple DeoptBase factories (2-operand + FrameState, no custom fields) ---- */

#define DEOPT2_FACTORY(name, CppType) \
HirInstr hir_c_create_##name(HirFunction func, HirRegister lhs, \
                              HirRegister rhs, void *frame_state) { \
  if (!frame_state) return nullptr; \
  auto* f = static_cast<Function*>(func); \
  auto* dst = f->env.AllocateRegister(); \
  return CppType::create( \
      dst, as_reg(lhs), as_reg(rhs), \
      *static_cast<const FrameState*>(frame_state)); \
}

DEOPT2_FACTORY(dict_subscr, DictSubscr)
DEOPT2_FACTORY(unicode_subscr, UnicodeSubscr)
DEOPT2_FACTORY(unicode_repeat, UnicodeRepeat)
DEOPT2_FACTORY(unicode_concat, UnicodeConcat)
DEOPT2_FACTORY(list_append, ListAppend)
DEOPT2_FACTORY(is_instance, IsInstance)
#undef DEOPT2_FACTORY

HirInstr hir_c_create_get_length(HirFunction func, HirRegister src,
                                  void *frame_state) {
  if (!frame_state) return nullptr;
  auto* f = static_cast<Function*>(func);
  auto* dst = f->env.AllocateRegister();
  return GetLength::create(
      dst, as_reg(src),
      *static_cast<const FrameState*>(frame_state));
}

HirInstr hir_c_create_long_in_place_op(HirFunction func, int32_t op_kind,
                                        HirRegister left, HirRegister right,
                                        void *frame_state) {
  if (!frame_state) return nullptr;
  auto* f = static_cast<Function*>(func);
  auto* dst = f->env.AllocateRegister();
  return LongInPlaceOp::create(
      dst, static_cast<InPlaceOpKind>(op_kind),
      as_reg(left), as_reg(right),
      *static_cast<const FrameState*>(frame_state));
}

/* ---- FloatBinaryOp / LongBinaryOp / IsNegativeAndErrOccurred factories ---- */

HirInstr hir_c_create_float_binary_op(HirFunction func, int32_t op_kind,
                                       HirRegister left, HirRegister right,
                                       void *frame_state) {
  if (!frame_state) return nullptr;
  auto* f = static_cast<Function*>(func);
  auto* dst = f->env.AllocateRegister();
  return FloatBinaryOp::create(
      dst, static_cast<BinaryOpKind>(op_kind),
      as_reg(left), as_reg(right),
      *static_cast<const FrameState*>(frame_state));
}

HirInstr hir_c_create_long_binary_op(HirFunction func, int32_t op_kind,
                                      HirRegister left, HirRegister right,
                                      void *frame_state) {
  if (!frame_state) return nullptr;
  auto* f = static_cast<Function*>(func);
  auto* dst = f->env.AllocateRegister();
  return LongBinaryOp::create(
      dst, static_cast<BinaryOpKind>(op_kind),
      as_reg(left), as_reg(right),
      *static_cast<const FrameState*>(frame_state));
}

HirInstr hir_c_create_is_neg_and_err(HirFunction func, HirRegister src,
                                      void *frame_state) {
  if (!frame_state) return nullptr;
  auto* f = static_cast<Function*>(func);
  auto* dst = f->env.AllocateRegister();
  return IsNegativeAndErrOccurred::create(
      dst, as_reg(src),
      *static_cast<const FrameState*>(frame_state));
}

/* ---- Branch/CondBranch factories (C++ bridge for Edge management) ---- */

HirInstr hir_c_create_branch_cpp(void *target_block) {
  auto* bb = static_cast<BasicBlock*>(target_block);
  return Branch::create(bb);
}

HirInstr hir_c_create_check_seq_bounds_reg(HirRegister dst, HirRegister seq,
                                            HirRegister idx, void *frame_state) {
  if (frame_state)
    return CheckSequenceBounds::create(
        as_reg(dst), as_reg(seq), as_reg(idx),
        *static_cast<const FrameState*>(frame_state));
  return CheckSequenceBounds::create(as_reg(dst), as_reg(seq), as_reg(idx));
}

HirInstr hir_c_create_check_field_reg(HirRegister dst, HirRegister src,
                                       void *name, void *frame_state) {
  if (frame_state)
    return CheckField::create(
        as_reg(dst), as_reg(src), static_cast<PyObject*>(name),
        *static_cast<const FrameState*>(frame_state));
  return CheckField::create(as_reg(dst), as_reg(src), static_cast<PyObject*>(name));
}

HirInstr hir_c_create_binary_op_reg(HirRegister dst, int32_t op_kind,
                                     HirRegister left, HirRegister right,
                                     void *frame_state) {
  return BinaryOp::create(
      as_reg(dst), static_cast<BinaryOpKind>(op_kind),
      as_reg(left), as_reg(right),
      *static_cast<const FrameState*>(frame_state));
}

HirInstr hir_c_create_guard_is_reg(HirRegister dst, void *target,
                                    HirRegister src) {
  return GuardIs::create(
      as_reg(dst), static_cast<PyObject*>(target), as_reg(src));
}

HirInstr hir_c_create_set_current_awaiter_reg(HirRegister src) {
  return SetCurrentAwaiter::create(as_reg(src));
}

HirInstr hir_c_create_decref_reg(HirRegister src) {
  return Decref::create(as_reg(src));
}

HirInstr hir_c_create_make_cell_reg(HirRegister dst, HirRegister src,
                                     void *frame_state) {
  return MakeCell::create(as_reg(dst), as_reg(src),
                           *static_cast<const FrameState*>(frame_state));
}

HirInstr hir_c_create_initial_yield_reg(HirRegister dst, void *frame_state) {
  return InitialYield::create(as_reg(dst),
                               *static_cast<const FrameState*>(frame_state));
}

HirInstr hir_c_create_load_arg_reg(HirRegister dst, int32_t idx, HirType type) {
  Type cpp_type = Type::fromHirType(type);
  return LoadArg::create(as_reg(dst), idx, cpp_type);
}

HirInstr hir_c_create_store_subscr_reg(HirRegister container, HirRegister sub,
                                        HirRegister value, void *frame_state) {
  return StoreSubscr::create(as_reg(container), as_reg(sub), as_reg(value),
                              *static_cast<const FrameState*>(frame_state));
}

HirInstr hir_c_create_set_set_item_reg(HirRegister dst, HirRegister set,
                                        HirRegister item, void *frame_state) {
  return SetSetItem::create(as_reg(dst), as_reg(set), as_reg(item),
                             *static_cast<const FrameState*>(frame_state));
}

HirInstr hir_c_create_in_place_op_reg(HirRegister dst, int32_t op_kind,
                                       HirRegister left, HirRegister right,
                                       void *frame_state) {
  return InPlaceOp::create(as_reg(dst), static_cast<InPlaceOpKind>(op_kind),
                            as_reg(left), as_reg(right),
                            *static_cast<const FrameState*>(frame_state));
}

HirInstr hir_c_create_compare_reg(HirRegister dst, int32_t op,
                                   HirRegister left, HirRegister right,
                                   void *frame_state) {
  return Compare::create(as_reg(dst), static_cast<CompareOp>(op),
                          as_reg(left), as_reg(right),
                          *static_cast<const FrameState*>(frame_state));
}

HirInstr hir_c_create_format_with_spec_reg(HirRegister dst, HirRegister value,
                                            HirRegister fmt_spec, void *frame_state) {
  return FormatWithSpec::create(as_reg(dst), as_reg(value), as_reg(fmt_spec),
                                 *static_cast<const FrameState*>(frame_state));
}

HirInstr hir_c_create_make_dict_reg(HirRegister dst, int32_t dict_size,
                                     void *frame_state) {
  return MakeDict::create(as_reg(dst), dict_size,
                           *static_cast<const FrameState*>(frame_state));
}

HirInstr hir_c_create_get_a_iter_reg(HirRegister dst, HirRegister src, void *fs) {
  return GetAIter::create(as_reg(dst), as_reg(src),
                           *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_get_a_next_reg(HirRegister dst, HirRegister src, void *fs) {
  return GetANext::create(as_reg(dst), as_reg(src),
                           *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_get_tuple_reg(HirRegister dst, HirRegister src, void *fs) {
  return GetTuple::create(as_reg(dst), as_reg(src),
                           *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_is_neg_and_err_reg(HirRegister dst, HirRegister src, void *fs) {
  return IsNegativeAndErrOccurred::create(as_reg(dst), as_reg(src),
                                           *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_load_cell_item_reg(HirRegister dst, HirRegister src) {
  return LoadCellItem::create(as_reg(dst), as_reg(src));
}
HirInstr hir_c_create_load_current_func_reg(HirRegister dst) {
  return LoadCurrentFunc::create(as_reg(dst));
}
HirInstr hir_c_create_load_eval_breaker_reg(HirRegister dst) {
  return LoadEvalBreaker::create(as_reg(dst));
}
HirInstr hir_c_create_load_frame_reg(void) {
  return LoadFrame::create();
}
HirInstr hir_c_create_load_var_object_size_reg(HirRegister dst, HirRegister src) {
  return LoadVarObjectSize::create(as_reg(dst), as_reg(src));
}
HirInstr hir_c_create_check_err_occurred_reg(void *frame_state) {
  return CheckErrOccurred::create(
      *static_cast<const FrameState*>(frame_state));
}

HirInstr hir_c_create_raise_reg(void *fs) {
  return Raise::create(*static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_wait_handle_release_reg(HirRegister src) {
  return WaitHandleRelease::create(as_reg(src));
}
HirInstr hir_c_create_make_set_reg(HirRegister dst, void *fs) {
  return MakeSet::create(as_reg(dst), *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_delete_attr_reg(HirRegister receiver, int32_t idx, void *fs) {
  return DeleteAttr::create(as_reg(receiver), idx, *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_delete_subscr_reg(HirRegister container, HirRegister sub, void *fs) {
  return DeleteSubscr::create(as_reg(container), as_reg(sub), *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_store_attr_reg(HirRegister receiver, HirRegister value, int32_t idx, void *fs) {
  return StoreAttr::create(as_reg(receiver), as_reg(value), idx, *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_swap_cell_item_reg(HirRegister dst, HirRegister cell, HirRegister value) {
  return SwapCellItem::create(as_reg(dst), as_reg(cell), as_reg(value));
}
HirInstr hir_c_create_steal_cell_item_reg(HirRegister dst, HirRegister cell) {
  return StealCellItem::create(as_reg(dst), as_reg(cell));
}
HirInstr hir_c_create_set_cell_item_reg(HirRegister cell, HirRegister value, HirRegister old) {
  return SetCellItem::create(as_reg(cell), as_reg(value), as_reg(old));
}
HirInstr hir_c_create_at_quiescent_state_reg(void) {
  return AtQuiescentState::create();
}
HirInstr hir_c_create_run_periodic_tasks_reg(HirRegister dst, void *fs) {
  return RunPeriodicTasks::create(as_reg(dst), *static_cast<const FrameState*>(fs));
}

HirInstr hir_c_create_wait_handle_load_waiter_reg(HirRegister dst, HirRegister src) {
  return WaitHandleLoadWaiter::create(as_reg(dst), as_reg(src));
}
HirInstr hir_c_create_wait_handle_load_coro_reg(HirRegister dst, HirRegister src) {
  return WaitHandleLoadCoroOrResult::create(as_reg(dst), as_reg(src));
}
HirInstr hir_c_create_set_update_reg(HirRegister dst, HirRegister set, HirRegister iter, void *fs) {
  return SetUpdate::create(as_reg(dst), as_reg(set), as_reg(iter), *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_dict_update_reg(HirRegister dst, HirRegister dict, HirRegister update, void *fs) {
  return DictUpdate::create(as_reg(dst), as_reg(dict), as_reg(update), *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_list_extend_reg(HirRegister dst, HirRegister list, HirRegister iter, void *fs) {
  return ListExtend::create(as_reg(dst), as_reg(list), as_reg(iter), *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_copy_dict_without_keys_reg(HirRegister dst, HirRegister subj, HirRegister keys, void *fs) {
  return CopyDictWithoutKeys::create(as_reg(dst), as_reg(subj), as_reg(keys), *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_make_tuple_from_list_reg(HirRegister dst, HirRegister list, void *fs) {
  return MakeTupleFromList::create(as_reg(dst), as_reg(list), *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_list_append_reg(HirRegister dst, HirRegister list, HirRegister item, void *fs) {
  return ListAppend::create(as_reg(dst), as_reg(list), as_reg(item), *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_check_freevar_reg(HirRegister dst, HirRegister src, void *name, void *fs) {
  return CheckFreevar::create(as_reg(dst), as_reg(src), static_cast<PyObject*>(name), *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_load_global_reg(HirRegister dst, int32_t name_idx, void *fs) {
  return LoadGlobal::create(as_reg(dst), name_idx, *static_cast<const FrameState*>(fs));
}

HirInstr hir_c_create_dict_merge_reg(HirRegister dst, HirRegister dict, HirRegister update, HirRegister func, void *fs) {
  return DictMerge::create(as_reg(dst), as_reg(dict), as_reg(update), as_reg(func), *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_dict_subscr_reg(HirRegister dst, HirRegister dict, HirRegister key, void *fs) {
  return DictSubscr::create(as_reg(dst), as_reg(dict), as_reg(key), *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_send_reg(HirRegister iter, HirRegister vout, HirRegister vin, void *fs) {
  return Send::create(as_reg(iter), as_reg(vout), as_reg(vin), *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_convert_value_reg(HirRegister dst, HirRegister value, int32_t conversion, void *fs) {
  return ConvertValue::create(as_reg(dst), as_reg(value), conversion, *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_unary_op_reg(HirRegister dst, int32_t op_kind, HirRegister operand, void *fs) {
  return UnaryOp::create(as_reg(dst), static_cast<UnaryOpKind>(op_kind), as_reg(operand), *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_import_from_reg(HirRegister dst, HirRegister name, int32_t name_idx, void *fs) {
  return ImportFrom::create(as_reg(dst), as_reg(name), name_idx, *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_invoke_iter_next_reg(HirRegister dst, HirRegister iter, void *fs) {
  return InvokeIterNext::create(as_reg(dst), as_reg(iter), *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_primitive_unbox_reg(HirRegister dst, HirRegister src, HirType type) {
  Type cpp_type = Type::fromHirType(type);
  return PrimitiveUnbox::create(as_reg(dst), as_reg(src), cpp_type);
}

HirInstr hir_c_create_make_tuple_reg(size_t n, HirRegister dst, void *fs) {
  return MakeTuple::create(n, as_reg(dst), *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_make_list_reg(size_t n, HirRegister dst, void *fs) {
  return MakeList::create(n, as_reg(dst), *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_tp_alloc_reg(HirRegister dst, void *pytype, void *fs) {
  return TpAlloc::create(as_reg(dst), static_cast<PyTypeObject*>(pytype), *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_unpack_ex_to_tuple_reg(HirRegister dst, HirRegister seq, int32_t before, int32_t after, void *fs) {
  return UnpackExToTuple::create(as_reg(dst), as_reg(seq), before, after, *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_load_method_reg(HirRegister dst, HirRegister receiver, int32_t name_idx, void *fs) {
  return LoadMethod::create(as_reg(dst), as_reg(receiver), name_idx, *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_load_special_reg(HirRegister dst, HirRegister self, int32_t oparg, void *fs) {
  return LoadSpecial::create(as_reg(dst), as_reg(self), oparg, *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_match_keys_reg(HirRegister dst, HirRegister subj, HirRegister keys, void *fs) {
  return MatchKeys::create(as_reg(dst), as_reg(subj), as_reg(keys), *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_raise_awaitable_error_reg(HirRegister type, int32_t is_aenter, void *fs) {
  return RaiseAwaitableError::create(as_reg(type), is_aenter, *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_format_value_reg(HirRegister dst, HirRegister fmt, HirRegister val, int32_t conv, void *fs) {
  return FormatValue::create(as_reg(dst), as_reg(fmt), as_reg(val), conv, *static_cast<const FrameState*>(fs));
}

HirInstr hir_c_create_eager_import_name_reg(HirRegister dst, int32_t name_idx, HirRegister fromlist, HirRegister level, void *fs) {
  return EagerImportName::create(as_reg(dst), name_idx, as_reg(fromlist), as_reg(level), *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_make_checked_dict_reg(HirRegister dst, int32_t size, HirType type, void *fs) {
  Type cpp_type = Type::fromHirType(type);
  return MakeCheckedDict::create(as_reg(dst), size, cpp_type, *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_make_checked_list_reg(int32_t size, HirRegister dst, HirType type, void *fs) {
  Type cpp_type = Type::fromHirType(type);
  return MakeCheckedList::create(size, as_reg(dst), cpp_type, *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_make_function_reg(HirRegister dst, HirRegister code, HirRegister qualname, void *fs) {
  return MakeFunction::create(as_reg(dst), as_reg(code), as_reg(qualname), *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_build_template_reg(HirRegister strings, HirRegister interps, HirRegister dst, void *fs) {
  return BuildTemplate::create(as_reg(strings), as_reg(interps), as_reg(dst), *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_build_interpolation_reg(HirRegister dst, HirRegister val, HirRegister str, HirRegister fmt, int32_t conv, void *fs) {
  return BuildInterpolation::create(as_reg(dst), as_reg(val), as_reg(str), as_reg(fmt), conv, *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_load_attr_reg2(HirRegister dst, HirRegister receiver, int32_t name_idx, void *fs) {
  return LoadAttr::create(as_reg(dst), as_reg(receiver), name_idx, *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_init_frame_cell_vars_reg(HirRegister func, int32_t nfree) {
  return InitFrameCellVars::create(as_reg(func), nfree);
}

HirInstr hir_c_create_store_field_reg(HirRegister receiver, const char *name, intptr_t offset, HirRegister value, HirType type, HirRegister previous) {
  Type cpp_type = Type::fromHirType(type);
  return StoreField::create(as_reg(receiver), name, offset, as_reg(value), cpp_type, as_reg(previous));
}
HirInstr hir_c_create_yield_and_yield_from_reg(HirRegister dst, HirRegister waiter, HirRegister coro, void *fs) {
  return YieldAndYieldFrom::create(as_reg(dst), as_reg(waiter), as_reg(coro), *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_yield_from_handle_stop_async_reg(HirRegister dst, HirRegister send, HirRegister awaitable, void *fs) {
  return YieldFromHandleStopAsyncIteration::create(as_reg(dst), as_reg(send), as_reg(awaitable), *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_call_ex_reg(HirRegister dst, HirRegister func, HirRegister pargs, HirRegister kwargs, uint32_t flags, void *fs) {
  return CallEx::create(as_reg(dst), as_reg(func), as_reg(pargs), as_reg(kwargs), static_cast<CallFlags>(flags), *static_cast<const FrameState*>(fs));
}
HirInstr hir_c_create_import_name_reg(HirRegister dst, int32_t name_idx, HirRegister fromlist, HirRegister level, void *fs) {
  return ImportName::create(as_reg(dst), name_idx, as_reg(fromlist), as_reg(level), *static_cast<const FrameState*>(fs));
}

HirInstr hir_c_create_call_method_reg(size_t n_operands, HirRegister dst, uint32_t flags) {
  return CallMethod::create(n_operands, as_reg(dst), static_cast<CallFlags>(flags));
}
HirInstr hir_c_create_call_static_ret_void_reg(size_t n_operands, void *addr) {
  return CallStaticRetVoid::create(n_operands, addr);
}
HirInstr hir_c_create_invoke_static_function_reg(size_t n_operands, HirRegister dst, void *func, HirType ret_type) {
  Type cpp_type = Type::fromHirType(ret_type);
  return InvokeStaticFunction::create(n_operands, as_reg(dst), static_cast<PyFunctionObject*>(func), cpp_type);
}

HirInstr hir_c_create_cond_branch_iter_not_done_cpp(
    HirRegister src, void *body_block, void *done_block) {
  return CondBranchIterNotDone::create(
      as_reg(src), static_cast<BasicBlock*>(body_block),
      static_cast<BasicBlock*>(done_block));
}

HirInstr hir_c_create_int_convert_reg(HirRegister dst, HirRegister src,
                                       HirType type) {
  Type cpp_type = Type::fromHirType(type);
  return IntConvert::create(as_reg(dst), as_reg(src), cpp_type);
}

HirInstr hir_c_create_get_iter_reg(HirRegister dst, HirRegister src,
                                    void *frame_state) {
  return GetIter::create(as_reg(dst), as_reg(src),
                          *static_cast<const FrameState*>(frame_state));
}

HirInstr hir_c_create_load_field_address_reg(HirRegister dst, HirRegister object,
                                              HirRegister offset) {
  return LoadFieldAddress::create(as_reg(dst), as_reg(object), as_reg(offset));
}

HirInstr hir_c_create_yield_value_reg(HirRegister dst, HirRegister src,
                                       void *frame_state) {
  return YieldValue::create(as_reg(dst), as_reg(src),
                             *static_cast<const FrameState*>(frame_state));
}

HirInstr hir_c_create_yield_from_reg(HirRegister dst, HirRegister send_value,
                                      HirRegister iter, void *frame_state) {
  return YieldFrom::create(as_reg(dst), as_reg(send_value), as_reg(iter),
                            *static_cast<const FrameState*>(frame_state));
}

HirInstr hir_c_create_check_var_reg(HirRegister dst, HirRegister src,
                                     void *name, void *frame_state) {
  return CheckVar::create(as_reg(dst), as_reg(src),
                           static_cast<PyObject*>(name),
                           *static_cast<const FrameState*>(frame_state));
}

HirInstr hir_c_create_set_dict_item_reg(HirRegister dst, HirRegister dict,
                                         HirRegister key, HirRegister value,
                                         void *frame_state) {
  return SetDictItem::create(as_reg(dst), as_reg(dict), as_reg(key),
                              as_reg(value),
                              *static_cast<const FrameState*>(frame_state));
}

HirInstr hir_c_create_load_tuple_item_reg(HirRegister dst, HirRegister tuple,
                                           int32_t idx) {
  return LoadTupleItem::create(as_reg(dst), as_reg(tuple), idx);
}

HirInstr hir_c_create_is_truthy_reg(HirRegister dst, HirRegister src,
                                     void *frame_state) {
  return IsTruthy::create(as_reg(dst), as_reg(src),
                           *static_cast<const FrameState*>(frame_state));
}

HirInstr hir_c_create_get_second_output_reg(HirRegister dst, HirType type,
                                             HirRegister src) {
  Type cpp_type = Type::fromHirType(type);
  return GetSecondOutput::create(as_reg(dst), cpp_type, as_reg(src));
}

HirInstr hir_c_create_set_function_attr_reg(HirRegister value, HirRegister base,
                                             int32_t field) {
  return SetFunctionAttr::create(as_reg(value), as_reg(base),
                                  static_cast<FunctionAttr>(field));
}

HirInstr hir_c_create_check_neg_reg(HirRegister dst, HirRegister src,
                                     void *frame_state) {
  if (frame_state)
    return CheckNeg::create(as_reg(dst), as_reg(src),
                             *static_cast<const FrameState*>(frame_state));
  return CheckNeg::create(as_reg(dst), as_reg(src));
}

HirInstr hir_c_create_get_length_reg(HirRegister dst, HirRegister src,
                                      void *frame_state) {
  if (frame_state)
    return GetLength::create(as_reg(dst), as_reg(src),
                              *static_cast<const FrameState*>(frame_state));
  return GetLength::create(as_reg(dst), as_reg(src));
}

HirInstr hir_c_create_primitive_box_reg(HirRegister dst, HirRegister src,
                                         HirType type, void *frame_state) {
  Type cpp_type = Type::fromHirType(type);
  return PrimitiveBox::create(as_reg(dst), as_reg(src), cpp_type,
                               *static_cast<const FrameState*>(frame_state));
}

HirInstr hir_c_create_load_array_item_reg(HirRegister dst, HirRegister arr,
                                           HirRegister idx, HirRegister container,
                                           intptr_t offset, HirType type) {
  Type cpp_type = Type::fromHirType(type);
  return LoadArrayItem::create(as_reg(dst), as_reg(arr), as_reg(idx),
                                as_reg(container), offset, cpp_type);
}

HirInstr hir_c_create_vectorcall_reg(size_t n_operands, HirRegister dst,
                                      uint32_t flags) {
  return VectorCall::create(
      n_operands, as_reg(dst), static_cast<CallFlags>(flags));
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
  Type cpp_type = Type::fromHirType(type);
  return LoadField::create(
      as_reg(dst), as_reg(receiver), name, offset, cpp_type, borrowed != 0);
}

HirInstr hir_c_create_guard_type_reg(HirRegister dst, HirType target,
                                      HirRegister src) {
  Type cpp_target = Type::fromHirType(target);
  return GuardType::create(as_reg(dst), cpp_target, as_reg(src));
}

HirInstr hir_c_create_refine_type_reg(HirRegister dst, HirType type,
                                       HirRegister src) {
  Type cpp_type = Type::fromHirType(type);
  return RefineType::create(as_reg(dst), cpp_type, as_reg(src));
}

HirInstr hir_c_create_check_exc_reg(HirRegister dst, HirRegister src) {
  return CheckExc::create(as_reg(dst), as_reg(src));
}

HirInstr hir_c_create_deopt(void) {
  return Deopt::create();
}

HirInstr hir_c_create_return(HirRegister src, HirType type) {
  Type cpp_type = Type::fromHirType(type);
  return Return::create(as_reg(src), cpp_type);
}

/* ---- Frame state ---- */

void *hir_get_frame_state(HirInstr instr) {
  return const_cast<FrameState*>(get_frame_state(*as_instr(instr)));
}

} /* extern "C" */
