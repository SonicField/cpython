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

/* ---- Cross-boundary layout verification ---- */
/* Runs during JIT init to catch C/C++ bit-packing divergence. */

static void verify_hir_type_layout() {
    /* Create a known C++ Type, reinterpret as HirType, verify accessors agree */
    Type cpp_type = TLong;  /* known: kLong bits, kLifetimeTop, kSpecTop */
    const HirType *c_type = reinterpret_cast<const HirType*>(&cpp_type);

    assert(hir_type_bits(c_type) == Type::kLong &&
           "C/C++ HirType bits_ divergence");
    assert(hir_type_lifetime(c_type) == Type::kLifetimeTop &&
           "C/C++ HirType lifetime_ divergence");
    assert(hir_type_spec_kind(c_type) == 0 /* kSpecTop */ &&
           "C/C++ HirType spec_kind_ divergence");

    /* Verify a specialized type (kSpecObject) */
    Type cpp_obj = Type::fromObject(Py_None);
    const HirType *c_obj = reinterpret_cast<const HirType*>(&cpp_obj);
    assert(hir_type_has_object_spec(c_obj) &&
           "C/C++ HirType object spec divergence");
    assert(c_obj->pyobject == Py_None &&
           "C/C++ HirType object spec value divergence");

    /* Verify kSpecDouble */
    Type cpp_dbl = Type::fromCDouble(3.14);
    const HirType *c_dbl = reinterpret_cast<const HirType*>(&cpp_dbl);
    assert(hir_type_has_double_spec(c_dbl) &&
           "C/C++ HirType kSpecDouble divergence");
    assert(c_dbl->double_val == 3.14 &&
           "C/C++ HirType double value divergence");

    /* Verify kSpecInt */
    Type cpp_int = Type::fromCInt(42, TCInt64);
    const HirType *c_int = reinterpret_cast<const HirType*>(&cpp_int);
    assert(hir_type_has_int_spec(c_int) &&
           "C/C++ HirType kSpecInt divergence");
    assert(c_int->int_val == 42 &&
           "C/C++ HirType int value divergence");

    /* Verify kSpecType (Long is the most common type-specialized kind) */
    Type cpp_tyspec = Type::fromType(&PyLong_Type);
    const HirType *c_tyspec = reinterpret_cast<const HirType*>(&cpp_tyspec);
    assert(hir_type_has_type_spec(c_tyspec) &&
           "C/C++ HirType kSpecType divergence");
    assert(c_tyspec->pytype == &PyLong_Type &&
           "C/C++ HirType kSpecType value divergence");

    /* Verify kSpecTypeExact */
    Type cpp_tyexact = Type::fromTypeExact(&PyLong_Type);
    const HirType *c_tyexact = reinterpret_cast<const HirType*>(&cpp_tyexact);
    assert(hir_type_has_type_exact_spec(c_tyexact) &&
           "C/C++ HirType kSpecTypeExact divergence");
    assert(c_tyexact->pytype == &PyLong_Type &&
           "C/C++ HirType kSpecTypeExact value divergence");
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

PyObject *hir_type_as_object(const HirType *t) {
  const auto& type = *reinterpret_cast<const Type*>(t);
  return type.asObject();
}

int hir_type_is_exact(const HirType *t) {
  const auto& type = *reinterpret_cast<const Type*>(t);
  return type.isExact() ? 1 : 0;
}

int hir_type_has_known_destructor(const HirType *t) {
  const auto& type = *reinterpret_cast<const Type*>(t);
  return type.runtimePyTypeDestructor().has_value() ? 1 : 0;
}

PyTypeObject *hir_type_runtime_py_type(const HirType *t) {
  const auto& type = *reinterpret_cast<const Type*>(t);
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

/* ---- Instruction predicates (T2-D) ---- */

int hir_instr_is_condbranch(HirInstr instr) {
  return as_instr(instr)->IsCondBranch() ? 1 : 0;
}

int hir_instr_is_istruthy(HirInstr instr) {
  return as_instr(instr)->IsIsTruthy() ? 1 : 0;
}

int hir_instr_is_compare(HirInstr instr) {
  return as_instr(instr)->IsCompare() ? 1 : 0;
}

int hir_instr_is_vectorcall(HirInstr instr) {
  return as_instr(instr)->IsVectorCall() ? 1 : 0;
}

int hir_instr_opcode(HirInstr instr) {
  return static_cast<int>(as_instr(instr)->opcode());
}

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

HirBasicBlock hir_instr_block(HirInstr instr) {
  return as_instr(instr)->block();
}

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

/* ---- Frame state ---- */

void *hir_get_frame_state(HirInstr instr) {
  return const_cast<FrameState*>(get_frame_state(*as_instr(instr)));
}

} /* extern "C" */
