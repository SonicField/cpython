// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cinderx/Jit/hir/hir.h"

#include "cinderx/Common/log.h"
#include "cinderx/Jit/hir/hir_basic_block_c.h"
#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/hir/hir_instr_info_c.h"
#include "cinderx/Jit/hir/hir_operand_types_c.h"
#include "cinderx/Jit/hir/typed_argument_c.h"
#include "cinderx/Jit/hir/phx_ptr_set.h"        /* X3b Environment::references_ */
#include "cinderx/Jit/hir/phx_threaded_ref.h"   /* X3b incref/decref bridges */
#include "cinderx/Jit/threaded_compile.h"

#include <algorithm>
#include <cstring>

namespace jit::hir {

// Phase H1a: Cross-validate HirBasicBlock C struct against C++ BasicBlock.
static_assert(sizeof(HirBasicBlock) == sizeof(BasicBlock),
    "HirBasicBlock and BasicBlock size mismatch");
static_assert(offsetof(HirBasicBlock, id) == offsetof(BasicBlock, id),
    "HirBasicBlock.id offset mismatch");
static_assert(offsetof(HirBasicBlock, cfg_node) == offsetof(BasicBlock, cfg_node),
    "HirBasicBlock.cfg_node offset mismatch");
// IntrusiveListNode layout validation
static_assert(sizeof(HirIntrusiveListNode) == sizeof(IntrusiveListNode),
    "HirIntrusiveListNode size mismatch");
// Instr::List layout validation
static_assert(sizeof(HirInstrList) == sizeof(Instr::List),
    "HirInstrList size mismatch");

// T2-C3: GetOperandType dispatch.
// 164/168 opcodes use the C operand-type table (hir_operand_types_c.c).
// 4 types with instance-dependent GetOperandTypeImpl use C++ dispatch directly:
// PrimitiveCompare, PrimitiveUnbox, Return, UseType.
// No FOREACH_OPCODE or _OperandTypes mixin references — class deletion safe.

DeoptBase::DeoptBase(Opcode op) : Instr(op) {}

DeoptBase::DeoptBase(Opcode op, const FrameState& frame) : Instr(op) {
  setFrameState(frame);
}

DeoptBase::DeoptBase(const DeoptBase& other) : Instr(other) {
  hir_c_deopt_base_init_copy(this, &other);
}

DeoptBase::~DeoptBase() {
  hir_c_deopt_base_destroy(this);
}

const PhxRegStateArray& DeoptBase::live_regs() const {
  return *static_cast<const PhxRegStateArray*>(
      hir_c_deopt_live_regs(const_cast<DeoptBase*>(this)));
}

PhxRegStateArray& DeoptBase::live_regs() {
  return *static_cast<PhxRegStateArray*>(hir_c_deopt_live_regs(this));
}

// asDeoptBase() devirtualized in T2-C1 — implementation moved to Instr
// (opcode metadata check). DeoptBase no longer overrides.

bool DeoptBase::visitUsesDeopt(const std::function<bool(Register*&)>& func) {
  // Phase 4.A W7b Batch 74: thunk pattern matches Instr::visitUses
  // (line 356-366). hir_c_deopt_visit_uses_deopt walks frame_state +
  // live_regs + guilty_reg in pure-C; this shim wraps the std::function
  // callback as a C visitor + user pointer.
  auto thunk = +[](void **slot, void *user) -> int {
    auto& f = *static_cast<const std::function<bool(Register*&)>*>(user);
    Register*& reg_ref = *reinterpret_cast<Register**>(slot);
    return f(reg_ref) ? 1 : 0;
  };
  return hir_c_deopt_visit_uses_deopt(
      this,
      thunk,
      const_cast<std::function<bool(Register*&)>*>(&func)) != 0;
}

void DeoptBase::sortLiveRegs() {
  hir_c_deopt_sort_live_regs(this);

  if (kPyDebug) {
    // Phase 4.X-mini X-mini-b (Batch 80, E-4 discharge per supervisor
    // 22:42:40Z): std::adjacent_find lambda replaced with inline C
    // helper hir_c_deopt_find_adjacent_dup_reg (hir_instr_c.h). Same
    // post-sort uniqueness check; same fail-loud message format.
    Register* dup = static_cast<Register*>(
        hir_c_deopt_find_adjacent_dup_reg(this));
    JIT_DCHECK(dup == nullptr, "Register {} is live twice", *dup);
  }
}

void DeoptBase::setFrameState(const FrameState& state) {
  hir_c_deopt_set_frame_state(this, &state);
}

int DeoptBase::nonce() const {
  return hir_c_deopt_get_nonce(this);
}

void DeoptBase::set_nonce(int nonce) {
  hir_c_deopt_set_nonce(this, nonce);
}

const char* DeoptBase::descr() const {
  return hir_c_deopt_get_descr(this);
}

void DeoptBase::setDescr(const char* r) {
  hir_c_deopt_set_descr(this, r);
}

Register* DeoptBase::guiltyReg() const {
  return static_cast<Register*>(hir_c_deopt_get_guilty_reg(this));
}

void DeoptBase::setGuiltyReg(Register* reg) {
  hir_c_deopt_set_guilty_reg(this, reg);
}

// CallCFunc::Func enum count — pinned to the C-side name table in
// hir_c_call_cfunc_func_name (hir_instr_c.h). Update both together
// when CallCFunc_FUNCS gains/loses an entry.
namespace {
constexpr std::size_t kCallCFuncCount = []() {
  std::size_t count = 0;
#define COUNT_FUNC(...) ++count;
  CallCFunc_FUNCS(COUNT_FUNC)
#undef COUNT_FUNC
  return count;
}();
static_assert(kCallCFuncCount == 4,
    "CallCFunc_FUNCS count diverged from hir_c_call_cfunc_func_name table");
}

std::string_view CallCFunc::funcName() const {
  return hir_c_call_cfunc_func_name(static_cast<int32_t>(func_));
}

void Phi::setArgs(const std::unordered_map<BasicBlock*, Register*>& args) {
  // Phase 4.A W2 Batch 68: post-iteration sort/split/apply/free factored
  // into hir_c_phi_apply_args_from_pairs (hir_basic_block_c.h). The C++
  // shim only does what STL forces — JIT_DCHECK invariant + iterate
  // unordered_map into the HirPhiArgPair buffer.
  JIT_DCHECK(NumOperands() == args.size(), "arg mismatch");
  size_t n = args.size();
  if (n == 0) {
    hir_c_phi_apply_args_from_pairs(this, nullptr, 0);
    return;
  }
  HirPhiArgPair* pairs =
      static_cast<HirPhiArgPair*>(malloc(n * sizeof(HirPhiArgPair)));
  size_t i = 0;
  for (auto& kv : args) {
    pairs[i].key = kv.first;
    pairs[i].value = kv.second;
    ++i;
  }
  hir_c_phi_apply_args_from_pairs(this, pairs, n);
  free(pairs);
}

std::size_t Phi::blockIndex(const BasicBlock* block) const {
  return hir_phi_block_index(this,
      reinterpret_cast<const HirBasicBlock*>(block));
}

Edge::Edge(const Edge& other) {
  hir_c_edge_copy_init(reinterpret_cast<HirEdge*>(this),
                       reinterpret_cast<const HirEdge*>(&other));
}

Edge::~Edge() {
  hir_edge_destroy(reinterpret_cast<HirEdge*>(this));
}

BasicBlock* Edge::from() const {
  // Phase 4.A W3 Batch 69: read-only endpoint accessor delegates to C
  // (hir_edge_from). Mutators (set_from/set_to) STAY C++ per
  // feedback_edge_management — they manage in_edges_ on the target
  // BasicBlock and must route through the C++ bridge.
  return reinterpret_cast<BasicBlock*>(
      hir_edge_from(reinterpret_cast<const HirEdge*>(this)));
}

BasicBlock* Edge::to() const {
  // Phase 4.A W3 Batch 69: see Edge::from() comment above.
  return reinterpret_cast<BasicBlock*>(
      hir_edge_to(reinterpret_cast<const HirEdge*>(this)));
}

void Edge::set_from(BasicBlock* new_from) {
  hir_edge_set_from(
      reinterpret_cast<HirEdge*>(this),
      reinterpret_cast<HirBasicBlock*>(new_from));
}

void Edge::set_to(BasicBlock* new_to) {
  hir_edge_set_to(
      reinterpret_cast<HirEdge*>(this),
      reinterpret_cast<HirBasicBlock*>(new_to));
}

void* Instr::allocate(std::size_t fixed_size, std::size_t num_operands) {
  return hir_c_instr_allocate(fixed_size, num_operands);
}

void* Instr::operator new(std::size_t count, void* ptr) {
  return ::operator new(count, ptr);
}

void Instr::operator delete(void* ptr) {
  hir_c_instr_free(ptr);
}

void Instr::Destroy(Instr* instr) {
  // H2-C: Delegate to pure C implementation.
  // Per-type cleanup is handled explicitly in hir_c_destroy_instr_impl()
  // (hir_instr_c.h). FrameState deletion goes through the C++ helper
  // hir_c_destroy_frame_state() to chain ~FrameState (PhxPtrArray
  // localsplus/stack + PhxExecBlockArray block_stack member dtors).
  hir_c_destroy_instr_impl(instr);
}

Instr::Instr(Opcode opcode) {
  hir_c_instr_init(this, static_cast<int32_t>(opcode));
}

Instr::Instr(const Instr& other) {
  hir_c_instr_init_copy(this, &other);
}

std::string_view Instr::opname() const {
  return hir_opcode_name(static_cast<HirOpcode>(opcode_));
}

std::size_t Instr::NumOperands() const {
  return *(reinterpret_cast<const std::size_t*>(this) - 1);
}

Register* Instr::GetOperand(std::size_t i) const {
  return static_cast<Register*>(hir_c_get_operand(this, i));
}

std::span<Register* const> Instr::GetOperands() const {
  return {operands(), NumOperands()};
}

// Phase 4.A Batch 14: Instr::GetOperandType port. Pure-C dispatcher
// (hir_c_instr_get_operand_type) routes 4 instance-dependent opcodes
// to extern "C" wrappers below; all others use the static
// hir_operand_type_get_info table. C++ shim decodes
// HirOperandTypeEntry into OperandType.
}  // namespace jit::hir

extern "C" HirOperandTypeEntry hir_primitive_compare_operand_type_c(
    const void *instr, size_t i) {
  auto ot = static_cast<const jit::hir::PrimitiveCompare*>(instr)
                ->GetOperandTypeImpl(i);
  return hir_c_encode_operand_type(
      static_cast<int>(ot.kind), jit::hir::Type::toHirType(ot.type));
}

extern "C" HirOperandTypeEntry hir_primitive_unbox_operand_type_c(
    const void *instr, size_t i) {
  auto ot = static_cast<const jit::hir::PrimitiveUnbox*>(instr)
                ->GetOperandTypeImpl(i);
  return hir_c_encode_operand_type(
      static_cast<int>(ot.kind), jit::hir::Type::toHirType(ot.type));
}

extern "C" HirOperandTypeEntry hir_return_operand_type_c(
    const void *instr, size_t i) {
  auto ot = static_cast<const jit::hir::Return*>(instr)
                ->GetOperandTypeImpl(i);
  return hir_c_encode_operand_type(
      static_cast<int>(ot.kind), jit::hir::Type::toHirType(ot.type));
}

extern "C" HirOperandTypeEntry hir_use_type_operand_type_c(
    const void *instr, size_t i) {
  auto ot = static_cast<const jit::hir::UseType*>(instr)
                ->GetOperandTypeImpl(i);
  return hir_c_encode_operand_type(
      static_cast<int>(ot.kind), jit::hir::Type::toHirType(ot.type));
}

namespace jit::hir {

OperandType Instr::GetOperandType(std::size_t i) const {
  JIT_DCHECK(
      i < NumOperands(),
      "operand {} out of range (max is {})",
      i,
      NumOperands() - 1);
  HirOperandTypeEntry entry = hir_c_instr_get_operand_type(this, i);
  Constraint constraint = static_cast<Constraint>(entry.kind);
  if (constraint == Constraint::kType) {
    return Type::fromHirType(entry.type);
  }
  return constraint;
}

void Instr::SetOperand(std::size_t i, Register* reg) {
  hir_c_set_operand(this, i, reg);
}

// Phase 4.A Batch 10: pure-C ports of Snapshot::visitUses +
// DeoptBase::visitUsesDeopt (replaces Batch 9 extern "C" bridges).
// Two C-callable C++ helpers remain: FrameState::visitUses (data
// traversal, kept in C++ for now) and DeoptBase live_regs iteration
// (V4 pure-C deferred to Batch 11 with HirRegState struct).
}  // namespace jit::hir

extern "C" int hir_frame_state_visit_uses_c(void *fs,
                                            HirRegVisitor visitor,
                                            void *user) {
  auto* f = static_cast<jit::hir::FrameState*>(fs);
  bool ok = f->visitUses([&](jit::hir::Register*& reg) {
    return visitor(reinterpret_cast<void**>(&reg), user) != 0;
  });
  return ok ? 1 : 0;
}

/* Phase 4.A Batch 18: copy-construct a new FrameState. C++ class with
 * std::vector members; the actual copy stays C++, exposed via this
 * thin extern "C" bridge so hir_c_deopt_set_frame_state can use it. */
extern "C" void *hir_make_frame_state_c(const void *src) {
  return new jit::hir::FrameState(*static_cast<const jit::hir::FrameState*>(src));
}

/* Phase 4.A Batch 25: allocate + placement-new a Phi shell with the
 * given operand count. Bypasses the std::unordered_map setArgs path
 * in Phi::create so the C-side add/remove predecessor flows can apply
 * pre-sorted parallel arrays directly via hir_c_phi_apply_args. */
extern "C" void *hir_make_phi_with_count_c(void *dst, size_t count) {
  return jit::hir::Phi::createWithCount(
      static_cast<jit::hir::Register*>(dst), count);
}

/* Phase 4.A Batch 26: heap-construct a Register with the given id.
 * Register's ctor stays C++ (initializer-list pinning + member inits);
 * exposed via this thin extern C bridge so the C-side Environment
 * AllocateRegister path can hand off creation without repeating the
 * C++ constructor body. */
extern "C" void *hir_make_register_c(int id) {
  return new jit::hir::Register(id);
}

namespace jit::hir {

bool Instr::visitUses(const std::function<bool(Register*&)>& func) {
  auto thunk = +[](void **slot, void *user) -> int {
    auto& f = *static_cast<const std::function<bool(Register*&)>*>(user);
    Register*& reg_ref = *reinterpret_cast<Register**>(slot);
    return f(reg_ref) ? 1 : 0;
  };
  return hir_c_instr_visit_uses(
      this,
      thunk,
      const_cast<std::function<bool(Register*&)>*>(&func)) != 0;
}

bool Instr::visitUses(const std::function<bool(Register*)>& func) const {
  return const_cast<Instr*>(this)->visitUses(
      [&func](Register*& reg) { return func(reg); });
}

bool Instr::Uses(Register* needle) const {
  /* Stop iteration on first match. user-data is the needle pointer;
   * visitor returns 0 to halt + signal found. */
  auto thunk = +[](void **slot, void *user) -> int {
    return (*slot == user) ? 0 : 1;
  };
  return hir_c_instr_visit_uses(
      const_cast<Instr*>(this), thunk, needle) == 0;
}

void Instr::ReplaceUsesOf(Register* orig, Register* replacement) {
  /* user-data: pair of {orig, replacement} pointers. Visitor writes
   * replacement into any slot equal to orig and continues iteration. */
  struct Pair { void *orig; void *replacement; } pair{orig, replacement};
  auto thunk = +[](void **slot, void *user) -> int {
    auto* p = static_cast<Pair*>(user);
    if (*slot == p->orig) {
      *slot = p->replacement;
    }
    return 1;
  };
  hir_c_instr_visit_uses(this, thunk, &pair);
}

Register* Instr::output() const {
  return static_cast<Register*>(hir_c_output(this));
}

void Instr::setOutput(Register* dst) {
  hir_c_set_output(this, dst);
}

bool Instr::IsTerminator() const {
  return hir_c_is_terminator(this) != 0;
}

std::size_t Instr::numEdges() const {
  return hir_c_num_edges(this);
}

Edge* Instr::edge(std::size_t i) {
  return const_cast<Edge*>(const_cast<const Instr*>(this)->edge(i));
}

const Edge* Instr::edge(std::size_t i) const {
  size_t n = hir_c_num_edges(this);
  JIT_CHECK(
      i < n,
      "Trying to access edge {} of {} but it only has {}",
      i,
      opname(),
      n);
  return reinterpret_cast<const Edge*>(
      hir_c_edge_at(const_cast<Instr*>(this), i));
}

std::span<const Edge> Instr::edges() const {
  size_t n = hir_c_num_edges(this);
  if (n == 0) {
    return {};
  }
  return std::span<const Edge>{
      reinterpret_cast<const Edge*>(
          hir_c_edge_at(const_cast<Instr*>(this), 0)),
      n};
}

BasicBlock* Instr::successor(std::size_t i) const {
  return edge(i)->to();
}

void Instr::set_successor(std::size_t i, BasicBlock* to) {
  edge(i)->set_to(to);
}

bool Instr::isReplayable() const {
  return hir_c_is_replayable(this) != 0;
}

void Instr::set_block(BasicBlock* block) {
  hir_c_set_block(this, block);
}

void Instr::InsertBefore(Instr& instr) {
  hir_c_insert_before_pure(
      this, &instr, reinterpret_cast<const HirBasicBlock*>(instr.block()));
}

void Instr::InsertAfter(Instr& instr) {
  hir_c_insert_after_pure(
      this, &instr, reinterpret_cast<const HirBasicBlock*>(instr.block()));
}

void Instr::ReplaceWith(Instr& instr) {
  hir_c_instr_replace_with(this, &instr);
}

void Instr::ExpandInto(const std::vector<Instr*>& expansion) {
  hir_c_instr_expand_into(
      this,
      reinterpret_cast<void**>(const_cast<Instr**>(expansion.data())),
      expansion.size());
}

void Instr::link(BasicBlock* block) {
  hir_c_instr_link(this, block);
}

void Instr::unlink() {
  hir_c_instr_unlink(this);
}

BasicBlock* Instr::block() const {
  return static_cast<BasicBlock*>(hir_c_block(this));
}

BCOffset Instr::bytecodeOffset() const {
  return BCOffset{hir_c_bytecode_offset(this)};
}

void Instr::setBytecodeOffset(BCOffset off) {
  hir_c_set_bytecode_offset(this, off.value());
}

void Instr::copyBytecodeOffset(const Instr& instr) {
  hir_c_copy_bytecode_offset(this, &instr);
}

const FrameState* Instr::getDominatingFrameState() const {
  return static_cast<const FrameState*>(hir_c_get_dominating_frame_state(this));
}

DeoptBase* Instr::asDeoptBase() {
  return static_cast<DeoptBase*>(hir_c_as_deopt_base(this));
}

const DeoptBase* Instr::asDeoptBase() const {
  return static_cast<const DeoptBase*>(
      hir_c_as_deopt_base(const_cast<Instr*>(this)));
}

void* Instr::base() {
  return hir_c_instr_base(this);
}

Register** Instr::operands() {
  return static_cast<Register**>(base());
}

Register* const* Instr::operands() const {
  return const_cast<Instr*>(this)->operands();
}

Register*& Instr::operandAt(std::size_t i) {
  JIT_DCHECK(
      i < NumOperands(),
      "operand {} out of range (max is {})",
      i,
      NumOperands() - 1);
  return operands()[i];
}

bool isLoadMethodBase(const Instr& instr) {
  return hir_c_is_load_method_base(&instr) != 0;
}

bool isAnyLoadMethod(const Instr& instr) {
  return hir_c_is_any_load_method(&instr) != 0;
}

/* extern decl for the pure-C predicate body (defined at
 * Python/jit/hir/pass_output_type_c.c:789). */
extern "C" int hir_is_passthrough_c(const void *instr);

bool isPassthrough(const Instr& instr) {
  // Tier 7 Batch 4 Cat-A: 180-line opcode-switch body extracted via
  // delegation to hir_is_passthrough_c (already exists, used by other
  // C-side passes per refcount_env_c.c:218 + pass_output_type_c.c:821).
  // The C version returns 0 for the original 'no-output' JIT_ABORT
  // cases (Branch/Decref/Deopt/etc.) instead of asserting; correct
  // callers should not query passthrough on those opcodes.
  return hir_is_passthrough_c(&instr) != 0;
}

Register* modelReg(Register* reg) {
  return static_cast<Register*>(hir_c_model_reg(reg));
}

Instr* BasicBlock::Append(Instr* instr) {
  return static_cast<Instr*>(
      hir_c_bb_append(reinterpret_cast<HirBasicBlock*>(this), instr));
}

void BasicBlock::retargetPreds(BasicBlock* target) {
  hir_bb_retarget_preds(
      reinterpret_cast<HirBasicBlock*>(this),
      reinterpret_cast<HirBasicBlock*>(target));
}

void BasicBlock::push_front(Instr* instr) {
  hir_c_bb_push_front(reinterpret_cast<HirBasicBlock*>(this), instr);
}

Instr* BasicBlock::pop_front() {
  return static_cast<Instr*>(
      hir_c_bb_pop_front(reinterpret_cast<HirBasicBlock*>(this)));
}

void BasicBlock::insert(Instr* instr, Instr::List::iterator it) {
  Instr* before = (it == instrs_.end()) ? nullptr : &*it;
  hir_c_bb_insert(reinterpret_cast<HirBasicBlock*>(this), instr, before);
}

void BasicBlock::clear() {
  hir_bb_clear(reinterpret_cast<HirBasicBlock*>(this));
}

BasicBlock::~BasicBlock() {
  hir_c_bb_destroy(reinterpret_cast<HirBasicBlock*>(this));
}

Instr* BasicBlock::GetTerminator() {
  return static_cast<Instr*>(
      hir_bb_get_terminator(reinterpret_cast<const HirBasicBlock*>(this)));
}

Snapshot* BasicBlock::entrySnapshot() {
  return static_cast<Snapshot*>(
      hir_bb_entry_snapshot(reinterpret_cast<const HirBasicBlock*>(this)));
}

bool BasicBlock::IsTrampoline() {
  return hir_bb_is_trampoline(reinterpret_cast<const HirBasicBlock*>(this));
}

void BasicBlock::fixupPhis(BasicBlock* old_pred, BasicBlock* new_pred) {
  // Phase 4.A W7b Batch 74: thunk pattern over hir_c_bb_for_each_phi.
  // The two-incoming-edges-from-same-block caveat from the prior C++
  // forEachPhi-loop comment applies here too — Phi setup currently
  // can't represent that case.
  struct FixupPair { void* old_pred; void* new_pred; } pair{old_pred, new_pred};
  auto thunk = +[](void* phi, void* user) -> int {
    auto* p = static_cast<FixupPair*>(user);
    hir_c_phi_fixup_predecessor(static_cast<Phi*>(phi), p->old_pred, p->new_pred);
    return 1;  // continue iteration
  };
  hir_c_bb_for_each_phi(
      reinterpret_cast<HirBasicBlock*>(this), thunk, &pair);
}

void BasicBlock::addPhiPredecessor(BasicBlock* old_pred, BasicBlock* new_pred) {
  // Phase 4.A W7c Batch 75: std::vector replaced with C-side scratch via
  // hir_c_bb_collect_phis_alloc. Per-Phi pred-presence filter dropped
  // from the C++ shim — hir_c_phi_collect_add_args (called inside
  // hir_c_phi_add_predecessor) returns 0 + early-returns when old_pred
  // is not a current predecessor, so unconditional iteration is correct
  // and the shim only owns the collect-then-mutate split required by
  // mid-iter Phi replacement.
  size_t n = 0;
  void** phis = hir_c_bb_collect_phis_alloc(
      reinterpret_cast<HirBasicBlock*>(this), &n);
  for (size_t i = 0; i < n; i++) {
    hir_c_phi_add_predecessor(phis[i], old_pred, new_pred);
  }
  free(phis);
}

void BasicBlock::removePhiPredecessor(BasicBlock* old_pred) {
  // Phase 4.A W7c Batch 75: std::vector replaced with C-side scratch.
  // See addPhiPredecessor comment above for the collect-then-mutate
  // rationale.
  size_t n = 0;
  void** phis = hir_c_bb_collect_phis_alloc(
      reinterpret_cast<HirBasicBlock*>(this), &n);
  for (size_t i = 0; i < n; i++) {
    hir_c_phi_remove_predecessor(phis[i], old_pred);
  }
  free(phis);
}

std::string_view GetCompareOpName(CompareOp op) {
  return hir_c_get_compare_op_name(static_cast<int>(op));
}

CompareOp ParseCompareOpName(std::string_view name) {
  return static_cast<CompareOp>(
      hir_c_parse_compare_op_name(name.data(), name.size()));
}

std::string_view GetPrimitiveCompareOpName(PrimitiveCompareOp op) {
  return hir_c_get_primitive_compare_op_name(static_cast<int>(op));
}

PrimitiveCompareOp ParsePrimitiveCompareOpName(std::string_view name) {
  return static_cast<PrimitiveCompareOp>(
      hir_c_parse_primitive_compare_op_name(name.data(), name.size()));
}

std::optional<PrimitiveCompareOp> toPrimitiveCompareOp(CompareOp op) {
  int result = hir_c_to_primitive_compare_op(static_cast<int>(op));
  if (result < 0) return std::nullopt;
  return static_cast<PrimitiveCompareOp>(result);
}

std::string_view GetBinaryOpName(BinaryOpKind op) {
  return hir_c_get_binary_op_name(static_cast<int>(op));
}

BinaryOpKind ParseBinaryOpName(std::string_view name) {
  return static_cast<BinaryOpKind>(
      hir_c_parse_binary_op_name(name.data(), name.size()));
}

std::string_view GetUnaryOpName(UnaryOpKind op) {
  return hir_c_get_unary_op_name(static_cast<int>(op));
}

UnaryOpKind ParseUnaryOpName(std::string_view name) {
  return static_cast<UnaryOpKind>(
      hir_c_parse_unary_op_name(name.data(), name.size()));
}

std::string_view GetPrimitiveUnaryOpName(PrimitiveUnaryOpKind op) {
  return hir_c_get_primitive_unary_op_name(static_cast<int>(op));
}

PrimitiveUnaryOpKind ParsePrimitiveUnaryOpName(std::string_view name) {
  return static_cast<PrimitiveUnaryOpKind>(
      hir_c_parse_primitive_unary_op_name(name.data(), name.size()));
}

std::string_view GetInPlaceOpName(InPlaceOpKind op) {
  return hir_c_get_inplace_op_name(static_cast<int>(op));
}

InPlaceOpKind ParseInPlaceOpName(std::string_view name) {
  return static_cast<InPlaceOpKind>(
      hir_c_parse_inplace_op_name(name.data(), name.size()));
}

const char* functionFieldName(FunctionAttr field) {
  return hir_c_get_function_field_name(static_cast<int>(field));
}

TypedArgument::TypedArgument(
    long locals_idx,
    PyTypeObject* pytype,
    int optional,
    int exact,
    Type jit_type)
    : locals_idx(locals_idx),
      optional(optional),
      exact(exact),
      jit_type(jit_type) {
  ThreadedCompileSerialize guard;
  this->pytype = pytype;
  Py_XINCREF(this->pytype);
  thread_safe_flags = pytype->tp_flags & kThreadSafeFlagsMask;
}

TypedArgument::~TypedArgument() {
  ThreadedCompileSerialize guard;
  Py_XDECREF(pytype);
  pytype = nullptr;
}

TypedArgument::TypedArgument(const TypedArgument& other)
    : locals_idx(other.locals_idx),
      optional(other.optional),
      exact(other.exact),
      jit_type(other.jit_type),
      thread_safe_flags(other.thread_safe_flags) {
  ThreadedCompileSerialize guard;
  // Phase 4.A W7d Batch 76: refcount swap via C body. pytype starts NULL
  // (default-init), so phx_typed_argument_pytype_swap acts as a pure
  // INCREF on the new value (Py_XDECREF(NULL) is a no-op).
  pytype = nullptr;
  phx_typed_argument_pytype_swap(&pytype, other.pytype);
}

TypedArgument& TypedArgument::operator=(const TypedArgument& other) {
  if (this != &other) {
    ThreadedCompileSerialize guard;
    // Phase 4.A W7d Batch 76: refcount swap delegated to C body
    // (typed_argument_c.{h,c}). The GIL + serialize guard stay C++ per
    // Q-W7-3 stay-C++ exception for ThreadedCompileSerialize RAII.
    // Field-copy assignments (incl. Type operator=) stay C++ — pure-POD
    // for locals_idx/optional/exact/thread_safe_flags; jit_type uses
    // Type::operator= which is non-POD C++.
    phx_typed_argument_pytype_swap(&pytype, other.pytype);
    locals_idx = other.locals_idx;
    optional = other.optional;
    exact = other.exact;
    jit_type = other.jit_type;
    thread_safe_flags = other.thread_safe_flags;
  }
  return *this;
}

unsigned long TypedArgument::threadSafeTpFlags() const {
  // Phase 4.A W1: tp_flags-mask read delegated to phx_typed_argument C body
  // (typed_argument_c.{h,c}). Header mask is _Static_assert'd ==
  // Py_TPFLAGS_BASETYPE, matching hir.h kThreadSafeFlagsMask. PyTypeObject
  // is a typedef for struct _typeobject (Include/pytypedefs.h:20) so the
  // pointer is implicitly compatible — no cast needed.
  JIT_DCHECK(
      thread_safe_flags ==
          phx_typed_argument_thread_safe_tp_flags(pytype),
      "thread safe flags changed");
  return thread_safe_flags;
}

Environment::~Environment() {
  /* X3b (Batch 96) E-1+E-2+E-3 DISCHARGE: references_ migrated from
   * std::unordered_set<ThreadedRef<>> to PhxPtrSet (X2b void*-keyed
   * open-address hash). Teardown: serialize guard (ThreadedCompileSerialize
   * stays C++ per Phoenix concurrency-infra class) + iterate raw slots +
   * phx_threaded_decref each + phx_ptr_set_destroy. Replaces the prior
   * STL-clear-via-~ThreadedRef path. */
  ThreadedCompileSerialize guard;
  for (size_t i = 0; i < phx_ptr_set_capacity(&references_); i++) {
    void *obj = phx_ptr_set_at(&references_, i);
    if (obj != NULL) {
      phx_threaded_decref(static_cast<PyObject *>(obj));
    }
  }
  phx_ptr_set_destroy(&references_);
  for (size_t i = 0; i < reg_count_; i++) {
    delete reg_data_[i];
  }
  free(reg_data_);
}

Register* Environment::AllocateRegister() {
  return static_cast<Register*>(hir_c_env_allocate_register(this));
}

Register* Environment::getRegister(int id) {
  return static_cast<Register*>(hir_env_get_register(this, id));
}

Register* Environment::addRegister(std::unique_ptr<Register> reg) {
  return static_cast<Register*>(
      hir_c_env_add_register(this, reg.release()));
}

PyObject* Environment::addReference(PyObject* obj) {
  /* X3b (Batch 96) E-1+E-2+E-3 DISCHARGE: references_ now PhxPtrSet
   * (void*-keyed). Dedup semantic preserved via contains-check BEFORE
   * phx_threaded_incref (theologian 04:25:09Z watchpoint #1: prevent
   * double-incref on duplicate adds; matches prior unordered_set::emplace
   * dedup which would discard the new ThreadedRef temporary on dup hit).
   * Serialize guard retained for ThreadedRef-class refcount safety. */
  ThreadedCompileSerialize guard;
  if (!phx_ptr_set_contains(&references_, obj)) {
    phx_threaded_incref(obj);
    phx_ptr_set_insert(&references_, obj);
  }
  return obj;
}

PyObject* Environment::addReference(Ref<> obj) {
  // Phase 4.A W7d Batch 76: STAY C++ per Q-W7-3 (overload-of stay-C++).
  // ThreadedRef cannot steal from Ref, so have to go through the raw pointer
  // overload and accept the extra increfs and decrefs.
  return addReference((PyObject*)obj);
}

const Environment::ReferenceSet& Environment::references() const {
  /* X3b: ReferenceSet now PhxPtrSet (POD); direct reference return,
   * no opaque-blob bridge cast needed. hir_c_env_references at
   * hir_instr_c.h:265 still exposes the field address as void* for
   * C-side consumers (offset preserved by HirEnvironmentLayoutVerifier). */
  return references_;
}

bool usesRuntimeFunc([[maybe_unused]] PyCodeObject* code) {
#if PY_VERSION_HEX < 0x030C0000
  return PyTuple_GET_SIZE(PyCode_GetFreevars(code)) > 0;
#else
  // In 3.12+ we always need the runtime function because we use it to
  // initialize the _PyInterpreterFrame object.
  return true;
#endif
}

const char* getInlineFailureMessage(InlineFailureType failure_type) {
  return hir_c_get_inline_failure_message(static_cast<int>(failure_type));
}

const char* getInlineFailureName(InlineFailureType failure_type) {
  return hir_c_get_inline_failure_name(static_cast<int>(failure_type));
}

std::ostream& operator<<(std::ostream& os, OperandType op) {
  if (op.kind == Constraint::kType) {
    return os << op.type;
  }
  return os << hir_c_constraint_name(static_cast<int>(op.kind));
}

const FrameState* get_frame_state(const Instr& instr) {
  return static_cast<const FrameState*>(hir_c_instr_get_frame_state(&instr));
}

FrameState* get_frame_state(Instr& instr) {
  return static_cast<FrameState*>(hir_c_instr_get_frame_state(&instr));
}

} // namespace jit::hir
