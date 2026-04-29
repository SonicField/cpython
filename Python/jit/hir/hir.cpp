// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cinderx/Jit/hir/hir.h"

#include "cinderx/Common/log.h"
#include "cinderx/Jit/hir/hir_basic_block_c.h"
#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/hir/hir_instr_info_c.h"
#include "cinderx/Jit/hir/hir_operand_types_c.h"
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
  if (auto fs = frameState()) {
    if (!fs->visitUses(func)) {
      return false;
    }
  }
  for (auto& reg_state : live_regs_) {
    if (!func(reg_state.reg)) {
      return false;
    }
  }
  if (guilty_reg_ != nullptr) {
    if (!func(guilty_reg_)) {
      return false;
    }
  }
  return true;
}

void DeoptBase::sortLiveRegs() {
  hir_c_deopt_sort_live_regs(this);

  if (kPyDebug) {
    // Check for uniqueness after sorting rather than inside the predicate
    // passed to qsort(), in case sort performs extra comparisons to
    // sanity-check our predicate.
    auto it = std::adjacent_find(
        live_regs_.begin(),
        live_regs_.end(),
        [](const RegState& a, const RegState& b) { return a.reg == b.reg; });
    JIT_DCHECK(it == live_regs_.end(), "Register {} is live twice", *it->reg);
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
  JIT_DCHECK(NumOperands() == args.size(), "arg mismatch");
  size_t n = args.size();

  if (n == 0) {
    hir_c_phi_apply_args(this, nullptr, nullptr, 0);
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
  qsort(pairs, n, sizeof(HirPhiArgPair), hir_c_phi_pair_cmp_by_block_id);

  void** keys = static_cast<void**>(malloc(n * sizeof(void*)));
  void** values = static_cast<void**>(malloc(n * sizeof(void*)));
  for (size_t j = 0; j < n; ++j) {
    keys[j] = pairs[j].key;
    values[j] = pairs[j].value;
  }
  free(pairs);

  hir_c_phi_apply_args(this, keys, values, n);

  free(keys);
  free(values);
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
  return from_;
}

BasicBlock* Edge::to() const {
  return to_;
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
  // hir_c_destroy_frame_state() since FrameState still has std::vector.
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

namespace {
inline HirOperandTypeEntry encode_operand_type(jit::hir::OperandType ot) {
  HirOperandTypeEntry e;
  e.kind = static_cast<int>(ot.kind);
  e.type = jit::hir::Type::toHirType(ot.type);
  return e;
}
}  // namespace

extern "C" HirOperandTypeEntry hir_primitive_compare_operand_type_c(
    const void *instr, size_t i) {
  auto* p = static_cast<const jit::hir::PrimitiveCompare*>(instr);
  return encode_operand_type(p->GetOperandTypeImpl(i));
}

extern "C" HirOperandTypeEntry hir_primitive_unbox_operand_type_c(
    const void *instr, size_t i) {
  auto* p = static_cast<const jit::hir::PrimitiveUnbox*>(instr);
  return encode_operand_type(p->GetOperandTypeImpl(i));
}

extern "C" HirOperandTypeEntry hir_return_operand_type_c(
    const void *instr, size_t i) {
  auto* p = static_cast<const jit::hir::Return*>(instr);
  return encode_operand_type(p->GetOperandTypeImpl(i));
}

extern "C" HirOperandTypeEntry hir_use_type_operand_type_c(
    const void *instr, size_t i) {
  auto* p = static_cast<const jit::hir::UseType*>(instr);
  return encode_operand_type(p->GetOperandTypeImpl(i));
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
  return instr.IsLoadMethod() || instr.IsLoadMethodCached() ||
         instr.IsLoadModuleMethodCached();
}

bool isAnyLoadMethod(const Instr& instr) {
  if (isLoadMethodBase(instr)) {
    return true;
  }
  if (!instr.IsPhi() || instr.NumOperands() != 2) {
    return false;
  }
  const Instr* arg1 = instr.GetOperand(0)->instr();
  const Instr* arg2 = instr.GetOperand(1)->instr();
  return (arg1->IsLoadTypeMethodCacheEntryValue() &&
          arg2->IsFillTypeMethodCache()) ||
      (arg2->IsLoadTypeMethodCacheEntryValue() &&
       arg1->IsFillTypeMethodCache());
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
  auto orig_reg = reg;
  // Even though GuardIs is a passthrough, it verifies that a runtime value is a
  // specific object, breaking the dependency on the instruction that produced
  // the runtime value
  while (isPassthrough(*reg->instr()) && !(reg->instr()->IsGuardIs())) {
    reg = reg->instr()->GetOperand(0);
    JIT_DCHECK(reg != orig_reg, "Hit cycle while looking for model reg");
  }
  return reg;
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
  // This won't work correctly if this block has two incoming edges from the
  // same block, but we already can't handle that correctly with our current Phi
  // setup.
  forEachPhi([&](Phi& phi) {
    hir_c_phi_fixup_predecessor(&phi, old_pred, new_pred);
  });
}

void BasicBlock::addPhiPredecessor(BasicBlock* old_pred, BasicBlock* new_pred) {
  std::vector<Phi*> replacements;
  forEachPhi([&](Phi& phi) {
    for (auto block : phi.basic_blocks()) {
      if (block == old_pred) {
        replacements.push_back(&phi);
        break;
      }
    }
  });
  for (auto phi : replacements) {
    hir_c_phi_add_predecessor(phi, old_pred, new_pred);
  }
}

void BasicBlock::removePhiPredecessor(BasicBlock* old_pred) {
  std::vector<Phi*> all_phis;
  forEachPhi([&](Phi& phi) {
    all_phis.push_back(&phi);
  });
  for (auto phi : all_phis) {
    hir_c_phi_remove_predecessor(phi, old_pred);
  }
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
  pytype = other.pytype;
  Py_XINCREF(pytype);
}

TypedArgument& TypedArgument::operator=(const TypedArgument& other) {
  if (this != &other) {
    ThreadedCompileSerialize guard;
    Py_XDECREF(pytype);
    locals_idx = other.locals_idx;
    pytype = other.pytype;
    Py_XINCREF(pytype);
    optional = other.optional;
    exact = other.exact;
    jit_type = other.jit_type;
    thread_safe_flags = other.thread_safe_flags;
  }
  return *this;
}

unsigned long TypedArgument::threadSafeTpFlags() const {
  JIT_DCHECK(
      thread_safe_flags == (pytype->tp_flags & kThreadSafeFlagsMask),
      "thread safe flags changed");
  return thread_safe_flags;
}

Environment::~Environment() {
  // Serialize as we modify the ref-count of objects which may be widely
  // accessible.
  ThreadedCompileSerialize guard;
  references_.clear();
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
  // Serialize as we modify the ref-count to obj which may be widely accessible.
  ThreadedCompileSerialize guard;
  return references_.emplace(ThreadedRef<>::create(obj)).first->get();
}

PyObject* Environment::addReference(Ref<> obj) {
  // ThreadedRef cannot steal from Ref, so have to go through the raw pointer
  // overload and accept the extra increfs and decrefs.
  return addReference((PyObject*)obj);
}

const Environment::ReferenceSet& Environment::references() const {
  return *reinterpret_cast<const ReferenceSet*>(
      hir_c_env_references(const_cast<Environment*>(this)));
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
  switch (op.kind) {
    case Constraint::kType:
      return os << op.type;
    case Constraint::kOptObjectOrCIntOrCBool:
      return os << "(OptObject, CInt, CBool)";
    case Constraint::kOptObjectOrCInt:
      return os << "(OptObject, CInt)";
    case Constraint::kTupleExactOrCPtr:
      return os << "(TupleExact, CPtr)";
    case Constraint::kOptObjectOrCPtr:
      return os << "(OptObject, CPtr)";
    case Constraint::kOptObjectOrCUInt64:
      return os << "(OptObject, CUInt64)";
    case Constraint::kListOrChkList:
      return os << "(List, chklist)";
    case Constraint::kDictOrChkDict:
      return os << "(Dict, chkdict)";
    case Constraint::kMatchAllAsCInt:
      return os << "CInt";
    case Constraint::kMatchAllAsPrimitive:
      return os << "Primitive";
  }
  JIT_ABORT("unknown constraint");
}

const FrameState* get_frame_state(const Instr& instr) {
  return static_cast<const FrameState*>(hir_c_instr_get_frame_state(&instr));
}

FrameState* get_frame_state(Instr& instr) {
  return static_cast<FrameState*>(hir_c_instr_get_frame_state(&instr));
}

} // namespace jit::hir
