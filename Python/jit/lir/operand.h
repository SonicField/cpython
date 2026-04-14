// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "Python.h"
#include "cinderx/Common/log.h"
#include "cinderx/Jit/lir/arch.h"
#include "cinderx/Jit/lir/lir_c_api.h"
#include "cinderx/Jit/lir/type.h"

#include <cstdint>
#include <variant>

namespace jit::lir {

struct BasicBlock;
class Instruction;
class OperandBase;
class Operand;
class LinkedOperand;
class MemoryIndirect;

// Defines the interface for all the operand kinds.
//
// Phase 3D: Devirtualized. All data fields live in OperandBase; dispatch
// is via is_linked_ flag instead of vtable.  Operand and LinkedOperand
// are thin subclasses with constructors and setters only.
class OperandBase {
 public:
  using Type = OperandType;

  // Use PyMem_RawMalloc for all operand allocation — matches C-side
  // lir_operand_free() which uses PyMem_RawFree. Fixes ASAN
  // alloc-dealloc mismatch at the C/C++ boundary.
  void* operator new(size_t size) { return PyMem_RawCalloc(1, size); }
  void operator delete(void* ptr) { PyMem_RawFree(ptr); }

  OperandBase() = default;
  explicit OperandBase(Instruction* parent) : parent_instr_{parent} {}

  // Phase B5: Non-virtual destructor. Operand/LinkedOperand add no data
  // members, so destruction through OperandBase* is safe.
  // Uses lir_memind_free (C API) to avoid needing full MemoryIndirect definition.
  ~OperandBase() {
    if (!is_linked_ && type_ == kInd && value_.indirect) {
      lir_memind_free(reinterpret_cast<LirMemoryIndirect*>(value_.indirect));
      value_.indirect = nullptr;
    }
  }

  OperandBase(const OperandBase& ob)
      : parent_instr_{ob.parent_instr_},
        last_use_{ob.last_use_},
        is_linked_{ob.is_linked_},
        type_{ob.type_},
        data_type_{ob.data_type_},
        def_opnd_{ob.def_opnd_} {
    // value_ is not copied (same as original Operand copy ctor behavior).
  }

  // Phase B5: Delete copy assignment — raw MemoryIndirect* in value_.indirect
  // would be shallow-copied, creating double-free risk. Copy constructor
  // deliberately skips value_ (safe); assignment must not silently shallow-copy.
  OperandBase& operator=(const OperandBase&) = delete;

#define OPERAND_TYPE_DEFINES(V, ...) \
  using OperandType::k##V;           \
                                     \
  bool is##V() const {               \
    return type() == Type::k##V;     \
  }
  FOREACH_OPERAND_TYPE(OPERAND_TYPE_DEFINES)
#undef OPERAND_TYPE_DEFINES

#define OPERAND_DATA_TYPE_DEFINES(V, ...) using DataType::k##V;
  FOREACH_OPERAND_DATA_TYPE(OPERAND_DATA_TYPE_DEFINES)
#undef OPERAND_DATA_TYPE_DEFINES

  size_t sizeInBits() const;

  // Get the instruction using this operand.
  Instruction* instr() { return parent_instr_; }
  const Instruction* instr() const { return parent_instr_; }

  // Set and unset the instruction using this operand.
  void assignToInstr(Instruction* instr) { parent_instr_ = instr; }
  void releaseFromInstr() { parent_instr_ = nullptr; }

  bool isFp() const;
  bool isVecD() const;

  bool isLastUse() const { return last_use_; }
  void setLastUse() { last_use_ = true; }

  // Data accessors — dispatch via is_linked_ flag.
  // When linked, delegates to def_opnd_ (the defining Operand).
  // Defined out-of-class (bottom of header) because they reference Operand.
  uint64_t getConstant() const;
  double getFPConstant() const;
  PhyLocation getPhyRegister() const;
  PhyLocation getStackSlot() const;
  PhyLocation getPhyRegOrStackSlot() const;
  void* getMemoryAddress() const;
  MemoryIndirect* getMemoryIndirect() const;
  BasicBlock* getBasicBlock() const;
  uint64_t getConstantOrAddress() const;

  Operand* getDefine();
  const Operand* getDefine() const;

  DataType dataType() const;
  Type type() const;

  bool isLinked() const { return is_linked_; }

 // Phase B4d: All fields public for C struct compatibility.
 // B5 devirtualized dispatch — protected modifier no longer guards invariants.
 public:
  // Phase B5: Free owned MemoryIndirect before type change.
  // Uses lir_memind_free (C API) to avoid needing full MemoryIndirect definition.
  void clearIndirect() {
    if (type_ == kInd && value_.indirect) {
      lir_memind_free(reinterpret_cast<LirMemoryIndirect*>(value_.indirect));
      value_.indirect = nullptr;
    }
  }

  // Raw value accessor for diagnostics (works on non-linked operands only).
  uint64_t rawValue() const {
    switch (type_) {
      case kImm: return value_.constant;
      case kMem: return reinterpret_cast<uint64_t>(value_.address);
      case kLabel: return reinterpret_cast<uint64_t>(value_.block);
      case kInd: return reinterpret_cast<uint64_t>(value_.indirect);
      case kReg:
      case kStack: return static_cast<uint64_t>(value_.phy_loc.loc);
      case kVreg:
      case kNone: return 0;
    }
    JIT_ABORT("Unknown operand type");
  }

  Instruction* parent_instr_{nullptr};
  bool last_use_{false};
  bool is_linked_{false};

  // Fields from Operand (unused when is_linked_):
  Type type_{kNone};
  DataType data_type_{kObject};

  // Phase B5: Tagged union replaces std::variant. type_ is the discriminator.
  // kImm → constant, kMem → address, kLabel → block, kInd → indirect,
  // kReg/kStack → phy_loc, kNone/kVreg → unused.
  union {
    uint64_t constant;
    void* address;
    BasicBlock* block;
    MemoryIndirect* indirect;
    PhyLocation phy_loc;
  } value_{};

  // Field from LinkedOperand (unused when !is_linked_):
  Operand* def_opnd_{nullptr};
};

// Memory reference: [base_reg + index_reg * (2^index_multiplier) + offset]
class MemoryIndirect {
 public:
  void* operator new(size_t size) { return PyMem_RawCalloc(1, size); }
  void operator delete(void* ptr) { PyMem_RawFree(ptr); }

  explicit MemoryIndirect(Instruction* parent) : parent_(parent) {}
  ~MemoryIndirect() {
    delete base_reg_;
    delete index_reg_;
  }

  void setMemoryIndirect(Instruction* base, int32_t offset) {
    setMemoryIndirect(base, nullptr, 0, offset);
  }
  void setMemoryIndirect(PhyLocation base, int32_t offset = 0) {
    setMemoryIndirect(base, PhyLocation::REG_INVALID, 0, offset);
  }
  void setMemoryIndirect(
      PhyLocation base, PhyLocation index_reg, uint8_t multiplier) {
    setMemoryIndirect(base, index_reg, multiplier, 0);
  }
  void setMemoryIndirect(
      std::variant<Instruction*, PhyLocation> base,
      std::variant<Instruction*, PhyLocation> index,
      uint8_t multiplier, int32_t offset) {
    setBaseIndex(base_reg_, base);
    setBaseIndex(index_reg_, index);
    multiplier_ = multiplier;
    offset_ = offset;
  }

  OperandBase* getBaseRegOperand() const { return base_reg_; }
  OperandBase* getIndexRegOperand() const { return index_reg_; }
  uint8_t getMultipiler() const { return multiplier_; }
  int32_t getOffset() const { return offset_; }

 // Phase 3D: Fields and helpers public for inline implementation.
 public:
  void setBaseIndex(OperandBase*& opnd, Instruction* base_index) {
    if (opnd) { lir_operand_free(reinterpret_cast<LirOperand*>(opnd)); }
    if (base_index != nullptr) {
      opnd = reinterpret_cast<OperandBase*>(
          lir_operand_new_linked(
              reinterpret_cast<LirInstruction*>(parent_),
              reinterpret_cast<LirInstruction*>(base_index)));
    } else {
      opnd = nullptr;
    }
  }
  void setBaseIndex(OperandBase*& opnd, PhyLocation base_index) {
    if (opnd) { lir_operand_free(reinterpret_cast<LirOperand*>(opnd)); }
    if (base_index != PhyLocation::REG_INVALID) {
      auto* operand = lir_operand_new(
          reinterpret_cast<LirInstruction*>(parent_));
      lir_operand_set_phy_register(operand, {base_index.loc, base_index.bitSize});
      opnd = reinterpret_cast<OperandBase*>(operand);
    } else {
      opnd = nullptr;
    }
  }
  void setBaseIndex(OperandBase*& opnd,
      std::variant<Instruction*, PhyLocation> base_index) {
    if (Instruction** instrp = std::get_if<Instruction*>(&base_index)) {
      setBaseIndex(opnd, *instrp);
    } else {
      setBaseIndex(opnd, std::get<PhyLocation>(base_index));
    }
  }

  Instruction* parent_{nullptr};
  OperandBase* base_reg_{nullptr};
  OperandBase* index_reg_{nullptr};
  uint8_t multiplier_{0};
  int32_t offset_{0};
};

// An operand that is either an immediate value, or a value being defined by an
// instruction.
//
// Phase 3D: Devirtualized — all data fields now in OperandBase.
// Operand is a thin subclass with constructors and setters.
class Operand : public OperandBase {
 public:
  Operand() = default;
  explicit Operand(Instruction* parent) : OperandBase{parent} {}
  ~Operand() = default;

  // Only copies simple fields (type and data type) from operand.
  Operand(Instruction* parent, Operand* operand) : OperandBase(parent) {
    type_ = operand->type_;
    data_type_ = operand->data_type_;
  }
  Operand(Instruction* parent, DataType data_type, Type type, uint64_t data)
      : OperandBase(parent) {
    type_ = type;
    data_type_ = data_type;
    value_.constant = data;
  }
  Operand(Instruction* parent, Type type, double data)
      : OperandBase(parent) {
    type_ = type;
    data_type_ = kDouble;
    value_.constant = bit_cast<uint64_t>(data);
  }

  // Setters (modify base class fields directly):
  void setConstant(uint64_t n, DataType data_type = k64bit) {
    clearIndirect();
    type_ = kImm;
    value_.constant = n;
    data_type_ = data_type;
  }
  void setFPConstant(double n) {
    clearIndirect();
    type_ = kImm;
    value_.constant = bit_cast<uint64_t>(n);
    data_type_ = kDouble;
  }
  void setPhyRegister(PhyLocation reg) {
    clearIndirect();
    type_ = kReg;
    value_.phy_loc = reg;
  }
  void setStackSlot(PhyLocation slot) {
    clearIndirect();
    type_ = kStack;
    value_.phy_loc = slot;
  }
  void setPhyRegOrStackSlot(PhyLocation loc) {
    if (loc.loc < 0) { setStackSlot(loc); }
    else { setPhyRegister(loc); }
  }
  void setMemoryAddress(void* addr) {
    clearIndirect();
    type_ = kMem;
    value_.address = addr;
  }

  void setMemoryIndirect(Instruction* base, int32_t offset) {
    clearIndirect();
    type_ = kInd;
    auto* ind = new MemoryIndirect(instr());
    ind->setMemoryIndirect(base, offset);
    value_.indirect = ind;
  }
  void setMemoryIndirect(PhyLocation base, int32_t offset = 0) {
    clearIndirect();
    type_ = kInd;
    auto* ind = new MemoryIndirect(instr());
    ind->setMemoryIndirect(base, offset);
    value_.indirect = ind;
  }
  void setMemoryIndirect(
      PhyLocation base, PhyLocation index_reg, uint8_t multiplier) {
    clearIndirect();
    type_ = kInd;
    auto* ind = new MemoryIndirect(instr());
    ind->setMemoryIndirect(base, index_reg, multiplier);
    value_.indirect = ind;
  }
  void setMemoryIndirect(
      std::variant<Instruction*, PhyLocation> base,
      std::variant<Instruction*, PhyLocation> index,
      uint8_t multiplier, int32_t offset) {
    clearIndirect();
    type_ = kInd;
    auto* ind = new MemoryIndirect(instr());
    ind->setMemoryIndirect(base, index, multiplier, offset);
    value_.indirect = ind;
  }

  void setBasicBlock(BasicBlock* block) {
    clearIndirect();
    type_ = kLabel;
    data_type_ = kObject;
    value_.block = block;
  }
  void setDataType(DataType data_type) {
    data_type_ = data_type;
    if (type_ == kReg || type_ == kStack) {
      value_.phy_loc.bitSize = bitSize(data_type);
    }
  }
  void setNone() {
    clearIndirect();
    type_ = kNone;
  }
  void setVirtualRegister() {
    clearIndirect();
    type_ = kVreg;
  }
};

// An operand that points to the output value of an instruction.  Represents a
// def-use relationship.
//
// Can only be the input of an instruction.
//
// Phase 3D: Devirtualized — all data accessors now in OperandBase via
// is_linked_ dispatch.  LinkedOperand just has constructors and helpers.
class LinkedOperand : public OperandBase {
 public:
  explicit LinkedOperand(Instruction* def) {
    def_opnd_ = (def != nullptr)
        ? static_cast<Operand*>(jit_lir_instr_output(
              static_cast<JitLirInstr>(def)))
        : nullptr;
    is_linked_ = (def_opnd_ != nullptr);
  }
  LinkedOperand(Instruction* parent, Instruction* def)
      : LinkedOperand{def} {
    assignToInstr(parent);
  }

  ~LinkedOperand() = default;

  Operand* getLinkedOperand() { return def_opnd_; }
  const Operand* getLinkedOperand() const { return def_opnd_; }

  Instruction* getLinkedInstr() { return def_opnd_->instr(); }
  const Instruction* getLinkedInstr() const { return def_opnd_->instr(); }

  void setLinkedInstr(Instruction* def) {
    def_opnd_ = (def != nullptr)
        ? static_cast<Operand*>(jit_lir_instr_output(
              static_cast<JitLirInstr>(def)))
        : nullptr;
    is_linked_ = (def_opnd_ != nullptr);
  }
};

// OperandArg reqresents different operand data types, and is used as
// arguments to BasicBlock::allocateInstr* instructions. The latter
// will create the operands accordingly for the instructions after
// allocating them.
template <typename Type, bool Output>
struct OperandArg {
  explicit OperandArg(Type v, DataType dt = OperandBase::kObject)
      : value(v), data_type(dt) {}

  Type value;
  DataType data_type{OperandBase::kObject};
  static constexpr bool is_output = Output;
};

template <bool Output>
struct OperandArg<uint64_t, Output> {
  explicit OperandArg(uint64_t v, DataType dt = OperandBase::k64bit)
      : value(v), data_type(dt) {}

  uint64_t value;
  DataType data_type{OperandBase::k64bit};
  static constexpr bool is_output = Output;
};

// Operand is typed through its linked instruction.
template <>
struct OperandArg<Instruction*, false> {
  explicit OperandArg(Instruction* v) : value{v} {}

  Instruction* value{nullptr};
  static constexpr bool is_output = false;
};

template <bool Output>
struct OperandArg<MemoryIndirect, Output> {
  using Reg = std::variant<Instruction*, PhyLocation>;

  explicit OperandArg(Reg b, DataType dt = OperandBase::kObject)
      : base(b), data_type(dt) {}
  explicit OperandArg(Reg b, int32_t off, DataType dt = OperandBase::kObject)
      : base(b), offset(off), data_type(dt) {}
  OperandArg(Reg b, Reg i, DataType dt = OperandBase::kObject)
      : base(b), index(i), data_type(dt) {}
  OperandArg(Reg b, Reg i, int32_t off, DataType dt = OperandBase::kObject)
      : base(b), index(i), offset(off), data_type(dt) {}
  OperandArg(
      Reg b,
      Reg i,
      unsigned int num_bytes,
      int32_t off,
      DataType dt = OperandBase::kObject)
      : base(b), index(i), offset(off), data_type(dt) {
    // x86 encodes scales as size==2**X, so this does log2(num_bytes), but we
    // have a limited set of inputs.
    switch (num_bytes) {
      case 1:
        multiplier = 0;
        break;
      case 2:
        multiplier = 1;
        break;
      case 4:
        multiplier = 2;
        break;
      case 8:
        multiplier = 3;
        break;
      default:
        JIT_ABORT("Unexpected num_bytes {}", num_bytes);
    }
  }

  Reg base{PhyLocation::REG_INVALID};
  Reg index{PhyLocation::REG_INVALID};
  uint8_t multiplier{0};
  int32_t offset{0};
  DataType data_type{OperandBase::kObject};
  static constexpr bool is_output = Output;
};

template <>
struct OperandArg<void, true> {
  OperandArg(const DataType& dt = OperandBase::kObject) : data_type(dt) {}

  DataType data_type{OperandBase::kObject};
  static constexpr bool is_output = true;
};

// Creates a new struct type so that types like Stk and PhyReg are different
// from each other. This is needed because we need std::is_same_v<Stk, PhyReg> =
// false. If we used `using` then they would be aliases of each other.
#define DECLARE_TYPE_ARG(__T, __V, __O)      \
  struct __T : public OperandArg<__V, __O> { \
    using OperandArg::OperandArg;            \
  };

DECLARE_TYPE_ARG(PhyReg, PhyLocation, false)
DECLARE_TYPE_ARG(Imm, uint64_t, false)
DECLARE_TYPE_ARG(FPImm, double, false)
DECLARE_TYPE_ARG(MemImm, void*, false)
DECLARE_TYPE_ARG(Stk, PhyLocation, false)
DECLARE_TYPE_ARG(Lbl, BasicBlock*, false)
DECLARE_TYPE_ARG(VReg, Instruction*, false)
DECLARE_TYPE_ARG(Ind, MemoryIndirect, false)

DECLARE_TYPE_ARG(OutPhyReg, PhyLocation, true)
DECLARE_TYPE_ARG(OutImm, uint64_t, true)
DECLARE_TYPE_ARG(OutFPImm, double, true)
DECLARE_TYPE_ARG(OutMemImm, void*, true)
DECLARE_TYPE_ARG(OutStk, PhyLocation, true)
DECLARE_TYPE_ARG(OutLbl, BasicBlock*, true)
DECLARE_TYPE_ARG(OutDbl, double, true)
DECLARE_TYPE_ARG(OutInd, MemoryIndirect, true)
DECLARE_TYPE_ARG(OutVReg, void, true)

// Phase B5: Verify that Operand/LinkedOperand add no data members beyond
// OperandBase. Prerequisite for safe deletion through OperandBase*.
static_assert(sizeof(OperandBase) == sizeof(Operand),
    "Operand must not add data members beyond OperandBase");
static_assert(sizeof(OperandBase) == sizeof(LinkedOperand),
    "LinkedOperand must not add data members beyond OperandBase");

// ---- Deferred inline definitions (need complete Operand type) ----

inline size_t OperandBase::sizeInBits() const { return bitSize(dataType()); }
inline bool OperandBase::isFp() const { return dataType() == kDouble; }
inline bool OperandBase::isVecD() const { return getPhyRegister().is_fp_register(); }

inline uint64_t OperandBase::getConstant() const {
  if (is_linked_) return def_opnd_->getConstant();
  return value_.constant;
}
inline double OperandBase::getFPConstant() const {
  return bit_cast<double>(getConstant());
}
inline PhyLocation OperandBase::getPhyRegister() const {
  if (is_linked_) return def_opnd_->getPhyRegister();
  JIT_CHECK(type_ == kReg,
      "Trying to treat operand [type={},val={:#x}] as a physical register",
      type_, rawValue());
  return value_.phy_loc;
}
inline PhyLocation OperandBase::getStackSlot() const {
  if (is_linked_) return def_opnd_->getStackSlot();
  JIT_CHECK(type_ == kStack,
      "Trying to treat operand [type={},val={:#x}] as a stack slot",
      type_, rawValue());
  return value_.phy_loc;
}
inline PhyLocation OperandBase::getPhyRegOrStackSlot() const {
  if (is_linked_) return def_opnd_->getPhyRegOrStackSlot();
  switch (type_) {
    case kReg: return getPhyRegister();
    case kStack: return getStackSlot();
    default:
      JIT_ABORT(
          "Trying to treat operand [type={},val={:#x} as a physical register "
          "or a stack slot", type_, rawValue());
  }
  return -1;
}
inline void* OperandBase::getMemoryAddress() const {
  if (is_linked_) return def_opnd_->getMemoryAddress();
  JIT_CHECK(type_ == kMem,
      "Trying to treat operand [type={},val={:#x}] as a memory address",
      type_, rawValue());
  return value_.address;
}
inline MemoryIndirect* OperandBase::getMemoryIndirect() const {
  if (is_linked_) return def_opnd_->getMemoryIndirect();
  JIT_CHECK(type_ == kInd,
      "Trying to treat operand [type={},val={:#x}] as a memory indirect",
      type_, rawValue());
  return value_.indirect;
}
inline BasicBlock* OperandBase::getBasicBlock() const {
  if (is_linked_) return def_opnd_->getBasicBlock();
  JIT_CHECK(type_ == kLabel,
      "Trying to treat operand [type={},val={:#x}] as a basic block address",
      type_, rawValue());
  return value_.block;
}
inline uint64_t OperandBase::getConstantOrAddress() const {
  if (type_ == kMem) return reinterpret_cast<uint64_t>(getMemoryAddress());
  return getConstant();
}
inline Operand* OperandBase::getDefine() {
  if (is_linked_) return def_opnd_;
  return static_cast<Operand*>(this);
}
inline const Operand* OperandBase::getDefine() const {
  if (is_linked_) return def_opnd_;
  return static_cast<const Operand*>(this);
}
inline DataType OperandBase::dataType() const {
  if (is_linked_) return def_opnd_->dataType();
  return data_type_;
}
inline OperandBase::Type OperandBase::type() const {
  if (is_linked_) return def_opnd_->type();
  return type_;
}

} // namespace jit::lir
