// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "Python.h"
#include "cinderx/Common/log.h"
#include "cinderx/Jit/lir/arch.h"
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
  void* operator new(size_t size) { return PyMem_RawMalloc(size); }
  void operator delete(void* ptr) { PyMem_RawFree(ptr); }

  OperandBase() = default;
  explicit OperandBase(Instruction* parent);

  // Phase B5: Non-virtual destructor. Operand/LinkedOperand add no data
  // members, so destruction through OperandBase* is safe.
  ~OperandBase();

  OperandBase(const OperandBase& ob);

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
  Instruction* instr();
  const Instruction* instr() const;

  // Set and unset the instruction using this operand.
  void assignToInstr(Instruction* instr);
  void releaseFromInstr();

  bool isFp() const;
  bool isVecD() const;

  bool isLastUse() const;
  void setLastUse();

  // Data accessors — dispatch via is_linked_ flag.
  // When linked, delegates to def_opnd_ (the defining Operand).
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
  void clearIndirect() {
    if (type_ == kInd && value_.indirect) {
      delete value_.indirect;
      value_.indirect = nullptr;
    }
  }

  // Raw value accessor for diagnostics (works on non-linked operands only).
  uint64_t rawValue() const;

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
  explicit MemoryIndirect(Instruction* parent);
  ~MemoryIndirect();

  void setMemoryIndirect(Instruction* base, int32_t offset);
  void setMemoryIndirect(PhyLocation base, int32_t offset = 0);

  void setMemoryIndirect(
      PhyLocation base,
      PhyLocation index_reg,
      uint8_t multiplier);

  void setMemoryIndirect(
      std::variant<Instruction*, PhyLocation> base,
      std::variant<Instruction*, PhyLocation> index,
      uint8_t multiplier,
      int32_t offset);

  OperandBase* getBaseRegOperand() const;
  OperandBase* getIndexRegOperand() const;

  uint8_t getMultipiler() const;
  int32_t getOffset() const;

 private:
  // Phase B5: Raw pointers replace unique_ptr<OperandBase>.
  void setBaseIndex(OperandBase*& base_index_opnd, Instruction* base_index);
  void setBaseIndex(OperandBase*& base_index_opnd, PhyLocation base_index);
  void setBaseIndex(
      OperandBase*& base_index_opnd,
      std::variant<Instruction*, PhyLocation> base_index);

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
  explicit Operand(Instruction* parent);

  ~Operand() = default;

  // Only copies simple fields (type and data type) from operand.
  // The value_ field is not copied.
  Operand(Instruction* parent, Operand* operand);

  Operand(Instruction* parent, DataType data_type, Type type, uint64_t data);
  Operand(Instruction* parent, Type type, double data);

  // Setters (modify base class fields directly):
  void setConstant(uint64_t n, DataType data_type = k64bit);
  void setFPConstant(double n);
  void setPhyRegister(PhyLocation reg);
  void setStackSlot(PhyLocation slot);
  void setPhyRegOrStackSlot(PhyLocation loc);
  void setMemoryAddress(void* addr);

  // Phase B5: Explicit overloads replace variadic template (no more
  // std::make_unique / std::move into variant).
  void setMemoryIndirect(Instruction* base, int32_t offset);
  void setMemoryIndirect(PhyLocation base, int32_t offset = 0);
  void setMemoryIndirect(
      PhyLocation base,
      PhyLocation index_reg,
      uint8_t multiplier);
  void setMemoryIndirect(
      std::variant<Instruction*, PhyLocation> base,
      std::variant<Instruction*, PhyLocation> index,
      uint8_t multiplier,
      int32_t offset);

  void setBasicBlock(BasicBlock* block);
  void setDataType(DataType data_type);
  void setNone();
  void setVirtualRegister();
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
  explicit LinkedOperand(Instruction* def);
  LinkedOperand(Instruction* parent, Instruction* def);

  ~LinkedOperand() = default;

  Operand* getLinkedOperand();
  const Operand* getLinkedOperand() const;

  Instruction* getLinkedInstr();
  const Instruction* getLinkedInstr() const;

  void setLinkedInstr(Instruction* def);
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

} // namespace jit::lir
