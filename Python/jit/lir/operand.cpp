// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cinderx/Jit/lir/operand.h"

#include "cinderx/Jit/lir/arch.h"
#include "cinderx/Jit/lir/instruction.h"

namespace jit::lir {

// Phase B5: Verify that Operand/LinkedOperand add no data members beyond
// OperandBase. This is a prerequisite for safe deletion through OperandBase*
// after vtable removal.
static_assert(sizeof(OperandBase) == sizeof(Operand),
    "Operand must not add data members beyond OperandBase");
static_assert(sizeof(OperandBase) == sizeof(LinkedOperand),
    "LinkedOperand must not add data members beyond OperandBase");

OperandBase::OperandBase(Instruction* parent) : parent_instr_{parent} {}

// Phase B5: Non-virtual destructor. Frees owned MemoryIndirect when type_ ==
// kInd. Safe for deletion through OperandBase* because Operand/LinkedOperand
// add zero data members.
OperandBase::~OperandBase() {
  if (!is_linked_ && type_ == kInd && value_.indirect) {
    delete value_.indirect;
    value_.indirect = nullptr;
  }
}

OperandBase::OperandBase(const OperandBase& ob)
    : parent_instr_{ob.parent_instr_},
      last_use_{ob.last_use_},
      is_linked_{ob.is_linked_},
      type_{ob.type_},
      data_type_{ob.data_type_},
      def_opnd_{ob.def_opnd_} {
  // value_ is not copied (same as original Operand copy ctor behavior).
}

size_t OperandBase::sizeInBits() const {
  return bitSize(dataType());
}

Instruction* OperandBase::instr() {
  return parent_instr_;
}

const Instruction* OperandBase::instr() const {
  return parent_instr_;
}

void OperandBase::assignToInstr(Instruction* instr) {
  parent_instr_ = instr;
}

void OperandBase::releaseFromInstr() {
  parent_instr_ = nullptr;
}

bool OperandBase::isFp() const {
  return dataType() == kDouble;
}

bool OperandBase::isVecD() const {
  return getPhyRegister().is_fp_register();
}

bool OperandBase::isLastUse() const {
  return last_use_;
}

void OperandBase::setLastUse() {
  last_use_ = true;
}

// --- Devirtualized data accessors (dispatch via is_linked_) ---

uint64_t OperandBase::getConstant() const {
  if (is_linked_) return def_opnd_->getConstant();
  return value_.constant;
}

double OperandBase::getFPConstant() const {
  if (is_linked_) return def_opnd_->getFPConstant();
  return bit_cast<double>(value_.constant);
}

PhyLocation OperandBase::getPhyRegister() const {
  if (is_linked_) return def_opnd_->getPhyRegister();
  JIT_CHECK(
      type_ == kReg,
      "Trying to treat operand [type={},val={:#x}] as a physical register",
      type_,
      rawValue());
  return value_.phy_loc;
}

PhyLocation OperandBase::getStackSlot() const {
  if (is_linked_) return def_opnd_->getStackSlot();
  JIT_CHECK(
      type_ == kStack,
      "Trying to treat operand [type={},val={:#x}] as a stack slot",
      type_,
      rawValue());
  return value_.phy_loc;
}

PhyLocation OperandBase::getPhyRegOrStackSlot() const {
  if (is_linked_) return def_opnd_->getPhyRegOrStackSlot();
  switch (type_) {
    case kReg:
      return getPhyRegister();
    case kStack:
      return getStackSlot();
    default:
      JIT_ABORT(
          "Trying to treat operand [type={},val={:#x} as a physical register "
          "or a stack slot",
          type_,
          rawValue());
  }
  return -1;
}

void* OperandBase::getMemoryAddress() const {
  if (is_linked_) return def_opnd_->getMemoryAddress();
  JIT_CHECK(
      type_ == kMem,
      "Trying to treat operand [type={},val={:#x}] as a memory address",
      type_,
      rawValue());
  return value_.address;
}

MemoryIndirect* OperandBase::getMemoryIndirect() const {
  if (is_linked_) return def_opnd_->getMemoryIndirect();
  JIT_CHECK(
      type_ == kInd,
      "Trying to treat operand [type={},val={:#x}] as a memory indirect",
      type_,
      rawValue());
  return value_.indirect;
}

BasicBlock* OperandBase::getBasicBlock() const {
  if (is_linked_) return def_opnd_->getBasicBlock();
  JIT_CHECK(
      type_ == kLabel,
      "Trying to treat operand [type={},val={:#x}] as a basic block address",
      type_,
      rawValue());
  return value_.block;
}

uint64_t OperandBase::getConstantOrAddress() const {
  if (is_linked_) return def_opnd_->getConstantOrAddress();
  if (type_ == kImm) {
    return value_.constant;
  }
  return reinterpret_cast<uint64_t>(getMemoryAddress());
}

Operand* OperandBase::getDefine() {
  if (is_linked_) return def_opnd_;
  return static_cast<Operand*>(this);
}

const Operand* OperandBase::getDefine() const {
  if (is_linked_) return def_opnd_;
  return static_cast<const Operand*>(this);
}

DataType OperandBase::dataType() const {
  if (is_linked_) return def_opnd_->dataType();
  return data_type_;
}

OperandBase::Type OperandBase::type() const {
  if (is_linked_) {
    if (def_opnd_ == nullptr) return kNone;
    return def_opnd_->type();
  }
  return type_;
}

uint64_t OperandBase::rawValue() const {
  switch (type_) {
    case kImm:
      return value_.constant;
    case kMem:
      return reinterpret_cast<uint64_t>(value_.address);
    case kLabel:
      return reinterpret_cast<uint64_t>(value_.block);
    case kInd:
      return reinterpret_cast<uint64_t>(value_.indirect);
    case kReg:
    case kStack:
      return static_cast<uint64_t>(value_.phy_loc.loc);
    case kVreg:
    case kNone:
      return 0;
  }
  JIT_ABORT("Unknown operand type");
}

MemoryIndirect::MemoryIndirect(Instruction* parent) : parent_(parent) {}

MemoryIndirect::~MemoryIndirect() {
  delete base_reg_;
  delete index_reg_;
}

void MemoryIndirect::setMemoryIndirect(Instruction* base, int32_t offset) {
  setMemoryIndirect(base, nullptr /* index */, 0, offset);
}

void MemoryIndirect::setMemoryIndirect(PhyLocation base, int32_t offset) {
  setMemoryIndirect(base, PhyLocation::REG_INVALID, 0, offset);
}

void MemoryIndirect::setMemoryIndirect(
    PhyLocation base,
    PhyLocation index_reg,
    uint8_t multiplier) {
  setMemoryIndirect(base, index_reg, multiplier, 0);
}

void MemoryIndirect::setMemoryIndirect(
    std::variant<Instruction*, PhyLocation> base,
    std::variant<Instruction*, PhyLocation> index,
    uint8_t multiplier,
    int32_t offset) {
  setBaseIndex(base_reg_, base);
  setBaseIndex(index_reg_, index);
  multiplier_ = multiplier;
  offset_ = offset;
}

OperandBase* MemoryIndirect::getBaseRegOperand() const {
  return base_reg_;
}

OperandBase* MemoryIndirect::getIndexRegOperand() const {
  return index_reg_;
}

uint8_t MemoryIndirect::getMultipiler() const {
  return multiplier_;
}

int32_t MemoryIndirect::getOffset() const {
  return offset_;
}

void MemoryIndirect::setBaseIndex(
    OperandBase*& base_index_opnd,
    Instruction* base_index) {
  delete base_index_opnd;
  if (base_index != nullptr) {
    base_index_opnd = new LinkedOperand(parent_, base_index);
  } else {
    base_index_opnd = nullptr;
  }
}

void MemoryIndirect::setBaseIndex(
    OperandBase*& base_index_opnd,
    PhyLocation base_index) {
  delete base_index_opnd;
  if (base_index != PhyLocation::REG_INVALID) {
    auto* operand = new Operand(parent_);
    operand->setPhyRegister(base_index);
    base_index_opnd = operand;
  } else {
    base_index_opnd = nullptr;
  }
}

void MemoryIndirect::setBaseIndex(
    OperandBase*& base_index_opnd,
    std::variant<Instruction*, PhyLocation> base_index) {
  if (Instruction** instrp = std::get_if<Instruction*>(&base_index)) {
    setBaseIndex(base_index_opnd, *instrp);
  } else {
    setBaseIndex(base_index_opnd, std::get<PhyLocation>(base_index));
  }
}

Operand::Operand(Instruction* parent) : OperandBase{parent} {}

// Only copies simple fields (type and data type) from operand.
// The value_ field is not copied.
Operand::Operand(Instruction* parent, Operand* operand)
    : OperandBase(parent) {
  type_ = operand->type_;
  data_type_ = operand->data_type_;
}

Operand::Operand(
    Instruction* parent,
    DataType data_type,
    Operand::Type type,
    uint64_t data)
    : OperandBase(parent) {
  type_ = type;
  data_type_ = data_type;
  value_.constant = data;
}

Operand::Operand(Instruction* parent, Operand::Type type, double data)
    : OperandBase(parent) {
  type_ = type;
  data_type_ = kDouble;
  value_.constant = bit_cast<uint64_t>(data);
}

void Operand::setConstant(uint64_t n, DataType data_type) {
  clearIndirect();
  type_ = kImm;
  value_.constant = n;
  data_type_ = data_type;
}

void Operand::setFPConstant(double n) {
  clearIndirect();
  type_ = kImm;
  data_type_ = kDouble;
  value_.constant = bit_cast<uint64_t>(n);
}

void Operand::setPhyRegister(PhyLocation reg) {
  clearIndirect();
  type_ = kReg;
  value_.phy_loc = reg;
}

void Operand::setStackSlot(PhyLocation slot) {
  clearIndirect();
  type_ = kStack;
  value_.phy_loc = slot;
}

void Operand::setPhyRegOrStackSlot(PhyLocation loc) {
  if (loc.loc < 0) {
    setStackSlot(loc);
  } else {
    setPhyRegister(loc);
  }
}

void Operand::setMemoryAddress(void* addr) {
  clearIndirect();
  type_ = kMem;
  value_.address = addr;
}

void Operand::setBasicBlock(BasicBlock* block) {
  clearIndirect();
  type_ = kLabel;
  data_type_ = kObject;
  value_.block = block;
}

void Operand::setDataType(DataType data_type) {
  data_type_ = data_type;
  if (type_ == kReg || type_ == kStack) {
    value_.phy_loc.bitSize = bitSize(data_type);
  }
}

void Operand::setNone() {
  clearIndirect();
  type_ = kNone;
}

void Operand::setVirtualRegister() {
  clearIndirect();
  type_ = kVreg;
}

// Phase B5: Explicit setMemoryIndirect overloads replace template.
void Operand::setMemoryIndirect(Instruction* base, int32_t offset) {
  clearIndirect();
  type_ = kInd;
  auto* ind = new MemoryIndirect(instr());
  ind->setMemoryIndirect(base, offset);
  value_.indirect = ind;
}

void Operand::setMemoryIndirect(PhyLocation base, int32_t offset) {
  clearIndirect();
  type_ = kInd;
  auto* ind = new MemoryIndirect(instr());
  ind->setMemoryIndirect(base, offset);
  value_.indirect = ind;
}

void Operand::setMemoryIndirect(
    PhyLocation base,
    PhyLocation index_reg,
    uint8_t multiplier) {
  clearIndirect();
  type_ = kInd;
  auto* ind = new MemoryIndirect(instr());
  ind->setMemoryIndirect(base, index_reg, multiplier);
  value_.indirect = ind;
}

void Operand::setMemoryIndirect(
    std::variant<Instruction*, PhyLocation> base,
    std::variant<Instruction*, PhyLocation> index,
    uint8_t multiplier,
    int32_t offset) {
  clearIndirect();
  type_ = kInd;
  auto* ind = new MemoryIndirect(instr());
  ind->setMemoryIndirect(base, index, multiplier, offset);
  value_.indirect = ind;
}

LinkedOperand::LinkedOperand(Instruction* def_instr) {
  is_linked_ = true;
  def_opnd_ = def_instr->output();
}

LinkedOperand::LinkedOperand(Instruction* parent, Instruction* def_instr)
    : LinkedOperand{def_instr} {
  assignToInstr(parent);
}

Operand* LinkedOperand::getLinkedOperand() {
  return def_opnd_;
}

const Operand* LinkedOperand::getLinkedOperand() const {
  return def_opnd_;
}

Instruction* LinkedOperand::getLinkedInstr() {
  return def_opnd_->instr();
}

const Instruction* LinkedOperand::getLinkedInstr() const {
  return def_opnd_->instr();
}

void LinkedOperand::setLinkedInstr(Instruction* def) {
  def_opnd_ = def->output();
}

} // namespace jit::lir
