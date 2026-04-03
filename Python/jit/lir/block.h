// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/Jit/codegen/code_section.h"
#include "cinderx/Jit/lir/instruction.h"

#include <cstring>

namespace jit::hir {
class Instr;
} // namespace jit::hir

namespace jit::lir {

class Function;

// Lightweight non-owning view over a BasicBlock* array (range-for compatible).
struct BlockSpan {
  BasicBlock** data_;
  size_t size_;
  BasicBlock** begin() const { return data_; }
  BasicBlock** end() const { return data_ + size_; }
  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }
  BasicBlock*& operator[](size_t i) { return data_[i]; }
  BasicBlock* operator[](size_t i) const { return data_[i]; }
  BasicBlock*& at(size_t i) { return data_[i]; }
  BasicBlock* at(size_t i) const { return data_[i]; }
  BasicBlock*& front() { return data_[0]; }
  BasicBlock*& back() { return data_[size_ - 1]; }
};

struct ConstBlockSpan {
  BasicBlock* const* data_;
  size_t size_;
  BasicBlock* const* begin() const { return data_; }
  BasicBlock* const* end() const { return data_ + size_; }
  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }
  BasicBlock* operator[](size_t i) const { return data_[i]; }
  BasicBlock* at(size_t i) const { return data_[i]; }
  BasicBlock* front() const { return data_[0]; }
  BasicBlock* back() const { return data_[size_ - 1]; }
};

// Basic block class for LIR
class BasicBlock {
 public:
  // Phase B3c-2: instr_iter_t is now Instruction* (linked list position).
  using instr_iter_t = Instruction*;

  explicit BasicBlock(Function* func);
  ~BasicBlock();

  // Get the unique ID representing this block within its function.
  int id() const;

  // Change the block's ID.  This is only meant to be used by the LIR
  // parser.  LIR strongly expects unique instruction IDs.
  void setId(int id);

  // Get the function that has this block as part of its CFG.
  Function* function();
  const Function* function() const;

  void addSuccessor(BasicBlock* bb);

  // Set successor at index to bb.
  // Expects index to be within the current size of successors.
  void setSuccessor(size_t index, BasicBlock* bb);

  BlockSpan successors();
  ConstBlockSpan successors() const;

  void swapSuccessors();

  BasicBlock* getTrueSuccessor() const;
  BasicBlock* getFalseSuccessor() const;

  BlockSpan predecessors();
  ConstBlockSpan predecessors() const;

  // Allocate an instruction and its operands and append it to the
  // instruction list. For the details on how to allocate instruction
  // operands, please refer to Instruction::addOperands() function.
  template <typename... T>
  Instruction* allocateInstr(
      Instruction::Opcode opcode,
      const hir::Instr* origin,
      T&&... args) {
    auto* instr = new Instruction(this, opcode, origin);
    appendInstr(instr);
    instr->addOperands(std::forward<T>(args)...);
    return instr;
  }

  // Allocate an instruction and its operands and insert it before the
  // instruction specified by pos. For the details on how to allocate
  // instruction operands, please refer to Instruction::addOperands() function.
  template <typename... T>
  Instruction* allocateInstrBefore(
      instr_iter_t pos,
      Instruction::Opcode opcode,
      T&&... args) {
    const hir::Instr* origin = nullptr;
    if (pos != nullptr) {
      origin = pos->origin();
    } else if (instr_tail_ != nullptr) {
      origin = instr_tail_->origin();
    }

    auto* instr = new Instruction(this, opcode, origin);
    insertInstrBefore(pos, instr);
    instr->addOperands(std::forward<T>(args)...);
    return instr;
  }

  // Append an instruction to the end of this block. Takes ownership.
  void appendInstr(Instruction* instr);

  // Insert an instruction before pos in the linked list. Takes ownership.
  void insertInstrBefore(Instruction* pos, Instruction* instr);

  // Remove an instruction from this block. Caller takes ownership of the
  // returned pointer (must delete it or transfer elsewhere).
  Instruction* removeInstr(instr_iter_t pos);

  InstrRange instructions();
  InstrRange instructions() const;

  bool isEmpty() const;

  size_t getNumInstrs() const;

  Instruction* getFirstInstr();
  const Instruction* getFirstInstr() const;

  Instruction* getLastInstr();
  const Instruction* getLastInstr() const;

  instr_iter_t getLastInstrIter();

  template <typename Func>
  void foreachPhiInstr(const Func& f) const {
    for (Instruction* instr = instr_head_; instr; instr = instr->next_) {
      if (instr->opcode_ == Instruction::kPhi) {
        f(instr);
      }
    }
  }

  // insert a basic block on the edge between the current basic
  // block and another basic block specified by block.
  BasicBlock* insertBasicBlockBetween(BasicBlock* block);

  // Split this block before instr.
  // Current basic block contains all instructions up to (but excluding) instr.
  // Return a new block with all instructions (including and) after instr.
  BasicBlock* splitBefore(Instruction* instr);

  // Replace any references to old_pred in this block's Phis with new_pred.
  void fixupPhis(BasicBlock* old_pred, BasicBlock* new_pred);

  codegen::CodeSection section() const;
  void setSection(codegen::CodeSection section);

  // Return an iterator to the given instruction. Behavior is undefined if the
  // given Instruction is not in this block.
  //
  // This function is O(getNumInstrs()) due to implementation details in
  // InstrList.
  instr_iter_t iterator_to(Instruction* instr);

 private:
  int id_;
  Function* func_;

  BasicBlock** successors_{nullptr};
  size_t num_succs_{0};
  size_t succs_capacity_{0};

  BasicBlock** predecessors_{nullptr};
  size_t num_preds_{0};
  size_t preds_capacity_{0};

  // Phase B3c-2: intrusive doubly-linked list of instructions.
  Instruction* instr_head_{nullptr};
  Instruction* instr_tail_{nullptr};
  size_t num_instrs_{0};

  codegen::CodeSection section_{codegen::CodeSection::kHot};
};

using instr_iter_t = Instruction*;

} // namespace jit::lir
