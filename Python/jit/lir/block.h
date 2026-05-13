// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/Jit/codegen/code_section.h"
#include "cinderx/Jit/lir/instruction.h"

#include <cstring>

namespace jit::hir {
class Instr;
} // namespace jit::hir

namespace jit::lir {

struct Function;

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

// Basic block for LIR — Phase B4b: struct with public fields.
struct BasicBlock {
  void* operator new(size_t size) { return PyMem_RawCalloc(1, size); }
  void operator delete(void* ptr) { PyMem_RawFree(ptr); }

  // Phase B3c-2: instr_iter_t is now Instruction* (linked list position).
  using instr_iter_t = Instruction*;

  // Phase 3D: Constructor uses C API to break circular dep with function.h.
  explicit BasicBlock(Function* func)
      : id_(lir_function_allocate_id(
            reinterpret_cast<LirFunction*>(func))),
        func_(func) {}
  // Phase 5.B c4: dtor inlined; calls lir_block_destroy (cleanup children
  // only — operator delete handles PyMem_RawFree per destroy-vs-free
  // pattern, see D-1776379922).
  ~BasicBlock() {
    lir_block_destroy(reinterpret_cast<LirBasicBlock*>(this));
  }

  int id() const { return id_; }
  void setId(int id) { id_ = id; }
  Function* function() { return func_; }
  const Function* function() const { return func_; }

  // Phase 5.B c4: inlined; setSuccessor C++ wrapper deleted (was unused;
  // C-side lir_block_set_successor remains, called from function_impl.c).
  void addSuccessor(BasicBlock* bb) {
    lir_block_add_successor(
        reinterpret_cast<LirBasicBlock*>(this),
        reinterpret_cast<LirBasicBlock*>(bb));
  }

  BlockSpan successors() { return {successors_, num_succs_}; }
  ConstBlockSpan successors() const { return {successors_, num_succs_}; }

  void swapSuccessors() {
    if (num_succs_ >= 2) {
      BasicBlock* tmp = successors_[0];
      successors_[0] = successors_[1];
      successors_[1] = tmp;
    }
  }

  BasicBlock* getTrueSuccessor() const { return num_succs_ > 1 ? successors_[1] : nullptr; }
  BasicBlock* getFalseSuccessor() const { return num_succs_ > 0 ? successors_[0] : nullptr; }

  BlockSpan predecessors() { return {predecessors_, num_preds_}; }
  ConstBlockSpan predecessors() const { return {predecessors_, num_preds_}; }

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

  // Phase 5.B c22b-api: Allocate an instruction NOT linked into any
  // basic block's instruction list. Caller owns the returned pointer
  // and MUST `delete` it when done.
  //
  // Lifetime contract (per c22b-api 5-criterion pre-spec ratified by
  // gatekeeper 14:04:49Z):
  //   API-1 NO LEAK: caller-owned raw ptr, must `delete` after use
  //   API-2 NO DOUBLE-FREE: DO NOT pass the returned ptr to
  //         appendInstr() / insertInstrBefore() — those assume
  //         ownership via list linkage
  //   API-3 NO BB-LIST POLLUTION: instr is NOT in parent->instr_head_
  //         chain; getFirstInstr / instructions() iteration will not
  //         yield it
  //   API-5 INVARIANT-PRESERVING: lifetime contract = "alloc + populate
  //         + use + delete in single function scope, never iterated by
  //         Block". No setBlock / removeFromBlock / list-cascade calls
  //   API-6 NO OPERAND-LEAK: `delete instr` cascades operand cleanup
  //         via Instruction dtor → lir_instruction_destroy
  //
  // Intended use: c22b-mech shadow-emit (criterion 2 SYMMETRY) — build
  // a shadow Instruction, compare its operand list against the wrapper-
  // emitted instr, then delete the shadow. Single-scope discipline
  // enforced by audit.
  template <typename... T>
  Instruction* allocateInstrUnlinked(
      Instruction::Opcode opcode,
      const hir::Instr* origin,
      T&&... args) {
    auto* instr = new Instruction(this, opcode, origin);
    instr->addOperands(std::forward<T>(args)...);
    return instr;
  }

  // Append an instruction to the end of this block. Takes ownership.
  // Phase 5.B c4: inlined.
  void appendInstr(Instruction* instr) {
    lir_block_append_instr(
        reinterpret_cast<LirBasicBlock*>(this),
        reinterpret_cast<LirInstruction*>(instr));
  }

  // Insert an instruction before pos in the linked list. Takes ownership.
  // Phase 5.B c4: inlined; pos==nullptr falls back to appendInstr per
  // prior block.cpp::insertInstrBefore semantics.
  void insertInstrBefore(Instruction* pos, Instruction* instr) {
    if (pos == nullptr) {
      appendInstr(instr);
      return;
    }
    lir_block_insert_instr_before(
        reinterpret_cast<LirBasicBlock*>(this),
        reinterpret_cast<LirInstruction*>(pos),
        reinterpret_cast<LirInstruction*>(instr));
  }

  // Remove an instruction from this block. Caller takes ownership of the
  // returned pointer (must delete it or transfer elsewhere).
  // Phase 5.B c4: inlined.
  Instruction* removeInstr(instr_iter_t pos) {
    return reinterpret_cast<Instruction*>(
        lir_block_remove_instr(
            reinterpret_cast<LirBasicBlock*>(this),
            reinterpret_cast<LirInstruction*>(pos)));
  }

  InstrRange instructions() { return {instr_head_, instr_tail_, num_instrs_}; }
  InstrRange instructions() const { return {instr_head_, instr_tail_, num_instrs_}; }

  bool isEmpty() const { return num_instrs_ == 0; }
  size_t getNumInstrs() const { return num_instrs_; }

  Instruction* getFirstInstr() { return instr_head_; }
  const Instruction* getFirstInstr() const { return instr_head_; }

  Instruction* getLastInstr() { return instr_tail_; }
  const Instruction* getLastInstr() const { return instr_tail_; }

  instr_iter_t getLastInstrIter() { return instr_tail_; }

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
  // Phase 5.B c4: inlined.
  BasicBlock* insertBasicBlockBetween(BasicBlock* block) {
    return reinterpret_cast<BasicBlock*>(
        lir_block_insert_between(
            reinterpret_cast<LirBasicBlock*>(this),
            reinterpret_cast<LirBasicBlock*>(block)));
  }

  // Phase 5.B c4: splitBefore + fixupPhis C++ wrappers deleted (0 callers
  // tree-wide; C-side lir_block_split_before + lir_block_fixup_phis remain
  // alive in lir_impl_internal.h for cross-impl-file use).

  codegen::CodeSection section() const { return section_; }
  void setSection(codegen::CodeSection section) { section_ = section; }

  // Phase B3c-2: iterator_to is identity — Instruction* IS the iterator.
  instr_iter_t iterator_to(Instruction* instr) { return instr; }

  // Phase B4b: all fields public.
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
