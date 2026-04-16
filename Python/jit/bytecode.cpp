// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cinderx/Jit/bytecode.h"
#include "cinderx/Jit/bytecode_c.h"

namespace jit {

int BytecodeInstruction::opcode() const {
  JitBytecodeInstr c;
  jit_bc_instr_init(&c, code_, baseOffset_.value());
  return jit_bc_instr_opcode(&c);
}

void BytecodeInstruction::calcOpcodeOffsetAndOparg() const {
  if (opcodeIndex_.value() != std::numeric_limits<int>::min()) {
    return;
  }
  // Delegate to C implementation, copy results back to mutable fields.
  JitBytecodeInstr c;
  jit_bc_instr_init(&c, code_, baseOffset_.value());
  jit_bc_instr_opcode_offset(&c);  // triggers lazy computation
  opcodeIndex_ = BCIndex{c.opcode_index};
  extendedOparg_ = c.extended_oparg;
  extendedOpcode_ = c.extended_opcode;
}

int BytecodeInstruction::specializedOpcode() const {
  JitBytecodeInstr c;
  jit_bc_instr_init(&c, code_, baseOffset_.value());
  return jit_bc_instr_specialized_opcode(&c);
}

bool BytecodeInstruction::isBranch() const {
  JitBytecodeInstr c;
  jit_bc_instr_init(&c, code_, baseOffset_.value());
  return jit_bc_instr_is_branch(&c) != 0;
}

bool BytecodeInstruction::isReturn() const {
  JitBytecodeInstr c;
  jit_bc_instr_init(&c, code_, baseOffset_.value());
  return jit_bc_instr_is_return(&c) != 0;
}

bool BytecodeInstruction::isTerminator() const {
  JitBytecodeInstr c;
  jit_bc_instr_init(&c, code_, baseOffset_.value());
  return jit_bc_instr_is_terminator(&c) != 0;
}

BCOffset BytecodeInstruction::getJumpTarget() const {
  JIT_DCHECK(
      isBranch(), "Calling getJumpTarget() on a non-branch gives nonsense");
  JitBytecodeInstr c;
  jit_bc_instr_init(&c, code_, baseOffset_.value());
  return BCOffset{jit_bc_instr_get_jump_target(&c)};
}

BCOffset BytecodeInstruction::nextInstrOffset() const {
  JitBytecodeInstr c;
  jit_bc_instr_init(&c, code_, baseOffset_.value());
  return BCOffset{jit_bc_instr_next_offset(&c)};
}

// word() and isAbsoluteControlFlow() removed — logic now in bytecode_c.c.

BytecodeInstructionBlock::BytecodeInstructionBlock(
    BorrowedRef<PyCodeObject> code)
    : BytecodeInstructionBlock{code, BCIndex{0}, BCIndex{countIndices(code)}} {}

BytecodeInstructionBlock::BytecodeInstructionBlock(
    BorrowedRef<PyCodeObject> code,
    BCIndex start,
    BCIndex end)
    : code_{ThreadedRef<PyCodeObject>::create(code)},
      start_idx_{start},
      end_idx_{end} {}

BytecodeInstructionBlock::Iterator BytecodeInstructionBlock::begin() const {
  return Iterator{code_, start_idx_, end_idx_};
}

BytecodeInstructionBlock::Iterator BytecodeInstructionBlock::end() const {
  return Iterator{code_, end_idx_, end_idx_};
}

BCOffset BytecodeInstructionBlock::startOffset() const {
  return start_idx_;
}

BCOffset BytecodeInstructionBlock::endOffset() const {
  return end_idx_;
}

Py_ssize_t BytecodeInstructionBlock::size() const {
  return end_idx_ - start_idx_;
}

BytecodeInstruction BytecodeInstructionBlock::at(BCIndex idx) const {
  JIT_CHECK(
      idx >= start_idx_ && idx < end_idx_,
      "Invalid index {}, bytecode block is [{}, {})",
      idx,
      start_idx_,
      end_idx_);
  return BytecodeInstruction{code_, idx};
}

BorrowedRef<PyCodeObject> BytecodeInstructionBlock::code() const {
  return code_;
}

} // namespace jit
