// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cinderx/Jit/bytecode.h"
#include "cinderx/Jit/bytecode_c.h"

namespace jit {

// All C functions in bytecode_c.c work with instruction indices (array indices
// into _Py_CODEUNIT[]).  The C++ BCOffset type stores byte offsets (index * 2).
// These helpers convert at the C/C++ boundary.
//
// baseOffset_.value() is a byte offset.  To pass to C: divide by sizeof.
// C returns instruction indices.  To return as BCOffset: wrap in BCIndex
// (which implicitly converts to BCOffset by multiplying by sizeof).

static int toInstrIndex(BCOffset off) {
  return BCIndex{off}.value();
}

static BCOffset fromInstrIndex(int idx) {
  return BCIndex{idx};  // implicit BCIndex → BCOffset: idx * sizeof(_Py_CODEUNIT)
}

int BytecodeInstruction::opcode() const {
  JitBytecodeInstr c;
  jit_bc_instr_init(&c, code_, toInstrIndex(baseOffset_));
  return jit_bc_instr_opcode(&c);
}

void BytecodeInstruction::calcOpcodeOffsetAndOparg() const {
  if (opcodeIndex_.value() != std::numeric_limits<int>::min()) {
    return;
  }
  // Delegate to C implementation, copy results back to mutable fields.
  JitBytecodeInstr c;
  jit_bc_instr_init(&c, code_, toInstrIndex(baseOffset_));
  jit_bc_instr_opcode_offset(&c);  // triggers lazy computation
  opcodeIndex_ = BCIndex{c.opcode_index};
  extendedOparg_ = c.extended_oparg;
  extendedOpcode_ = c.extended_opcode;
}

int BytecodeInstruction::specializedOpcode() const {
  JitBytecodeInstr c;
  jit_bc_instr_init(&c, code_, toInstrIndex(baseOffset_));
  return jit_bc_instr_specialized_opcode(&c);
}

bool BytecodeInstruction::isBranch() const {
  JitBytecodeInstr c;
  jit_bc_instr_init(&c, code_, toInstrIndex(baseOffset_));
  return jit_bc_instr_is_branch(&c) != 0;
}

bool BytecodeInstruction::isReturn() const {
  JitBytecodeInstr c;
  jit_bc_instr_init(&c, code_, toInstrIndex(baseOffset_));
  return jit_bc_instr_is_return(&c) != 0;
}

bool BytecodeInstruction::isTerminator() const {
  JitBytecodeInstr c;
  jit_bc_instr_init(&c, code_, toInstrIndex(baseOffset_));
  return jit_bc_instr_is_terminator(&c) != 0;
}

BCOffset BytecodeInstruction::getJumpTarget() const {
  JIT_DCHECK(
      isBranch(), "Calling getJumpTarget() on a non-branch gives nonsense");
  JitBytecodeInstr c;
  jit_bc_instr_init(&c, code_, toInstrIndex(baseOffset_));
  // C function handles all cases including FOR_ITER END_FOR skip.
  return fromInstrIndex(jit_bc_instr_get_jump_target(&c));
}

BCOffset BytecodeInstruction::nextInstrOffset() const {
  JitBytecodeInstr c;
  jit_bc_instr_init(&c, code_, toInstrIndex(baseOffset_));
  return fromInstrIndex(jit_bc_instr_next_offset(&c));
}

_Py_CODEUNIT BytecodeInstruction::word() const {
  JitBytecodeInstr c;
  jit_bc_instr_init(&c, code_, toInstrIndex(baseOffset_));
  return jit_bc_instr_word(&c);
}

bool BytecodeInstruction::isAbsoluteControlFlow() const {
  JitBytecodeInstr c;
  jit_bc_instr_init(&c, code_, toInstrIndex(baseOffset_));
  return jit_bc_instr_is_absolute_control_flow(&c) != 0;
}

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
