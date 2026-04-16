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
#if PY_VERSION_HEX >= 0x030C0000
  int opcode = uninstrumentedOpcode();

  switch (opcode) {
    case BINARY_OP_ADD_FLOAT:
    case BINARY_OP_ADD_INT:
    case BINARY_OP_ADD_UNICODE:
    case BINARY_OP_MULTIPLY_FLOAT:
    case BINARY_OP_MULTIPLY_INT:
    case BINARY_OP_SUBTRACT_FLOAT:
    case BINARY_OP_SUBTRACT_INT:
    case BINARY_SUBSCR_DICT:
    case BINARY_SUBSCR_LIST_INT:
    case BINARY_SUBSCR_TUPLE_INT:
    case COMPARE_OP_FLOAT:
    case COMPARE_OP_INT:
    case COMPARE_OP_STR:
    case LOAD_ATTR_MODULE:
    case LOAD_ATTR_INSTANCE_VALUE:
    case STORE_ATTR_INSTANCE_VALUE:
    case STORE_ATTR_SLOT:
    case LOAD_ATTR_SLOT:
    case STORE_SUBSCR_DICT:
    case STORE_SUBSCR_LIST_INT:
    case UNPACK_SEQUENCE_LIST:
    case UNPACK_SEQUENCE_TUPLE:
    case UNPACK_SEQUENCE_TWO_TUPLE:
    case FOR_ITER_RANGE:
    case FOR_ITER_LIST:
    case FOR_ITER_TUPLE:
      return opcode;
    default:
      return unspecialize(opcode);
  }
#else
  return opcode();
#endif
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

  if (isAbsoluteControlFlow()) {
    return BCIndex{oparg()};
  }

  int delta = oparg();
  if (opcode() == JUMP_BACKWARD || opcode() == JUMP_BACKWARD_NO_INTERRUPT) {
    delta = -delta;
  }
  BCIndex target = BCIndex{nextInstrOffset()} + delta;
  // In 3.11+ the FOR_ITER bytecode encodes a jump to an END_FOR instruction
  // then at runtime it usually dynamically jumps past this. The only time it
  // actually goes through the END_FOR is if the FOR_ITER is operating
  // on a generator and gets adaptively specialized. We always compile
  // unspecialized bytecode so we can always skip the END_FOR.
  //
  // We make this tweak here so it applies both when generating the branching
  // HIR operation, and when creating block boundaries for bytecode. The END_FOR
  // will end up on its own in an unreachable block.
  if (PY_VERSION_HEX >= 0x030B0000 && opcode() == FOR_ITER) {
    BytecodeInstruction target_bc{code_, target};
    JIT_CHECK(target_bc.opcode() == END_FOR, "Expected END_FOR");
    return target_bc.nextInstrOffset();
  }
  return target;
}

BCOffset BytecodeInstruction::nextInstrOffset() const {
  JitBytecodeInstr c;
  jit_bc_instr_init(&c, code_, baseOffset_.value());
  return BCOffset{jit_bc_instr_next_offset(&c)};
}

_Py_CODEUNIT BytecodeInstruction::word() const {
#if PY_VERSION_HEX >= 0x030C0000
  int opcode = unspecialize(uninstrumentedOpcode());
  int oparg = _Py_OPARG(codeUnit(code_)[opcodeIndex().value()]);
  return _Py_MAKE_CODEUNIT(opcode, oparg);
#else
  return codeUnit(code_)[opcodeIndex().value()];
#endif
}

bool BytecodeInstruction::isAbsoluteControlFlow() const {
  switch (opcode()) {
    case JUMP_ABSOLUTE:
    case JUMP_IF_FALSE_OR_POP:
    case JUMP_IF_NONZERO_OR_POP:
    case JUMP_IF_NOT_EXC_MATCH:
    case JUMP_IF_TRUE_OR_POP:
    case JUMP_IF_ZERO_OR_POP:
      return true;
    case POP_JUMP_IF_NONZERO:
    case POP_JUMP_IF_ZERO:
    case POP_JUMP_IF_FALSE:
    case POP_JUMP_IF_TRUE:
      // These instructions switched from absolute to relative in 3.11.
      return PY_VERSION_HEX < 0x030B0000;
    default:
      return false;
  }
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
