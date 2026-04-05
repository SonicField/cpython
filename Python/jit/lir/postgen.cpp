// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cinderx/Jit/lir/postgen.h"

#include "cinderx/Jit/lir/inliner.h"
#include "cinderx/Jit/lir/printer.h"

using namespace jit::codegen;

namespace jit::lir {

namespace {

// Inline C helper functions.
RewriteResult rewriteInlineHelper(function_rewrite_arg_t func) {
  if (!getConfig().lir_opts.inliner) {
    return kUnchanged;
  }

  return LIRInliner::inlineCalls(func) ? kChanged : kUnchanged;
}

// Fix constant input position. If a binary operation has a constant input,
// always put it as the second operand (or move the 2nd to a register for div
// instructions)
RewriteResult rewriteBinaryOpConstantPosition(instr_iter_t instr_iter) {
  auto instr = instr_iter;
  auto block = instr->basic_block_;

  if (instr->isDiv() || instr->isDivUn()) {
    auto divisor = instr->inputs_[2];
    if (!divisor->isImm()) {
      return kUnchanged;
    }

    // div doesn't support an immediate as the divisor.
    auto constant = divisor->getConstant();
    auto constant_size = divisor->dataType();

    auto move = block->allocateInstrBefore(instr_iter, Instruction::kMove);
    move->output_.setVirtualRegister();
    move->output_.setDataType(constant_size);
    move->allocateImmediateInput(constant)->setDataType(constant_size);

    instr->setInput(2, new LinkedOperand(move));
    return kChanged;
  }

  if (!instr->isAdd() && !instr->isSub() && !instr->isXor() &&
      !instr->isAnd() && !instr->isOr() && !instr->isMul() &&
      !instr->isCompare()) {
    return kUnchanged;
  }

  bool is_commutative_or_compare = !instr->isSub();
  auto input0 = instr->inputs_[0];
  auto input1 = instr->inputs_[1];

  if (!input0->isImm()) {
    return kUnchanged;
  }

  // TODO: If both are registers we could constant fold here
  if (is_commutative_or_compare && !input1->isImm()) {
    // if the operation is commutative and the second input is not also an
    // immediate, just swap the operands
    if (instr->isCompare()) {
      instr->setOpcode(Instruction::flipComparisonDirection(instr->opcode_));
    }
    auto* imm = instr->removeInput(0);
    instr->appendInput(imm);
    return kChanged;
  }

  // Otherwise, replace the immediate with a new move instruction.
  auto constant = input0->getConstant();
  auto constant_size = input0->dataType();

  auto move = block->allocateInstrBefore(instr_iter, Instruction::kMove);
  move->output_.setVirtualRegister();
  move->output_.setDataType(constant_size);
  move->allocateImmediateInput(constant)->setDataType(constant_size);
  instr->setInput(0, new LinkedOperand(move));

  return kChanged;
}

// Rewrite binary instructions with > 32-bit constant.
RewriteResult rewriteBinaryOpLargeConstant(instr_iter_t instr_iter) {
  // rewrite
  //     Vreg2 = BinOp Vreg1, Imm64
  // to
  //     Vreg0 = Mov Imm64
  //     Vreg2 = BinOp Vreg1, VReg0

  Instruction* instr = instr_iter;
  if (!instr->isAdd() && !instr->isSub() && !instr->isXor() &&
      !instr->isAnd() && !instr->isOr() && !instr->isMul() &&
      !instr->isCompare()) {
    return kUnchanged;
  }

  // If first operand is an immediate, we need to swap the operands.
  if (instr->inputs_[0]->isImm()) {
    // another rewrite will fix this later
    return kUnchanged;
  }

  JIT_CHECK(
      !instr->inputs_[0]->isImm(),
      "The first input operand of a binary op instruction should not be "
      "constant");

  auto in1 = instr->inputs_[1];
  if (!in1->isImm()) {
    return kUnchanged;
  }

  auto constant = in1->getConstantOrAddress();
#if defined(CINDER_X86_64)
  // All of these instructions support a register operand and a 32-bit immediate
  // operand. None of them support a 64-bit immediate.
  if ((in1->sizeInBits() < 64) || fitsSignedInt<32>(constant)) {
    return kUnchanged;
  }
#elif defined(CINDER_AARCH64)
  if (instr->isAdd() || instr->isSub() || instr->isCompare()) {
    // add, sub, and cmp (which is a pseudo-instruction aliased to subs) all
    // support a 12-bit immediate optionally shifted by 20 bits.
    if (asmjit::arm::Utils::isAddSubImm(constant)) {
      return kUnchanged;
    }
  } else if (instr->isAnd() || instr->isOr() || instr->isXor()) {
    // and, or, and xor use a logical immediate, which is a 13-bit-encoded
    // operand that represents repeated 1 patterns.
    size_t bits = (&instr->output_)->sizeInBits();
    if (asmjit::arm::Utils::isLogicalImm(constant, bits < 32 ? 32 : bits)) {
      return kUnchanged;
    }
  } else {
    // mul has to use registers and does not support immediates.
  }
#else
  CINDER_UNSUPPORTED
#endif

  auto block = instr->basic_block_;
  auto move = block->allocateInstrBefore(instr_iter, Instruction::kMove);
  move->output_.setVirtualRegister();
  move->output_.setDataType(OperandBase::kObject);
  move->allocateImmediateInput(constant)->setDataType(in1->dataType());

  // If the first operand is smaller in size than the second operand, replace
  // the first operand with a sign-extended version that matches the size of the
  // second operand.
  if (instr->inputs_[0]->sizeInBits() < in1->sizeInBits()) {
    auto movsx = block->allocateInstrBefore(
        instr_iter, Instruction::kMovSX);
    movsx->output_.setVirtualRegister();
    movsx->output_.setDataType(in1->dataType());
    movsx->appendInput(instr->releaseInput(0));
    instr->setInput(0, new LinkedOperand(movsx));
  }

  // Replace the constant with the move.
  instr->setInput(1, new LinkedOperand(move));

  return kChanged;
}

#if defined(CINDER_X86_64)
// Rewrite storing a large immediate to a memory location in x86-64. Other
// architectures handle this explicitly in the autogen layer.
RewriteResult rewriteMoveToMemoryLargeConstant(instr_iter_t instr_iter) {
  // rewrite
  //     [Vreg0 + offset] = Imm64
  // to
  //     Vreg1 = Mov Imm64
  //     [Vreg0 + offset] = Vreg1

  auto instr = instr_iter;
  auto out = (&instr->output_);

  if (!(instr->isMove() || instr->isMoveRelaxed()) || !out->isInd()) {
    return kUnchanged;
  }

  auto input = instr->inputs_[0];
  if (!input->isImm() && !input->isMem()) {
    return kUnchanged;
  }

  auto constant = input->getConstantOrAddress();
  if (fitsSignedInt<32>(constant)) {
    return kUnchanged;
  }

  auto block = instr->basic_block_;
  auto move = block->allocateInstrBefore(instr_iter, Instruction::kMove);
  move->output_.setVirtualRegister();
  move->output_.setDataType(OperandBase::kObject);
  move->allocateImmediateInput(constant)->setDataType(input->dataType());

  // Replace the constant input with the move.
  instr->setInput(0, new LinkedOperand(move));

  return kChanged;
}
#endif

// Most guards involve comparing against a constant immediate. This rewrite
// ensures those immediates fit into comparison instructions (and if they do
// not it splits them).
RewriteResult rewriteGuardLargeConstant(instr_iter_t instr_iter) {
  auto instr = instr_iter;
  if (!instr->isGuard()) {
    return kUnchanged;
  }

  constexpr size_t kTargetIndex = 3;
  auto target_opnd = instr->inputs_[kTargetIndex];
  if (!target_opnd->isImm() && !target_opnd->isMem()) {
    return kUnchanged;
  }

  auto target_imm = target_opnd->getConstantOrAddress();

#if defined(CINDER_X86_64)
  if (fitsSignedInt<32>(target_imm)) {
    return kUnchanged;
  }
#elif defined(CINDER_AARCH64)
  if (asmjit::arm::Utils::isAddSubImm(target_imm)) {
    return kUnchanged;
  }
#else
  CINDER_UNSUPPORTED
#endif

  auto block = instr->basic_block_;
  auto move = block->allocateInstrBefore(instr_iter, Instruction::kMove);
  move->output_.setVirtualRegister();
  move->output_.setDataType(OperandBase::kObject);
  move->allocateImmediateInput(target_imm)->setDataType(
      target_opnd->dataType());
  instr->setInput(kTargetIndex, new LinkedOperand(move));
  return kChanged;
}

// Rewrite LoadArg to Bind and allocate a physical register for its input.
RewriteResult rewriteLoadArg(instr_iter_t instr_iter, Environ* env) {
  auto instr = instr_iter;
  if (!instr->isLoadArg()) {
    return kUnchanged;
  }
  instr->setOpcode(Instruction::kBind);
  JIT_CHECK(instr->num_inputs_ == 1, "expected one input");
  OperandBase* input = instr->inputs_[0];
  JIT_CHECK(input->isImm(), "expected constant arg index as input");
  auto arg_idx = input->getConstant();
  auto loc = env->arg_locations[arg_idx];
  static_cast<Operand*>(input)->setPhyRegOrStackSlot(loc);
  static_cast<Operand*>(input)->setDataType((&instr->output_)->dataType());
  return kChanged;
}

void populateLoadSecondCallResultPhi(
    DataType data_type,
    Instruction* phi1,
    Instruction* phi2,
    UnorderedMap<Operand*, Instruction*>& seen_srcs);

// Return an Instruction* (which may already exist) defining the second call
// result for src, with the given DataType.
//
// instr, if given, will be reused rather than inserting a new instruction (to
// preserve its vreg identity).
//
// seen_srcs is used to ensure only one Move is inserted for each root Call
// instruction in the presence of loops or repeated Phi uses of the same vreg.
Instruction* getSecondCallResult(
    DataType data_type,
    Operand* src,
    Instruction* instr,
    UnorderedMap<Operand*, Instruction*>& seen_srcs) {
  auto it = seen_srcs.find(src);
  if (it != seen_srcs.end()) {
    return it->second;
  }
  Instruction* src_instr = src->instr();
  BasicBlock* src_block = src_instr->basic_block_;
  auto src_it = src_block->iterator_to(src_instr);
  JIT_CHECK(
      src_instr->isCall() || src_instr->isPhi(),
      "LoadSecondCallResult input must come from Call or Phi, not '{}'",
      *src_instr);

  if (src_instr->isCall()) {
    // Check that this Call hasn't already been handled on behalf of another
    // LoadSecondCallResult. If we need to support this pattern in the future,
    // this rewrite function should probably become a standalone pass, with the
    // scope of seen_srcs expanded to the whole function.
    Instruction* next_instr_ptr = src_it->next_;
    if (next_instr_ptr != nullptr) {
      Instruction* next_instr = next_instr_ptr;
      JIT_CHECK(
          !(next_instr->isMove() && next_instr->num_inputs_ == 1 &&
            next_instr->inputs_[0]->isReg() &&
            next_instr->inputs_[0]->getPhyRegister() == RETURN_REGS[1]),
          "Call output consumed by multiple LoadSecondCallResult instructions");
    }
  }

  if (instr) {
    // We want to keep using the vreg defined by instr, so move it to after
    // src_instr, rather than allocating a new one.
    BasicBlock* instr_block = instr->basic_block_;
    instr_block->removeInstr(instr);
    // Insert after src_it (which is src_instr)
    Instruction* after_src = src_it->next_;
    src_block->insertInstrBefore(after_src, instr);
    instr->setNumInputs(0);
  }

  Instruction::Opcode new_op =
      src_instr->isCall() ? Instruction::kMove : Instruction::kPhi;
  if (instr) {
    instr->setOpcode(new_op);
  } else {
    instr = src_block->allocateInstrBefore(src_it->next_, new_op);
    instr->output_.setVirtualRegister();
    instr->output_.setDataType(data_type);
  }
  seen_srcs[src] = instr;
  if (new_op == Instruction::kMove) {
    instr->allocatePhyRegisterInput(RETURN_REGS[1])->setDataType(data_type);
  } else {
    // instr is now a Phi (either newly-created or a replacement for
    // instr). Recursively populate its inputs with the second result of all
    // original Calls.
    populateLoadSecondCallResultPhi(data_type, src_instr, instr, seen_srcs);
  }

  return instr;
}

// Given a Phi that joins the outputs of multiple Calls (or more Phis that
// ultimately join the outputs of Calls), populate a second, parallel Phi to
// join the second result of all original Calls.
void populateLoadSecondCallResultPhi(
    DataType data_type,
    Instruction* phi1,
    Instruction* phi2,
    UnorderedMap<Operand*, Instruction*>& seen_srcs) {
  for (size_t i = 1; i < phi1->num_inputs_; i += 2) {
    Operand* src1 = phi1->inputs_[i]->getDefine();
    Instruction* instr2 =
        getSecondCallResult(data_type, src1, nullptr, seen_srcs);
    phi2->allocateLabelInput(phi1->inputs_[i - 1]->getBasicBlock());
    phi2->allocateLinkedInput(instr2);
  }
}

// Replace LoadSecondCallResult instructions with an appropriate Move.
RewriteResult rewriteLoadSecondCallResult(instr_iter_t instr_iter) {
  // Replace "%x = LoadSecondCallResult %y" with "%x = Move RDX" immediately
  // after the call that defines %y. If necessary, trace through Phis,
  // inserting multiple Moves and a new Phi to reconcile them.

  Instruction* instr = instr_iter;
  if (!instr->isLoadSecondCallResult()) {
    return kUnchanged;
  }

  Operand* src = instr->inputs_[0]->getDefine();
  UnorderedMap<Operand*, Instruction*> seen_srcs;
  getSecondCallResult((&instr->output_)->dataType(), src, instr, seen_srcs);
  return kRemoved;
}

#if defined(CINDER_AARCH64)
// On AArch64, we never are going to produce an output that is less than 32-bits
// for our comparisons so promote all of these to 32-bits so we don't need to
// mask them.
RewriteResult rewritePromoteOutputSize(instr_iter_t instr_iter) {
  auto instr = instr_iter;
  switch (instr->opcode_) {
    case Instruction::kEqual:
    case Instruction::kNotEqual:
    case Instruction::kGreaterThanSigned:
    case Instruction::kGreaterThanEqualSigned:
    case Instruction::kLessThanSigned:
    case Instruction::kLessThanEqualSigned:
    case Instruction::kGreaterThanUnsigned:
    case Instruction::kGreaterThanEqualUnsigned:
    case Instruction::kLessThanUnsigned:
    case Instruction::kLessThanEqualUnsigned:
      if ((&instr->output_)->sizeInBits() < 32) {
        (&instr->output_)->setDataType(DataType::k32bit);
        return kChanged;
      }
      return kUnchanged;
    default:
      return kUnchanged;
  }
}
#endif

} // namespace

void PostGenerationRewrite::registerRewrites() {
  // rewriteInlineHelper should occur before other rewrites.
  registerOneRewriteFunction(rewriteInlineHelper, 0);

  registerOneRewriteFunction(rewriteBinaryOpConstantPosition, 1);
  registerOneRewriteFunction(rewriteBinaryOpLargeConstant, 1);
  registerOneRewriteFunction(rewriteGuardLargeConstant, 1);
  registerOneRewriteFunction(rewriteLoadArg, 1);

#if defined(CINDER_X86_64)
  registerOneRewriteFunction(rewriteMoveToMemoryLargeConstant, 1);
#elif defined(CINDER_AARCH64)
  registerOneRewriteFunction(rewritePromoteOutputSize, 1);
#endif

  registerOneRewriteFunction(rewriteLoadSecondCallResult, 1);
}

} // namespace jit::lir
