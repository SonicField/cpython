// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/Jit/hir/function.h"
#include "cinderx/Jit/hir/hir.h"

#include <unordered_set>

namespace jit::hir {

class BasicBlock;
class Register;

using RegisterSet = std::unordered_set<Register*>;

bool operandsMustMatch(OperandType op_type);
bool registerTypeMatches(Type op_type, OperandType expected_type);

} // namespace jit::hir
