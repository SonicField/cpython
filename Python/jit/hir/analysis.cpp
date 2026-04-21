// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cinderx/Jit/hir/analysis.h"

#include "cinderx/Jit/hir/hir_type_c.h"
#include "cinderx/StaticPython/checked_dict.h"
#include "cinderx/StaticPython/checked_list.h"

namespace jit::hir {

/* Convert C++ Type to C HirType via field-by-field conversion. */
static inline HirType to_hir(Type t) {
  return Type::toHirType(t);
}

static bool isSingleCInt(Type t) {
  HirType h = to_hir(t);
  return hir_type_is_subtype(h, to_hir(TCInt8)) ||
      hir_type_is_subtype(h, to_hir(TCUInt8)) ||
      hir_type_is_subtype(h, to_hir(TCInt16)) ||
      hir_type_is_subtype(h, to_hir(TCUInt16)) ||
      hir_type_is_subtype(h, to_hir(TCInt32)) ||
      hir_type_is_subtype(h, to_hir(TCUInt32)) ||
      hir_type_is_subtype(h, to_hir(TCInt64)) ||
      hir_type_is_subtype(h, to_hir(TCUInt64));
}

bool registerTypeMatches(Type op_type, OperandType expected_type) {
  HirType op_hir = to_hir(op_type);
  switch (expected_type.kind) {
    case Constraint::kType:
      return hir_type_is_subtype(op_hir, to_hir(expected_type.type));
    case Constraint::kTupleExactOrCPtr:
      return hir_type_is_subtype(op_hir, to_hir(TTupleExact)) ||
          hir_type_is_subtype(op_hir, to_hir(TCPtr));
    case Constraint::kListOrChkList:
      return hir_type_is_subtype(op_hir, to_hir(TList)) ||
          (hir_type_has_type_spec(&op_hir) &&
           Ci_CheckedList_TypeCheck(hir_type_type_spec(&op_hir)));
    case Constraint::kDictOrChkDict:
      return hir_type_is_subtype(op_hir, to_hir(TDict)) ||
          (hir_type_has_type_spec(&op_hir) &&
           Ci_CheckedDict_TypeCheck(hir_type_type_spec(&op_hir)));
    case Constraint::kOptObjectOrCIntOrCBool:
      return hir_type_is_subtype(op_hir, to_hir(TOptObject)) ||
          hir_type_is_subtype(op_hir, to_hir(TCInt)) ||
          hir_type_is_subtype(op_hir, to_hir(TCBool));
    case Constraint::kOptObjectOrCInt:
      return hir_type_is_subtype(op_hir, to_hir(TOptObject)) ||
          hir_type_is_subtype(op_hir, to_hir(TCInt));
    case Constraint::kMatchAllAsCInt:
      return isSingleCInt(op_type);
    case Constraint::kMatchAllAsPrimitive:
      return isSingleCInt(op_type) ||
          hir_type_is_subtype(op_hir, to_hir(TCBool)) ||
          hir_type_is_subtype(op_hir, to_hir(TCDouble)) ||
          hir_type_is_subtype(op_hir, to_hir(TCPtr));
  }
  JIT_ABORT("unknown constraint");
}

bool operandsMustMatch(OperandType op_type) {
  switch (op_type.kind) {
    case Constraint::kMatchAllAsCInt:
    case Constraint::kMatchAllAsPrimitive:
      return true;

    case Constraint::kType:
    case Constraint::kTupleExactOrCPtr:
    case Constraint::kListOrChkList:
    case Constraint::kDictOrChkDict:
    case Constraint::kOptObjectOrCInt:
    case Constraint::kOptObjectOrCIntOrCBool:
      return false;
  }
  JIT_ABORT("unknown constraint");
}

} // namespace jit::hir
