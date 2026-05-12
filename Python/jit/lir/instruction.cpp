// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Phase 5.B c5: instruction.cpp now contains only InstrProperty
// (out-of-line getProperties + static prop_map_ vector built from
// FOREACH_INSTR_TYPE, plus 3 extern "C" query wrappers). Layout-pin
// static_asserts migrated to lir_instr_c_verify.cpp per
// feedback_verifier_pattern. All Instruction:: method bodies
// (destructor + 20 methods) were moved inline to instruction.h in
// Phase 5.A2 C2-C5. InstrProperty C-port + class elimination deferred
// to Phase 5.E bridge dissolution.

#include "cinderx/Jit/lir/instruction.h"

#include <array>
#include <utility>

namespace jit::lir {

// ---- InstrProperty (static data — stays in .cpp until C equivalent exists) ----

InstrProperty::InstrInfo& InstrProperty::getProperties(
    Instruction::Opcode opcode) {
  return prop_map_.at(opcode);
}

#define BEGIN_INSTR_PROPERTY \
  std::vector<InstrProperty::InstrInfo> InstrProperty::prop_map_ = {
#define END_INSTR_PROPERTY \
  }                        \
  ;

#define PROPERTY(__t, __p...) {#__t, __p},

// clang-format off
BEGIN_INSTR_PROPERTY
  FOREACH_INSTR_TYPE(PROPERTY)
END_INSTR_PROPERTY
// clang-format on

} // namespace jit::lir

// ---- Extern C wrappers for InstrProperty queries ----
// These query the authoritative C++ prop_map_ (generated from
// FOREACH_INSTR_TYPE). No data duplication — correct by construction.

extern "C" int lir_instr_get_output_phy_reg_use(int opcode) {
  using IP = jit::lir::InstrProperty;
  using Op = jit::lir::Instruction::Opcode;
  if (opcode < 0 || opcode > jit::lir::Instruction::kYieldValue) {
    return 1; // default: output uses phy reg
  }
  return IP::getProperties(static_cast<Op>(opcode)).output_phy_use;
}

extern "C" int lir_instr_get_input_phy_reg_use(int opcode, size_t i) {
  using IP = jit::lir::InstrProperty;
  using Op = jit::lir::Instruction::Opcode;
  if (opcode < 0 || opcode > jit::lir::Instruction::kYieldValue) {
    return 0;
  }
  auto& uses = IP::getProperties(static_cast<Op>(opcode)).input_phy_uses;
  if (i >= uses.size()) return 0;
  return uses[i];
}

extern "C" int lir_instr_inputs_live_across(int opcode) {
  using IP = jit::lir::InstrProperty;
  using Op = jit::lir::Instruction::Opcode;
  if (opcode < 0 || opcode > jit::lir::Instruction::kYieldValue) {
    return 0;
  }
  return IP::getProperties(static_cast<Op>(opcode)).inputs_live_across;
}
