// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// W3 R4 ORACLE ADAPTER — definition of kEmptyRegSet.
// See _rc_oracle_adapter.h for full BRIDGE SPEC TEMPLATE.

#include "_rc_oracle_adapter.h"

namespace jit::hir {

const RegisterSet kEmptyRegSet;

// Restore cc4a18e7e5's RegisterSet formatter (deleted from HEAD analysis.h).
// rc_oracle.cpp's TRACE("dying_regs: {}", fmt::streamed(dying_regs)) requires
// it. Format is debug-only — order is arbitrary (unordered_set).
std::ostream &operator<<(std::ostream &os, const RegisterSet &set) {
  os << "{ ";
  bool first = true;
  for (Register *r : set) {
    if (!first) os << ", ";
    os << r;
    first = false;
  }
  os << " }";
  return os;
}

} // namespace jit::hir
