// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

// NOLINTNEXTLINE(facebook-unused-include-check)
#include "cinderx/Jit/codegen/arch/detection.h"
#include "fmt/ostream.h"

#ifdef PHOENIX_ASM
// Phoenix assembler — pure C replacement for asmjit
#include "jit/phoenix_asm/asmjit_compat.h"
#else
#include <asmjit/asmjit.h>
#endif

#include <ostream>

#if defined(CINDER_X86_64)

#include "cinderx/Jit/codegen/arch/x86_64.h"

#ifndef PHOENIX_ASM
#include <asmjit/x86/x86builder.h>
#include <asmjit/x86/x86emitter.h>
#include <asmjit/x86/x86operand.h>
#endif

namespace jit::codegen::arch {

using Builder = asmjit::x86::Builder;
using Emitter = asmjit::x86::Emitter;
using Gp = asmjit::x86::Gp;
using Mem = asmjit::x86::Mem;
using Reg = asmjit::x86::Reg;
using VecD = asmjit::x86::Xmm;

template <typename T>
using EmitterExplicitT = asmjit::x86::EmitterExplicitT<T>;

// If you change this register you'll also need to change the deopt
// trampoline code that saves all registers.
constexpr auto reg_scratch_deopt = asmjit::x86::r15;

constexpr auto reg_scratch_0_loc = RAX;

constexpr auto reg_general_return_loc = RAX;
constexpr auto reg_general_auxilary_return_loc = RDX;
constexpr auto reg_double_return_loc = XMM0;
constexpr auto reg_double_auxilary_return_loc = XMM1;
constexpr auto reg_frame_pointer_loc = RBP;
constexpr auto reg_stack_pointer_loc = RSP;

} // namespace jit::codegen::arch

#elif defined(CINDER_AARCH64)

#include "cinderx/Jit/codegen/arch/aarch64.h"

#ifndef PHOENIX_ASM
#include <asmjit/arm/a64builder.h>
#include <asmjit/arm/a64emitter.h>
#include <asmjit/arm/a64operand.h>
#include <asmjit/arm/armutils.h>
#endif

namespace jit::codegen::arch {

using Builder = asmjit::a64::Builder;
using Emitter = asmjit::a64::Emitter;
using Gp = asmjit::a64::Gp;
using Mem = asmjit::a64::Mem;
using Reg = asmjit::a64::Reg;
using VecD = asmjit::a64::Vec;

template <typename T>
using EmitterExplicitT = asmjit::a64::EmitterExplicitT<T>;

// If you change this register you'll also need to change the deopt
// trampoline code that saves all registers.
constexpr auto reg_scratch_deopt = asmjit::a64::x28;

constexpr auto reg_scratch_0 = asmjit::a64::x12;
constexpr auto reg_scratch_1 = asmjit::a64::x13;
constexpr auto reg_scratch_br = asmjit::a64::x16;

constexpr auto reg_scratch_0_loc = X12;

constexpr auto reg_general_return_loc = X0;
constexpr auto reg_general_auxilary_return_loc = X1;
constexpr auto reg_double_return_loc = D0;
constexpr auto reg_double_auxilary_return_loc = D1;
constexpr auto reg_frame_pointer_loc = X29;
constexpr auto reg_stack_pointer_loc = SP;

constexpr auto fp = asmjit::a64::x29;
constexpr auto lr = asmjit::a64::x30;

} // namespace jit::codegen::arch

#else

#include "cinderx/Jit/codegen/arch/unknown.h"

namespace jit::codegen::arch {

class Builder : public asmjit::BaseBuilder {
 public:
  explicit Builder(asmjit::CodeHolder* code = nullptr) noexcept {
    if (code) {
      code->attach(this);
    }
  }
};

using Emitter = asmjit::BaseEmitter;
using Gp = asmjit::BaseReg;
using Mem = asmjit::BaseMem;
using Reg = asmjit::BaseReg;
using VecD = asmjit::BaseReg;

template <typename T>
struct EmitterExplicitT {};

constexpr auto reg_scratch_deopt = asmjit::BaseReg();

constexpr auto reg_scratch_0_loc = R3;

constexpr auto reg_general_return_loc = R0;
constexpr auto reg_general_auxilary_return_loc = R1;
constexpr auto reg_double_return_loc = D0;
constexpr auto reg_double_auxilary_return_loc = D1;
constexpr auto reg_frame_pointer_loc = R3;
constexpr auto reg_stack_pointer_loc = SP;

} // namespace jit::codegen::arch

#endif

#if defined(CINDER_AARCH64)

#ifdef PHOENIX_ASM
/* Pure C implementations in arch.c */
#ifdef __cplusplus
extern "C" {
#endif
PhxMem jit_arch_ptr_offset(PhxGp base, int32_t offset, int32_t access_size);
PhxMem jit_arch_ptr_resolve(PhxBuilder *as, PhxGp base, int32_t offset,
                            PhxGp scratch, int32_t access_size);
#ifdef __cplusplus
}
#endif
#endif /* PHOENIX_ASM */

namespace jit::codegen::arch {

enum class AccessSize : int32_t { k8 = 1, k16 = 2, k32 = 4, k64 = 8 };

#ifdef PHOENIX_ASM
/* Inline wrappers calling the C implementations */
inline asmjit::a64::Mem ptr_offset(
    const asmjit::a64::Gp& base,
    int32_t offset,
    AccessSize access_size = AccessSize::k64) {
  return asmjit::a64::Mem(
      jit_arch_ptr_offset(base, offset, static_cast<int32_t>(access_size)));
}

inline asmjit::a64::Mem ptr_resolve(
    asmjit::a64::Builder* as,
    const asmjit::a64::Gp& base,
    int32_t offset,
    const asmjit::a64::Gp& scratch,
    AccessSize access_size = AccessSize::k64) {
  return asmjit::a64::Mem(
      jit_arch_ptr_resolve(as->impl(), base, offset, scratch,
                           static_cast<int32_t>(access_size)));
}
#else
asmjit::a64::Mem ptr_offset(
    const asmjit::a64::Gp& base,
    int32_t offset,
    AccessSize access_size = AccessSize::k64);

asmjit::a64::Mem ptr_resolve(
    asmjit::a64::Builder* as,
    const asmjit::a64::Gp& base,
    int32_t offset,
    const asmjit::a64::Gp& scratch,
    AccessSize access_size = AccessSize::k64);
#endif /* PHOENIX_ASM */

} // namespace jit::codegen::arch

#endif

namespace jit::codegen {

inline std::ostream& operator<<(std::ostream& out, const PhyLocation& loc) {
  return out << loc.toString();
}

} // namespace jit::codegen

inline auto format_as(jit::codegen::RegId reg) {
  return fmt::underlying(reg);
}

namespace std {

template <>
struct hash<jit::codegen::PhyLocation> {
  std::size_t operator()(jit::codegen::PhyLocation const& s) const noexcept {
    return s.loc;
  }
};

} // namespace std

template <>
struct fmt::formatter<jit::codegen::PhyLocation> : fmt::ostream_formatter {};
