/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * T2-A verification: cross-boundary check that hir_instr_info_c table
 * matches C++ InstrT<> metadata. Runs at program startup via
 * __attribute__((constructor)).
 */

#include "cinderx/Jit/hir/hir_instr_info_c.h"
#include "cinderx/Jit/hir/hir.h"

#include <cassert>
#include <cstdio>
#include <type_traits>

using namespace jit::hir;

/* Note: HIR_OP_COUNT (from hir_opcode_c.h) may differ from the C++
 * FOREACH_OPCODE count (hir_ops.h) — the C enum is a subset.
 * The table uses C++ FOREACH_OPCODE order (168 entries).
 * This verification checks names and is_deopt_base match the C++ enum. */

/* Standalone is_terminator check — mirrors Instr::IsTerminator() switch.
 * Cross-references: table (hir_instr_info_c.c) vs C++ (hir.cpp:337). */
static bool cpp_is_terminator(Opcode op) {
    switch (op) {
        case Opcode::kBranch:
        case Opcode::kDeopt:
        case Opcode::kCondBranch:
        case Opcode::kCondBranchIterNotDone:
        case Opcode::kCondBranchCheckType:
        case Opcode::kRaise:
        case Opcode::kRaiseAwaitableError:
        case Opcode::kRaiseStatic:
        case Opcode::kReturn:
        case Opcode::kUnreachable:
            return true;
        default:
            return false;
    }
}

/* Verify table names, is_deopt_base, has_output, is_terminator match C++ */
static void verify_hir_instr_info() {
#define CHECK_OPCODE(opname)                                            \
    {                                                                   \
        int op = static_cast<int>(Opcode::k##opname);                  \
        const HirInstrInfo *info = hir_instr_get_info(op);             \
        assert(info != nullptr);                                        \
        /* Name check */                                                \
        assert(strcmp(info->name, #opname) == 0 &&                     \
               "hir_instr_info name mismatch: " #opname);              \
        /* is_deopt_base must match C++ hierarchy (T2-C1 gate) */      \
        { bool cpp_is_deopt = std::is_base_of_v<DeoptBase, opname>;   \
        assert(info->is_deopt_base == (int)cpp_is_deopt &&             \
               "hir_instr_info is_deopt_base mismatch: " #opname); }  \
        /* has_output must match C++ constexpr (T2-C6 hardening) */    \
        { bool cpp_has_output = opname::has_output;                    \
        assert(info->has_output == (int)cpp_has_output &&              \
               "hir_instr_info has_output mismatch: " #opname); }     \
        /* is_terminator must match C++ switch (T2-C6 hardening) */    \
        { bool cpp_is_term = cpp_is_terminator(Opcode::k##opname);    \
        assert(info->is_terminator == (int)cpp_is_term &&              \
               "hir_instr_info is_terminator mismatch: " #opname); }  \
    }
    FOREACH_OPCODE(CHECK_OPCODE)
#undef CHECK_OPCODE
}

__attribute__((constructor))
static void hir_instr_info_startup_check() {
    verify_hir_instr_info();
}
