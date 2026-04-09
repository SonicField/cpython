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

/* Verify table names and is_deopt_base match FOREACH_OPCODE order */
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
        assert(info->is_deopt_base ==                                  \
               (int)std::is_base_of_v<DeoptBase, opname> &&            \
               "hir_instr_info is_deopt_base mismatch: " #opname);    \
    }
    FOREACH_OPCODE(CHECK_OPCODE)
#undef CHECK_OPCODE
}

__attribute__((constructor))
static void hir_instr_info_startup_check() {
    verify_hir_instr_info();
}
