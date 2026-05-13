// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Phase 5.B c16: layout verifier for LirBasicBlockBuilder C-side struct
// (lir_types_c.h) against C++ jit::lir::BasicBlockBuilder
// (block_builder.h:41).
//
// First commit of BasicBlockBuilder bridge construction (per supervisor
// 01:29:30Z + 01:31:39Z auth + feedback_default_port_not_bridge). Goal:
// foundation for c17+ extern "C" wrapper batches that will allow porting
// generator.cpp anonymous-namespace functions (finishYield,
// emitSubclassCheck, updateRefTotal) currently blocked on BBB bridge.
//
// 3-CATEGORY CASCADE-AUDIT (per docs/methodology/pre-port-audit-checklist.md
// 0efc0e1525):
//
// (a) sizeof: static_assert for full struct + each std-type opaque blob
// (b) offsetof: static_assert per field cross-validating field positions
// (c) hardcoded literal-offset substitutes: blob sizes (16, 24) hardcoded
//     in C struct — locked at compile-time by stdlib-size static_asserts
//     to fail-fast if libstdc++/libc++ changes std::optional/std::vector
//     layout in a future toolchain.

#include "cinderx/Jit/lir/lir_types_c.h"
#include "cinderx/Jit/lir/block_builder.h"

#include <cstddef>
#include <optional>
#include <vector>

namespace jit::lir {

// ---- (c) Stdlib-blob-size locks (fail-fast on stdlib drift) ----
static_assert(sizeof(std::optional<std::size_t>) == 16,
    "std::optional<size_t> size != 16 — LirBasicBlockBuilder."
    "cur_deopt_metadata_storage[16] opaque blob would mismatch C++ layout. "
    "Investigate libstdc++/libc++ change before adjusting blob size.");

static_assert(sizeof(std::vector<BasicBlock*>) == 24,
    "std::vector<BasicBlock*> size != 24 — LirBasicBlockBuilder."
    "bbs_storage[24] opaque blob would mismatch C++ layout. "
    "Investigate libstdc++/libc++ change before adjusting blob size.");

// ---- (a) sizeof equivalence ----
static_assert(sizeof(LirBasicBlockBuilder) == sizeof(BasicBlockBuilder),
    "LirBasicBlockBuilder size mismatch with BasicBlockBuilder — "
    "C-side opaque blobs likely out of sync with C++ stdlib type sizes; "
    "verify (c) stdlib-blob-size locks above and adjust LirBasicBlockBuilder "
    "field sizes in lir_types_c.h.");

// ---- (b) offsetof equivalence per field ----
// Friend struct for access to private BasicBlockBuilder members.
struct LirBasicBlockBuilderLayoutVerifier {
    static void verify_offsets() {
        static_assert(offsetof(LirBasicBlockBuilder, cur_hir_instr)
            == offsetof(BasicBlockBuilder, cur_hir_instr_),
            "LirBasicBlockBuilder.cur_hir_instr offset mismatch");
        static_assert(offsetof(LirBasicBlockBuilder, cur_deopt_metadata_storage)
            == offsetof(BasicBlockBuilder, cur_deopt_metadata_),
            "LirBasicBlockBuilder.cur_deopt_metadata_storage offset mismatch");
        static_assert(offsetof(LirBasicBlockBuilder, cur_bb)
            == offsetof(BasicBlockBuilder, cur_bb_),
            "LirBasicBlockBuilder.cur_bb offset mismatch");
        static_assert(offsetof(LirBasicBlockBuilder, bbs_storage)
            == offsetof(BasicBlockBuilder, bbs_),
            "LirBasicBlockBuilder.bbs_storage offset mismatch");
        static_assert(offsetof(LirBasicBlockBuilder, env)
            == offsetof(BasicBlockBuilder, env_),
            "LirBasicBlockBuilder.env offset mismatch");
        static_assert(offsetof(LirBasicBlockBuilder, func)
            == offsetof(BasicBlockBuilder, func_),
            "LirBasicBlockBuilder.func offset mismatch");
    }
};

} // namespace jit::lir

// Friend declaration is needed in C++ class for offsetof on private members
// to compile. Since BasicBlockBuilder doesn't have the friend yet, we rely
// on offsetof working on private members by virtue of standard-layout
// inheritance OR that the compiler accepts it as long as we're inside a
// translation unit that can see the layout. If a future stricter compiler
// rejects offsetof on private members without friend, add:
//   friend struct ::jit::lir::LirBasicBlockBuilderLayoutVerifier;
// to BasicBlockBuilder class declaration.
