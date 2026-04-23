/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Pure-C port of codeCost from inliner.cpp:100-108
 * (Tier 7 Batch 3 Cat-A per theologian 20:52:14Z + supervisor 20:52:14Z).
 *
 * Uses the existing JitBytecodeBlock C iteration API (Python/jit/
 * bytecode_c.h) — no new bridges needed (CACHE-skip + EXTENDED_ARG
 * handling already implemented in jit_bc_block_first/next).
 */

#include "cinderx/Jit/hir/inliner_helpers_c.h"
#include "cinderx/Jit/bytecode_c.h"

size_t phx_code_cost(PyCodeObject* code) {
    /* Manually iterating through the code block to count real opcodes
     * and not inline caches. Matches the original C++ semantics of
     * BytecodeInstructionBlock{code} iteration. */
    JitBytecodeBlock block;
    jit_bc_block_init(&block, code);

    JitBytecodeInstr instr;
    size_t num_opcodes = 0;
    if (jit_bc_block_first(&block, &instr)) {
        do {
            num_opcodes++;
        } while (jit_bc_block_next(&block, &instr));
    }
    return num_opcodes;
}
