// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/Jit/lir/lir_c_api.h"

/* ---- C API (implemented in blocksorter.c) ---- */
#ifdef __cplusplus
extern "C" {
#endif

/* Sort basic blocks in reverse post-order with SCC handling.
 * blocks[0] is the entry block, blocks[count-1] is the exit block.
 * Returns a newly allocated array of sorted blocks; caller frees with
 * PyMem_RawFree(). Sets *out_count to the number of blocks returned. */
JitLirBlock *jit_lir_sort_blocks_rpo(
    JitLirBlock *blocks, size_t count, size_t *out_count);

#ifdef __cplusplus
} /* extern "C" */
#endif

#ifdef __cplusplus
#include "cinderx/Jit/lir/block.h"

namespace jit::lir {

// Phase B3d: BasicBlockSorter removed — Function::sortBasicBlocks()
// now calls jit_lir_sort_blocks_rpo() directly on raw array.

} // namespace jit::lir
#endif
