// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

/* ---- C API (implemented in cold_block_marker.c) ---- */
#ifdef __cplusplus
extern "C" {
#endif

void jit_lir_mark_cold_blocks(void* func);

#ifdef __cplusplus
} /* extern "C" */
#endif

#ifdef __cplusplus
namespace jit::lir {

class Function;

// Mark cold basic blocks in the LIR function.
//
// Uses static heuristics to identify blocks that are unlikely to execute
// on the hot path. Cold blocks are moved to a separate code section by
// the code generator, improving I-cache utilisation.
//
// Heuristics (in order of application):
//   H1: Guard failure targets — blocks reached only via guard failure
//   H2: Blocks containing only a deopt (single-instruction deopt stubs)
//   H3: Transitive closure — blocks reachable only from cold blocks
//
// Call after LIR generation, before register allocation.
inline void markColdBlocks(Function* func) {
  jit_lir_mark_cold_blocks(static_cast<void*>(func));
}

} // namespace jit::lir
#endif
