/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * F-1 codegen-time gate-correctness oracle.  See _oracle.h header
 * comment for purpose.  Independence rationale below.
 *
 * Structural independence from production gate at
 * frame_asm_c.c:527-532:
 *
 *   - Different translation unit (NEW file; production gate edits
 *     stay confined to frame_asm_c.c).
 *   - Reverse condition ordering (cellvars/freevars first; frame
 *     mode middle; generator last).  Production orders generator
 *     check first.
 *   - Different field-load paths for frame_mode: queries
 *     jit_get_config()->frame_mode against the JIT_FRAME_LIGHTWEIGHT
 *     enum value defined in jit_config_c.h, instead of going through
 *     jit_hir_func_get_frame_mode + the FRAME_MODE_LIGHTWEIGHT
 *     #define local to frame_asm_c.c.  The two paths must agree at
 *     compile time because jit_hir_func's frame_mode is initialised
 *     from jit_get_config().
 *   - Different control-flow shape (sequence of early-exit returns
 *     vs &&-chained boolean compose).
 *   - Authored from design intent in theologian 06:28:41Z +
 *     06:29:45Z confirm (4 conditions: lightweight + non-generator
 *     + numCellvars==0 + numFreevars==0) without re-reading
 *     frame_asm_c.c:527-532 mid-author.
 *
 * Pydebug-only consumer.  Production builds compile this TU but
 * the JIT_DCHECK at the gate site under !defined(Py_DEBUG) is
 * compiled out, leaving this oracle an unused symbol the linker
 * may strip.
 */

#include "Python.h"

#include "cinderx/Jit/codegen/frame_asm_c_oracle.h"
#include "cinderx/Jit/jit_config_c.h"
#include "cinderx/Jit/lir/lir_c_api.h"

int
frame_asm_c_full_inline_gate_oracle(const void* hir_func)
{
    PyCodeObject* code = (PyCodeObject*)jit_hir_func_get_code(hir_func);

    /* (1) zero cellvar slots — inline body has no cell init/clear. */
    if (code->co_ncellvars != 0) {
        return 0;
    }

    /* (2) zero freevar slots — inline body has no freevar
     * Ci_STACK_CLEAR loop. */
    if (code->co_nfreevars != 0) {
        return 0;
    }

    /* (3) lightweight frame mode — re-fetched from JitConfig rather
     * than from the HIR function's mirrored copy, so an init-time
     * mismatch between config and hir_func->frameMode would surface
     * as oracle/gate divergence. */
    if (jit_get_config()->frame_mode != JIT_FRAME_LIGHTWEIGHT) {
        return 0;
    }

    /* (4) non-generator — read directly from PyCodeObject->co_flags
     * with the bitmask spelled out in-line, distinct from the
     * jit_hir_func_is_gen helper that computes the same predicate. */
    const int gen_flags =
        CO_ASYNC_GENERATOR
        | CO_COROUTINE
        | CO_GENERATOR
        | CO_ITERABLE_COROUTINE;
    if (code->co_flags & gen_flags) {
        return 0;
    }

    return 1;
}
