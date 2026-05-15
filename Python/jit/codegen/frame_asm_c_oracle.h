/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * F-1 codegen-time gate-correctness oracle for the W-PERF Class A
 * full-inline scope-expansion gate at frame_asm_c.c:527-532.
 *
 * Authored from theologian 2026-05-15T06:28:41Z + 06:29:45Z design
 * intent (4-condition gate-passes-full-inline) WITHOUT consulting the
 * production gate composition; agreement under JIT_DCHECK at the
 * production gate site is the F-1.a falsifier per supervisor
 * 06:27:29Z dispatch.
 *
 * Pydebug-only.  Not referenced from production code paths.
 */

#ifndef Py_JIT_CODEGEN_FRAME_ASM_C_ORACLE_H
#define Py_JIT_CODEGEN_FRAME_ASM_C_ORACLE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Returns 1 iff the JIT-compiled function described by hir_func
 * matches the intended full-inline gate predicate independently
 * derived from theologian's design.  See frame_asm_c_oracle.c for
 * structural-independence rationale. */
int frame_asm_c_full_inline_gate_oracle(const void* hir_func);

#ifdef __cplusplus
}
#endif

#endif /* Py_JIT_CODEGEN_FRAME_ASM_C_ORACLE_H */
