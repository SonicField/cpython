/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C entry points for the HIR printer — Phase 4.A pilot #2 conversion
 * of printer.cpp (theologian invariant pre-audit 09:16:10Z; supervisor
 * 09:12:00Z).
 *
 * Pattern adopted from Python/jit/lir/printer_c.{h,c} (D-1775548957),
 * with two deliberate divergences flagged by librarian 09:16:36Z:
 *  - open_memstream auto-grow buffer in the C++ shell (printer.h)
 *    instead of LIR's 4 KiB fmemopen stack buffer — avoids silent
 *    truncation on large HIR Function dumps with full snapshots.
 *  - fmt/ostream.h include strictly inside #ifdef __cplusplus in
 *    printer.h — never reached from this .c TU.
 *
 * Skeleton stage (B1): all entries have empty bodies. Print(Function)
 * etc. are wired in subsequent batches. Existing C++ HIRPrinter in
 * printer.h continues to own behavior until the C entries are
 * sole-path swapped (B5).
 */
#pragma once

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Per-call printer state. Mirrors the C++ HIRPrinter members
 * (indent_level_, line_prefix_, full_snapshots_, func_); allocated by
 * the caller (typically C++ shell) and threaded through every entry
 * point so the C side stays stateless. */
typedef struct PhxHirPrinter {
    const void *func;            /* HirFunction; NULL when printing
                                    a free-standing instr / block */
    const char *line_prefix;     /* NULL → no prefix */
    int indent_level;
    int full_snapshots;          /* 0 = Snapshot opcode only;
                                    1 = full FrameState body */
} PhxHirPrinter;

/* Convenience initializer for the default printer (no prefix, indent
 * 0, summary snapshots). */
static inline PhxHirPrinter phx_hir_printer_default(void) {
    PhxHirPrinter p = {NULL, NULL, 0, 0};
    return p;
}

/* ---- Print entries (FILE*-output) ----
 *
 * Each entry writes the canonical text representation of its argument
 * to `out`. Inputs are opaque void* matching the C++ class via
 * reinterpret_cast in the C++ shell.
 *
 * B1 NOTE: bodies are empty stubs. Wiring happens in subsequent
 * batches. Calling these now produces no output. */
void phx_hir_print_function(FILE *out, PhxHirPrinter *p, const void *func);
void phx_hir_print_cfg(FILE *out, PhxHirPrinter *p, const void *cfg);
void phx_hir_print_basic_block(FILE *out, PhxHirPrinter *p, const void *block);
void phx_hir_print_instr(FILE *out, PhxHirPrinter *p, const void *instr);
void phx_hir_print_frame_state(FILE *out, PhxHirPrinter *p, const void *state);

#ifdef __cplusplus
} /* extern "C" */
#endif
