/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Pure-C body for the HIR printer — Phase 4.A pilot #2 conversion.
 *
 * Skeleton stage (B1). Each entry is an empty stub. Wiring happens
 * per-entry in subsequent batches; the C++ HIRPrinter in printer.cpp
 * continues to own behavior until those land + a sole-path swap
 * lands.
 *
 * No fmt/ostream.h or other C++ header is reachable from this TU —
 * that include lives strictly behind #ifdef __cplusplus in
 * printer.h, per librarian D-1775548957 trap-class avoidance.
 */

#include "cinderx/Jit/hir/printer_c.h"

void phx_hir_print_function(FILE *out, PhxHirPrinter *p, const void *func) {
    (void)out;
    (void)p;
    (void)func;
    /* B1 stub: see printer.cpp:HIRPrinter::Print(Function&) */
}

void phx_hir_print_cfg(FILE *out, PhxHirPrinter *p, const void *cfg) {
    (void)out;
    (void)p;
    (void)cfg;
    /* B1 stub: see printer.cpp:HIRPrinter::Print(CFG&) */
}

void phx_hir_print_basic_block(FILE *out, PhxHirPrinter *p, const void *block) {
    (void)out;
    (void)p;
    (void)block;
    /* B1 stub: see printer.cpp:HIRPrinter::Print(BasicBlock&) */
}

void phx_hir_print_instr(FILE *out, PhxHirPrinter *p, const void *instr) {
    (void)out;
    (void)p;
    (void)instr;
    /* B1 stub: see printer.cpp:HIRPrinter::Print(Instr&) */
}

void phx_hir_print_frame_state(FILE *out, PhxHirPrinter *p, const void *state) {
    (void)out;
    (void)p;
    (void)state;
    /* B1 stub: see printer.cpp:HIRPrinter::Print(FrameState&) */
}
