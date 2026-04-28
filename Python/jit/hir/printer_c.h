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

/* B2 commit-1: helper ports of HIRPrinter::Indent/Dedent/Indented from
 * printer.cpp:40-54.  State lives entirely in the caller-provided
 * PhxHirPrinter; helpers are inline so each Print(*) consumer pays no
 * cross-TU function-call overhead.  Behavior matches the C++ side
 * exactly (verified golden-byte-diff in commit-5 sole-path swap). */
static inline void phx_hir_printer_indent(PhxHirPrinter *p) {
    p->indent_level += 1;
}

static inline void phx_hir_printer_dedent(PhxHirPrinter *p) {
    p->indent_level -= 1;
}

static inline void phx_hir_printer_write_indent(FILE *out, const PhxHirPrinter *p) {
    if (p->line_prefix != NULL) {
        fputs(p->line_prefix, out);
    }
    for (int i = 0; i < p->indent_level; i++) {
        fputs("  ", out);
    }
}

/* B2 commit-2: C++ → FILE* bridges (defined in printer.cpp).  Forwards
 * Print(Instr) / Print(FrameState) calls into the not-yet-ported C++
 * HIRPrinter so phx_hir_print_basic_block can recurse correctly during
 * the staged port.  Removed in commit-5 once the C-side bodies land. */
void phx_hir_print_instr_cpp(FILE *out, const PhxHirPrinter *p, const void *instr);
void phx_hir_print_frame_state_cpp(FILE *out, const PhxHirPrinter *p, const void *state);

/* B2 commit-4: bridge into the static C++ format_immediates (185
 * case-branches; full port deferred to W-PRINTER-IMMEDIATES-PORT
 * residual workstream per theologian 11:23:20Z option A).  Used by
 * commit-5 Print(Instr) C wrapper. */
void phx_format_immediates_cpp(FILE *out, const PhxHirPrinter *p, const void *instr);

/* B2 commit-3: thin C-callable bridges for non-trivial Function /
 * LoadSuperBase methods used by the format_name family ports below.
 * Pointers are opaque (void*) — implementation in printer.cpp casts
 * to jit::hir::* directly.  Removed in commit-5 if the underlying
 * classes get full C-side coverage; until then 1-line bridges keep
 * hir_c_api free of printer-specific surface. */
const void *phx_hir_func_code_for(const void *func, const void *instr);
int phx_hir_load_super_name_idx(const void *instr);
int phx_hir_load_super_no_args_in_super_call(const void *instr);

/* B2 commit-5a: structural HIR accessors for print_reg_states +
 * Print(FrameState) C-side bodies.  Implementations in printer.cpp
 * (1-line wrappers).  Per theologian 11:35:47Z principled distinction:
 * these are structural-and-reusable accessors (RegState / Register /
 * FrameState / BlockStack), not the instruction-specific surface that
 * format_immediates would require — proper C-API additions, not the
 * single-use bridge pattern that justifies W-PRINTER-IMMEDIATES-PORT. */

/* PhxRegStateArray (printer.cpp:107 print_reg_states consumer). */
size_t phx_hir_reg_state_array_size(const void *array);
const void *phx_hir_reg_state_array_at(const void *array, size_t i);

/* RegState fields (register.h struct). */
const void *phx_hir_reg_state_reg(const void *rs);
int phx_hir_reg_state_ref_kind(const void *rs);    /* 0=Uncounted 1=Borrowed 2=Owned */
int phx_hir_reg_state_value_kind(const void *rs);  /* 0=Object 1=Signed 2=Unsigned 3=Bool 4=Double */

/* Register accessors. */
int phx_hir_register_id(const void *reg);
const char *phx_hir_register_name(const void *reg);

/* FrameState accessors (frame_state.h struct). */
int phx_hir_frame_state_cur_instr_offs(const void *state);
int phx_hir_frame_state_nlocals(const void *state);
size_t phx_hir_frame_state_localsplus_count(const void *state);
const void *phx_hir_frame_state_localsplus_at(const void *state, size_t i);
size_t phx_hir_frame_state_stack_count(const void *state);
const void *phx_hir_frame_state_stack_at(const void *state, size_t i);

/* BlockStack accessors (frame_state.h ExecutionBlock). */
size_t phx_hir_frame_state_block_stack_size(const void *state);
int phx_hir_frame_state_block_stack_opcode(const void *state, size_t i);
int phx_hir_frame_state_block_stack_handler_off(const void *state, size_t i);
int phx_hir_frame_state_block_stack_stack_level(const void *state, size_t i);

/* B2 commit-5a: escape_unicode PyObject overload — declared in the
 * Python-aware caller printer_c.c (Python.h required for PyObject).
 * Forward decl here uses opaque void* per
 * feedback_no_pythonh_headers.md keep-Python.h-out-of-headers rule. */
void phx_hir_escape_unicode_pyobject(FILE *out, const void *str);

/* B2 commit-5a: print_reg_states port (printer.cpp:107-158). */
void phx_hir_print_reg_states(FILE *out, const void *reg_states_array);

/* B2 commit-3: format_name family C ports — printer.cpp:200-237.
 * Each writes its formatted text directly to FILE* `out` (no
 * intermediate std::string).  PhxHirPrinter `p` carries the Function*
 * via p->func; bodies handle the func==NULL fallback (writes just the
 * raw idx).  format_name_impl is internal-only. */
void phx_format_name(FILE *out, const PhxHirPrinter *p, const void *instr, int idx);
void phx_format_load_super(FILE *out, const PhxHirPrinter *p, const void *load_instr);
void phx_format_varname(FILE *out, const PhxHirPrinter *p, const void *instr, int idx);

/* B2 commit-1: helper port of escape_unicode (printer.cpp:162-187).
 * Writes a JSON-style escaped representation of the input bytes to
 * `out`: ASCII printable preserved, '"' and '\\' backslash-escaped,
 * '\\n' as "\\n", non-ASCII bytes as decimal-escaped "\NNN".
 *
 * `size` uses ptrdiff_t — semantically equivalent to Py_ssize_t and
 * avoids forcing Python.h into every includer of printer_c.h
 * (feedback_no_pythonh_headers.md).
 *
 * The C++ side has a second overload taking PyObject* (printer.cpp:189);
 * that variant is added in commit-1b alongside the PhxRegStateArray
 * accessor since its only intended caller (print_reg_states /
 * Print(Instr)) is part of that batch. */
#include <stddef.h>
void phx_hir_escape_unicode_chars(FILE *out, const char *data, ptrdiff_t size);

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
