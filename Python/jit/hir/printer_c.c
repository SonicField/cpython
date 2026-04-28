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

/* B2 commit-3: format_name family needs Python.h (PyTuple_GET_ITEM,
 * PyUnicode_AsUTF8AndSize) and getVarnameTuple.  Per
 * feedback_no_pythonh_headers.md the include lives only in the .c
 * body, never in printer_c.h. */
#include "cinderx/python.h"

/* getVarnameTuple is declared in cinderx/Common/code.h, but that header
 * transitively includes code_extra.h which uses C++ 'bool' without
 * <stdbool.h> guard.  Forward-declare locally per JIT C-header pattern
 * (feedback_no_pythonh_headers.md) until the upstream cleanup lands. */
extern PyObject *getVarnameTuple(PyCodeObject *code, int *idx);

#include "cinderx/Jit/hir/hir_basic_block_c.h"  /* HirBasicBlock, HirCFG, HirEdge, hir_bb_* */
#include "cinderx/Jit/hir/hir_c_api.h"          /* hir_func_fullname, hir_func_cfg, hir_cfg_get_rpo */

#include <stdlib.h>  /* qsort */

/* B2 commit-1 port of escape_unicode (printer.cpp:162-187).  Writes
 * directly to `out` instead of building a std::string; semantically
 * identical otherwise.  kMaxASCII (printer.cpp:160) inlined as 127. */
void phx_hir_escape_unicode_chars(FILE *out, const char *data, ptrdiff_t size) {
    fputc('"', out);
    for (ptrdiff_t i = 0; i < size; i++) {
        char c = data[i];
        switch (c) {
            case '"':
            case '\\':
                fputc('\\', out);
                fputc(c, out);
                break;
            case '\n':
                fputs("\\n", out);
                break;
            default:
                if ((unsigned char)c > 127) {
                    fprintf(out, "\\%u", (unsigned)(unsigned char)c);
                } else {
                    fputc(c, out);
                }
                break;
        }
    }
    fputc('"', out);
}

/* B2 commit-2 port of HIRPrinter::Print(Function) — printer.cpp:56-66.
 * Writes "fun {name} {\n" + indented CFG + "}\n".  func is HirFunction*
 * (opaque).  Behavior matches C++ exactly. */
void phx_hir_print_function(FILE *out, PhxHirPrinter *p, const void *func) {
    HirFunction f = (HirFunction)(void *)func;
    const char *name = hir_func_fullname(f);
    fprintf(out, "fun %s {\n", (name && name[0]) ? name : "<unknown>");
    p->func = func;
    phx_hir_printer_indent(p);
    phx_hir_print_cfg(out, p, hir_func_cfg(f));
    phx_hir_printer_dedent(p);
    p->func = NULL;
    fputs("}\n", out);
}

/* B2 commit-2 port of HIRPrinter::Print(CFG) — printer.cpp:68-78.
 * Walks the CFG in RPO order (matching C++ GetRPOTraversal output) and
 * prints each block separated by blank lines. */
void phx_hir_print_cfg(FILE *out, PhxHirPrinter *p, const void *cfg) {
    struct HirCFG *cfg_mut = (struct HirCFG *)(void *)cfg;
    /* Two-call pattern: first to size, then to fill.  Matches existing
     * hir_cfg_get_rpo callers in the JIT pipeline. */
    size_t n = hir_cfg_get_rpo(cfg_mut, NULL, 0);
    if (n == 0) {
        return;
    }
    struct HirBasicBlock **blocks =
        (struct HirBasicBlock **)malloc(n * sizeof(*blocks));
    if (blocks == NULL) {
        return;
    }
    hir_cfg_get_rpo(cfg_mut, blocks, n);
    for (size_t i = 0; i < n; i++) {
        phx_hir_print_basic_block(out, p, blocks[i]);
        if (i + 1 < n) {
            fputc('\n', out);
        }
    }
    free(blocks);
}

/* qsort comparator: sort in_edges by source-block id (matches the
 * std::sort(... e1->from()->id < e2->from()->id) call at printer.cpp:86). */
static int phx_hir_in_edge_cmp(const void *a, const void *b) {
    const HirEdge *ea = *(const HirEdge *const *)a;
    const HirEdge *eb = *(const HirEdge *const *)b;
    int ida = hir_bb_id((const HirBasicBlock *)ea->from);
    int idb = hir_bb_id((const HirBasicBlock *)eb->from);
    return (ida > idb) - (ida < idb);
}

/* B2 commit-2 port of HIRPrinter::Print(BasicBlock) — printer.cpp:80-105.
 * Writes "bb {id}" + optional " (preds N1, N2, ...)" + " {\n", recurses
 * into each instruction (via the C++ bridge until commit-5 lands the
 * pure-C Print(Instr)), then closes "}\n".  in_edges sorted by source
 * block id for deterministic output. */
void phx_hir_print_basic_block(FILE *out, PhxHirPrinter *p, const void *block) {
    const HirBasicBlock *bb = (const HirBasicBlock *)block;
    phx_hir_printer_write_indent(out, p);
    fprintf(out, "bb %d", hir_bb_id(bb));

    size_t n_in = hir_bb_in_edges_count(bb);
    if (n_in > 0) {
        const HirEdge **edges =
            (const HirEdge **)malloc(n_in * sizeof(*edges));
        if (edges != NULL) {
            for (size_t i = 0; i < n_in; i++) {
                edges[i] = hir_bb_in_edge(bb, i);
            }
            qsort(edges, n_in, sizeof(*edges), phx_hir_in_edge_cmp);
            fputs(" (preds ", out);
            for (size_t i = 0; i < n_in; i++) {
                if (i > 0) {
                    fputs(", ", out);
                }
                fprintf(out, "%d",
                        hir_bb_id((const HirBasicBlock *)edges[i]->from));
            }
            fputc(')', out);
            free(edges);
        }
    }
    fputs(" {\n", out);

    phx_hir_printer_indent(p);
    /* Cast away const for hir_bb_first_instr/next_instr which take
     * non-const HirBasicBlock* (they don't mutate; const-correctness
     * of the underlying API is a separate cleanup). */
    HirBasicBlock *bb_mut = (HirBasicBlock *)(void *)bb;
    for (void *instr = hir_bb_first_instr(bb_mut); instr != NULL;
         instr = hir_bb_next_instr(bb_mut, instr)) {
        phx_hir_print_instr_cpp(out, p, instr);
    }
    phx_hir_printer_dedent(p);

    phx_hir_printer_write_indent(out, p);
    fputs("}\n", out);
}

/* B2 commit-3 internal: write "{idx}; \"{escaped name}\"" — port of
 * format_name_impl (printer.cpp:200-202) but with FILE* output. */
static void phx_format_name_impl(FILE *out, int idx, PyObject *names) {
    fprintf(out, "%d; ", idx);
    PyObject *name = PyTuple_GET_ITEM(names, idx);
    Py_ssize_t size;
    const char *data = PyUnicode_AsUTF8AndSize(name, &size);
    if (data == NULL) {
        PyErr_Clear();
        fputs("\"\"", out);
        return;
    }
    phx_hir_escape_unicode_chars(out, data, (ptrdiff_t)size);
}

/* B2 commit-3 port of format_name (printer.cpp:204-213).  Writes either
 * "{idx}; \"name\"" (when a code object is available) or just "{idx}"
 * (when func is NULL or codeFor returns NULL). */
void phx_format_name(FILE *out, const PhxHirPrinter *p, const void *instr, int idx) {
    const PyCodeObject *code = (p->func != NULL)
        ? (const PyCodeObject *)phx_hir_func_code_for(p->func, instr)
        : NULL;
    if (idx < 0 || code == NULL) {
        fprintf(out, "%d", idx);
        return;
    }
    phx_format_name_impl(out, idx, code->co_names);
}

/* B2 commit-3 port of format_load_super (printer.cpp:215-226). */
void phx_format_load_super(FILE *out, const PhxHirPrinter *p, const void *load_instr) {
    int name_idx = phx_hir_load_super_name_idx(load_instr);
    int no_args = phx_hir_load_super_no_args_in_super_call(load_instr);
    const PyCodeObject *code = (p->func != NULL)
        ? (const PyCodeObject *)phx_hir_func_code_for(p->func, load_instr)
        : NULL;
    if (code == NULL) {
        fprintf(out, "%d %d", name_idx, no_args);
        return;
    }
    phx_format_name_impl(out, name_idx, code->co_names);
    fprintf(out, ", %d", no_args);
}

/* B2 commit-3 port of format_varname (printer.cpp:228-237). */
void phx_format_varname(FILE *out, const PhxHirPrinter *p, const void *instr, int idx) {
    PyCodeObject *code = (p->func != NULL)
        ? (PyCodeObject *)phx_hir_func_code_for(p->func, instr)
        : NULL;
    if (idx < 0 || code == NULL) {
        fprintf(out, "%d", idx);
        return;
    }
    int adjusted = idx;
    PyObject *names = getVarnameTuple(code, &adjusted);
    phx_format_name_impl(out, adjusted, names);
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
