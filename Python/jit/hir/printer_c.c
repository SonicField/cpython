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

/* getVarnameTuple is C++-mangled (declared in jit/getVarnameTuple inside
 * `namespace jit`); use the C-shim phx_getVarnameTuple wrapper added in
 * jit_common/code.cpp per testkeeper 12:07:56Z option A.  Pre-swap LTO
 * dead-code-elimination was hiding the missing-symbol because
 * phx_format_varname had no live caller. */
extern PyObject *phx_getVarnameTuple(PyCodeObject *code, int *idx);

#include "cinderx/Jit/hir/hir_basic_block_c.h"  /* HirBasicBlock, HirCFG, HirEdge, hir_bb_* */
#include "cinderx/Jit/hir/hir_c_api.h"          /* hir_func_fullname, hir_func_cfg, hir_cfg_get_rpo */
#include "cinderx/Jit/hir/hir_instr_c.h"        /* hir_c_opcode + HirInstr layout */
#include "cinderx/Jit/hir/hir_opcode_c.h"       /* HIR_OP_* enum */

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
        /* B2 commit-5b sole-path: route through the C-side
         * phx_hir_print_instr port; commit-2's
         * phx_hir_print_instr_cpp bridge is now unused and removed. */
        phx_hir_print_instr(out, p, instr);
        fputc('\n', out);
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
    PyObject *names = phx_getVarnameTuple(code, &adjusted);
    phx_format_name_impl(out, adjusted, names);
}

/* B2 commit-5b: Print(Instr) port — printer.cpp:803-863.  Writes
 * "  {dst}:{type} = {opname}<{immediates}> {operands}" and an optional
 * DeoptBase / FrameState block.  format_immediates stays C++ (called
 * via commit-4 phx_format_immediates_cpp bridge); print_reg_states +
 * Print(FrameState) are commit-5a C ports. */
void phx_hir_print_instr(FILE *out, PhxHirPrinter *p, const void *instr) {
    phx_hir_printer_write_indent(out, p);

    const void *dst = hir_c_output(instr);
    if (dst != NULL) {
        fputs(phx_hir_register_name(dst), out);
        if (!phx_hir_register_type_is_top(dst)) {
            fputc(':', out);
            fputs(phx_hir_register_type_to_string(dst), out);
        }
        fputs(" = ", out);
    }
    fputs(phx_hir_instr_opname(instr), out);

    /* Immediate text. The C-side dispatcher handles ported cases
     * directly and falls through to phx_format_immediates_cpp (the
     * commit-4 bridge) for opcodes not yet migrated. Each ported case
     * does its own <...> wrapping if it has anything to write. */
    phx_format_immediates(out, p, instr);

    /* Operand list. */
    size_t n = hir_c_num_operands(instr);
    for (size_t i = 0; i < n; i++) {
        const void *op = hir_c_get_operand(instr, i);
        if (op != NULL) {
            fputc(' ', out);
            fputs(phx_hir_register_name(op), out);
        } else {
            fputs(" nullptr", out);
        }
    }

    if (hir_c_is_snapshot(instr) && !p->full_snapshots) {
        return;
    }

    /* Type-aware FrameState dispatch (matches C++ get_frame_state at
     * hir.cpp:1250-1261).  hir_c_get_frame_state blindly casts to
     * HirDeoptLayout and reads the frame_state field — wrong for
     * non-deopt non-snapshot non-BIF instrs (would return garbage and
     * crash here).  testkeeper 12:13:37Z SIGSEGV evidence. */
    const void *fs = phx_hir_instr_get_frame_state(instr);
    const void *db = hir_c_as_deopt(instr);
    if (db != NULL) {
        fputs(" {\n", out);
        phx_hir_printer_indent(p);
        const char *descr = phx_hir_deopt_descr(db);
        if (descr != NULL && descr[0] != '\0') {
            phx_hir_printer_write_indent(out, p);
            fprintf(out, "Descr '%s'\n", descr);
        }
        const void *guilty = phx_hir_deopt_guilty_reg(db);
        if (guilty != NULL) {
            phx_hir_printer_write_indent(out, p);
            fprintf(out, "GuiltyReg %s\n", phx_hir_register_name(guilty));
        }
        const void *live_regs = phx_hir_deopt_live_regs(db);
        if (phx_hir_reg_state_array_size(live_regs) > 0) {
            phx_hir_printer_write_indent(out, p);
            fputs("LiveValues", out);
            phx_hir_print_reg_states(out, live_regs);
            fputc('\n', out);
        }
        if (fs != NULL) {
            phx_hir_printer_write_indent(out, p);
            fputs("FrameState {\n", out);
            phx_hir_printer_indent(p);
            phx_hir_print_frame_state(out, p, fs);
            phx_hir_printer_dedent(p);
            phx_hir_printer_write_indent(out, p);
            fputs("}\n", out);
        }
        phx_hir_printer_dedent(p);
        phx_hir_printer_write_indent(out, p);
        fputc('}', out);
    } else if (fs != NULL) {
        fputs(" {\n", out);
        phx_hir_printer_indent(p);
        phx_hir_print_frame_state(out, p, fs);
        phx_hir_printer_dedent(p);
        phx_hir_printer_write_indent(out, p);
        fputc('}', out);
    }
}

/* B2 commit-5a: escape_unicode PyObject overload (printer.cpp:189-197).
 * Fetches UTF-8 bytes from the PyObject and delegates to the chars
 * version (commit-1). Errors during decode are silently swallowed
 * matching the C++ side's PyErr_Clear(). */
void phx_hir_escape_unicode_pyobject(FILE *out, const void *str) {
    Py_ssize_t size;
    const char *data = PyUnicode_AsUTF8AndSize((PyObject *)str, &size);
    if (data == NULL) {
        PyErr_Clear();
        fputs("\"\"", out);
        return;
    }
    phx_hir_escape_unicode_chars(out, data, (ptrdiff_t)size);
}

/* B2 commit-5a: qsort comparator for sorting RegState array by reg id
 * (matches std::sort lambda at printer.cpp:111-113). */
static int phx_hir_reg_state_cmp(const void *a, const void *b) {
    const void *ra = phx_hir_reg_state_reg(*(const void *const *)a);
    const void *rb = phx_hir_reg_state_reg(*(const void *const *)b);
    int ida = phx_hir_register_id(ra);
    int idb = phx_hir_register_id(rb);
    return (ida > idb) - (ida < idb);
}

/* B2 commit-5a: ref_kind/value_kind ordinals → prefix string.  Mirrors
 * the switch at printer.cpp:121-154.  Returns "?" for unrecognised
 * combinations (defensive; should not happen with valid HIR). */
static const char *phx_hir_reg_state_prefix(int value_kind, int ref_kind) {
    /* ValueKind ordinals: 0=Object 1=Signed 2=Unsigned 3=Bool 4=Double */
    /* RefKind ordinals: 0=Uncounted 1=Borrowed 2=Owned */
    switch (value_kind) {
        case 1: return "s";
        case 2: return "uns";
        case 3: return "bool";
        case 4: return "double";
        case 0:
            switch (ref_kind) {
                case 0: return "unc";
                case 1: return "b";
                case 2: return "o";
                default: return "?";
            }
        default: return "?";
    }
}

/* B2 commit-5a: print_reg_states port (printer.cpp:107-158). Sorts by
 * reg id, writes "<count>" header followed by space-separated
 * "{prefix}:{name}" entries. */
void phx_hir_print_reg_states(FILE *out, const void *reg_states_array) {
    size_t n = phx_hir_reg_state_array_size(reg_states_array);
    fprintf(out, "<%zu>", n);
    if (n == 0) {
        return;
    }
    fputc(' ', out);
    /* Sort a pointer array by reg id (avoids touching the source array). */
    const void **sorted = (const void **)malloc(n * sizeof(*sorted));
    if (sorted == NULL) {
        return;
    }
    for (size_t i = 0; i < n; i++) {
        sorted[i] = phx_hir_reg_state_array_at(reg_states_array, i);
    }
    qsort(sorted, n, sizeof(*sorted), phx_hir_reg_state_cmp);
    const char *sep = "";
    for (size_t i = 0; i < n; i++) {
        const void *rs = sorted[i];
        const char *prefix = phx_hir_reg_state_prefix(
            phx_hir_reg_state_value_kind(rs),
            phx_hir_reg_state_ref_kind(rs));
        fprintf(out, "%s%s:%s",
                sep,
                prefix,
                phx_hir_register_name(phx_hir_reg_state_reg(rs)));
        sep = " ";
    }
    free(sorted);
}

/* B2 commit-5a: Print(FrameState) port (printer.cpp:864-920).
 * Writes "CurInstrOffset N\n" header, then optional Locals/Cells/Stack/
 * BlockStack sections, each indented + newline-terminated. */
void phx_hir_print_frame_state(FILE *out, PhxHirPrinter *p, const void *state) {
    phx_hir_printer_write_indent(out, p);
    fprintf(out, "CurInstrOffset %d\n", phx_hir_frame_state_cur_instr_offs(state));

    int nlocals = phx_hir_frame_state_nlocals(state);
    if (nlocals > 0) {
        phx_hir_printer_write_indent(out, p);
        fprintf(out, "Locals<%d>", nlocals);
        for (int i = 0; i < nlocals; i++) {
            const void *reg = phx_hir_frame_state_localsplus_at(state, (size_t)i);
            if (reg == NULL) {
                fputs(" <null>", out);
            } else {
                fprintf(out, " %s", phx_hir_register_name(reg));
            }
        }
        fputc('\n', out);
    }

    size_t nlocalsplus = phx_hir_frame_state_localsplus_count(state);
    size_t ncells = (nlocalsplus > (size_t)nlocals) ? nlocalsplus - (size_t)nlocals : 0;
    if (ncells > 0) {
        phx_hir_printer_write_indent(out, p);
        fprintf(out, "Cells<%zu>", ncells);
        for (size_t i = (size_t)nlocals; i < nlocalsplus; i++) {
            const void *reg = phx_hir_frame_state_localsplus_at(state, i);
            if (reg == NULL) {
                fputs(" <null>", out);
            } else {
                fprintf(out, " %s", phx_hir_register_name(reg));
            }
        }
        fputc('\n', out);
    }

    size_t opstack_size = phx_hir_frame_state_stack_count(state);
    if (opstack_size > 0) {
        phx_hir_printer_write_indent(out, p);
        fprintf(out, "Stack<%zu>", opstack_size);
        for (size_t i = 0; i < opstack_size; i++) {
            const void *reg = phx_hir_frame_state_stack_at(state, i);
            fprintf(out, " %s", phx_hir_register_name(reg));
        }
        fputc('\n', out);
    }

    size_t bs_size = phx_hir_frame_state_block_stack_size(state);
    if (bs_size > 0) {
        phx_hir_printer_write_indent(out, p);
        fputs("BlockStack {\n", out);
        phx_hir_printer_indent(p);
        for (size_t i = 0; i < bs_size; i++) {
            phx_hir_printer_write_indent(out, p);
            fprintf(out, "Opcode %d HandlerOff %d StackLevel %d\n",
                    phx_hir_frame_state_block_stack_opcode(state, i),
                    phx_hir_frame_state_block_stack_handler_off(state, i),
                    phx_hir_frame_state_block_stack_stack_level(state, i));
        }
        phx_hir_printer_dedent(p);
        phx_hir_printer_write_indent(out, p);
        fputs("}\n", out);
    }
}

/* W-PRINTER-IMMEDIATES-PORT P-1 (Alex 2026-04-28 default-port directive):
 * Begins migrating format_immediates (printer.cpp:218-803, 168 cases) from
 * the C++ static into a C-side switch. Each ported case writes its own
 * "<...>" wrapping (matches printer.cpp:815-818
 * `if (!immed.empty()) os << "<" << immed << ">"` semantics — empty cases
 * write nothing). Unported opcodes fall through to phx_format_immediates_cpp,
 * the commit-4 bridge into the C++ static.
 *
 * P-1 scope: the 75-opcode "no immediates" cluster (printer.cpp:220-295)
 * — these all return "" in the C++ side, i.e. the format_immediates
 * function emits nothing. Porting them is purely deletion-of-fallback:
 * the C-side switch case body is empty (`return;`), no formatting work.
 * Subsequent commits port the per-class formatting cases.
 */
void phx_format_immediates(FILE *out, const PhxHirPrinter *p, const void *instr) {
    int op = hir_c_opcode(instr);
    switch (op) {
        /* P-1: empty-immediates cluster (no <...> emitted).  Mirror of
         * printer.cpp:220-294 case-fallthrough returning "". */
        case HIR_OP_Assign:
        case HIR_OP_BatchDecref:
        case HIR_OP_BitCast:
        case HIR_OP_BuildString:
        case HIR_OP_BuildTemplate:
        case HIR_OP_CheckErrOccurred:
        case HIR_OP_CheckExc:
        case HIR_OP_CheckNeg:
        case HIR_OP_CheckSequenceBounds:
        case HIR_OP_CIntToCBool:
        case HIR_OP_CopyDictWithoutKeys:
        case HIR_OP_Decref:
        case HIR_OP_DeleteSubscr:
        case HIR_OP_Deopt:
        case HIR_OP_DictMerge:
        case HIR_OP_DictSubscr:
        case HIR_OP_DictUpdate:
        case HIR_OP_EndInlinedFunction:
        case HIR_OP_FormatWithSpec:
        case HIR_OP_GetAIter:
        case HIR_OP_GetANext:
        case HIR_OP_GetIter:
        case HIR_OP_GetLength:
        case HIR_OP_GetTuple:
        case HIR_OP_Guard:
        case HIR_OP_Incref:
        case HIR_OP_InitialYield:
        case HIR_OP_InvokeIterNext:
        case HIR_OP_IsInstance:
        case HIR_OP_IsNegativeAndErrOccurred:
        case HIR_OP_IsTruthy:
        case HIR_OP_ListAppend:
        case HIR_OP_ListExtend:
        case HIR_OP_LoadCellItem:
        case HIR_OP_LoadCurrentFunc:
        case HIR_OP_LoadFrame:
        case HIR_OP_LoadEvalBreaker:
        case HIR_OP_AtQuiescentState:
        case HIR_OP_LoadFieldAddress:
        case HIR_OP_LoadVarObjectSize:
        case HIR_OP_MakeCell:
        case HIR_OP_MakeFunction:
        case HIR_OP_MakeSet:
        case HIR_OP_MakeTupleFromList:
        case HIR_OP_MatchClass:
        case HIR_OP_MatchKeys:
        case HIR_OP_MergeSetUnpack:
        case HIR_OP_PrimitiveBoxBool:
        case HIR_OP_Raise:
        case HIR_OP_RunPeriodicTasks:
        case HIR_OP_Send:
        case HIR_OP_SetCurrentAwaiter:
        case HIR_OP_SetCellItem:
        case HIR_OP_SetDictItem:
        case HIR_OP_SetSetItem:
        case HIR_OP_SetUpdate:
        case HIR_OP_Snapshot:
        case HIR_OP_StealCellItem:
        case HIR_OP_SwapCellItem:
        case HIR_OP_StoreArrayItem:
        case HIR_OP_StoreSubscr:
        case HIR_OP_WaitHandleLoadCoroOrResult:
        case HIR_OP_WaitHandleLoadWaiter:
        case HIR_OP_WaitHandleRelease:
        case HIR_OP_XDecref:
        case HIR_OP_XIncref:
        case HIR_OP_YieldAndYieldFrom:
        case HIR_OP_YieldFrom:
        case HIR_OP_YieldFromHandleStopAsyncIteration:
        case HIR_OP_UnicodeConcat:
        case HIR_OP_UnicodeRepeat:
        case HIR_OP_UnicodeSubscr:
        case HIR_OP_Unreachable:
        case HIR_OP_YieldValue:
            return;

        /* P-2: simple 1-line cases that read a single field via the
         * C-side instr struct or a thin bridge.  Each writes "<...>"
         * with the formatted value. */
        case HIR_OP_BinaryOp: {
            const HirBinaryOp *bop = (const HirBinaryOp *)instr;
            fprintf(out, "<%s>", phx_hir_binary_op_name(bop->op));
            return;
        }
        case HIR_OP_UnaryOp: {
            const HirUnaryOp *uop = (const HirUnaryOp *)instr;
            fprintf(out, "<%s>", phx_hir_unary_op_name(uop->op));
            return;
        }
        case HIR_OP_LongBinaryOp: {
            const HirLongBinaryOp *bop = (const HirLongBinaryOp *)instr;
            fprintf(out, "<%s>", phx_hir_binary_op_name(bop->op));
            return;
        }
        case HIR_OP_LongInPlaceOp: {
            const HirLongInPlaceOp *ip = (const HirLongInPlaceOp *)instr;
            fprintf(out, "<%s>", phx_hir_in_place_op_name(ip->op));
            return;
        }
        case HIR_OP_FloatBinaryOp: {
            const HirFloatBinaryOp *bop = (const HirFloatBinaryOp *)instr;
            fprintf(out, "<%s>", phx_hir_binary_op_name(bop->op));
            return;
        }
        case HIR_OP_Compare: {
            const HirCompare *cmp = (const HirCompare *)instr;
            fprintf(out, "<%s>", phx_hir_compare_op_name(cmp->op));
            return;
        }
        case HIR_OP_FloatCompare: {
            /* Different layout from HirCompare: HIR_INSTR_FIELDS (no
             * deopt prefix) so op field sits at a different offset.
             * testkeeper 13:16:01Z caught this via 4-bench golden-diff. */
            const HirFloatCompare *cmp = (const HirFloatCompare *)instr;
            fprintf(out, "<%s>", phx_hir_compare_op_name(cmp->op));
            return;
        }
        case HIR_OP_LongCompare: {
            const HirLongCompare *cmp = (const HirLongCompare *)instr;
            fprintf(out, "<%s>", phx_hir_compare_op_name(cmp->op));
            return;
        }
        case HIR_OP_UnicodeCompare: {
            const HirUnicodeCompare *cmp = (const HirUnicodeCompare *)instr;
            fprintf(out, "<%s>", phx_hir_compare_op_name(cmp->op));
            return;
        }
        case HIR_OP_BeginInlinedFunction: {
            const char *name = phx_hir_begin_inlined_function_fullname(instr);
            if (name != NULL && name[0] != '\0') {
                fprintf(out, "<%s>", name);
            }
            return;
        }
        case HIR_OP_LoadArrayItem: {
            intptr_t off = phx_hir_load_array_item_offset(instr);
            if (off != 0) {
                fprintf(out, "<Offset[%ld]>", (long)off);
            }
            return;
        }
        case HIR_OP_LoadSplitDictItem: {
            int idx = phx_hir_load_split_dict_item_idx(instr);
            fprintf(out, "<%d>", idx);
            return;
        }
        case HIR_OP_Return: {
            const char *type_str = phx_hir_return_type_or_empty(instr);
            if (type_str != NULL && type_str[0] != '\0') {
                fprintf(out, "<%s>", type_str);
            }
            return;
        }
        case HIR_OP_Branch: {
            int target = phx_hir_branch_target_id(instr);
            fprintf(out, "<%d>", target);
            return;
        }

        /* All other opcodes: bridge fallback to C++ format_immediates
         * (commit-4 phx_format_immediates_cpp). Subsequent P-N commits
         * migrate cases out of this default. */
        default:
            phx_format_immediates_cpp(out, p, instr);
            return;
    }
}
