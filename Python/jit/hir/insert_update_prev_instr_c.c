/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C port of InsertUpdatePrevInstr pass.
 */

#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/hir/hir_basic_block_c.h"
#include "cinderx/Jit/hir/instr_effects_c.h"
#include "cinderx/Jit/jit_config_c.h"
#include "Python.h"
#include "pycore_code.h"

#include <limits.h>
#include <string.h>

typedef void *HirFunction;

/* ---- BytecodeIndexToLine: maps bytecode index → line number ---- */

typedef struct {
    int *lines;      /* lines[idx] = line number for bytecode index idx */
    size_t count;    /* number of entries */
} BcIdxToLine;

static BcIdxToLine bc_idx_to_line_create(PyCodeObject *co) {
    BcIdxToLine result = {NULL, 0};
    size_t num_indices = _PyCode_NBYTES(co) / sizeof(_Py_CODEUNIT);
    result.lines = (int *)PyMem_RawCalloc(num_indices, sizeof(int));
    if (!result.lines) return result;
    result.count = num_indices;

    PyCodeAddressRange range;
    _PyCode_InitAddressRange(co, &range);
    size_t idx = 0;
    while (_PyLineTable_NextAddressRange(&range)) {
        if (idx >= num_indices) break;
        for (; idx < (size_t)(range.ar_end / 2) && idx < num_indices; idx++) {
            result.lines[idx] = range.ar_line;
        }
    }
    return result;
}

static void bc_idx_to_line_destroy(BcIdxToLine *tbl) {
    if (tbl->lines) PyMem_RawFree(tbl->lines);
    tbl->lines = NULL;
    tbl->count = 0;
}

static int bc_idx_to_line_lookup(const BcIdxToLine *tbl, int index) {
    if (index < 0) return -1;
    if ((size_t)index >= tbl->count) return -1;
    return tbl->lines[index];
}

/* ---- Code-to-line-table map (small fixed array) ---- */

#define MAX_CODE_OBJECTS 64

typedef struct {
    PyCodeObject *code;
    BcIdxToLine tbl;
} CodeLineEntry;

typedef struct {
    CodeLineEntry entries[MAX_CODE_OBJECTS];
    size_t count;
} CodeLineMap;

static BcIdxToLine *code_line_map_get_or_create(CodeLineMap *map, PyCodeObject *co) {
    for (size_t i = 0; i < map->count; i++) {
        if (map->entries[i].code == co) return &map->entries[i].tbl;
    }
    if (map->count >= MAX_CODE_OBJECTS) return NULL;
    map->entries[map->count].code = co;
    map->entries[map->count].tbl = bc_idx_to_line_create(co);
    return &map->entries[map->count++].tbl;
}

static void code_line_map_destroy(CodeLineMap *map) {
    for (size_t i = 0; i < map->count; i++) {
        bc_idx_to_line_destroy(&map->entries[i].tbl);
    }
    map->count = 0;
}

/* ---- Worklist stack ---- */

#define MAX_WORKLIST 4096

typedef struct {
    void *block;   /* BasicBlock* */
    void *parent;  /* BeginInlinedFunction* or NULL */
} WorkItem;

/* ---- Parent map (BeginInlinedFunction* → parent BeginInlinedFunction*) ---- */

#define MAX_PARENTS 256

typedef struct {
    void *begin;
    void *parent;
} ParentEntry;

typedef struct {
    ParentEntry entries[MAX_PARENTS];
    size_t count;
} ParentMap;

static void parent_map_set(ParentMap *map, void *begin, void *parent) {
    for (size_t i = 0; i < map->count; i++) {
        if (map->entries[i].begin == begin) {
            map->entries[i].parent = parent;
            return;
        }
    }
    if (map->count >= MAX_PARENTS) return;
    map->entries[map->count].begin = begin;
    map->entries[map->count].parent = parent;
    map->count++;
}

static void *parent_map_get(const ParentMap *map, void *begin) {
    for (size_t i = 0; i < map->count; i++) {
        if (map->entries[i].begin == begin) return map->entries[i].parent;
    }
    return NULL;
}

/* ---- Helper: get code object from parent (BeginInlinedFunction or func) ---- */

static PyCodeObject *get_target_code(void *func, void *parent) {
    if (parent == NULL) {
        return (PyCodeObject *)hir_func_code(func);
    }
    HirBeginInlinedFunction *bif = (HirBeginInlinedFunction *)parent;
    PyFunctionObject *pyfunc = (PyFunctionObject *)bif->func;
    return (PyCodeObject *)pyfunc->func_code;
}

/* ---- Main pass ---- */

void hir_insert_update_prev_instr_run(HirFunction func) {
    PyCodeObject *main_code = (PyCodeObject *)hir_func_code(func);
    void *cfg = hir_func_cfg_ptr(func);
    HirCFG *hcfg = (HirCFG *)cfg;
    void *entry = hcfg->entry_block;
    if (!entry) return;

    /* Initialize code→line map with main function's code */
    CodeLineMap clmap;
    memset(&clmap, 0, sizeof(clmap));
    code_line_map_get_or_create(&clmap, main_code);

    /* Worklist (BFS-like traversal) */
    WorkItem *worklist = (WorkItem *)PyMem_RawMalloc(MAX_WORKLIST * sizeof(WorkItem));
    size_t wl_top = 0;

    /* Visited set (by block id) */
    int max_bid = 0;
    for (HirBasicBlock *b = hir_cfg_first_block(hcfg); b; b = hir_cfg_next_block(hcfg, b)) {
        if (b->id > max_bid) max_bid = b->id;
    }
    char *visited = (char *)PyMem_RawCalloc((size_t)(max_bid + 1), 1);

    /* Parent map */
    ParentMap pmap;
    memset(&pmap, 0, sizeof(pmap));

    /* Push entry block */
    worklist[wl_top].block = entry;
    worklist[wl_top].parent = NULL;
    wl_top++;
    visited[((HirBasicBlock *)entry)->id] = 1;

    int frame_mode = jit_get_config()->frame_mode;
    int inited_once = 0;

    while (wl_top > 0) {
        wl_top--;
        void *block = worklist[wl_top].block;
        void *parent = worklist[wl_top].parent;

        int prev_emitted_lno_or_bc = INT_MAX;

        for (void *instr = hir_bb_first_instr(block); instr;
             instr = hir_bb_next_instr(block, instr)) {
            int opcode = hir_c_opcode(instr);
            int32_t bc_off = hir_c_bytecode_offset(instr);
            int bc_idx = bc_off / (int)sizeof(_Py_CODEUNIT);

            /* update_one lambda equivalent */
            #define DO_UPDATE_ONE() do { \
                int update_every_bc = (main_code->co_linetable == NULL || \
                    PyBytes_Size(main_code->co_linetable) == 0); \
                if (update_every_bc) { \
                    if (bc_off != prev_emitted_lno_or_bc) { \
                        void *ui = hir_c_create_update_prev_instr(-1, parent); \
                        hir_c_copy_bytecode_offset(ui, instr); \
                        hir_c_insert_before_pure(ui, instr, block); \
                        prev_emitted_lno_or_bc = bc_off; \
                    } \
                } else { \
                    PyCodeObject *tc = get_target_code(func, parent); \
                    BcIdxToLine *tbl = code_line_map_get_or_create(&clmap, tc); \
                    int cur_line_no = bc_idx_to_line_lookup(tbl, bc_idx); \
                    if (cur_line_no != prev_emitted_lno_or_bc) { \
                        void *ui = hir_c_create_update_prev_instr(cur_line_no, parent); \
                        hir_c_copy_bytecode_offset(ui, instr); \
                        hir_c_insert_before_pure(ui, instr, block); \
                        prev_emitted_lno_or_bc = cur_line_no; \
                    } \
                } \
            } while (0)

            if (opcode == HIR_OP_BeginInlinedFunction) {
                DO_UPDATE_ONE();

                HirBeginInlinedFunction *bif = (HirBeginInlinedFunction *)instr;
                PyFunctionObject *pyfunc = (PyFunctionObject *)bif->func;
                PyCodeObject *code = (PyCodeObject *)pyfunc->func_code;
                code_line_map_get_or_create(&clmap, code);

                parent_map_set(&pmap, instr, parent);
                parent = instr;
                if (frame_mode == JIT_FRAME_LIGHTWEIGHT) {
                    inited_once = 0;
                }
            } else if (opcode == HIR_OP_EndInlinedFunction) {
                HirEndInlinedFunction *eif = (HirEndInlinedFunction *)instr;
                parent = parent_map_get(&pmap, eif->begin);
            }

            if (frame_mode == JIT_FRAME_LIGHTWEIGHT) {
                if (!inited_once && opcode == HIR_OP_LoadEvalBreaker) {
                    PyCodeObject *tc = get_target_code(func, parent);
                    BcIdxToLine *tbl = code_line_map_get_or_create(&clmap, tc);
                    int line_no = bc_idx_to_line_lookup(tbl, tc->_co_firsttraceable);
                    void *ui = hir_c_create_update_prev_instr(line_no, parent);
                    hir_c_set_bytecode_offset(ui,
                        tc->_co_firsttraceable * (int)sizeof(_Py_CODEUNIT));
                    hir_c_insert_before_pure(ui, instr, block);
                    inited_once = 1;
                }
            } else if (hir_has_arbitrary_execution(instr)) {
                DO_UPDATE_ONE();
            }

            #undef DO_UPDATE_ONE
        }

        /* Push successors */
        void *term = hir_bb_get_terminator(block);
        if (term) {
            size_t n_edges = hir_c_num_edges(term);
            for (size_t i = 0; i < n_edges; i++) {
                void *succ = hir_c_successor(term, i);
                if (succ && !visited[((HirBasicBlock *)succ)->id]) {
                    visited[((HirBasicBlock *)succ)->id] = 1;
                    if (wl_top < MAX_WORKLIST) {
                        worklist[wl_top].block = succ;
                        worklist[wl_top].parent = parent;
                        wl_top++;
                    }
                }
            }
        }
    }

    code_line_map_destroy(&clmap);
    PyMem_RawFree(visited);
    PyMem_RawFree(worklist);
}
