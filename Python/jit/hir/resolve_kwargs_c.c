/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Pure C ResolveKwargs — resolve keyword arguments to positional order.
 */

#include "cinderx/Jit/hir/hir_c_api.h"
#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/hir/hir_basic_block_c.h"
#include "cinderx/Jit/hir/hir_type_c.h"
#include "cinderx/Common/jit_log_c.h"
#include "Python.h"

#include <string.h>

/* hir_register_type canonical decl in hir_c_api.h (post-W25b). */

/* ---- Pointer array ---- */

typedef struct {
    void **data;
    size_t count, cap;
} RkArr;

static void rk_arr_init(RkArr *a) { memset(a, 0, sizeof(*a)); }
static void rk_arr_destroy(RkArr *a) { PyMem_RawFree(a->data); }

static void rk_arr_push(RkArr *a, void *v) {
    if (a->count >= a->cap) {
        a->cap = a->cap ? a->cap * 2 : 8;
        a->data = (void **)PyMem_RawRealloc(a->data, a->cap * sizeof(void *));
    }
    a->data[a->count++] = v;
}

/* ---- Resolve VectorCall kwargs ---- */

static int resolve_vectorcall_kwargs(void *call) {
    void *target = hir_c_get_operand(call, 0);
    HirType target_hir = hir_register_type(target);
    HirType tfunc_hir = HIR_TYPE_SIMPLE(0x00000000020ULL, HIR_TYPE_LIFETIME_TOP);

    if (!hir_type_has_value_spec(&target_hir, tfunc_hir)) {
        return 0;
    }

    PyFunctionObject *callee = (PyFunctionObject *)hir_type_object_spec(&target_hir);
    PyCodeObject *code = (PyCodeObject *)callee->func_code;

    if (code->co_flags & (CO_VARKEYWORDS | CO_VARARGS)) {
        return 0;
    }

    size_t total_operands = hir_c_num_operands(call);
    if (total_operands < 2) return 0;

    void *kwnames_reg = hir_c_get_operand(call, total_operands - 1);
    HirType kwnames_hir = hir_register_type(kwnames_reg);
    if (!hir_type_has_object_spec(&kwnames_hir)) return 0;

    PyObject *kwnames_obj = hir_type_object_spec(&kwnames_hir);
    if (!PyTuple_Check(kwnames_obj)) return 0;

    Py_ssize_t n_kw = PyTuple_GET_SIZE(kwnames_obj);
    if (n_kw == 0) return 0;

    size_t total_args = total_operands - 2;
    size_t n_pos = total_args - (size_t)n_kw;

    int co_argcount = code->co_argcount;
    int co_kwonly = code->co_kwonlyargcount;
    int total_params = co_argcount + co_kwonly;

    if ((int)total_args > total_params) return 0;

    void **reordered = (void **)PyMem_RawCalloc(total_args, sizeof(void *));

    for (size_t i = 0; i < n_pos; i++) {
        reordered[i] = hir_c_get_operand(call, i + 1);
    }

    PyObject *varnames = PyCode_GetVarnames(code);
    if (!PyTuple_Check(varnames)) {
        PyMem_RawFree(reordered);
        return 0;
    }

    for (Py_ssize_t kw_idx = 0; kw_idx < n_kw; kw_idx++) {
        PyObject *kwname = PyTuple_GET_ITEM(kwnames_obj, kw_idx);
        void *kw_arg = hir_c_get_operand(call, n_pos + kw_idx + 1);

        int found = 0;
        for (int param_idx = 0; param_idx < total_params; param_idx++) {
            PyObject *param_name = PyTuple_GET_ITEM(varnames, param_idx);
            int cmp = PyUnicode_Compare(kwname, param_name);
            if (cmp == 0 && !PyErr_Occurred()) {
                if (reordered[param_idx] != NULL) {
                    PyMem_RawFree(reordered);
                    return 0;
                }
                reordered[param_idx] = kw_arg;
                found = 1;
                break;
            }
            if (PyErr_Occurred()) {
                PyErr_Clear();
                PyMem_RawFree(reordered);
                return 0;
            }
        }
        if (!found) {
            PyMem_RawFree(reordered);
            return 0;
        }
    }

    for (size_t i = 0; i < total_args; i++) {
        if (reordered[i] == NULL) {
            PyMem_RawFree(reordered);
            return 0;
        }
    }

    size_t new_num_operands = total_args + 1;
    uint32_t old_flags = hir_c_call_flags(call);
    uint32_t new_flags = old_flags & ~HIR_CALL_FLAG_KWARGS;

    void *new_call = hir_c_create_vectorcall_reg(new_num_operands,
                                                  hir_c_output(call), new_flags);
    hir_c_set_operand(new_call, 0, target);
    for (size_t i = 0; i < total_args; i++) {
        hir_c_set_operand(new_call, i + 1, reordered[i]);
    }

    hir_c_copy_frame_state(new_call, call);
    hir_c_copy_bytecode_offset(new_call, call);

    hir_instr_replace_with(call, new_call);

    PyMem_RawFree(reordered);
    return 1;
}

/* ---- Resolve CallMethod kwargs ---- */

static int resolve_call_method_kwargs(void *call) {
    void *target = hir_c_get_operand(call, 0);
    HirType target_hir = hir_register_type(target);
    HirType tfunc_hir = HIR_TYPE_SIMPLE(0x00000000020ULL, HIR_TYPE_LIFETIME_TOP);

    if (!hir_type_has_value_spec(&target_hir, tfunc_hir)) {
        return 0;
    }

    PyFunctionObject *callee = (PyFunctionObject *)hir_type_object_spec(&target_hir);
    PyCodeObject *code = (PyCodeObject *)callee->func_code;

    if (code->co_flags & (CO_VARKEYWORDS | CO_VARARGS)) {
        return 0;
    }

    size_t total_operands = hir_c_num_operands(call);
    if (total_operands < 3) return 0;

    void *kwnames_reg = hir_c_get_operand(call, total_operands - 1);
    HirType kwnames_hir = hir_register_type(kwnames_reg);
    if (!hir_type_has_object_spec(&kwnames_hir)) return 0;

    PyObject *kwnames_obj = hir_type_object_spec(&kwnames_hir);
    if (!PyTuple_Check(kwnames_obj)) return 0;

    Py_ssize_t n_kw = PyTuple_GET_SIZE(kwnames_obj);
    if (n_kw == 0) return 0;

    size_t total_args = total_operands - 3;
    size_t n_pos = total_args - (size_t)n_kw;

    int co_argcount = code->co_argcount;
    int co_kwonly = code->co_kwonlyargcount;
    int total_params = co_argcount + co_kwonly;

    if ((int)total_args > total_params) return 0;

    void **reordered = (void **)PyMem_RawCalloc(total_args, sizeof(void *));

    for (size_t i = 0; i < n_pos; i++) {
        reordered[i] = hir_c_get_operand(call, i + 2);
    }

    PyObject *varnames = PyCode_GetVarnames(code);
    if (!PyTuple_Check(varnames)) {
        PyMem_RawFree(reordered);
        return 0;
    }

    for (Py_ssize_t kw_idx = 0; kw_idx < n_kw; kw_idx++) {
        PyObject *kwname = PyTuple_GET_ITEM(kwnames_obj, kw_idx);
        void *kw_arg = hir_c_get_operand(call, n_pos + kw_idx + 2);

        int found = 0;
        for (int param_idx = 0; param_idx < total_params; param_idx++) {
            PyObject *param_name = PyTuple_GET_ITEM(varnames, param_idx);
            int cmp = PyUnicode_Compare(kwname, param_name);
            if (cmp == 0 && !PyErr_Occurred()) {
                if (reordered[param_idx] != NULL) {
                    PyMem_RawFree(reordered);
                    return 0;
                }
                reordered[param_idx] = kw_arg;
                found = 1;
                break;
            }
            if (PyErr_Occurred()) {
                PyErr_Clear();
                PyMem_RawFree(reordered);
                return 0;
            }
        }
        if (!found) {
            PyMem_RawFree(reordered);
            return 0;
        }
    }

    for (size_t i = 0; i < total_args; i++) {
        if (reordered[i] == NULL) {
            PyMem_RawFree(reordered);
            return 0;
        }
    }

    size_t new_num_operands = total_args + 2;
    uint32_t old_flags = hir_c_call_flags(call);
    uint32_t new_flags = old_flags & ~HIR_CALL_FLAG_KWARGS;

    void *self_reg = hir_c_get_operand(call, 1);
    void *new_call = hir_c_create_call_method_reg(new_num_operands,
                                                   hir_c_output(call), new_flags);
    hir_c_set_operand(new_call, 0, target);
    hir_c_set_operand(new_call, 1, self_reg);
    for (size_t i = 0; i < total_args; i++) {
        hir_c_set_operand(new_call, i + 2, reordered[i]);
    }

    hir_c_copy_frame_state(new_call, call);
    hir_c_copy_bytecode_offset(new_call, call);

    hir_instr_replace_with(call, new_call);

    PyMem_RawFree(reordered);
    return 1;
}

/* ---- Public API ---- */

void hir_resolve_kwargs_run(HirFunction func) {
    HirCFG *cfg = (HirCFG *)hir_func_cfg_ptr(func);

    int resolved = 0;

    for (HirBasicBlock *block = hir_cfg_first_block(cfg); block;
         block = hir_cfg_next_block(cfg, block)) {
        RkArr kwargs_instrs;
        rk_arr_init(&kwargs_instrs);

        void *instr = hir_bb_first_instr(block);
        while (instr) {
            int op = hir_c_opcode(instr);
            if (op == HIR_OP_VectorCall || op == HIR_OP_CallMethod) {
                uint32_t flags = hir_c_call_flags(instr);
                if (flags & HIR_CALL_FLAG_KWARGS) {
                    rk_arr_push(&kwargs_instrs, instr);
                }
            }
            instr = hir_bb_next_instr(block, instr);
        }

        for (size_t i = 0; i < kwargs_instrs.count; i++) {
            void *ki = kwargs_instrs.data[i];
            int op = hir_c_opcode(ki);
            if (op == HIR_OP_VectorCall) {
                resolved += resolve_vectorcall_kwargs(ki);
            } else if (op == HIR_OP_CallMethod) {
                resolved += resolve_call_method_kwargs(ki);
            }
        }

        rk_arr_destroy(&kwargs_instrs);
    }
}
