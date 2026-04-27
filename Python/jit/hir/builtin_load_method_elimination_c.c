/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Pure-C body for the BuiltinLoadMethodElimination pass.
 *
 * Walks every CallMethod, identifies the LoadMethod /
 * GetSecondOutput / CallMethod triple that vanilla CPython
 * specialisation produces, and — when the receiver type's MRO yields
 * a directly-invokable method object — rewrites the triple to
 * UseType + LoadConst (expanded from LoadMethod), Assign (replaces
 * GetSecondOutput) and a Static-flagged VectorCall (replaces
 * CallMethod).
 *
 * Cat-A MRO / type-lookup helpers live in blme_helpers_c.{c,h}.
 */

#include "cinderx/Jit/hir/builtin_load_method_elimination_c.h"

#include "cinderx/Common/jit_log_c.h"
#include "cinderx/Jit/hir/blme_helpers_c.h"
#include "cinderx/Jit/hir/hir_basic_block_c.h"
#include "cinderx/Jit/hir/hir_c_api.h"
#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/hir/hir_opcode_c.h"
#include "cinderx/Jit/hir/hir_type_c.h"
#include "cinderx/Jit/threaded_compile_c.h"

#include <stdint.h>

/* hir_reflow_types_c is the C-side type-flow re-trigger. Type-changing
 * rewrites (LoadConst, Assign) propagate freshly-discovered types
 * through the rest of the function so the next outer-loop iteration
 * may unlock further eliminations. Declared extern here per the
 * established C-port pattern (see simplify_c.c L2665, ssa.h L17). */
extern void hir_reflow_types_c(void *func, void *start_block);

/* Defensive iteration cap on the outer fixed-point loop. Per
 * theologian risk flag #8 (HIGH): an incorrect 'changed' flag
 * propagation would spin the loop forever. The cap turns that into a
 * JIT_DCHECK in pydebug + a benign early-out in release.
 *
 * Rationale for 100: each iteration eliminates at least one
 * LoadMethod/CallMethod triple if it returns true. Real workloads do
 * not have 100 nested-elidable LoadMethod chains in a single
 * function. */
#define PHX_BLME_MAX_ITERATIONS 100

/* ---------- PhxMethodInvoke + linear-scan array ---------- */

typedef struct {
    void *load_method;       /* LoadMethodBase* (key) */
    void *get_instance;      /* GetSecondOutput* */
    void *call_method;       /* CallMethod* */
    int   collided;          /* 1 if a 2nd CallMethod for the same LoadMethod was seen */
} PhxMethodInvoke;

typedef struct {
    PhxMethodInvoke *data;
    size_t count;
    size_t capacity;
} PhxBlmeArray;

static void phx_blme_array_destroy(PhxBlmeArray *arr) {
    PyMem_RawFree(arr->data);
    arr->data = NULL;
    arr->count = arr->capacity = 0;
}

static int phx_blme_array_grow(PhxBlmeArray *arr) {
    size_t new_cap = arr->capacity ? arr->capacity * 2 : 16;
    PhxMethodInvoke *new_data = (PhxMethodInvoke *)PyMem_RawRealloc(
        arr->data, new_cap * sizeof(PhxMethodInvoke));
    if (!new_data) return 0;
    arr->data = new_data;
    arr->capacity = new_cap;
    return 1;
}

/* Linear-scan find by load_method pointer. Returns index or arr->count. */
static size_t phx_blme_array_find(const PhxBlmeArray *arr, const void *load_method) {
    for (size_t i = 0; i < arr->count; i++) {
        if (arr->data[i].load_method == load_method) return i;
    }
    return arr->count;
}

/* Insert or, if a row for load_method already exists, mark it collided
 * (two CallMethods for the same LoadMethod cannot be safely rewritten;
 * see invariant #2 in theologian pre-audit / TASK T138839090). */
static int phx_blme_array_insert(PhxBlmeArray *arr,
                                 void *load_method,
                                 void *get_instance,
                                 void *call_method) {
    size_t idx = phx_blme_array_find(arr, load_method);
    if (idx < arr->count) {
        arr->data[idx].collided = 1;
        return 1;
    }
    if (arr->count == arr->capacity && !phx_blme_array_grow(arr)) {
        return 0;
    }
    arr->data[arr->count++] = (PhxMethodInvoke){
        .load_method = load_method,
        .get_instance = get_instance,
        .call_method = call_method,
        .collided = 0,
    };
    return 1;
}

/* ---------- Single-triple rewrite ---------- */

/* Returns 1 if the LoadMethod/GetSecondOutput/CallMethod triple was
 * eliminated, 0 otherwise. Mirrors C++ tryEliminateLoadMethod
 * (builtin_load_method_elimination.cpp:46-124). */
static int phx_blme_try_eliminate_load_method(void *func,
                                              const PhxMethodInvoke *invoke) {
    JIT_COMPILE_GUARD();

    void *fs = hir_c_get_frame_state(invoke->load_method);
    /* LoadMethod / LoadMethodCached / LoadModuleMethodCached all use
     * HIR_DEOPT_NAMEIDX_FIELDS — name_idx lives at the same offset in
     * each, so a single HirLoadMethod cast is correct for the whole
     * isLoadMethodBase set. */
    int32_t name_idx = ((HirLoadMethod *)invoke->load_method)->name_idx;
    PyObject *name = hir_frame_state_get_name(fs, name_idx);
    JIT_DCHECK_C(name != NULL, "name must not be null");

    void *receiver = hir_c_get_operand(invoke->load_method, 0);
    HirType receiver_type = hir_reg_type(receiver);

    PyObject *method_obj = phx_get_method_object_from_type(receiver_type, name);

    /* Mirror the .cpp's DCHECK on TBottom-when-no-runtime-type:
     * receiver_type with hasTypeExactSpec but a NULL runtime PyType
     * MUST be TBottom (anything else is a type-system bug). The C
     * helper returned NULL early in that branch so we test the same
     * predicate on receiver_type here in the caller. */
    if (method_obj == NULL) {
        if (hir_type_has_type_exact_spec(&receiver_type) &&
            hir_type_runtime_py_type(&receiver_type) == NULL) {
            HirType tbottom = HIR_TYPE_BOTTOM;
            JIT_DCHECK_C(hir_type_equal(&receiver_type, &tbottom),
                       "Type expected to have PyTypeObject*");
        }
        /* No such method. Let the LoadMethod fail at runtime;
         * _PyType_Lookup does not raise an exception. */
        return 0;
    }

    /* PyStaticMethod: only used by bytearray/bytes/str.maketrans;
     * not worth optimising (matches C++ comment line 78). */
    if (Py_TYPE(method_obj) == &PyStaticMethod_Type) {
        return 0;
    }

    void *method_reg = hir_c_output(invoke->load_method);
    /* Refcount: addReference keeps method_obj live for the function's
     * lifetime; without it the LoadConst would dangle on type unload
     * (theologian invariant #6). */
    HirType const_type = hir_type_from_object(
        hir_func_add_reference(func, method_obj));
    void *load_const = hir_c_create_load_const(method_reg, const_type);

    void *call_static = hir_c_create_vectorcall_fs_reg(
        hir_c_num_operands(invoke->call_method),
        hir_c_output(invoke->call_method),
        hir_c_call_flags(invoke->call_method) | HIR_CALL_FLAG_STATIC,
        hir_c_get_frame_state(invoke->call_method));
    hir_c_set_operand(call_static, 0, method_reg);

    if (Py_TYPE(method_obj) == &PyClassMethodDescr_Type) {
        /* Pass the type as the first argument (e.g. dict.fromkeys). */
        void *type_reg = hir_func_alloc_register(func);
        PyObject *type_obj = (PyObject *)hir_type_runtime_py_type(&receiver_type);
        HirType type_const_type = hir_type_from_object(
            hir_func_add_reference(func, type_obj));
        void *load_type = hir_c_create_load_const(type_reg, type_const_type);
        hir_c_set_bytecode_offset(
            load_type, hir_c_bytecode_offset(invoke->load_method));
        hir_c_insert_before(load_type, invoke->call_method);
        hir_c_set_operand(call_static, 1, type_reg);
    } else {
        JIT_DCHECK_C(
            Py_TYPE(method_obj) == &PyMethodDescr_Type ||
                Py_TYPE(method_obj) == &PyWrapperDescr_Type ||
                Py_TYPE(method_obj) == &PyFunction_Type,
            "unexpected type");
        /* Pass the instance as the first argument (e.g. str.join,
         * str.__mod__). */
        hir_c_set_operand(call_static, 1, receiver);
    }

    /* Operand layout: [0]=method, [1]=type-or-self, [2..]=original
     * positional args (theologian invariant #9). */
    size_t n_ops = hir_c_num_operands(invoke->call_method);
    for (size_t i = 2; i < n_ops; i++) {
        hir_c_set_operand(call_static, i, hir_c_get_operand(invoke->call_method, i));
    }

    /* LoadMethod expands into [UseType(receiver, unspecialized),
     * LoadConst(method_reg, method_obj-typed)]. */
    HirType unspec = hir_type_unspecialized(&receiver_type);
    void *use_type = hir_c_create_use_type(receiver, unspec);
    void *expansion[2] = { use_type, load_const };
    hir_instr_expand_into(invoke->load_method, expansion, 2);

    /* GetSecondOutput becomes Assign(get_instance->output, receiver). */
    void *assign = hir_c_create_assign(hir_c_output(invoke->get_instance), receiver);
    hir_c_replace_with(invoke->get_instance, assign);

    /* CallMethod becomes the prepared CallStatic. */
    hir_c_replace_with(invoke->call_method, call_static);

    hir_c_destroy_instr(invoke->load_method);
    hir_c_destroy_instr(invoke->get_instance);
    hir_c_destroy_instr(invoke->call_method);
    return 1;
}

/* ---------- Outer fixed-point Run() loop ---------- */

void hir_builtin_load_method_elimination_run(HirFunction func) {
    HirCFG *cfg = (HirCFG *)hir_func_cfg_ptr((void *)func);

    int changed = 1;
    int iter = 0;
    while (changed && iter < PHX_BLME_MAX_ITERATIONS) {
        changed = 0;
        iter++;

        PhxBlmeArray invokes = {NULL, 0, 0};

        /* Walk every block, every instruction; collect candidate triples. */
        for (HirBasicBlock *block = hir_cfg_first_block(cfg);
             block != NULL;
             block = hir_cfg_next_block(cfg, block)) {
            for (void *instr = hir_bb_first_instr(block);
                 instr != NULL;
                 instr = hir_bb_next_instr(block, instr)) {

                if (!hir_c_is_call_method(instr)) continue;

                /* func() is operand[0]; self() is operand[1] for
                 * CallMethod (matches HIR class header layout). */
                void *func_reg = hir_c_get_operand(instr, 0);
                void *func_instr = hir_reg_instr(func_reg);
                if (hir_c_is_load_method_super(func_instr)) continue;

                if (!hir_c_is_load_method_base(func_instr)) {
                    /* {FillTypeMethodCache | LoadTypeMethodCacheEntryValue}
                     * + CallMethod is the type-method-on-instance path
                     * (e.g. dict.fromkeys(...)). Doesn't need the
                     * LoadMethod/CallMethod pairing invariant and isn't
                     * eliminable by this pass. */
                    continue;
                }

                void *self_reg = hir_c_get_operand(instr, 1);
                void *self_instr = hir_reg_instr(self_reg);
                JIT_DCHECK_C(
                    hir_c_is_get_second_output(self_instr),
                    "GetSecondOutput/CallMethod should be paired");

                if (!phx_blme_array_insert(&invokes, func_instr, self_instr, instr)) {
                    /* Allocation failure — bail out of the entire pass for
                     * this iteration (better than partial mutation). */
                    phx_blme_array_destroy(&invokes);
                    return;
                }
            }
        }

        /* Attempt elimination on every uncollided triple. Skipped
         * collided rows so 1:LoadMethod-multi:CallMethod patterns
         * remain untouched (theologian invariant #2 / TASK
         * T138839090). */
        for (size_t i = 0; i < invokes.count; i++) {
            if (invokes.data[i].collided) continue;
            if (phx_blme_try_eliminate_load_method(func, &invokes.data[i])) {
                changed = 1;
            }
        }

        phx_blme_array_destroy(&invokes);

        /* Type-flow re-trigger: LoadConst + Assign rewrites change
         * the types of method_reg / get_instance->output;
         * propagate so the next iteration sees the fresh types
         * (theologian invariant #8). */
        hir_reflow_types_c((void *)func, cfg->entry_block);
    }

    JIT_DCHECK_C(
        iter < PHX_BLME_MAX_ITERATIONS,
        "BuiltinLoadMethodElimination outer loop hit iteration cap");
}
