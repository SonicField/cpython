/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Pure C simplify handlers — incremental port of simplify.cpp.
 */

#include "cinderx/Jit/hir/simplify_c.h"
#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/hir/hir_type_c.h"
#include "cinderx/Common/py-portability.h"
#include "Python.h"
#include "pycore_global_strings.h"
#include "cinderx/Jit/jit_config_c.h"
#include "pycore_long.h"
#include "structmember.h"

/* Forward declarations (avoid hir_c_api.h typedef conflicts) */
extern HirType hir_register_type(void *reg);
extern void *hir_func_alloc_register(void *func);
extern void hir_c_insert_before(void *new_instr, void *before);
extern HirType hir_output_type(void *instr);
extern void *hir_bb_append_instr(void *bb, void *instr);
extern void *hir_bb_first_instr(const void *bb);
extern void *hir_cfg_alloc_block(void *func);
extern void *hir_cfg_split_after(void *func, void *instr);
extern void *hir_phi_create_2way(void *func, void *bb1, void *reg1,
                                  void *bb2, void *reg2);
extern void *hir_c_create_branch_cpp(void *target_block);
extern void *hir_c_create_cond_branch_cpp(void *cond_reg,
                                           void *true_block,
                                           void *false_block);
extern int hir_func_env_allocate_type_method_cache(void *func);
extern int hir_func_env_allocate_type_attr_cache(void *func);
extern void hir_c_set_suppress_exc_deopt(void *instr, int val);
extern void *hir_c_create_fill_type_method_cache(void *func,
    void *receiver, int name_idx, int cache_id, void *fs);
extern void *hir_c_create_load_method_cached(void *func,
    void *receiver, int name_idx, void *fs);
extern void *hir_c_create_load_module_method_cached(void *func,
    void *receiver, int name_idx, void *fs);

/* ---- SimplifyEnv: C equivalent of the C++ Env struct ---- */

void *simplify_env_emit(SimplifyEnv *env, void *new_instr) {
    env->optimized = 1;
    hir_c_set_bytecode_offset(new_instr, env->bc_off);
    if (env->cursor_instr) {
        hir_c_insert_before(new_instr, env->cursor_instr);
    } else {
        hir_bb_append_instr(env->block, new_instr);
    }
    void *out = hir_c_output(new_instr);
    if (out) {
        HirType out_type = hir_output_type(new_instr);
        hir_reg_set_type(out, out_type);
    }
    return out;
}

void *simplify_env_emit_load_const(SimplifyEnv *env, HirType type) {
    void *reg = hir_func_alloc_register(env->func);
    void *instr = hir_c_create_load_const(reg, type);
    return simplify_env_emit(env, instr);
}

void *simplify_env_emit_use_type(SimplifyEnv *env, void *val, HirType type) {
    void *instr = hir_c_create_use_type(val, type);
    return simplify_env_emit(env, instr);
}

/* ---- simplifyCheck ----
 * CheckVar, CheckExc, CheckField all check input for null.
 * If input is known to be an Object (non-null), the check is redundant. */
void *simplify_check_c(const void *instr) {
    void *operand = hir_c_get_operand(instr, 0);
    HirType reg_type = hir_register_type(operand);
    HirType t_object = HIR_TYPE_OBJECT;
    if (hir_type_is_subtype(reg_type, t_object)) {
        return operand;
    }
    return NULL;
}

void *simplify_env_emit_primitive_unbox(SimplifyEnv *env, void *src, HirType type) {
    void *reg = hir_func_alloc_register(env->func);
    void *instr = hir_c_create_primitive_unbox(reg, src, type);
    return simplify_env_emit(env, instr);
}

void *simplify_env_emit_primitive_unary_op(SimplifyEnv *env, int32_t op, void *src) {
    void *reg = hir_func_alloc_register(env->func);
    void *instr = hir_c_create_primitive_unary_op(reg, op, src);
    return simplify_env_emit(env, instr);
}

/* ---- simplifyCast ----
 * If input already has the cast target type, Cast is redundant. */
void *simplify_cast_c(const void *instr) {
    void *input = hir_c_get_operand(instr, 0);
    const HirCast *cast = (const HirCast *)instr;
    HirType type = hir_type_from_pytype((PyTypeObject *)cast->pytype, cast->exact);
    if (cast->optional) {
        HirType t_none = HIR_TYPE_NONETYPE;
        type = hir_type_union(type, t_none);
    }
    HirType input_type = hir_register_type(input);
    if (hir_type_is_subtype(input_type, type)) {
        return input;
    }
    return NULL;
}

/* ---- simplifyRefineType ----
 * If input already has the target type, RefineType is redundant. */
void *simplify_refine_type_c(const void *instr) {
    void *input = hir_c_get_operand(instr, 0);
    HirType input_type = hir_register_type(input);
    HirType target = ((const HirRefineType *)instr)->type;
    if (hir_type_is_subtype(input_type, target)) {
        return input;
    }
    return NULL;
}

/* ---- simplifyGuardType ----
 * If input already has the guarded type, redundant.
 * If target is NoneType, convert to GuardIs(Py_None). */
void *simplify_guard_type_c(SimplifyEnv *env, const void *instr) {
    void *input = hir_c_get_operand(instr, 0);
    HirType input_type = hir_register_type(input);
    HirType target = hir_c_guard_type_target(instr);
    if (hir_type_is_subtype(input_type, target)) {
        return input;
    }
    HirType t_none = HIR_TYPE_NONETYPE;
    if (hir_type_equal(&target, &t_none)) {
        extern void *hir_c_create_guard_is(void *func, void *target_obj, void *src);
        void *guard = hir_c_create_guard_is(env->func, Py_None, input);
        return simplify_env_emit(env, guard);
    }
    return NULL;
}

/* ---- Additional SimplifyEnv emit helpers ---- */

void *simplify_env_emit_primitive_compare(SimplifyEnv *env, int32_t op,
                                          void *left, void *right) {
    void *reg = hir_func_alloc_register(env->func);
    extern void *hir_c_create_primitive_compare(void *dst, int32_t op,
                                                 void *left, void *right);
    void *instr = hir_c_create_primitive_compare(reg, op, left, right);
    return simplify_env_emit(env, instr);
}

void *simplify_env_emit_float_compare(SimplifyEnv *env, int32_t op,
                                       void *left, void *right) {
    void *reg = hir_func_alloc_register(env->func);
    extern void *hir_c_create_float_compare(void *dst, int32_t op,
                                             void *left, void *right);
    void *instr = hir_c_create_float_compare(reg, op, left, right);
    return simplify_env_emit(env, instr);
}

void *simplify_env_emit_long_compare(SimplifyEnv *env, int32_t op,
                                      void *left, void *right) {
    void *reg = hir_func_alloc_register(env->func);
    extern void *hir_c_create_long_compare(void *dst, int32_t op,
                                            void *left, void *right);
    void *instr = hir_c_create_long_compare(reg, op, left, right);
    return simplify_env_emit(env, instr);
}

void *simplify_env_emit_primitive_box_bool(SimplifyEnv *env, void *src) {
    void *reg = hir_func_alloc_register(env->func);
    extern void *hir_c_create_primitive_box_bool_reg(void *dst, void *src);
    void *instr = hir_c_create_primitive_box_bool_reg(reg, src);
    return simplify_env_emit(env, instr);
}

/* ---- simplifyUnaryOp ----
 * If not(bool) → PrimitiveUnbox + NotInt + PrimitiveBoxBool */
#define HIR_UNARY_NOT 0
#define HIR_PRIM_UNARY_NOT_INT 2
void *simplify_unary_op_c(SimplifyEnv *env, const void *instr) {
    void *operand = hir_c_get_operand(instr, 0);
    int32_t op = hir_c_unary_op_kind(instr);
    HirType t_bool = HIR_TYPE_BOOL;
    if (op == HIR_UNARY_NOT && hir_type_is_subtype(hir_register_type(operand), t_bool)) {
        simplify_env_emit_use_type(env, operand, t_bool);
        HirType t_cbool = HIR_TYPE_CBOOL;
        void *unboxed = simplify_env_emit_primitive_unbox(env, operand, t_cbool);
        void *negated = simplify_env_emit_primitive_unary_op(env, HIR_PRIM_UNARY_NOT_INT, unboxed);
        return simplify_env_emit_primitive_box_bool(env, negated);
    }
    return NULL;
}

/* Returns the INSTRUCTION (not output reg) so caller can set operands */
void *simplify_env_emit_call_static_instr(SimplifyEnv *env, size_t n_operands,
                                           void *addr, HirType ret_type) {
    void *reg = hir_func_alloc_register(env->func);
    extern void *hir_c_create_call_static_reg(size_t n, void *dst, void *addr, HirType ret);
    void *instr = hir_c_create_call_static_reg(n_operands, reg, addr, ret_type);
    simplify_env_emit(env, instr);
    return instr;
}

void *simplify_env_emit_load_field(SimplifyEnv *env, void *receiver,
                                    const char *name, intptr_t offset,
                                    HirType type, int borrowed) {
    extern void *hir_c_create_load_field(void *func, void *recv, const char *name,
                                          intptr_t offset, HirType type, int borrowed);
    void *instr = hir_c_create_load_field(env->func, receiver, name, offset, type, borrowed);
    return simplify_env_emit(env, instr);
}

void *simplify_env_emit_cint_to_cbool(SimplifyEnv *env, void *src) {
    void *reg = hir_func_alloc_register(env->func);
    extern void *hir_c_create_cint_to_cbool(void *dst, void *src);
    void *instr = hir_c_create_cint_to_cbool(reg, src);
    return simplify_env_emit(env, instr);
}

/* ---- emitGetLengthInt64 C helper ----
 * Returns ob_size/ma_used/used/length field as CInt64, or NULL. */
static void *emit_get_length_int64_c(SimplifyEnv *env, void *obj) {
    HirType obj_type = hir_register_type(obj);
    HirType t_tuple_exact = HIR_TYPE_TUPLEEXACT;
    HirType t_cint64 = HIR_TYPE_CINT64;

#ifndef Py_GIL_DISABLED
    HirType t_list_exact = HIR_TYPE_LISTEXACT;
    if (hir_type_is_subtype(obj_type, t_list_exact) ||
        hir_type_is_subtype(obj_type, t_tuple_exact)) {
#else
    if (hir_type_is_subtype(obj_type, t_tuple_exact)) {
#endif
        HirType unspec = hir_type_unspecialized(&obj_type);
        simplify_env_emit_use_type(env, obj, unspec);
        return simplify_env_emit_load_field(env, obj, "ob_size",
            (intptr_t)offsetof(PyVarObject, ob_size), t_cint64, 0);
    }

    HirType t_unicode_exact = HIR_TYPE_UNICODEEXACT;
#ifndef Py_GIL_DISABLED
    HirType t_dict_exact = HIR_TYPE_DICTEXACT;
    HirType t_set_exact = HIR_TYPE_SETEXACT;
    if (hir_type_is_subtype(obj_type, t_dict_exact)) {
        HirType unspec = hir_type_unspecialized(&obj_type);
        simplify_env_emit_use_type(env, obj, unspec);
        return simplify_env_emit_load_field(env, obj, "ma_used",
            (intptr_t)offsetof(PyDictObject, ma_used), t_cint64, 0);
    }
    if (hir_type_is_subtype(obj_type, t_set_exact)) {
        HirType unspec = hir_type_unspecialized(&obj_type);
        simplify_env_emit_use_type(env, obj, unspec);
        return simplify_env_emit_load_field(env, obj, "used",
            (intptr_t)offsetof(PySetObject, used), t_cint64, 0);
    }
#endif
    if (hir_type_is_subtype(obj_type, t_unicode_exact)) {
        HirType unspec = hir_type_unspecialized(&obj_type);
        simplify_env_emit_use_type(env, obj, unspec);
        return simplify_env_emit_load_field(env, obj, "length",
            (intptr_t)offsetof(PyASCIIObject, length), t_cint64, 0);
    }
    return NULL;
}

void *simplify_env_emit_check_neg(SimplifyEnv *env, void *src, void *frame_state) {
    extern void *hir_c_create_check_neg(void *func, void *src, void *fs);
    void *instr = hir_c_create_check_neg(env->func, src, frame_state);
    return simplify_env_emit(env, instr);
}

/* ---- isBuiltin C helper ----
 * Checks if a register holding a callable is a known builtin with given name. */
static int is_builtin_c(void *callable_reg, const char *name) {
    HirType callable_type = hir_register_type(callable_reg);
    if (!hir_type_has_object_spec(&callable_type)) return 0;

    PyObject *callable_obj = hir_type_object_spec(&callable_type);
    PyMethodDef *meth = NULL;

    if (Py_TYPE(callable_obj) == &PyCFunction_Type) {
        meth = ((PyCFunctionObject *)callable_obj)->m_ml;
    } else if (Py_TYPE(callable_obj) == &PyMethodDescr_Type) {
        meth = ((PyMethodDescrObject *)callable_obj)->d_method;
    }

    if (meth == NULL) return 0;

    extern const char *jit_builtins_find(void *method_def);
    const char *found = jit_builtins_find(meth);
    if (found == NULL) return 0;
    return strcmp(found, name) == 0;
}

/* ---- Long slot method lookup ---- */
static binaryfunc long_slot_method(int32_t op) {
    PyNumberMethods *nb = PyLong_Type.tp_as_number;
    switch (op) {
        case HIR_BOP_Add: return nb->nb_add;
        case HIR_BOP_And: return nb->nb_and;
        case HIR_BOP_FloorDivide: return nb->nb_floor_divide;
        case HIR_BOP_LShift: return nb->nb_lshift;
        case HIR_BOP_Modulo: return nb->nb_remainder;
        case HIR_BOP_Multiply: return nb->nb_multiply;
        case HIR_BOP_Or: return nb->nb_or;
        case HIR_BOP_RShift: return nb->nb_rshift;
        case HIR_BOP_Subtract: return nb->nb_subtract;
        case HIR_BOP_TrueDivide: return nb->nb_true_divide;
        case HIR_BOP_Xor: return nb->nb_xor;
        default: return NULL;
    }
}

/* ---- simplifyStoreAttr ---- */
void *simplify_store_attr_c(SimplifyEnv *env, const void *instr) {
    if (!jit_get_config()->attr_caches) return NULL;
    void *receiver = hir_c_get_operand(instr, 0);
    void *value = hir_c_get_operand(instr, 1);
    int32_t name_idx = ((const HirStoreAttr *)instr)->name_idx;
    void *fs = hir_c_get_frame_state(instr);
    extern void *hir_c_create_store_attr_cached(void *func, void *recv, void *val,
                                                 int32_t idx, void *fs);
    void *cached = hir_c_create_store_attr_cached(env->func, receiver, value, name_idx, fs);
    return simplify_env_emit(env, cached);
}

/* ---- simplifyGetIter ---- */
void *simplify_get_iter_c(SimplifyEnv *env, const void *instr) {
    extern PyTypeObject *jit_g_range_iterator_type;
    if (jit_g_range_iterator_type == NULL) return NULL;
    void *input = hir_c_get_operand(instr, 0);
    HirType input_type = hir_register_type(input);
    HirType t_range = hir_type_from_pytype(&PyRange_Type, 1);
    if (!hir_type_is_subtype(input_type, t_range)) return NULL;
    void *output = hir_c_output(instr);
    if (output == NULL) return NULL;
    HirType iter_type = hir_type_from_pytype(jit_g_range_iterator_type, 1);
    simplify_env_emit_use_type(env, output, iter_type);
    return NULL;
}

/* ---- simplifyInvokeIterNext ---- */
void *simplify_invoke_iter_next_c(SimplifyEnv *env, const void *instr) {
    extern PyTypeObject *jit_g_range_iterator_type;
    extern PyTypeObject *jit_g_list_iterator_type;
    extern PyTypeObject *jit_g_tuple_iterator_type;
    void *iterator = hir_c_get_operand(instr, 0);
    HirType iter_type = hir_register_type(iterator);
    PyTypeObject *iter_pytype = hir_type_runtime_py_type(&iter_type);
    if (iter_pytype == NULL) return NULL;
    if (!((jit_g_range_iterator_type != NULL && iter_pytype == jit_g_range_iterator_type) ||
          (jit_g_list_iterator_type != NULL && iter_pytype == jit_g_list_iterator_type) ||
          (jit_g_tuple_iterator_type != NULL && iter_pytype == jit_g_tuple_iterator_type)))
        return NULL;
    extern void *jit_rt_invoke_iter_next_addr(void);
    HirType t_object = HIR_TYPE_OBJECT;
    void *call = simplify_env_emit_call_static_instr(env, 1,
        jit_rt_invoke_iter_next_addr(), t_object);
    hir_c_set_operand(call, 0, iterator);
    return hir_c_output(call);
}

/* Forward declarations for helpers defined later */
static binaryfunc float_slot_method(int32_t op);

/* ---- InPlaceOp → BinaryOp conversion ---- */
static int32_t inplace_to_binary(int32_t iop) {
    switch (iop) {
        case HIR_IOP_Add: return HIR_BOP_Add;
        case HIR_IOP_Subtract: return HIR_BOP_Subtract;
        case HIR_IOP_Multiply: return HIR_BOP_Multiply;
        case HIR_IOP_TrueDivide: return HIR_BOP_TrueDivide;
        case HIR_IOP_FloorDivide: return HIR_BOP_FloorDivide;
        case HIR_IOP_Modulo: return HIR_BOP_Modulo;
        case HIR_IOP_Power: return HIR_BOP_Power;
        default: return -1;
    }
}

void *simplify_env_emit_long_in_place_op(SimplifyEnv *env, int32_t op,
                                          void *left, void *right, void *fs) {
    extern void *hir_c_create_long_in_place_op(void *func, int32_t op,
                                                void *left, void *right, void *fs);
    void *instr = hir_c_create_long_in_place_op(env->func, op, left, right, fs);
    return simplify_env_emit(env, instr);
}

void *simplify_env_emit_float_binary_op_deopt(SimplifyEnv *env, int32_t op,
                                               void *left, void *right, void *fs) {
    extern void *hir_c_create_float_binary_op(void *func, int32_t op,
                                               void *left, void *right, void *fs);
    void *instr = hir_c_create_float_binary_op(env->func, op, left, right, fs);
    return simplify_env_emit(env, instr);
}

void *simplify_env_emit_guard_type_deopt(SimplifyEnv *env, HirType target,
                                          void *src, void *fs) {
    extern void *hir_c_create_guard_type(void *func, HirType target,
                                          void *src, void *fs);
    void *instr = hir_c_create_guard_type(env->func, target, src, fs);
    return simplify_env_emit(env, instr);
}

/* ---- simplifyInPlaceOp ---- */
void *simplify_in_place_op_c(SimplifyEnv *env, const void *instr) {
    void *lhs = hir_c_get_operand(instr, 0);
    void *rhs = hir_c_get_operand(instr, 1);
    HirType lhs_type = hir_register_type(lhs);
    HirType rhs_type = hir_register_type(rhs);
    int32_t op = hir_c_binary_op_kind(instr);
    void *fs = hir_c_get_frame_state(instr);
    HirType t_long_exact = HIR_TYPE_LONGEXACT;
    HirType t_float_exact = HIR_TYPE_FLOATEXACT;

    /* Path 1: Long in-place */
    if (hir_type_is_subtype(lhs_type, t_long_exact) &&
        hir_type_is_subtype(rhs_type, t_long_exact) &&
        op != HIR_IOP_MatrixMultiply) {
        simplify_env_emit_use_type(env, lhs, t_long_exact);
        simplify_env_emit_use_type(env, rhs, t_long_exact);
        return simplify_env_emit_long_in_place_op(env, op, lhs, rhs, fs);
    }

    /* Path 2: Float speculation (LHS not Float, RHS is Float) */
    if (!hir_type_is_subtype(lhs_type, t_float_exact) &&
        hir_type_is_subtype(rhs_type, t_float_exact)) {
        int32_t binop = inplace_to_binary(op);
        if (binop >= 0 && (float_slot_method(binop) != NULL || binop == HIR_BOP_Power)) {
            void *guarded = simplify_env_emit_guard_type_deopt(env, t_float_exact, lhs, fs);
            simplify_env_emit_use_type(env, rhs, t_float_exact);
            return simplify_env_emit_float_binary_op_deopt(env, binop, guarded, rhs, fs);
        }
    }

    /* Path 3: Float in-place (both Float) */
    if (hir_type_is_subtype(lhs_type, t_float_exact) &&
        hir_type_is_subtype(rhs_type, t_float_exact)) {
        int32_t binop = inplace_to_binary(op);
        if (binop >= 0 && (float_slot_method(binop) != NULL || binop == HIR_BOP_Power)) {
            simplify_env_emit_use_type(env, lhs, t_float_exact);
            simplify_env_emit_use_type(env, rhs, t_float_exact);
            return simplify_env_emit_float_binary_op_deopt(env, binop, lhs, rhs, fs);
        }
    }

    /* Path 4: Float speculation (LHS is Float, RHS not Float but could be) */
    if (hir_type_is_subtype(lhs_type, t_float_exact) &&
        !hir_type_is_subtype(rhs_type, t_float_exact) &&
        hir_type_could_be(&rhs_type, &t_float_exact)) {
        int32_t binop = inplace_to_binary(op);
        if (binop >= 0 && (float_slot_method(binop) != NULL || binop == HIR_BOP_Power)) {
            simplify_env_emit_use_type(env, lhs, t_float_exact);
            void *guarded = simplify_env_emit_guard_type_deopt(env, t_float_exact, rhs, fs);
            return simplify_env_emit_float_binary_op_deopt(env, binop, lhs, guarded, fs);
        }
    }

    /* Path 5: Reverse float speculation (RHS is Float, LHS not but could be) */
    if (hir_type_is_subtype(rhs_type, t_float_exact) &&
        !hir_type_is_subtype(lhs_type, t_float_exact) &&
        hir_type_could_be(&lhs_type, &t_float_exact)) {
        int32_t binop = inplace_to_binary(op);
        if (binop >= 0 && (float_slot_method(binop) != NULL || binop == HIR_BOP_Power)) {
            simplify_env_emit_use_type(env, rhs, t_float_exact);
            void *guarded = simplify_env_emit_guard_type_deopt(env, t_float_exact, lhs, fs);
            return simplify_env_emit_float_binary_op_deopt(env, binop, guarded, rhs, fs);
        }
    }

    return NULL;
}

/* ---- Float slot method lookup ---- */
static binaryfunc float_slot_method(int32_t op) {
    PyNumberMethods *nb = PyFloat_Type.tp_as_number;
    switch (op) {
        case HIR_BOP_Add: return nb->nb_add;
        case HIR_BOP_FloorDivide: return nb->nb_floor_divide;
        case HIR_BOP_Modulo: return nb->nb_remainder;
        case HIR_BOP_Multiply: return nb->nb_multiply;
        case HIR_BOP_Subtract: return nb->nb_subtract;
        case HIR_BOP_TrueDivide: return nb->nb_true_divide;
        default: return NULL;
    }
}

void *simplify_env_emit_double_binary_op(SimplifyEnv *env, int32_t op,
                                          void *left, void *right) {
    void *reg = hir_func_alloc_register(env->func);
    void *instr = hir_c_create_double_binary_op(reg, op, left, right);
    return simplify_env_emit(env, instr);
}

/* ---- simplifyFloatBinaryOp ---- */
void *simplify_float_binary_op_c(SimplifyEnv *env, const void *instr) {
    int32_t op = hir_c_binary_op_kind(instr);
    void *left = hir_c_get_operand(instr, 0);
    void *right = hir_c_get_operand(instr, 1);

    /* Path 1: unbox + native double arithmetic + box */
    if (op != HIR_BOP_Power && float_slot_method(op) != NULL) {
        HirType t_cdouble = HIR_TYPE_CDOUBLE;
        void *left_unboxed = simplify_env_emit_primitive_unbox(env, left, t_cdouble);
        void *right_unboxed = simplify_env_emit_primitive_unbox(env, right, t_cdouble);
        void *result = simplify_env_emit_double_binary_op(env, op, left_unboxed, right_unboxed);
        void *fs = hir_c_get_frame_state(instr);
        HirType t_float_exact = HIR_TYPE_FLOATEXACT;
        extern void *hir_c_create_primitive_box(void *func, void *src, HirType type, void *fs);
        void *box = hir_c_create_primitive_box(env->func, result, t_float_exact, fs);
        return simplify_env_emit(env, box);
    }

    /* Path 2: constant folding (same pattern as LongBinaryOp) */
    extern int jit_compile_running(void);
    if (jit_compile_running()) return NULL;

    HirType left_type = hir_register_type(left);
    HirType right_type = hir_register_type(right);
    if (!hir_type_has_object_spec(&left_type) ||
        !hir_type_has_object_spec(&right_type)) return NULL;

    PyObject *left_obj = hir_type_object_spec(&left_type);
    PyObject *right_obj = hir_type_object_spec(&right_type);

    extern void jit_compile_lock(void);
    extern void jit_compile_unlock(void);
    jit_compile_lock();

    PyObject *result;
    if (op == HIR_BOP_Power) {
        result = PyFloat_Type.tp_as_number->nb_power(left_obj, right_obj, Py_None);
    } else {
        binaryfunc slot = float_slot_method(op);
        if (slot == NULL) { jit_compile_unlock(); return NULL; }
        result = (*slot)(left_obj, right_obj);
    }

    jit_compile_unlock();

    if (result == NULL) {
        PyErr_Clear();
        return NULL;
    }

    simplify_env_emit_use_type(env, left, left_type);
    simplify_env_emit_use_type(env, right, right_type);
    extern PyObject *hir_func_add_reference(void *func, PyObject *obj);
    PyObject *ref = hir_func_add_reference(env->func, result);
    Py_DECREF(result);
    return simplify_env_emit_load_const(env, hir_type_from_object(ref));
}

/* ---- simplifyLongBinaryOp ---- */
void *simplify_long_binary_op_c(SimplifyEnv *env, const void *instr) {
    extern int jit_compile_running(void);
    if (jit_compile_running()) return NULL;

    void *left = hir_c_get_operand(instr, 0);
    void *right = hir_c_get_operand(instr, 1);
    HirType left_type = hir_register_type(left);
    HirType right_type = hir_register_type(right);

    if (!hir_type_has_object_spec(&left_type) ||
        !hir_type_has_object_spec(&right_type)) return NULL;

    int32_t op = hir_c_binary_op_kind(instr);
    PyObject *left_obj = hir_type_object_spec(&left_type);
    PyObject *right_obj = hir_type_object_spec(&right_type);

    extern void jit_compile_lock(void);
    extern void jit_compile_unlock(void);
    jit_compile_lock();

    PyObject *result;
    if (op == HIR_BOP_Power) {
        result = PyLong_Type.tp_as_number->nb_power(left_obj, right_obj, Py_None);
    } else {
        binaryfunc slot = long_slot_method(op);
        if (slot == NULL) { jit_compile_unlock(); return NULL; }
        result = (*slot)(left_obj, right_obj);
    }

    jit_compile_unlock();

    if (result == NULL) {
        PyErr_Clear();
        return NULL;
    }

    simplify_env_emit_use_type(env, left, left_type);
    simplify_env_emit_use_type(env, right, right_type);
    extern PyObject *hir_func_add_reference(void *func, PyObject *obj);
    PyObject *ref = hir_func_add_reference(env->func, result);
    Py_DECREF(result);
    return simplify_env_emit_load_const(env, hir_type_from_object(ref));
}

/* ---- simplifyGetLength ----
 * If obj is a collection with known length field, emit LoadField + PrimitiveBox. */
void *simplify_get_length_c(SimplifyEnv *env, const void *instr) {
    void *obj = hir_c_get_operand(instr, 0);
    void *size = emit_get_length_int64_c(env, obj);
    if (size == NULL) return NULL;

    void *fs = hir_c_get_frame_state(instr);
    HirType t_cint64 = HIR_TYPE_CINT64;
    extern void *hir_c_create_primitive_box(void *func, void *src, HirType type, void *fs);
    void *box = hir_c_create_primitive_box(env->func, size, t_cint64, fs);
    return simplify_env_emit(env, box);
}

/* ---- simplifyStoreSubscr ----
 * If target is DictExact, call mp_ass_subscript directly + check_neg. */
void *simplify_store_subscr_c(SimplifyEnv *env, const void *instr) {
    void *target = hir_c_get_operand(instr, 0);
    HirType target_type = hir_register_type(target);
    HirType t_dict_exact = HIR_TYPE_DICTEXACT;

    if (!hir_type_is_subtype(target_type, t_dict_exact)) return NULL;

    void *addr = (void *)PyDict_Type.tp_as_mapping->mp_ass_subscript;
    HirType t_cint32 = HIR_TYPE_CINT32;
    void *call = simplify_env_emit_call_static_instr(env, 3, addr, t_cint32);
    hir_c_set_operand(call, 0, hir_c_get_operand(instr, 0));
    hir_c_set_operand(call, 1, hir_c_get_operand(instr, 1));
    hir_c_set_operand(call, 2, hir_c_get_operand(instr, 2));

    void *fs = hir_c_get_frame_state(instr);
    simplify_env_emit_check_neg(env, hir_c_output(call), fs);
    return NULL;
}

/* Helper: emit LoadConst for a known PyObject* (registers reference + creates type) */
void *simplify_env_emit_load_const_object(SimplifyEnv *env, PyObject *obj) {
    extern PyObject *hir_func_add_reference(void *func, PyObject *obj);
    PyObject *ref = hir_func_add_reference(env->func, obj);
    return simplify_env_emit_load_const(env, hir_type_from_object(ref));
}

void *simplify_env_emit_unicode_compare(SimplifyEnv *env, int32_t op,
                                         void *left, void *right) {
    void *reg = hir_func_alloc_register(env->func);
    void *instr = hir_c_create_unicode_compare(reg, op, left, right);
    return simplify_env_emit(env, instr);
}

/* Helper: create HirType with int specialization */
static HirType make_int_spec_type(HirType base, intptr_t val) {
    base.bits_and_flags |= ((uint64_t)HIR_SPEC_INT << HIR_TYPE_SPEC_SHIFT);
    base.int_val = val;
    return base;
}

static HirType make_cbool_type(intptr_t val) {
    return make_int_spec_type(((HirType)HIR_TYPE_CBOOL), val != 0 ? 1 : 0);
}

static HirType make_cint_type(HirType target_type, intptr_t val) {
    uint64_t bits = target_type.bits_and_flags & HIR_TYPE_BITS_MASK;
    HirType t = HIR_TYPE_SIMPLE(bits, HIR_TYPE_LIFETIME_BOTTOM);
    return make_int_spec_type(t, val);
}

/* ---- simplifyCIntToCBool ----
 * If input is a known int constant, fold to CBool(val != 0). */
void *simplify_cint_to_cbool_c(SimplifyEnv *env, const void *instr) {
    void *input = hir_c_get_operand(instr, 0);
    HirType input_type = hir_register_type(input);
    if (hir_type_has_int_spec(&input_type)) {
        return simplify_env_emit_load_const(env, make_cbool_type(hir_type_int_spec(&input_type)));
    }
    return NULL;
}

/* ---- simplifyCompare (partial — None, Float, Long paths) ---- */
void *simplify_compare_c(SimplifyEnv *env, const void *instr) {
    void *left = hir_c_get_operand(instr, 0);
    void *right = hir_c_get_operand(instr, 1);
    HirType left_type = hir_register_type(left);
    HirType right_type = hir_register_type(right);
    int32_t op = hir_c_compare_op(instr);

    HirType t_none = HIR_TYPE_NONETYPE;

    /* None == None or None != None */
    if (hir_type_is_subtype(left_type, t_none) &&
        hir_type_is_subtype(right_type, t_none)) {
        if (op == HIR_CMP_Equal || op == HIR_CMP_NotEqual) {
            simplify_env_emit_use_type(env, left, t_none);
            simplify_env_emit_use_type(env, right, t_none);
            PyObject *result = (op == HIR_CMP_Equal) ? Py_True : Py_False;
            return simplify_env_emit_load_const(env, hir_type_from_object(result));
        }
    }

    /* Bool == Bool or Bool != Bool → PrimitiveCompare + PrimitiveBoxBool */
    HirType t_bool = HIR_TYPE_BOOL;
    if (hir_type_is_subtype(left_type, t_bool) &&
        hir_type_is_subtype(right_type, t_bool) &&
        (op == HIR_CMP_Equal || op == HIR_CMP_NotEqual)) {
        int32_t prim_op = (op == HIR_CMP_Equal) ? HIR_PCMP_Equal : HIR_PCMP_NotEqual;
        simplify_env_emit_use_type(env, left, t_bool);
        simplify_env_emit_use_type(env, right, t_bool);
        void *result = simplify_env_emit_primitive_compare(env, prim_op, left, right);
        return simplify_env_emit_primitive_box_bool(env, result);
    }

    /* Float comparison (not In/NotIn/ExcMatch) */
    HirType t_float_exact = HIR_TYPE_FLOATEXACT;
    if (hir_type_is_subtype(left_type, t_float_exact) &&
        hir_type_is_subtype(right_type, t_float_exact) &&
        op != HIR_CMP_In && op != HIR_CMP_NotIn && op != HIR_CMP_ExcMatch) {
        return simplify_env_emit_float_compare(env, op, left, right);
    }

    /* Long comparison (not In/NotIn/ExcMatch) */
    HirType t_long_exact = HIR_TYPE_LONGEXACT;
    if (hir_type_is_subtype(left_type, t_long_exact) &&
        hir_type_is_subtype(right_type, t_long_exact) &&
        op != HIR_CMP_In && op != HIR_CMP_NotIn && op != HIR_CMP_ExcMatch) {
        return simplify_env_emit_long_compare(env, op, left, right);
    }

    /* Unicode comparison (not In/NotIn/ExcMatch) */
    HirType t_unicode_exact = HIR_TYPE_UNICODEEXACT;
    if (hir_type_is_subtype(left_type, t_unicode_exact) &&
        hir_type_is_subtype(right_type, t_unicode_exact) &&
        op != HIR_CMP_In && op != HIR_CMP_NotIn && op != HIR_CMP_ExcMatch) {
        return simplify_env_emit_unicode_compare(env, op, left, right);
    }

    return NULL;
}

/* ---- simplifyPrimitiveCompare (full C port) ---- */
void *simplify_primitive_compare_c(SimplifyEnv *env, const void *instr) {
    void *left = hir_c_get_operand(instr, 0);
    void *right = hir_c_get_operand(instr, 1);
    int32_t op = hir_c_compare_op(instr);

    if (op == HIR_PCMP_Equal || op == HIR_PCMP_NotEqual) {
        HirType left_type = hir_register_type(left);
        HirType right_type = hir_register_type(right);

        /* Types can't overlap → always false (or true for !=) */
        if (!hir_type_could_be(&left_type, &right_type)) {
            simplify_env_emit_use_type(env, left, left_type);
            simplify_env_emit_use_type(env, right, right_type);
            int val = (op == HIR_PCMP_NotEqual) ? 1 : 0;
            return simplify_env_emit_load_const(env, make_cbool_type(val));
        }
        /* Both have int specialization → compare values */
        if (hir_type_has_int_spec(&left_type) && hir_type_has_int_spec(&right_type)) {
            int equal = (hir_type_int_spec(&left_type) == hir_type_int_spec(&right_type));
            int val = (op == HIR_PCMP_NotEqual) ? !equal : equal;
            simplify_env_emit_use_type(env, left, left_type);
            simplify_env_emit_use_type(env, right, right_type);
            return simplify_env_emit_load_const(env, make_cbool_type(val));
        }
        /* Both have object specialization → compare pointers */
        if (hir_type_has_object_spec(&left_type) && hir_type_has_object_spec(&right_type)) {
            int equal = (hir_type_object_spec(&left_type) == hir_type_object_spec(&right_type));
            int val = (op == HIR_PCMP_NotEqual) ? !equal : equal;
            simplify_env_emit_use_type(env, left, left_type);
            simplify_env_emit_use_type(env, right, right_type);
            return simplify_env_emit_load_const(env, make_cbool_type(val));
        }
    }

    /* box(b) == True → b */
    if (op == HIR_PCMP_Equal) {
        extern void *hir_reg_instr(void *reg);
        void *left_def = hir_reg_instr(left);
        if (left_def != NULL && hir_c_opcode(left_def) == HIR_OP_PrimitiveBoxBool) {
            HirType right_type = hir_register_type(right);
            PyObject *right_obj = hir_type_as_object(&right_type);
            if (right_obj == Py_True) {
                return hir_c_get_operand(left_def, 0);
            }
        }
    }

    return NULL;
}

/* ---- simplifyIsTruthy (partial — Bool + LongExact paths) ---- */
void *simplify_is_truthy_c(SimplifyEnv *env, const void *instr) {
    void *input = hir_c_get_operand(instr, 0);
    HirType input_type = hir_register_type(input);

    /* Known immutable object: constant-fold PyObject_IsTrue */
    PyObject *obj = hir_type_as_object(&input_type);
    if (obj != NULL) {
        PyTypeObject *tp = Py_TYPE(obj);
        if (tp == &PyBool_Type || tp == &PyFloat_Type ||
            tp == &PyLong_Type || tp == &PyFrozenSet_Type ||
            tp == &PySlice_Type || tp == &PyTuple_Type ||
            tp == &PyUnicode_Type || tp == Py_TYPE(Py_None)) {
            int res = PyObject_IsTrue(obj);
            if (res >= 0) {
                simplify_env_emit_use_type(env, input, input_type);
                return simplify_env_emit_load_const(env, make_cbool_type(res));
            }
        }
    }

    /* TBool: compare with Py_True */
    HirType t_bool = HIR_TYPE_BOOL;
    if (hir_type_is_subtype(input_type, t_bool)) {
        simplify_env_emit_use_type(env, input, t_bool);
        void *right = simplify_env_emit_load_const(env, hir_type_from_object(Py_True));
        return simplify_env_emit_primitive_compare(env, HIR_PCMP_Equal, input, right);
    }

    /* Collection length path: emit GetLengthInt64 + CIntToCBool */
    void *size = emit_get_length_int64_c(env, input);
    if (size != NULL) {
        return simplify_env_emit_cint_to_cbool(env, size);
    }

    /* TLongExact: compare with _PyLong_GetZero() */
    HirType t_long_exact = HIR_TYPE_LONGEXACT;
    if (hir_type_is_subtype(input_type, t_long_exact)) {
        simplify_env_emit_use_type(env, input, input_type);
        void *right = simplify_env_emit_load_const(env, hir_type_from_object(_PyLong_GetZero()));
        return simplify_env_emit_primitive_compare(env, HIR_PCMP_NotEqual, input, right);
    }

    return NULL;
}

/* ---- simplifyCheckSequenceBounds ----
 * If sequence is MakeTuple with known length + idx is known int, fold bounds check. */
void *simplify_check_sequence_bounds_c(SimplifyEnv *env, const void *instr) {
    void *sequence = hir_c_get_operand(instr, 0);
    void *idx_reg = hir_c_get_operand(instr, 1);
    HirType seq_type = hir_register_type(sequence);
    HirType idx_type = hir_register_type(idx_reg);

    HirType t_tuple_exact = HIR_TYPE_TUPLEEXACT;
    HirType t_cint = HIR_TYPE_CINT;

    if (!hir_type_is_subtype(seq_type, t_tuple_exact)) return NULL;
    if (!hir_type_is_subtype(idx_type, t_cint)) return NULL;

    extern void *hir_reg_instr(void *reg);
    void *seq_def = hir_reg_instr(sequence);
    if (seq_def == NULL || hir_c_opcode(seq_def) != HIR_OP_MakeTuple) return NULL;

    if (!hir_type_has_int_spec(&idx_type)) return NULL;

    size_t length = hir_c_num_operands(seq_def);
    intptr_t idx_value = hir_type_int_spec(&idx_type);
    int adjusted = 0;
    if (idx_value < 0) {
        idx_value += (intptr_t)length;
        adjusted = 1;
    }
    if ((size_t)idx_value < length) {
        simplify_env_emit_use_type(env, sequence, seq_type);
        simplify_env_emit_use_type(env, idx_reg, idx_type);
        if (adjusted) {
            HirType t_cint64 = HIR_TYPE_CINT64;
            return simplify_env_emit_load_const(env, make_cint_type(t_cint64, idx_value));
        }
        return idx_reg;
    }
    return NULL;
}

/* ---- simplifyLoadArrayItem (partial — known tuple path) ----
 * If src is a known tuple and idx is a known int, fold to constant item. */
void *simplify_load_array_item_tuple_c(SimplifyEnv *env, const void *instr) {
    void *src = hir_c_get_operand(instr, 0);
    void *idx_reg = hir_c_get_operand(instr, 1);
    HirType idx_type = hir_register_type(idx_reg);
    if (!hir_type_has_int_spec(&idx_type)) return NULL;

    intptr_t idx_signed = hir_type_int_spec(&idx_type);
    if (idx_signed < 0) return NULL;

    /* MakeTuple path: return the idx-th operand directly */
    extern void *hir_reg_instr(void *reg);
    void *src_def = hir_reg_instr(src);
    if (src_def != NULL && hir_c_opcode(src_def) == HIR_OP_MakeTuple) {
        size_t length = hir_c_num_operands(src_def);
        if ((size_t)idx_signed < length) {
            HirType t_tuple_exact_use = HIR_TYPE_TUPLEEXACT;
            simplify_env_emit_use_type(env, src, t_tuple_exact_use);
            simplify_env_emit_use_type(env, idx_reg, idx_type);
            return hir_c_get_operand(src_def, (size_t)idx_signed);
        }
    }

    /* Known tuple object path */
    HirType src_type = hir_register_type(src);
    HirType t_tuple_exact = HIR_TYPE_TUPLEEXACT;
    if (hir_type_has_value_spec(&src_type, t_tuple_exact)) {
        PyObject *tuple_obj = hir_type_object_spec(&src_type);
        if (idx_signed < PyTuple_GET_SIZE(tuple_obj)) {
            simplify_env_emit_use_type(env, src, src_type);
            simplify_env_emit_use_type(env, idx_reg, idx_type);
            PyObject *item = PyTuple_GET_ITEM(tuple_obj, idx_signed);
            extern PyObject *hir_func_add_reference(void *func, PyObject *obj);
            PyObject *ref = hir_func_add_reference(env->func, item);
            return simplify_env_emit_load_const(env, hir_type_from_object(ref));
        }
    }
    return NULL;
}

/* ---- simplifyLoadTupleItem ----
 * If src is a known tuple object, fold to the constant item. */
void *simplify_load_tuple_item_c(SimplifyEnv *env, const void *instr) {
    void *src = hir_c_get_operand(instr, 0);
    HirType src_type = hir_register_type(src);
    HirType t_tuple = HIR_TYPE_TUPLE;
    if (!hir_type_has_value_spec(&src_type, t_tuple)) return NULL;

    simplify_env_emit_use_type(env, src, src_type);
    PyObject *tuple_obj = hir_type_object_spec(&src_type);
    size_t idx = hir_c_load_tuple_item_idx(instr);
    PyObject *item = PyTuple_GET_ITEM(tuple_obj, idx);
    extern PyObject *hir_func_add_reference(void *func, PyObject *obj);
    PyObject *ref = hir_func_add_reference(env->func, item);
    HirType item_type = hir_type_from_object(ref);
    return simplify_env_emit_load_const(env, item_type);
}

/* ---- simplifyLoadField (partial — known float object) ----
 * If loadee is a known float and we're loading ob_fval, fold to constant. */
void *simplify_load_field_float_c(SimplifyEnv *env, const void *instr) {
    void *loadee = hir_c_get_operand(instr, 0);
    HirType loadee_type = hir_register_type(loadee);
    if (!hir_type_has_object_spec(&loadee_type)) return NULL;

    void *output_reg = hir_c_output(instr);
    if (output_reg == NULL) return NULL;
    HirType output_type = hir_register_type(output_reg);
    HirType t_cdouble = HIR_TYPE_CDOUBLE;
    if (!hir_type_is_subtype(output_type, t_cdouble)) return NULL;

    PyObject *value = hir_type_object_spec(&loadee_type);
    if (value == NULL || !PyFloat_Check(value)) return NULL;

    intptr_t offset = hir_c_load_field_offset(instr);
    if (offset != (intptr_t)offsetof(PyFloatObject, ob_fval)) return NULL;

    double number = PyFloat_AS_DOUBLE(value);
    simplify_env_emit_use_type(env, loadee, loadee_type);
    return simplify_env_emit_load_const(env, hir_type_from_cdouble(number));
}

/* ---- simplifyCondBranchCheckType ----
 * If value type is subtype of expected → Branch to true_bb.
 * If value type can't be expected → Branch to false_bb. */
void *simplify_cond_branch_check_type_c(SimplifyEnv *env, const void *instr) {
    void *value = hir_c_get_operand(instr, 0);
    HirType actual_type = hir_register_type(value);
    const HirCondBranchCheckType *cbct = (const HirCondBranchCheckType *)instr;
    HirType expected_type = cbct->type;

    if (hir_type_is_subtype(actual_type, expected_type)) {
        simplify_env_emit_use_type(env, value, actual_type);
        extern void *hir_c_create_branch_cpp(void *target_block);
        void *branch = hir_c_create_branch_cpp(cbct->true_edge.to);
        return simplify_env_emit(env, branch);
    }
    if (!hir_type_could_be(&actual_type, &expected_type)) {
        simplify_env_emit_use_type(env, value, actual_type);
        extern void *hir_c_create_branch_cpp(void *target_block);
        void *branch = hir_c_create_branch_cpp(cbct->false_edge.to);
        return simplify_env_emit(env, branch);
    }
    return NULL;
}

/* ---- simplifyUnbox — full C port ---- */
static int cint_fits_type(int64_t val, HirType target) {
    uint64_t bits = target.bits_and_flags & HIR_TYPE_BITS_MASK;
    if (bits == 0x01000000000ULL) return 1; /* CInt64 — always fits */
    if (bits == 0x00800000000ULL) return val >= INT32_MIN && val <= INT32_MAX;
    if (bits == 0x00400000000ULL) return val >= INT16_MIN && val <= INT16_MAX;
    return val >= INT8_MIN && val <= INT8_MAX; /* CInt8 */
}

static int cuint_fits_type(int64_t val, HirType target) {
    if (val < 0) return 0;
    uint64_t bits = target.bits_and_flags & HIR_TYPE_BITS_MASK;
    if (bits == 0x10000000000ULL) return 1; /* CUInt64 */
    if (bits == 0x08000000000ULL) return (uint64_t)val <= UINT32_MAX;
    if (bits == 0x04000000000ULL) return (uint64_t)val <= UINT16_MAX;
    return (uint64_t)val <= UINT8_MAX; /* CUInt8 */
}

/* ---- simplifyUnbox — full C port ---- */
void *simplify_unbox_box_c(SimplifyEnv *env, const void *instr) {
    void *input = hir_c_get_operand(instr, 0);
    void *output_reg = hir_c_output(instr);
    if (output_reg == NULL) return NULL;
    HirType output_type = hir_register_type(output_reg);

    extern void *hir_reg_instr(void *reg);
    void *box_instr = hir_reg_instr(input);
    if (box_instr == NULL || hir_c_opcode(box_instr) != HIR_OP_PrimitiveBox)
        return NULL;

    HirType box_type = ((const HirPrimitiveBox *)box_instr)->type;
    if (hir_type_equal(&box_type, &output_type)) {
        return hir_c_get_operand(box_instr, 0);
    }

    /* Constant-folding for known int/float values */
    HirType input_type = hir_register_type(input);
    if (!hir_type_has_object_spec(&input_type)) return NULL;
    PyObject *value = hir_type_object_spec(&input_type);

    HirType t_csigned = HIR_TYPE_SIMPLE(0x01e00000000ULL, HIR_TYPE_LIFETIME_BOTTOM);
    HirType t_cunsigned = HIR_TYPE_SIMPLE(0x1e000000000ULL, HIR_TYPE_LIFETIME_BOTTOM);
    HirType t_csigned_or_unsigned = {t_csigned.bits_and_flags | t_cunsigned.bits_and_flags, {0}};

    if (hir_type_is_subtype(output_type, t_csigned_or_unsigned)) {
        if (!PyLong_Check(value)) return NULL;
        int overflow = 0;
        long number = PyLong_AsLongAndOverflow(value, &overflow);
        if (overflow != 0) return NULL;

        if (hir_type_is_subtype(output_type, t_csigned)) {
            if (!cint_fits_type(number, output_type)) return NULL;
            return simplify_env_emit_load_const(env, make_cint_type(output_type, (intptr_t)number));
        } else {
            if (!cuint_fits_type(number, output_type)) return NULL;
            return simplify_env_emit_load_const(env, make_cint_type(output_type, (intptr_t)number));
        }
    }

    HirType t_cdouble = HIR_TYPE_CDOUBLE;
    if (hir_type_is_subtype(output_type, t_cdouble)) {
        if (!PyFloat_Check(value)) return NULL;
        double number = PyFloat_AS_DOUBLE(value);
        return simplify_env_emit_load_const(env, hir_type_from_cdouble(number));
    }

    return NULL;
}

/* ---- simplifyIntConvert ----
 * If input already has the target type, IntConvert is redundant. */
void *simplify_int_convert_c(SimplifyEnv *env, const void *instr) {
    void *src = hir_c_get_operand(instr, 0);
    HirType src_type = hir_register_type(src);
    HirType target = ((const HirIntConvert *)instr)->type;
    if (hir_type_is_subtype(src_type, target)) {
        simplify_env_emit_use_type(env, src, target);
        return src;
    }
    return NULL;
}

/* ---- simplifyLoadVarObjectSize (partial — known object path) ----
 * If input is a known tuple/bytes object, fold to constant ob_size. */
void *simplify_load_var_object_size_c(SimplifyEnv *env, const void *instr) {
    void *obj_reg = hir_c_get_operand(instr, 0);
    HirType obj_type = hir_register_type(obj_reg);

    /* MakeTuple path: nvalues = NumOperands */
    extern void *hir_reg_instr(void *reg);
    void *obj_def = hir_reg_instr(obj_reg);
    if (obj_def != NULL && hir_c_opcode(obj_def) == HIR_OP_MakeTuple) {
        simplify_env_emit_use_type(env, obj_reg, obj_type);
        size_t size = hir_c_num_operands(obj_def);
        HirType output_type = hir_register_type(hir_c_output(instr));
        return simplify_env_emit_load_const(env, make_cint_type(output_type, (intptr_t)size));
    }

    /* Known tuple/bytes object path */
    HirType t_tuple_exact = HIR_TYPE_TUPLEEXACT;
    HirType t_bytes_exact = HIR_TYPE_BYTESEXACT;

    if (hir_type_has_value_spec(&obj_type, t_tuple_exact) ||
        hir_type_has_value_spec(&obj_type, t_bytes_exact)) {
        PyObject *obj = hir_type_as_object(&obj_type);
        if (obj != NULL) {
            Py_ssize_t size = ((PyVarObject *)obj)->ob_size;
            simplify_env_emit_use_type(env, obj_reg, obj_type);
            HirType output_type = hir_register_type(hir_c_output(instr));
            return simplify_env_emit_load_const(env, make_cint_type(output_type, (intptr_t)size));
        }
    }
    return NULL;
}

/* ---- simplifyIsNegativeAndErrOccurred ----
 * If input is a LoadConst, we know no exception is active → result is 0. */
void *simplify_is_neg_and_err_c(SimplifyEnv *env, const void *instr) {
    void *input = hir_c_get_operand(instr, 0);
    extern void *hir_reg_instr(void *reg);
    void *def = hir_reg_instr(input);
    if (def == NULL || hir_c_opcode(def) != HIR_OP_LoadConst) {
        return NULL;
    }
    HirType output_type = hir_register_type(hir_c_output(instr));
    return simplify_env_emit_load_const(env, make_cint_type(output_type, 0));
}

/* ---- simplifyCondBranch (partial — constant condition folding) ----
 * If condition is a known int constant, fold to unconditional Branch. */
void *simplify_cond_branch_const_c(SimplifyEnv *env, const void *instr) {
    void *cond = hir_c_get_operand(instr, 0);
    HirType cond_type = hir_register_type(cond);
    if (hir_type_has_int_spec(&cond_type)) {
        intptr_t spec = hir_type_int_spec(&cond_type);
        void *target = hir_c_successor(instr, spec ? 0 : 1);
        extern void *hir_c_create_branch_cpp(void *target_block);
        void *branch = hir_c_create_branch_cpp(target);
        return simplify_env_emit(env, branch);
    }
    /* IntConvert forwarding: if cond is from a widening IntConvert, use src */
    extern void *hir_reg_instr(void *reg);
    void *cond_def = hir_reg_instr(cond);
    if (cond_def != NULL && hir_c_opcode(cond_def) == HIR_OP_IntConvert) {
        void *src = hir_c_get_operand(cond_def, 0);
        HirType convert_type = ((const HirIntConvert *)cond_def)->type;
        HirType src_type = hir_register_type(src);
        if (hir_type_size_in_bytes(&convert_type) >= hir_type_size_in_bytes(&src_type)) {
            extern void *hir_c_create_cond_branch_cpp(void *cond_reg, void *true_bb, void *false_bb);
            void *true_bb = hir_c_successor(instr, 0);
            void *false_bb = hir_c_successor(instr, 1);
            void *new_cb = hir_c_create_cond_branch_cpp(src, true_bb, false_bb);
            return simplify_env_emit(env, new_cb);
        }
    }
    return NULL;
}

/* ---- simplifyPrimitiveBoxBool ----
 * If input is a known int constant, replace with Py_True/Py_False. */
void *simplify_primitive_box_bool_c(SimplifyEnv *env, const void *instr) {
    void *input = hir_c_get_operand(instr, 0);
    HirType input_type = hir_register_type(input);
    if (hir_type_has_int_spec(&input_type)) {
        simplify_env_emit_use_type(env, input, input_type);
        PyObject *bool_obj = hir_type_int_spec(&input_type) ? Py_True : Py_False;
        HirType result_type = hir_type_from_object(bool_obj);
        return simplify_env_emit_load_const(env, result_type);
    }
    return NULL;
}

/* ==== emitCond infrastructure ==== */

void *simplify_emit_cond(SimplifyEnv *env, void *cond_reg,
                         SimplifyCondBodyFn do_bb1, void *ctx1,
                         SimplifyCondBodyFn do_bb2, void *ctx2) {
    env->new_blocks += 3;

    void *bb1 = hir_cfg_alloc_block(env->func);
    void *bb2 = hir_cfg_alloc_block(env->func);

    void *cond_br = hir_c_create_cond_branch_cpp(cond_reg, bb1, bb2);
    simplify_env_emit(env, cond_br);

    void *tail = hir_cfg_split_after(env->func, cond_br);

    env->block = bb1;
    env->cursor_instr = NULL;
    void *bb1_reg = do_bb1(env, ctx1);
    void *br1 = hir_c_create_branch_cpp(tail);
    simplify_env_emit(env, br1);

    env->block = bb2;
    env->cursor_instr = NULL;
    void *bb2_reg = do_bb2(env, ctx2);
    void *br2 = hir_c_create_branch_cpp(tail);
    simplify_env_emit(env, br2);

    void *phi = hir_phi_create_2way(env->func, bb1, bb1_reg, bb2, bb2_reg);
    env->block = tail;
    env->cursor_instr = hir_bb_first_instr(tail);
    return simplify_env_emit(env, phi);
}

/* ==== LoadMethod emit helpers ==== */

static void *emit_primitive_compare_eq(SimplifyEnv *env, void *lhs, void *rhs) {
    void *reg = hir_func_alloc_register(env->func);
    void *instr = hir_c_create_primitive_compare(reg, HIR_PCMP_Equal, lhs, rhs);
    return simplify_env_emit(env, instr);
}

static void *emit_load_type_method_cache_entry_type(SimplifyEnv *env, int cache_id) {
    void *reg = hir_func_alloc_register(env->func);
    void *instr = hir_c_create_load_type_method_cache_entry_type(reg, cache_id);
    return simplify_env_emit(env, instr);
}

static void *emit_load_type_method_cache_entry_value(SimplifyEnv *env,
                                                      int cache_id, void *receiver) {
    void *reg = hir_func_alloc_register(env->func);
    void *instr = hir_c_create_load_type_method_cache_entry_value(reg, cache_id, receiver);
    return simplify_env_emit(env, instr);
}

static void *emit_fill_type_method_cache(SimplifyEnv *env, void *receiver,
                                          int32_t name_idx, int32_t cache_id, void *fs) {
    void *instr = hir_c_create_fill_type_method_cache(env->func, receiver,
                                                       name_idx, cache_id, fs);
    return simplify_env_emit(env, instr);
}

static void *emit_load_method_cached(SimplifyEnv *env, void *receiver,
                                      int32_t name_idx, void *fs) {
    void *instr = hir_c_create_load_method_cached(env->func, receiver, name_idx, fs);
    return simplify_env_emit(env, instr);
}

static void *emit_load_module_method_cached(SimplifyEnv *env, void *receiver,
                                             int32_t name_idx, void *fs) {
    void *instr = hir_c_create_load_module_method_cached(env->func, receiver, name_idx, fs);
    return simplify_env_emit(env, instr);
}

/* ==== simplifyLoadTypeMethodCached emitCond callbacks ==== */

typedef struct {
    int32_t cache_id;
    void *receiver;
} TypeMethodFastCtx;

typedef struct {
    int32_t cache_id;
    void *receiver;
    int32_t name_idx;
    void *fs;
} TypeMethodSlowCtx;

static void *type_method_fast_path(SimplifyEnv *env, void *ctx) {
    TypeMethodFastCtx *c = (TypeMethodFastCtx *)ctx;
    return emit_load_type_method_cache_entry_value(env, c->cache_id, c->receiver);
}

static void *type_method_slow_path(SimplifyEnv *env, void *ctx) {
    TypeMethodSlowCtx *c = (TypeMethodSlowCtx *)ctx;
    return emit_fill_type_method_cache(env, c->receiver, c->name_idx, c->cache_id, c->fs);
}

/* ==== simplifyLoadMethod ==== */
void *simplify_load_method_c(SimplifyEnv *env, const void *instr) {
    if (!jit_get_config()->attr_caches) return NULL;

    void *receiver = hir_c_get_operand(instr, 0);
    HirType recv_type = hir_register_type(receiver);
    int32_t name_idx = ((const HirLoadMethod *)instr)->name_idx;
    void *fs = hir_c_get_frame_state(instr);

    HirType t_type = HIR_TYPE_TYPE;
    if (hir_type_is_subtype(recv_type, t_type)) {
        int cache_id = hir_func_env_allocate_type_method_cache(env->func);
        simplify_env_emit_use_type(env, receiver, t_type);
        void *guard = emit_load_type_method_cache_entry_type(env, cache_id);
        void *type_matches = emit_primitive_compare_eq(env, guard, receiver);

        TypeMethodFastCtx fast_ctx = {cache_id, receiver};
        TypeMethodSlowCtx slow_ctx = {cache_id, receiver, name_idx, fs};

        return simplify_emit_cond(env, type_matches,
                                  type_method_fast_path, &fast_ctx,
                                  type_method_slow_path, &slow_ctx);
    }

    extern PyTypeObject Ci_StrictModule_Type;
    PyTypeObject *pytype = hir_type_runtime_py_type(&recv_type);
    if (pytype == &PyModule_Type || pytype == &Ci_StrictModule_Type) {
        return emit_load_module_method_cached(env, receiver, name_idx, fs);
    }

    return emit_load_method_cached(env, receiver, name_idx, fs);
}

/* ==== BinaryOp emit helpers ==== */

static void *simplify_env_emit_dict_subscr(SimplifyEnv *env, void *lhs, void *rhs, void *fs) {
    extern void *hir_c_create_dict_subscr(void *func, void *lhs, void *rhs, void *fs);
    void *instr = hir_c_create_dict_subscr(env->func, lhs, rhs, fs);
    return simplify_env_emit(env, instr);
}

static void *simplify_env_emit_index_unbox(SimplifyEnv *env, void *src, void *exc) {
    void *reg = hir_func_alloc_register(env->func);
    void *instr = hir_c_create_index_unbox(reg, src, exc);
    return simplify_env_emit(env, instr);
}

static void simplify_env_emit_is_neg_and_err(SimplifyEnv *env, void *src, void *fs) {
    extern void *hir_c_create_is_neg_and_err(void *func, void *src, void *fs);
    void *instr = hir_c_create_is_neg_and_err(env->func, src, fs);
    simplify_env_emit(env, instr);
}

static void *simplify_env_emit_check_sequence_bounds(SimplifyEnv *env,
        void *seq, void *idx, void *fs) {
    extern void *hir_c_create_check_sequence_bounds_reg(void *dst, void *seq, void *idx, void *fs);
    void *reg = hir_func_alloc_register(env->func);
    void *instr = hir_c_create_check_sequence_bounds_reg(reg, seq, idx, fs);
    return simplify_env_emit(env, instr);
}

static void *simplify_env_emit_load_array_item(SimplifyEnv *env,
        void *arr, void *idx, void *container, intptr_t offset, HirType type) {
    extern void *hir_c_create_load_array_item(void *func, void *arr, void *idx,
        void *container, intptr_t offset, HirType type);
    void *instr = hir_c_create_load_array_item(env->func, arr, idx, container, offset, type);
    return simplify_env_emit(env, instr);
}

static void *simplify_env_emit_unicode_subscr(SimplifyEnv *env,
        void *lhs, void *idx, void *fs) {
    extern void *hir_c_create_unicode_subscr(void *func, void *lhs, void *idx, void *fs);
    void *instr = hir_c_create_unicode_subscr(env->func, lhs, idx, fs);
    return simplify_env_emit(env, instr);
}

static void *simplify_env_emit_long_binary_op_deopt(SimplifyEnv *env,
        int32_t op, void *lhs, void *rhs, void *fs) {
    extern void *hir_c_create_long_binary_op(void *func, int32_t op,
        void *lhs, void *rhs, void *fs);
    void *instr = hir_c_create_long_binary_op(env->func, op, lhs, rhs, fs);
    return simplify_env_emit(env, instr);
}

static void *simplify_env_emit_unicode_repeat(SimplifyEnv *env,
        void *lhs, void *rhs, void *fs) {
    extern void *hir_c_create_unicode_repeat(void *func, void *lhs, void *rhs, void *fs);
    void *instr = hir_c_create_unicode_repeat(env->func, lhs, rhs, fs);
    return simplify_env_emit(env, instr);
}

static void *simplify_env_emit_unicode_concat(SimplifyEnv *env,
        void *lhs, void *rhs, void *fs) {
    extern void *hir_c_create_unicode_concat(void *func, void *lhs, void *rhs, void *fs);
    void *instr = hir_c_create_unicode_concat(env->func, lhs, rhs, fs);
    return simplify_env_emit(env, instr);
}

/* ==== simplifyBinaryOp ==== */
void *simplify_binary_op_c(SimplifyEnv *env, const void *instr) {
    int32_t op = hir_c_binary_op_kind(instr);
    void *lhs = hir_c_get_operand(instr, 0);
    void *rhs = hir_c_get_operand(instr, 1);
    HirType lhs_type = hir_register_type(lhs);
    HirType rhs_type = hir_register_type(rhs);
    void *fs = hir_c_get_frame_state(instr);
    HirType t_long_exact = HIR_TYPE_LONGEXACT;
    HirType t_float_exact = HIR_TYPE_FLOATEXACT;
    HirType t_dict_exact = HIR_TYPE_DICTEXACT;
    HirType t_unicode_exact = HIR_TYPE_UNICODEEXACT;
    HirType t_tuple_exact = HIR_TYPE_TUPLEEXACT;
    HirType t_list_exact = HIR_TYPE_LISTEXACT;
    HirType t_object = HIR_TYPE_OBJECT;
    HirType t_cptr = HIR_TYPE_CPTR;

    /* Subscript paths */
    if (op == HIR_BOP_Subscript) {
        if (hir_type_is_subtype(lhs_type, t_dict_exact)) {
            return simplify_env_emit_dict_subscr(env, lhs, rhs, fs);
        }
        if (!hir_type_is_subtype(rhs_type, t_long_exact)) return NULL;

        /* Known tuple constant fold */
        if (hir_type_is_subtype(lhs_type, t_tuple_exact) &&
            hir_type_has_object_spec(&lhs_type) &&
            hir_type_has_object_spec(&rhs_type)) {
            int overflow;
            Py_ssize_t index = PyLong_AsLongAndOverflow(
                hir_type_object_spec(&rhs_type), &overflow);
            if (!overflow) {
                PyObject *lhs_obj = hir_type_object_spec(&lhs_type);
                if (index >= 0 && index < PyTuple_GET_SIZE(lhs_obj)) {
                    simplify_env_emit_use_type(env, lhs, lhs_type);
                    simplify_env_emit_use_type(env, rhs, rhs_type);
                    PyObject *item = PyTuple_GET_ITEM(lhs_obj, index);
                    return simplify_env_emit_load_const_object(env, item);
                }
            }
        }

#ifndef Py_GIL_DISABLED
        /* List/Tuple subscript → IndexUnbox + CheckBounds + LoadArrayItem */
        if (hir_type_is_subtype(lhs_type, t_list_exact) ||
            hir_type_is_subtype(lhs_type, t_tuple_exact)) {
            HirType use_type = hir_type_is_subtype(lhs_type, t_list_exact)
                ? t_list_exact : t_tuple_exact;
            simplify_env_emit_use_type(env, lhs, use_type);
            simplify_env_emit_use_type(env, rhs, t_long_exact);
            void *right_index = simplify_env_emit_index_unbox(env, rhs, PyExc_IndexError);
            simplify_env_emit_is_neg_and_err(env, right_index, fs);
            void *adjusted_idx = simplify_env_emit_check_sequence_bounds(env, lhs, right_index, fs);
            intptr_t offset = (intptr_t)offsetof(PyTupleObject, ob_item);
            void *array = lhs;
            if (hir_type_is_subtype(lhs_type, t_list_exact)) {
                array = simplify_env_emit_load_field(env, lhs, "ob_item",
                    (intptr_t)offsetof(PyListObject, ob_item), t_cptr, 0);
                offset = 0;
            }
            return simplify_env_emit_load_array_item(env, array, adjusted_idx,
                lhs, offset, t_object);
        }
#endif
        /* Unicode subscript constant fold */
        if (hir_type_is_subtype(lhs_type, t_unicode_exact) &&
            hir_type_has_object_spec(&lhs_type) &&
            hir_type_has_object_spec(&rhs_type)) {
            int overflow;
            Py_ssize_t index = PyLong_AsLongAndOverflow(
                hir_type_object_spec(&rhs_type), &overflow);
            if (!overflow && index >= 0 &&
                index < PyUnicode_GET_LENGTH(hir_type_object_spec(&lhs_type))) {
                simplify_env_emit_use_type(env, lhs, lhs_type);
                simplify_env_emit_use_type(env, rhs, rhs_type);
                Py_UCS4 ch = PyUnicode_READ_CHAR(hir_type_object_spec(&lhs_type), index);
                extern void jit_compile_lock(void);
                extern void jit_compile_unlock(void);
                jit_compile_lock();
                PyObject *result = PyUnicode_FromOrdinal(ch);
                jit_compile_unlock();
                if (result != NULL) {
                    extern PyObject *hir_func_add_reference(void *func, PyObject *obj);
                    PyObject *ref = hir_func_add_reference(env->func, result);
                    Py_DECREF(result);
                    return simplify_env_emit_load_const(env, hir_type_from_object(ref));
                }
                PyErr_Clear();
            }
        }
#ifndef Py_GIL_DISABLED
        /* Unicode subscript runtime */
        if (hir_type_is_subtype(lhs_type, t_unicode_exact)) {
            simplify_env_emit_use_type(env, lhs, t_unicode_exact);
            simplify_env_emit_use_type(env, rhs, t_long_exact);
            void *unboxed_idx = simplify_env_emit_index_unbox(env, rhs, PyExc_IndexError);
            simplify_env_emit_is_neg_and_err(env, unboxed_idx, fs);
            void *adjusted_idx = simplify_env_emit_check_sequence_bounds(
                env, lhs, unboxed_idx, fs);
            return simplify_env_emit_unicode_subscr(env, lhs, adjusted_idx, fs);
        }
#endif
        return NULL;
    }

    /* Long + Long → LongBinaryOp */
    if (hir_type_is_subtype(lhs_type, t_long_exact) &&
        hir_type_is_subtype(rhs_type, t_long_exact)) {
        if (op == HIR_BOP_MatrixMultiply || op == HIR_BOP_Subscript)
            return NULL;
        simplify_env_emit_use_type(env, lhs, t_long_exact);
        simplify_env_emit_use_type(env, rhs, t_long_exact);
        return simplify_env_emit_long_binary_op_deopt(env, op, lhs, rhs, fs);
    }

    /* Float speculation */
    if (op != HIR_BOP_Subscript && op != HIR_BOP_MatrixMultiply) {
        if (hir_type_is_subtype(lhs_type, t_float_exact) &&
            !hir_type_is_subtype(rhs_type, t_float_exact) &&
            hir_type_could_be(&rhs_type, &t_float_exact)) {
            if (float_slot_method(op) != NULL || op == HIR_BOP_Power) {
                simplify_env_emit_use_type(env, lhs, t_float_exact);
                void *guarded = simplify_env_emit_guard_type_deopt(env, t_float_exact, rhs, fs);
                return simplify_env_emit_float_binary_op_deopt(env, op, lhs, guarded, fs);
            }
        }
        if (hir_type_is_subtype(rhs_type, t_float_exact) &&
            !hir_type_is_subtype(lhs_type, t_float_exact) &&
            hir_type_could_be(&lhs_type, &t_float_exact)) {
            if (float_slot_method(op) != NULL || op == HIR_BOP_Power) {
                simplify_env_emit_use_type(env, rhs, t_float_exact);
                void *guarded = simplify_env_emit_guard_type_deopt(env, t_float_exact, lhs, fs);
                return simplify_env_emit_float_binary_op_deopt(env, op, guarded, rhs, fs);
            }
        }
    }

    /* Long speculation */
    if (op != HIR_BOP_Subscript && op != HIR_BOP_MatrixMultiply) {
        if (hir_type_is_subtype(lhs_type, t_long_exact) &&
            !hir_type_is_subtype(rhs_type, t_long_exact) &&
            hir_type_could_be(&rhs_type, &t_long_exact)) {
            simplify_env_emit_use_type(env, lhs, t_long_exact);
            void *guarded = simplify_env_emit_guard_type_deopt(env, t_long_exact, rhs, fs);
            return simplify_env_emit_long_binary_op_deopt(env, op, lhs, guarded, fs);
        }
        if (hir_type_is_subtype(rhs_type, t_long_exact) &&
            !hir_type_is_subtype(lhs_type, t_long_exact) &&
            hir_type_could_be(&lhs_type, &t_long_exact)) {
            simplify_env_emit_use_type(env, rhs, t_long_exact);
            void *guarded = simplify_env_emit_guard_type_deopt(env, t_long_exact, lhs, fs);
            return simplify_env_emit_long_binary_op_deopt(env, op, guarded, rhs, fs);
        }
    }

    /* Float + Float → FloatBinaryOp */
    if (hir_type_is_subtype(lhs_type, t_float_exact) &&
        hir_type_is_subtype(rhs_type, t_float_exact) &&
        (op == HIR_BOP_Power || float_slot_method(op) != NULL)) {
        simplify_env_emit_use_type(env, lhs, t_float_exact);
        simplify_env_emit_use_type(env, rhs, t_float_exact);
        return simplify_env_emit_float_binary_op_deopt(env, op, lhs, rhs, fs);
    }

    /* Int-to-float constant fold */
    {
        void *float_reg = NULL, *int_reg = NULL;
        if (hir_type_is_subtype(lhs_type, t_float_exact) &&
            hir_type_is_subtype(rhs_type, t_long_exact) &&
            hir_type_has_object_spec(&rhs_type)) {
            float_reg = lhs; int_reg = rhs;
        } else if (hir_type_is_subtype(rhs_type, t_float_exact) &&
                   hir_type_is_subtype(lhs_type, t_long_exact) &&
                   hir_type_has_object_spec(&lhs_type)) {
            float_reg = rhs; int_reg = lhs;
        }
        if (float_reg != NULL &&
            (op == HIR_BOP_Power || float_slot_method(op) != NULL)) {
            extern int jit_compile_running(void);
            if (jit_compile_running()) return NULL;
            HirType int_type = hir_register_type(int_reg);
            double dval = PyLong_AsDouble(hir_type_object_spec(&int_type));
            if (dval != -1.0 || !PyErr_Occurred()) {
                extern void jit_compile_lock(void);
                extern void jit_compile_unlock(void);
                jit_compile_lock();
                PyObject *float_obj = PyFloat_FromDouble(dval);
                jit_compile_unlock();
                if (float_obj != NULL) {
                    simplify_env_emit_use_type(env, float_reg, t_float_exact);
                    simplify_env_emit_use_type(env, int_reg, int_type);
                    extern PyObject *hir_func_add_reference(void *func, PyObject *obj);
                    PyObject *ref = hir_func_add_reference(env->func, float_obj);
                    Py_DECREF(float_obj);
                    void *float_const = simplify_env_emit_load_const(env, hir_type_from_object(ref));
                    return simplify_env_emit_float_binary_op_deopt(env, op,
                        (int_reg == lhs) ? float_const : lhs,
                        (int_reg == rhs) ? float_const : rhs, fs);
                }
            }
            PyErr_Clear();
        }
    }

    /* Unicode multiply */
    if (hir_type_is_subtype(lhs_type, t_unicode_exact) &&
        hir_type_is_subtype(rhs_type, t_long_exact) &&
        op == HIR_BOP_Multiply) {
        void *unboxed_rhs = simplify_env_emit_index_unbox(env, rhs, PyExc_OverflowError);
        simplify_env_emit_is_neg_and_err(env, unboxed_rhs, fs);
        return simplify_env_emit_unicode_repeat(env, lhs, unboxed_rhs, fs);
    }

    /* Unicode concat */
    if (hir_type_is_subtype(lhs_type, t_unicode_exact) &&
        hir_type_is_subtype(rhs_type, t_unicode_exact) &&
        op == HIR_BOP_Add) {
        return simplify_env_emit_unicode_concat(env, lhs, rhs, fs);
    }

    return NULL;
}

/* ==== LoadAttr emit helpers ==== */

static void *emit_load_type_attr_cache_entry_type(SimplifyEnv *env, int cache_id) {
    void *reg = hir_func_alloc_register(env->func);
    void *instr = hir_c_create_load_type_attr_cache_entry_type(reg, cache_id);
    return simplify_env_emit(env, instr);
}

static void *emit_load_type_attr_cache_entry_value(SimplifyEnv *env, int cache_id) {
    void *reg = hir_func_alloc_register(env->func);
    void *instr = hir_c_create_load_type_attr_cache_entry_value(reg, cache_id);
    return simplify_env_emit(env, instr);
}

extern void *hir_c_create_fill_type_attr_cache(void *func,
    void *receiver, int name_idx, int cache_id, void *fs);

static void *emit_fill_type_attr_cache(SimplifyEnv *env, void *receiver,
                                        int32_t name_idx, int32_t cache_id, void *fs) {
    void *instr = hir_c_create_fill_type_attr_cache(env->func, receiver,
                                                     name_idx, cache_id, fs);
    return simplify_env_emit(env, instr);
}

extern void *hir_c_create_load_attr_cached(void *func,
    void *receiver, int name_idx, void *fs);

static void *emit_load_attr_cached(SimplifyEnv *env, void *receiver,
                                    int32_t name_idx, void *fs) {
    void *instr = hir_c_create_load_attr_cached(env->func, receiver, name_idx, fs);
    return simplify_env_emit(env, instr);
}

extern void *hir_c_create_load_module_attr_cached(void *func,
    void *receiver, int name_idx, void *fs);

static void *emit_load_module_attr_cached(SimplifyEnv *env, void *receiver,
                                           int32_t name_idx, void *fs) {
    void *instr = hir_c_create_load_module_attr_cached(env->func, receiver, name_idx, fs);
    return simplify_env_emit(env, instr);
}

/* ==== simplifyLoadAttrTypeReceiver emitCond callbacks ==== */

typedef struct {
    int32_t cache_id;
} TypeAttrFastCtx;

typedef struct {
    int32_t cache_id;
    void *receiver;
    int32_t name_idx;
    void *fs;
} TypeAttrSlowCtx;

static void *type_attr_fast_path(SimplifyEnv *env, void *ctx) {
    TypeAttrFastCtx *c = (TypeAttrFastCtx *)ctx;
    return emit_load_type_attr_cache_entry_value(env, c->cache_id);
}

static void *type_attr_slow_path(SimplifyEnv *env, void *ctx) {
    TypeAttrSlowCtx *c = (TypeAttrSlowCtx *)ctx;
    return emit_fill_type_attr_cache(env, c->receiver, c->name_idx, c->cache_id, c->fs);
}

/* ==== simplifyLoadAttr ====
 * Partial port: handles type receiver (emitCond), module, and default cached
 * paths. Instance receiver (splitDict, memberDescr, property, genericDescr)
 * returns NULL to fall through to C++ simplifyLoadAttrInstanceReceiver.
 * Returns: register or NULL. Sets env->optimized if handled. */
/* ==== Type construction helpers ==== */
static inline HirType hir_type_from_cuint(uint64_t val, HirType base) {
    HirType t = base;
    t.bits_and_flags = (t.bits_and_flags & ~HIR_TYPE_SPEC_MASK) |
                       ((uint64_t)HIR_SPEC_INT << HIR_TYPE_SPEC_SHIFT);
    t.int_val = (intptr_t)val;
    return t;
}

static inline HirType hir_type_from_cptr(void *ptr) {
    HirType t = HIR_TYPE_CPTR;
    t.bits_and_flags = (t.bits_and_flags & ~HIR_TYPE_SPEC_MASK) |
                       ((uint64_t)HIR_SPEC_OBJECT << HIR_TYPE_SPEC_SHIFT);
    t.pyobject = (PyObject *)ptr;
    return t;
}

/* ==== simplifyLoadAttrSplitDict ==== */
#include "cinderx/Common/dict.h"

static void *simplify_load_attr_split_dict_c(SimplifyEnv *env, const void *instr,
                                              PyTypeObject *py_type, PyObject *attr_name) {
#if PY_VERSION_HEX >= 0x030C0000
    if (!PyType_HasFeature(py_type, Py_TPFLAGS_MANAGED_DICT))
        return NULL;
#else
    if (!PyType_HasFeature(py_type, Py_TPFLAGS_HEAPTYPE) ||
        py_type->tp_dictoffset < 0)
        return NULL;
#endif

    PyHeapTypeObject *ht = (PyHeapTypeObject *)py_type;
    if (ht->ht_cached_keys == NULL) return NULL;
    PyDictKeysObject *keys = ht->ht_cached_keys;
    Py_ssize_t attr_idx = getDictKeysIndex(keys, attr_name);
    if (attr_idx == -1) return NULL;

    void *receiver = hir_c_get_operand(instr, 0);
    HirType recv_type = hir_register_type(receiver);
    void *fs = hir_c_get_frame_state(instr);

    extern void *hir_func_allocate_split_dict_deopt_patcher(
        void *func, void *type, void *attr_name, void *keys);
    void *patcher = hir_func_allocate_split_dict_deopt_patcher(
        env->func, py_type, attr_name, keys);
    extern void *hir_c_create_deopt_patchpoint(void *patcher);
    void *pp = hir_c_create_deopt_patchpoint(patcher);
    simplify_env_emit(env, pp);
    extern void hir_c_set_guilty_reg(void *instr, void *reg);
    hir_c_set_guilty_reg(pp, receiver);
    extern void hir_c_set_descr(void *instr, const char *descr);
    hir_c_set_descr(pp, "SplitDictDeoptPatcher");
    simplify_env_emit_use_type(env, receiver, recv_type);

    HirType t_optobj = HIR_TYPE_OPTOBJECT;
#if PY_VERSION_HEX >= 0x030C0000
    void *obj_dict = simplify_env_emit_load_field(env, receiver, "__dict__",
        -3 * (intptr_t)sizeof(PyObject *), t_optobj, 0);
#else
    void *obj_dict = simplify_env_emit_load_field(env, receiver, "__dict__",
        py_type->tp_dictoffset, t_optobj, 0);
#endif

    extern void *hir_c_create_check_field(void *func, void *src,
        void *name, void *frame_state);
    void *check_dict_instr = hir_c_create_check_field(env->func, obj_dict,
        attr_name, fs);
    void *checked_dict = simplify_env_emit(env, check_dict_instr);
    hir_c_set_guilty_reg(check_dict_instr, receiver);

#if PY_VERSION_HEX >= 0x030C0000
    HirType t_cuint64 = HIR_TYPE_CUINT64;
    void *one = simplify_env_emit_load_const(env, hir_type_from_cuint(1, t_cuint64));
    void *dict_ptr_reg = hir_func_alloc_register(env->func);
    void *bit_cast1 = hir_c_create_bit_cast(dict_ptr_reg, checked_dict, t_cuint64);
    void *dict_ptr = simplify_env_emit(env, bit_cast1);

    void *is_values_reg = hir_func_alloc_register(env->func);
    void *and_instr = hir_c_create_int_binary_op(is_values_reg, HIR_BOP_And, dict_ptr, one);
    void *is_values = simplify_env_emit(env, and_instr);

    extern void *hir_c_create_guard(void *src);
    void *guard_instr = hir_c_create_guard(is_values);
    simplify_env_emit(env, guard_instr);
    hir_c_set_guilty_reg(guard_instr, receiver);
    hir_c_set_descr(guard_instr, "dict values check");

    void *values_reg = hir_func_alloc_register(env->func);
    void *add_instr = hir_c_create_int_binary_op(values_reg, HIR_BOP_Add, dict_ptr, one);
    void *values = simplify_env_emit(env, add_instr);

    void *values_obj_reg = hir_func_alloc_register(env->func);
    void *bit_cast2 = hir_c_create_bit_cast(values_obj_reg, values, t_optobj);
    void *values_obj = simplify_env_emit(env, bit_cast2);

    void *attr = simplify_env_emit_load_field(env, values_obj, "attr",
        attr_idx * (intptr_t)sizeof(PyObject *), t_optobj, 0);
#else
    HirType t_cptr = HIR_TYPE_CPTR;
    void *dict_keys = simplify_env_emit_load_field(env, checked_dict,
        "ma_keys", (intptr_t)offsetof(PyDictObject, ma_keys), t_cptr, 0);
    void *expected_keys = simplify_env_emit_load_const(env, hir_type_from_cptr(keys));
    void *equal = simplify_env_emit_primitive_compare(env, HIR_PCMP_Equal,
        dict_keys, expected_keys);

    extern void *hir_c_create_guard(void *src);
    void *guard_instr = hir_c_create_guard(equal);
    simplify_env_emit(env, guard_instr);
    hir_c_set_guilty_reg(guard_instr, receiver);
    hir_c_set_descr(guard_instr, "ht_cached_keys comparison");

    void *split_item_reg = hir_func_alloc_register(env->func);
    extern void *hir_c_create_load_split_dict_item(void *dst, void *src, intptr_t idx);
    void *split_item = hir_c_create_load_split_dict_item(split_item_reg,
        checked_dict, attr_idx);
    void *attr = simplify_env_emit(env, split_item);
#endif

    void *check_attr_instr = hir_c_create_check_field(env->func, attr,
        attr_name, fs);
    void *checked_attr = simplify_env_emit(env, check_attr_instr);
    hir_c_set_guilty_reg(check_attr_instr, receiver);

    return checked_attr;
}

/* ==== emitTypeAttrDeoptPatcher helper ==== */
static void emit_type_attr_deopt_patcher(SimplifyEnv *env,
        PyTypeObject *py_type, PyObject *attr_name, PyObject *descr,
        void *receiver, const char *description) {
    extern int _PyClassLoader_IsImmutable(PyObject *container);
    if (_PyClassLoader_IsImmutable((PyObject *)py_type)) return;

    extern void *hir_func_allocate_type_attr_deopt_patcher(
        void *func, void *type, void *attr_name, void *method);
    void *patcher = hir_func_allocate_type_attr_deopt_patcher(
        env->func, py_type, attr_name, descr);
    extern void *hir_c_create_deopt_patchpoint(void *patcher);
    void *pp = hir_c_create_deopt_patchpoint(patcher);
    simplify_env_emit(env, pp);
    extern void hir_c_set_guilty_reg(void *instr, void *reg);
    hir_c_set_guilty_reg(pp, receiver);
    extern void hir_c_set_descr(void *instr, const char *descr);
    hir_c_set_descr(pp, description);
}

/* ==== MemberDescr emitCond callbacks ==== */

static void *member_descr_field_set(SimplifyEnv *env, void *ctx) {
    void *field = ctx;
    void *reg = hir_func_alloc_register(env->func);
    HirType t_object = HIR_TYPE_OBJECT;
    void *instr = hir_c_create_refine_type(reg, t_object, field);
    return simplify_env_emit(env, instr);
}

static void *member_descr_field_null(SimplifyEnv *env, void *ctx) {
    (void)ctx;
    HirType t_nonetype = HIR_TYPE_NONETYPE;
    return simplify_env_emit_load_const(env, t_nonetype);
}

/* ==== simplifyLoadAttrMemberDescr ==== */
static void *simplify_load_attr_member_descr_c(SimplifyEnv *env,
        const void *instr, void *receiver, HirType recv_type,
        PyTypeObject *py_type, PyObject *attr_name, PyObject *descr, void *fs) {
    if (Py_TYPE(descr) != &PyMemberDescr_Type) return NULL;

    PyMemberDef *def = ((PyMemberDescrObject *)descr)->d_member;
    if (def->flags & READ_RESTRICTED) return NULL;

    if (def->type == T_OBJECT || def->type == T_OBJECT_EX) {
        const char *name_cstr = PyUnicode_AsUTF8(attr_name);
        if (name_cstr == NULL) {
            PyErr_Clear();
            name_cstr = "<unknown>";
        }
        emit_type_attr_deopt_patcher(env, py_type, attr_name, descr,
                                     receiver, "member descriptor attribute");
        simplify_env_emit_use_type(env, receiver, recv_type);
        HirType t_optobj = HIR_TYPE_OPTOBJECT;
        void *field = simplify_env_emit_load_field(env, receiver, name_cstr,
            def->offset, t_optobj, 0);

        if (def->type == T_OBJECT_EX) {
            extern void *hir_c_create_check_field(void *func, void *src,
                void *name, void *frame_state);
            void *check = hir_c_create_check_field(env->func, field, attr_name, fs);
            void *result = simplify_env_emit(env, check);
            extern void hir_c_set_guilty_reg(void *instr, void *reg);
            hir_c_set_guilty_reg(check, receiver);
            return result;
        }

        return simplify_emit_cond(env, field,
            member_descr_field_set, field,
            member_descr_field_null, NULL);
    }
    return NULL;
}

/* ==== simplifyLoadAttrProperty ==== */
#include "cinderx/Common/property.h"

static void *simplify_load_attr_property_c(SimplifyEnv *env,
        void *receiver, HirType recv_type, PyTypeObject *py_type,
        PyObject *attr_name, PyObject *descr, void *fs) {
    if (Py_TYPE(descr) != &PyProperty_Type) return NULL;

    Ci_propertyobject *prop = (Ci_propertyobject *)descr;
    PyObject *getter = prop->prop_get;
    if (getter == NULL) return NULL;

    emit_type_attr_deopt_patcher(env, py_type, attr_name, descr,
                                 receiver, "property attribute");
    simplify_env_emit_use_type(env, receiver, recv_type);

    extern PyObject *hir_func_add_reference(void *func, PyObject *obj);
    PyObject *ref = hir_func_add_reference(env->func, getter);
    void *getter_obj = simplify_env_emit_load_const(env, hir_type_from_object(ref));

    extern void *hir_c_create_vectorcall(void *func, size_t n_ops,
                                          uint32_t flags, void *fs);
    void *call = hir_c_create_vectorcall(env->func, 2, HIR_CALL_FLAG_NONE, fs);
    hir_c_set_operand(call, 0, getter_obj);
    hir_c_set_operand(call, 1, receiver);
    return simplify_env_emit(env, call);
}

/* ==== simplifyLoadAttrGenericDescriptor ==== */
static void *simplify_load_attr_generic_descr_c(SimplifyEnv *env,
        void *receiver, HirType recv_type, PyTypeObject *py_type,
        PyObject *attr_name, PyObject *descr, void *fs) {
    PyTypeObject *descr_type = Py_TYPE(descr);
    descrgetfunc descr_get = descr_type->tp_descr_get;
    descrsetfunc descr_set = descr_type->tp_descr_set;
    if (descr_get == NULL || descr_set == NULL) return NULL;

    emit_type_attr_deopt_patcher(env, py_type, attr_name, descr,
                                 receiver, "generic descriptor attribute");

    extern int _PyClassLoader_IsImmutable(PyObject *container);
    if (!_PyClassLoader_IsImmutable((PyObject *)descr_type)) {
        extern void *hir_func_allocate_type_deopt_patcher(void *func, void *type);
        void *patcher = hir_func_allocate_type_deopt_patcher(env->func, descr_type);
        extern void *hir_c_create_deopt_patchpoint(void *patcher);
        void *pp = hir_c_create_deopt_patchpoint(patcher);
        simplify_env_emit(env, pp);
        extern void hir_c_set_guilty_reg(void *instr, void *reg);
        hir_c_set_guilty_reg(pp, receiver);
        extern void hir_c_set_descr(void *instr, const char *d);
        hir_c_set_descr(pp, "tp_descr_get/tp_descr_set");
    }

    simplify_env_emit_use_type(env, receiver, recv_type);

    extern PyObject *hir_func_add_reference(void *func, PyObject *obj);
    PyObject *descr_ref = hir_func_add_reference(env->func, descr);
    void *descr_reg = simplify_env_emit_load_const(env, hir_type_from_object(descr_ref));

    PyObject *type_ref = hir_func_add_reference(env->func, (PyObject *)py_type);
    void *type_reg = simplify_env_emit_load_const(env, hir_type_from_object(type_ref));

    HirType t_optobj = HIR_TYPE_OPTOBJECT;
    extern void *hir_c_create_call_static(void *func, size_t n_ops,
                                           void *addr, HirType ret_type);
    void *call = hir_c_create_call_static(env->func, 3,
                                           (void *)descr_get, t_optobj);
    hir_c_set_operand(call, 0, descr_reg);
    hir_c_set_operand(call, 1, receiver);
    hir_c_set_operand(call, 2, type_reg);
    void *call_out = simplify_env_emit(env, call);

    extern void *hir_c_create_check_exc(void *func, void *src, void *fs);
    void *check = hir_c_create_check_exc(env->func, call_out, fs);
    return simplify_env_emit(env, check);
}

/* ==== simplifyLoadAttrInstanceReceiver ==== */
static void *simplify_load_attr_instance_c(SimplifyEnv *env, const void *instr) {
    void *receiver = hir_c_get_operand(instr, 0);
    HirType recv_type = hir_register_type(receiver);
    PyTypeObject *py_type = hir_type_runtime_py_type(&recv_type);

    if (!hir_type_is_exact(&recv_type) || py_type == NULL ||
        !PyType_HasFeature(py_type, Py_TPFLAGS_READY) ||
        py_type->tp_getattro != PyObject_GenericGetAttr)
        return NULL;

    extern int jit_compile_running(void);
    if (jit_compile_running()) {
        if (!Ci_Type_HasValidVersionTag(py_type)) return NULL;
    } else {
        extern int jit_ensure_version_tag(PyTypeObject *type);
        if (!jit_ensure_version_tag(py_type)) return NULL;
    }

    int32_t name_idx = ((const HirLoadAttr *)instr)->name_idx;
    void *fs = hir_c_get_frame_state(instr);
    extern PyObject *hir_frame_state_get_name(void *fs, int name_idx);
    PyObject *attr_name = hir_frame_state_get_name(fs, name_idx);
    if (!PyUnicode_CheckExact(attr_name)) return NULL;

    extern PyObject *jit_type_lookup_safe(PyTypeObject *type, PyObject *name);
    PyObject *descr = jit_type_lookup_safe(py_type, attr_name);
    if (descr == NULL) {
        return simplify_load_attr_split_dict_c(env, instr, py_type, attr_name);
    }

    /* Try descriptor handlers in order */
    void *result = simplify_load_attr_member_descr_c(env, instr, receiver,
        recv_type, py_type, attr_name, descr, fs);
    if (result) return result;

    result = simplify_load_attr_property_c(env, receiver, recv_type,
        py_type, attr_name, descr, fs);
    if (result) return result;

    result = simplify_load_attr_generic_descr_c(env, receiver, recv_type,
        py_type, attr_name, descr, fs);
    if (result) return result;

    return NULL;
}

/* Called after C++ has already checked alreadyOptimized.
 * Handles instance receiver (SplitDict), type receiver, module receiver.
 * Returns NULL if not handled (caller falls through to LoadAttrCached). */
void *simplify_load_attr_c(SimplifyEnv *env, const void *instr) {
    const HirLoadAttr *la = (const HirLoadAttr *)instr;
    if (la->already_optimized) return NULL;

    /* Try instance receiver first (SplitDict path) */
    void *result = simplify_load_attr_instance_c(env, instr);
    if (result) return result;

    if (!jit_get_config()->attr_caches) return NULL;

    void *receiver = hir_c_get_operand(instr, 0);
    HirType recv_type = hir_register_type(receiver);
    int32_t name_idx = ((const HirLoadAttr *)instr)->name_idx;
    void *fs = hir_c_get_frame_state(instr);

    /* Module receiver: module attr cached */
    extern PyTypeObject Ci_StrictModule_Type;
    PyTypeObject *pytype = hir_type_runtime_py_type(&recv_type);
    if (pytype == &PyModule_Type || pytype == &Ci_StrictModule_Type) {
        return emit_load_module_attr_cached(env, receiver, name_idx, fs);
    }

    /* Type receiver: type attr cache with emitCond */
    HirType t_type = HIR_TYPE_TYPE;
    if (hir_type_is_subtype(recv_type, t_type)) {
        int cache_id = hir_func_env_allocate_type_attr_cache(env->func);
        simplify_env_emit_use_type(env, receiver, t_type);
        void *guard = emit_load_type_attr_cache_entry_type(env, cache_id);
        void *type_matches = emit_primitive_compare_eq(env, guard, receiver);

        TypeAttrFastCtx fast_ctx = {cache_id};
        TypeAttrSlowCtx slow_ctx = {cache_id, receiver, name_idx, fs};

        return simplify_emit_cond(env, type_matches,
                                  type_attr_fast_path, &fast_ctx,
                                  type_attr_slow_path, &slow_ctx);
    }

    /* Default: LoadAttrCached */
    return emit_load_attr_cached(env, receiver, name_idx, fs);
}

/* ==== simplifyCallMethod ====
 * Path 1: func()->type() <= TNullptr → rewrite as VectorCall.
 * Path 2: __exit__/__aexit__ bound method resolution (complex, needs deep bridges).
 * Currently only Path 1 is ported. Path 2 returns NULL to fall through to C++. */
void *simplify_call_method_c(SimplifyEnv *env, const void *instr) {
    const HirCallMethod *cm = (const HirCallMethod *)instr;
    size_t n_operands = hir_c_num_operands(instr);
    void *fs = hir_c_get_frame_state(instr);

    /* Path 1: func is known to be NULL (self was None) → rewrite as VectorCall */
    {
        void *func_reg = hir_c_get_operand(instr, 0);
        HirType func_type = hir_register_type(func_reg);
        HirType t_nullptr = HIR_TYPE_NULLPTR;
        if (hir_type_is_subtype(func_type, t_nullptr)) {
            extern void *hir_c_create_vectorcall(void *func, size_t n_ops,
                                                  uint32_t flags, void *fs);
            void *call = hir_c_create_vectorcall(env->func, n_operands - 1,
                                                  cm->flags, fs);
            hir_c_set_suppress_exc_deopt(call, cm->suppress_exception_deopt);

            for (size_t i = 1; i < n_operands; ++i) {
                hir_c_set_operand(call, i - 1, hir_c_get_operand(instr, i));
            }

            void *out = simplify_env_emit(env, call);

            void *callable = hir_c_get_operand(call, 0);
            HirType callable_type = hir_register_type(callable);
            if (hir_type_has_object_spec(&callable_type)) {
                PyObject *callable_obj = hir_type_object_spec(&callable_type);
                if (PyType_Check(callable_obj)) {
                    PyTypeObject *cls = (PyTypeObject *)callable_obj;
                    if (cls->tp_new == PyBaseObject_Type.tp_new) {
                        HirType exact_type = hir_type_from_pytype(cls, 1);
                        hir_reg_set_type(out, exact_type);
                    }
                }
            }

            return out;
        }
    }

    /* Path 2: __exit__/__aexit__ bound method resolution */
#if PY_VERSION_HEX >= 0x030C0000
    {
        void *func_reg = hir_c_get_operand(instr, 0);
        extern void *hir_reg_instr(void *reg);
        void *func_def = hir_reg_instr(func_reg);
        if (func_def == NULL || hir_c_opcode(func_def) != HIR_OP_LoadAttrSpecial)
            return NULL;

        const HirLoadAttrSpecial *las = (const HirLoadAttrSpecial *)func_def;
        PyObject *attr_id = las->id;

        if (attr_id != &_Py_ID(__exit__) && attr_id != &_Py_ID(__aexit__))
            return NULL;

        void *receiver = hir_c_get_operand(func_def, 0);
        HirType recv_type = hir_register_type(receiver);
        PyTypeObject *py_type = hir_type_runtime_py_type(&recv_type);

        if (!hir_type_is_exact(&recv_type) || py_type == NULL ||
            !PyType_HasFeature(py_type, Py_TPFLAGS_READY))
            return NULL;

        extern int jit_compile_running(void);
        int version_ok;
        if (jit_compile_running()) {
            version_ok = Ci_Type_HasValidVersionTag(py_type);
        } else {
            extern int jit_ensure_version_tag(PyTypeObject *type);
            version_ok = jit_ensure_version_tag(py_type);
        }
        if (!version_ok) return NULL;

        extern PyObject *jit_type_lookup_safe(PyTypeObject *type, PyObject *name);
        PyObject *method = jit_type_lookup_safe(py_type, attr_id);
        if (method == NULL || !PyFunction_Check(method))
            return NULL;

        if (!jit_compile_running()) {
            extern void hir_preloader_ensure(void *py_func);
            hir_preloader_ensure(method);
        }

        /* Emit snapshot for deopt patchpoint */
        extern void *hir_c_create_snapshot(void *fs);
        void *snapshot = hir_c_create_snapshot(fs);
        simplify_env_emit(env, snapshot);

        extern int _PyClassLoader_IsImmutable(PyObject *container);
        if (!_PyClassLoader_IsImmutable((PyObject *)py_type)) {
            extern void *hir_func_allocate_type_attr_deopt_patcher(
                void *func, void *type, void *attr_name, void *method);
            void *patcher = hir_func_allocate_type_attr_deopt_patcher(
                env->func, py_type, attr_id, method);
            extern void *hir_c_create_deopt_patchpoint(void *patcher);
            void *pp = hir_c_create_deopt_patchpoint(patcher);
            simplify_env_emit(env, pp);
            extern void hir_c_set_guilty_reg(void *instr, void *reg);
            hir_c_set_guilty_reg(pp, receiver);
            extern void hir_c_set_descr(void *instr, const char *descr);
            hir_c_set_descr(pp, "CallMethod __exit__ method resolution");
        }

        HirType recv_unspec = hir_type_unspecialized(&recv_type);
        simplify_env_emit_use_type(env, receiver, recv_unspec);

        extern PyObject *hir_func_add_reference(void *func, PyObject *obj);
        PyObject *ref = hir_func_add_reference(env->func, method);
        void *func_const = simplify_env_emit_load_const(env, hir_type_from_object(ref));

        size_t cm_noperands = hir_c_num_operands(instr);
        extern void *hir_c_create_vectorcall(void *func, size_t n_ops,
                                              uint32_t flags, void *fs);
        void *new_call = hir_c_create_vectorcall(env->func,
            cm_noperands + 1, HIR_CALL_FLAG_STATIC, fs);
        hir_c_set_suppress_exc_deopt(new_call, cm->suppress_exception_deopt);
        hir_c_set_operand(new_call, 0, func_const);
        hir_c_set_operand(new_call, 1, receiver);
        for (size_t i = 1; i < cm_noperands; ++i) {
            hir_c_set_operand(new_call, i + 1, hir_c_get_operand(instr, i));
        }

        return simplify_env_emit(env, new_call);
    }
#endif

    return NULL;
}

/* ==== VectorCall Phase 1: static/bound/global sub-handlers ==== */

/* trySpecializeCCall: specialize PyMethodDescr METH_NOARGS/METH_O to CallStatic */
static void *try_specialize_ccall_c(SimplifyEnv *env, const void *instr) {
    const HirVectorCall *vc = (const HirVectorCall *)instr;
    if (vc->flags & HIR_CALL_FLAG_AWAITED) return NULL;

    void *callable = hir_c_get_operand(instr, 0);
    HirType callable_type = hir_register_type(callable);
    PyObject *callable_obj = hir_type_as_object(&callable_type);
    if (callable_obj == NULL) return NULL;

    if (Py_TYPE(callable_obj) == &PyMethodDescr_Type) {
        PyMethodDescrObject *meth = (PyMethodDescrObject *)callable_obj;
        PyMethodDef *def = meth->d_method;
        size_t n_args = hir_c_num_operands(instr) - 1;
        /* Use original output type | TNullptr for CallStatic return type */
        void *orig_out = hir_c_output(instr);
        HirType orig_type = hir_register_type(orig_out);
        HirType t_nullptr = HIR_TYPE_NULLPTR;
        orig_type.bits_and_flags |= hir_type_bits(&t_nullptr);
        HirType ret_type = orig_type;

        if ((def->ml_flags & METH_NOARGS) && n_args == 1) {
            extern void *hir_c_create_call_static(void *func, size_t n_ops,
                                                   void *addr, HirType ret_type);
            void *call = hir_c_create_call_static(env->func, 1,
                (void *)def->ml_meth, ret_type);
            hir_c_set_operand(call, 0, hir_c_get_operand(instr, 1));
            void *call_out = simplify_env_emit(env, call);
            extern void *hir_c_create_check_exc(void *func, void *src, void *fs);
            void *fs = hir_c_get_frame_state(instr);
            void *check = hir_c_create_check_exc(env->func, call_out, fs);
            return simplify_env_emit(env, check);
        }
        if ((def->ml_flags & METH_O) && n_args == 2) {
            extern void *hir_c_create_call_static(void *func, size_t n_ops,
                                                   void *addr, HirType ret_type);
            void *call = hir_c_create_call_static(env->func, 2,
                (void *)def->ml_meth, ret_type);
            hir_c_set_operand(call, 0, hir_c_get_operand(instr, 1));
            hir_c_set_operand(call, 1, hir_c_get_operand(instr, 2));
            void *call_out = simplify_env_emit(env, call);
            extern void *hir_c_create_check_exc(void *func, void *src, void *fs);
            void *fs = hir_c_get_frame_state(instr);
            void *check = hir_c_create_check_exc(env->func, call_out, fs);
            return simplify_env_emit(env, check);
        }
    }
    return NULL;
}

/* simplifyVectorCallStatic: list.append or trySpecializeCCall */
static void *simplify_vectorcall_static_c(SimplifyEnv *env, const void *instr) {
    const HirVectorCall *vc = (const HirVectorCall *)instr;
    if (!(vc->flags & HIR_CALL_FLAG_STATIC)) return NULL;

    void *func = hir_c_get_operand(instr, 0);
    if (is_builtin_c(func, "list.append") && hir_c_num_operands(instr) - 1 == 2) {
        HirType func_type = hir_register_type(func);
        simplify_env_emit_use_type(env, func, func_type);
        extern void *hir_c_create_list_append(void *func_h, void *list,
                                               void *item, void *fs);
        void *fs = hir_c_get_frame_state(instr);
        void *la = hir_c_create_list_append(env->func,
            hir_c_get_operand(instr, 1), hir_c_get_operand(instr, 2), fs);
        simplify_env_emit(env, la);
        HirType t_nonetype = HIR_TYPE_NONETYPE;
        return simplify_env_emit_load_const(env, t_nonetype);
    }

    return try_specialize_ccall_c(env, instr);
}

/* simplifyVectorCallBoundMethod: resolve LoadAttrSpecial bound methods */
static void *simplify_vectorcall_bound_method_c(SimplifyEnv *env, const void *instr) {
#if PY_VERSION_HEX < 0x030C0000
    return NULL;
#else
    const HirVectorCall *vc = (const HirVectorCall *)instr;
    if (vc->flags & (HIR_CALL_FLAG_KWARGS | HIR_CALL_FLAG_STATIC | HIR_CALL_FLAG_AWAITED))
        return NULL;

    void *func_reg = hir_c_get_operand(instr, 0);
    extern void *hir_reg_instr(void *reg);
    void *func_def = hir_reg_instr(func_reg);
    if (func_def == NULL || hir_c_opcode(func_def) != HIR_OP_LoadAttrSpecial)
        return NULL;

    const HirLoadAttrSpecial *las = (const HirLoadAttrSpecial *)func_def;
    PyObject *attr_id = las->id;
    if (attr_id != &_Py_ID(__enter__) && attr_id != &_Py_ID(__aenter__) &&
        attr_id != &_Py_ID(__exit__) && attr_id != &_Py_ID(__aexit__))
        return NULL;

    void *receiver = hir_c_get_operand(func_def, 0);
    HirType recv_type = hir_register_type(receiver);
    PyTypeObject *py_type = hir_type_runtime_py_type(&recv_type);
    if (!hir_type_is_exact(&recv_type) || py_type == NULL ||
        !PyType_HasFeature(py_type, Py_TPFLAGS_READY))
        return NULL;

    extern int jit_compile_running(void);
    if (jit_compile_running()) {
        if (!Ci_Type_HasValidVersionTag(py_type)) return NULL;
    } else {
        extern int jit_ensure_version_tag(PyTypeObject *type);
        if (!jit_ensure_version_tag(py_type)) return NULL;
    }

    extern PyObject *jit_type_lookup_safe(PyTypeObject *type, PyObject *name);
    PyObject *method = jit_type_lookup_safe(py_type, attr_id);
    if (method == NULL || !PyFunction_Check(method)) return NULL;

    if (!jit_compile_running()) {
        extern void hir_preloader_ensure(void *py_func);
        hir_preloader_ensure(method);
    }

    void *fs = hir_c_get_frame_state(instr);
    extern void *hir_c_create_snapshot(void *fs);
    void *snapshot = hir_c_create_snapshot(fs);
    simplify_env_emit(env, snapshot);

    extern int _PyClassLoader_IsImmutable(PyObject *container);
    if (!_PyClassLoader_IsImmutable((PyObject *)py_type)) {
        extern void *hir_func_allocate_type_attr_deopt_patcher(
            void *func, void *type, void *attr_name, void *method);
        void *patcher = hir_func_allocate_type_attr_deopt_patcher(
            env->func, py_type, attr_id, method);
        extern void *hir_c_create_deopt_patchpoint(void *patcher);
        void *pp = hir_c_create_deopt_patchpoint(patcher);
        simplify_env_emit(env, pp);
        extern void hir_c_set_guilty_reg(void *instr, void *reg);
        hir_c_set_guilty_reg(pp, receiver);
        extern void hir_c_set_descr(void *instr, const char *descr);
        hir_c_set_descr(pp, "LoadAttrSpecial method resolution");
    }

    HirType recv_unspec = hir_type_unspecialized(&recv_type);
    simplify_env_emit_use_type(env, receiver, recv_unspec);

    extern PyObject *hir_func_add_reference(void *func, PyObject *obj);
    PyObject *ref = hir_func_add_reference(env->func, method);
    void *func_const = simplify_env_emit_load_const(env, hir_type_from_object(ref));

    size_t orig_nargs = hir_c_num_operands(instr) - 1;
    extern void *hir_c_create_vectorcall(void *func, size_t n_ops,
                                          uint32_t flags, void *fs);
    void *new_call = hir_c_create_vectorcall(env->func,
        2 + orig_nargs, vc->flags | HIR_CALL_FLAG_STATIC, fs);
    hir_c_set_operand(new_call, 0, func_const);
    hir_c_set_operand(new_call, 1, receiver);
    for (size_t i = 0; i < orig_nargs; ++i) {
        hir_c_set_operand(new_call, 2 + i, hir_c_get_operand(instr, 1 + i));
    }
    return simplify_env_emit(env, new_call);
#endif
}

/* simplifyVectorCallGlobal: GuardIs → GlobalDeoptPatcher → static call */
static void *simplify_vectorcall_global_c(SimplifyEnv *env, const void *instr) {
    const HirVectorCall *vc = (const HirVectorCall *)instr;
    if (vc->flags & (HIR_CALL_FLAG_KWARGS | HIR_CALL_FLAG_STATIC | HIR_CALL_FLAG_AWAITED))
        return NULL;

    void *func_reg = hir_c_get_operand(instr, 0);
    extern void *hir_reg_instr(void *reg);
    void *func_def = hir_reg_instr(func_reg);
    if (func_def == NULL || hir_c_opcode(func_def) != HIR_OP_GuardIs)
        return NULL;

    const HirGuardIs *guard_is = (const HirGuardIs *)func_def;
    PyObject *expected = guard_is->target;
    if (!PyFunction_Check(expected)) return NULL;

    void *guarded_input = hir_c_get_operand(func_def, 0);
    extern void *hir_reg_instr(void *reg);
    void *input_def = hir_reg_instr(guarded_input);
    if (input_def == NULL || hir_c_opcode(input_def) != HIR_OP_LoadGlobalCached)
        return NULL;

    const HirLoadGlobalCached *lg = (const HirLoadGlobalCached *)input_def;
    PyObject *name = PyTuple_GET_ITEM(((PyCodeObject *)lg->code)->co_names, lg->name_idx);
    if (!PyUnicode_CheckExact(name)) return NULL;

    extern int jit_compile_running(void);
    if (jit_compile_running()) return NULL;

    void *fs = hir_c_get_frame_state(instr);
    extern void *hir_c_create_snapshot(void *fs);
    void *snapshot = hir_c_create_snapshot(fs);
    simplify_env_emit(env, snapshot);

    extern void *hir_func_allocate_global_deopt_patcher(
        void *func, void *globals, void *key_name, void *expected);
    void *patcher = hir_func_allocate_global_deopt_patcher(
        env->func, lg->globals, name, expected);
    extern void *hir_c_create_deopt_patchpoint(void *patcher);
    void *pp = hir_c_create_deopt_patchpoint(patcher);
    simplify_env_emit(env, pp);
    extern void hir_c_set_guilty_reg(void *instr, void *reg);
    hir_c_set_guilty_reg(pp, func_reg);
    extern void hir_c_set_descr(void *instr, const char *descr);
    hir_c_set_descr(pp, "Global callee guard elimination");

    extern PyObject *hir_func_add_reference(void *func, PyObject *obj);
    PyObject *ref = hir_func_add_reference(env->func, expected);
    void *func_const = simplify_env_emit_load_const(env, hir_type_from_object(ref));

    size_t orig_nargs = hir_c_num_operands(instr) - 1;
    extern void *hir_c_create_vectorcall(void *func, size_t n_ops,
                                          uint32_t flags, void *fs);
    void *new_call = hir_c_create_vectorcall(env->func,
        1 + orig_nargs, vc->flags | HIR_CALL_FLAG_STATIC, fs);
    hir_c_set_operand(new_call, 0, func_const);
    for (size_t i = 0; i < orig_nargs; ++i) {
        hir_c_set_operand(new_call, 1 + i, hir_c_get_operand(instr, 1 + i));
    }
    return simplify_env_emit(env, new_call);
}

/* ==== simplifyVectorCall ==== */
void *simplify_vectorcall_c(SimplifyEnv *env, const void *instr) {
    /* Try static sub-handler */
    void *result = simplify_vectorcall_static_c(env, instr);
    if (result) return result;

    /* Try bound method sub-handler */
    result = simplify_vectorcall_bound_method_c(env, instr);
    if (result) return result;

    /* Try global sub-handler */
    result = simplify_vectorcall_global_c(env, instr);
    if (result) return result;

    /* BoundMethod and Global sub-handlers — return NULL to fall through to C++ */

    const HirVectorCall *vc = (const HirVectorCall *)instr;
    if (vc->flags & HIR_CALL_FLAG_KWARGS) return NULL;

    void *target = hir_c_get_operand(instr, 0);
    size_t n_operands = hir_c_num_operands(instr);

    /* type(x) → LoadField(x, ob_type) */
    HirType target_type = hir_register_type(target);
    HirType t_type = HIR_TYPE_TYPE;
    if (hir_type_is_subtype(target_type, t_type) &&
        hir_type_has_object_spec(&target_type) &&
        hir_type_object_spec(&target_type) == (PyObject *)&PyType_Type &&
        n_operands == 2) {
        simplify_env_emit_use_type(env, target, target_type);
        return simplify_env_emit_load_field(env, hir_c_get_operand(instr, 1),
            "ob_type", (intptr_t)offsetof(PyObject, ob_type), t_type, 0);
    }

    /* len(x) → GetLength */
    if (is_builtin_c(target, "len") && n_operands - 1 == 1) {
        HirType func_type = hir_register_type(target);
        simplify_env_emit_use_type(env, target, func_type);
        extern void *hir_c_create_get_length(void *func, void *src, void *fs);
        void *fs = hir_c_get_frame_state(instr);
        void *gl = hir_c_create_get_length(env->func, hir_c_get_operand(instr, 1), fs);
        return simplify_env_emit(env, gl);
    }

    return NULL;
}
