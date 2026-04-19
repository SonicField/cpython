/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Pure C simplify handlers — incremental port of simplify.cpp.
 */

#include "cinderx/Jit/hir/simplify_c.h"
#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/hir/hir_type_c.h"
#include "Python.h"

/* Forward declarations (avoid hir_c_api.h typedef conflicts) */
extern HirType hir_register_type(void *reg);
extern void *hir_func_alloc_register(void *func);
extern void hir_c_insert_before(void *new_instr, void *before);
extern HirType hir_output_type(void *instr);

/* ---- SimplifyEnv: C equivalent of the C++ Env struct ---- */

void *simplify_env_emit(SimplifyEnv *env, void *new_instr) {
    env->optimized = 1;
    hir_c_set_bytecode_offset(new_instr, env->bc_off);
    hir_c_insert_before(new_instr, env->cursor_instr);
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

/* ---- simplifyGuardType (partial — type-already-matches path) ----
 * If input already has the guarded type, GuardType is redundant. */
void *simplify_guard_type_identity_c(const void *instr) {
    void *input = hir_c_get_operand(instr, 0);
    HirType input_type = hir_register_type(input);
    HirType target = hir_c_guard_type_target(instr);
    if (hir_type_is_subtype(input_type, target)) {
        return input;
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
    extern void *hir_c_create_primitive_box(void *dst, void *src, HirType type);
    HirType t_cbool = HIR_TYPE_CBOOL;
    void *instr = hir_c_create_primitive_box(reg, src, t_cbool);
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

    HirType t_none = HIR_TYPE_SIMPLE(0x00000000080ULL, HIR_TYPE_LIFETIME_TOP);

    /* None == None or None != None */
    if (hir_type_is_subtype(left_type, t_none) &&
        hir_type_is_subtype(right_type, t_none)) {
        /* CompareOp::kEqual = 2, kNotEqual = 3 */
        if (op == 2 || op == 3) {
            simplify_env_emit_use_type(env, left, t_none);
            simplify_env_emit_use_type(env, right, t_none);
            PyObject *result = (op == 2) ? Py_True : Py_False;
            return simplify_env_emit_load_const(env, hir_type_from_object(result));
        }
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

    HirType t_tuple_exact = HIR_TYPE_SIMPLE(0x00000040000ULL, HIR_TYPE_LIFETIME_TOP);
    HirType t_bytes_exact = HIR_TYPE_SIMPLE(0x00000002000ULL, HIR_TYPE_LIFETIME_TOP);

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
