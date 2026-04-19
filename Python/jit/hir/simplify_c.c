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

/* ---- SimplifyEnv: C equivalent of the C++ Env struct ---- */

void *simplify_env_emit(SimplifyEnv *env, void *new_instr) {
    env->optimized = 1;
    hir_c_set_bytecode_offset(new_instr, env->bc_off);
    hir_c_insert_before(new_instr, env->cursor_instr);
    return hir_c_output(new_instr);
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

/* Helper: create HirType for CBool(val) with int specialization */
static HirType make_cbool_type(intptr_t val) {
    HirType t = HIR_TYPE_CBOOL;
    t.bits_and_flags |= ((uint64_t)HIR_SPEC_INT << HIR_TYPE_SPEC_SHIFT);
    t.int_val = val != 0 ? 1 : 0;
    return t;
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
