/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Pure C simplify handlers — incremental port of simplify.cpp.
 */

#include "cinderx/Jit/hir/simplify_c.h"
#include "cinderx/Jit/hir/hir_instr_c.h"
#include "cinderx/Jit/hir/hir_type_c.h"
#include "Python.h"
#include "pycore_long.h"

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

    return NULL;
}

/* ---- simplifyPrimitiveCompare (partial — box(b)==True → b) ---- */
void *simplify_primitive_compare_box_true_c(const void *instr) {
    void *left = hir_c_get_operand(instr, 0);
    void *right = hir_c_get_operand(instr, 1);
    int32_t op = hir_c_compare_op(instr);
    if (op != HIR_PCMP_Equal) return NULL;

    extern void *hir_reg_instr(void *reg);
    void *left_def = hir_reg_instr(left);
    if (left_def == NULL || hir_c_opcode(left_def) != HIR_OP_PrimitiveBoxBool)
        return NULL;

    HirType right_type = hir_register_type(right);
    PyObject *right_obj = hir_type_as_object(&right_type);
    if (right_obj != Py_True) return NULL;

    return hir_c_get_operand(left_def, 0);
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

/* ---- simplifyUnbox (partial — unbox(box(x)) → x) ---- */
void *simplify_unbox_box_c(const void *instr) {
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
