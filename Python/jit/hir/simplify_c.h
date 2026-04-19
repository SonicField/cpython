/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C simplify handler declarations — incremental port of simplify.cpp.
 */
#pragma once

#include "cinderx/Jit/hir/hir_type_c.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* C equivalent of the C++ Env struct (subset needed by C handlers). */
typedef struct {
    void *func;           /* HirFunction */
    void *block;          /* current BasicBlock* */
    void *cursor_instr;   /* instruction being optimized (insert before) */
    int32_t bc_off;       /* bytecode offset for new instructions */
    int optimized;        /* set to 1 when emit is called */
} SimplifyEnv;

/* Emit helpers */
void *simplify_env_emit(SimplifyEnv *env, void *new_instr);
void *simplify_env_emit_load_const(SimplifyEnv *env, HirType type);
void *simplify_env_emit_use_type(SimplifyEnv *env, void *val, HirType type);

/* Env-free handlers (return existing register or NULL) */
void *simplify_check_c(const void *instr);
void *simplify_refine_type_c(const void *instr);
void *simplify_guard_type_identity_c(const void *instr);
void *simplify_primitive_compare_box_true_c(const void *instr);
void *simplify_unbox_box_c(const void *instr);
void *simplify_cond_branch_check_type_c(SimplifyEnv *env, const void *instr);
void *simplify_load_array_item_tuple_c(SimplifyEnv *env, const void *instr);
void *simplify_load_tuple_item_c(SimplifyEnv *env, const void *instr);
void *simplify_load_field_float_c(SimplifyEnv *env, const void *instr);
void *simplify_int_convert_c(SimplifyEnv *env, const void *instr);

/* Env-using handlers (Category 2) */
void *simplify_primitive_box_bool_c(SimplifyEnv *env, const void *instr);
void *simplify_cint_to_cbool_c(SimplifyEnv *env, const void *instr);
void *simplify_cond_branch_const_c(SimplifyEnv *env, const void *instr);
void *simplify_compare_c(SimplifyEnv *env, const void *instr);
void *simplify_is_neg_and_err_c(SimplifyEnv *env, const void *instr);
void *simplify_load_var_object_size_c(SimplifyEnv *env, const void *instr);

#ifdef __cplusplus
}
#endif
