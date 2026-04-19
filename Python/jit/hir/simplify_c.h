/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C simplify handler declarations — incremental port of simplify.cpp.
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/* Returns replacement Register* or NULL if not optimizable. */
void *simplify_check_c(const void *instr);

#ifdef __cplusplus
}
#endif
