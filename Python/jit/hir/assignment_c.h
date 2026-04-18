/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C AssignmentAnalysis — forward dataflow for definite/maybe assignment.
 */
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* HirFunction;
typedef void* HirRegister;

typedef struct PhxAssignmentState PhxAssignmentState;

/* Create an assignment analysis (definite or maybe).
 * is_definite=1: definite assignment (AND meet, all paths must assign).
 * is_definite=0: maybe assignment (OR meet, any path assigns). */
PhxAssignmentState *phx_assign_create(HirFunction func, int is_definite);

/* Check if register is assigned at entry to block (by block ID). */
int phx_assign_is_assigned_in(const PhxAssignmentState *state,
                              int block_id, HirRegister reg);

/* Check if register is assigned at exit of block (by block ID). */
int phx_assign_is_assigned_out(const PhxAssignmentState *state,
                               int block_id, HirRegister reg);

/* Differential verification against C++ AssignmentAnalysis. */
int phx_assign_verify(HirFunction func, const PhxAssignmentState *c_state,
                      int is_definite);

/* Free the assignment analysis state. */
void phx_assign_destroy(PhxAssignmentState *state);

#ifdef __cplusplus
} /* extern "C" */
#endif
