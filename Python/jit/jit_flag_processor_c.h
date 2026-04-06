/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C-compatible JIT flag processor — table-driven option parsing.
 * Replaces C++ FlagProcessor with lambdas/std::function.
 */
#pragma once

#include "cinderx/python.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Option value types */
typedef enum {
    JIT_OPT_BOOL,      /* target is int* (set to 0 or 1) */
    JIT_OPT_INT,       /* target is int* */
    JIT_OPT_SIZE_T,    /* target is size_t* */
    JIT_OPT_UINT32,    /* target is uint32_t* */
    JIT_OPT_STRING,    /* target is char* (fixed buffer, max 256) */
    JIT_OPT_CALLBACK   /* uses callback function pointer */
} JitOptType;

/* Callback for JIT_OPT_CALLBACK options */
typedef void (*jit_opt_callback_fn)(const char* value, void* ctx);

/* Single option descriptor */
typedef struct {
    const char* cmdline_flag;       /* e.g. "jit-enable" */
    const char* env_var;            /* e.g. "PYTHONJITENABLE" */
    const char* description;        /* human-readable help text */
    JitOptType type;
    void* target;                   /* pointer to variable to bind */
    jit_opt_callback_fn callback;   /* for JIT_OPT_CALLBACK only */
    void* callback_ctx;             /* context for callback */
    const char* param_name;         /* optional, for help formatting */
    const char* debug_msg;          /* optional override for debug message */
    int hidden;                     /* if true, hide from help */
    int handled;                    /* set to 1 when matched */
} JitOption;

/* Flag processor: owns an array of options */
typedef struct {
    JitOption* options;
    size_t count;
    size_t capacity;
} JitFlagProcessor;

/* Initialize/free */
void jit_flagproc_init(JitFlagProcessor* fp);
void jit_flagproc_free(JitFlagProcessor* fp);

/* Add an option. Returns pointer to the added option (for chaining). */
JitOption* jit_flagproc_add_bool(
    JitFlagProcessor* fp,
    const char* cmdline_flag,
    const char* env_var,
    bool* target,
    const char* description);

JitOption* jit_flagproc_add_int(
    JitFlagProcessor* fp,
    const char* cmdline_flag,
    const char* env_var,
    int* target,
    const char* description);

JitOption* jit_flagproc_add_size_t(
    JitFlagProcessor* fp,
    const char* cmdline_flag,
    const char* env_var,
    size_t* target,
    const char* description);

JitOption* jit_flagproc_add_uint32(
    JitFlagProcessor* fp,
    const char* cmdline_flag,
    const char* env_var,
    uint32_t* target,
    const char* description);

JitOption* jit_flagproc_add_string(
    JitFlagProcessor* fp,
    const char* cmdline_flag,
    const char* env_var,
    char* target,  /* must be char[256] */
    const char* description);

JitOption* jit_flagproc_add_callback(
    JitFlagProcessor* fp,
    const char* cmdline_flag,
    const char* env_var,
    jit_opt_callback_fn callback,
    void* ctx,
    const char* description);

/* Process flags from Python X-options dict */
void jit_flagproc_set_flags(JitFlagProcessor* fp, PyObject* xoptions);

/* Query */
int jit_flagproc_has_options(const JitFlagProcessor* fp);
int jit_flagproc_can_handle(const JitFlagProcessor* fp, const char* name);
int jit_flagproc_has_handled(const JitFlagProcessor* fp, const char* name);

/* Option modifiers (for chaining after add) */
JitOption* jit_opt_set_debug_msg(JitOption* opt, const char* msg);
JitOption* jit_opt_set_param_name(JitOption* opt, const char* name);
JitOption* jit_opt_set_hidden(JitOption* opt, int hidden);

/* Print formatted help message to stream */
void jit_flagproc_print_help(const JitFlagProcessor* fp, FILE* out);

#ifdef __cplusplus
} /* extern "C" */
#endif
