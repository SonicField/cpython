/* Copyright (c) Meta Platforms, Inc. and affiliates. */

#include "cinderx/Jit/jit_flag_processor_c.h"

#include <assert.h>
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Internal helpers ---- */

static void ensure_capacity(JitFlagProcessor* fp) {
    if (fp->count >= fp->capacity) {
        size_t new_cap = fp->capacity == 0 ? 16 : fp->capacity * 2;
        JitOption* new_opts = (JitOption*)realloc(
            fp->options, new_cap * sizeof(JitOption));
        assert(new_opts != NULL);
        fp->options = new_opts;
        fp->capacity = new_cap;
    }
}

static JitOption* add_option_common(
    JitFlagProcessor* fp,
    const char* cmdline_flag,
    const char* env_var,
    const char* description,
    JitOptType type,
    void* target,
    jit_opt_callback_fn callback,
    void* callback_ctx) {
    ensure_capacity(fp);
    JitOption* opt = &fp->options[fp->count++];
    memset(opt, 0, sizeof(*opt));
    opt->cmdline_flag = cmdline_flag;
    opt->env_var = env_var;
    opt->description = description;
    opt->type = type;
    opt->target = target;
    opt->callback = callback;
    opt->callback_ctx = callback_ctx;
    return opt;
}

static void apply_value(JitOption* opt, const char* value) {
    if (value == NULL) {
        value = "";
    }

    switch (opt->type) {
    case JIT_OPT_BOOL: {
        int v = (value[0] == '\0') ? 1 : atoi(value);
        *(bool*)opt->target = (v != 0);
        break;
    }
    case JIT_OPT_INT: {
        int v = (value[0] == '\0') ? 1 : atoi(value);
        *(int*)opt->target = v;
        break;
    }
    case JIT_OPT_SIZE_T: {
        if (value[0] == '\0') {
            *(size_t*)opt->target = 1;
        } else {
            char* end;
            errno = 0;
            unsigned long long v = strtoull(value, &end, 10);
            if (errno == 0 && end != value) {
                *(size_t*)opt->target = (size_t)v;
            } else {
                fprintf(stderr, "JIT: Invalid unsigned long value for %s/%s: %s\n",
                        opt->cmdline_flag, opt->env_var, value);
            }
        }
        break;
    }
    case JIT_OPT_UINT32: {
        if (value[0] == '\0') {
            *(uint32_t*)opt->target = 1;
        } else {
            char* end;
            errno = 0;
            unsigned long v = strtoul(value, &end, 10);
            if (errno == 0 && end != value) {
                *(uint32_t*)opt->target = (uint32_t)v;
            } else {
                fprintf(stderr, "JIT: Invalid uint32 value for %s/%s: %s\n",
                        opt->cmdline_flag, opt->env_var, value);
            }
        }
        break;
    }
    case JIT_OPT_STRING: {
        if (opt->target != NULL) {
            strncpy((char*)opt->target, value, 255);
            ((char*)opt->target)[255] = '\0';
        }
        break;
    }
    case JIT_OPT_CALLBACK: {
        if (opt->callback != NULL) {
            opt->callback(value, opt->callback_ctx);
        }
        break;
    }
    }
}

/* ---- Public API ---- */

void jit_flagproc_init(JitFlagProcessor* fp) {
    fp->options = NULL;
    fp->count = 0;
    fp->capacity = 0;
}

void jit_flagproc_free(JitFlagProcessor* fp) {
    free(fp->options);
    fp->options = NULL;
    fp->count = 0;
    fp->capacity = 0;
}

JitOption* jit_flagproc_add_bool(
    JitFlagProcessor* fp,
    const char* cmdline_flag,
    const char* env_var,
    bool* target,
    const char* description) {
    return add_option_common(fp, cmdline_flag, env_var, description,
                             JIT_OPT_BOOL, target, NULL, NULL);
}

JitOption* jit_flagproc_add_int(
    JitFlagProcessor* fp,
    const char* cmdline_flag,
    const char* env_var,
    int* target,
    const char* description) {
    return add_option_common(fp, cmdline_flag, env_var, description,
                             JIT_OPT_INT, target, NULL, NULL);
}

JitOption* jit_flagproc_add_size_t(
    JitFlagProcessor* fp,
    const char* cmdline_flag,
    const char* env_var,
    size_t* target,
    const char* description) {
    return add_option_common(fp, cmdline_flag, env_var, description,
                             JIT_OPT_SIZE_T, target, NULL, NULL);
}

JitOption* jit_flagproc_add_uint32(
    JitFlagProcessor* fp,
    const char* cmdline_flag,
    const char* env_var,
    uint32_t* target,
    const char* description) {
    return add_option_common(fp, cmdline_flag, env_var, description,
                             JIT_OPT_UINT32, target, NULL, NULL);
}

JitOption* jit_flagproc_add_string(
    JitFlagProcessor* fp,
    const char* cmdline_flag,
    const char* env_var,
    char* target,
    const char* description) {
    return add_option_common(fp, cmdline_flag, env_var, description,
                             JIT_OPT_STRING, target, NULL, NULL);
}

JitOption* jit_flagproc_add_callback(
    JitFlagProcessor* fp,
    const char* cmdline_flag,
    const char* env_var,
    jit_opt_callback_fn callback,
    void* ctx,
    const char* description) {
    return add_option_common(fp, cmdline_flag, env_var, description,
                             JIT_OPT_CALLBACK, NULL, callback, ctx);
}

void jit_flagproc_set_flags(JitFlagProcessor* fp, PyObject* xoptions) {
    assert(xoptions != NULL);

    for (size_t i = 0; i < fp->count; i++) {
        JitOption* opt = &fp->options[i];
        opt->handled = 0;

        /* Check X-options dict */
        PyObject* key = PyUnicode_FromString(opt->cmdline_flag);
        assert(key != NULL);

        PyObject* resolves_to = PyDict_GetItem(xoptions, key);
        Py_DECREF(key);
        int found = 0;

        if (resolves_to != NULL) {
            const char* got = PyUnicode_Check(resolves_to)
                ? PyUnicode_AsUTF8(resolves_to)
                : "";
            apply_value(opt, got);
            found = 1;
        }

        /* Check environment variable */
        if (!found && opt->env_var != NULL && opt->env_var[0] != '\0') {
            const char* envval = Py_GETENV(opt->env_var);
            if (envval != NULL && envval[0] != '\0') {
                apply_value(opt, envval);
                found = 1;
            }
        }

        if (found) {
            opt->handled = 1;
        }
    }

    /* Warn about unrecognized jit- options */
    PyObject* k;
    PyObject* v;
    Py_ssize_t pos = 0;
    while (PyDict_Next(xoptions, &pos, &k, &v)) {
        const char* option_name = PyUnicode_AsUTF8(k);
        if (option_name == NULL) continue;
        if (strncmp(option_name, "jit", 3) == 0) {
            if (!jit_flagproc_can_handle(fp, option_name)) {
                fprintf(stderr, "JIT: Warning: cannot handle X-option %s\n",
                        option_name);
            }
        }
    }
}

int jit_flagproc_has_options(const JitFlagProcessor* fp) {
    return fp->count > 0;
}

int jit_flagproc_can_handle(const JitFlagProcessor* fp, const char* name) {
    for (size_t i = 0; i < fp->count; i++) {
        if (strcmp(fp->options[i].cmdline_flag, name) == 0) {
            return 1;
        }
    }
    return 0;
}

int jit_flagproc_has_handled(const JitFlagProcessor* fp, const char* name) {
    for (size_t i = 0; i < fp->count; i++) {
        if (strcmp(fp->options[i].cmdline_flag, name) == 0) {
            return fp->options[i].handled;
        }
    }
    return 0;
}

JitOption* jit_opt_set_debug_msg(JitOption* opt, const char* msg) {
    opt->debug_msg = msg;
    return opt;
}

JitOption* jit_opt_set_param_name(JitOption* opt, const char* name) {
    opt->param_name = name;
    return opt;
}

JitOption* jit_opt_set_hidden(JitOption* opt, int hidden) {
    opt->hidden = hidden;
    return opt;
}

void jit_flagproc_print_help(const JitFlagProcessor* fp, FILE* out) {
    fprintf(out,
            "-X opt : set Cinder JIT-specific option. "
            "The following options are available:\n\n");
    for (size_t i = 0; i < fp->count; i++) {
        const JitOption* opt = &fp->options[i];
        if (opt->hidden) {
            continue;
        }
        fprintf(out, "         -X %s", opt->cmdline_flag);
        if (opt->param_name != NULL && opt->param_name[0] != '\0') {
            fprintf(out, "=<%s>", opt->param_name);
        }
        fprintf(out, ": %s", opt->description);
        if (opt->env_var != NULL && opt->env_var[0] != '\0') {
            fprintf(out, "; also %s", opt->env_var);
            if (opt->param_name != NULL && opt->param_name[0] != '\0') {
                fprintf(out, "=<%s>", opt->param_name);
            }
        }
        fprintf(out, "\n");
    }
}
