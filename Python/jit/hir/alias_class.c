/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * AliasClass string conversion — C port of alias_class.cpp.
 */

#include "cinderx/Jit/hir/alias_class_c.h"

#include <stdio.h>
#include <string.h>

const char *hir_alias_class_to_string(uint64_t bits, char *buf, size_t bufsz) {
    /* Check for exact named matches first */
#define X(name, ...) \
    if (bits == A##name) { \
        snprintf(buf, bufsz, "%s", #name); \
        return buf; \
    }
    HIR_ACLS(X)
#undef X

    /* Build composite string from basic bits */
    buf[0] = '{';
    buf[1] = '\0';
    size_t pos = 1;
    const char *sep = "";

#define X(name) \
    if (bits & (1ull << k##name##Bit)) { \
        pos += (size_t)snprintf(buf + pos, bufsz - pos, "%s" #name, sep); \
        sep = "|"; \
    }
    HIR_BASIC_ACLS(X)
#undef X

    if (pos + 1 < bufsz) {
        buf[pos] = '}';
        buf[pos + 1] = '\0';
    }
    return buf;
}
