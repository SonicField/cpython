/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C-compatible AliasClass definitions for HIR memory alias analysis.
 * Mirrors the C++ AliasClass in alias_class.h.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Disjoint memory locations — each gets one bit */
#define HIR_BASIC_ACLS(X) \
    X(ArrayItem)          \
    X(CellItem)           \
    X(DictItem)           \
    X(FuncArgs)           \
    X(FuncAttr)           \
    X(Global)             \
    X(InObjectAttr)       \
    X(ListItem)           \
    X(Other)              \
    X(TupleItem)          \
    X(TypeAttrCache)      \
    X(TypeMethodCache)

/* Bit indexes */
enum HirAliasClassBit {
#define X(name) k##name##Bit,
    HIR_BASIC_ACLS(X)
#undef X
    kHirAliasClassNumBits
};

/* Predefined constants */
#define HIR_OR_BITS(name) | (1ull << k##name##Bit)

#define HIR_UNION_ACLS(X)                                          \
    X(Empty, 0)                                                     \
    X(Any, 0 HIR_BASIC_ACLS(HIR_OR_BITS))                          \
    X(ManagedHeapAny, (0 HIR_BASIC_ACLS(HIR_OR_BITS)) & ~(1ull << kFuncArgsBit))

#define HIR_ACLS(X) HIR_BASIC_ACLS(X) HIR_UNION_ACLS(X)

/* Named constants */
#define X(name, ...) static const uint64_t A##name = __VA_ARGS__ + 0 ? (__VA_ARGS__) : (1ull << k##name##Bit);
/* Simpler: basic acls get one bit each, union acls use explicit values */
#undef X

#define X(name) static const uint64_t A##name = (1ull << k##name##Bit);
HIR_BASIC_ACLS(X)
#undef X

#define X(name, val) static const uint64_t A##name = (val);
HIR_UNION_ACLS(X)
#undef X

/* Convert alias class bits to string representation.
 * Writes to buf (max bufsz chars). Returns buf. */
const char *hir_alias_class_to_string(uint64_t bits, char *buf, size_t bufsz);

#ifdef __cplusplus
} /* extern "C" */
#endif
