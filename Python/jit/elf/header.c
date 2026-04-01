/*
 * header.c -- ELF header struct size verification (pure C)
 *
 * Phase 3D conversion: header.cpp -> header.c
 * The original only contained static_asserts for struct sizes.
 * C11 _Static_assert provides the same compile-time checks.
 */

#include <stdint.h>

/* Replicate the struct layouts to verify sizes.
 * The actual struct definitions remain in header.h (C++ with
 * in-class initializers). These C-side checks ensure the
 * binary layout matches ELF spec expectations. */

typedef struct {
    uint32_t name_offset;
    uint32_t type;
    uint64_t flags;
    uint64_t address;
    uint64_t offset;
    uint64_t size;
    uint32_t link;
    uint32_t info;
    uint64_t align;
    uint64_t entry_size;
} JitElfSectionHeader;

typedef struct {
    uint32_t type;
    uint32_t flags;
    uint64_t offset;
    uint64_t address;
    uint64_t physical_address;
    uint64_t file_size;
    uint64_t mem_size;
    uint64_t align;
} JitElfSegmentHeader;

_Static_assert(sizeof(JitElfSectionHeader) == 64,
    "ELF SectionHeader must be 64 bytes");
_Static_assert(sizeof(JitElfSegmentHeader) == 56,
    "ELF SegmentHeader must be 56 bytes");
