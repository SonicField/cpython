/*
 * phoenix-asm: Code Allocator
 *
 * Manages executable memory for JIT-compiled code via mmap/munmap.
 * Replaces asmjit::JitRuntime.
 *
 * Each allocation is its own mmap region (page-aligned). No sub-page
 * pooling at this stage -- keeps the implementation simple and ensures
 * individual allocations can be released independently.
 *
 * Linux only. Works on both x86_64 and ARM64.
 */

/* Required for MAP_ANONYMOUS on some glibc versions */
#define _GNU_SOURCE

#include "alloc.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

/* --------------------------------------------------------------------
 * Internal types
 * -------------------------------------------------------------------- */

typedef struct {
    void*  ptr;
    size_t size;
} PhxAllocation;

/* Initial capacity for the allocations array */
#define PHX_ALLOC_INITIAL_CAP 32

struct PhxRuntime {
    PhxAllocation* allocations;
    uint32_t       alloc_count;
    uint32_t       alloc_capacity;
    size_t         page_size;
    size_t         used_bytes;
};

/* --------------------------------------------------------------------
 * Helpers
 * -------------------------------------------------------------------- */

/* Round 'n' up to the next multiple of 'align'. 'align' must be a power of 2. */
static size_t align_up(size_t n, size_t align)
{
    return (n + (align - 1)) & ~(align - 1);
}

/* Find the index of 'ptr' in the allocations array.
 * Returns the index, or (uint32_t)-1 if not found. */
static uint32_t find_allocation(const PhxRuntime* rt, const void* ptr)
{
    for (uint32_t i = 0; i < rt->alloc_count; i++) {
        if (rt->allocations[i].ptr == ptr) {
            return i;
        }
    }
    return (uint32_t)-1;
}

/* Grow the allocations array to accommodate at least one more entry.
 * Returns 0 on success, -1 on allocation failure. */
static int ensure_capacity(PhxRuntime* rt)
{
    if (rt->alloc_count < rt->alloc_capacity) {
        return 0;
    }

    uint32_t new_cap = rt->alloc_capacity * 2;
    if (new_cap == 0) {
        new_cap = PHX_ALLOC_INITIAL_CAP;
    }

    PhxAllocation* new_arr = (PhxAllocation*)realloc(
        rt->allocations, (size_t)new_cap * sizeof(PhxAllocation));
    if (!new_arr) {
        return -1;
    }

    rt->allocations    = new_arr;
    rt->alloc_capacity = new_cap;
    return 0;
}

/* Remove the allocation at index 'idx' by swapping with the last element. */
static void remove_allocation(PhxRuntime* rt, uint32_t idx)
{
    rt->alloc_count--;
    if (idx < rt->alloc_count) {
        rt->allocations[idx] = rt->allocations[rt->alloc_count];
    }
}

/* --------------------------------------------------------------------
 * Public API
 * -------------------------------------------------------------------- */

PhxRuntime* phx_runtime_create(void)
{
    PhxRuntime* rt = (PhxRuntime*)calloc(1, sizeof(PhxRuntime));
    if (!rt) {
        return NULL;
    }

    long ps = sysconf(_SC_PAGESIZE);
    if (ps <= 0) {
        ps = 4096;  /* fallback */
    }
    rt->page_size = (size_t)ps;

    rt->allocations   = (PhxAllocation*)malloc(
        PHX_ALLOC_INITIAL_CAP * sizeof(PhxAllocation));
    if (!rt->allocations) {
        free(rt);
        return NULL;
    }
    rt->alloc_count    = 0;
    rt->alloc_capacity = PHX_ALLOC_INITIAL_CAP;
    rt->used_bytes     = 0;

    return rt;
}

void phx_runtime_destroy(PhxRuntime* rt)
{
    if (!rt) {
        return;
    }

    /* Unmap all live allocations */
    for (uint32_t i = 0; i < rt->alloc_count; i++) {
        if (rt->allocations[i].ptr) {
            munmap(rt->allocations[i].ptr, rt->allocations[i].size);
        }
    }

    free(rt->allocations);
    free(rt);
}

void* phx_runtime_add(PhxRuntime* rt, const void* code, size_t code_size,
                       size_t* out_size)
{
    if (!rt || !code || code_size == 0) {
        return NULL;
    }

    /* Ensure we have room in the tracking array before mapping memory */
    if (ensure_capacity(rt) != 0) {
        return NULL;
    }

    /* Round up to page boundary */
    size_t alloc_size = align_up(code_size, rt->page_size);

    /* Allocate RWX pages */
    void* ptr = mmap(
        NULL,
        alloc_size,
        PROT_READ | PROT_WRITE | PROT_EXEC,
        MAP_PRIVATE | MAP_ANONYMOUS,
        -1,
        0);

    if (ptr == MAP_FAILED) {
        fprintf(stderr, "phx_runtime_add: mmap(%zu) failed: %s\n",
                alloc_size, strerror(errno));
        return NULL;
    }

    /* Copy the machine code into the executable region */
    memcpy(ptr, code, code_size);

    /*
     * Zero any trailing bytes in the last page. This is not strictly
     * necessary (mmap already provides zeroed pages), but is good
     * hygiene -- prevents stale data if the region is ever re-examined.
     */
    if (alloc_size > code_size) {
        memset((char*)ptr + code_size, 0, alloc_size - code_size);
    }

    /*
     * ARM64 requires an explicit instruction-cache / data-cache
     * coherency flush after writing code into memory.
     * On x86_64 this is a no-op (self-modifying code is coherent by
     * hardware design), but the compiler builtin is safe to call on
     * any architecture.
     */
    __builtin___clear_cache((char*)ptr, (char*)ptr + code_size);

    /* Track the allocation */
    PhxAllocation* alloc = &rt->allocations[rt->alloc_count++];
    alloc->ptr  = ptr;
    alloc->size = alloc_size;

    rt->used_bytes += alloc_size;

    if (out_size) {
        *out_size = alloc_size;
    }

    return ptr;
}

void phx_runtime_release(PhxRuntime* rt, void* code_ptr)
{
    if (!rt || !code_ptr) {
        return;
    }

    uint32_t idx = find_allocation(rt, code_ptr);
    if (idx == (uint32_t)-1) {
        fprintf(stderr,
                "phx_runtime_release: %p not found in runtime allocations\n",
                code_ptr);
        return;
    }

    size_t alloc_size = rt->allocations[idx].size;

    if (munmap(code_ptr, alloc_size) != 0) {
        fprintf(stderr, "phx_runtime_release: munmap(%p, %zu) failed: %s\n",
                code_ptr, alloc_size, strerror(errno));
        return;
    }

    rt->used_bytes -= alloc_size;
    remove_allocation(rt, idx);
}

int phx_runtime_contains(const PhxRuntime* rt, const void* ptr)
{
    if (!rt || !ptr) {
        return 0;
    }

    for (uint32_t i = 0; i < rt->alloc_count; i++) {
        const void* base = rt->allocations[i].ptr;
        size_t      size = rt->allocations[i].size;
        if ((const char*)ptr >= (const char*)base &&
            (const char*)ptr < (const char*)base + size) {
            return 1;
        }
    }
    return 0;
}

size_t phx_runtime_used_bytes(const PhxRuntime* rt)
{
    if (!rt) {
        return 0;
    }
    return rt->used_bytes;
}

uint32_t phx_runtime_alloc_count(const PhxRuntime* rt)
{
    if (!rt) {
        return 0;
    }
    return rt->alloc_count;
}

/* --------------------------------------------------------------------
 * Self-test / falsifier
 *
 * Compile with -DPHX_ALLOC_MAIN to build a standalone test:
 *   cc -std=c11 -DPHX_ALLOC_MAIN -o test_alloc alloc.c && ./test_alloc
 *
 * Allocates RWX memory, writes a tiny function that returns 42,
 * calls it, and verifies the result.
 * -------------------------------------------------------------------- */

#ifdef PHX_ALLOC_MAIN

#include <assert.h>

typedef int (*ReturnInt)(void);

/*
 * Convert a void* to a function pointer without triggering
 * -Wpedantic "ISO C forbids conversion of object pointer to
 * function pointer type". Uses memcpy through a union-sized buffer.
 */
static ReturnInt void_to_fn(void* p)
{
    ReturnInt fn;
    memcpy(&fn, &p, sizeof(fn));
    return fn;
}

int main(void)
{
    printf("=== phoenix-asm alloc self-test ===\n");

    PhxRuntime* rt = phx_runtime_create();
    assert(rt != NULL);
    assert(phx_runtime_alloc_count(rt) == 0);
    assert(phx_runtime_used_bytes(rt) == 0);

#if defined(__x86_64__) || defined(_M_X64)
    /*
     * x86_64 machine code:
     *   mov eax, 42    ->  B8 2A 00 00 00
     *   ret            ->  C3
     */
    uint8_t code[] = { 0xB8, 0x2A, 0x00, 0x00, 0x00, 0xC3 };
#elif defined(__aarch64__)
    /*
     * ARM64 machine code:
     *   mov w0, #42    ->  d2800540
     *   ret            ->  d65f03c0
     */
    uint32_t code[] = { 0xd2800540, 0xd65f03c0 };
#else
    #error "Unsupported architecture -- expected x86_64 or ARM64"
#endif

    size_t alloc_size = 0;
    void* exec_ptr = phx_runtime_add(rt, code, sizeof(code), &alloc_size);
    assert(exec_ptr != NULL);
    printf("  allocated %zu bytes at %p (code size: %zu)\n",
           alloc_size, exec_ptr, sizeof(code));

    assert(phx_runtime_alloc_count(rt) == 1);
    assert(phx_runtime_used_bytes(rt) == alloc_size);
    assert(alloc_size >= sizeof(code));

    /* The allocation must contain both the start and a byte within range */
    assert(phx_runtime_contains(rt, exec_ptr) == 1);
    assert(phx_runtime_contains(rt, (char*)exec_ptr + 1) == 1);

    /* A pointer outside must not be contained */
    assert(phx_runtime_contains(rt, (char*)exec_ptr + alloc_size) == 0);

    /* Cast to function pointer and call */
    ReturnInt fn = void_to_fn(exec_ptr);
    int result = fn();
    printf("  function returned: %d (expected 42)\n", result);
    assert(result == 42);

    /* Test a second allocation */
    void* exec_ptr2 = phx_runtime_add(rt, code, sizeof(code), NULL);
    assert(exec_ptr2 != NULL);
    assert(phx_runtime_alloc_count(rt) == 2);

    ReturnInt fn2 = void_to_fn(exec_ptr2);
    assert(fn2() == 42);

    /* Release first allocation */
    phx_runtime_release(rt, exec_ptr);
    assert(phx_runtime_alloc_count(rt) == 1);
    assert(phx_runtime_contains(rt, exec_ptr) == 0);

    /* Second allocation should still work */
    assert(phx_runtime_contains(rt, exec_ptr2) == 1);
    assert(fn2() == 42);

    /* Release second allocation */
    phx_runtime_release(rt, exec_ptr2);
    assert(phx_runtime_alloc_count(rt) == 0);
    assert(phx_runtime_used_bytes(rt) == 0);

    /* Releasing NULL should be a no-op */
    phx_runtime_release(rt, NULL);

    /* Destroy the runtime */
    phx_runtime_destroy(rt);

    /* Destroying NULL should be safe */
    phx_runtime_destroy(NULL);

    printf("  all tests passed.\n");
    return 0;
}

#endif /* PHX_ALLOC_MAIN */
