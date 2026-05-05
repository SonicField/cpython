/* test_phx_ptr_int_map.c — C unit tests for PhxPtrIntMap data structure.
 *
 * Phase 5.A3 commit 2 (per docs/5a3-function-cpp-bridge-spec-2026-05-05.md
 * §2.3 + spec template). Mirrors test_phx_int_ptr_map.c pattern.
 *
 * Build standalone (no Phoenix dependencies — only stdlib + Python.h):
 *   cc -I. -I../../../Include -o test_phx_ptr_int_map \
 *     test_phx_ptr_int_map.c
 *
 * Test corpus:
 *   #1 init/destroy is idempotent
 *   #2 insert + lookup_or round-trip (small N)
 *   #3 resize triggers at expected count (load 0.7 of cap=16)
 *   #4 N=300 distinct keys all retrievable post-resize
 *   #5 colliding keys (same hash slot) all retrievable
 *   #6 overwrite on duplicate key
 *   #7 lookup miss returns default value
 *   #8 clear preserves capacity, resets count
 *   #9 zero-value-with-key-present: contains() returns 1 even when
 *      lookup_or returns 0 (absent-vs-zero-value disambig)
 *   #10 negative int values stored & retrieved correctly
 *   #11 get_strict returns the stored value (smoke; loud-fail path
 *       not exercised — JIT_CHECK_C aborts the process)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "Python.h"
#include "cinderx/Jit/hir/phx_ptr_int_map.h"

static int g_pass = 0, g_fail = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  FAIL  %s  (line %d): %s\n", __func__, __LINE__, msg); \
        g_fail++; \
        return; \
    } \
} while (0)

#define PASS() do { \
    printf("  PASS  %s\n", __func__); \
    g_pass++; \
} while (0)

static void test_01_init_destroy(void) {
    PhxPtrIntMap m;
    phx_ptr_int_map_init(&m);
    ASSERT(m.entries == NULL, "init: entries NULL");
    ASSERT(m.count == 0, "init: count 0");
    ASSERT(m.capacity == 0, "init: capacity 0");
    phx_ptr_int_map_destroy(&m);
    ASSERT(m.entries == NULL, "destroy: entries NULL");
    ASSERT(m.count == 0, "destroy: count 0");
    ASSERT(m.capacity == 0, "destroy: capacity 0");
    PASS();
}

static void test_02_insert_lookup(void) {
    PhxPtrIntMap m;
    phx_ptr_int_map_init(&m);
    int sentinels[5];
    for (int i = 0; i < 5; i++) {
        phx_ptr_int_map_insert(&m, &sentinels[i], i * 10 + 1);
    }
    ASSERT(m.count == 5, "count after 5 inserts");
    ASSERT(m.capacity == PHX_PTR_INT_MAP_INITIAL_CAP, "no resize yet");
    for (int i = 0; i < 5; i++) {
        int v = phx_ptr_int_map_lookup_or(&m, &sentinels[i], -1);
        ASSERT(v == i * 10 + 1, "lookup matches");
    }
    phx_ptr_int_map_destroy(&m);
    PASS();
}

static void test_03_resize_triggers(void) {
    PhxPtrIntMap m;
    phx_ptr_int_map_init(&m);
    int sentinels[16];
    /* Initial cap=16 * 0.7 = 11.2; insert #12 triggers resize to cap=32. */
    for (int i = 0; i < 11; i++) {
        phx_ptr_int_map_insert(&m, &sentinels[i], i + 1);
    }
    ASSERT(m.capacity == 16, "cap=16 before #12 insert");
    phx_ptr_int_map_insert(&m, &sentinels[11], 12);
    ASSERT(m.capacity == 32, "cap doubled at #12");
    ASSERT(m.count == 12, "count 12");
    /* All 12 keys still retrievable post-rehash. */
    for (int i = 0; i < 12; i++) {
        int v = phx_ptr_int_map_lookup_or(&m, &sentinels[i], -1);
        ASSERT(v == i + 1, "lookup post-resize");
    }
    phx_ptr_int_map_destroy(&m);
    PASS();
}

static void test_04_n300(void) {
    PhxPtrIntMap m;
    phx_ptr_int_map_init(&m);
    /* 300 distinct keys forces multiple resizes (16→32→64→128→256→512). */
    static int sentinels[300];
    for (int i = 0; i < 300; i++) {
        phx_ptr_int_map_insert(&m, &sentinels[i], i * 7 + 1);
    }
    ASSERT(m.count == 300, "count 300");
    ASSERT(m.capacity >= 512, "cap >= 512");
    /* Lookup all in random-ish order to exercise collision chains. */
    for (int i = 299; i >= 0; i--) {
        int v = phx_ptr_int_map_lookup_or(&m, &sentinels[i], -1);
        ASSERT(v == i * 7 + 1, "n300 lookup");
    }
    phx_ptr_int_map_destroy(&m);
    PASS();
}

static void test_05_collisions(void) {
    PhxPtrIntMap m;
    phx_ptr_int_map_init(&m);
    /* Force collisions by passing pointers whose hash mod 16 collides.
     * Pointer addresses are not directly controllable; instead, allocate
     * a large array and pick entries whose addresses hash to the same
     * slot. */
    static char buf[8192];
    void *picked[8];
    int n_picked = 0;
    int target_slot = -1;
    for (size_t off = 0; off < sizeof(buf) && n_picked < 8; off += 8) {
        void *p = &buf[off];
        size_t slot = phx_ptr_int_map_slot(16, p);
        if (target_slot < 0) {
            target_slot = (int)slot;
            picked[n_picked++] = p;
        } else if ((int)slot == target_slot) {
            picked[n_picked++] = p;
        }
    }
    ASSERT(n_picked == 8, "found 8 colliding keys");
    for (int i = 0; i < 8; i++) {
        phx_ptr_int_map_insert(&m, picked[i], i * 100 + 7);
    }
    for (int i = 0; i < 8; i++) {
        int v = phx_ptr_int_map_lookup_or(&m, picked[i], -1);
        ASSERT(v == i * 100 + 7, "collision lookup");
    }
    phx_ptr_int_map_destroy(&m);
    PASS();
}

static void test_06_overwrite(void) {
    PhxPtrIntMap m;
    phx_ptr_int_map_init(&m);
    int k;
    int rc1 = phx_ptr_int_map_insert(&m, &k, 42);
    ASSERT(rc1 == 1, "first insert returns 1 (newly inserted)");
    ASSERT(m.count == 1, "count 1 after first insert");
    int rc2 = phx_ptr_int_map_insert(&m, &k, 99);
    ASSERT(rc2 == 0, "overwrite returns 0 (updated)");
    ASSERT(m.count == 1, "count still 1 after overwrite");
    ASSERT(phx_ptr_int_map_lookup_or(&m, &k, -1) == 99,
           "overwrite returned latest");
    phx_ptr_int_map_destroy(&m);
    PASS();
}

static void test_07_lookup_miss(void) {
    PhxPtrIntMap m;
    phx_ptr_int_map_init(&m);
    int x, y;
    /* Lookup before any insert (capacity=0). */
    ASSERT(phx_ptr_int_map_lookup_or(&m, &x, -7) == -7,
           "miss on empty returns default");
    ASSERT(phx_ptr_int_map_contains(&m, &x) == 0, "contains 0 on empty");
    phx_ptr_int_map_insert(&m, &x, 100);
    ASSERT(phx_ptr_int_map_lookup_or(&m, &y, -7) == -7,
           "miss when key absent returns default");
    ASSERT(phx_ptr_int_map_contains(&m, &y) == 0,
           "contains 0 when absent");
    phx_ptr_int_map_destroy(&m);
    PASS();
}

static void test_08_clear(void) {
    PhxPtrIntMap m;
    phx_ptr_int_map_init(&m);
    int sentinels[5];
    for (int i = 0; i < 5; i++) {
        phx_ptr_int_map_insert(&m, &sentinels[i], i * 3);
    }
    size_t cap_before = m.capacity;
    phx_ptr_int_map_clear(&m);
    ASSERT(m.count == 0, "count reset");
    ASSERT(m.capacity == cap_before, "capacity preserved");
    for (int i = 0; i < 5; i++) {
        ASSERT(phx_ptr_int_map_lookup_or(&m, &sentinels[i], -1) == -1,
               "miss post-clear");
    }
    /* Re-fill works. */
    for (int i = 0; i < 5; i++) {
        phx_ptr_int_map_insert(&m, &sentinels[i], i + 1);
    }
    for (int i = 0; i < 5; i++) {
        ASSERT(phx_ptr_int_map_lookup_or(&m, &sentinels[i], -1) == i + 1,
               "post-clear refill");
    }
    phx_ptr_int_map_destroy(&m);
    PASS();
}

static void test_09_zero_value_disambig(void) {
    /* Int values may legitimately be 0 (e.g. instr_refs storing
     * Instruction id_=0). lookup_or with default 0 cannot
     * distinguish absent-vs-zero-stored; contains() must be used. */
    PhxPtrIntMap m;
    phx_ptr_int_map_init(&m);
    int k;
    phx_ptr_int_map_insert(&m, &k, 0);
    ASSERT(m.count == 1, "zero-value insert counted");
    ASSERT(phx_ptr_int_map_lookup_or(&m, &k, -1) == 0,
           "lookup_or returns the stored 0");
    ASSERT(phx_ptr_int_map_contains(&m, &k) == 1,
           "contains returns 1 for zero-value-present key");
    int absent;
    ASSERT(phx_ptr_int_map_lookup_or(&m, &absent, 0) == 0,
           "lookup_or absent returns default 0 (ambiguous with stored 0)");
    ASSERT(phx_ptr_int_map_contains(&m, &absent) == 0,
           "contains disambiguates absent");
    phx_ptr_int_map_destroy(&m);
    PASS();
}

static void test_10_negative_values(void) {
    PhxPtrIntMap m;
    phx_ptr_int_map_init(&m);
    int a, b, c;
    phx_ptr_int_map_insert(&m, &a, -1);
    phx_ptr_int_map_insert(&m, &b, INT32_MIN);
    phx_ptr_int_map_insert(&m, &c, -2147483647);
    ASSERT(phx_ptr_int_map_lookup_or(&m, &a, 0) == -1, "store -1");
    ASSERT(phx_ptr_int_map_lookup_or(&m, &b, 0) == INT32_MIN, "store INT32_MIN");
    ASSERT(phx_ptr_int_map_lookup_or(&m, &c, 0) == -2147483647, "store negative");
    phx_ptr_int_map_destroy(&m);
    PASS();
}

static void test_11_get_strict(void) {
    /* Smoke for get_strict on present key. The loud-fail path is not
     * exercised — JIT_CHECK_C aborts the process on miss. */
    PhxPtrIntMap m;
    phx_ptr_int_map_init(&m);
    int k;
    phx_ptr_int_map_insert(&m, &k, 12345);
    int v = phx_ptr_int_map_get_strict(&m, &k);
    ASSERT(v == 12345, "get_strict returns stored value");
    phx_ptr_int_map_destroy(&m);
    PASS();
}

int main(void) {
    printf("PhxPtrIntMap unit tests\n");
    printf("=======================\n");
    test_01_init_destroy();
    test_02_insert_lookup();
    test_03_resize_triggers();
    test_04_n300();
    test_05_collisions();
    test_06_overwrite();
    test_07_lookup_miss();
    test_08_clear();
    test_09_zero_value_disambig();
    test_10_negative_values();
    test_11_get_strict();
    printf("=======================\n");
    printf("Result: %d PASS / %d FAIL\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
