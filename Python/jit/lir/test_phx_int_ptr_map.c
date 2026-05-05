/* test_phx_int_ptr_map.c — C unit tests for PhxIntPtrMap data structure.
 *
 * Phase 5.A3 commit 1 (per docs/5a3-function-cpp-bridge-spec-2026-05-05.md
 * §2.3 + spec template). Mirrors test_phx_block_map.c pattern.
 *
 * Build standalone (no Phoenix dependencies — only stdlib + Python.h):
 *   cc -I. -I../../../Include -o test_phx_int_ptr_map \
 *     test_phx_int_ptr_map.c
 * (The header is fully inline; no .c link needed.)
 *
 * Test corpus:
 *   #1 init/destroy is idempotent
 *   #2 insert + lookup round-trip (small N)
 *   #3 resize triggers at expected count (load 0.7 of cap=16)
 *   #4 N=300 distinct keys all retrievable post-resize
 *   #5 colliding keys (same hash slot) all retrievable
 *   #6 overwrite on duplicate key
 *   #7 lookup miss returns NULL
 *   #8 clear preserves capacity, resets count
 *   #9 NULL-value-with-key-present: contains() returns 1 even when
 *      lookup() returns NULL (absent-vs-NULL-value disambig)
 *   #10 zero-key works (key=0 must not collide with empty-slot sentinel)
 *   #11 get_strict returns the stored value (smoke; loud-fail path not
 *       exercised here — that would JIT_CHECK_C abort the process)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "Python.h"
#include "cinderx/Jit/hir/phx_int_ptr_map.h"

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
    PhxIntPtrMap m;
    phx_int_ptr_map_init(&m);
    ASSERT(m.entries == NULL, "init: entries NULL");
    ASSERT(m.count == 0, "init: count 0");
    ASSERT(m.capacity == 0, "init: capacity 0");
    phx_int_ptr_map_destroy(&m);
    ASSERT(m.entries == NULL, "destroy: entries NULL");
    ASSERT(m.count == 0, "destroy: count 0");
    ASSERT(m.capacity == 0, "destroy: capacity 0");
    PASS();
}

static void test_02_insert_lookup(void) {
    PhxIntPtrMap m;
    phx_int_ptr_map_init(&m);
    int sentinels[5];
    for (int i = 0; i < 5; i++) {
        phx_int_ptr_map_insert(&m, i * 10, &sentinels[i]);
    }
    ASSERT(m.count == 5, "count after 5 inserts");
    ASSERT(m.capacity == PHX_INT_PTR_MAP_INITIAL_CAP, "no resize yet");
    for (int i = 0; i < 5; i++) {
        void *v = phx_int_ptr_map_lookup(&m, i * 10);
        ASSERT(v == &sentinels[i], "lookup matches");
    }
    phx_int_ptr_map_destroy(&m);
    PASS();
}

static void test_03_resize_triggers(void) {
    PhxIntPtrMap m;
    phx_int_ptr_map_init(&m);
    int sentinels[16];
    /* Initial cap=16 * 0.7 = 11.2; insert #12 triggers resize to cap=32. */
    for (int i = 0; i < 11; i++) {
        phx_int_ptr_map_insert(&m, i + 1, &sentinels[i]);
    }
    ASSERT(m.capacity == 16, "cap=16 before #12 insert");
    phx_int_ptr_map_insert(&m, 12, &sentinels[11]);
    ASSERT(m.capacity == 32, "cap doubled at #12");
    ASSERT(m.count == 12, "count 12");
    /* All 12 keys still retrievable post-rehash. */
    for (int i = 0; i < 12; i++) {
        void *v = phx_int_ptr_map_lookup(&m, i + 1);
        ASSERT(v == &sentinels[i], "lookup post-resize");
    }
    phx_int_ptr_map_destroy(&m);
    PASS();
}

static void test_04_n300(void) {
    PhxIntPtrMap m;
    phx_int_ptr_map_init(&m);
    /* 300 distinct keys forces multiple resizes (16→32→64→128→256→512). */
    static int sentinels[300];
    for (int i = 0; i < 300; i++) {
        phx_int_ptr_map_insert(&m, i * 7 + 1, &sentinels[i]);
    }
    ASSERT(m.count == 300, "count 300");
    ASSERT(m.capacity >= 512, "cap >= 512");
    /* Lookup all in random-ish order to exercise collision chains. */
    for (int i = 299; i >= 0; i--) {
        void *v = phx_int_ptr_map_lookup(&m, i * 7 + 1);
        ASSERT(v == &sentinels[i], "n300 lookup");
    }
    phx_int_ptr_map_destroy(&m);
    PASS();
}

static void test_05_collisions(void) {
    PhxIntPtrMap m;
    phx_int_ptr_map_init(&m);
    int sentinels[8];
    /* Knuth multiplicative h = key * 2654435761u; for cap=16 mask is 0xF.
     * Synthesize 8 keys whose hashes mod 16 collide. */
    int target_slot = -1;
    int picked[8];
    int n_picked = 0;
    for (int k = 1; k < 1000 && n_picked < 8; k++) {
        uint32_t h = (uint32_t)k * 2654435761u;
        size_t slot = (size_t)h & 15u;
        if (target_slot < 0) {
            target_slot = (int)slot;
            picked[n_picked++] = k;
        } else if ((int)slot == target_slot) {
            picked[n_picked++] = k;
        }
    }
    ASSERT(n_picked == 8, "found 8 colliding keys");
    for (int i = 0; i < 8; i++) {
        phx_int_ptr_map_insert(&m, picked[i], &sentinels[i]);
    }
    for (int i = 0; i < 8; i++) {
        void *v = phx_int_ptr_map_lookup(&m, picked[i]);
        ASSERT(v == &sentinels[i], "collision lookup");
    }
    phx_int_ptr_map_destroy(&m);
    PASS();
}

static void test_06_overwrite(void) {
    PhxIntPtrMap m;
    phx_int_ptr_map_init(&m);
    int a, b;
    int rc1 = phx_int_ptr_map_insert(&m, 42, &a);
    ASSERT(rc1 == 1, "first insert returns 1 (newly inserted)");
    ASSERT(m.count == 1, "count 1 after first insert");
    int rc2 = phx_int_ptr_map_insert(&m, 42, &b);
    ASSERT(rc2 == 0, "overwrite returns 0 (updated)");
    ASSERT(m.count == 1, "count still 1 after overwrite");
    ASSERT(phx_int_ptr_map_lookup(&m, 42) == &b, "overwrite returned latest");
    phx_int_ptr_map_destroy(&m);
    PASS();
}

static void test_07_lookup_miss(void) {
    PhxIntPtrMap m;
    phx_int_ptr_map_init(&m);
    int x;
    /* Lookup before any insert (capacity=0). */
    ASSERT(phx_int_ptr_map_lookup(&m, 0) == NULL, "miss on empty");
    ASSERT(phx_int_ptr_map_contains(&m, 0) == 0, "contains 0 on empty");
    phx_int_ptr_map_insert(&m, 100, &x);
    ASSERT(phx_int_ptr_map_lookup(&m, 200) == NULL, "miss when key absent");
    ASSERT(phx_int_ptr_map_contains(&m, 200) == 0, "contains 0 when absent");
    phx_int_ptr_map_destroy(&m);
    PASS();
}

static void test_08_clear(void) {
    PhxIntPtrMap m;
    phx_int_ptr_map_init(&m);
    int sentinels[5];
    for (int i = 0; i < 5; i++) {
        phx_int_ptr_map_insert(&m, i * 3, &sentinels[i]);
    }
    size_t cap_before = m.capacity;
    phx_int_ptr_map_clear(&m);
    ASSERT(m.count == 0, "count reset");
    ASSERT(m.capacity == cap_before, "capacity preserved");
    for (int i = 0; i < 5; i++) {
        ASSERT(phx_int_ptr_map_lookup(&m, i * 3) == NULL, "miss post-clear");
    }
    /* Re-fill works. */
    for (int i = 0; i < 5; i++) {
        phx_int_ptr_map_insert(&m, i * 3 + 1, &sentinels[i]);
    }
    for (int i = 0; i < 5; i++) {
        ASSERT(phx_int_ptr_map_lookup(&m, i * 3 + 1) == &sentinels[i],
               "post-clear refill");
    }
    phx_int_ptr_map_destroy(&m);
    PASS();
}

static void test_09_null_value_disambig(void) {
    /* Critical differentiator from PhxBlockMap: PhxIntPtrMap uses an
     * explicit `occupied` flag, so storing NULL as a value is a valid
     * operation that must be distinguishable from absent. */
    PhxIntPtrMap m;
    phx_int_ptr_map_init(&m);
    phx_int_ptr_map_insert(&m, 7, NULL);
    ASSERT(m.count == 1, "NULL-value insert counted");
    ASSERT(phx_int_ptr_map_lookup(&m, 7) == NULL, "lookup returns NULL value");
    ASSERT(phx_int_ptr_map_contains(&m, 7) == 1,
           "contains returns 1 for NULL-value-present key");
    ASSERT(phx_int_ptr_map_contains(&m, 8) == 0,
           "contains returns 0 for absent key");
    /* Overwrite NULL with non-NULL works. */
    int x;
    phx_int_ptr_map_insert(&m, 7, &x);
    ASSERT(m.count == 1, "overwrite preserves count");
    ASSERT(phx_int_ptr_map_lookup(&m, 7) == &x, "overwrite to non-NULL");
    /* Overwrite non-NULL back to NULL works. */
    phx_int_ptr_map_insert(&m, 7, NULL);
    ASSERT(phx_int_ptr_map_lookup(&m, 7) == NULL, "overwrite back to NULL");
    ASSERT(phx_int_ptr_map_contains(&m, 7) == 1, "still contains key 7");
    phx_int_ptr_map_destroy(&m);
    PASS();
}

static void test_10_zero_key(void) {
    /* BasicBlock id_ + Instruction id_ start at 0; the explicit-occupied
     * sentinel design must permit key=0 without collision against the
     * empty-slot marker (which an int-zero sentinel would have caused). */
    PhxIntPtrMap m;
    phx_int_ptr_map_init(&m);
    int v0, v1;
    phx_int_ptr_map_insert(&m, 0, &v0);
    phx_int_ptr_map_insert(&m, 1, &v1);
    ASSERT(m.count == 2, "two distinct keys including 0");
    ASSERT(phx_int_ptr_map_lookup(&m, 0) == &v0, "key 0 retrievable");
    ASSERT(phx_int_ptr_map_lookup(&m, 1) == &v1, "key 1 retrievable");
    ASSERT(phx_int_ptr_map_contains(&m, 0) == 1, "contains 0");
    /* Key not present (negative also works since slot is masked). */
    ASSERT(phx_int_ptr_map_contains(&m, -1) == 0, "negative absent");
    phx_int_ptr_map_destroy(&m);
    PASS();
}

static void test_11_get_strict(void) {
    /* Smoke for get_strict on present key. The loud-fail path is not
     * exercised here — JIT_CHECK_C aborts the process on miss, which
     * cannot be caught from the test harness. */
    PhxIntPtrMap m;
    phx_int_ptr_map_init(&m);
    int x;
    phx_int_ptr_map_insert(&m, 99, &x);
    void *v = phx_int_ptr_map_get_strict(&m, 99);
    ASSERT(v == &x, "get_strict returns stored value");
    phx_int_ptr_map_destroy(&m);
    PASS();
}

int main(void) {
    printf("PhxIntPtrMap unit tests\n");
    printf("=======================\n");
    test_01_init_destroy();
    test_02_insert_lookup();
    test_03_resize_triggers();
    test_04_n300();
    test_05_collisions();
    test_06_overwrite();
    test_07_lookup_miss();
    test_08_clear();
    test_09_null_value_disambig();
    test_10_zero_key();
    test_11_get_strict();
    printf("=======================\n");
    printf("Result: %d PASS / %d FAIL\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
