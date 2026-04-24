/* test_phx_block_map.c — C unit tests for PhxBlockMap data structure.
 *
 * Tier 8 SECOND-PILOT Phase A (theologian 10:25:08Z + supervisor
 * 10:29:05Z + 10:29:22Z, testkeeper 10:29:06Z gap-flag): exercises
 * insert / lookup / resize / collision paths. Companion to the
 * deterministic resize-trigger fixture added to scripts/gate_phoenix.sh
 * in the same commit.
 *
 * Build standalone (no Phoenix dependencies — only stdlib):
 *   cc -I. -I../../../Include -o test_phx_block_map \
 *     test_phx_block_map.c
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
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "Python.h"
#include "cinderx/Jit/hir/builder_state_c.h"

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
    PhxBlockMap m;
    phx_block_map_init(&m);
    ASSERT(m.entries == NULL, "init: entries NULL");
    ASSERT(m.count == 0, "init: count 0");
    ASSERT(m.capacity == 0, "init: capacity 0");
    phx_block_map_destroy(&m);
    ASSERT(m.entries == NULL, "destroy: entries NULL");
    ASSERT(m.count == 0, "destroy: count 0");
    ASSERT(m.capacity == 0, "destroy: capacity 0");
    PASS();
}

static void test_02_insert_lookup(void) {
    PhxBlockMap m;
    phx_block_map_init(&m);
    int sentinels[5];
    for (int i = 0; i < 5; i++) {
        phx_block_map_insert(&m, i * 10, &sentinels[i]);
    }
    ASSERT(m.count == 5, "count after 5 inserts");
    ASSERT(m.capacity == PHX_BLOCK_MAP_INITIAL_CAP, "no resize yet");
    for (int i = 0; i < 5; i++) {
        void *v = phx_block_map_lookup(&m, i * 10);
        ASSERT(v == &sentinels[i], "lookup matches");
    }
    phx_block_map_destroy(&m);
    PASS();
}

static void test_03_resize_triggers(void) {
    PhxBlockMap m;
    phx_block_map_init(&m);
    int sentinels[16];
    /* Initial cap=16 * 0.7 = 11.2; insert #12 triggers resize to cap=32. */
    for (int i = 0; i < 11; i++) {
        phx_block_map_insert(&m, i + 1, &sentinels[i]);
    }
    ASSERT(m.capacity == 16, "cap=16 before #12 insert");
    phx_block_map_insert(&m, 12, &sentinels[11]);
    ASSERT(m.capacity == 32, "cap doubled at #12");
    ASSERT(m.count == 12, "count 12");
    /* All 12 keys still retrievable post-rehash. */
    for (int i = 0; i < 12; i++) {
        void *v = phx_block_map_lookup(&m, i + 1);
        ASSERT(v == &sentinels[i], "lookup post-resize");
    }
    phx_block_map_destroy(&m);
    PASS();
}

static void test_04_n300(void) {
    PhxBlockMap m;
    phx_block_map_init(&m);
    /* 300 distinct keys forces multiple resizes (16→32→64→128→256→512). */
    static int sentinels[300];
    for (int i = 0; i < 300; i++) {
        phx_block_map_insert(&m, i * 7 + 1, &sentinels[i]);
    }
    ASSERT(m.count == 300, "count 300");
    ASSERT(m.capacity >= 512, "cap >= 512");
    /* Lookup all in random-ish order to exercise collision chains. */
    for (int i = 299; i >= 0; i--) {
        void *v = phx_block_map_lookup(&m, i * 7 + 1);
        ASSERT(v == &sentinels[i], "n300 lookup");
    }
    phx_block_map_destroy(&m);
    PASS();
}

static void test_05_collisions(void) {
    PhxBlockMap m;
    phx_block_map_init(&m);
    int sentinels[8];
    /* Knuth multiplicative h = key * 2654435761u; for cap=16 mask is 0xF.
     * Synthesize 8 keys whose hashes mod 16 collide, by stepping through
     * keys and selecting those landing in the same slot. */
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
        phx_block_map_insert(&m, picked[i], &sentinels[i]);
    }
    for (int i = 0; i < 8; i++) {
        void *v = phx_block_map_lookup(&m, picked[i]);
        ASSERT(v == &sentinels[i], "collision lookup");
    }
    phx_block_map_destroy(&m);
    PASS();
}

static void test_06_overwrite(void) {
    PhxBlockMap m;
    phx_block_map_init(&m);
    int a, b;
    phx_block_map_insert(&m, 42, &a);
    ASSERT(m.count == 1, "count 1 after first insert");
    phx_block_map_insert(&m, 42, &b);
    ASSERT(m.count == 1, "count still 1 after overwrite");
    ASSERT(phx_block_map_lookup(&m, 42) == &b, "overwrite returned latest");
    phx_block_map_destroy(&m);
    PASS();
}

static void test_07_lookup_miss(void) {
    PhxBlockMap m;
    phx_block_map_init(&m);
    int x;
    /* Lookup before any insert (capacity=0). */
    ASSERT(phx_block_map_lookup(&m, 0) == NULL, "miss on empty");
    phx_block_map_insert(&m, 100, &x);
    ASSERT(phx_block_map_lookup(&m, 200) == NULL, "miss when key absent");
    phx_block_map_destroy(&m);
    PASS();
}

static void test_08_clear(void) {
    PhxBlockMap m;
    phx_block_map_init(&m);
    int sentinels[5];
    for (int i = 0; i < 5; i++) {
        phx_block_map_insert(&m, i * 3, &sentinels[i]);
    }
    size_t cap_before = m.capacity;
    phx_block_map_clear(&m);
    ASSERT(m.count == 0, "count reset");
    ASSERT(m.capacity == cap_before, "capacity preserved");
    for (int i = 0; i < 5; i++) {
        ASSERT(phx_block_map_lookup(&m, i * 3) == NULL, "miss post-clear");
    }
    /* Re-fill works. */
    for (int i = 0; i < 5; i++) {
        phx_block_map_insert(&m, i * 3 + 1, &sentinels[i]);
    }
    for (int i = 0; i < 5; i++) {
        ASSERT(phx_block_map_lookup(&m, i * 3 + 1) == &sentinels[i],
               "post-clear refill");
    }
    phx_block_map_destroy(&m);
    PASS();
}

int main(void) {
    printf("PhxBlockMap unit tests\n");
    printf("======================\n");
    test_01_init_destroy();
    test_02_insert_lookup();
    test_03_resize_triggers();
    test_04_n300();
    test_05_collisions();
    test_06_overwrite();
    test_07_lookup_miss();
    test_08_clear();
    printf("======================\n");
    printf("Result: %d PASS / %d FAIL\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
