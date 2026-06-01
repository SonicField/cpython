// Shared random-walk adaptive worker count controller for parallel GC.
//
// Used by both the GIL parallel GC (Python/gc.c, Python/gc_parallel.c) and the
// free-threaded parallel GC (Python/gc_free_threading*.c) so they exhibit
// IDENTICAL worker-count adaptation behaviour.
//
// Algorithm:
// - Each collection, with 20% probability, step adaptive_workers by ±1.
// - 50/50 unbiased direction (no uphill bias).
// - Clamp result to [2, num_workers].
// - First collection (prev_cost <= 0) only records baseline; no step.
//
// State (caller-owned, must be initialised by caller):
// - prev_cost_per_obj_ns: previous collection's per-object cost in ns; 0.0
//   means no measurement yet.
// - explore_rng: xorshift32 PRNG state; must be non-zero (caller seeds from
//   GC_TEST_SEED env var or a perf counter, then clamps to 1 if zero).
// - adaptive_workers: current worker count, in [2, num_workers]; caller
//   typically initialises to min(4, num_workers).

#ifndef Py_INTERNAL_GC_RANDOM_WALK_H
#define Py_INTERNAL_GC_RANDOM_WALK_H

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include "Python.h"
#include <stdint.h>

// Update *adaptive_workers, *explore_rng, *prev_cost_per_obj_ns based on the
// observed collection cost. Pure logic — no allocations, no locks, no globals.
//
// parallel_time_ns: wall-clock time the parallel work took (gc_start to
//   cleanup_end), in nanoseconds. Must be > 0 for the update to fire.
// candidates: number of objects considered by the collection. Must be > 0 for
//   the update to fire.
//
// If parallel_time_ns <= 0 or candidates <= 0 (e.g. trivial collection),
// state is not modified.
static inline void
_PyGC_RandomWalkUpdate(int64_t parallel_time_ns,
                       Py_ssize_t candidates,
                       double *prev_cost_per_obj_ns,
                       uint32_t *explore_rng,
                       size_t *adaptive_workers,
                       size_t num_workers)
{
    if (parallel_time_ns <= 0 || candidates <= 0) {
        return;
    }

    double cost = (double)parallel_time_ns / (double)candidates;
    double prev_cost = *prev_cost_per_obj_ns;
    *prev_cost_per_obj_ns = cost;

    // Skip adjustment on first collection (no baseline yet)
    if (prev_cost <= 0.0) {
        return;
    }

    // xorshift32 PRNG
    uint32_t rng = *explore_rng;
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    *explore_rng = rng;
    double rand_val = (double)(rng & 0xFFFF) / 65535.0;

    // 20% chance to step ±1 (proactive exploration)
    if (rand_val < 0.2) {
        // No directional bias: 50/50 chance to increase or decrease
        double dir_val = (double)((rng >> 16) & 0xFFFF) / 65535.0;
        int delta = (dir_val < 0.5) ? 1 : -1;

        // Always step when the dice fires. Good values stick because they
        // don't trigger further corrective steps.
        size_t trial = *adaptive_workers;
        if (delta > 0 && trial < num_workers) {
            trial++;
        } else if (delta < 0 && trial > 2) {
            trial--;
        }
        *adaptive_workers = trial;
    }
}

// Seed an xorshift32 PRNG state. Reads GC_TEST_SEED env var if set
// (for deterministic tests), otherwise uses PyTime_PerfCounterRaw().
// Guarantees a non-zero seed (xorshift32 absorbing state).
static inline uint32_t
_PyGC_RandomWalkSeed(void)
{
    uint32_t seed;
    const char *seed_env = getenv("GC_TEST_SEED");
    if (seed_env != NULL) {
        seed = (uint32_t)atoi(seed_env);
    } else {
        PyTime_t seed_time;
        (void)PyTime_PerfCounterRaw(&seed_time);
        seed = (uint32_t)seed_time;
    }
    if (seed == 0) {
        seed = 1;  // xorshift32 absorbing state guard
    }
    return seed;
}

#ifdef __cplusplus
}
#endif

#endif /* !Py_INTERNAL_GC_RANDOM_WALK_H */
