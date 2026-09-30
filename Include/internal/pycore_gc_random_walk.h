// Shared stochastic hill-climbing worker controller for parallel GC.
//
// Used by both the GIL parallel GC (Python/gc.c, Python/gc_parallel.c) and the
// free-threaded parallel GC (Python/gc_free_threading*.c) so they exhibit
// IDENTICAL worker-count adaptation behaviour.
//
// Algorithm:
// - Start at min(4, num_workers).
// - With 20% probability, try an adjacent worker count in an unbiased random
//   direction.
// - The next valid collection measures that trial. Keep it only if its cost
//   per candidate is lower; otherwise restore the previous worker count.
// - Clamp trials to [2, num_workers].
// - The first collection only records a baseline.
//
// State (caller-owned, must be initialised by caller):
// - prev_cost_per_obj_ns: previous collection's per-object cost in ns; 0.0
//   means no measurement yet.
// - trial_previous_workers: worker count before the current trial; 0 means no
//   trial is pending.
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
                       size_t *trial_previous_workers,
                       uint32_t *explore_rng,
                       size_t *adaptive_workers,
                       size_t num_workers)
{
    if (parallel_time_ns <= 0 || candidates <= 0) {
        return;
    }

    double cost = (double)parallel_time_ns / (double)candidates;
    if (*prev_cost_per_obj_ns <= 0.0) {
        *prev_cost_per_obj_ns = cost;
        return;
    }

    // The current collection measured a trial chosen after the previous
    // collection. Keep an improvement; otherwise walk back to the previously
    // accepted worker count.
    if (*trial_previous_workers != 0) {
        if (cost < *prev_cost_per_obj_ns) {
            *prev_cost_per_obj_ns = cost;
        }
        else {
            *adaptive_workers = *trial_previous_workers;
        }
        *trial_previous_workers = 0;
        return;
    }

    // Refresh the comparison point while running at an accepted worker count.
    *prev_cost_per_obj_ns = cost;

    // xorshift32 PRNG
    uint32_t rng = *explore_rng;
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    *explore_rng = rng;
    double rand_val = (double)(rng & 0xFFFF) / 65535.0;

    // 20% chance to step ±1 (proactive exploration)
    if (rand_val < 0.2) {
        // No directional bias: 50/50 chance to increase or decrease.
        double dir_val = (double)((rng >> 16) & 0xFFFF) / 65535.0;
        int delta = (dir_val < 0.5) ? 1 : -1;

        size_t trial = *adaptive_workers;
        if (trial <= 2) {
            trial++;
        }
        else if (trial >= num_workers) {
            trial--;
        }
        else if (delta > 0) {
            trial++;
        }
        else {
            trial--;
        }

        *trial_previous_workers = *adaptive_workers;
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
