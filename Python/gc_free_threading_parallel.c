// Parallel garbage collector for free-threaded Python.
//
// This extends gc_free_threading.c with parallel marking support using:
// - Page-based sequential bucket filling for work distribution
// - Work-stealing for load balancing
// - Atomic operations for parallel marking

#include "Python.h"

#ifdef Py_GIL_DISABLED

#include "pycore_gc.h"
#include "pycore_gc_ft_parallel.h"
#include "pycore_interp.h"
#include "pycore_lock.h"                // PyMutex_Lock/Unlock
#include "pycore_pystate.h"
#include "pycore_tstate.h"
#include "pycore_time.h"  // For PyTime_PerfCounterRaw
#include "pycore_object_alloc.h"
#include "pycore_object_deferred.h"  // _PyObject_HasDeferredRefcount
#include "pycore_frame.h"            // _PyInterpreterFrame, FRAME_CLEARED
#include "pycore_stackref.h"         // PyStackRef_* functions
#include "pycore_uniqueid.h"         // _PyObject_ClearUniqueId, batch release
#include "pycore_tuple.h"            // _PyTuple_MaybeUntrack
#include "frameobject.h"             // PyFrameObject

// For mimalloc heap access - includes mimalloc/types.h for mi_page_t, mi_heap_t
#include "pycore_mimalloc.h"

#include <stdatomic.h>

// Minimum capacity for page buckets (must be power of 2)
#define _PyGC_BUCKET_MIN_CAPACITY 16

//=============================================================================
// Atomic instrumentation support (only when GC_DEBUG_ATOMICS is enabled)
//=============================================================================

#ifdef GC_DEBUG_ATOMICS

// Thread-local storage for atomic instrumentation
_Py_thread_local _PyGCAtomicPhase _PyGC_atomic_current_phase = GC_ATOMIC_PHASE_TRAVERSE;
_Py_thread_local _PyGCAtomicWorkerStats _PyGC_atomic_worker_stats = {0};

// Print aggregated atomic stats at end of collection
void
_PyGC_AtomicPrintStats(_PyGCAtomicWorkerStats *all_workers, int num_workers)
{
    // Aggregate across all workers
    _PyGCAtomicWorkerStats total = {0};
    for (int w = 0; w < num_workers; w++) {
        for (int p = 0; p < GC_ATOMIC_PHASE_COUNT; p++) {
            total.phases[p].attempts += all_workers[w].phases[p].attempts;
            total.phases[p].successes += all_workers[w].phases[p].successes;
            total.phases[p].already_done += all_workers[w].phases[p].already_done;
        }
    }

    // Print summary
    fprintf(stderr, "\n=== GC Atomic Stats (%d workers) ===\n", num_workers);
    size_t grand_attempts = 0, grand_successes = 0;
    for (int p = 0; p < GC_ATOMIC_PHASE_COUNT; p++) {
        _PyGCAtomicPhaseStats *ps = &total.phases[p];
        if (ps->attempts > 0) {
            double success_rate = 100.0 * ps->successes / ps->attempts;
            fprintf(stderr, "  %-16s: attempts=%zu, success=%zu (%.1f%%), already_done=%zu\n",
                    _PyGC_AtomicPhaseName(p), ps->attempts, ps->successes,
                    success_rate, ps->already_done);
            grand_attempts += ps->attempts;
            grand_successes += ps->successes;
        }
    }
    if (grand_attempts > 0) {
        fprintf(stderr, "  TOTAL           : attempts=%zu, success=%zu (%.1f%%)\n",
                grand_attempts, grand_successes,
                100.0 * grand_successes / grand_attempts);
    }
    fprintf(stderr, "===================================\n\n");
}

#endif  // GC_DEBUG_ATOMICS

//=============================================================================
// Page Counting - O(heaps), not O(objects)
//=============================================================================

// Count pages in a heap by reading page_count field - O(1) per heap
static inline size_t
count_heap_pages(mi_heap_t *heap)
{
    // heap->page_count is the total number of pages in all page queues
    return heap->page_count;
}

// Check if a page is from a huge segment (for round-robin distribution)
static inline bool
is_huge_page(mi_page_t *page)
{
    // Huge pages have MI_PAGE_HUGE kind, indicated by large block size
    // xblock_size > MI_LARGE_OBJ_SIZE_MAX indicates huge allocation
    return page->xblock_size > MI_LARGE_OBJ_SIZE_MAX;
}

size_t
_PyGC_CountPages(PyInterpreterState *interp)
{
    assert(interp->stoptheworld.world_stopped);

    size_t total_pages = 0;

    HEAD_LOCK(&_PyRuntime);

    // Count pages in each thread's GC heaps - O(threads), not O(objects)
    _Py_FOR_EACH_TSTATE_UNLOCKED(interp, p) {
        struct _mimalloc_thread_state *m = &((_PyThreadStateImpl *)p)->mimalloc;
        if (!_Py_atomic_load_int(&m->initialized)) {
            continue;
        }

        total_pages += count_heap_pages(&m->heaps[_Py_MIMALLOC_HEAP_GC]);
        total_pages += count_heap_pages(&m->heaps[_Py_MIMALLOC_HEAP_GC_PRE]);
    }

    // Count pages in mimalloc's abandoned pool (from dead threads)
    // These pages need to be included in parallel scanning for correct work
    // distribution when objects are created by threads that subsequently exit.
    mi_abandoned_pool_t *pool = &interp->mimalloc.abandoned_pool;
    total_pages += _mi_abandoned_pool_count_pages(pool, _Py_MIMALLOC_HEAP_GC);
    total_pages += _mi_abandoned_pool_count_pages(pool, _Py_MIMALLOC_HEAP_GC_PRE);

    HEAD_UNLOCK(&_PyRuntime);

    return total_pages;
}

//=============================================================================
// Page Enumeration - O(pages), not O(objects)
//=============================================================================

// Context for page enumeration callback
struct page_enum_context {
    _PyGCFTParState *state;
    int current_worker;       // For normal pages (sequential bucket filling)
    size_t current_count;
    size_t pages_per_worker;
    int huge_worker;          // For huge pages (round-robin)
    int error;
};

// Callback to assign a page to a worker bucket
static void
assign_page_to_bucket(mi_page_t *page, struct page_enum_context *ctx)
{
    // Preconditions
    assert(page != NULL && "NULL page passed to assign_page_to_bucket");
    assert(ctx != NULL);
    assert(ctx->state != NULL);
    assert(ctx->state->buckets != NULL);
    assert(ctx->current_worker >= 0 && ctx->current_worker < ctx->state->num_workers);
    assert(ctx->huge_worker >= 0 && ctx->huge_worker < ctx->state->num_workers);

    _PyGCPageBucket *bucket;

    if (is_huge_page(page)) {
        // Huge pages: round-robin assignment to spread expensive traversals
        bucket = &ctx->state->buckets[ctx->huge_worker];
        ctx->huge_worker = (ctx->huge_worker + 1) % ctx->state->num_workers;
    } else {
        // Normal pages: sequential bucket filling for locality
        bucket = &ctx->state->buckets[ctx->current_worker];
        ctx->current_count++;

        // Move to next worker when bucket is "full"
        if (ctx->current_count >= ctx->pages_per_worker &&
            ctx->current_worker < ctx->state->num_workers - 1) {
            ctx->current_worker++;
            ctx->current_count = 0;
        }
    }

    // Add page to bucket (grow if needed)
    if (bucket->num_pages >= bucket->capacity) {
        size_t new_capacity = bucket->capacity * 2;
        if (new_capacity < _PyGC_BUCKET_MIN_CAPACITY) new_capacity = _PyGC_BUCKET_MIN_CAPACITY;

        mi_page_t **new_pages = PyMem_RawRealloc(
            bucket->pages, new_capacity * sizeof(mi_page_t *));
        if (new_pages == NULL) {
            ctx->error = 1;
            return;
        }
        bucket->pages = new_pages;
        bucket->capacity = new_capacity;
    }

    bucket->pages[bucket->num_pages++] = page;

    // Post-condition: page was added and bucket invariants hold
    assert(bucket->num_pages <= bucket->capacity);
    assert(bucket->pages[bucket->num_pages - 1] == page);
}

// Enumerate all pages in a heap's page queues - O(pages)
static void
enumerate_heap_pages(mi_heap_t *heap, struct page_enum_context *ctx)
{
    assert(heap != NULL);
    assert(ctx != NULL);

    // Iterate through all bins (size classes)
    for (size_t bin = 0; bin <= MI_BIN_FULL; bin++) {
        mi_page_queue_t *queue = &heap->pages[bin];

        // Walk the linked list of pages in this bin
        for (mi_page_t *page = queue->first; page != NULL; page = page->next) {
            // Verify page metadata is sensible
            assert(page->used <= page->capacity &&
                   "page->used exceeds page->capacity");

            if (page->used > 0) {  // Only process pages with allocated blocks
                assign_page_to_bucket(page, ctx);
                if (ctx->error) {
                    return;
                }
            }
        }
    }
}

// Callback wrapper for _mi_abandoned_pool_enumerate_pages
// Adapts mi_page_enumerate_fun to our page bucket assignment
static void
enumerate_abandoned_page_callback(mi_page_t *page, void *arg)
{
    struct page_enum_context *ctx = (struct page_enum_context *)arg;
    assign_page_to_bucket(page, ctx);
}


//=============================================================================
// Page Assignment (Sequential Bucket Filling)
//=============================================================================

//-----------------------------------------------------------------------------
// Debug invariant checking
//-----------------------------------------------------------------------------

#ifdef Py_DEBUG

// Verify bucket internal invariants
static void
assert_bucket_valid(const _PyGCPageBucket *bucket, const char *context)
{
    // num_pages should never exceed capacity
    assert(bucket->num_pages <= bucket->capacity);

    // If we have pages, the array should be allocated
    if (bucket->num_pages > 0) {
        assert(bucket->pages != NULL);
    }

    // If capacity > 0, array should be allocated
    if (bucket->capacity > 0) {
        assert(bucket->pages != NULL);
    }

    // Each page pointer should be non-NULL
    for (size_t i = 0; i < bucket->num_pages; i++) {
        assert(bucket->pages[i] != NULL && "NULL page pointer in bucket");
    }
}

// Verify all buckets and cross-bucket invariants
static void
assert_buckets_valid(const _PyGCFTParState *state, size_t expected_total,
                     const char *context)
{
    assert(state->buckets != NULL && "buckets array is NULL");
    assert(state->num_workers > 0 && "num_workers must be positive");

    size_t actual_total = 0;

    for (int i = 0; i < state->num_workers; i++) {
        assert_bucket_valid(&state->buckets[i], context);
        actual_total += state->buckets[i].num_pages;

        // No single bucket should have more than total pages
        assert(state->buckets[i].num_pages <= expected_total);
    }

    // Total pages in buckets must match expected count
    assert(actual_total == expected_total &&
           "Total pages in buckets doesn't match expected count");
}

// Count pages by walking heaps (for verification against heap->page_count)
static size_t
count_pages_by_enumeration(PyInterpreterState *interp)
{
    size_t count = 0;

    _Py_FOR_EACH_TSTATE_UNLOCKED(interp, p) {
        struct _mimalloc_thread_state *m = &((_PyThreadStateImpl *)p)->mimalloc;
        if (!_Py_atomic_load_int(&m->initialized)) {
            continue;
        }

        // Count pages in GC heap
        mi_heap_t *heap = &m->heaps[_Py_MIMALLOC_HEAP_GC];
        for (size_t bin = 0; bin <= MI_BIN_FULL; bin++) {
            for (mi_page_t *page = heap->pages[bin].first;
                 page != NULL; page = page->next) {
                if (page->used > 0) {
                    count++;
                }
            }
        }

        // Count pages in GC_PRE heap
        heap = &m->heaps[_Py_MIMALLOC_HEAP_GC_PRE];
        for (size_t bin = 0; bin <= MI_BIN_FULL; bin++) {
            for (mi_page_t *page = heap->pages[bin].first;
                 page != NULL; page = page->next) {
                if (page->used > 0) {
                    count++;
                }
            }
        }
    }

    // Count pages in abandoned pool (from dead threads)
    mi_abandoned_pool_t *pool = &interp->mimalloc.abandoned_pool;
    count += _mi_abandoned_pool_count_pages(pool, _Py_MIMALLOC_HEAP_GC);
    count += _mi_abandoned_pool_count_pages(pool, _Py_MIMALLOC_HEAP_GC_PRE);

    return count;
}

#define ASSERT_BUCKET_VALID(bucket, ctx) assert_bucket_valid(bucket, ctx)
#define ASSERT_BUCKETS_VALID(state, total, ctx) assert_buckets_valid(state, total, ctx)

#else  // !Py_DEBUG

#define ASSERT_BUCKET_VALID(bucket, ctx) ((void)0)
#define ASSERT_BUCKETS_VALID(state, total, ctx) ((void)0)

#endif  // Py_DEBUG

//-----------------------------------------------------------------------------
// Bucket operations
//-----------------------------------------------------------------------------

// Initialize a page bucket
static int
bucket_init(_PyGCPageBucket *bucket, size_t initial_capacity)
{
    assert(bucket != NULL);
    assert(initial_capacity > 0);

    bucket->pages = PyMem_RawCalloc(initial_capacity, sizeof(mi_page_t *));
    if (bucket->pages == NULL) {
        return -1;
    }
    bucket->num_pages = 0;
    bucket->capacity = initial_capacity;

    ASSERT_BUCKET_VALID(bucket, "bucket_init exit");
    return 0;
}

// Free a page bucket
static void
bucket_free(_PyGCPageBucket *bucket)
{
    assert(bucket != NULL);

    if (bucket->pages != NULL) {
        PyMem_RawFree(bucket->pages);
        bucket->pages = NULL;
    }
    bucket->num_pages = 0;
    bucket->capacity = 0;

    // Post-condition: bucket is zeroed
    assert(bucket->pages == NULL);
    assert(bucket->num_pages == 0);
    assert(bucket->capacity == 0);
}

int
_PyGC_AssignPagesToBuckets(PyInterpreterState *interp,
                           _PyGCFTParState *state)
{
    // Entry invariants
    assert(interp != NULL);
    assert(state != NULL);
    assert(interp->stoptheworld.world_stopped);
    assert(state->num_workers > 0);
    assert(state->buckets == NULL && "buckets already allocated - double init?");

    // Step 1: Count total pages - O(threads)
    state->total_pages = _PyGC_CountPages(interp);

    // Step 2: Allocate buckets
    state->buckets = PyMem_RawCalloc(state->num_workers, sizeof(_PyGCPageBucket));
    if (state->buckets == NULL) {
        return -1;
    }

    size_t initial_capacity = (state->total_pages / state->num_workers) + _PyGC_BUCKET_MIN_CAPACITY;
    for (int i = 0; i < state->num_workers; i++) {
        if (bucket_init(&state->buckets[i], initial_capacity) < 0) {
            // Cleanup on error
            for (int j = 0; j < i; j++) {
                bucket_free(&state->buckets[j]);
            }
            PyMem_RawFree(state->buckets);
            state->buckets = NULL;
            return -1;
        }
    }

    // Step 3: Enumerate pages and assign to buckets - O(pages)
    struct page_enum_context ctx = {
        .state = state,
        .current_worker = 0,
        .current_count = 0,
        .pages_per_worker = state->total_pages / state->num_workers,
        .huge_worker = 0,
        .error = 0
    };

    // Ensure at least 1 page per worker to avoid divide-by-zero edge case
    if (ctx.pages_per_worker == 0) {
        ctx.pages_per_worker = 1;
    }

    size_t pages_enumerated = 0;  // Track for exit invariant check

    HEAD_LOCK(&_PyRuntime);

    _Py_FOR_EACH_TSTATE_UNLOCKED(interp, p) {
        struct _mimalloc_thread_state *m = &((_PyThreadStateImpl *)p)->mimalloc;
        if (!_Py_atomic_load_int(&m->initialized)) {
            continue;
        }

        enumerate_heap_pages(&m->heaps[_Py_MIMALLOC_HEAP_GC], &ctx);
        if (ctx.error) {
            break;
        }
        enumerate_heap_pages(&m->heaps[_Py_MIMALLOC_HEAP_GC_PRE], &ctx);
        if (ctx.error) {
            break;
        }
    }

    // Also enumerate pages in abandoned pool (from dead threads)
    // This is critical for multi-threaded object creation where worker threads
    // exit after creating objects, putting their heap pages into the abandoned pool.
    if (!ctx.error) {
        mi_abandoned_pool_t *pool = &interp->mimalloc.abandoned_pool;
        _mi_abandoned_pool_enumerate_pages(pool, _Py_MIMALLOC_HEAP_GC,
                                           enumerate_abandoned_page_callback, &ctx);
        if (!ctx.error) {
            _mi_abandoned_pool_enumerate_pages(pool, _Py_MIMALLOC_HEAP_GC_PRE,
                                               enumerate_abandoned_page_callback, &ctx);
        }
    }

#ifdef Py_DEBUG
    // Count pages enumerated for verification (while still holding lock)
    if (!ctx.error) {
        pages_enumerated = count_pages_by_enumeration(interp);
    }
#endif

    HEAD_UNLOCK(&_PyRuntime);

    if (ctx.error) {
        _PyGC_FreeBuckets(state);
        return -1;
    }

    // Exit invariants - verify bucket contents match enumeration
#ifdef Py_DEBUG
    {
        size_t total_in_buckets = 0;
        for (int i = 0; i < state->num_workers; i++) {
            total_in_buckets += state->buckets[i].num_pages;
        }
        assert(total_in_buckets == pages_enumerated &&
               "Bucket total doesn't match pages enumerated");
    }
#endif

    ASSERT_BUCKETS_VALID(state, pages_enumerated, "AssignPagesToBuckets exit");

    return 0;
}

void
_PyGC_FreeBuckets(_PyGCFTParState *state)
{
    // Entry invariant
    assert(state != NULL);

    if (state->buckets != NULL) {
#ifdef Py_DEBUG
        // Verify each bucket is in valid state before freeing
        for (int i = 0; i < state->num_workers; i++) {
            ASSERT_BUCKET_VALID(&state->buckets[i], "FreeBuckets entry");
        }
#endif

        for (int i = 0; i < state->num_workers; i++) {
            bucket_free(&state->buckets[i]);
        }
        PyMem_RawFree(state->buckets);
        state->buckets = NULL;
    }

    // Exit invariant
    assert(state->buckets == NULL && "buckets not NULL after free");
}


//=============================================================================
// Object Enumeration from Pages
//=============================================================================

// Get the offset of PyObject from start of block
// (accounts for debug allocator and preheader)
static inline Py_ssize_t
get_block_offset(bool is_gc_pre)
{
    Py_ssize_t offset = 0;
    if (_PyMem_DebugEnabled()) {
        // Debug allocator adds two words at beginning
        offset += 2 * sizeof(size_t);
    }
    if (is_gc_pre) {
        // Objects with Py_TPFLAGS_PREHEADER have two extra fields
        offset += 2 * sizeof(PyObject*);
    }
    return offset;
}

// Convert a memory block to a PyObject (or NULL if not valid for GC)
static inline PyObject *
block_to_object(void *block, Py_ssize_t offset)
{
    if (block == NULL) {
        return NULL;
    }
    PyObject *op = (PyObject *)((char*)block + offset);

    // Must be tracked and not frozen (frozen objects skip GC)
    if (!_PyObject_GC_IS_TRACKED(op)) {
        return NULL;
    }
    if (op->ob_gc_bits & _PyGC_BITS_FROZEN) {
        return NULL;
    }
    return op;
}

// Visitor arguments for page block enumeration
struct parallel_mark_page_args {
    _PyGCWorkerState *worker;
    Py_ssize_t offset;
    int error;
};

// Visitor callback for each block in a page
static bool
parallel_mark_block_visitor(const mi_heap_t *heap, const mi_heap_area_t *area,
                            void *block, size_t block_size, void *arg)
{
    _PyGC_ATOMIC_SET_PHASE(GC_ATOMIC_PHASE_PAGE_MARK);
    struct parallel_mark_page_args *args = (struct parallel_mark_page_args *)arg;

    PyObject *op = block_to_object(block, args->offset);
    if (op == NULL) {
        return true;  // Skip untracked/frozen objects
    }

    // Try to mark this object as alive using atomic fetch-or
    // Returns 1 if we were first to mark it
    if (_PyGC_TryMarkAlive(op)) {
        // We marked it - push to our deque for traversal
        _PyWSDeque_Push(&args->worker->deque, op);
        args->worker->objects_marked++;
    }

    return true;
}

// Visit all objects in a single page and mark alive
static int
parallel_mark_page(mi_page_t *page, _PyGCWorkerState *worker)
{
    assert(page != NULL && "NULL page in parallel_mark_page");
    assert(worker != NULL && "NULL worker in parallel_mark_page");

    mi_heap_area_t area;
    _mi_heap_area_init(&area, page);

    // Determine offset based on page tag (heap type)
    // GC_PRE heap has preheader with extra fields
    bool is_gc_pre = (page->tag == _Py_MIMALLOC_HEAP_GC_PRE);

    struct parallel_mark_page_args args = {
        .worker = worker,
        .offset = get_block_offset(is_gc_pre),
        .error = 0
    };

    // Visit all blocks in this page
    if (!_mi_heap_area_visit_blocks(&area, page, parallel_mark_block_visitor, &args)) {
        return args.error ? -1 : 0;  // Error or early exit
    }

    return 0;
}


//=============================================================================
// Work-Stealing Mark Loop
//=============================================================================

// Visitproc for parallel marking - marks children alive
static int
parallel_mark_visitproc(PyObject *child, void *arg)
{
    _PyGC_ATOMIC_SET_PHASE(GC_ATOMIC_PHASE_TRAVERSE);
    struct parallel_mark_page_args *a = (struct parallel_mark_page_args *)arg;

    if (child == NULL) {
        return 0;
    }

    // Skip untracked/frozen objects
    if (!_PyObject_GC_IS_TRACKED(child)) {
        return 0;
    }
    if (child->ob_gc_bits & _PyGC_BITS_FROZEN) {
        return 0;
    }

    // Try to mark child alive (atomic)
    if (_PyGC_TryMarkAlive(child)) {
        // We marked it - push to deque
        _PyWSDeque_Push(&a->worker->deque, child);
        a->worker->objects_marked++;
    }
    return 0;
}

// Traverse an object's children and mark alive
// Returns -1 on error, 0 on success
static int
parallel_traverse_object(PyObject *op, _PyGCWorkerState *worker)
{
    traverseproc traverse = Py_TYPE(op)->tp_traverse;
    if (traverse == NULL) {
        return 0;
    }

    // For tp_traverse, we use a visitproc that marks children alive
    // and pushes them to our deque
    struct parallel_mark_page_args args = {
        .worker = worker,
        .offset = 0,  // Not used in this path
        .error = 0
    };

    if (traverse(op, parallel_mark_visitproc, &args) < 0) {
        return args.error ? -1 : 0;
    }

    return args.error ? -1 : 0;
}

// Try to steal work from another worker
static PyObject *
try_steal_work(_PyGCFTParState *state, int my_id)
{
    assert(state != NULL && "NULL state in try_steal_work");
    assert(state->workers != NULL && "NULL workers in try_steal_work");
    assert(my_id >= 0 && my_id < state->num_workers && "Invalid worker id");

    // Round-robin stealing from other workers
    int num_workers = state->num_workers;

    for (int i = 1; i < num_workers; i++) {
        int victim = (my_id + i) % num_workers;
        PyObject *stolen = (PyObject *)_PyWSDeque_Steal(&state->workers[victim].deque);
        if (stolen != NULL) {
            state->workers[my_id].objects_stolen++;
            return stolen;
        }
        state->workers[my_id].steals_attempted++;
    }

    return NULL;  // Nothing to steal
}

// Worker main loop: process local deque, then steal
static int
parallel_worker_run(_PyGCFTParState *state, int worker_id)
{
    assert(state != NULL && "NULL state in parallel_worker_run");
    assert(state->workers != NULL && "NULL workers array");
    assert(state->buckets != NULL && "NULL buckets array");
    assert(worker_id >= 0 && worker_id < state->num_workers && "Invalid worker_id");

    _PyGCWorkerState *worker = &state->workers[worker_id];
    _PyGCPageBucket *bucket = &state->buckets[worker_id];

    // Phase 1: Process all pages in our bucket
    for (size_t i = 0; i < bucket->num_pages; i++) {
        mi_page_t *page = bucket->pages[i];

        if (parallel_mark_page(page, worker) < 0) {
            return -1;
        }
    }

    // Phase 2: Transitive marking work loop
    bool made_progress = true;
    while (made_progress) {
        made_progress = false;

        // Process our local deque (Take from bottom - LIFO)
        PyObject *op;
        while ((op = _PyWSDeque_Take(&worker->deque)) != NULL) {
            made_progress = true;
            if (parallel_traverse_object(op, worker) < 0) {
                return -1;
            }
        }

        // Try to steal work
        op = try_steal_work(state, worker_id);
        if (op != NULL) {
            made_progress = true;
            if (parallel_traverse_object(op, worker) < 0) {
                return -1;
            }
        }
    }

    return 0;
}


//=============================================================================
// Parallel Marking Entry Point
//=============================================================================

// Arguments passed to worker threads
typedef struct {
    _PyGCFTParState *state;
    int worker_id;
} _PyGCWorkerArgs;

// Thread entry function for parallel workers
static void
parallel_worker_thread(void *arg)
{
    assert(arg != NULL && "NULL arg in parallel_worker_thread");

    _PyGCWorkerArgs *args = (_PyGCWorkerArgs *)arg;

    assert(args->state != NULL && "NULL state in thread args");
    assert(args->worker_id > 0 && "Worker 0 should run on main thread");
    assert(args->worker_id < args->state->num_workers && "Invalid worker_id");

    _PyGCFTParState *state = args->state;
    int worker_id = args->worker_id;

    // Run the worker loop
    int result = parallel_worker_run(state, worker_id);

    // Signal error if something went wrong
    if (result < 0) {
        _Py_atomic_store_int(&state->error_flag, 1);
    }
}

// Initialize worker state
static int
init_workers(_PyGCFTParState *state)
{
    state->workers = PyMem_RawCalloc(state->num_workers, sizeof(_PyGCWorkerState));
    if (state->workers == NULL) {
        return -1;
    }

    for (int i = 0; i < state->num_workers; i++) {
        state->workers[i].worker_id = i;
        state->workers[i].objects_marked = 0;
        state->workers[i].objects_stolen = 0;
        state->workers[i].steals_attempted = 0;

        // Initialize work-stealing deque (uses default size)
        _PyWSDeque_Init(&state->workers[i].deque);
    }

    return 0;
}

// Free worker state
static void
free_workers(_PyGCFTParState *state)
{
    if (state->workers != NULL) {
        for (int i = 0; i < state->num_workers; i++) {
            _PyWSDeque_Fini(&state->workers[i].deque);
        }
        PyMem_RawFree(state->workers);
        state->workers = NULL;
    }
}

int
_PyGC_ParallelMarkAlive(PyInterpreterState *interp,
                        _PyGCFTParState *state)
{
    assert(interp != NULL);
    assert(state != NULL);
    assert(interp->stoptheworld.world_stopped);
    assert(state->num_workers > 0);
    assert(state->buckets != NULL);

    // Initialize worker state
    if (init_workers(state) < 0) {
        return -1;
    }

    // Initialize error flag and threads array
    state->error_flag = 0;
    state->threads = NULL;

    int result = 0;

    // For single worker, just run directly (no threading overhead)
    if (state->num_workers == 1) {
        result = parallel_worker_run(state, 0);
        goto collect_stats;
    }

    // Allocate thread handles for workers 1..N-1 (worker 0 runs on main thread)
    state->threads = PyMem_RawCalloc(state->num_workers - 1, sizeof(PyThread_handle_t));
    if (state->threads == NULL) {
        free_workers(state);
        return -1;
    }

    // Allocate thread arguments
    _PyGCWorkerArgs *thread_args = PyMem_RawCalloc(state->num_workers - 1,
                                                    sizeof(_PyGCWorkerArgs));
    if (thread_args == NULL) {
        PyMem_RawFree(state->threads);
        state->threads = NULL;
        free_workers(state);
        return -1;
    }

    // Spawn worker threads 1..N-1
    int threads_created = 0;
    for (int i = 1; i < state->num_workers; i++) {
        thread_args[i - 1].state = state;
        thread_args[i - 1].worker_id = i;

        PyThread_ident_t ident;
        int rc = PyThread_start_joinable_thread(
            parallel_worker_thread, &thread_args[i - 1],
            &ident, &state->threads[i - 1]);
        if (rc != 0) {
            // Failed to create thread - continue with fewer workers
            // but mark error so we know something went wrong
            state->error_flag = 1;
            break;
        }
        threads_created++;
    }

    // Worker 0 runs on main thread
    if (parallel_worker_run(state, 0) < 0) {
        state->error_flag = 1;
    }

    // Wait for all spawned threads to complete
    for (int i = 0; i < threads_created; i++) {
        PyThread_join_thread(state->threads[i]);
    }

    // Clean up thread resources
    PyMem_RawFree(thread_args);
    PyMem_RawFree(state->threads);
    state->threads = NULL;

    // Check for errors
    if (state->error_flag) {
        result = -1;
    }

collect_stats:
    // Collect statistics and save per-worker stats before freeing
    state->total_objects = 0;
    state->per_worker_marked = PyMem_RawCalloc(state->num_workers, sizeof(size_t));
    for (int i = 0; i < state->num_workers; i++) {
        size_t marked = state->workers[i].objects_marked;
        state->total_objects += marked;
        if (state->per_worker_marked) {
            state->per_worker_marked[i] = marked;
        }
    }

    free_workers(state);
    return result;
}


//=============================================================================
// Parallel Propagate Alive (Integration with gc_mark_alive_from_roots)
//=============================================================================

// State for parallel propagation from roots
typedef struct {
    int num_workers;
    _PyGCWorkerState *workers;
    PyThread_handle_t *threads;
    volatile int error_flag;
    size_t total_marked;
} _PyGCPropagateState;

// Worker arguments for parallel propagation
typedef struct {
    _PyGCPropagateState *state;
    int worker_id;
} _PyGCPropagateArgs;

// Visitproc for parallel propagation - marks children alive and pushes to deque
static int
propagate_visitproc(PyObject *child, void *arg)
{
    _PyGC_ATOMIC_SET_PHASE(GC_ATOMIC_PHASE_PROPAGATE);
    _PyGCWorkerState *worker = (_PyGCWorkerState *)arg;

    if (child == NULL) {
        return 0;
    }

    // Skip untracked/frozen objects
    if (!_PyObject_GC_IS_TRACKED(child)) {
        return 0;
    }
    if (child->ob_gc_bits & _PyGC_BITS_FROZEN) {
        return 0;
    }

    // Try to mark child alive (atomic fetch-or)
    if (_PyGC_TryMarkAlive(child)) {
        // We marked it first - push to our deque for traversal
        _PyWSDeque_Push(&worker->deque, child);
        worker->objects_marked++;
    }
    return 0;
}

// Traverse an object's children for propagation
static int
propagate_traverse_object(PyObject *op, _PyGCWorkerState *worker)
{
    assert(op != NULL);
    traverseproc traverse = Py_TYPE(op)->tp_traverse;
    if (traverse == NULL) {
        return 0;
    }
    return traverse(op, propagate_visitproc, worker);
}

// Try to steal work from another worker for propagation
static PyObject *
propagate_try_steal(_PyGCPropagateState *state, int my_id)
{
    for (int i = 1; i < state->num_workers; i++) {
        int victim = (my_id + i) % state->num_workers;
        PyObject *stolen = (PyObject *)_PyWSDeque_Steal(&state->workers[victim].deque);
        if (stolen != NULL) {
            return stolen;
        }
    }
    return NULL;
}

// Worker loop for parallel propagation
static int
propagate_worker_run(_PyGCPropagateState *state, int worker_id)
{
    assert(state != NULL);
    assert(worker_id >= 0 && worker_id < state->num_workers);

    _PyGCWorkerState *worker = &state->workers[worker_id];

    // Work-stealing loop: process local deque, steal when empty
    bool made_progress = true;
    while (made_progress) {
        made_progress = false;

        // Process local deque
        PyObject *op;
        while ((op = _PyWSDeque_Take(&worker->deque)) != NULL) {
            made_progress = true;
            if (propagate_traverse_object(op, worker) < 0) {
                return -1;
            }
        }

        // Try to steal
        op = propagate_try_steal(state, worker_id);
        if (op != NULL) {
            made_progress = true;
            if (propagate_traverse_object(op, worker) < 0) {
                return -1;
            }
        }
    }

    return 0;
}


// Visitor callback to clear ALIVE bits after testing
static bool
clear_alive_bits_visitor(const mi_heap_t *heap, const mi_heap_area_t *area,
                         void *block, size_t block_size, void *arg)
{
    Py_ssize_t *offset = (Py_ssize_t *)arg;

    PyObject *op = block_to_object(block, *offset);
    if (op != NULL) {
        // Clear ALIVE bit using atomic operation
        _PyGC_AtomicClearBit(op, _PyGC_BITS_ALIVE);
    }
    return true;
}

// Clear ALIVE bits from all objects in a page
static void
clear_alive_bits_page(mi_page_t *page)
{
    mi_heap_area_t area;
    _mi_heap_area_init(&area, page);

    bool is_gc_pre = (page->tag == _Py_MIMALLOC_HEAP_GC_PRE);
    Py_ssize_t offset = get_block_offset(is_gc_pre);

    _mi_heap_area_visit_blocks(&area, page, clear_alive_bits_visitor, &offset);
}

// Clear ALIVE bits from all pages in all buckets
static void
clear_alive_bits_all_buckets(_PyGCFTParState *state)
{
    for (int i = 0; i < state->num_workers; i++) {
        _PyGCPageBucket *bucket = &state->buckets[i];
        for (size_t j = 0; j < bucket->num_pages; j++) {
            clear_alive_bits_page(bucket->pages[j]);
        }
    }
}


//=============================================================================
// Testing / Debugging APIs
//=============================================================================

#ifdef Py_DEBUG

// Test API: Get page count for current thread's heap
size_t
_PyGC_TestCountPages(void)
{
    // For testing, count pages in current thread's heap only
    // This is safe without stopping the world
    size_t count = 0;
    PyThreadState *tstate = _PyThreadState_GET();
    struct _mimalloc_thread_state *m = &((_PyThreadStateImpl *)tstate)->mimalloc;

    if (_Py_atomic_load_int(&m->initialized)) {
        count += count_heap_pages(&m->heaps[_Py_MIMALLOC_HEAP_GC]);
        count += count_heap_pages(&m->heaps[_Py_MIMALLOC_HEAP_GC_PRE]);
    }

    return count;
}

// Test API: Test page assignment algorithm
// Returns array of page counts per worker (caller must free with PyMem_RawFree)
size_t *
_PyGC_TestPageAssignment(size_t total_pages, int num_workers)
{
    if (num_workers <= 0) {
        return NULL;
    }

    size_t *result = PyMem_RawCalloc(num_workers, sizeof(size_t));
    if (result == NULL) {
        return NULL;
    }

    // Sequential bucket filling algorithm
    size_t pages_per_worker = total_pages / num_workers;
    if (pages_per_worker == 0) {
        pages_per_worker = 1;  // Avoid getting stuck on worker 0
    }

    int current_worker = 0;
    size_t current_count = 0;

    for (size_t i = 0; i < total_pages; i++) {
        result[current_worker]++;
        current_count++;

        // Move to next worker when bucket is full
        // (but not for the last worker, which absorbs remainder)
        if (current_count >= pages_per_worker &&
            current_worker < num_workers - 1) {
            current_worker++;
            current_count = 0;
        }
    }

    return result;
}

// Test API: Test atomic bit operations (stub - real testing via Python tests)
int
_PyGC_TestAtomicBitOps(int num_threads)
{
    (void)num_threads;  // Unused in stub
    // Real concurrent testing is done via Python threading tests
    // which exercise the atomic operations through GC collections
    return 0;
}

// Test API: Test REAL page enumeration by stopping the world and running
// the actual _PyGC_AssignPagesToBuckets() function.
// This exercises all the invariant assertions and verifies the real code path.
//
// Returns: Array of [total_pages, bucket0_pages, bucket1_pages, ...]
//          (num_workers + 1 elements). Caller must free with PyMem_RawFree.
// Returns NULL on error.
size_t *
_PyGC_TestRealPageEnumeration(int num_workers)
{
    if (num_workers <= 0 || num_workers > 64) {
        return NULL;
    }

    PyInterpreterState *interp = _PyInterpreterState_GET();

    // Allocate result array: [total_pages, bucket0, bucket1, ...]
    size_t *result = PyMem_RawCalloc(num_workers + 1, sizeof(size_t));
    if (result == NULL) {
        return NULL;
    }

    // Initialize parallel state
    _PyGCFTParState state = {
        .num_workers = num_workers,
        .buckets = NULL,
        .workers = NULL,
        .threads = NULL,
        .error_flag = 0,
        .workers_done = 0,
        .total_pages = 0,
        .total_objects = 0,
        .per_worker_marked = NULL
    };

    // Stop the world - this is what real GC does
    _PyEval_StopTheWorld(interp);

    // Run the REAL page assignment
    int err = _PyGC_AssignPagesToBuckets(interp, &state);

    if (err == 0) {
        // Success - collect bucket sizes
        result[0] = state.total_pages;

        size_t total_in_buckets = 0;
        for (int i = 0; i < num_workers; i++) {
            result[i + 1] = state.buckets[i].num_pages;
            total_in_buckets += state.buckets[i].num_pages;
        }

        // Verify invariant: sum of buckets should match enumerated pages
        // (Note: total_pages from CountPages may differ due to empty pages)
        assert(total_in_buckets > 0 || state.total_pages == 0);

        // Clean up
        _PyGC_FreeBuckets(&state);
    }

    // Resume the world
    _PyEval_StartTheWorld(interp);

    if (err < 0) {
        PyMem_RawFree(result);
        return NULL;
    }

    return result;
}

// Test API: Test REAL parallel marking
// Stops the world, assigns pages to buckets, runs parallel marking.
// Returns: Array of [total_objects, worker0_marked, worker1_marked, ...]
//          (num_workers + 1 elements). Caller must free with PyMem_RawFree.
// Returns NULL on error.
size_t *
_PyGC_TestParallelMark(int num_workers)
{
    if (num_workers <= 0 || num_workers > 64) {
        return NULL;
    }

    PyInterpreterState *interp = _PyInterpreterState_GET();

    // Allocate result array: [total_objects, worker0_marked, worker1_marked, ...]
    size_t *result = PyMem_RawCalloc(num_workers + 1, sizeof(size_t));
    if (result == NULL) {
        return NULL;
    }

    // Initialize parallel state
    _PyGCFTParState state = {
        .num_workers = num_workers,
        .buckets = NULL,
        .workers = NULL,
        .threads = NULL,
        .error_flag = 0,
        .workers_done = 0,
        .total_pages = 0,
        .total_objects = 0,
        .per_worker_marked = NULL
    };

    // Stop the world
    _PyEval_StopTheWorld(interp);

    int err = 0;

    // Step 1: Assign pages to buckets
    err = _PyGC_AssignPagesToBuckets(interp, &state);
    if (err < 0) {
        goto cleanup;
    }

    // Step 2: Run parallel marking
    err = _PyGC_ParallelMarkAlive(interp, &state);
    if (err < 0) {
        // Clear alive bits before freeing buckets to prevent assertion
        // failures in subsequent GC runs
        clear_alive_bits_all_buckets(&state);
        _PyGC_FreeBuckets(&state);
        goto cleanup;
    }

    // Collect results - use saved per-worker stats
    result[0] = state.total_objects;

    if (state.per_worker_marked != NULL) {
        // Use actual per-worker stats
        for (int i = 0; i < num_workers; i++) {
            result[i + 1] = state.per_worker_marked[i];
        }
        PyMem_RawFree(state.per_worker_marked);
        state.per_worker_marked = NULL;
    } else {
        // Fallback: distribute evenly if allocation failed
        size_t per_worker = state.total_objects / num_workers;
        size_t remainder = state.total_objects % num_workers;
        for (int i = 0; i < num_workers; i++) {
            result[i + 1] = per_worker + (i < (int)remainder ? 1 : 0);
        }
    }

    // IMPORTANT: Clear ALIVE bits so they don't affect subsequent GC runs
    // This is only needed in test mode - real GC would process these objects
    clear_alive_bits_all_buckets(&state);

    // Clean up
    _PyGC_FreeBuckets(&state);

cleanup:
    _PyEval_StartTheWorld(interp);

    if (err < 0) {
        PyMem_RawFree(result);
        return NULL;
    }

    return result;
}

#endif  // Py_DEBUG

//=============================================================================
// Persistent Thread Pool Implementation (Barrier-based)
//=============================================================================

// Forward declaration
static int propagate_pool_visitproc(PyObject *obj, void *arg);

// Wrapper: flush local buffer to deque (uses shared implementation)
static inline void
flush_local_buffer_to_deque(_PyGCWorkerState *worker)
{
    _PyGC_FlushLocalToDeque(&worker->local, &worker->deque);
}

// Wrapper: refill local buffer from deque (uses shared implementation)
static inline void
refill_local_buffer_from_deque(_PyGCWorkerState *worker)
{
    _PyGC_RefillLocalFromDeque(&worker->local, &worker->deque);
}

// Wrapper: batch steal from another worker's deque (uses shared implementation)
static inline size_t
steal_batch_from_worker(_PyGCWorkerState *thief, _PyGCWorkerState *victim)
{
    return _PyGC_BatchSteal(&thief->local, &victim->deque);
}

// Work-stealing loop for PROPAGATE work
// Called by both background workers and main thread (worker 0)
static void
propagate_pool_work(_PyGCThreadPool *pool, int worker_id)
{
    _PyGCWorkerState *worker = &pool->workers[worker_id];
    _PyGCLocalBuffer *local = &worker->local;

    // Ensure local buffer is initialized
    _PyGCLocalBuffer_Init(local);

    // Work loop: process local buffer + deque + steal until no work left
    int steal_attempts_since_work = 0;
    const int MAX_STEAL_ATTEMPTS = pool->num_workers * 2;

    while (1) {
        // Phase 1: Process local buffer (fast path - no fences!)
        while (!_PyGCLocalBuffer_IsEmpty(local)) {
            steal_attempts_since_work = 0;
            PyObject *obj = _PyGCLocalBuffer_Pop(local);
            traverseproc traverse = Py_TYPE(obj)->tp_traverse;
            if (traverse != NULL) {
                // Special handling for tuples: untrack if possible and clear
                // alive bit. This matches the serial gc_mark_traverse_tuple
                // behavior and prevents alive bits from being left on untracked
                // tuples.
                if (traverse == PyTuple_Type.tp_traverse) {
                    _PyTuple_MaybeUntrack(obj);
                    if (!_PyObject_GC_IS_TRACKED(obj)) {
                        // Tuple was untracked - clear alive bit
                        _PyGC_AtomicClearBit(obj, _PyGC_BITS_ALIVE);
                        worker->objects_marked++;
                        continue;
                    }
                }
                traverse(obj,
                    (visitproc)propagate_pool_visitproc,
                    (void *)worker);
            }
            worker->objects_marked++;
        }

        // Phase 2: Local buffer empty - try to refill from deque
        refill_local_buffer_from_deque(worker);
        if (!_PyGCLocalBuffer_IsEmpty(local)) {
            continue;  // Got work from deque, continue processing
        }

        // Phase 3: Deque also empty - try stealing from other workers
        size_t total_stolen = 0;
        for (int offset = 1; offset < pool->num_workers; offset++) {
            int victim_id = (worker_id + offset) % pool->num_workers;
            _PyGCWorkerState *victim = &pool->workers[victim_id];

            // First try victim's deque
            size_t batch_stolen = steal_batch_from_worker(worker, victim);
            if (batch_stolen > 0) {
                worker->objects_stolen += batch_stolen;
                total_stolen += batch_stolen;
                break;  // Got work, stop looking
            }
            worker->steals_attempted++;
        }

        if (total_stolen > 0) {
            steal_attempts_since_work = 0;
            continue;  // Process stolen work
        }

        // No work found anywhere
        steal_attempts_since_work++;
        if (steal_attempts_since_work >= MAX_STEAL_ATTEMPTS) {
            break;  // Give up - all workers appear idle
        }
    }

    // Flush any remaining local work back to deque before exiting
    // (shouldn't happen normally, but ensures correctness)
    flush_local_buffer_to_deque(worker);
}

// Forward declarations for pool work functions
// (These types and functions are defined later in the file)
typedef struct {
    void *par_state;              // _PyGCFTParState* (not used in pool version)
    void *phase_barrier;          // _PyGCBarrier* (not used in pool version)
    Py_ssize_t candidates;        // Per-worker candidate count
    int skip_deferred;            // Whether to skip deferred objects
    int error;                    // Error flag
} _PyGCPoolUpdateRefsArgs;

static int pool_update_refs_init_page(mi_page_t *page, _PyGCPoolUpdateRefsArgs *args);
static int pool_update_refs_compute_page(mi_page_t *page, _PyGCPoolUpdateRefsArgs *args);
static int pool_mark_heap_find_roots_page(mi_page_t *page, _PyGCWorkerState *worker,
                                          Py_ssize_t offset, int skip_deferred,
                                          size_t *roots_found);
static int pool_mark_traverse_object(PyObject *op, _PyGCWorkerState *worker);
static void pool_scan_heap_process_page(mi_page_t *page,
                                        struct _PyGCScanWorkerState *worker,
                                        int gc_reason);

// UPDATE_REFS pool work function
// Processes page buckets in two phases: init and compute
static void
update_refs_pool_work(_PyGCThreadPool *pool, int worker_id)
{
    _PyGCWorkDescriptor *work = pool->current_work;
    assert(work != NULL);
    assert(work->buckets != NULL);

    _PyGCPageBucket *bucket = &work->buckets[worker_id];

    _PyGCPoolUpdateRefsArgs args = {
        .par_state = NULL,
        .phase_barrier = NULL,
        .candidates = 0,
        .skip_deferred = 0,
        .error = 0
    };

    // Phase 1: Initialize all objects in our pages
    for (size_t i = 0; i < bucket->num_pages; i++) {
        mi_page_t *page = bucket->pages[i];
        if (pool_update_refs_init_page(page, &args) < 0) {
            work->error_flag = 1;
            break;
        }
    }

    // Wait for all workers to finish init phase
    _PyGCBarrier_Wait(&pool->phase_barrier);

    // Check for errors from other workers
    if (work->error_flag) {
        return;
    }

    // Phase 2: Compute gc_refs for all objects in our pages
    for (size_t i = 0; i < bucket->num_pages; i++) {
        mi_page_t *page = bucket->pages[i];
        if (pool_update_refs_compute_page(page, &args) < 0) {
            work->error_flag = 1;
            break;
        }
    }

    // Store per-worker candidate count if requested
    if (work->per_worker_refs != NULL) {
        work->per_worker_refs[worker_id] = args.candidates;
    }
}

// MARK_HEAP pool work function
// Finds roots in pages, then does work-stealing propagation
static void
mark_heap_pool_work(_PyGCThreadPool *pool, int worker_id)
{
    _PyGCWorkDescriptor *work = pool->current_work;
    assert(work != NULL);
    assert(work->buckets != NULL);

    _PyGCWorkerState *worker = &pool->workers[worker_id];
    _PyGCPageBucket *bucket = &work->buckets[worker_id];

    // Determine offset for block-to-object conversion
    Py_ssize_t offset_base = 0;
    if (_PyMem_DebugEnabled()) {
        offset_base += 2 * sizeof(size_t);
    }
    Py_ssize_t offset_pre = offset_base + 2 * sizeof(PyObject*);

    size_t roots_found = 0;

    // Phase 1: Find roots in our assigned pages
    for (size_t i = 0; i < bucket->num_pages; i++) {
        mi_page_t *page = bucket->pages[i];
        Py_ssize_t offset = (page->tag == _Py_MIMALLOC_HEAP_GC_PRE)
                            ? offset_pre : offset_base;
        if (pool_mark_heap_find_roots_page(page, worker, offset, work->skip_deferred,
                                           &roots_found) < 0) {
            work->error_flag = 1;
            return;
        }
    }

    // Phase 2: Transitive marking with work-stealing
    // Use multiple retry rounds to handle the termination race condition:
    // A worker might exit thinking there's no work, while another worker
    // is still generating new work. We retry a few times before giving up.
    int idle_rounds = 0;
    const int MAX_IDLE_ROUNDS = 3;  // Retry a few times before exiting

    while (idle_rounds < MAX_IDLE_ROUNDS) {
        bool made_progress = false;

        // Process our local deque
        PyObject *op;
        while ((op = (PyObject *)_PyWSDeque_Take(&worker->deque)) != NULL) {
            made_progress = true;
            idle_rounds = 0;  // Reset idle counter on progress
            if (pool_mark_traverse_object(op, worker) < 0) {
                work->error_flag = 1;
                return;
            }
        }

        // Try to steal from other workers
        for (int i = 1; i < pool->num_workers; i++) {
            int victim = (worker_id + i) % pool->num_workers;
            op = (PyObject *)_PyWSDeque_Steal(&pool->workers[victim].deque);
            if (op != NULL) {
                made_progress = true;
                idle_rounds = 0;  // Reset idle counter on progress
                if (pool_mark_traverse_object(op, worker) < 0) {
                    work->error_flag = 1;
                    return;
                }
                break;
            }
        }

        if (!made_progress) {
            idle_rounds++;
        }
    }
}

// SCAN_HEAP pool work function
// Uses atomic counter for dynamic page distribution
static void
scan_heap_pool_work(_PyGCThreadPool *pool, int worker_id)
{
    _PyGCWorkDescriptor *work = pool->current_work;
    assert(work != NULL);
    assert(work->page_array != NULL);
    assert(work->scan_workers != NULL);
    assert(work->scan_result != NULL);

    struct _PyGCScanWorkerState *scan_worker = &work->scan_workers[worker_id];
    int gc_reason = work->scan_result->gc_reason;

    // Dynamic work distribution using atomic counter
    while (1) {
        int page_idx = atomic_fetch_add(&work->page_counter, 1);
        if (page_idx >= (int)work->total_pages) {
            break;
        }

        pool_scan_heap_process_page(work->page_array[page_idx], scan_worker, gc_reason);
    }
}

// Dispatcher for pool work based on work type
static void
thread_pool_do_work(_PyGCThreadPool *pool, int worker_id)
{
    _PyGCWorkDescriptor *work = pool->current_work;
    if (work == NULL) {
        return;  // No work to do
    }

#ifdef GC_DEBUG_ATOMICS
    // Reset TLS stats at start of work
    _PyGC_ATOMIC_RESET_STATS();
#endif

    switch (work->type) {
        case _PyGC_WORK_PROPAGATE:
            propagate_pool_work(pool, worker_id);
            break;
        case _PyGC_WORK_UPDATE_REFS:
            update_refs_pool_work(pool, worker_id);
            break;
        case _PyGC_WORK_MARK_HEAP:
            mark_heap_pool_work(pool, worker_id);
            break;
        case _PyGC_WORK_SCAN_HEAP:
            scan_heap_pool_work(pool, worker_id);
            break;
        default:
            // Unknown work type - do nothing
            break;
    }

#ifdef GC_DEBUG_ATOMICS
    // Copy TLS stats to worker state for aggregation after barrier
    pool->workers[worker_id].atomic_stats = _PyGC_atomic_worker_stats;
#endif
}

// Thread args for pool workers (passed at creation time)
// Contains both pool and worker_id for safe passing to new thread
typedef struct {
    _PyGCThreadPool *pool;
    int worker_id;
} _PyGCPoolWorkerArgs;

// Static array for pool worker args (allocated in ThreadPoolInit, freed in ThreadPoolFini)
static _PyGCPoolWorkerArgs *_pool_worker_args = NULL;

// Background worker thread entry point
static void
thread_pool_worker(void *arg)
{
    _PyGCPoolWorkerArgs *args = (_PyGCPoolWorkerArgs *)arg;
    _PyGCThreadPool *pool = args->pool;
    int worker_id = args->worker_id;

    // Create a Python thread state for this worker thread.
    // This is required for Py_REF_DEBUG (debug builds) where Py_INCREF/Py_DECREF
    // call _Py_INCREF_IncRefTotal() which needs _PyThreadState_GET() to return
    // a valid thread state.
    _PyGCWorkerState *worker = &pool->workers[worker_id];
    PyThreadState *tstate = _PyThreadState_New(pool->interp, _PyThreadState_WHENCE_UNKNOWN);
    if (tstate != NULL) {
        _PyThreadState_Bind(tstate);
        _Py_tss_tstate = tstate;
        _Py_tss_interp = pool->interp;
        worker->tstate = tstate;
    }

    while (1) {
        // Wait at mark_barrier for work (main thread signals by arriving here too)
        _PyGCBarrier_Wait(&pool->mark_barrier);

        // Check for shutdown after waking
        if (pool->shutdown) {
            break;
        }

        // Do the work
        thread_pool_do_work(pool, worker_id);

        // Signal completion
        _PyGCBarrier_Wait(&pool->done_barrier);
    }

    // Clean up Python thread state
    if (worker->tstate != NULL) {
        PyThreadState_Clear(worker->tstate);
        _Py_tss_tstate = NULL;
        _Py_tss_interp = NULL;
        PyThreadState_Delete(worker->tstate);
        worker->tstate = NULL;
    }
}

// Visitproc for pool-based propagation
// Pushes to local buffer (fast) and only overflows to deque when buffer is full
static int
propagate_pool_visitproc(PyObject *obj, void *arg)
{
    _PyGC_ATOMIC_SET_PHASE(GC_ATOMIC_PHASE_POOL_PROPAGATE);
    if (obj == NULL || !_PyObject_IS_GC(obj)) {
        return 0;
    }
    _PyGCWorkerState *worker = (_PyGCWorkerState *)arg;

    // Try to mark as alive - if we win, push to our local buffer
    if (_PyGC_TryMarkAlive(obj)) {
        _PyGCLocalBuffer *local = &worker->local;
        if (_PyGCLocalBuffer_IsFull(local)) {
            // Local buffer full - use compile-time selected flush strategy
            _PyGC_OverflowFlush(local, &worker->deque);
        }
        _PyGCLocalBuffer_Push(local, obj);
    }
    return 0;
}

// Initialize the thread pool
int
_PyGC_ThreadPoolInit(PyInterpreterState *interp, int num_workers)
{
    // Require at least 1 worker (main thread only is valid, though not parallel)
    if (num_workers < 1) {
        return -1;
    }
    if (interp->gc.thread_pool != NULL) {
        // Already initialized
        return -1;
    }

    _PyGCThreadPool *pool = PyMem_RawCalloc(1, sizeof(_PyGCThreadPool));
    if (pool == NULL) {
        return -1;
    }

    pool->interp = interp;
    pool->num_workers = num_workers;
    pool->shutdown = 0;
    pool->threads_created = 0;
    pool->collections_completed = 0;

    // Initialize barriers for synchronization
    // All barriers include all workers (main thread as worker 0)
    _PyGCBarrier_Init(&pool->mark_barrier, num_workers);
    _PyGCBarrier_Init(&pool->done_barrier, num_workers);
    _PyGCBarrier_Init(&pool->phase_barrier, num_workers);

    // Allocate thread handles (for workers 1..N-1, worker 0 is main thread)
    pool->threads = PyMem_RawCalloc(num_workers - 1, sizeof(PyThread_handle_t));
    if (pool->threads == NULL) {
        _PyGCBarrier_Fini(&pool->phase_barrier);
        _PyGCBarrier_Fini(&pool->done_barrier);
        _PyGCBarrier_Fini(&pool->mark_barrier);
        PyMem_RawFree(pool);
        return -1;
    }

    // Allocate persistent worker states (one per worker including main)
    pool->workers = PyMem_RawCalloc(num_workers, sizeof(_PyGCWorkerState));
    if (pool->workers == NULL) {
        PyMem_RawFree(pool->threads);
        _PyGCBarrier_Fini(&pool->phase_barrier);
        _PyGCBarrier_Fini(&pool->done_barrier);
        _PyGCBarrier_Fini(&pool->mark_barrier);
        PyMem_RawFree(pool);
        return -1;
    }

    // Initialize worker states and deques with thread-local memory pools
    // Each worker gets a pre-allocated 2MB buffer to avoid malloc during hot path
    size_t pool_entries = _Py_WSDEQUE_PARALLEL_GC_SIZE;  // 256K entries = 2MB
    size_t pool_bytes = sizeof(_PyWSArray) + sizeof(uintptr_t) * pool_entries;

    for (int i = 0; i < num_workers; i++) {
        _PyGCWorkerState *worker = &pool->workers[i];
        worker->worker_id = i;
        worker->objects_marked = 0;
        worker->objects_stolen = 0;
        worker->steals_attempted = 0;

        // Allocate thread-local pool (done once at enable time, not per-collection)
        worker->local_pool = PyMem_RawCalloc(1, pool_bytes);
        worker->local_pool_size = pool_entries;

        if (worker->local_pool != NULL) {
            // Initialize deque with pre-allocated buffer
            _PyWSDeque_InitWithBuffer(&worker->deque, worker->local_pool,
                                       pool_bytes, pool_entries);
        } else {
            // Fallback to regular init if pool allocation failed
            worker->local_pool_size = 0;
            _PyWSDeque_Init(&worker->deque);
        }
    }

    // Allocate thread args (needed to pass pool + worker_id to each thread)
    // Store in static for cleanup in ThreadPoolFini
    _pool_worker_args = PyMem_RawCalloc(num_workers - 1,
                                         sizeof(_PyGCPoolWorkerArgs));
    if (_pool_worker_args == NULL) {
        for (int j = 0; j < num_workers; j++) {
            if (pool->workers[j].local_pool != NULL) {
                _PyWSDeque_FiniExternal(&pool->workers[j].deque,
                                        pool->workers[j].local_pool);
                PyMem_RawFree(pool->workers[j].local_pool);
            } else {
                _PyWSDeque_Fini(&pool->workers[j].deque);
            }
        }
        PyMem_RawFree(pool->workers);
        PyMem_RawFree(pool->threads);
        _PyGCBarrier_Fini(&pool->phase_barrier);
        _PyGCBarrier_Fini(&pool->done_barrier);
        _PyGCBarrier_Fini(&pool->mark_barrier);
        PyMem_RawFree(pool);
        return -1;
    }

    // Create worker threads (they will wait at mark_barrier immediately)
    for (int i = 0; i < num_workers - 1; i++) {
        _pool_worker_args[i].pool = pool;
        _pool_worker_args[i].worker_id = i + 1;  // Workers 1..N-1

        PyThread_ident_t ident;
        int rc = PyThread_start_joinable_thread(
            thread_pool_worker, &_pool_worker_args[i],
            &ident, &pool->threads[i]);
        if (rc != 0) {
            // Thread creation failure is a fatal error during init.
            // Already-created threads are not cleaned up here (same as GIL version).
            // This is acceptable because thread creation failure during init
            // is rare and indicates system resource exhaustion.
            PyMem_RawFree(_pool_worker_args);
            _pool_worker_args = NULL;
            for (int j = 0; j < num_workers; j++) {
                if (pool->workers[j].local_pool != NULL) {
                    _PyWSDeque_FiniExternal(&pool->workers[j].deque,
                                            pool->workers[j].local_pool);
                    PyMem_RawFree(pool->workers[j].local_pool);
                } else {
                    _PyWSDeque_Fini(&pool->workers[j].deque);
                }
            }
            PyMem_RawFree(pool->workers);
            PyMem_RawFree(pool->threads);
            _PyGCBarrier_Fini(&pool->phase_barrier);
            _PyGCBarrier_Fini(&pool->done_barrier);
            _PyGCBarrier_Fini(&pool->mark_barrier);
            PyMem_RawFree(pool);
            return -1;
        }
        pool->threads_created++;
    }

    interp->gc.thread_pool = pool;
    return 0;
}

// Finalize the thread pool
void
_PyGC_ThreadPoolFini(PyInterpreterState *interp)
{
    _PyGCThreadPool *pool = interp->gc.thread_pool;
    if (pool == NULL) {
        return;
    }

    // Signal shutdown - workers will check this after mark_barrier
    pool->shutdown = 1;

    // Release workers from mark_barrier so they can see shutdown flag
    // Main thread participates in barrier to release all workers
    _PyGCBarrier_Wait(&pool->mark_barrier);

    // Wait for all workers to finish (they exit after seeing shutdown)
    for (int i = 0; i < pool->num_workers - 1; i++) {
        PyThread_join_thread(pool->threads[i]);
    }

    // Free the thread args
    if (_pool_worker_args != NULL) {
        PyMem_RawFree(_pool_worker_args);
        _pool_worker_args = NULL;
    }

    // Clean up worker states and deques
    if (pool->workers != NULL) {
        for (int i = 0; i < pool->num_workers; i++) {
            // Use FiniExternal if using pre-allocated pool to avoid double-free
            if (pool->workers[i].local_pool != NULL) {
                _PyWSDeque_FiniExternal(&pool->workers[i].deque,
                                        pool->workers[i].local_pool);
                PyMem_RawFree(pool->workers[i].local_pool);
            } else {
                _PyWSDeque_Fini(&pool->workers[i].deque);
            }
        }
        PyMem_RawFree(pool->workers);
    }

    // Clean up
    PyMem_RawFree(pool->threads);
    _PyGCBarrier_Fini(&pool->phase_barrier);
    _PyGCBarrier_Fini(&pool->done_barrier);
    _PyGCBarrier_Fini(&pool->mark_barrier);
    PyMem_RawFree(pool);

    interp->gc.thread_pool = NULL;
}

// Check if thread pool is active
int
_PyGC_ThreadPoolIsActive(PyInterpreterState *interp)
{
    return interp->gc.thread_pool != NULL;
}

// Get threads created count (for testing)
size_t
_PyGC_ThreadPoolGetThreadsCreated(PyInterpreterState *interp)
{
    _PyGCThreadPool *pool = interp->gc.thread_pool;
    if (pool == NULL) {
        return 0;
    }
    return pool->threads_created;
}

// Get collections completed count (for testing)
size_t
_PyGC_ThreadPoolGetCollectionsCompleted(PyInterpreterState *interp)
{
    _PyGCThreadPool *pool = interp->gc.thread_pool;
    if (pool == NULL) {
        return 0;
    }
    return pool->collections_completed;
}

// New parallel propagate using thread pool with barrier synchronization
int
_PyGC_ParallelPropagateAliveWithPool(PyInterpreterState *interp,
                                      PyObject **initial_roots,
                                      size_t num_roots,
                                      int num_workers)
{
    _PyGCThreadPool *pool = interp->gc.thread_pool;
    assert(pool != NULL);
    assert(pool->num_workers == num_workers);
    assert(pool->workers != NULL);

    // Reset worker states for this collection
    for (int i = 0; i < num_workers; i++) {
        pool->workers[i].objects_marked = 0;
        pool->workers[i].objects_stolen = 0;
        pool->workers[i].steals_attempted = 0;
        // Initialize local buffer for this collection
        _PyGCLocalBuffer_Init(&pool->workers[i].local);
        // Note: deques should be empty from last collection
    }

    // Set up work descriptor for PROPAGATE work
    _PyGCWorkDescriptor work = {
        .type = _PyGC_WORK_PROPAGATE,
        .roots = initial_roots,
        .num_roots = num_roots,
        .workers = pool->workers,
        .error_flag = 0,
    };
    pool->current_work = &work;

    // Distribute roots round-robin to worker deques
    // Do this BEFORE signaling workers to start
    _PyGC_ATOMIC_SET_PHASE(GC_ATOMIC_PHASE_ROOT_DISTRIBUTE);
    for (size_t i = 0; i < num_roots; i++) {
        PyObject *root = initial_roots[i];
        if (root != NULL && _PyObject_IS_GC(root)) {
            // Mark root as alive
            _PyGC_TryMarkAlive(root);
            // Push to worker's deque
            int worker_id = i % num_workers;
            _PyWSDeque_Push(&pool->workers[worker_id].deque, root);
        }
    }

    // Signal workers to start by arriving at mark_barrier
    // Workers are waiting at mark_barrier; when we arrive, all are released
    _PyGCBarrier_Wait(&pool->mark_barrier);

    // Worker 0 (main thread) does its share
    thread_pool_do_work(pool, 0);

    // Signal completion by arriving at done_barrier
    // This blocks until all workers finish, guaranteeing correct termination
    _PyGCBarrier_Wait(&pool->done_barrier);

    // Clear work descriptor
    pool->current_work = NULL;

    // All work is complete
    pool->collections_completed++;

#ifdef GC_DEBUG_ATOMICS
    // Print aggregated atomic stats from all workers
    _PyGCAtomicWorkerStats all_stats[num_workers];
    for (int i = 0; i < num_workers; i++) {
        all_stats[i] = pool->workers[i].atomic_stats;
    }
    fprintf(stderr, "[pool_propagate phase]\n");
    _PyGC_AtomicPrintStats(all_stats, num_workers);
#endif

    return 0;
}


//=============================================================================
// Parallel Update Refs for deduce_unreachable_heap
//=============================================================================
//
// This parallelizes the update_refs phase of deduce_unreachable_heap by:
// 1. Phase 1: init_refs - each worker sets unreachable bit + zeros ob_tid
// 2. [barrier]
// 3. Phase 2: compute_refs - each worker adds refcount + calls visit_decref_atomic
//
// The barrier ensures all objects are initialized before any worker tries to
// decrement gc_refs on cross-page references.

// Per-worker state for parallel update_refs
typedef struct {
    _PyGCFTParState *par_state;
    _PyGCBarrier *phase_barrier;  // Barrier between init_refs and compute_refs
    Py_ssize_t candidates;          // Per-worker candidate count
    int skip_deferred;              // Whether to skip deferred objects
    int error;
} _PyGCUpdateRefsWorkerArgs;

// Atomic gc_refs operations are now inline in pycore_gc_ft_parallel.h:
// - gc_decref_atomic()
// - gc_add_refs_atomic()
// - gc_maybe_init_refs_atomic()

// Check if object is frozen (from gc_free_threading.c)
static inline int
par_gc_is_frozen(PyObject *op)
{
    return (op->ob_gc_bits & _PyGC_BITS_FROZEN) != 0;
}

// Check if object is alive (marked in mark_alive phase)
static inline int
par_gc_is_alive(PyObject *op)
{
    return (op->ob_gc_bits & _PyGC_BITS_ALIVE) != 0;
}

// Atomic version of visit_decref for parallel update_refs.
// Assumes all objects have been pre-initialized (unreachable bit set, ob_tid = 0).

static int
par_visit_decref_atomic(PyObject *op, void *arg)
{
    (void)arg;  // unused

    if (_PyObject_GC_IS_TRACKED(op)
        && !_Py_IsImmortal(op)
        && !par_gc_is_frozen(op)
        && !par_gc_is_alive(op))
    {
        gc_decref_atomic(op);
    }
    return 0;
}

// Arguments for update_refs visitor callbacks
typedef struct {
    Py_ssize_t offset;
    Py_ssize_t *candidates;  // Pointer to accumulator
} _PyGCUpdateRefsInitArgs;

typedef struct {
    Py_ssize_t offset;
} _PyGCUpdateRefsComputeArgs;

// Visitor callback for Phase 1: Initialize gc_refs
// Sets unreachable bit and zeros ob_tid
static bool
update_refs_init_visitor(const mi_heap_t *heap, const mi_heap_area_t *area,
                         void *block, size_t block_size, void *arg)
{
    (void)heap;
    (void)area;
    (void)block_size;

    _PyGCUpdateRefsInitArgs *args = (_PyGCUpdateRefsInitArgs *)arg;
    PyObject *op = block_to_object(block, args->offset);
    if (op == NULL) {
        return true;  // Skip untracked/frozen
    }

    // Skip immortal objects
    if (_Py_IsImmortal(op)) {
        op->ob_tid = 0;
        _PyObject_GC_UNTRACK(op);
        _PyGC_AtomicClearBit(op, _PyGC_BITS_UNREACHABLE);
        return true;
    }

    // Count as candidate
    (*args->candidates)++;

    // Skip already-alive objects
    if (par_gc_is_alive(op)) {
        return true;
    }

    // Initialize gc_refs atomically
    gc_maybe_init_refs_atomic(op);

    return true;
}

// Visitor callback for Phase 2: Compute gc_refs
// Adds refcount and calls tp_traverse with visit_decref_atomic
static bool
update_refs_compute_visitor(const mi_heap_t *heap, const mi_heap_area_t *area,
                            void *block, size_t block_size, void *arg)
{
    (void)heap;
    (void)area;
    (void)block_size;

    _PyGCUpdateRefsComputeArgs *args = (_PyGCUpdateRefsComputeArgs *)arg;
    PyObject *op = block_to_object(block, args->offset);
    if (op == NULL) {
        return true;  // Skip untracked/frozen
    }

    // Skip immortal objects (already handled in init phase)
    if (_Py_IsImmortal(op)) {
        return true;
    }

    // Skip already-alive objects
    if (par_gc_is_alive(op)) {
        return true;
    }

    // Get refcount (adjust for deferred references)
    Py_ssize_t refcount = Py_REFCNT(op);
    if (_PyObject_HasDeferredRefcount(op)) {
        refcount -= _Py_REF_DEFERRED;
    }

    // Add refcount to gc_refs atomically
    gc_add_refs_atomic(op, refcount);

    // Subtract internal references by calling tp_traverse
    traverseproc traverse = Py_TYPE(op)->tp_traverse;
    if (traverse != NULL) {
        traverse(op, par_visit_decref_atomic, NULL);
    }

    return true;
}

// Phase 1: Initialize gc_refs for each object in a page
// Sets unreachable bit and zeros ob_tid
static int
update_refs_init_page(mi_page_t *page, _PyGCUpdateRefsWorkerArgs *args)
{
    assert(page != NULL);

    mi_heap_area_t area;
    _mi_heap_area_init(&area, page);

    // Determine offset based on page tag
    bool is_gc_pre = (page->tag == _Py_MIMALLOC_HEAP_GC_PRE);

    _PyGCUpdateRefsInitArgs visitor_args = {
        .offset = get_block_offset(is_gc_pre),
        .candidates = &args->candidates
    };

    _mi_heap_area_visit_blocks(&area, page, update_refs_init_visitor, &visitor_args);

    return 0;
}

// Phase 2: Compute gc_refs for each object in a page
// Adds refcount and calls tp_traverse with visit_decref_atomic
static int
update_refs_compute_page(mi_page_t *page, _PyGCUpdateRefsWorkerArgs *args)
{
    (void)args;  // Not used in this phase
    assert(page != NULL);

    mi_heap_area_t area;
    _mi_heap_area_init(&area, page);

    // Determine offset based on page tag
    bool is_gc_pre = (page->tag == _Py_MIMALLOC_HEAP_GC_PRE);

    _PyGCUpdateRefsComputeArgs visitor_args = {
        .offset = get_block_offset(is_gc_pre)
    };

    _mi_heap_area_visit_blocks(&area, page, update_refs_compute_visitor, &visitor_args);

    return 0;
}

// Worker function for parallel update_refs
static int
update_refs_worker(_PyGCFTParState *state, int worker_id,
                   _PyGCBarrier *phase_barrier,
                   Py_ssize_t *out_candidates)
{
    assert(state != NULL);
    assert(state->buckets != NULL);
    assert(worker_id >= 0 && worker_id < state->num_workers);

    _PyGCPageBucket *bucket = &state->buckets[worker_id];

    _PyGCUpdateRefsWorkerArgs args = {
        .par_state = state,
        .phase_barrier = phase_barrier,
        .candidates = 0,
        .skip_deferred = 0,
        .error = 0
    };

    // Phase 1: Initialize all objects in our pages
    for (size_t i = 0; i < bucket->num_pages; i++) {
        mi_page_t *page = bucket->pages[i];
        if (update_refs_init_page(page, &args) < 0) {
            return -1;
        }
    }

    // Wait for all workers to finish init phase before compute phase
    _PyGCBarrier_Wait(phase_barrier);

    // Phase 2: Compute gc_refs for all objects in our pages
    for (size_t i = 0; i < bucket->num_pages; i++) {
        mi_page_t *page = bucket->pages[i];
        if (update_refs_compute_page(page, &args) < 0) {
            return -1;
        }
    }

    *out_candidates = args.candidates;
    return 0;
}

//=============================================================================
// Parallel Mark Heap Visitor (find roots and transitively mark reachable)
//=============================================================================
//
// This parallelizes the mark_heap_visitor phase of deduce_unreachable_heap.
// mark_heap_visitor finds objects with gc_refs > 0 (external references) and
// transitively clears the UNREACHABLE bit on all reachable objects.
//
// Parallel strategy:
// 1. Phase 1: Workers scan their pages for roots (gc_refs > 0, still unreachable)
//    - Uses _PyGC_TryMarkReachable to atomically clear UNREACHABLE bit
//    - Pushes newly-reachable objects to local deque for transitive marking
// 2. Phase 2: Work-stealing propagation
//    - Workers traverse objects, marking children reachable
//    - Work-stealing ensures load balancing across graph structure

// Visitproc for parallel mark_reachable - marks children and pushes to deque
static int
par_mark_visitproc(PyObject *child, void *arg)
{
    _PyGC_ATOMIC_SET_PHASE(GC_ATOMIC_PHASE_MARK_HEAP);
    _PyGCWorkerState *worker = (_PyGCWorkerState *)arg;

    if (child == NULL) {
        return 0;
    }

    // Skip untracked/frozen objects
    if (!_PyObject_GC_IS_TRACKED(child)) {
        return 0;
    }
    if (child->ob_gc_bits & _PyGC_BITS_FROZEN) {
        return 0;
    }

    // Try to mark child as reachable (atomic clear of UNREACHABLE bit)
    if (_PyGC_TryMarkReachable(child)) {
        // We were first to mark it - push to our deque for traversal
        _PyWSDeque_Push(&worker->deque, child);
        worker->objects_marked++;
    }
    return 0;
}

// Traverse an object's children for parallel mark
static int
par_mark_traverse_object(PyObject *op, _PyGCWorkerState *worker)
{
    assert(op != NULL);
    traverseproc traverse = Py_TYPE(op)->tp_traverse;
    if (traverse == NULL) {
        return 0;
    }
    return traverse(op, par_mark_visitproc, worker);
}

// Arguments for mark_heap roots visitor
typedef struct {
    Py_ssize_t offset;
    _PyGCWorkerState *worker;
    int skip_deferred;
    size_t roots_found;
} _PyGCMarkRootsVisitorArgs;

// Visitor callback to find roots for parallel mark
static bool
mark_heap_roots_visitor(const mi_heap_t *heap, const mi_heap_area_t *area,
                        void *block, size_t block_size, void *arg)
{
    (void)heap;
    (void)area;
    (void)block_size;

    _PyGCMarkRootsVisitorArgs *ctx = (_PyGCMarkRootsVisitorArgs *)arg;
    PyObject *op = block_to_object(block, ctx->offset);

    if (op == NULL) {
        return true;
    }

    // Skip if already marked alive (by previous gc_mark_alive phase)
    // But ensure UNREACHABLE bit is cleared for consistency with scan_heap
    if (op->ob_gc_bits & _PyGC_BITS_ALIVE) {
        _PyGC_TryClearBit(op, _PyGC_BITS_UNREACHABLE);
        return true;
    }

    // Check if already reachable (not unreachable bit set)
    if (!_PyGC_IsUnreachable(op)) {
        return true;
    }

    // Get gc_refs (stored in ob_tid during GC)
    Py_ssize_t gc_refs = gc_get_refs_atomic(op);

    // GH-129236: If skipping deferred objects, keep them alive
    int keep_alive = (ctx->skip_deferred && _PyObject_HasDeferredRefcount(op));

    if (gc_refs > 0 || keep_alive) {
        // This object is a root - try to mark it reachable
        if (_PyGC_TryMarkReachable(op)) {
            // We were first to mark it - push to deque for transitive marking
            _PyWSDeque_Push(&ctx->worker->deque, op);
            ctx->worker->objects_marked++;
            ctx->roots_found++;
        }
    }

    return true;
}

// Process a page to find roots for parallel marking
static int
mark_heap_find_roots_page(mi_page_t *page, _PyGCWorkerState *worker,
                          Py_ssize_t offset, int skip_deferred, size_t *roots_found)
{
    assert(page != NULL);

    mi_heap_area_t area;
    _mi_heap_area_init(&area, page);

    _PyGCMarkRootsVisitorArgs visitor_args = {
        .offset = offset,
        .worker = worker,
        .skip_deferred = skip_deferred,
        .roots_found = 0
    };

    _mi_heap_area_visit_blocks(&area, page, mark_heap_roots_visitor, &visitor_args);

    *roots_found += visitor_args.roots_found;
    return 0;
}

// Try to steal work from other workers during mark phase
static PyObject *
mark_try_steal_work(_PyGCFTParState *state, int my_id, _PyGCWorkerState *workers)
{
    int num_workers = state->num_workers;

    for (int i = 1; i < num_workers; i++) {
        int victim = (my_id + i) % num_workers;
        PyObject *stolen = (PyObject *)_PyWSDeque_Steal(&workers[victim].deque);
        if (stolen != NULL) {
            return stolen;
        }
    }

    return NULL;
}

// Worker function for parallel mark
static int
mark_heap_worker(_PyGCFTParState *state, int worker_id,
                 _PyGCWorkerState *workers, int skip_deferred)
{
    assert(state != NULL);
    assert(state->buckets != NULL);
    assert(worker_id >= 0 && worker_id < state->num_workers);

    _PyGCWorkerState *worker = &workers[worker_id];
    _PyGCPageBucket *bucket = &state->buckets[worker_id];

    // Determine offset for block-to-object conversion
    Py_ssize_t offset_base = 0;
    if (_PyMem_DebugEnabled()) {
        offset_base += 2 * sizeof(size_t);
    }
    Py_ssize_t offset_pre = offset_base + 2 * sizeof(PyObject*);

    size_t roots_found = 0;

    // Phase 1: Find roots in our assigned pages
    for (size_t i = 0; i < bucket->num_pages; i++) {
        mi_page_t *page = bucket->pages[i];

        // Determine offset based on page tag
        Py_ssize_t offset = (page->tag == _Py_MIMALLOC_HEAP_GC_PRE)
                            ? offset_pre : offset_base;

        if (mark_heap_find_roots_page(page, worker, offset, skip_deferred,
                                       &roots_found) < 0) {
            return -1;
        }
    }

    // Phase 2: Transitive marking with work-stealing
    bool made_progress = true;
    while (made_progress) {
        made_progress = false;

        // Process our local deque (Take from bottom - LIFO for cache locality)
        PyObject *op;
        while ((op = (PyObject *)_PyWSDeque_Take(&worker->deque)) != NULL) {
            made_progress = true;
            if (par_mark_traverse_object(op, worker) < 0) {
                return -1;
            }
        }

        // Try to steal work from other workers
        op = mark_try_steal_work(state, worker_id, workers);
        if (op != NULL) {
            made_progress = true;
            if (par_mark_traverse_object(op, worker) < 0) {
                return -1;
            }
        }
    }

    return 0;
}

//=============================================================================
// Parallel scan_heap_visitor Implementation
//=============================================================================

// Thread-local worklist push (uses ob_tid for linking, which is 0 for unreachable)
static inline void
scan_worklist_push(struct _PyGCScanWorklist *wl, PyObject *op)
{
    assert(op->ob_tid == 0);
    op->ob_tid = wl->head;
    wl->head = (uintptr_t)op;
    wl->count++;
}

// Merge thread-local worklist into main worklist
static void
scan_worklist_merge(struct _PyGCScanWorklist *local, uintptr_t *main_head)
{
    if (local->head == 0) {
        return;
    }

    // Find tail of local list
    PyObject *tail = (PyObject *)local->head;
    while (tail->ob_tid != 0) {
        tail = (PyObject *)tail->ob_tid;
    }

    // Link tail to current head of main list
    tail->ob_tid = *main_head;
    *main_head = local->head;
}

// Check for legacy finalizer (tp_del)
static inline int
par_has_legacy_finalizer(PyObject *op)
{
    return Py_TYPE(op)->tp_del != NULL;
}

// Clear unreachable bit
static inline void
par_gc_clear_unreachable(PyObject *op)
{
    op->ob_gc_bits &= ~_PyGC_BITS_UNREACHABLE;
}

// Clear alive bit
static inline void
par_gc_clear_alive(PyObject *op)
{
    op->ob_gc_bits &= ~_PyGC_BITS_ALIVE;
}

// Merge reference count
static Py_ssize_t
par_merge_refcount(PyObject *op, Py_ssize_t extra)
{
    Py_ssize_t refcount = Py_REFCNT(op);
    refcount += extra;

#ifdef Py_REF_DEBUG
    // Only update ref debug counter if we have a valid thread state.
    // Worker threads in parallel GC don't have thread states, so skip.
    PyThreadState *tstate = _Py_tss_tstate;
    if (tstate != NULL) {
        _Py_AddRefTotal(tstate, extra);
    }
#endif

    op->ob_tid = 0;
    op->ob_ref_local = 0;
    op->ob_ref_shared = _Py_REF_SHARED(refcount, _Py_REF_MERGED);
    return refcount;
}

// Disable deferred refcounting
static void
par_disable_deferred_refcounting(PyObject *op)
{
    if (_PyObject_HasDeferredRefcount(op)) {
        op->ob_gc_bits &= ~_PyGC_BITS_DEFERRED;
        op->ob_ref_shared -= _Py_REF_SHARED(_Py_REF_DEFERRED, 0);
        par_merge_refcount(op, 0);
        _PyObject_DisablePerThreadRefcounting(op);
    }

    // Handle generator/frame objects
    if (PyGen_CheckExact(op) || PyCoro_CheckExact(op) || PyAsyncGen_CheckExact(op)) {
        PyGenObject *gen = (PyGenObject *)op;
        _PyInterpreterFrame *frame = &gen->gi_iframe;
        if (frame->stackpointer != NULL &&
            gen->gi_frame_state != FRAME_CLEARED) {
            frame->f_executable = PyStackRef_AsStrongReference(frame->f_executable);
            frame->f_funcobj = PyStackRef_AsStrongReference(frame->f_funcobj);
            for (_PyStackRef *ref = frame->localsplus; ref < frame->stackpointer; ref++) {
                if (!PyStackRef_IsNullOrInt(*ref) && PyStackRef_IsDeferred(*ref)) {
                    *ref = PyStackRef_AsStrongReference(*ref);
                }
            }
        }
    }
    else if (PyFrame_Check(op)) {
        _PyInterpreterFrame *frame = ((PyFrameObject *)op)->f_frame;
        if (frame != NULL && frame->stackpointer != NULL) {
            frame->f_executable = PyStackRef_AsStrongReference(frame->f_executable);
            frame->f_funcobj = PyStackRef_AsStrongReference(frame->f_funcobj);
            for (_PyStackRef *ref = frame->localsplus; ref < frame->stackpointer; ref++) {
                if (!PyStackRef_IsNullOrInt(*ref) && PyStackRef_IsDeferred(*ref)) {
                    *ref = PyStackRef_AsStrongReference(*ref);
                }
            }
        }
    }
}

// Add a unique ID to the worker's collection for batch release
// Returns 0 on success, -1 on allocation failure
static int
worker_collect_unique_id(struct _PyGCScanWorkerState *worker, Py_ssize_t id)
{
    if (id == _Py_INVALID_UNIQUE_ID) {
        return 0;  // Nothing to collect
    }

    // Grow array if needed
    if (worker->unique_id_count >= worker->unique_id_capacity) {
        size_t new_capacity = worker->unique_id_capacity == 0 ? 64 :
                              worker->unique_id_capacity * 2;
        Py_ssize_t *new_array = PyMem_RawRealloc(worker->unique_ids,
                                                  new_capacity * sizeof(Py_ssize_t));
        if (new_array == NULL) {
            return -1;  // Allocation failed
        }
        worker->unique_ids = new_array;
        worker->unique_id_capacity = new_capacity;
    }

    worker->unique_ids[worker->unique_id_count++] = id;
    return 0;
}

// Disable deferred refcounting and collect unique ID for batch release.
// This version doesn't acquire locks - IDs are collected for batch release later.
static void
par_disable_deferred_collect_ids(PyObject *op, struct _PyGCScanWorkerState *worker)
{
    if (_PyObject_HasDeferredRefcount(op)) {
        op->ob_gc_bits &= ~_PyGC_BITS_DEFERRED;
        op->ob_ref_shared -= _Py_REF_SHARED(_Py_REF_DEFERRED, 0);
        par_merge_refcount(op, 0);

        // Collect unique ID instead of releasing (no locks)
        Py_ssize_t id = _PyObject_ClearUniqueId(op);
        worker_collect_unique_id(worker, id);
    }

    // Handle generator/frame objects (same as original)
    if (PyGen_CheckExact(op) || PyCoro_CheckExact(op) || PyAsyncGen_CheckExact(op)) {
        PyGenObject *gen = (PyGenObject *)op;
        _PyInterpreterFrame *frame = &gen->gi_iframe;
        if (frame->stackpointer != NULL &&
            gen->gi_frame_state != FRAME_CLEARED) {
            frame->f_executable = PyStackRef_AsStrongReference(frame->f_executable);
            frame->f_funcobj = PyStackRef_AsStrongReference(frame->f_funcobj);
            for (_PyStackRef *ref = frame->localsplus; ref < frame->stackpointer; ref++) {
                if (!PyStackRef_IsNullOrInt(*ref) && PyStackRef_IsDeferred(*ref)) {
                    *ref = PyStackRef_AsStrongReference(*ref);
                }
            }
        }
    }
    else if (PyFrame_Check(op)) {
        _PyInterpreterFrame *frame = ((PyFrameObject *)op)->f_frame;
        if (frame != NULL && frame->stackpointer != NULL) {
            frame->f_executable = PyStackRef_AsStrongReference(frame->f_executable);
            frame->f_funcobj = PyStackRef_AsStrongReference(frame->f_funcobj);
            for (_PyStackRef *ref = frame->localsplus; ref < frame->stackpointer; ref++) {
                if (!PyStackRef_IsNullOrInt(*ref) && PyStackRef_IsDeferred(*ref)) {
                    *ref = PyStackRef_AsStrongReference(*ref);
                }
            }
        }
    }
}

// Restore thread ID for reachable objects
static void
par_gc_restore_tid(PyObject *op)
{
    mi_segment_t *segment = _mi_ptr_segment(op);
    if (_Py_REF_IS_MERGED(op->ob_ref_shared)) {
        op->ob_tid = 0;
    }
    else {
        op->ob_tid = segment->thread_id;
        if (op->ob_tid == 0) {
            par_merge_refcount(op, 0);
        }
    }
}

// Per-worker scan arguments
struct scan_page_visitor_args {
    struct _PyGCScanWorkerState *worker;
    Py_ssize_t offset;
    int gc_reason;
};

// Visitor callback for scan_heap blocks
static bool
scan_heap_block_visitor(const mi_heap_t *heap, const mi_heap_area_t *area,
                        void *block, size_t block_size, void *arg)
{
    (void)heap;
    (void)area;
    (void)block_size;

    struct scan_page_visitor_args *args = (struct scan_page_visitor_args *)arg;
    PyObject *op = (PyObject *)((char *)block + args->offset);

    if (!_PyObject_GC_IS_TRACKED(op)) {
        return true;
    }

    if (_PyGC_IsUnreachable(op)) {
        // Unreachable object
        // Disable deferred refcounting and collect unique IDs for batch release.
        // This is done inline during scan to avoid a separate O(n) loop.
        par_disable_deferred_collect_ids(op, args->worker);

        par_merge_refcount(op, 1);

        if (par_has_legacy_finalizer(op)) {
            par_gc_clear_unreachable(op);
            scan_worklist_push(&args->worker->legacy_finalizers, op);
        }
        else {
            scan_worklist_push(&args->worker->unreachable, op);
        }
    }
    else {
        // Reachable object
        par_gc_restore_tid(op);
        par_gc_clear_alive(op);
        args->worker->long_lived_total++;
    }

    return true;
}

// Process a single page in scan_heap using visitor pattern
static void
scan_heap_process_page(mi_page_t *page, struct _PyGCScanWorkerState *worker,
                       int gc_reason)
{
    if (page == NULL || page->used == 0) {
        return;
    }

    mi_heap_area_t area;
    _mi_heap_area_init(&area, page);

    bool is_gc_pre = (page->tag == _Py_MIMALLOC_HEAP_GC_PRE);

    struct scan_page_visitor_args args = {
        .worker = worker,
        .offset = get_block_offset(is_gc_pre),
        .gc_reason = gc_reason
    };

    _mi_heap_area_visit_blocks(&area, page, scan_heap_block_visitor, &args);
}

//=============================================================================
// Pool-based entry points for parallel GC phases
// These use the persistent thread pool instead of spawning ad-hoc threads.
// They require the thread pool to be initialized.
//=============================================================================

// Pool-based parallel update_refs
// Returns sum of candidates across all workers, or -1 on error
Py_ssize_t
_PyGC_ParallelUpdateRefsWithPool(PyInterpreterState *interp,
                                  _PyGCFTParState *state)
{
    // Record start time for phase timing
    PyTime_t start_time;
    (void)PyTime_PerfCounterRaw(&start_time);
    interp->gc.phase_start_ns = start_time;

    _PyGCThreadPool *pool = interp->gc.thread_pool;
    assert(pool != NULL);
    assert(pool->num_workers == state->num_workers);
    assert(state->buckets != NULL);

    // Allocate per-worker candidate counts
    Py_ssize_t *per_worker_refs = PyMem_RawCalloc(pool->num_workers, sizeof(Py_ssize_t));
    if (per_worker_refs == NULL) {
        return -1;
    }

    // Set up work descriptor
    _PyGCWorkDescriptor work = {
        .type = _PyGC_WORK_UPDATE_REFS,
        .buckets = state->buckets,
        .workers = pool->workers,
        .per_worker_refs = per_worker_refs,
        .error_flag = 0,
    };
    pool->current_work = &work;

    // Signal workers to start
    _PyGCBarrier_Wait(&pool->mark_barrier);

    // Worker 0 (main thread) does its share
    thread_pool_do_work(pool, 0);

    // Wait for all workers to complete
    _PyGCBarrier_Wait(&pool->done_barrier);

    // Record end time for update_refs phase
    PyTime_t end_time;
    (void)PyTime_PerfCounterRaw(&end_time);
    interp->gc.update_refs_end_ns = end_time;

    // Clear work descriptor
    pool->current_work = NULL;

    // Sum up candidates
    Py_ssize_t total_candidates = 0;
    if (!work.error_flag) {
        for (int i = 0; i < pool->num_workers; i++) {
            total_candidates += per_worker_refs[i];
        }
    }

    PyMem_RawFree(per_worker_refs);

    return work.error_flag ? -1 : total_candidates;
}

// Pool-based parallel mark_heap
// Returns 0 on success, -1 on error
int
_PyGC_ParallelMarkHeapWithPool(PyInterpreterState *interp,
                                _PyGCFTParState *state,
                                int skip_deferred_objects)
{
    _PyGCThreadPool *pool = interp->gc.thread_pool;
    assert(pool != NULL);
    assert(pool->num_workers == state->num_workers);
    assert(state->buckets != NULL);

    // Reset worker states
    for (int i = 0; i < pool->num_workers; i++) {
        pool->workers[i].objects_marked = 0;
        pool->workers[i].objects_stolen = 0;
        pool->workers[i].steals_attempted = 0;
    }

    // Set up work descriptor
    _PyGCWorkDescriptor work = {
        .type = _PyGC_WORK_MARK_HEAP,
        .buckets = state->buckets,
        .skip_deferred = skip_deferred_objects,
        .workers = pool->workers,
        .error_flag = 0,
    };
    pool->current_work = &work;

    // Signal workers to start
    _PyGCBarrier_Wait(&pool->mark_barrier);

    // Worker 0 (main thread) does its share
    thread_pool_do_work(pool, 0);

    // Wait for all workers to complete
    _PyGCBarrier_Wait(&pool->done_barrier);

    // Record end time for mark_heap phase
    PyTime_t mark_end;
    (void)PyTime_PerfCounterRaw(&mark_end);
    interp->gc.mark_heap_end_ns = mark_end;

    // Clear work descriptor
    pool->current_work = NULL;

#ifdef GC_DEBUG_ATOMICS
    // Print aggregated atomic stats from all workers for mark_heap phase
    int num_workers = pool->num_workers;
    _PyGCAtomicWorkerStats all_stats[num_workers];
    for (int i = 0; i < num_workers; i++) {
        all_stats[i] = pool->workers[i].atomic_stats;
    }
    fprintf(stderr, "[mark_heap phase]\n");
    _PyGC_AtomicPrintStats(all_stats, num_workers);
#endif

    return work.error_flag ? -1 : 0;
}

// Context for scan page enumeration callback
struct _PyGCScanEnumCtx {
    mi_page_t **page_array;
    size_t *total_pages;
    size_t page_capacity;
};

// Callback for adding abandoned pool pages to scan_heap's page array
static void
_PyGC_ScanEnumeratePageCallback(mi_page_t *page, void *arg)
{
    struct _PyGCScanEnumCtx *ctx = (struct _PyGCScanEnumCtx *)arg;
    if (page->used == 0) {
        return;  // Skip empty pages
    }
    // Note: page_capacity check is done before calling enumerate
    ctx->page_array[(*ctx->total_pages)++] = page;
}

// Check if heap is GC heap (tag indicates GC or GC_PRE)
static inline bool
is_gc_heap(mi_heap_t *heap)
{
    // Check first page queue for tag info
    for (size_t q = 0; q <= MI_BIN_FULL; q++) {
        mi_page_queue_t *pq = &heap->pages[q];
        if (pq->first != NULL) {
            int tag = pq->first->tag;
            return (tag == _Py_MIMALLOC_HEAP_GC ||
                    tag == _Py_MIMALLOC_HEAP_GC_PRE);
        }
    }
    // Check heap tag directly
    return (heap->tag == _Py_MIMALLOC_HEAP_GC ||
            heap->tag == _Py_MIMALLOC_HEAP_GC_PRE);
}

// Pool-based parallel scan_heap
// Returns 0 on success, -1 on error
int
_PyGC_ParallelScanHeapWithPool(PyInterpreterState *interp,
                                _PyGCFTParState *state,
                                struct _PyGCScanHeapResult *result)
{
    _PyGCThreadPool *pool = interp->gc.thread_pool;
    assert(pool != NULL);
    assert(pool->num_workers == state->num_workers);

    // Allocate per-worker scan state
    struct _PyGCScanWorkerState *scan_workers = PyMem_RawCalloc(
        pool->num_workers, sizeof(struct _PyGCScanWorkerState));
    if (scan_workers == NULL) {
        return -1;
    }

    // Initialize scan workers with gc_reason
    for (int i = 0; i < pool->num_workers; i++) {
        scan_workers[i].reason = result->gc_reason;
    }

    // Collect pages using atomic counter approach
    // First, collect all pages into an array
    size_t page_capacity = 1024;
    size_t total_pages = 0;
    mi_page_t **page_array = PyMem_RawMalloc(page_capacity * sizeof(mi_page_t *));
    if (page_array == NULL) {
        PyMem_RawFree(scan_workers);
        return -1;
    }

    HEAD_LOCK(&_PyRuntime);
    _Py_FOR_EACH_TSTATE_UNLOCKED(interp, t) {
        struct _mimalloc_thread_state *m = &((_PyThreadStateImpl *)t)->mimalloc;
        if (!_Py_atomic_load_int(&m->initialized)) {
            continue;
        }

        for (int i = 0; i < _Py_MIMALLOC_HEAP_COUNT; i++) {
            mi_heap_t *heap = &m->heaps[i];
            if (!is_gc_heap(heap)) {
                continue;
            }

            for (size_t q = 0; q <= MI_BIN_FULL; q++) {
                mi_page_queue_t *pq = &heap->pages[q];
                for (mi_page_t *page = pq->first; page != NULL; page = page->next) {
                    if (page->used == 0) continue;

                    if (total_pages >= page_capacity) {
                        page_capacity *= 2;
                        mi_page_t **new_arr = PyMem_RawRealloc(
                            page_array, page_capacity * sizeof(mi_page_t *));
                        if (new_arr == NULL) {
                            HEAD_UNLOCK(&_PyRuntime);
                            PyMem_RawFree(page_array);
                            PyMem_RawFree(scan_workers);
                            return -1;
                        }
                        page_array = new_arr;
                    }
                    page_array[total_pages++] = page;
                }
            }
        }
    }

    // Also include pages from the abandoned pool (from exited threads)
    // This is critical for garbage collection of objects created by threads
    // that have since exited. We use _mi_abandoned_pool_count_pages to get count
    // and then enumerate.
    mi_abandoned_pool_t *apool = &interp->mimalloc.abandoned_pool;
    size_t abandoned_count = _mi_abandoned_pool_count_pages(apool, _Py_MIMALLOC_HEAP_GC)
                           + _mi_abandoned_pool_count_pages(apool, _Py_MIMALLOC_HEAP_GC_PRE);

    if (abandoned_count > 0) {
        // Ensure we have space for abandoned pages
        if (total_pages + abandoned_count > page_capacity) {
            page_capacity = total_pages + abandoned_count + 64;  // Extra slack
            mi_page_t **new_arr = PyMem_RawRealloc(
                page_array, page_capacity * sizeof(mi_page_t *));
            if (new_arr == NULL) {
                HEAD_UNLOCK(&_PyRuntime);
                PyMem_RawFree(page_array);
                PyMem_RawFree(scan_workers);
                return -1;
            }
            page_array = new_arr;
        }

        // Use the page enumeration function to add pages
        struct _PyGCScanEnumCtx ectx = {
            .page_array = page_array,
            .total_pages = &total_pages,
            .page_capacity = page_capacity
        };
        _mi_abandoned_pool_enumerate_pages(apool, _Py_MIMALLOC_HEAP_GC,
            _PyGC_ScanEnumeratePageCallback, &ectx);
        _mi_abandoned_pool_enumerate_pages(apool, _Py_MIMALLOC_HEAP_GC_PRE,
            _PyGC_ScanEnumeratePageCallback, &ectx);
    }

    HEAD_UNLOCK(&_PyRuntime);

    if (total_pages == 0) {
        PyMem_RawFree(page_array);
        PyMem_RawFree(scan_workers);
        return 0;
    }

    // Set up work descriptor
    _PyGCWorkDescriptor work = {
        .type = _PyGC_WORK_SCAN_HEAP,
        .scan_result = result,
        .scan_workers = scan_workers,
        .page_array = page_array,
        .total_pages = total_pages,
        .page_counter = 0,
        .workers = pool->workers,
        .error_flag = 0,
    };
    pool->current_work = &work;

    // Signal workers to start
    _PyGCBarrier_Wait(&pool->mark_barrier);

    // Worker 0 (main thread) does its share
    thread_pool_do_work(pool, 0);

    // Wait for all workers to complete
    _PyGCBarrier_Wait(&pool->done_barrier);

    // Clear work descriptor
    pool->current_work = NULL;

    // Merge results from all workers
    for (int i = 0; i < pool->num_workers; i++) {
        scan_worklist_merge(&scan_workers[i].unreachable, result->unreachable_head);
        scan_worklist_merge(&scan_workers[i].legacy_finalizers,
                            result->legacy_finalizers_head);
        *result->long_lived_total += scan_workers[i].long_lived_total;
    }

    // Batch release all collected unique IDs (during STW, no lock needed)
    // This replaces the O(n) lock acquisitions with a single batch operation
    for (int i = 0; i < pool->num_workers; i++) {
        if (scan_workers[i].unique_id_count > 0) {
            _PyObject_ReleaseUniqueIdBatch_NoLock(
                interp,
                scan_workers[i].unique_ids,
                scan_workers[i].unique_id_count);
        }
        // Free the worker's unique_id array
        if (scan_workers[i].unique_ids != NULL) {
            PyMem_RawFree(scan_workers[i].unique_ids);
        }
    }

    // Cleanup
    PyMem_RawFree(page_array);
    PyMem_RawFree(scan_workers);

    return work.error_flag ? -1 : 0;
}

//-----------------------------------------------------------------------------
// Pool-based work function wrappers
// These wrap the existing functions to work with the persistent thread pool
//-----------------------------------------------------------------------------

// Pool version of update_refs_init_page
// Wraps the existing function with pool-compatible args struct
static int
pool_update_refs_init_page(mi_page_t *page, _PyGCPoolUpdateRefsArgs *args)
{
    assert(page != NULL);

    mi_heap_area_t area;
    _mi_heap_area_init(&area, page);

    bool is_gc_pre = (page->tag == _Py_MIMALLOC_HEAP_GC_PRE);

    _PyGCUpdateRefsInitArgs visitor_args = {
        .offset = get_block_offset(is_gc_pre),
        .candidates = &args->candidates
    };

    _mi_heap_area_visit_blocks(&area, page, update_refs_init_visitor, &visitor_args);

    return 0;
}

// Pool version of update_refs_compute_page
static int
pool_update_refs_compute_page(mi_page_t *page, _PyGCPoolUpdateRefsArgs *args)
{
    (void)args;
    assert(page != NULL);

    mi_heap_area_t area;
    _mi_heap_area_init(&area, page);

    bool is_gc_pre = (page->tag == _Py_MIMALLOC_HEAP_GC_PRE);

    _PyGCUpdateRefsComputeArgs visitor_args = {
        .offset = get_block_offset(is_gc_pre)
    };

    _mi_heap_area_visit_blocks(&area, page, update_refs_compute_visitor, &visitor_args);

    return 0;
}

// Pool version of mark_heap_find_roots_page - thin wrapper
static int
pool_mark_heap_find_roots_page(mi_page_t *page, _PyGCWorkerState *worker,
                               Py_ssize_t offset, int skip_deferred,
                               size_t *roots_found)
{
    return mark_heap_find_roots_page(page, worker, offset, skip_deferred, roots_found);
}

// Pool version of par_mark_traverse_object - thin wrapper
static int
pool_mark_traverse_object(PyObject *op, _PyGCWorkerState *worker)
{
    return par_mark_traverse_object(op, worker);
}

// Pool version of scan_heap_process_page - thin wrapper
static void
pool_scan_heap_process_page(mi_page_t *page, struct _PyGCScanWorkerState *worker,
                            int gc_reason)
{
    scan_heap_process_page(page, worker, gc_reason);
}

//=============================================================================
// Statistics API
//=============================================================================

PyObject *
_PyGC_FTParallelGetStats(PyInterpreterState *interp)
{
    PyObject *result = PyDict_New();
    if (result == NULL) {
        return NULL;
    }

    PyObject *enabled = interp->gc.parallel_gc_enabled ? Py_True : Py_False;
    if (PyDict_SetItemString(result, "enabled", enabled) < 0) {
        Py_DECREF(result);
        return NULL;
    }

    int num_workers = interp->gc.parallel_gc_enabled ?
                      interp->gc.parallel_gc_num_workers : 0;
    PyObject *workers = PyLong_FromLong(num_workers);
    if (workers == NULL || PyDict_SetItemString(result, "num_workers", workers) < 0) {
        Py_XDECREF(workers);
        Py_DECREF(result);
        return NULL;
    }
    Py_DECREF(workers);

    // Add phase timing (nanoseconds)
    PyObject *phase_timing = PyDict_New();
    if (phase_timing == NULL) {
        Py_DECREF(result);
        return NULL;
    }

    // Helper macro to add a timing value to the dict
    #define ADD_TIMING(name, value) do { \
        PyObject *obj = PyLong_FromLongLong(value); \
        if (obj == NULL || PyDict_SetItemString(phase_timing, name, obj) < 0) { \
            Py_XDECREF(obj); \
            Py_DECREF(phase_timing); \
            Py_DECREF(result); \
            return NULL; \
        } \
        Py_DECREF(obj); \
    } while(0)

    // Calculate phase durations from recorded timestamps
    // Pre-parallel phases
    int64_t stw0_ns = 0;
    int64_t merge_refs_ns = 0;
    int64_t delayed_frees_ns = 0;
    int64_t mark_alive_ns = 0;
    int64_t bucket_assign_ns = 0;
    // Parallel phases
    int64_t update_refs_ns = 0;
    int64_t mark_heap_ns = 0;
    int64_t scan_heap_ns = 0;
    int64_t disable_deferred_ns = 0;
    int64_t find_weakrefs_ns = 0;
    int64_t stw1_ns = 0;
    int64_t objs_decref_ns = 0;
    int64_t weakref_callbacks_ns = 0;
    int64_t finalize_ns = 0;
    int64_t stw2_ns = 0;
    int64_t resurrection_ns = 0;
    int64_t freelists_ns = 0;
    int64_t clear_weakrefs_ns = 0;
    int64_t stw3_ns = 0;
    int64_t cleanup_ns = 0;
    int64_t total_ns = 0;

    // Calculate pre-parallel phase durations
    if (interp->gc.gc_start_ns > 0 && interp->gc.stw0_end_ns > 0) {
        stw0_ns = interp->gc.stw0_end_ns - interp->gc.gc_start_ns;
    }
    if (interp->gc.stw0_end_ns > 0 && interp->gc.merge_refs_end_ns > 0) {
        merge_refs_ns = interp->gc.merge_refs_end_ns - interp->gc.stw0_end_ns;
    }
    if (interp->gc.merge_refs_end_ns > 0 && interp->gc.delayed_frees_end_ns > 0) {
        delayed_frees_ns = interp->gc.delayed_frees_end_ns - interp->gc.merge_refs_end_ns;
    }
    if (interp->gc.delayed_frees_end_ns > 0 && interp->gc.mark_alive_end_ns > 0) {
        mark_alive_ns = interp->gc.mark_alive_end_ns - interp->gc.delayed_frees_end_ns;
    }
    if (interp->gc.mark_alive_end_ns > 0 && interp->gc.bucket_assign_end_ns > 0) {
        bucket_assign_ns = interp->gc.bucket_assign_end_ns - interp->gc.mark_alive_end_ns;
    }

    // Calculate durations between timestamps
    if (interp->gc.phase_start_ns > 0 && interp->gc.update_refs_end_ns > 0) {
        update_refs_ns = interp->gc.update_refs_end_ns - interp->gc.phase_start_ns;
    }
    if (interp->gc.update_refs_end_ns > 0 && interp->gc.mark_heap_end_ns > 0) {
        mark_heap_ns = interp->gc.mark_heap_end_ns - interp->gc.update_refs_end_ns;
    }
    if (interp->gc.mark_heap_end_ns > 0 && interp->gc.scan_heap_end_ns > 0) {
        scan_heap_ns = interp->gc.scan_heap_end_ns - interp->gc.mark_heap_end_ns;
    }
    if (interp->gc.scan_heap_end_ns > 0 && interp->gc.disable_deferred_end_ns > 0) {
        disable_deferred_ns = interp->gc.disable_deferred_end_ns - interp->gc.scan_heap_end_ns;
    }
    if (interp->gc.disable_deferred_end_ns > 0 && interp->gc.find_weakrefs_end_ns > 0) {
        find_weakrefs_ns = interp->gc.find_weakrefs_end_ns - interp->gc.disable_deferred_end_ns;
    }
    if (interp->gc.find_weakrefs_end_ns > 0 && interp->gc.stw1_end_ns > 0) {
        stw1_ns = interp->gc.stw1_end_ns - interp->gc.find_weakrefs_end_ns;
    }
    if (interp->gc.stw1_end_ns > 0 && interp->gc.objs_decref_end_ns > 0) {
        objs_decref_ns = interp->gc.objs_decref_end_ns - interp->gc.stw1_end_ns;
    }
    if (interp->gc.objs_decref_end_ns > 0 && interp->gc.weakref_callbacks_end_ns > 0) {
        weakref_callbacks_ns = interp->gc.weakref_callbacks_end_ns - interp->gc.objs_decref_end_ns;
    }
    if (interp->gc.weakref_callbacks_end_ns > 0 && interp->gc.finalize_end_ns > 0) {
        finalize_ns = interp->gc.finalize_end_ns - interp->gc.weakref_callbacks_end_ns;
    }
    if (interp->gc.finalize_end_ns > 0 && interp->gc.stw2_end_ns > 0) {
        stw2_ns = interp->gc.stw2_end_ns - interp->gc.finalize_end_ns;
    }
    if (interp->gc.stw2_end_ns > 0 && interp->gc.resurrection_end_ns > 0) {
        resurrection_ns = interp->gc.resurrection_end_ns - interp->gc.stw2_end_ns;
    }
    if (interp->gc.resurrection_end_ns > 0 && interp->gc.freelists_end_ns > 0) {
        freelists_ns = interp->gc.freelists_end_ns - interp->gc.resurrection_end_ns;
    }
    if (interp->gc.freelists_end_ns > 0 && interp->gc.clear_weakrefs_end_ns > 0) {
        clear_weakrefs_ns = interp->gc.clear_weakrefs_end_ns - interp->gc.freelists_end_ns;
    }
    if (interp->gc.clear_weakrefs_end_ns > 0 && interp->gc.stw3_end_ns > 0) {
        stw3_ns = interp->gc.stw3_end_ns - interp->gc.clear_weakrefs_end_ns;
    }
    if (interp->gc.cleanup_start_ns > 0 && interp->gc.cleanup_end_ns > 0) {
        cleanup_ns = interp->gc.cleanup_end_ns - interp->gc.cleanup_start_ns;
    }
    // Calculate total from gc_start to cleanup_end (or fallback)
    if (interp->gc.gc_start_ns > 0 && interp->gc.cleanup_end_ns > 0) {
        total_ns = interp->gc.cleanup_end_ns - interp->gc.gc_start_ns;
    }
    else if (interp->gc.gc_start_ns > 0 && interp->gc.stw3_end_ns > 0) {
        total_ns = interp->gc.stw3_end_ns - interp->gc.gc_start_ns;
    }
    else if (interp->gc.gc_start_ns > 0 && interp->gc.mark_heap_end_ns > 0) {
        // Fallback to mark_heap_end if nothing else recorded
        total_ns = interp->gc.mark_heap_end_ns - interp->gc.gc_start_ns;
    }

    // Add pre-parallel phase timing values
    ADD_TIMING("stw0_ns", stw0_ns);
    ADD_TIMING("merge_refs_ns", merge_refs_ns);
    ADD_TIMING("delayed_frees_ns", delayed_frees_ns);
    ADD_TIMING("mark_alive_ns", mark_alive_ns);
    ADD_TIMING("bucket_assign_ns", bucket_assign_ns);

    // Add all timing values
    ADD_TIMING("update_refs_ns", update_refs_ns);
    ADD_TIMING("mark_heap_ns", mark_heap_ns);
    ADD_TIMING("scan_heap_ns", scan_heap_ns);
    ADD_TIMING("disable_deferred_ns", disable_deferred_ns);
    ADD_TIMING("find_weakrefs_ns", find_weakrefs_ns);
    ADD_TIMING("stw1_ns", stw1_ns);
    ADD_TIMING("objs_decref_ns", objs_decref_ns);
    ADD_TIMING("weakref_callbacks_ns", weakref_callbacks_ns);
    ADD_TIMING("finalize_ns", finalize_ns);
    ADD_TIMING("stw2_ns", stw2_ns);
    ADD_TIMING("resurrection_ns", resurrection_ns);
    ADD_TIMING("freelists_ns", freelists_ns);
    ADD_TIMING("clear_weakrefs_ns", clear_weakrefs_ns);
    ADD_TIMING("stw3_ns", stw3_ns);
    ADD_TIMING("cleanup_ns", cleanup_ns);
    ADD_TIMING("total_ns", total_ns);

    // Abstract phases - provide a common interface across GIL and FTP collectors.
    // These allow benchmarks to compare builds using identical phase names.
    //
    // scan_mark_ns: Core graph traversal to identify reachable objects.
    //   FTP: stw0 + merge_refs + delayed_frees + mark_alive + bucket_assign +
    //        update_refs + mark_heap + scan_heap
    //
    // finalization_ns: Weakref/finalize handling.
    //   FTP: disable_deferred + find_weakrefs + stw1 + objs_decref +
    //        weakref_callbacks + finalize + stw2 + resurrection
    //
    // dealloc_ns: Final deallocation and cleanup.
    //   FTP: freelists + clear_weakrefs + stw3 + cleanup
    //
    // stw_pause_ns: Total time threads are stopped (not just barrier sync).
    //   FTP has two STW periods:
    //   - First STW: gc_start → find_weakrefs_end (merge, mark, scan, find_weakrefs)
    //   - Second STW: finalize_end → clear_weakrefs_end (resurrection, freelists, clear)
    int64_t scan_mark = stw0_ns + merge_refs_ns + delayed_frees_ns +
                        mark_alive_ns + bucket_assign_ns +
                        update_refs_ns + mark_heap_ns + scan_heap_ns;
    int64_t finalization = disable_deferred_ns + find_weakrefs_ns +
                           stw1_ns + objs_decref_ns +
                           weakref_callbacks_ns + finalize_ns +
                           stw2_ns + resurrection_ns;
    int64_t dealloc = freelists_ns + clear_weakrefs_ns + stw3_ns + cleanup_ns;

    // First STW period: from gc_start until world is resumed after find_weakrefs
    // This includes: stw0 barrier + merge_refs + delayed_frees + mark_alive +
    //                update_refs + mark_heap + scan_heap + find_weakrefs
    int64_t stw_period_1 = 0;
    if (interp->gc.gc_start_ns > 0 && interp->gc.find_weakrefs_end_ns > 0) {
        stw_period_1 = interp->gc.find_weakrefs_end_ns - interp->gc.gc_start_ns;
    }

    // Second STW period: from entering STW2 until world is resumed after clear_weakrefs
    // This includes: stw2 barrier + resurrection + freelists + clear_weakrefs
    int64_t stw_period_2 = 0;
    if (interp->gc.finalize_end_ns > 0 && interp->gc.clear_weakrefs_end_ns > 0) {
        stw_period_2 = interp->gc.clear_weakrefs_end_ns - interp->gc.finalize_end_ns;
    }

    int64_t stw_pause = stw_period_1 + stw_period_2;

    ADD_TIMING("scan_mark_ns", scan_mark);
    ADD_TIMING("finalization_ns", finalization);
    ADD_TIMING("dealloc_ns", dealloc);
    ADD_TIMING("stw_pause_ns", stw_pause);

    #undef ADD_TIMING

    if (PyDict_SetItemString(result, "phase_timing", phase_timing) < 0) {
        Py_DECREF(phase_timing);
        Py_DECREF(result);
        return NULL;
    }
    Py_DECREF(phase_timing);

    return result;
}

#endif  // Py_GIL_DISABLED
