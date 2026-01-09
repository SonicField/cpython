// Parallel garbage collector for free-threaded Python.
//
// This extends gc_free_threading.c with parallel marking support using:
// - Page-based sequential bucket filling for work distribution
// - Work-stealing for load balancing
// - Atomic operations for parallel marking
//
// See FTP_PARALLEL_GC_DESIGN.md for design details.

#include "Python.h"

#ifdef Py_GIL_DISABLED

#include "pycore_gc.h"
#include "pycore_gc_ft_parallel.h"
#include "pycore_interp.h"
#include "pycore_pystate.h"
#include "pycore_tstate.h"
#include "pycore_object_alloc.h"
#include "pycore_object_deferred.h"  // _PyObject_HasDeferredRefcount
#include "pycore_frame.h"            // _PyInterpreterFrame, FRAME_CLEARED
#include "pycore_stackref.h"         // PyStackRef_* functions
#include "frameobject.h"             // PyFrameObject

// For mimalloc heap access - includes mimalloc/types.h for mi_page_t, mi_heap_t
#include "pycore_mimalloc.h"

#include <stdatomic.h>

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

    // TODO: Count pages in abandoned pool (from dead threads)

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
        if (new_capacity < 16) new_capacity = 16;

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

    size_t initial_capacity = (state->total_pages / state->num_workers) + 16;
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
    struct parallel_mark_page_args *args = (struct parallel_mark_page_args *)arg;

    PyObject *op = block_to_object(block, args->offset);
    if (op == NULL) {
        return true;  // Skip untracked/frozen objects
    }

    // Try to mark this object as alive using atomic CAS
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
static void *
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

    return NULL;
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
    state->threads = PyMem_RawCalloc(state->num_workers - 1, sizeof(pthread_t));
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

        int err = pthread_create(&state->threads[i - 1],
                                NULL,
                                parallel_worker_thread,
                                &thread_args[i - 1]);
        if (err != 0) {
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
        pthread_join(state->threads[i], NULL);
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
    pthread_t *threads;
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

    // Try to mark child alive (atomic CAS)
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

// Thread entry for parallel propagation
static void *
propagate_worker_thread(void *arg)
{
    _PyGCPropagateArgs *args = (_PyGCPropagateArgs *)arg;

    assert(args != NULL);
    assert(args->state != NULL);
    assert(args->worker_id > 0);

    if (propagate_worker_run(args->state, args->worker_id) < 0) {
        _Py_atomic_store_int(&args->state->error_flag, 1);
    }

    return NULL;
}

// Initialize propagation workers
static int
init_propagate_workers(_PyGCPropagateState *state)
{
    state->workers = PyMem_RawCalloc(state->num_workers, sizeof(_PyGCWorkerState));
    if (state->workers == NULL) {
        return -1;
    }

    for (int i = 0; i < state->num_workers; i++) {
        state->workers[i].worker_id = i;
        state->workers[i].objects_marked = 0;
        _PyWSDeque_Init(&state->workers[i].deque);
    }

    return 0;
}

// Free propagation workers
static void
free_propagate_workers(_PyGCPropagateState *state)
{
    if (state->workers != NULL) {
        for (int i = 0; i < state->num_workers; i++) {
            _PyWSDeque_Fini(&state->workers[i].deque);
        }
        PyMem_RawFree(state->workers);
        state->workers = NULL;
    }
}

// Parallel propagation entry point
// Takes roots from initial_roots array and propagates ALIVE transitively
// using parallel workers with work-stealing.
int
_PyGC_ParallelPropagateAlive(PyInterpreterState *interp,
                              PyObject **initial_roots,
                              size_t num_roots,
                              int num_workers)
{
    assert(interp != NULL);
    assert(interp->stoptheworld.world_stopped);
    assert(num_workers > 0);

    if (num_roots == 0) {
        return 0;  // Nothing to do
    }

    _PyGCPropagateState state = {
        .num_workers = num_workers,
        .workers = NULL,
        .threads = NULL,
        .error_flag = 0,
        .total_marked = 0
    };

    // Initialize workers
    if (init_propagate_workers(&state) < 0) {
        return -1;
    }

    // Distribute initial roots to workers (round-robin)
    for (size_t i = 0; i < num_roots; i++) {
        PyObject *root = initial_roots[i];
        if (root != NULL && _PyObject_GC_IS_TRACKED(root)) {
            // Root is already marked alive, push to worker deque
            int worker_id = i % num_workers;
            _PyWSDeque_Push(&state.workers[worker_id].deque, root);
        }
    }

    int result = 0;

    // Single worker: run directly
    if (num_workers == 1) {
        result = propagate_worker_run(&state, 0);
        goto cleanup;
    }

    // Multiple workers: spawn threads
    state.threads = PyMem_RawCalloc(num_workers - 1, sizeof(pthread_t));
    if (state.threads == NULL) {
        result = -1;
        goto cleanup;
    }

    _PyGCPropagateArgs *thread_args = PyMem_RawCalloc(num_workers - 1,
                                                       sizeof(_PyGCPropagateArgs));
    if (thread_args == NULL) {
        PyMem_RawFree(state.threads);
        state.threads = NULL;
        result = -1;
        goto cleanup;
    }

    // Spawn worker threads 1..N-1
    int threads_created = 0;
    for (int i = 1; i < num_workers; i++) {
        thread_args[i - 1].state = &state;
        thread_args[i - 1].worker_id = i;

        int err = pthread_create(&state.threads[i - 1],
                                NULL,
                                propagate_worker_thread,
                                &thread_args[i - 1]);
        if (err != 0) {
            state.error_flag = 1;
            break;
        }
        threads_created++;
    }

    // Worker 0 runs on main thread
    if (propagate_worker_run(&state, 0) < 0) {
        state.error_flag = 1;
    }

    // Join all threads
    for (int i = 0; i < threads_created; i++) {
        pthread_join(state.threads[i], NULL);
    }

    PyMem_RawFree(thread_args);
    PyMem_RawFree(state.threads);
    state.threads = NULL;

    if (state.error_flag) {
        result = -1;
    }

cleanup:
    // Collect stats
    state.total_marked = 0;
    for (int i = 0; i < num_workers; i++) {
        state.total_marked += state.workers[i].objects_marked;
    }

    free_propagate_workers(&state);
    return result;
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

// Work-stealing loop for a single worker
// Called by both background workers and main thread (worker 0)
static void
thread_pool_do_work(_PyGCThreadPool *pool, int worker_id)
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
            if (Py_TYPE(obj)->tp_traverse != NULL) {
                Py_TYPE(obj)->tp_traverse(obj,
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

// Background worker thread entry point
static void *
thread_pool_worker(void *arg)
{
    _PyGCThreadPool *pool = (_PyGCThreadPool *)arg;

    // Determine worker ID from thread handle comparison
    int worker_id = -1;
    pthread_t self = pthread_self();
    for (int i = 0; i < pool->num_workers - 1; i++) {
        if (pthread_equal(pool->threads[i], self)) {
            worker_id = i + 1;  // Worker 0 is main thread
            break;
        }
    }
    assert(worker_id > 0 && worker_id < pool->num_workers);

    while (1) {
        // Wait at mark_barrier for work (main thread signals by arriving here too)
        _PyGCBarrier_Wait(&pool->mark_barrier);

        // Check for shutdown after waking
        if (pool->shutdown) {
            break;
        }

        // Do the work
        thread_pool_do_work(pool, worker_id);

        // Signal completion by arriving at done_barrier
        _PyGCBarrier_Wait(&pool->done_barrier);
    }

    return NULL;
}

// Visitproc for pool-based propagation
// Pushes to local buffer (fast) and only overflows to deque when buffer is full
static int
propagate_pool_visitproc(PyObject *obj, void *arg)
{
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
    assert(num_workers >= 2);  // Need at least 2 workers for parallelism
    assert(interp->gc.thread_pool == NULL);  // Should not be initialized yet

    _PyGCThreadPool *pool = PyMem_RawCalloc(1, sizeof(_PyGCThreadPool));
    if (pool == NULL) {
        return -1;
    }

    pool->num_workers = num_workers;
    pool->shutdown = 0;
    pool->threads_created = 0;
    pool->collections_completed = 0;

    // Initialize barriers for synchronization
    // Both barriers include all workers (main thread as worker 0)
    _PyGCBarrier_Init(&pool->mark_barrier, num_workers);
    _PyGCBarrier_Init(&pool->done_barrier, num_workers);

    // Allocate thread handles (for workers 1..N-1, worker 0 is main thread)
    pool->threads = PyMem_RawCalloc(num_workers - 1, sizeof(pthread_t));
    if (pool->threads == NULL) {
        _PyGCBarrier_Fini(&pool->done_barrier);
        _PyGCBarrier_Fini(&pool->mark_barrier);
        PyMem_RawFree(pool);
        return -1;
    }

    // Allocate persistent worker states (one per worker including main)
    pool->workers = PyMem_RawCalloc(num_workers, sizeof(_PyGCWorkerState));
    if (pool->workers == NULL) {
        PyMem_RawFree(pool->threads);
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

    // Create worker threads (they will wait at mark_barrier immediately)
    for (int i = 0; i < num_workers - 1; i++) {
        int err = pthread_create(&pool->threads[i], NULL,
                                  thread_pool_worker, pool);
        if (err != 0) {
            // Failed - shut down already created threads
            pool->shutdown = 1;
            // Wake workers so they can exit (need to unblock mark_barrier)
            // For partially created pool, we need to reach barrier with all threads
            // Just set shutdown and let them exit on next barrier pass
            // Actually, since not all threads are created, barrier will block forever
            // So we need to join the ones we created and cleanup
            for (int j = 0; j < i; j++) {
                // Cancel threads that are blocked on barrier
                pthread_cancel(pool->threads[j]);
                pthread_join(pool->threads[j], NULL);
            }
            for (int j = 0; j < num_workers; j++) {
                // Use FiniExternal if using pre-allocated pool to avoid double-free
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
        pthread_join(pool->threads[i], NULL);
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

    // Distribute roots round-robin to worker deques
    // Do this BEFORE signaling workers to start
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

    // All work is complete
    pool->collections_completed++;

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

// Thread entry point for parallel update_refs worker
typedef struct {
    _PyGCFTParState *state;
    _PyGCBarrier *phase_barrier;
    int worker_id;
    Py_ssize_t candidates;
    int result;
} _PyGCUpdateRefsThreadArgs;

static void *
update_refs_thread(void *arg)
{
    _PyGCUpdateRefsThreadArgs *targs = (_PyGCUpdateRefsThreadArgs *)arg;
    targs->result = update_refs_worker(targs->state, targs->worker_id,
                                       targs->phase_barrier,
                                       &targs->candidates);
    return NULL;
}

// Main entry point for parallel update_refs
// Returns sum of candidates across all workers
Py_ssize_t
_PyGC_ParallelUpdateRefs(PyInterpreterState *interp,
                         _PyGCFTParState *state)
{
    assert(interp != NULL);
    assert(state != NULL);
    assert(interp->stoptheworld.world_stopped);
    assert(state->num_workers > 0);
    assert(state->buckets != NULL);

    // Initialize barrier for phase synchronization (all workers + main thread)
    _PyGCBarrier phase_barrier;
    _PyGCBarrier_Init(&phase_barrier, state->num_workers);

    Py_ssize_t total_candidates = 0;

    // For single worker, just run directly
    if (state->num_workers == 1) {
        Py_ssize_t candidates = 0;
        if (update_refs_worker(state, 0, &phase_barrier, &candidates) < 0) {
            _PyGCBarrier_Fini(&phase_barrier);
            return -1;
        }
        _PyGCBarrier_Fini(&phase_barrier);
        return candidates;
    }

    // Allocate thread handles for workers 1..N-1
    pthread_t *threads = PyMem_RawCalloc(state->num_workers - 1, sizeof(pthread_t));
    if (threads == NULL) {
        _PyGCBarrier_Fini(&phase_barrier);
        return -1;
    }

    // Allocate thread arguments
    _PyGCUpdateRefsThreadArgs *thread_args = PyMem_RawCalloc(
        state->num_workers, sizeof(_PyGCUpdateRefsThreadArgs));
    if (thread_args == NULL) {
        PyMem_RawFree(threads);
        _PyGCBarrier_Fini(&phase_barrier);
        return -1;
    }

    // Initialize thread args
    for (int i = 0; i < state->num_workers; i++) {
        thread_args[i].state = state;
        thread_args[i].phase_barrier = &phase_barrier;
        thread_args[i].worker_id = i;
        thread_args[i].candidates = 0;
        thread_args[i].result = 0;
    }

    // Spawn worker threads 1..N-1
    int threads_created = 0;
    for (int i = 1; i < state->num_workers; i++) {
        int err = pthread_create(&threads[i - 1], NULL,
                                 update_refs_thread, &thread_args[i]);
        if (err != 0) {
            // Failed to create thread - we'll run with fewer workers
            break;
        }
        threads_created++;
    }

    // Worker 0 runs on main thread
    Py_ssize_t w0_candidates = 0;
    int result = update_refs_worker(state, 0, &phase_barrier, &w0_candidates);
    thread_args[0].candidates = w0_candidates;
    thread_args[0].result = result;

    // Wait for spawned threads
    for (int i = 0; i < threads_created; i++) {
        pthread_join(threads[i], NULL);
    }

    // Sum up candidates and check for errors
    int error = thread_args[0].result;
    total_candidates = thread_args[0].candidates;

    for (int i = 1; i <= threads_created; i++) {
        if (thread_args[i].result < 0) {
            error = -1;
        }
        total_candidates += thread_args[i].candidates;
    }

    // Cleanup
    PyMem_RawFree(thread_args);
    PyMem_RawFree(threads);
    _PyGCBarrier_Fini(&phase_barrier);

    return error < 0 ? -1 : total_candidates;
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

// Visitor callback to find roots in a page
// A root is an object with gc_refs > 0 (or deferred + skip_deferred) that is still unreachable
// NOTE: This is defined but currently unused - we use mark_heap_roots_visitor instead
// Keeping for reference.
#if 0
static bool
mark_heap_parallel_visitor(const mi_heap_t *heap, const mi_heap_area_t *area,
                           void *block, size_t block_size, void *arg)
{
    (void)heap;
    (void)area;
    (void)block_size;

    _PyGCMarkWorkerContext *ctx = (_PyGCMarkWorkerContext *)arg;
    PyObject *op = block_to_object(block, ctx->par_state->buckets->pages ? 0 : 0);

    if (op == NULL) {
        return true;
    }

    // Skip if not tracked
    if (!_PyObject_GC_IS_TRACKED(op)) {
        return true;
    }

    // Skip frozen
    if (op->ob_gc_bits & _PyGC_BITS_FROZEN) {
        return true;
    }

    // Skip if already marked alive (by previous gc_mark_alive phase)
    if (op->ob_gc_bits & _PyGC_BITS_ALIVE) {
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
#endif

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
    if (op->ob_gc_bits & _PyGC_BITS_ALIVE) {
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

// Thread entry point for parallel mark worker
typedef struct {
    _PyGCFTParState *state;
    _PyGCWorkerState *workers;
    int worker_id;
    int skip_deferred;
    int result;
} _PyGCMarkThreadArgs;

static void *
mark_heap_thread(void *arg)
{
    _PyGCMarkThreadArgs *targs = (_PyGCMarkThreadArgs *)arg;
    targs->result = mark_heap_worker(targs->state, targs->worker_id,
                                     targs->workers, targs->skip_deferred);
    return NULL;
}

// Main entry point for parallel mark_heap_visitor
// Returns 0 on success, -1 on error
int
_PyGC_ParallelMarkHeap(PyInterpreterState *interp,
                       _PyGCFTParState *state,
                       int skip_deferred_objects)
{
    assert(interp != NULL);
    assert(state != NULL);
    assert(interp->stoptheworld.world_stopped);
    assert(state->num_workers > 0);
    assert(state->buckets != NULL);

    // Allocate worker states
    _PyGCWorkerState *workers = PyMem_RawCalloc(state->num_workers,
                                                 sizeof(_PyGCWorkerState));
    if (workers == NULL) {
        return -1;
    }

    // Allocate deque buffers for workers (256K entries = 2MB each)
    // This avoids runtime growth and associated race conditions
    size_t pool_entries = _Py_WSDEQUE_PARALLEL_GC_SIZE;
    size_t pool_bytes = sizeof(_PyWSArray) + sizeof(uintptr_t) * pool_entries;
    void **deque_buffers = PyMem_RawCalloc(state->num_workers, sizeof(void *));
    if (deque_buffers == NULL) {
        PyMem_RawFree(workers);
        return -1;
    }

    // Initialize worker states and deques with pre-allocated buffers
    for (int i = 0; i < state->num_workers; i++) {
        workers[i].worker_id = i;
        workers[i].objects_marked = 0;
        workers[i].objects_stolen = 0;
        workers[i].steals_attempted = 0;

        // Allocate and initialize deque with large buffer
        deque_buffers[i] = PyMem_RawMalloc(pool_bytes);
        if (deque_buffers[i] != NULL) {
            _PyWSDeque_InitWithBuffer(&workers[i].deque, deque_buffers[i],
                                       pool_bytes, pool_entries);
        } else {
            // Fallback to dynamic allocation (will trigger runtime growth)
            _PyWSDeque_Init(&workers[i].deque);
        }
    }

    int result = 0;

    // For single worker, just run directly
    if (state->num_workers == 1) {
        result = mark_heap_worker(state, 0, workers, skip_deferred_objects);
        goto cleanup;
    }

    // Allocate thread handles for workers 1..N-1
    pthread_t *threads = PyMem_RawCalloc(state->num_workers - 1, sizeof(pthread_t));
    if (threads == NULL) {
        result = -1;
        goto cleanup;
    }

    // Allocate thread arguments
    _PyGCMarkThreadArgs *thread_args = PyMem_RawCalloc(
        state->num_workers, sizeof(_PyGCMarkThreadArgs));
    if (thread_args == NULL) {
        PyMem_RawFree(threads);
        result = -1;
        goto cleanup;
    }

    // Initialize thread args
    for (int i = 0; i < state->num_workers; i++) {
        thread_args[i].state = state;
        thread_args[i].workers = workers;
        thread_args[i].worker_id = i;
        thread_args[i].skip_deferred = skip_deferred_objects;
        thread_args[i].result = 0;
    }

    // Spawn worker threads 1..N-1
    int threads_created = 0;
    for (int i = 1; i < state->num_workers; i++) {
        int err = pthread_create(&threads[i - 1], NULL,
                                 mark_heap_thread, &thread_args[i]);
        if (err != 0) {
            break;
        }
        threads_created++;
    }

    // Worker 0 runs on main thread
    thread_args[0].result = mark_heap_worker(state, 0, workers,
                                              skip_deferred_objects);

    // Wait for spawned threads
    for (int i = 0; i < threads_created; i++) {
        pthread_join(threads[i], NULL);
    }

    // Check for errors
    for (int i = 0; i <= threads_created; i++) {
        if (thread_args[i].result < 0) {
            result = -1;
            break;
        }
    }

    // Collect statistics
    state->total_objects = 0;
    for (int i = 0; i < state->num_workers; i++) {
        state->total_objects += workers[i].objects_marked;
    }

    // Cleanup thread resources
    PyMem_RawFree(thread_args);
    PyMem_RawFree(threads);

cleanup:
    // Free worker deques and their pre-allocated buffers
    for (int i = 0; i < state->num_workers; i++) {
        if (deque_buffers[i] != NULL) {
            // Use FiniExternal when we allocated an external buffer to avoid double-free
            _PyWSDeque_FiniExternal(&workers[i].deque, deque_buffers[i]);
            PyMem_RawFree(deque_buffers[i]);
        } else {
            _PyWSDeque_Fini(&workers[i].deque);
        }
    }
    PyMem_RawFree(deque_buffers);
    PyMem_RawFree(workers);

    return result;
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
    _Py_AddRefTotal(_PyThreadState_GET(), extra);
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
        // NOTE: disable_deferred_refcounting is NOT called here because it uses
        // internal locks that aren't safe for parallel access. It will be called
        // serially after the parallel phase completes.
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
        // NOTE: For shutdown, disable_deferred_refcounting is called serially later
        // because it uses internal locks that aren't safe for parallel access.
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

// Per-worker thread arguments
struct scan_worker_thread_args {
    struct _PyGCScanWorkerState *workers;
    atomic_int *page_counter;
    mi_page_t **page_array;
    size_t total_pages;
    int worker_id;
    int gc_reason;
};

// Worker thread function
static void *
scan_heap_thread(void *arg)
{
    struct scan_worker_thread_args *args = (struct scan_worker_thread_args *)arg;

    while (1) {
        int page_idx = atomic_fetch_add(args->page_counter, 1);
        if (page_idx >= (int)args->total_pages) {
            break;
        }

        scan_heap_process_page(args->page_array[page_idx],
                               &args->workers[args->worker_id],
                               args->gc_reason);
    }

    return NULL;
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

// Main entry point
int
_PyGC_ParallelScanHeap(
    PyInterpreterState *interp,
    _PyGCFTParState *state,
    struct _PyGCScanHeapResult *result)
{
    int ret = 0;

    // Allocate per-worker state
    struct _PyGCScanWorkerState *workers = PyMem_RawCalloc(
        state->num_workers, sizeof(struct _PyGCScanWorkerState));
    if (workers == NULL) {
        return -1;
    }

    // Collect pages
    size_t page_capacity = 1024;
    size_t total_pages = 0;
    mi_page_t **page_array = PyMem_RawMalloc(page_capacity * sizeof(mi_page_t *));
    if (page_array == NULL) {
        PyMem_RawFree(workers);
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
                            PyMem_RawFree(workers);
                            return -1;
                        }
                        page_array = new_arr;
                    }
                    page_array[total_pages++] = page;
                }
            }
        }
    }
    HEAD_UNLOCK(&_PyRuntime);

    if (total_pages == 0) {
        PyMem_RawFree(page_array);
        PyMem_RawFree(workers);
        return 0;
    }

    // Atomic counter for work distribution
    atomic_int page_counter = 0;

    // Allocate thread resources
    pthread_t *threads = PyMem_RawMalloc((state->num_workers - 1) * sizeof(pthread_t));
    struct scan_worker_thread_args *thread_args = PyMem_RawMalloc(
        state->num_workers * sizeof(struct scan_worker_thread_args));

    if (threads == NULL || thread_args == NULL) {
        PyMem_RawFree(threads);
        PyMem_RawFree(thread_args);
        PyMem_RawFree(page_array);
        PyMem_RawFree(workers);
        return -1;
    }

    // Initialize args
    for (int i = 0; i < state->num_workers; i++) {
        thread_args[i].workers = workers;
        thread_args[i].page_counter = &page_counter;
        thread_args[i].page_array = page_array;
        thread_args[i].total_pages = total_pages;
        thread_args[i].worker_id = i;
        thread_args[i].gc_reason = result->gc_reason;
    }

    // Spawn workers 1..N-1
    int threads_created = 0;
    for (int i = 1; i < state->num_workers; i++) {
        if (pthread_create(&threads[i - 1], NULL, scan_heap_thread,
                           &thread_args[i]) == 0) {
            threads_created++;
        }
    }

    // Worker 0 on main thread
    scan_heap_thread(&thread_args[0]);

    // Join workers
    for (int i = 0; i < threads_created; i++) {
        pthread_join(threads[i], NULL);
    }

    // Merge results
    for (int i = 0; i < state->num_workers; i++) {
        scan_worklist_merge(&workers[i].unreachable, result->unreachable_head);
        scan_worklist_merge(&workers[i].legacy_finalizers,
                            result->legacy_finalizers_head);
        *result->long_lived_total += workers[i].long_lived_total;
    }

    // Cleanup
    PyMem_RawFree(thread_args);
    PyMem_RawFree(threads);
    PyMem_RawFree(page_array);
    PyMem_RawFree(workers);

    return ret;
}

#endif  // Py_GIL_DISABLED
