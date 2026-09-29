// Parallel garbage collector for free-threaded Python.

#include "Python.h"

#if defined(Py_GIL_DISABLED) && defined(Py_PARALLEL_GC)

#include "pycore_gc.h"
#include "pycore_gc_ft_parallel.h"
#include "pycore_interp.h"
#include "pycore_lock.h"
#include "pycore_object_deferred.h"
#include "pycore_pystate.h"
#include "pycore_runtime.h"
#include "pycore_mimalloc.h"

#define _PyGC_BUCKET_MIN_CAPACITY 16

static inline bool
is_huge_page(mi_page_t *page)
{
    return page->xblock_size > MI_LARGE_OBJ_SIZE_MAX;
}

static size_t
count_pages(PyInterpreterState *interp)
{
    assert(interp->stoptheworld.world_stopped);

    size_t total_pages = 0;

    HEAD_LOCK(&_PyRuntime);

    _Py_FOR_EACH_TSTATE_UNLOCKED(interp, p) {
        struct _mimalloc_thread_state *m =
            &((_PyThreadStateImpl *)p)->mimalloc;
        if (!_Py_atomic_load_int(&m->initialized)) {
            continue;
        }

        total_pages += m->heaps[_Py_MIMALLOC_HEAP_GC].page_count;
        total_pages += m->heaps[_Py_MIMALLOC_HEAP_GC_PRE].page_count;
    }

    // Include pages left by threads that have exited.
    mi_abandoned_pool_t *pool = &interp->mimalloc.abandoned_pool;
    total_pages += _mi_abandoned_pool_count_pages(pool, _Py_MIMALLOC_HEAP_GC);
    total_pages += _mi_abandoned_pool_count_pages(
        pool, _Py_MIMALLOC_HEAP_GC_PRE);

    HEAD_UNLOCK(&_PyRuntime);

    return total_pages;
}

struct page_enum_context {
    _PyGCFTParState *state;
    int current_worker;       // For normal pages (sequential bucket filling)
    size_t current_count;
    size_t pages_per_worker;
    size_t assigned_pages;
    int huge_worker;          // For huge pages (round-robin)
    int error;
};

static void
assign_page_to_bucket(mi_page_t *page, struct page_enum_context *ctx)
{
    assert(page != NULL && "NULL page passed to assign_page_to_bucket");
    assert(ctx != NULL);
    assert(ctx->state != NULL);
    assert(ctx->state->buckets != NULL);
    assert(ctx->current_worker >= 0);
    assert(ctx->current_worker < ctx->state->num_workers);
    assert(ctx->huge_worker >= 0);
    assert(ctx->huge_worker < ctx->state->num_workers);

    _PyGCPageBucket *bucket;

    if (is_huge_page(page)) {
        // Spread huge pages because they can contain expensive traversals.
        bucket = &ctx->state->buckets[ctx->huge_worker];
        ctx->huge_worker = (ctx->huge_worker + 1) % ctx->state->num_workers;
    } else {
        // Keep ordinary pages contiguous for locality.
        bucket = &ctx->state->buckets[ctx->current_worker];
        ctx->current_count++;

        if (ctx->current_count >= ctx->pages_per_worker &&
            ctx->current_worker < ctx->state->num_workers - 1) {
            ctx->current_worker++;
            ctx->current_count = 0;
        }
    }

    if (bucket->num_pages >= bucket->capacity) {
        size_t new_capacity = bucket->capacity * 2;
        if (new_capacity < _PyGC_BUCKET_MIN_CAPACITY) {
            new_capacity = _PyGC_BUCKET_MIN_CAPACITY;
        }

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
    ctx->assigned_pages++;

    assert(bucket->num_pages <= bucket->capacity);
    assert(bucket->pages[bucket->num_pages - 1] == page);
}

static void
enumerate_heap_pages(mi_heap_t *heap, struct page_enum_context *ctx)
{
    assert(heap != NULL);
    assert(ctx != NULL);

    for (size_t bin = 0; bin <= MI_BIN_FULL; bin++) {
        mi_page_queue_t *queue = &heap->pages[bin];

        for (mi_page_t *page = queue->first; page != NULL; page = page->next) {
            assert(page->used <= page->capacity &&
                   "page->used exceeds page->capacity");

            if (page->used > 0) {
                assign_page_to_bucket(page, ctx);
                if (ctx->error) {
                    return;
                }
            }
        }
    }
}

static void
enumerate_abandoned_page_callback(mi_page_t *page, void *arg)
{
    struct page_enum_context *ctx = (struct page_enum_context *)arg;
    assign_page_to_bucket(page, ctx);
}


#ifdef Py_DEBUG

static void
assert_bucket_valid(const _PyGCPageBucket *bucket)
{
    assert(bucket->num_pages <= bucket->capacity);

    if (bucket->num_pages > 0) {
        assert(bucket->pages != NULL);
    }

    if (bucket->capacity > 0) {
        assert(bucket->pages != NULL);
    }

    for (size_t i = 0; i < bucket->num_pages; i++) {
        assert(bucket->pages[i] != NULL && "NULL page pointer in bucket");
    }
}

static void
assert_buckets_valid(const _PyGCFTParState *state, size_t expected_total)
{
    assert(state->buckets != NULL && "buckets array is NULL");
    assert(state->num_workers > 0 && "num_workers must be positive");

    size_t actual_total = 0;

    for (int i = 0; i < state->num_workers; i++) {
        assert_bucket_valid(&state->buckets[i]);
        actual_total += state->buckets[i].num_pages;

        assert(state->buckets[i].num_pages <= expected_total);
    }

    assert(actual_total == expected_total &&
           "Total pages in buckets doesn't match expected count");
}

#define ASSERT_BUCKET_VALID(bucket) assert_bucket_valid(bucket)
#define ASSERT_BUCKETS_VALID(state, total) assert_buckets_valid(state, total)

#else  // !Py_DEBUG

#define ASSERT_BUCKET_VALID(bucket) ((void)0)
#define ASSERT_BUCKETS_VALID(state, total) ((void)0)

#endif  // Py_DEBUG

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

    ASSERT_BUCKET_VALID(bucket);
    return 0;
}

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

    assert(bucket->pages == NULL);
    assert(bucket->num_pages == 0);
    assert(bucket->capacity == 0);
}

int
_PyGC_AssignPagesToBuckets(PyInterpreterState *interp,
                           _PyGCFTParState *state)
{
    assert(interp != NULL);
    assert(state != NULL);
    assert(interp->stoptheworld.world_stopped);
    assert(state->num_workers > 0);
    assert(state->buckets == NULL);

    size_t total_pages = count_pages(interp);

    state->buckets = PyMem_RawCalloc(
        state->num_workers, sizeof(_PyGCPageBucket));
    if (state->buckets == NULL) {
        return -1;
    }

    size_t initial_capacity = (
        total_pages / state->num_workers) + _PyGC_BUCKET_MIN_CAPACITY;
    for (int i = 0; i < state->num_workers; i++) {
        if (bucket_init(&state->buckets[i], initial_capacity) < 0) {
            for (int j = 0; j < i; j++) {
                bucket_free(&state->buckets[j]);
            }
            PyMem_RawFree(state->buckets);
            state->buckets = NULL;
            return -1;
        }
    }

    struct page_enum_context ctx = {
        .state = state,
        .current_worker = 0,
        .current_count = 0,
        .pages_per_worker = total_pages / state->num_workers,
        .assigned_pages = 0,
        .huge_worker = 0,
        .error = 0
    };

    if (ctx.pages_per_worker == 0) {
        ctx.pages_per_worker = 1;
    }

    HEAD_LOCK(&_PyRuntime);

    _Py_FOR_EACH_TSTATE_UNLOCKED(interp, p) {
        struct _mimalloc_thread_state *m =
            &((_PyThreadStateImpl *)p)->mimalloc;
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

    // Include pages abandoned by threads which have exited.
    if (!ctx.error) {
        mi_abandoned_pool_t *pool = &interp->mimalloc.abandoned_pool;
        _mi_abandoned_pool_visit_pages(
            pool, _Py_MIMALLOC_HEAP_GC,
            enumerate_abandoned_page_callback, &ctx);
        if (!ctx.error) {
            _mi_abandoned_pool_visit_pages(
                pool, _Py_MIMALLOC_HEAP_GC_PRE,
                enumerate_abandoned_page_callback, &ctx);
        }
    }

    HEAD_UNLOCK(&_PyRuntime);

    if (ctx.error) {
        _PyGC_FreeBuckets(state);
        return -1;
    }

    ASSERT_BUCKETS_VALID(state, ctx.assigned_pages);

    return 0;
}

void
_PyGC_FreeBuckets(_PyGCFTParState *state)
{
    assert(state != NULL);

    if (state->buckets != NULL) {
#ifdef Py_DEBUG
        for (int i = 0; i < state->num_workers; i++) {
            ASSERT_BUCKET_VALID(&state->buckets[i]);
        }
#endif

        for (int i = 0; i < state->num_workers; i++) {
            bucket_free(&state->buckets[i]);
        }
        PyMem_RawFree(state->buckets);
        state->buckets = NULL;
    }

    assert(state->buckets == NULL && "buckets not NULL after free");
}


static inline PyObject *
block_to_object(void *block, Py_ssize_t offset)
{
    if (block == NULL) {
        return NULL;
    }
    PyObject *op = (PyObject *)((char*)block + offset);

    if (!_PyObject_GC_IS_TRACKED(op)) {
        return NULL;
    }
    if (_Py_atomic_load_uint8_relaxed(&op->ob_gc_bits) & _PyGC_BITS_FROZEN) {
        return NULL;
    }
    return op;
}


typedef struct {
    _PyGCWorkDescriptor *work;
    _PyGCWorkerState *worker;
} _PyGCMarkContext;

static int par_mark_traverse_object(PyObject *op, _PyGCMarkContext *ctx);
static int mark_heap_find_roots_page(mi_page_t *page,
                                     _PyGCMarkContext *ctx,
                                     Py_ssize_t offset,
                                     int skip_deferred);

static void
mark_heap_pool_work(_PyGCThreadPool *pool, int worker_id)
{
    _PyGCWorkDescriptor *work = pool->current_work;
    assert(work != NULL);
    assert(work->buckets != NULL);

    _PyGCWorkerState *worker = &pool->workers[worker_id];
    _PyGCPageBucket *bucket = &work->buckets[worker_id];
    _PyGCMarkContext ctx = {
        .work = work,
        .worker = worker,
    };

    Py_ssize_t offset_base = 0;
    if (_PyMem_DebugEnabled()) {
        offset_base += 2 * sizeof(size_t);
    }
    Py_ssize_t offset_pre = offset_base + 2 * sizeof(PyObject*);

    for (size_t i = 0; i < bucket->num_pages; i++) {
        mi_page_t *page = bucket->pages[i];
        Py_ssize_t offset = (page->tag == _Py_MIMALLOC_HEAP_GC_PRE)
                            ? offset_pre : offset_base;
        if (mark_heap_find_roots_page(page, &ctx, offset,
                                      work->skip_deferred) < 0) {
            _Py_atomic_store_int_relaxed(&work->error_flag, 1);
            break;
        }
    }

    // Release this worker's root-scanning token.  Queued and in-flight
    // objects each hold a token until their traversal completes.
    _Py_atomic_add_ssize(&work->outstanding, -1);

    while (!_Py_atomic_load_int_relaxed(&work->error_flag)) {
        PyObject *op = (PyObject *)_PyWSDeque_Take(&worker->deque);

        if (op == NULL) {
            for (int i = 1; i < work->active_workers; i++) {
                int victim = (worker_id + i) % work->active_workers;
                op = (PyObject *)_PyWSDeque_Steal(
                    &pool->workers[victim].deque);
                if (op != NULL) {
                    break;
                }
            }
        }

        if (op != NULL) {
            int err = par_mark_traverse_object(op, &ctx);
            _Py_atomic_add_ssize(&work->outstanding, -1);
            if (err < 0) {
                _Py_atomic_store_int_relaxed(&work->error_flag, 1);
                return;
            }
            continue;
        }

        if (_Py_atomic_load_ssize(&work->outstanding) == 0) {
            return;
        }
        _Py_yield();
    }
}

// Wake the selected helpers and run worker zero on the calling thread.
static void
dispatch_and_wait(_PyGCThreadPool *pool, size_t active_workers)
{
    if (active_workers > (size_t)pool->num_workers) {
        active_workers = (size_t)pool->num_workers;
    }
    if (active_workers < 1) {
        active_workers = 1;
    }

    _PyGC_MUTEX_LOCK(&pool->done_mutex);
    pool->workers_done_count = 0;
    _PyGC_MUTEX_UNLOCK(&pool->done_mutex);

    for (size_t i = 1; i < active_workers; i++) {
        _PyGC_MUTEX_LOCK(&pool->workers[i].wake_mutex);
        pool->workers[i].wake_flag = 1;
        _PyGC_COND_SIGNAL(&pool->workers[i].wake_cond);
        _PyGC_MUTEX_UNLOCK(&pool->workers[i].wake_mutex);
    }

    mark_heap_pool_work(pool, 0);

    int expected = (int)(active_workers - 1);
    _PyGC_MUTEX_LOCK(&pool->done_mutex);
    while (pool->workers_done_count < expected) {
        _PyGC_COND_WAIT(&pool->done_cond, &pool->done_mutex);
    }
    _PyGC_MUTEX_UNLOCK(&pool->done_mutex);
}

struct _PyGCPoolWorkerArgs {
    _PyGCThreadPool *pool;
    int worker_id;
};

static void
thread_pool_worker(void *arg)
{
    _PyGCPoolWorkerArgs *args = (_PyGCPoolWorkerArgs *)arg;
    _PyGCThreadPool *pool = args->pool;
    int worker_id = args->worker_id;

    _PyGCWorkerState *worker = &pool->workers[worker_id];

    while (1) {
        _PyGC_MUTEX_LOCK(&worker->wake_mutex);
        while (!worker->wake_flag) {
            _PyGC_COND_WAIT(&worker->wake_cond, &worker->wake_mutex);
        }
        worker->wake_flag = 0;
        _PyGC_MUTEX_UNLOCK(&worker->wake_mutex);

        if (_Py_atomic_load_int_relaxed(&pool->shutdown)) {
            break;
        }

        mark_heap_pool_work(pool, worker_id);

        _PyGC_MUTEX_LOCK(&pool->done_mutex);
        pool->workers_done_count++;
        _PyGC_COND_SIGNAL(&pool->done_cond);
        _PyGC_MUTEX_UNLOCK(&pool->done_mutex);
    }

}

static void
thread_pool_stop_workers(_PyGCThreadPool *pool)
{
    if (pool->threads_created == 0) {
        return;
    }

    _Py_atomic_store_int_relaxed(&pool->shutdown, 1);
    for (size_t i = 1; i <= pool->threads_created; i++) {
        _PyGC_MUTEX_LOCK(&pool->workers[i].wake_mutex);
        pool->workers[i].wake_flag = 1;
        _PyGC_COND_SIGNAL(&pool->workers[i].wake_cond);
        _PyGC_MUTEX_UNLOCK(&pool->workers[i].wake_mutex);
    }
    for (size_t i = 0; i < pool->threads_created; i++) {
        if (PyThread_join_thread(pool->threads[i]) != 0) {
            Py_FatalError("failed to join parallel GC worker");
        }
    }
    pool->threads_created = 0;
}

static int
thread_pool_start_workers(_PyGCThreadPool *pool)
{
    assert(pool->threads_created == 0);
    _Py_atomic_store_int_relaxed(&pool->shutdown, 0);

    _PyGCPoolWorkerArgs *worker_args = pool->worker_args;
    for (int i = 0; i < pool->num_workers - 1; i++) {
        worker_args[i].pool = pool;
        worker_args[i].worker_id = i + 1;
        pool->workers[i + 1].wake_flag = 0;

        PyThread_ident_t ident;
        int rc = PyThread_start_joinable_thread(
            thread_pool_worker, &worker_args[i],
            &ident, &pool->threads[i]);
        if (rc != 0) {
            thread_pool_stop_workers(pool);
            PyErr_Format(PyExc_RuntimeError,
                         "failed to create parallel GC worker: error %d", rc);
            return -1;
        }
        pool->threads_created++;
    }
    return 0;
}

static void
thread_pool_free(_PyGCThreadPool *pool, int initialized_workers)
{
    PyMem_RawFree(pool->worker_args);
    for (int i = 0; i < initialized_workers; i++) {
        _PyGC_COND_FINI(&pool->workers[i].wake_cond);
        _PyGC_MUTEX_FINI(&pool->workers[i].wake_mutex);
        _PyWSDeque_Fini(&pool->workers[i].deque);
    }
    PyMem_RawFree(pool->workers);
    PyMem_RawFree(pool->threads);
    _PyGC_COND_FINI(&pool->done_cond);
    _PyGC_MUTEX_FINI(&pool->done_mutex);
    PyMem_RawFree(pool);
}

int
_PyGC_ThreadPoolInit(PyInterpreterState *interp, int num_workers)
{
    assert(interp != NULL);
    if (num_workers < _PyGC_PARALLEL_MIN_WORKERS ||
        num_workers > _PyGC_PARALLEL_MAX_WORKERS)
    {
        PyErr_Format(PyExc_ValueError,
                     "num_workers must be between %d and %d, got %d",
                     _PyGC_PARALLEL_MIN_WORKERS,
                     _PyGC_PARALLEL_MAX_WORKERS,
                     num_workers);
        return -1;
    }
    if (interp->gc.thread_pool != NULL) {
        return -1;
    }

    _PyGCThreadPool *pool = PyMem_RawCalloc(1, sizeof(_PyGCThreadPool));
    if (pool == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    pool->num_workers = num_workers;
    _Py_atomic_store_int_relaxed(&pool->shutdown, 0);
    pool->threads_created = 0;

    _PyGC_MUTEX_INIT(&pool->done_mutex);
    _PyGC_COND_INIT(&pool->done_cond);
    pool->workers_done_count = 0;
    int initialized_workers = 0;

    pool->threads = PyMem_RawCalloc(
        num_workers - 1, sizeof(PyThread_handle_t));
    if (pool->threads == NULL) {
        goto no_memory;
    }

    pool->workers = PyMem_RawCalloc(num_workers, sizeof(_PyGCWorkerState));
    if (pool->workers == NULL) {
        goto no_memory;
    }

    for (int i = 0; i < num_workers; i++) {
        _PyGCWorkerState *worker = &pool->workers[i];
        if (_PyWSDeque_Init(&worker->deque) < 0) {
            goto no_memory;
        }
        _PyGC_MUTEX_INIT(&worker->wake_mutex);
        _PyGC_COND_INIT(&worker->wake_cond);
        worker->wake_flag = 0;
        initialized_workers++;
    }

    pool->worker_args = PyMem_RawCalloc(
        num_workers - 1, sizeof(_PyGCPoolWorkerArgs));
    if (pool->worker_args == NULL) {
        goto no_memory;
    }

    if (thread_pool_start_workers(pool) < 0) {
        goto error;
    }

    interp->gc.thread_pool = pool;

    assert(pool->threads_created == (size_t)(pool->num_workers - 1));

    return 0;

no_memory:
    PyErr_NoMemory();
error:
    thread_pool_free(pool, initialized_workers);
    return -1;
}

void
_PyGC_ThreadPoolFini(PyInterpreterState *interp)
{
    assert(interp != NULL);
    _PyGCThreadPool *pool = interp->gc.thread_pool;
    if (pool == NULL) {
        return;
    }

    thread_pool_stop_workers(pool);
    thread_pool_free(pool, pool->num_workers);

    interp->gc.thread_pool = NULL;
    interp->gc.parallel_gc_enabled = 0;
    interp->gc.parallel_gc_num_workers = 0;
}

void
_PyGC_ThreadPoolBeforeFork(PyInterpreterState *interp)
{
    if (interp->gc.parallel_gc_enabled && interp->gc.thread_pool != NULL) {
        thread_pool_stop_workers(interp->gc.thread_pool);
    }
}

void
_PyGC_ThreadPoolAfterFork(PyInterpreterState *interp)
{
    _PyGCThreadPool *pool = interp->gc.thread_pool;
    if (!interp->gc.parallel_gc_enabled || pool == NULL ||
        pool->threads_created != 0)
    {
        return;
    }

    pool->current_work = NULL;
    pool->workers_done_count = 0;
    if (thread_pool_start_workers(pool) < 0) {
        // Fork itself succeeded.  If helper recreation fails (for example
        // under a thread-creation fault-injection test), keep the parent on
        // the serial collector rather than leaking an unrelated exception.
        interp->gc.parallel_gc_enabled = 0;
        interp->gc.parallel_gc_num_workers = 0;
        PyErr_Clear();
    }
}

void
_PyGC_ThreadPoolAfterForkChild(PyInterpreterState *interp)
{
    // Creating threads between fork() and exec() is unsafe.  Leave the child
    // serial; gc.enable_parallel() can create a fresh pool later.
    interp->gc.parallel_gc_enabled = 0;
    interp->gc.parallel_gc_num_workers = 0;
}

static int
par_mark_visitproc(PyObject *child, void *arg)
{
    _PyGCMarkContext *ctx = (_PyGCMarkContext *)arg;

    if (child == NULL) {
        return 0;
    }

    if (!_PyObject_GC_IS_TRACKED(child)) {
        return 0;
    }
    if (_Py_atomic_load_uint8_relaxed(&child->ob_gc_bits) &
        _PyGC_BITS_FROZEN)
    {
        return 0;
    }

    if (_PyGC_TryMarkReachable(child)) {
        _Py_atomic_add_ssize(&ctx->work->outstanding, 1);
        if (_PyWSDeque_Push(&ctx->worker->deque, child) < 0) {
            _Py_atomic_add_ssize(&ctx->work->outstanding, -1);
            return -1;
        }
    }
    return 0;
}

static int
par_mark_traverse_object(PyObject *op, _PyGCMarkContext *ctx)
{
    assert(op != NULL);
    traverseproc traverse = Py_TYPE(op)->tp_traverse;
    if (traverse == NULL) {
        return 0;
    }
    return traverse(op, par_mark_visitproc, ctx);
}

typedef struct {
    Py_ssize_t offset;
    _PyGCMarkContext *mark;
    int skip_deferred;
    int error;
} _PyGCMarkRootsVisitorArgs;

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

    if (_Py_atomic_load_uint8_relaxed(&op->ob_gc_bits) & _PyGC_BITS_ALIVE) {
        _PyGC_TryClearBit(op, _PyGC_BITS_UNREACHABLE);
        return true;
    }

    if (!_PyGC_IsUnreachable(op)) {
        return true;
    }

    Py_ssize_t gc_refs = gc_get_refs_atomic(op);
    _PyObject_ASSERT_WITH_MSG(op, gc_refs >= 0,
                              "refcount is too small");

    // GH-129236: Keep deferred objects alive if stack scanning was incomplete.
    int keep_alive = (ctx->skip_deferred && _PyObject_HasDeferredRefcount(op));

    if (gc_refs != 0 || keep_alive) {
        if (_PyGC_TryMarkReachable(op)) {
            _Py_atomic_add_ssize(&ctx->mark->work->outstanding, 1);
            if (_PyWSDeque_Push(&ctx->mark->worker->deque, op) < 0) {
                _Py_atomic_add_ssize(&ctx->mark->work->outstanding, -1);
                ctx->error = 1;
                return false;
            }
        }
    }

    return true;
}

static int
mark_heap_find_roots_page(mi_page_t *page, _PyGCMarkContext *ctx,
                          Py_ssize_t offset, int skip_deferred)
{
    assert(page != NULL);

    mi_heap_area_t area;
    _mi_heap_area_init(&area, page);

    _PyGCMarkRootsVisitorArgs visitor_args = {
        .offset = offset,
        .mark = ctx,
        .skip_deferred = skip_deferred,
        .error = 0,
    };

    _mi_heap_area_visit_blocks(
        &area, page, mark_heap_roots_visitor, &visitor_args);
    return visitor_args.error ? -1 : 0;
}


int
_PyGC_ParallelMarkHeapWithPool(PyInterpreterState *interp,
                                _PyGCFTParState *state,
                                int skip_deferred_objects)
{
    // Relaxed marking operations are only safe while the world is stopped.
    assert(interp->stoptheworld.world_stopped);
    _PyGCThreadPool *pool = interp->gc.thread_pool;
    assert(pool != NULL);
    assert(state->num_workers <= pool->num_workers);
    assert(state->buckets != NULL);

    _PyGCWorkDescriptor work = {
        .buckets = state->buckets,
        .skip_deferred = skip_deferred_objects,
        .error_flag = 0,
        .outstanding = state->num_workers,
        .active_workers = state->num_workers,
    };
    pool->current_work = &work;

    dispatch_and_wait(pool, state->num_workers);

    // A failed traversal can leave borrowed object pointers queued.  All
    // workers have stopped, so discard any remaining entries before objects
    // can be freed or the deques are reused by a later collection.
    for (int i = 0; i < pool->num_workers; i++) {
        _PyWSDeque_Reset(&pool->workers[i].deque);
    }

    pool->current_work = NULL;

    return _Py_atomic_load_int_relaxed(&work.error_flag) ? -1 : 0;
}

#endif  // Py_GIL_DISABLED && Py_PARALLEL_GC
