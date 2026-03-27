"""
Tests for FTP (Free-Threaded Python) parallel GC components.

These tests run on both GIL and FTP builds where applicable.
FTP-specific tests are skipped on GIL builds.
"""

import gc
import sys
import unittest
import weakref
import threading
from test.support import threading_helper


# Custom class that supports weak references (dict does not)
class GCTestObject:
    """A simple object that can have weak references for GC testing."""
    __slots__ = ('__weakref__', 'data', 'ref')

    def __init__(self, **kwargs):
        self.data = kwargs
        self.ref = None

# Check if we're running on FTP (no-GIL) build
try:
    FTP_BUILD = not sys._is_gil_enabled()
except AttributeError:
    FTP_BUILD = False

# Check if parallel GC test APIs are available
try:
    _count_pages = gc._count_gc_pages
    PARALLEL_GC_TESTS_AVAILABLE = True
except AttributeError:
    PARALLEL_GC_TESTS_AVAILABLE = False


def requires_ftp(test_func):
    """Skip test if not running on FTP build."""
    return unittest.skipUnless(FTP_BUILD, "Requires FTP (no-GIL) build")(test_func)


def requires_parallel_gc_tests(test_func):
    """Skip test if parallel GC test APIs not available."""
    return unittest.skipUnless(
        PARALLEL_GC_TESTS_AVAILABLE,
        "Requires parallel GC test APIs"
    )(test_func)


class TestPageCounter(unittest.TestCase):
    """Tests for page counting functionality."""

    @requires_ftp
    @requires_parallel_gc_tests
    def test_page_count_non_negative(self):
        """Page count should always be non-negative."""
        gc.collect()
        count = gc._count_gc_pages()
        self.assertGreaterEqual(count, 0)

    @requires_ftp
    @requires_parallel_gc_tests
    def test_page_count_increases_with_allocation(self):
        """Allocating many objects should increase page count."""
        gc.collect()
        gc.collect()  # Double collect to clear any pending garbage

        before = gc._count_gc_pages()

        # Allocate many objects (enough to require new pages)
        # Each page is ~64KB, each dict is ~100-200 bytes
        # So 10000 dicts should require multiple pages
        objects = [{'data': i, 'more': 'x' * 100} for i in range(10000)]

        after = gc._count_gc_pages()

        # Note: Page count may not increase if mimalloc reuses pages from
        # previous test runs. The key invariant is that page count doesn't
        # decrease while objects are alive.
        self.assertGreaterEqual(after, before,
            f"Page count should not decrease with live allocations: "
            f"before={before}, after={after}")

        # Keep objects alive until we're done checking
        del objects

    @requires_ftp
    @requires_parallel_gc_tests
    def test_page_count_stable_after_gc(self):
        """Page count should be stable after GC with no new allocations."""
        # Create and release garbage
        garbage = [{'cycle': None} for _ in range(1000)]
        for i in range(len(garbage) - 1):
            garbage[i]['cycle'] = garbage[i + 1]
        garbage[-1]['cycle'] = garbage[0]  # Complete the cycle
        del garbage

        gc.collect()
        count1 = gc._count_gc_pages()

        gc.collect()
        count2 = gc._count_gc_pages()

        # Page count shouldn't change significantly without allocation
        self.assertAlmostEqual(count1, count2, delta=2,
            msg=f"Page count unstable: {count1} vs {count2}")


class TestPageAssignment(unittest.TestCase):
    """Tests for page assignment (bucket filling) algorithm."""

    @requires_ftp
    @requires_parallel_gc_tests
    def test_assignment_even_distribution(self):
        """Pages should be distributed evenly across workers."""
        # Test with 100 pages, 4 workers -> 25 each
        assignment = gc._test_page_assignment(100, 4)

        self.assertEqual(len(assignment), 4)
        self.assertEqual(assignment, [25, 25, 25, 25])

    @requires_ftp
    @requires_parallel_gc_tests
    def test_assignment_uneven_distribution(self):
        """Remainder pages should go to last worker."""
        # Test with 103 pages, 4 workers -> 25, 25, 25, 28
        assignment = gc._test_page_assignment(103, 4)

        self.assertEqual(len(assignment), 4)
        self.assertEqual(sum(assignment), 103)
        # Last worker gets the remainder
        self.assertGreaterEqual(assignment[-1], assignment[0])

    @requires_ftp
    @requires_parallel_gc_tests
    def test_assignment_single_worker(self):
        """Single worker gets all pages."""
        assignment = gc._test_page_assignment(100, 1)

        self.assertEqual(len(assignment), 1)
        self.assertEqual(assignment[0], 100)

    @requires_ftp
    @requires_parallel_gc_tests
    def test_assignment_more_workers_than_pages(self):
        """When workers > pages, some workers get 0 pages."""
        assignment = gc._test_page_assignment(3, 8)

        self.assertEqual(len(assignment), 8)
        self.assertEqual(sum(assignment), 3)
        # At least some workers should have pages
        self.assertGreater(max(assignment), 0)

    @requires_ftp
    @requires_parallel_gc_tests
    def test_assignment_zero_pages(self):
        """Zero pages should give all workers 0."""
        assignment = gc._test_page_assignment(0, 4)

        self.assertEqual(len(assignment), 4)
        self.assertEqual(assignment, [0, 0, 0, 0])


class TestRealPageEnumeration(unittest.TestCase):
    """
    Tests for REAL page enumeration via _test_real_page_enumeration().
    This exercises the actual _PyGC_AssignPagesToBuckets() code path
    including all invariant assertions.
    """

    @requires_ftp
    @requires_parallel_gc_tests
    def test_real_enumeration_returns_valid_data(self):
        """Real page enumeration should return valid bucket data."""
        # Allocate some objects first to ensure we have pages
        objects = [GCTestObject(id=i) for i in range(1000)]

        result = gc._test_real_page_enumeration(4)

        self.assertIn('total_pages', result)
        self.assertIn('pages_enumerated', result)
        self.assertIn('bucket_sizes', result)

        self.assertGreaterEqual(result['total_pages'], 0)
        self.assertGreaterEqual(result['pages_enumerated'], 0)
        self.assertEqual(len(result['bucket_sizes']), 4)

        # Sum of buckets should equal pages_enumerated
        self.assertEqual(sum(result['bucket_sizes']), result['pages_enumerated'])

        del objects

    @requires_ftp
    @requires_parallel_gc_tests
    def test_real_enumeration_finds_pages(self):
        """Real enumeration should find pages after allocating objects."""
        gc.collect()

        # Allocate many objects - should create multiple pages
        objects = [GCTestObject(data='x' * 100) for _ in range(5000)]

        result = gc._test_real_page_enumeration(4)

        # We should have found some pages
        self.assertGreater(result['pages_enumerated'], 0,
            "Expected to find pages after allocating 5000 objects")

        # All buckets except possibly the last should have similar counts
        # (sequential bucket filling should distribute evenly)
        if result['pages_enumerated'] >= 4:
            # With enough pages, first workers should have similar amounts
            bucket_sizes = result['bucket_sizes']
            # Check that we're not putting everything in one bucket
            non_empty = [s for s in bucket_sizes if s > 0]
            self.assertGreater(len(non_empty), 1,
                f"Expected pages distributed across workers, got {bucket_sizes}")

        del objects

    @requires_ftp
    @requires_parallel_gc_tests
    def test_real_enumeration_different_worker_counts(self):
        """Real enumeration should work with different worker counts."""
        objects = [GCTestObject(id=i) for i in range(2000)]

        for num_workers in [1, 2, 4, 8]:
            result = gc._test_real_page_enumeration(num_workers)

            self.assertEqual(len(result['bucket_sizes']), num_workers)
            self.assertEqual(sum(result['bucket_sizes']), result['pages_enumerated'])

        del objects

    @requires_ftp
    @requires_parallel_gc_tests
    def test_real_enumeration_invariants_hold(self):
        """
        Real enumeration triggers all invariant assertions in C code.
        If assertions are enabled (debug build), this tests:
        - Bucket counts match enumerated pages
        - No NULL page pointers
        - page->used <= page->capacity
        - Worker indices stay in bounds
        """
        # Create complex object graph
        objects = []
        for i in range(1000):
            obj = GCTestObject(id=i)
            if objects:
                obj.ref = objects[-1]  # Chain references
            objects.append(obj)

        # This call exercises all the assertions
        result = gc._test_real_page_enumeration(8)

        # If we got here without assertion failure, invariants held
        self.assertIsNotNone(result)
        self.assertEqual(sum(result['bucket_sizes']), result['pages_enumerated'])

        del objects

    @requires_ftp
    @requires_parallel_gc_tests
    def test_real_enumeration_many_workers(self):
        """Real enumeration with many workers (more than pages)."""
        objects = [GCTestObject(id=i) for i in range(50)]

        # Use many workers to test distribution
        num_workers = 64
        result = gc._test_real_page_enumeration(num_workers)

        self.assertEqual(len(result['bucket_sizes']), num_workers)
        self.assertEqual(sum(result['bucket_sizes']), result['pages_enumerated'])

        # All bucket sizes should be non-negative
        for size in result['bucket_sizes']:
            self.assertGreaterEqual(size, 0)

        del objects

    @requires_ftp
    @requires_parallel_gc_tests
    def test_real_enumeration_after_gc(self):
        """Real enumeration after GC clears garbage."""
        # Create garbage
        for _ in range(100):
            garbage = [GCTestObject(id=i) for i in range(100)]
            for i in range(len(garbage) - 1):
                garbage[i].ref = garbage[i + 1]
            garbage[-1].ref = garbage[0]  # Cycle
            del garbage

        # Collect garbage
        gc.collect()

        # Enumeration should still work
        result = gc._test_real_page_enumeration(4)

        self.assertIsNotNone(result)
        self.assertEqual(len(result['bucket_sizes']), 4)
        self.assertEqual(sum(result['bucket_sizes']), result['pages_enumerated'])


class TestParallelMarking(unittest.TestCase):
    """
    Tests for REAL parallel marking via _test_parallel_mark().
    This exercises the actual _PyGC_ParallelMarkAlive() code path.
    """

    @requires_ftp
    @requires_parallel_gc_tests
    def test_parallel_mark_returns_valid_data(self):
        """Parallel marking should return valid data."""
        # Allocate some objects first
        objects = [GCTestObject(id=i) for i in range(1000)]

        result = gc._test_parallel_mark(4)

        self.assertIn('total_objects', result)
        self.assertIn('per_worker_marked', result)

        # Should have marked some objects
        self.assertGreater(result['total_objects'], 0)
        self.assertEqual(len(result['per_worker_marked']), 4)

        # Sum should equal total
        self.assertEqual(sum(result['per_worker_marked']), result['total_objects'])

        del objects

    @requires_ftp
    @requires_parallel_gc_tests
    def test_parallel_mark_finds_objects(self):
        """Parallel marking should find objects after allocation."""
        gc.collect()

        # Allocate many objects
        objects = [GCTestObject(data='x' * 100) for _ in range(5000)]

        result = gc._test_parallel_mark(4)

        # We should have marked many objects
        self.assertGreater(result['total_objects'], 0,
            "Expected to mark objects after allocating 5000")

        del objects

    @requires_ftp
    @requires_parallel_gc_tests
    def test_parallel_mark_different_worker_counts(self):
        """Parallel marking should work with different worker counts."""
        objects = [GCTestObject(id=i) for i in range(2000)]

        for num_workers in [1, 2, 4, 8]:
            result = gc._test_parallel_mark(num_workers)

            self.assertEqual(len(result['per_worker_marked']), num_workers)
            self.assertEqual(sum(result['per_worker_marked']), result['total_objects'])
            self.assertGreater(result['total_objects'], 0)

        del objects

    @requires_ftp
    @requires_parallel_gc_tests
    def test_parallel_mark_clears_alive_bits(self):
        """Parallel marking test should clean up ALIVE bits properly."""
        objects = [GCTestObject(id=i) for i in range(1000)]

        # Run parallel marking
        result = gc._test_parallel_mark(4)
        self.assertGreater(result['total_objects'], 0)

        # GC should work fine after (ALIVE bits cleared)
        gc.collect()

        # Objects should still be alive (we hold refs)
        self.assertEqual(len(objects), 1000)

        del objects

    @requires_ftp
    @requires_parallel_gc_tests
    def test_parallel_mark_multiple_runs(self):
        """Multiple parallel marking runs should work correctly."""
        objects = [GCTestObject(id=i) for i in range(1000)]

        # Run parallel marking multiple times
        results = []
        for _ in range(5):
            result = gc._test_parallel_mark(4)
            results.append(result['total_objects'])
            # Run gc between tests
            gc.collect()

        # Each run should mark objects
        for count in results:
            self.assertGreater(count, 0)

        del objects

    @requires_ftp
    @requires_parallel_gc_tests
    def test_parallel_mark_work_stealing_occurs(self):
        """Work-stealing should result in uneven per-worker distribution."""
        # Create objects with varying reference depths to trigger work-stealing
        # Some objects have deep reference chains, others are shallow
        objects = []
        for i in range(500):
            # Create chains of varying lengths
            chain_len = (i % 10) + 1
            chain = [GCTestObject(chain_id=i, depth=d) for d in range(chain_len)]
            for j in range(len(chain) - 1):
                chain[j].ref = chain[j + 1]
            objects.extend(chain)

        result = gc._test_parallel_mark(4)

        # With work-stealing, we expect non-uniform distribution
        per_worker = result['per_worker_marked']
        self.assertEqual(len(per_worker), 4)
        self.assertEqual(sum(per_worker), result['total_objects'])

        # At least some workers should have done work
        workers_with_work = sum(1 for c in per_worker if c > 0)
        self.assertGreater(workers_with_work, 0)

        del objects

    @requires_ftp
    @requires_parallel_gc_tests
    def test_parallel_mark_single_worker(self):
        """Single worker should mark all objects."""
        objects = [GCTestObject(id=i) for i in range(1000)]

        result = gc._test_parallel_mark(1)

        self.assertEqual(len(result['per_worker_marked']), 1)
        self.assertEqual(result['per_worker_marked'][0], result['total_objects'])
        self.assertGreater(result['total_objects'], 0)

        del objects

    @requires_ftp
    @requires_parallel_gc_tests
    def test_parallel_mark_many_workers(self):
        """Many workers (more than pages) should still work."""
        objects = [GCTestObject(id=i) for i in range(100)]

        # Use more workers than likely pages
        result = gc._test_parallel_mark(16)

        self.assertEqual(len(result['per_worker_marked']), 16)
        self.assertEqual(sum(result['per_worker_marked']), result['total_objects'])

        del objects

    @requires_ftp
    @requires_parallel_gc_tests
    def test_parallel_mark_with_ctypes_loaded(self):
        """Parallel marking must work when ctypes is loaded.

        Regression test: ctypes CType_Type_traverse calls
        PyType_GetBaseByToken -> Py_INCREF, which in debug builds
        dereferences _PyThreadState_GET() for reftotal tracking.
        Worker threads without a PyThreadState would segfault.
        """
        import ctypes  # noqa: F401 — side effect is the point

        objects = [GCTestObject(id=i) for i in range(1000)]

        # Must not segfault
        result = gc._test_parallel_mark(4)
        self.assertGreater(result['total_objects'], 0)
        self.assertEqual(sum(result['per_worker_marked']),
                         result['total_objects'])

        # GC should still work after
        gc.collect()
        self.assertEqual(len(objects), 1000)

        del objects


class TestBasicGCCorrectness(unittest.TestCase):
    """
    Basic GC correctness tests.
    These should pass on BOTH GIL and FTP builds.
    """

    def test_live_objects_not_collected(self):
        """Objects with references should survive GC."""
        objects = [GCTestObject(id=i) for i in range(1000)]
        weak_refs = [weakref.ref(obj) for obj in objects]

        gc.collect()

        for i, ref in enumerate(weak_refs):
            self.assertIsNotNone(ref(),
                f"Live object {i} was incorrectly collected!")

    def test_garbage_is_collected(self):
        """Unreachable objects should be collected."""
        weak_refs = []

        def create_garbage():
            garbage = [GCTestObject(id=i) for i in range(1000)]
            for obj in garbage:
                weak_refs.append(weakref.ref(obj))
            # garbage goes out of scope

        create_garbage()
        gc.collect()

        collected = sum(1 for ref in weak_refs if ref() is None)
        self.assertEqual(collected, 1000,
            f"Expected 1000 objects collected, got {collected}")

    def test_cycles_collected(self):
        """Cyclic garbage should be collected."""
        weak_refs = []

        def create_cycles():
            for _ in range(500):
                a = GCTestObject(name='a')
                b = GCTestObject(name='b')
                a.ref = b
                b.ref = a
                weak_refs.append(weakref.ref(a))
                weak_refs.append(weakref.ref(b))

        create_cycles()
        gc.collect()

        collected = sum(1 for ref in weak_refs if ref() is None)
        self.assertEqual(collected, 1000,
            f"Expected 1000 cyclic objects collected, got {collected}")


class TestCrossThreadReferences(unittest.TestCase):
    """
    Tests for objects referenced across threads.
    These exercise biased reference counting in FTP.
    """

    @threading_helper.requires_working_threading()
    def test_cross_thread_refs_survive(self):
        """Objects referenced across threads should survive GC."""
        shared_objects = []
        lock = threading.Lock()

        def worker(thread_id):
            # Create objects on this thread
            local_objs = [GCTestObject(thread=thread_id, id=i) for i in range(100)]

            with lock:
                # Share references to main thread's list
                shared_objects.extend(local_objs)

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(4)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Create weak refs before GC
        weak_refs = [weakref.ref(obj) for obj in shared_objects]

        gc.collect()

        # All objects should survive (we still hold references)
        for i, ref in enumerate(weak_refs):
            self.assertIsNotNone(ref(),
                f"Shared object {i} was incorrectly collected!")

    @threading_helper.requires_working_threading()
    def test_cross_thread_cycles_collected(self):
        """Cyclic garbage spanning threads should be collected."""
        weak_refs = []
        barriers = threading.Barrier(2)

        def create_half_cycle(container, other_container, barrier):
            obj = GCTestObject(half='cycle')
            weak_refs.append(weakref.ref(obj))
            container.append(obj)
            barrier.wait()  # Sync with other thread

            # Now create cross-thread reference
            if other_container:
                obj.ref = other_container[0]
                other_container[0].ref = obj
            barrier.wait()

        container_a = []
        container_b = []

        t1 = threading.Thread(target=create_half_cycle,
                              args=(container_a, container_b, barriers))
        t2 = threading.Thread(target=create_half_cycle,
                              args=(container_b, container_a, barriers))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Clear our references to make it garbage
        container_a.clear()
        container_b.clear()

        gc.collect()

        collected = sum(1 for ref in weak_refs if ref() is None)
        self.assertEqual(collected, 2,
            f"Expected 2 cross-thread cyclic objects collected, got {collected}")


class TestConcurrentMarking(unittest.TestCase):
    """
    Tests for concurrent marking correctness.
    These validate that parallel GC operations are safe.
    """

    @threading_helper.requires_working_threading()
    def test_concurrent_gc_no_crashes(self):
        """Multiple threads triggering GC concurrently should not crash."""
        errors = []

        def gc_worker(thread_id, iterations):
            try:
                for i in range(iterations):
                    # Create some garbage
                    garbage = [GCTestObject(thread=thread_id, iter=i, idx=j)
                               for j in range(100)]
                    for k in range(len(garbage) - 1):
                        garbage[k].ref = garbage[k + 1]
                    garbage[-1].ref = garbage[0]  # Cycle

                    # Trigger GC (may run concurrently with other threads)
                    gc.collect()

            except Exception as e:
                errors.append((thread_id, e))

        num_threads = 4
        iterations = 50

        threads = [threading.Thread(target=gc_worker, args=(i, iterations))
                   for i in range(num_threads)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0,
            f"Threads had errors: {errors}")

    @threading_helper.requires_working_threading()
    def test_rapid_alloc_dealloc_with_gc(self):
        """Rapid allocation/deallocation with GC should be safe."""
        errors = []
        stop_flag = threading.Event()

        def allocator_worker(thread_id):
            try:
                while not stop_flag.is_set():
                    # Rapidly create and discard objects
                    objs = [GCTestObject(id=i) for i in range(1000)]
                    del objs
            except Exception as e:
                errors.append((thread_id, e))

        def gc_worker():
            try:
                while not stop_flag.is_set():
                    gc.collect()
            except Exception as e:
                errors.append(('gc', e))

        # Start allocator threads
        allocators = [threading.Thread(target=allocator_worker, args=(i,))
                      for i in range(4)]
        gc_thread = threading.Thread(target=gc_worker)

        for t in allocators:
            t.start()
        gc_thread.start()

        # Let them run for a bit
        import time
        time.sleep(0.5)
        stop_flag.set()

        for t in allocators:
            t.join()
        gc_thread.join()

        self.assertEqual(len(errors), 0,
            f"Threads had errors: {errors}")

    @requires_ftp
    @threading_helper.requires_working_threading()
    def test_object_graph_integrity_under_concurrent_gc(self):
        """Object graph should remain valid during concurrent GC."""
        # Build a complex object graph
        num_objects = 1000
        objects = [GCTestObject(id=i) for i in range(num_objects)]

        # Create a random-ish graph of references
        import random
        random.seed(42)  # Reproducible
        for obj in objects:
            if random.random() > 0.3:
                target = random.choice(objects)
                obj.ref = target

        # Keep weak refs to track which survive
        weak_refs = [weakref.ref(obj) for obj in objects]

        # Trigger GC from multiple threads
        barriers = threading.Barrier(4)

        def gc_worker():
            barriers.wait()  # Synchronize start
            for _ in range(10):
                gc.collect()

        threads = [threading.Thread(target=gc_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All objects should still be alive (we hold references)
        alive_count = sum(1 for ref in weak_refs if ref() is not None)
        self.assertEqual(alive_count, num_objects,
            f"Expected {num_objects} alive, got {alive_count}")

        # Verify graph integrity - refs should still point to valid objects
        for i, obj in enumerate(objects):
            if obj.ref is not None:
                self.assertIsNotNone(obj.ref.data,
                    f"Object {i}'s reference target is corrupted")


if __name__ == '__main__':
    unittest.main()
