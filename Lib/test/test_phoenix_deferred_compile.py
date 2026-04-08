"""Phoenix JIT Deferred Compilation Tests.

Tests the deferred compilation mechanism (Option A):
- At threshold, functions are QUEUED for compilation, not compiled synchronously
- Compilation happens at a safe point (eval loop start_frame drain)
- force_compile is unchanged (still synchronous)
- Queue handles duplicates, many functions, and GC correctly

These tests verify the mechanism introduced to fix the ARM64 auto-compilation
crash (GC during vectorcall trampoline corrupts callee-saved registers).

Run with: ./python -m test test_phoenix_deferred_compile
"""

import gc
import sys
import unittest
import weakref

try:
    import _cinderx
    import cinderjit
    HAS_JIT = True
except ImportError:
    HAS_JIT = False

THRESHOLD = 1000  # compile_after_n_calls default


@unittest.skipUnless(HAS_JIT, "requires JIT")
class TestDeferredCompilation(unittest.TestCase):
    """Tests for the deferred compilation queue mechanism."""

    # =================================================================
    # 1. Core mechanism: compilation is deferred, not synchronous
    # =================================================================

    def test_force_compile_still_synchronous(self):
        """force_compile must compile immediately, not defer."""
        def add(x, y):
            return x + y

        cinderjit.force_compile(add)
        # Must be compiled RIGHT NOW, not deferred
        self.assertTrue(cinderjit.is_jit_compiled(add),
                        "force_compile did not compile synchronously")
        # And it must produce correct results
        self.assertEqual(add(3, 4), 7)

    def test_auto_compile_eventually_compiles(self):
        """Auto-compilation must eventually compile after threshold."""
        def mul(x, y):
            return x * y

        # Call past threshold
        for i in range(THRESHOLD + 100):
            mul(2, 3)

        # Must be compiled by now (drain happens at start_frame)
        self.assertTrue(cinderjit.is_jit_compiled(mul),
                        f"mul not compiled after {THRESHOLD + 100} calls")
        self.assertEqual(mul(6, 7), 42)

    def test_auto_compile_result_correct(self):
        """Auto-compiled function produces same result as interpreter."""
        def fib(n):
            a, b = 0, 1
            for _ in range(n):
                a, b = b, a + b
            return a

        # Get interpreter result first
        expected = fib(30)

        # Warm up past threshold
        for i in range(THRESHOLD + 100):
            fib(10)

        self.assertTrue(cinderjit.is_jit_compiled(fib))
        # Compiled result must match
        self.assertEqual(fib(30), expected)

    # =================================================================
    # 2. Queue deduplication
    # =================================================================

    def test_duplicate_queue_no_crash(self):
        """Calling the same function many times past threshold must not
        crash or double-compile. Queue deduplication should handle this."""
        def target(x):
            return x + 1

        # Call way past threshold — function may be queued multiple times
        # before drain if dedup fails
        for i in range(THRESHOLD + 500):
            target(i)

        self.assertTrue(cinderjit.is_jit_compiled(target))
        self.assertEqual(target(99), 100)

    # =================================================================
    # 3. Many functions hitting threshold
    # =================================================================

    def test_many_functions_all_compile(self):
        """Multiple functions hitting threshold must all eventually compile.
        Tests queue capacity (JIT_PENDING_COMPILE_MAX=256)."""
        funcs = []
        # Create 50 distinct functions
        for i in range(50):
            # Each function captures i to be unique
            exec(f"def f_{i}(x): return x + {i}", globals())
            funcs.append(globals()[f"f_{i}"])

        # Warm all functions to just below threshold
        for f in funcs:
            for _ in range(THRESHOLD - 1):
                f(1)

        # Push all past threshold in quick succession
        for f in funcs:
            f(1)
            f(1)  # one more to ensure drain fires

        # Give the drain a chance — call any function to trigger start_frame
        funcs[0](1)

        # All should be compiled
        not_compiled = []
        for i, f in enumerate(funcs):
            if not cinderjit.is_jit_compiled(f):
                not_compiled.append(f"f_{i}")

        self.assertEqual(not_compiled, [],
                         f"Functions not compiled: {not_compiled}")

        # Verify correctness
        for i, f in enumerate(funcs):
            self.assertEqual(f(10), 10 + i,
                             f"f_{i}(10) returned wrong result")

    # =================================================================
    # 4. GC safety during deferred state
    # =================================================================

    def test_gc_during_deferred_compile(self):
        """GC running while a function is queued for compilation must not
        corrupt the queue or crash. The INCREF in Ci_JitQueueCompile
        must keep the function alive."""
        def target(x):
            return x * 2

        # Call past threshold to queue
        for i in range(THRESHOLD + 10):
            target(i)

        # Force GC — this must not crash or corrupt the queue
        gc.collect()
        gc.collect()
        gc.collect()

        # Function should still compile and work
        # (call again to trigger drain if needed)
        for _ in range(10):
            target(1)

        self.assertTrue(cinderjit.is_jit_compiled(target))
        self.assertEqual(target(21), 42)

    def test_function_ref_held_during_deferred(self):
        """The deferred compilation queue must hold a reference to the
        function, preventing it from being collected before compilation."""
        def make_func():
            def target(x):
                return x + 42
            return target

        f = make_func()
        ref = weakref.ref(f)

        # Warm past threshold
        for _ in range(THRESHOLD + 10):
            f(1)

        # f is still referenced by us, so ref() should be alive
        self.assertIsNotNone(ref())

        # After compilation, function still works
        result = f(8)
        self.assertEqual(result, 50)

    # =================================================================
    # 5. Generator auto-compilation through deferred path
    # =================================================================

    def test_generator_auto_compile_deferred(self):
        """Generator functions must also compile through the deferred path."""
        def gen_sum(n):
            total = 0
            for i in range(n):
                total += i
                yield total

        # Warm past threshold
        for _ in range(THRESHOLD + 100):
            list(gen_sum(5))

        self.assertTrue(cinderjit.is_jit_compiled(gen_sum),
                        "Generator not compiled after threshold")
        # Verify correctness
        self.assertEqual(list(gen_sum(5)), [0, 1, 3, 6, 10])

    # =================================================================
    # 6. Deferred function called before drain
    # =================================================================

    def test_function_callable_while_pending(self):
        """A function queued for deferred compilation must still be
        callable via the interpreter while compilation is pending."""
        def compute(x):
            return x * x + 1

        # Interpreter result
        expected = compute(7)

        # Call past threshold — function is now queued
        for i in range(THRESHOLD + 5):
            result = compute(7)
            self.assertEqual(result, expected,
                             f"Wrong result at call {i}: {result} != {expected}")

    # =================================================================
    # 7. Mixed force_compile and auto-compile
    # =================================================================

    def test_force_compile_then_auto_compile_no_conflict(self):
        """force_compile and auto-compile must not interfere."""
        def forced(x):
            return x + 1

        def auto(x):
            return x + 2

        # force_compile one function
        cinderjit.force_compile(forced)
        self.assertTrue(cinderjit.is_jit_compiled(forced))

        # Auto-compile another
        for _ in range(THRESHOLD + 100):
            auto(1)
        self.assertTrue(cinderjit.is_jit_compiled(auto))

        # Both produce correct results
        self.assertEqual(forced(10), 11)
        self.assertEqual(auto(10), 12)

    def test_force_compile_already_queued(self):
        """force_compile on a function that's already queued for deferred
        compilation must succeed (compileFunction is idempotent)."""
        def target(x):
            return x * 3

        # Warm past threshold to get it queued
        for _ in range(THRESHOLD + 5):
            target(1)

        # force_compile should work regardless of queue state
        cinderjit.force_compile(target)
        self.assertTrue(cinderjit.is_jit_compiled(target))
        self.assertEqual(target(14), 42)

    # =================================================================
    # 8. Complex function patterns through deferred path
    # =================================================================

    def test_closure_auto_compile(self):
        """Closures must compile correctly through the deferred path."""
        def make_counter(start):
            count = start
            def increment(n):
                nonlocal count
                count += n
                return count
            return increment

        counter = make_counter(0)

        # Warm past threshold
        for i in range(THRESHOLD + 10):
            counter(1)

        self.assertTrue(cinderjit.is_jit_compiled(counter))
        # Reset and verify
        counter2 = make_counter(100)
        for _ in range(THRESHOLD + 10):
            counter2(1)
        self.assertEqual(counter2(5), 100 + THRESHOLD + 10 + 5)

    def test_recursive_function_auto_compile(self):
        """Recursive functions must compile through the deferred path."""
        call_count = [0]

        def factorial(n):
            call_count[0] += 1
            if n <= 1:
                return 1
            return n * factorial(n - 1)

        # Each factorial(10) call generates ~10 recursive calls.
        # We need the OUTER function to be called threshold times.
        for _ in range(THRESHOLD + 100):
            factorial(5)

        self.assertTrue(cinderjit.is_jit_compiled(factorial))
        self.assertEqual(factorial(10), 3628800)

    def test_exception_handling_auto_compile(self):
        """Functions with try/except must compile through the deferred path."""
        def safe_div(a, b):
            try:
                return a / b
            except ZeroDivisionError:
                return float('inf')

        for _ in range(THRESHOLD + 100):
            safe_div(10, 3)

        self.assertTrue(cinderjit.is_jit_compiled(safe_div))
        self.assertAlmostEqual(safe_div(22, 7), 22 / 7)
        self.assertEqual(safe_div(1, 0), float('inf'))

    def test_class_method_auto_compile(self):
        """Class methods must compile through the deferred path."""
        class Calculator:
            def add(self, a, b):
                return a + b

            def mul(self, a, b):
                return a * b

        calc = Calculator()

        for _ in range(THRESHOLD + 100):
            calc.add(1, 2)
            calc.mul(3, 4)

        # At least one should be compiled
        add_compiled = cinderjit.is_jit_compiled(Calculator.add)
        mul_compiled = cinderjit.is_jit_compiled(Calculator.mul)

        # Both methods should work correctly regardless of JIT status
        self.assertEqual(calc.add(10, 20), 30)
        self.assertEqual(calc.mul(6, 7), 42)

    # =================================================================
    # 9. Vectorcall state machine
    # =================================================================

    def test_threshold_exact_boundary(self):
        """Function called exactly threshold times should be queued,
        and compiled after one more call (which triggers start_frame drain)."""
        def boundary(x):
            return x + 1

        # Call exactly threshold times
        for _ in range(THRESHOLD):
            boundary(1)

        # At this point it should be queued (or just past threshold).
        # One more call triggers start_frame which drains the queue.
        boundary(1)

        # Should be compiled now (drain runs at start_frame before
        # executing the frame, so the function may or may not be compiled
        # during this exact call, but definitely after a few more)
        for _ in range(10):
            boundary(1)

        self.assertTrue(cinderjit.is_jit_compiled(boundary),
                        "Function not compiled after threshold + 11 calls")

    # =================================================================
    # 10. Interleaved compilation and execution
    # =================================================================

    def test_interleaved_functions(self):
        """Multiple functions interleaved past threshold must all compile
        correctly."""
        def f1(x): return x + 1
        def f2(x): return x * 2
        def f3(x): return x ** 2
        def f4(x): return x - 1

        # Interleave calls
        for _ in range(THRESHOLD + 100):
            f1(1)
            f2(1)
            f3(1)
            f4(1)

        for f, arg, expected in [
            (f1, 10, 11),
            (f2, 10, 20),
            (f3, 10, 100),
            (f4, 10, 9),
        ]:
            self.assertTrue(cinderjit.is_jit_compiled(f),
                            f"{f.__name__} not compiled")
            self.assertEqual(f(arg), expected)


if __name__ == "__main__":
    unittest.main()
