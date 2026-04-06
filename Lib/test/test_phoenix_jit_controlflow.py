"""Phoenix JIT Control Flow Correctness Tests.

Verifies that JIT-compiled control flow operations produce identical
results to the interpreter. Each test:
1. Defines a Python function exercising specific control flow opcode(s)
2. Runs it under the interpreter to get the expected result
3. Force-compiles the function via cinderjit.force_compile()
4. Runs under JIT and asserts the result matches

Covers: if/elif/else, for, while, break, continue, for_iter, get_iter,
nested loops, loop-else, try/except/else/finally, raise, exception chaining,
with statement, assert, ternary, short-circuit, chained comparisons,
match statement, return paths, recursion, StopIteration, and more.

Run with: ./python -m test test_phoenix_jit_controlflow
"""

import contextlib
import io
import sys
import unittest

try:
    import _cinderx
    import cinderjit
    HAS_JIT = True
except ImportError:
    HAS_JIT = False


@unittest.skipUnless(HAS_JIT, "Phoenix JIT not available")
class TestJitControlFlow(unittest.TestCase):
    """Test control flow patterns under JIT compilation."""

    def _jit_test(self, func, *args):
        """Run func under interpreter, force-compile, run under JIT, compare."""
        interp_result = func(*args)
        cinderjit.force_compile(func)
        self.assertTrue(
            cinderjit.is_jit_compiled(func),
            f"{func.__name__} did not compile"
        )
        jit_result = func(*args)
        self.assertEqual(interp_result, jit_result,
                         f"{func.__name__}{args}: "
                         f"interpreter={interp_result!r}, JIT={jit_result!r}")
        return jit_result

    # ================================================================
    # 1. IF / ELIF / ELSE
    # ================================================================

    def test_if_simple_true(self):
        def f(x):
            if x > 0:
                return "positive"
            return "non-positive"
        self._jit_test(f, 5)

    def test_if_simple_false(self):
        def f(x):
            if x > 0:
                return "positive"
            return "non-positive"
        self._jit_test(f, -3)

    def test_if_else(self):
        def f(x):
            if x % 2 == 0:
                return "even"
            else:
                return "odd"
        self._jit_test(f, 7)
        self._jit_test(f, 8)

    def test_if_elif_else(self):
        def f(x):
            if x < 0:
                return "negative"
            elif x == 0:
                return "zero"
            elif x < 10:
                return "small"
            elif x < 100:
                return "medium"
            else:
                return "large"
        for v in [-5, 0, 3, 50, 200]:
            self._jit_test(f, v)

    def test_if_nested(self):
        def f(x, y):
            if x > 0:
                if y > 0:
                    return "both positive"
                else:
                    return "x positive, y not"
            else:
                if y > 0:
                    return "y positive, x not"
                else:
                    return "both non-positive"
        for x, y in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            self._jit_test(f, x, y)

    def test_if_chained_conditions(self):
        def f(a, b, c):
            if a and b and c:
                return 1
            elif a or b:
                return 2
            elif not c:
                return 3
            else:
                return 4
        for vals in [(True, True, True), (True, False, False),
                     (False, False, False), (False, False, True)]:
            self._jit_test(f, *vals)

    # ================================================================
    # 2. FOR LOOP
    # ================================================================

    def test_for_range(self):
        def f(n):
            total = 0
            for i in range(n):
                total += i
            return total
        self._jit_test(f, 100)

    def test_for_range_step(self):
        def f(start, stop, step):
            result = []
            for i in range(start, stop, step):
                result.append(i)
            return result
        self._jit_test(f, 0, 20, 3)
        self._jit_test(f, 10, 0, -2)

    def test_for_list_iteration(self):
        def f(lst):
            total = 0
            for x in lst:
                total += x * x
            return total
        self._jit_test(f, [1, 2, 3, 4, 5])

    def test_for_dict_iteration(self):
        def f(d):
            keys = []
            vals = []
            for k in d:
                keys.append(k)
                vals.append(d[k])
            return (sorted(keys), sorted(vals))
        self._jit_test(f, {"a": 1, "b": 2, "c": 3})

    def test_for_dict_items(self):
        def f(d):
            result = []
            for k, v in d.items():
                result.append((k, v))
            return sorted(result)
        self._jit_test(f, {"x": 10, "y": 20, "z": 30})

    def test_for_enumerate(self):
        def f(lst):
            result = []
            for i, val in enumerate(lst):
                result.append(i * val)
            return result
        self._jit_test(f, [10, 20, 30, 40])

    def test_for_zip(self):
        def f(a, b):
            result = []
            for x, y in zip(a, b):
                result.append(x + y)
            return result
        self._jit_test(f, [1, 2, 3], [10, 20, 30])

    def test_for_zip_unequal(self):
        def f(a, b):
            result = []
            for x, y in zip(a, b):
                result.append(x + y)
            return result
        self._jit_test(f, [1, 2, 3, 4], [10, 20])

    def test_for_string_iteration(self):
        def f(s):
            chars = []
            for c in s:
                chars.append(c.upper())
            return "".join(chars)
        self._jit_test(f, "hello world")

    def test_for_tuple_unpacking(self):
        def f(pairs):
            total = 0
            for a, b in pairs:
                total += a * b
            return total
        self._jit_test(f, [(1, 2), (3, 4), (5, 6)])

    def test_for_empty(self):
        def f():
            result = 0
            for i in range(0):
                result += 1
            return result
        self._jit_test(f)

    # ================================================================
    # 3. WHILE LOOP
    # ================================================================

    def test_while_simple(self):
        def f(n):
            total = 0
            i = 0
            while i < n:
                total += i
                i += 1
            return total
        self._jit_test(f, 50)

    def test_while_with_break(self):
        def f(lst):
            i = 0
            while i < len(lst):
                if lst[i] < 0:
                    break
                i += 1
            return i
        self._jit_test(f, [1, 2, 3, -1, 5])
        self._jit_test(f, [1, 2, 3])

    def test_while_with_continue(self):
        def f(n):
            total = 0
            i = 0
            while i < n:
                i += 1
                if i % 3 == 0:
                    continue
                total += i
            return total
        self._jit_test(f, 20)

    def test_while_false(self):
        def f():
            result = "never entered"
            while False:
                result = "entered"
            return result
        self._jit_test(f)

    # ================================================================
    # 4. BREAK
    # ================================================================

    def test_break_for(self):
        def f(lst, target):
            for i, val in enumerate(lst):
                if val == target:
                    return i
            return -1
        self._jit_test(f, [10, 20, 30, 40], 30)
        self._jit_test(f, [10, 20, 30, 40], 99)

    def test_break_while(self):
        def f(n):
            i = 0
            while True:
                if i * i > n:
                    break
                i += 1
            return i
        self._jit_test(f, 100)

    def test_break_nested_inner_only(self):
        def f(matrix):
            results = []
            for row in matrix:
                for val in row:
                    if val < 0:
                        break
                    results.append(val)
            return results
        self._jit_test(f, [[1, 2, -3, 4], [5, 6], [7, -8, 9]])

    # ================================================================
    # 5. CONTINUE
    # ================================================================

    def test_continue_for(self):
        def f(lst):
            result = []
            for x in lst:
                if x % 2 == 0:
                    continue
                result.append(x)
            return result
        self._jit_test(f, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    def test_continue_while(self):
        def f(n):
            result = []
            i = 0
            while i < n:
                i += 1
                if i % 5 == 0:
                    continue
                result.append(i)
            return result
        self._jit_test(f, 20)

    # ================================================================
    # 6. FOR_ITER (iteration protocol)
    # ================================================================

    def test_for_iter_generator(self):
        def gen(n):
            for i in range(n):
                yield i * i

        def f(n):
            total = 0
            for val in gen(n):
                total += val
            return total
        self._jit_test(f, 10)

    def test_for_iter_custom_iterator(self):
        class Countdown:
            def __init__(self, start):
                self.current = start
            def __iter__(self):
                return self
            def __next__(self):
                if self.current <= 0:
                    raise StopIteration
                self.current -= 1
                return self.current + 1

        def f(n):
            result = []
            for val in Countdown(n):
                result.append(val)
            return result
        self._jit_test(f, 5)

    def test_for_iter_map_filter(self):
        def f(lst):
            result = list(filter(lambda x: x > 0, map(lambda x: x * 2 - 5, lst)))
            return result
        self._jit_test(f, [1, 2, 3, 4, 5, 6, 7])

    # ================================================================
    # 7. GET_ITER (iter() builtin)
    # ================================================================

    def test_get_iter_explicit(self):
        def f(lst):
            it = iter(lst)
            result = []
            result.append(next(it))
            result.append(next(it))
            result.append(next(it))
            return result
        self._jit_test(f, [10, 20, 30, 40, 50])

    def test_get_iter_next_default(self):
        def f(lst):
            it = iter(lst)
            results = []
            while True:
                val = next(it, None)
                if val is None:
                    break
                results.append(val)
            return results
        self._jit_test(f, [1, 2, 3])

    # ================================================================
    # 8. NESTED LOOPS
    # ================================================================

    def test_nested_loops_2_deep(self):
        def f(n, m):
            result = []
            for i in range(n):
                for j in range(m):
                    result.append(i * m + j)
            return result
        self._jit_test(f, 4, 3)

    def test_nested_loops_3_deep(self):
        def f(a, b, c):
            total = 0
            for i in range(a):
                for j in range(b):
                    for k in range(c):
                        total += i * 100 + j * 10 + k
            return total
        self._jit_test(f, 3, 4, 5)

    def test_nested_for_while(self):
        def f(n):
            result = []
            for i in range(n):
                j = 0
                while j < i:
                    result.append((i, j))
                    j += 1
            return result
        self._jit_test(f, 5)

    # ================================================================
    # 9. LOOP-ELSE
    # ================================================================

    def test_for_else_no_break(self):
        def f(lst, target):
            for x in lst:
                if x == target:
                    return "found"
            else:
                return "not found"
        self._jit_test(f, [1, 2, 3, 4], 5)

    def test_for_else_with_break(self):
        def f(lst, target):
            for x in lst:
                if x == target:
                    return "found"
            else:
                return "not found"
        self._jit_test(f, [1, 2, 3, 4], 3)

    def test_while_else_no_break(self):
        def f(n):
            i = 0
            while i < n:
                i += 1
            else:
                return "completed"
            return "broke out"
        self._jit_test(f, 5)

    def test_while_else_with_break(self):
        def f(lst):
            i = 0
            while i < len(lst):
                if lst[i] < 0:
                    break
                i += 1
            else:
                return "all positive"
            return f"negative at {i}"
        self._jit_test(f, [1, 2, 3])
        self._jit_test(f, [1, -2, 3])

    def test_for_else_empty_iterable(self):
        def f():
            for x in []:
                return "found"
            else:
                return "else on empty"
        self._jit_test(f)

    # ================================================================
    # 10. TRY / EXCEPT
    # ================================================================

    def test_try_except_simple(self):
        def f(x):
            try:
                return 10 // x
            except ZeroDivisionError:
                return -1
        self._jit_test(f, 5)
        self._jit_test(f, 0)

    def test_try_except_multiple(self):
        def f(action):
            try:
                if action == "div":
                    return 1 // 0
                elif action == "key":
                    d = {}
                    return d["missing"]
                elif action == "index":
                    lst = [1, 2]
                    return lst[10]
                else:
                    return "ok"
            except ZeroDivisionError:
                return "zero_div"
            except KeyError:
                return "key_error"
            except IndexError:
                return "index_error"
        for a in ["div", "key", "index", "none"]:
            self._jit_test(f, a)

    def test_try_except_bare(self):
        def f(x):
            try:
                if x == 0:
                    raise RuntimeError("boom")
                return x * 2
            except:
                return -999
        self._jit_test(f, 5)
        self._jit_test(f, 0)

    def test_try_except_as(self):
        def f(x):
            try:
                if x < 0:
                    raise ValueError(f"negative: {x}")
                return x
            except ValueError as e:
                return str(e)
        self._jit_test(f, 5)
        self._jit_test(f, -3)

    def test_try_except_exception_type(self):
        def f():
            try:
                raise TypeError("type mismatch")
            except TypeError as e:
                return type(e).__name__
        self._jit_test(f)

    # ================================================================
    # 11. TRY / EXCEPT / ELSE
    # ================================================================

    def test_try_except_else(self):
        def f(x):
            result = []
            try:
                val = 100 // x
            except ZeroDivisionError:
                result.append("caught")
                val = 0
            else:
                result.append("no error")
            result.append(val)
            return result
        self._jit_test(f, 5)
        self._jit_test(f, 0)

    def test_try_except_else_no_exception(self):
        def f():
            try:
                x = 42
            except Exception:
                return "error"
            else:
                return f"ok: {x}"
        self._jit_test(f)

    # ================================================================
    # 12. TRY / FINALLY
    # ================================================================

    def test_try_finally_no_exception(self):
        def f():
            result = []
            try:
                result.append("try")
            finally:
                result.append("finally")
            return result
        self._jit_test(f)

    def test_try_finally_with_exception(self):
        def f():
            result = []
            try:
                try:
                    result.append("try")
                    raise ValueError("oops")
                finally:
                    result.append("finally")
            except ValueError:
                result.append("caught")
            return result
        self._jit_test(f)

    def test_try_finally_with_return(self):
        def f(do_return):
            result = []
            try:
                result.append("try")
                if do_return:
                    return result + ["early return"]
                result.append("after check")
            finally:
                result.append("finally")
            return result
        self._jit_test(f, True)
        self._jit_test(f, False)

    # ================================================================
    # 13. TRY / EXCEPT / FINALLY
    # ================================================================

    def test_try_except_finally(self):
        def f(x):
            result = []
            try:
                result.append("try")
                if x == 0:
                    raise ValueError("zero")
                result.append(100 // x)
            except ValueError as e:
                result.append(f"caught: {e}")
            finally:
                result.append("finally")
            return result
        self._jit_test(f, 5)
        self._jit_test(f, 0)

    def test_try_except_else_finally(self):
        def f(x):
            result = []
            try:
                val = 10 // x
            except ZeroDivisionError:
                result.append("error")
                val = -1
            else:
                result.append("ok")
            finally:
                result.append("cleanup")
            result.append(val)
            return result
        self._jit_test(f, 2)
        self._jit_test(f, 0)

    # ================================================================
    # 14. NESTED TRY / EXCEPT
    # ================================================================

    def test_nested_try_except(self):
        def f(a, b):
            try:
                try:
                    return a // b
                except ZeroDivisionError:
                    return "inner caught"
            except Exception:
                return "outer caught"
        self._jit_test(f, 10, 3)
        self._jit_test(f, 10, 0)

    def test_nested_try_except_propagation(self):
        def f():
            result = []
            try:
                try:
                    raise ValueError("inner")
                except TypeError:
                    result.append("wrong handler")
            except ValueError as e:
                result.append(f"outer: {e}")
            return result
        self._jit_test(f)

    def test_try_in_loop(self):
        def f(lst):
            results = []
            for x in lst:
                try:
                    results.append(100 // x)
                except ZeroDivisionError:
                    results.append("inf")
            return results
        self._jit_test(f, [5, 0, 10, 0, 2])

    # ================================================================
    # 15. RAISE
    # ================================================================

    def test_raise_and_catch(self):
        def f():
            try:
                raise ValueError("test error")
            except ValueError as e:
                return str(e)
        self._jit_test(f)

    def test_raise_from(self):
        def f():
            try:
                try:
                    raise ValueError("original")
                except ValueError as e:
                    raise RuntimeError("wrapper") from e
            except RuntimeError as e:
                return (str(e), str(e.__cause__))
        self._jit_test(f)

    def test_reraise(self):
        def f():
            try:
                try:
                    raise ValueError("original")
                except ValueError:
                    raise  # re-raise
            except ValueError as e:
                return str(e)
        self._jit_test(f)

    # ================================================================
    # 16. EXCEPTION CHAINING
    # ================================================================

    def test_exception_chaining_implicit(self):
        def f():
            try:
                try:
                    raise ValueError("first")
                except ValueError:
                    raise TypeError("second")
            except TypeError as e:
                return (str(e), type(e.__context__).__name__, str(e.__context__))
        self._jit_test(f)

    def test_exception_chaining_explicit(self):
        def f():
            try:
                try:
                    raise ValueError("cause")
                except ValueError as orig:
                    raise RuntimeError("effect") from orig
            except RuntimeError as e:
                return (str(e), type(e.__cause__).__name__, str(e.__cause__))
        self._jit_test(f)

    def test_exception_chaining_suppress(self):
        def f():
            try:
                try:
                    raise ValueError("suppressed")
                except ValueError:
                    raise RuntimeError("new") from None
            except RuntimeError as e:
                return (str(e), e.__cause__, e.__suppress_context__)
        self._jit_test(f)

    # ================================================================
    # 17. WITH STATEMENT
    # ================================================================

    def test_with_simple(self):
        def f():
            result = []
            class Ctx:
                def __enter__(self):
                    result.append("enter")
                    return self
                def __exit__(self, *args):
                    result.append("exit")
                    return False
            with Ctx():
                result.append("body")
            return result
        self._jit_test(f)

    def test_with_as(self):
        def f():
            result = []
            class Ctx:
                def __enter__(self):
                    result.append("enter")
                    return 42
                def __exit__(self, *args):
                    result.append("exit")
                    return False
            with Ctx() as val:
                result.append(val)
            return result
        self._jit_test(f)

    def test_with_exception_not_suppressed(self):
        def f():
            result = []
            class Ctx:
                def __enter__(self):
                    result.append("enter")
                    return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    result.append(f"exit:{exc_type.__name__}")
                    return False  # don't suppress
            try:
                with Ctx():
                    result.append("body")
                    raise ValueError("boom")
            except ValueError:
                result.append("caught")
            return result
        self._jit_test(f)

    def test_with_exception_suppressed(self):
        def f():
            result = []
            class Ctx:
                def __enter__(self):
                    result.append("enter")
                    return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    result.append(f"exit:{exc_type.__name__}")
                    return True  # suppress
            with Ctx():
                result.append("body")
                raise ValueError("suppressed")
            result.append("after with")
            return result
        self._jit_test(f)

    # ================================================================
    # 18. NESTED WITH
    # ================================================================

    def test_nested_with(self):
        def f():
            result = []
            class Ctx:
                def __init__(self, name):
                    self.name = name
                def __enter__(self):
                    result.append(f"enter:{self.name}")
                    return self
                def __exit__(self, *args):
                    result.append(f"exit:{self.name}")
                    return False
            with Ctx("outer"):
                result.append("outer body")
                with Ctx("inner"):
                    result.append("inner body")
            return result
        self._jit_test(f)

    def test_multiple_with_single_statement(self):
        def f():
            result = []
            class Ctx:
                def __init__(self, name):
                    self.name = name
                def __enter__(self):
                    result.append(f"enter:{self.name}")
                    return self.name
                def __exit__(self, *args):
                    result.append(f"exit:{self.name}")
                    return False
            with Ctx("a") as a, Ctx("b") as b:
                result.append(f"body:{a},{b}")
            return result
        self._jit_test(f)

    # ================================================================
    # 19. ASSERT STATEMENT
    # ================================================================

    def test_assert_true(self):
        def f(x):
            assert x > 0, "must be positive"
            return x * 2
        self._jit_test(f, 5)

    def test_assert_false(self):
        def f(x):
            try:
                assert x > 0, f"must be positive, got {x}"
                return x * 2
            except AssertionError as e:
                return str(e)
        self._jit_test(f, -3)

    def test_assert_no_message(self):
        def f(x):
            try:
                assert x != 0
                return "ok"
            except AssertionError:
                return "assertion error"
        self._jit_test(f, 1)
        self._jit_test(f, 0)

    # ================================================================
    # 20. CONDITIONAL EXPRESSIONS (TERNARY)
    # ================================================================

    def test_ternary_true(self):
        def f(x):
            return "positive" if x > 0 else "non-positive"
        self._jit_test(f, 5)

    def test_ternary_false(self):
        def f(x):
            return "positive" if x > 0 else "non-positive"
        self._jit_test(f, -1)

    def test_ternary_nested(self):
        def f(x):
            return "pos" if x > 0 else ("zero" if x == 0 else "neg")
        self._jit_test(f, 5)
        self._jit_test(f, 0)
        self._jit_test(f, -3)

    def test_ternary_in_expression(self):
        def f(x, y):
            return (x + y) if x > y else (x * y)
        self._jit_test(f, 10, 3)
        self._jit_test(f, 3, 10)

    # ================================================================
    # 21. SHORT-CIRCUIT AND / OR
    # ================================================================

    def test_short_circuit_and(self):
        def f():
            log = []
            def a():
                log.append("a")
                return True
            def b():
                log.append("b")
                return True
            def c():
                log.append("c")
                return False
            result = a() and b() and c()
            return (result, log)
        self._jit_test(f)

    def test_short_circuit_and_early_false(self):
        def f():
            log = []
            def a():
                log.append("a")
                return False
            def b():
                log.append("b")
                return True
            result = a() and b()
            return (result, log)
        self._jit_test(f)

    def test_short_circuit_or(self):
        def f():
            log = []
            def a():
                log.append("a")
                return False
            def b():
                log.append("b")
                return False
            def c():
                log.append("c")
                return True
            result = a() or b() or c()
            return (result, log)
        self._jit_test(f)

    def test_short_circuit_or_early_true(self):
        def f():
            log = []
            def a():
                log.append("a")
                return True
            def b():
                log.append("b")
                return True
            result = a() or b()
            return (result, log)
        self._jit_test(f)

    def test_short_circuit_mixed(self):
        def f(a, b, c):
            return (a or b) and c
        for vals in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                     (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]:
            self._jit_test(f, *vals)

    def test_short_circuit_or_value(self):
        """or returns the first truthy value or last value."""
        def f(a, b, c):
            return a or b or c
        self._jit_test(f, 0, 0, 42)
        self._jit_test(f, 0, 7, 42)
        self._jit_test(f, 3, 7, 42)
        self._jit_test(f, 0, 0, 0)

    def test_short_circuit_and_value(self):
        """and returns the first falsy value or last value."""
        def f(a, b, c):
            return a and b and c
        self._jit_test(f, 1, 2, 3)
        self._jit_test(f, 1, 0, 3)
        self._jit_test(f, 0, 2, 3)

    # ================================================================
    # 22. CHAINED COMPARISONS
    # ================================================================

    def test_chained_lt(self):
        def f(a, b, c):
            return a < b < c
        self._jit_test(f, 1, 2, 3)
        self._jit_test(f, 1, 3, 2)
        self._jit_test(f, 3, 2, 1)

    def test_chained_eq_ne(self):
        def f(a, b, c):
            return a == b != c
        self._jit_test(f, 1, 1, 2)
        self._jit_test(f, 1, 1, 1)
        self._jit_test(f, 1, 2, 2)

    def test_chained_mixed(self):
        def f(a, b, c, d):
            return a <= b < c <= d
        self._jit_test(f, 1, 2, 3, 4)
        self._jit_test(f, 1, 2, 3, 3)
        self._jit_test(f, 1, 2, 2, 3)
        self._jit_test(f, 5, 2, 3, 4)

    def test_chained_comparison_side_effects(self):
        """Middle expression evaluated only once in chained comparison."""
        def f():
            log = []
            def val(x):
                log.append(x)
                return x
            result = val(1) < val(2) < val(3)
            return (result, log)
        self._jit_test(f)

    def test_chained_comparison_short_circuit(self):
        """Second comparison not evaluated if first is False."""
        def f():
            log = []
            def val(x):
                log.append(x)
                return x
            result = val(3) < val(1) < val(2)
            return (result, log)
        self._jit_test(f)

    # ================================================================
    # 23. MATCH STATEMENT (structural pattern matching, Python 3.10+)
    # ================================================================

    def test_match_literal(self):
        def f(command):
            match command:
                case "quit":
                    return 0
                case "hello":
                    return 1
                case "world":
                    return 2
                case _:
                    return -1
        for cmd in ["quit", "hello", "world", "other"]:
            self._jit_test(f, cmd)

    def test_match_capture(self):
        def f(point):
            match point:
                case (0, 0):
                    return "origin"
                case (x, 0):
                    return f"x-axis at {x}"
                case (0, y):
                    return f"y-axis at {y}"
                case (x, y):
                    return f"point at ({x}, {y})"
                case _:
                    return "not a point"
        self._jit_test(f, (0, 0))
        self._jit_test(f, (3, 0))
        self._jit_test(f, (0, 4))
        self._jit_test(f, (3, 4))

    def test_match_guard(self):
        def f(point):
            match point:
                case (x, y) if x == y:
                    return "diagonal"
                case (x, y) if x > y:
                    return "above"
                case (x, y):
                    return "below"
        self._jit_test(f, (3, 3))
        self._jit_test(f, (5, 2))
        self._jit_test(f, (2, 5))

    def test_match_or_pattern(self):
        def f(status):
            match status:
                case 200 | 201 | 204:
                    return "success"
                case 301 | 302:
                    return "redirect"
                case 400 | 404:
                    return "client error"
                case 500:
                    return "server error"
                case _:
                    return "unknown"
        for s in [200, 201, 204, 301, 302, 400, 404, 500, 999]:
            self._jit_test(f, s)

    def test_match_class_pattern(self):
        class Point:
            __match_args__ = ("x", "y")
            def __init__(self, x, y):
                self.x = x
                self.y = y

        def f(obj):
            match obj:
                case Point(x=0, y=0):
                    return "origin"
                case Point(x=x, y=0):
                    return f"x-axis:{x}"
                case Point(x=0, y=y):
                    return f"y-axis:{y}"
                case Point(x=x, y=y):
                    return f"({x},{y})"
                case _:
                    return "not a point"
        self._jit_test(f, Point(0, 0))
        self._jit_test(f, Point(3, 0))
        self._jit_test(f, Point(0, 5))
        self._jit_test(f, Point(3, 4))

    def test_match_sequence(self):
        def f(seq):
            match seq:
                case []:
                    return "empty"
                case [x]:
                    return f"single:{x}"
                case [x, y]:
                    return f"pair:{x},{y}"
                case [x, *rest]:
                    return f"head:{x},rest:{rest}"
        self._jit_test(f, [])
        self._jit_test(f, [1])
        self._jit_test(f, [1, 2])
        self._jit_test(f, [1, 2, 3, 4])

    def test_match_mapping(self):
        def f(action):
            match action:
                case {"type": "move", "x": x, "y": y}:
                    return f"move to ({x},{y})"
                case {"type": "click", "button": b}:
                    return f"click {b}"
                case {"type": t}:
                    return f"unknown action: {t}"
                case _:
                    return "not an action"
        self._jit_test(f, {"type": "move", "x": 10, "y": 20})
        self._jit_test(f, {"type": "click", "button": "left"})
        self._jit_test(f, {"type": "scroll"})
        self._jit_test(f, "not a dict")

    # ================================================================
    # 24. RETURN FROM NESTED SCOPES
    # ================================================================

    def test_return_from_nested_if(self):
        def f(x):
            if x > 100:
                if x > 200:
                    return "very large"
                return "large"
            if x > 50:
                return "medium"
            return "small"
        for v in [10, 60, 150, 250]:
            self._jit_test(f, v)

    def test_return_from_loop(self):
        def f(lst, target):
            for i, val in enumerate(lst):
                if val == target:
                    return i
            return -1
        self._jit_test(f, [10, 20, 30, 40], 30)
        self._jit_test(f, [10, 20, 30, 40], 99)

    def test_return_from_try(self):
        def f(x):
            try:
                if x == 0:
                    raise ValueError()
                return "from try"
            except ValueError:
                return "from except"
        self._jit_test(f, 1)
        self._jit_test(f, 0)

    # ================================================================
    # 25. MULTIPLE RETURN PATHS
    # ================================================================

    def test_multiple_return_paths(self):
        def f(x, y, z):
            if x < 0:
                return "path A"
            if y > 100:
                for i in range(z):
                    if i > 5:
                        return "path B"
                return "path C"
            try:
                result = x // y
                if result > 10:
                    return "path D"
            except ZeroDivisionError:
                return "path E"
            return "path F"
        test_cases = [
            (-1, 0, 0),    # path A
            (1, 200, 10),  # path B
            (1, 200, 3),   # path C
            (200, 5, 0),   # path D
            (1, 0, 0),     # path E
            (1, 5, 0),     # path F
        ]
        for args in test_cases:
            self._jit_test(f, *args)

    def test_multiple_returns_with_finally(self):
        def f(x):
            cleanup = []
            try:
                if x == 1:
                    return ("early", cleanup)
                if x == 2:
                    raise ValueError()
                return ("normal", cleanup)
            except ValueError:
                return ("error", cleanup)
            finally:
                cleanup.append("cleaned")
        self._jit_test(f, 1)
        self._jit_test(f, 2)
        self._jit_test(f, 3)

    # ================================================================
    # 26. RECURSIVE CALLS
    # ================================================================

    def test_direct_recursion(self):
        def factorial(n):
            if n <= 1:
                return 1
            return n * factorial(n - 1)
        self._jit_test(factorial, 10)

    def test_direct_recursion_fibonacci(self):
        def fib(n):
            if n < 2:
                return n
            return fib(n - 1) + fib(n - 2)
        self._jit_test(fib, 15)

    def test_mutual_recursion(self):
        def is_even(n):
            if n == 0:
                return True
            return is_odd(n - 1)

        def is_odd(n):
            if n == 0:
                return False
            return is_even(n - 1)

        # Compile both
        interp_even = is_even(7)
        interp_odd = is_odd(7)
        cinderjit.force_compile(is_even)
        cinderjit.force_compile(is_odd)
        self.assertTrue(cinderjit.is_jit_compiled(is_even))
        self.assertTrue(cinderjit.is_jit_compiled(is_odd))
        self.assertEqual(interp_even, is_even(7))
        self.assertEqual(interp_odd, is_odd(7))

    def test_recursion_with_accumulator(self):
        def sum_list(lst, acc=0):
            if not lst:
                return acc
            return sum_list(lst[1:], acc + lst[0])
        self._jit_test(sum_list, [1, 2, 3, 4, 5])

    # ================================================================
    # 27. WHILE TRUE WITH BREAK
    # ================================================================

    def test_while_true_break(self):
        def f(n):
            total = 0
            i = 0
            while True:
                if i >= n:
                    break
                total += i
                i += 1
            return total
        self._jit_test(f, 10)

    def test_while_true_multiple_breaks(self):
        def f(lst):
            i = 0
            while True:
                if i >= len(lst):
                    return "end"
                if lst[i] < 0:
                    return "negative"
                if lst[i] == 0:
                    return "zero"
                i += 1
        self._jit_test(f, [1, 2, 3])
        self._jit_test(f, [1, -2, 3])
        self._jit_test(f, [1, 0, 3])

    def test_while_true_with_try(self):
        def f(actions):
            results = []
            idx = 0
            while True:
                if idx >= len(actions):
                    break
                action = actions[idx]
                try:
                    if action == "error":
                        raise ValueError("oops")
                    results.append(action)
                except ValueError:
                    results.append("handled")
                idx += 1
            return results
        self._jit_test(f, ["a", "error", "b", "error", "c"])

    # ================================================================
    # 28. STOPITERATION HANDLING IN FOR LOOPS
    # ================================================================

    def test_stopiteration_in_for(self):
        """For loop correctly handles StopIteration from __next__."""
        class LimitedIter:
            def __init__(self, limit):
                self.limit = limit
                self.current = 0
            def __iter__(self):
                return self
            def __next__(self):
                if self.current >= self.limit:
                    raise StopIteration
                self.current += 1
                return self.current

        def f(n):
            result = []
            for x in LimitedIter(n):
                result.append(x)
            return result
        self._jit_test(f, 5)
        self._jit_test(f, 0)

    def test_stopiteration_generator(self):
        def gen(n):
            for i in range(n):
                yield i * 2

        def f(n):
            result = []
            for x in gen(n):
                result.append(x)
            return result
        self._jit_test(f, 5)

    def test_manual_next_stopiteration(self):
        def f(lst):
            it = iter(lst)
            result = []
            while True:
                try:
                    result.append(next(it))
                except StopIteration:
                    break
            return result
        self._jit_test(f, [10, 20, 30])
        self._jit_test(f, [])

    # ================================================================
    # 29. EXCEPTION IN LOOP BODY WITH CONTINUE
    # ================================================================

    def test_exception_in_loop_continue(self):
        def f(lst):
            results = []
            for x in lst:
                try:
                    if x == 0:
                        raise ValueError("skip zero")
                    results.append(100 // x)
                except (ValueError, ZeroDivisionError):
                    results.append("skipped")
                    continue
                results.append("processed")
            return results
        self._jit_test(f, [5, 0, 10, 0, 2])

    def test_exception_in_while_continue(self):
        def f(n):
            results = []
            i = 0
            while i < n:
                i += 1
                try:
                    if i % 3 == 0:
                        raise ValueError()
                    results.append(i)
                except ValueError:
                    results.append(f"skip:{i}")
                    continue
            return results
        self._jit_test(f, 10)

    # ================================================================
    # ADDITIONAL TESTS — complex/edge cases
    # ================================================================

    def test_complex_control_flow_fizzbuzz(self):
        def fizzbuzz(n):
            result = []
            for i in range(1, n + 1):
                if i % 15 == 0:
                    result.append("FizzBuzz")
                elif i % 3 == 0:
                    result.append("Fizz")
                elif i % 5 == 0:
                    result.append("Buzz")
                else:
                    result.append(i)
            return result
        self._jit_test(fizzbuzz, 30)

    def test_binary_search(self):
        def binary_search(lst, target):
            lo, hi = 0, len(lst) - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if lst[mid] == target:
                    return mid
                elif lst[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid - 1
            return -1
        data = list(range(0, 100, 2))
        self._jit_test(binary_search, data, 42)
        self._jit_test(binary_search, data, 43)
        self._jit_test(binary_search, data, 0)
        self._jit_test(binary_search, data, 98)

    def test_quicksort(self):
        def quicksort(lst):
            if len(lst) <= 1:
                return lst
            pivot = lst[len(lst) // 2]
            left = [x for x in lst if x < pivot]
            middle = [x for x in lst if x == pivot]
            right = [x for x in lst if x > pivot]
            return quicksort(left) + middle + quicksort(right)
        self._jit_test(quicksort, [3, 6, 8, 10, 1, 2, 1])
        self._jit_test(quicksort, [])
        self._jit_test(quicksort, [1])
        self._jit_test(quicksort, [5, 5, 5])

    def test_exception_in_with(self):
        def f():
            result = []
            class Logger:
                def __init__(self, name):
                    self.name = name
                def __enter__(self):
                    result.append(f"enter:{self.name}")
                    return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    if exc_type:
                        result.append(f"exit:{self.name}:err:{exc_type.__name__}")
                    else:
                        result.append(f"exit:{self.name}:ok")
                    return False
            try:
                with Logger("outer"):
                    with Logger("inner"):
                        result.append("body")
                        raise RuntimeError("boom")
            except RuntimeError:
                result.append("caught")
            return result
        self._jit_test(f)

    def test_generator_send(self):
        def accumulator():
            total = 0
            while True:
                val = yield total
                if val is None:
                    break
                total += val

        def f():
            gen = accumulator()
            next(gen)  # prime
            results = []
            for v in [10, 20, 30]:
                results.append(gen.send(v))
            return results
        self._jit_test(f)

    def test_list_comprehension_with_condition(self):
        def f(n):
            return [x * x for x in range(n) if x % 2 == 0]
        self._jit_test(f, 20)

    def test_nested_comprehension(self):
        def f(n):
            return [(i, j) for i in range(n) for j in range(i)]
        self._jit_test(f, 5)

    def test_dict_comprehension(self):
        def f(keys, vals):
            return {k: v for k, v in zip(keys, vals) if v > 0}
        self._jit_test(f, ["a", "b", "c", "d"], [1, -2, 3, -4])

    def test_set_comprehension(self):
        def f(lst):
            return sorted(list({x % 5 for x in lst}))
        self._jit_test(f, [1, 2, 3, 6, 7, 8, 11, 12, 13])

    def test_walrus_operator_in_while(self):
        def f(data):
            result = []
            it = iter(data)
            while (val := next(it, None)) is not None:
                result.append(val * 2)
            return result
        self._jit_test(f, [1, 2, 3, 4, 5])

    def test_walrus_operator_in_if(self):
        def f(lst):
            result = []
            for x in lst:
                if (doubled := x * 2) > 5:
                    result.append(doubled)
            return result
        self._jit_test(f, [1, 2, 3, 4, 5])

    def test_complex_loop_break_continue(self):
        """Break from inner, continue in outer."""
        def f(matrix):
            result = []
            for row in matrix:
                row_sum = 0
                found_negative = False
                for val in row:
                    if val < 0:
                        found_negative = True
                        break
                    row_sum += val
                if found_negative:
                    continue
                result.append(row_sum)
            return result
        self._jit_test(f, [[1, 2, 3], [4, -5, 6], [7, 8, 9]])

    def test_exception_reraise_chain(self):
        def f():
            errors = []
            for i in range(5):
                try:
                    if i == 2:
                        raise ValueError(f"err at {i}")
                    if i == 4:
                        raise TypeError(f"type err at {i}")
                    errors.append(f"ok:{i}")
                except ValueError as e:
                    errors.append(f"val:{e}")
                except TypeError as e:
                    errors.append(f"type:{e}")
            return errors
        self._jit_test(f)

    def test_deeply_nested_control_flow(self):
        def f(n):
            result = 0
            for i in range(n):
                if i % 2 == 0:
                    for j in range(i):
                        if j % 3 == 0:
                            try:
                                if j == 0:
                                    continue
                                result += i // j
                            except ZeroDivisionError:
                                result -= 1
                        else:
                            while result > 100:
                                result -= 50
                                if result < 75:
                                    break
                            result += 1
            return result
        self._jit_test(f, 15)

    def test_for_unpacking_star(self):
        def f(data):
            results = []
            for first, *rest in data:
                results.append((first, len(rest), sum(rest)))
            return results
        self._jit_test(f, [[1, 2, 3], [4, 5], [6, 7, 8, 9]])

    def test_contextmanager_decorator(self):
        def f():
            result = []

            @contextlib.contextmanager
            def managed(name):
                result.append(f"enter:{name}")
                try:
                    yield name.upper()
                finally:
                    result.append(f"exit:{name}")

            with managed("test") as val:
                result.append(f"body:{val}")
            return result
        self._jit_test(f)

    def test_try_except_in_comprehension(self):
        def safe_div(a, b):
            try:
                return a // b
            except ZeroDivisionError:
                return -1

        def f(pairs):
            return [safe_div(a, b) for a, b in pairs]
        self._jit_test(f, [(10, 2), (7, 0), (9, 3), (5, 0)])

    def test_multiple_exceptions_in_loop(self):
        def f():
            errors = []
            ops = [
                lambda: 1 / 0,
                lambda: [][5],
                lambda: {}["missing"],
                lambda: int("abc"),
                lambda: 42,
            ]
            for op in ops:
                try:
                    result = op()
                    errors.append(f"ok:{result}")
                except ZeroDivisionError:
                    errors.append("ZDE")
                except IndexError:
                    errors.append("IE")
                except KeyError:
                    errors.append("KE")
                except ValueError:
                    errors.append("VE")
            return errors
        self._jit_test(f)

    def test_finally_in_loop(self):
        def f(n):
            result = []
            for i in range(n):
                try:
                    if i == 2:
                        continue
                    if i == 4:
                        break
                    result.append(i)
                finally:
                    result.append(f"fin:{i}")
            return result
        self._jit_test(f, 6)

    def test_return_in_finally(self):
        """finally return overrides try return."""
        def f(use_finally_return):
            try:
                return "try"
            finally:
                if use_finally_return:
                    return "finally"
        self._jit_test(f, True)
        self._jit_test(f, False)

    def test_exception_in_finally(self):
        def f():
            result = []
            try:
                try:
                    raise ValueError("orig")
                finally:
                    result.append("finally1")
                    # This replaces the ValueError
                    raise TypeError("from finally")
            except TypeError as e:
                result.append(f"caught:{e}")
            except ValueError as e:
                result.append(f"orig:{e}")
            return result
        self._jit_test(f)

    def test_loop_with_else_and_exception(self):
        def f(lst, target):
            try:
                for x in lst:
                    if x == target:
                        break
                    if x < 0:
                        raise ValueError(f"negative: {x}")
                else:
                    return "not found, loop completed"
                return f"found: {target}"
            except ValueError as e:
                return f"error: {e}"
        self._jit_test(f, [1, 2, 3, 4], 3)
        self._jit_test(f, [1, 2, 3, 4], 99)
        self._jit_test(f, [1, -2, 3], 3)

    def test_truthiness_control_flow(self):
        """Test various truthy/falsy values in control flow."""
        def f(val):
            if val:
                return "truthy"
            return "falsy"
        for v in [0, 1, -1, 0.0, 0.1, "", "x", [], [1],
                  {}, {"a": 1}, None, True, False, (), (0,)]:
            self._jit_test(f, v)

    def test_is_none_control_flow(self):
        def f(x):
            if x is None:
                return "none"
            elif x is not None:
                return "not none"
        self._jit_test(f, None)
        self._jit_test(f, 0)
        self._jit_test(f, "")

    def test_in_operator_control_flow(self):
        def f(val, container):
            if val in container:
                return "found"
            else:
                return "missing"
        self._jit_test(f, 3, [1, 2, 3, 4])
        self._jit_test(f, 5, [1, 2, 3, 4])
        self._jit_test(f, "a", {"a": 1, "b": 2})
        self._jit_test(f, "c", {"a": 1, "b": 2})

    def test_not_in_operator(self):
        def f(val, lst):
            if val not in lst:
                return "absent"
            return "present"
        self._jit_test(f, 3, [1, 2, 4, 5])
        self._jit_test(f, 3, [1, 2, 3, 4])

    def test_isinstance_dispatch(self):
        def f(val):
            if isinstance(val, int):
                return f"int:{val}"
            elif isinstance(val, str):
                return f"str:{val}"
            elif isinstance(val, list):
                return f"list:{len(val)}"
            else:
                return "other"
        self._jit_test(f, 42)
        self._jit_test(f, "hello")
        self._jit_test(f, [1, 2])
        self._jit_test(f, 3.14)


if __name__ == "__main__":
    unittest.main()
