"""Phoenix JIT Auto-Compilation Tests.

Tests the interpreter specialization -> JIT handoff boundary.
Each test defines a function, warms it up past the compilation threshold
(1000 calls), then verifies the JIT-compiled result matches the
interpreter result.

CRITICAL DIFFERENCE from force_compile tests: these tests use the
auto-compilation path (function called 1100 times) rather than
cinderjit.force_compile().

Run with: ./python -m test test_phoenix_jit_autocompile
"""

import math
import sys
import unittest

try:
    import _cinderx
    import cinderjit
    HAS_JIT = True
except ImportError:
    HAS_JIT = False

WARMUP = 1100  # threshold=1000, add margin


@unittest.skipUnless(HAS_JIT, "requires JIT")
class TestJitAutoCompile(unittest.TestCase):

    def _auto_test(self, func, *args):
        """Warmup func to trigger auto-compilation, verify result."""
        expected = func(*args)
        for _ in range(WARMUP):
            func(*args)
        result = func(*args)
        self.assertEqual(expected, result,
                         f"{func.__name__}: expected={expected!r}, got={result!r}")
        return result

    def _auto_test_float(self, func, *args, tol=1e-9):
        """Like _auto_test but with float tolerance."""
        expected = func(*args)
        for _ in range(WARMUP):
            func(*args)
        result = func(*args)
        if isinstance(expected, float) and isinstance(result, float):
            self.assertAlmostEqual(expected, result, places=9,
                                   msg=f"{func.__name__}: expected={expected!r}, got={result!r}")
        else:
            self.assertEqual(expected, result,
                             f"{func.__name__}: expected={expected!r}, got={result!r}")
        return result

    # ===================================================================
    # 1. Integer arithmetic (10 tests)
    # ===================================================================

    def test_int_add(self):
        def f(a, b):
            return a + b
        self._auto_test(f, 17, 25)

    def test_int_sub(self):
        def f(a, b):
            return a - b
        self._auto_test(f, 100, 37)

    def test_int_mul(self):
        def f(a, b):
            return a * b
        self._auto_test(f, 13, 19)

    def test_int_div(self):
        def f(a, b):
            return a // b
        self._auto_test(f, 100, 7)

    def test_int_mod(self):
        def f(a, b):
            return a % b
        self._auto_test(f, 100, 7)

    def test_int_power(self):
        def f(a, b):
            return a ** b
        self._auto_test(f, 2, 20)

    def test_int_bitwise(self):
        def f(a, b):
            return (a & b) | (a ^ b)
        self._auto_test(f, 0xFF00, 0x0FF0)

    def test_int_augmented(self):
        def f(a, b):
            x = a
            x += b
            x *= 2
            x -= 1
            return x
        self._auto_test(f, 10, 20)

    def test_int_overflow(self):
        def f(n):
            return n * n * n
        self._auto_test(f, 10**6)

    def test_int_mixed_ops(self):
        def f(a, b, c):
            return (a + b) * c - (a // c) + (b % c)
        self._auto_test(f, 17, 23, 5)

    # ===================================================================
    # 2. Float arithmetic (8 tests)
    # ===================================================================

    def test_float_add(self):
        def f(a, b):
            return a + b
        self._auto_test_float(f, 1.5, 2.7)

    def test_float_sub(self):
        def f(a, b):
            return a - b
        self._auto_test_float(f, 10.0, 3.14)

    def test_float_mul(self):
        def f(a, b):
            return a * b
        self._auto_test_float(f, 3.14, 2.0)

    def test_float_div(self):
        def f(a, b):
            return a / b
        self._auto_test_float(f, 22.0, 7.0)

    def test_float_power(self):
        def f(a, b):
            return a ** b
        self._auto_test_float(f, 2.5, 3.0)

    def test_float_mixed_int(self):
        def f(a, b):
            return a * 2 + b / 3
        self._auto_test_float(f, 1.5, 9.0)

    def test_float_accumulation(self):
        def f(n):
            s = 0.0
            for i in range(n):
                s += 1.0 / (i + 1)
            return s
        self._auto_test_float(f, 20)

    def test_float_fma(self):
        def f(a, b, c):
            return a * b + c
        self._auto_test_float(f, 3.14, 2.71, 1.41)

    # ===================================================================
    # 3. Comparisons (8 tests)
    # ===================================================================

    def test_cmp_eq(self):
        def f(a, b):
            return a == b
        self._auto_test(f, 42, 42)

    def test_cmp_ne(self):
        def f(a, b):
            return a != b
        self._auto_test(f, 42, 43)

    def test_cmp_lt(self):
        def f(a, b):
            return a < b
        self._auto_test(f, 3, 7)

    def test_cmp_gt(self):
        def f(a, b):
            return a > b
        self._auto_test(f, 7, 3)

    def test_cmp_chained(self):
        def f(a, b, c):
            return a < b < c
        self._auto_test(f, 1, 5, 10)

    def test_cmp_is_none(self):
        def f(x):
            return x is None
        self._auto_test(f, None)

    def test_cmp_isinstance(self):
        def f(x):
            return isinstance(x, int)
        self._auto_test(f, 42)

    def test_cmp_truthiness(self):
        def f(x):
            if x:
                return "truthy"
            return "falsy"
        self._auto_test(f, 1)

    # ===================================================================
    # 4. Control flow (12 tests)
    # ===================================================================

    def test_cf_if_else(self):
        def f(x):
            if x > 0:
                return "pos"
            elif x == 0:
                return "zero"
            else:
                return "neg"
        self._auto_test(f, 5)

    def test_cf_for_range(self):
        def f(n):
            s = 0
            for i in range(n):
                s += i
            return s
        self._auto_test(f, 10)

    def test_cf_for_list(self):
        def f(lst):
            s = 0
            for x in lst:
                s += x
            return s
        self._auto_test(f, (1, 2, 3, 4, 5))

    def test_cf_while(self):
        def f(n):
            s = 0
            while n > 0:
                s += n
                n -= 1
            return s
        self._auto_test(f, 10)

    def test_cf_break(self):
        def f(n):
            s = 0
            for i in range(n):
                if i == 5:
                    break
                s += i
            return s
        self._auto_test(f, 20)

    def test_cf_continue(self):
        def f(n):
            s = 0
            for i in range(n):
                if i % 2 == 0:
                    continue
                s += i
            return s
        self._auto_test(f, 10)

    def test_cf_nested_loops(self):
        def f(n):
            s = 0
            for i in range(n):
                for j in range(n):
                    s += i * j
            return s
        self._auto_test(f, 5)

    def test_cf_loop_else(self):
        def f(n):
            for i in range(n):
                if i == 100:
                    break
            else:
                return "completed"
            return "broken"
        self._auto_test(f, 10)

    def test_cf_try_except(self):
        def f(a, b):
            try:
                return a // b
            except ZeroDivisionError:
                return -1
        self._auto_test(f, 10, 3)

    def test_cf_try_finally(self):
        def f(x):
            result = 0
            try:
                result = x * 2
            finally:
                result += 1
            return result
        self._auto_test(f, 5)

    def test_cf_ternary(self):
        def f(x):
            return "even" if x % 2 == 0 else "odd"
        self._auto_test(f, 7)

    def test_cf_short_circuit(self):
        def f(a, b):
            return (a > 0 and b > 0) or (a < -10)
        self._auto_test(f, 5, 3)

    # ===================================================================
    # 5. Container ops (15 tests)
    # ===================================================================

    def test_cont_list_build(self):
        def f(a, b, c):
            return [a, b, c]
        self._auto_test(f, 1, 2, 3)

    def test_cont_list_comprehension(self):
        def f(n):
            return [i * i for i in range(n)]
        self._auto_test(f, 8)

    def test_cont_dict_build(self):
        def f(k, v):
            return {k: v, "x": 10}
        self._auto_test(f, "key", 42)

    def test_cont_dict_comprehension(self):
        def f(n):
            return {i: i * i for i in range(n)}
        self._auto_test(f, 6)

    def test_cont_set_build(self):
        def f(a, b, c):
            return {a, b, c}
        self._auto_test(f, 1, 2, 3)

    def test_cont_set_comprehension(self):
        def f(n):
            return {i % 5 for i in range(n)}
        self._auto_test(f, 10)

    def test_cont_tuple_build(self):
        def f(a, b, c):
            return (a, b, c)
        self._auto_test(f, 10, 20, 30)

    def test_cont_list_append(self):
        def f(n):
            lst = []
            for i in range(n):
                lst.append(i)
            return lst
        self._auto_test(f, 5)

    def test_cont_dict_get_set(self):
        def f(k, v):
            d = {}
            d[k] = v
            return d.get(k, -1)
        self._auto_test(f, "hello", 99)

    def test_cont_set_add(self):
        def f(n):
            s = set()
            for i in range(n):
                s.add(i % 3)
            return sorted(s)
        self._auto_test(f, 10)

    def test_cont_string_join(self):
        def f(parts):
            return ",".join(parts)
        self._auto_test(f, ("a", "b", "c"))

    def test_cont_string_format(self):
        def f(name, age):
            return "Name: {}, Age: {}".format(name, age)
        self._auto_test(f, "Alice", 30)

    def test_cont_fstring(self):
        def f(x, y):
            return f"{x} + {y} = {x + y}"
        self._auto_test(f, 3, 4)

    def test_cont_in_operator(self):
        def f(x, lst):
            return x in lst
        self._auto_test(f, 3, (1, 2, 3, 4, 5))

    def test_cont_subscript(self):
        def f(lst, idx):
            return lst[idx]
        self._auto_test(f, (10, 20, 30, 40, 50), 2)

    # ===================================================================
    # 6. Function calls (12 tests)
    # ===================================================================

    def test_func_simple_call(self):
        def helper(x):
            return x + 1

        def f(x):
            return helper(x)
        self._auto_test(f, 10)

    def test_func_args(self):
        def helper(a, b, c):
            return a + b + c

        def f(x):
            return helper(x, x + 1, x + 2)
        self._auto_test(f, 5)

    def test_func_kwargs(self):
        def helper(a=0, b=0, c=0):
            return a * 100 + b * 10 + c

        def f(x):
            return helper(a=x, c=x + 2)
        self._auto_test(f, 3)

    def test_func_star_args(self):
        def helper(*args):
            return sum(args)

        def f(x):
            return helper(x, x + 1, x + 2, x + 3)
        self._auto_test(f, 1)

    def test_func_star_kwargs(self):
        def helper(**kwargs):
            return sorted(kwargs.items())

        def f(x):
            return helper(a=x, b=x + 1)
        self._auto_test(f, 10)

    def test_func_nested_call(self):
        def add(a, b):
            return a + b

        def mul(a, b):
            return a * b

        def f(x, y):
            return add(mul(x, y), mul(x + 1, y + 1))
        self._auto_test(f, 3, 4)

    def test_func_method_call(self):
        def f(s):
            return s.upper()
        self._auto_test(f, "hello")

    def test_func_builtin_len(self):
        def f(lst):
            return len(lst)
        self._auto_test(f, (1, 2, 3, 4, 5))

    def test_func_builtin_minmax(self):
        def f(lst):
            return (min(lst), max(lst))
        self._auto_test(f, (3, 1, 4, 1, 5, 9))

    def test_func_builtin_sum(self):
        def f(lst):
            return sum(lst)
        self._auto_test(f, (1, 2, 3, 4, 5))

    def test_func_lambda(self):
        def f(x):
            double = lambda a: a * 2
            return double(x) + double(x + 1)
        self._auto_test(f, 5)

    def test_func_closure(self):
        def make_adder(n):
            def add(x):
                return x + n
            return add

        adder5 = make_adder(5)

        def f(x):
            return adder5(x)
        self._auto_test(f, 10)

    # ===================================================================
    # 7. Attribute access (8 tests)
    # ===================================================================

    def test_attr_getattr(self):
        class Obj:
            def __init__(self):
                self.x = 42
        obj = Obj()

        def f(o):
            return o.x
        self._auto_test(f, obj)

    def test_attr_setattr(self):
        class Obj:
            def __init__(self):
                self.x = 0
        obj = Obj()

        def f(o, v):
            o.x = v
            return o.x
        self._auto_test(f, obj, 99)

    def test_attr_property(self):
        class Obj:
            def __init__(self, v):
                self._v = v
            @property
            def value(self):
                return self._v * 2
        obj = Obj(21)

        def f(o):
            return o.value
        self._auto_test(f, obj)

    def test_attr_method_lookup(self):
        class Obj:
            def compute(self, x):
                return x * x
        obj = Obj()

        def f(o, x):
            return o.compute(x)
        self._auto_test(f, obj, 7)

    def test_attr_class_attribute(self):
        class Obj:
            CLASS_VAL = 100
        obj = Obj()

        def f(o):
            return o.CLASS_VAL
        self._auto_test(f, obj)

    def test_attr_instance_dict(self):
        class Obj:
            pass
        obj = Obj()
        obj.a = 1
        obj.b = 2

        def f(o):
            return o.a + o.b
        self._auto_test(f, obj)

    def test_attr_slots(self):
        class Obj:
            __slots__ = ('x', 'y')
            def __init__(self, x, y):
                self.x = x
                self.y = y
        obj = Obj(3, 4)

        def f(o):
            return o.x + o.y
        self._auto_test(f, obj)

    def test_attr_super(self):
        class Base:
            def val(self):
                return 10
        class Child(Base):
            def val(self):
                return super().val() + 5
        obj = Child()

        def f(o):
            return o.val()
        self._auto_test(f, obj)

    # ===================================================================
    # 8. Generators (8 tests)
    # ===================================================================

    def test_gen_simple_yield(self):
        def f(n):
            def gen(n):
                for i in range(n):
                    yield i
            return list(gen(n))
        self._auto_test(f, 5)

    def test_gen_yield_in_loop(self):
        def f(n):
            def gen(n):
                s = 0
                for i in range(n):
                    s += i
                    yield s
            return list(gen(n))
        self._auto_test(f, 6)

    def test_gen_yield_from_list(self):
        def f(lst):
            def gen(lst):
                yield from lst
            return list(gen(lst))
        self._auto_test(f, (10, 20, 30))

    def test_gen_expression(self):
        def f(n):
            return sum(i * i for i in range(n))
        self._auto_test(f, 10)

    def test_gen_custom_iterator(self):
        class Counter:
            def __init__(self, n):
                self.n = n
                self.i = 0
            def __iter__(self):
                return self
            def __next__(self):
                if self.i >= self.n:
                    raise StopIteration
                val = self.i
                self.i += 1
                return val

        def f(n):
            return list(Counter(n))
        self._auto_test(f, 5)

    def test_gen_enumerate(self):
        def f(lst):
            result = []
            for i, v in enumerate(lst):
                result.append((i, v))
            return result
        self._auto_test(f, ("a", "b", "c"))

    def test_gen_zip(self):
        def f(a, b):
            return list(zip(a, b))
        self._auto_test(f, (1, 2, 3), (4, 5, 6))

    def test_gen_chain(self):
        def f(a, b):
            def chain(*iterables):
                for it in iterables:
                    yield from it
            return list(chain(a, b))
        self._auto_test(f, (1, 2), (3, 4))

    # ===================================================================
    # 9. String operations (8 tests)
    # ===================================================================

    def test_str_concat(self):
        def f(a, b):
            return a + b
        self._auto_test(f, "hello", " world")

    def test_str_join(self):
        def f(parts):
            return "-".join(parts)
        self._auto_test(f, ("x", "y", "z"))

    def test_str_split(self):
        def f(s):
            return s.split(",")
        self._auto_test(f, "a,b,c,d")

    def test_str_format(self):
        def f(x, y):
            return "%s=%d" % (x, y)
        self._auto_test(f, "val", 42)

    def test_str_fstring(self):
        def f(name):
            return f"Hello, {name}!"
        self._auto_test(f, "world")

    def test_str_slice(self):
        def f(s):
            return s[1:4]
        self._auto_test(f, "abcdef")

    def test_str_upper_lower(self):
        def f(s):
            return (s.upper(), s.lower())
        self._auto_test(f, "Hello World")

    def test_str_replace(self):
        def f(s, old, new):
            return s.replace(old, new)
        self._auto_test(f, "hello world", "world", "python")

    # ===================================================================
    # 10. Unary ops (5 tests)
    # ===================================================================

    def test_unary_negate(self):
        def f(x):
            return -x
        self._auto_test(f, 42)

    def test_unary_not(self):
        def f(x):
            return not x
        self._auto_test(f, False)

    def test_unary_positive(self):
        def f(x):
            return +x
        self._auto_test(f, -7)

    def test_unary_abs(self):
        def f(x):
            return abs(x)
        self._auto_test(f, -42)

    def test_unary_invert(self):
        def f(x):
            return ~x
        self._auto_test(f, 0xFF)

    # ===================================================================
    # 11. Global/local (4 tests)
    # ===================================================================

    def test_global_load(self):
        def f():
            return WARMUP
        self._auto_test(f)

    def test_local_multi(self):
        def f(a, b, c, d):
            x = a + b
            y = c + d
            z = x * y
            return z
        self._auto_test(f, 1, 2, 3, 4)

    def test_local_delete(self):
        def f(x):
            y = x * 2
            result = y + 1
            del y
            return result
        self._auto_test(f, 10)

    def test_nonlocal_var(self):
        def f(x):
            result = 0
            def inner():
                nonlocal result
                result = x * 3
            inner()
            return result
        self._auto_test(f, 7)

    # ===================================================================
    # 12. Exception handling (6 tests)
    # ===================================================================

    def test_exc_raise_catch(self):
        def f(x):
            try:
                if x < 0:
                    raise ValueError("neg")
                return x
            except ValueError:
                return -1
        self._auto_test(f, 5)

    def test_exc_multiple_except(self):
        def f(x):
            try:
                if x == 0:
                    raise ZeroDivisionError
                if x < 0:
                    raise ValueError
                return x
            except ZeroDivisionError:
                return -1
            except ValueError:
                return -2
        self._auto_test(f, 10)

    def test_exc_finally(self):
        def f(x):
            result = 0
            try:
                result = x * 2
            except Exception:
                result = -1
            finally:
                result += 100
            return result
        self._auto_test(f, 5)

    def test_exc_nested_try(self):
        def f(x):
            try:
                try:
                    return x // 1
                except TypeError:
                    return -1
            except Exception:
                return -2
        self._auto_test(f, 42)

    def test_exc_in_loop(self):
        def f(n):
            s = 0
            for i in range(n):
                try:
                    s += 10 // (i - 2)
                except ZeroDivisionError:
                    s += 0
            return s
        self._auto_test(f, 5)

    def test_exc_with_statement(self):
        class DummyCtx:
            def __enter__(self):
                return 42
            def __exit__(self, *args):
                return False
        ctx = DummyCtx()

        def f(c):
            with c as val:
                return val
        self._auto_test(f, ctx)

    # ===================================================================
    # 13. Additional coverage - mixed patterns (14 tests)
    # ===================================================================

    def test_mixed_fibonacci(self):
        def f(n):
            if n < 2:
                return n
            a, b = 0, 1
            for _ in range(n - 1):
                a, b = b, a + b
            return b
        self._auto_test(f, 20)

    def test_mixed_factorial(self):
        def f(n):
            result = 1
            for i in range(1, n + 1):
                result *= i
            return result
        self._auto_test(f, 10)

    def test_mixed_gcd(self):
        def f(a, b):
            while b:
                a, b = b, a % b
            return a
        self._auto_test(f, 48, 18)

    def test_mixed_binary_search(self):
        def f(target):
            lst = (1, 3, 5, 7, 9, 11, 13, 15, 17, 19)
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
        self._auto_test(f, 11)

    def test_mixed_matrix_mul(self):
        def f():
            a = ((1, 2), (3, 4))
            b = ((5, 6), (7, 8))
            r00 = a[0][0] * b[0][0] + a[0][1] * b[1][0]
            r01 = a[0][0] * b[0][1] + a[0][1] * b[1][1]
            r10 = a[1][0] * b[0][0] + a[1][1] * b[1][0]
            r11 = a[1][0] * b[0][1] + a[1][1] * b[1][1]
            return ((r00, r01), (r10, r11))
        self._auto_test(f)

    def test_mixed_dict_counting(self):
        def f(items):
            counts = {}
            for item in items:
                counts[item] = counts.get(item, 0) + 1
            return sorted(counts.items())
        self._auto_test(f, ("a", "b", "a", "c", "b", "a"))

    def test_mixed_flatten(self):
        def f(nested):
            result = []
            for sub in nested:
                for item in sub:
                    result.append(item)
            return result
        self._auto_test(f, ((1, 2), (3, 4), (5, 6)))

    def test_mixed_tuple_swap(self):
        def f(a, b):
            a, b = b, a
            return (a, b)
        self._auto_test(f, 10, 20)

    def test_mixed_multiple_return(self):
        def f(x):
            if x > 100:
                return "big"
            if x > 10:
                return "medium"
            if x > 0:
                return "small"
            return "zero_or_neg"
        self._auto_test(f, 42)

    def test_mixed_string_builder(self):
        def f(n):
            parts = []
            for i in range(n):
                parts.append(str(i))
            return ",".join(parts)
        self._auto_test(f, 8)

    def test_mixed_unpacking(self):
        def f():
            a, b, c = 1, 2, 3
            x, *rest = (10, 20, 30, 40)
            return (a + b + c, x, rest)
        # rest will be a list, that's fine
        self._auto_test(f)

    def test_mixed_walrus_like(self):
        def f(data):
            # Simulate walrus operator pattern
            result = []
            for x in data:
                y = x * 2
                if y > 5:
                    result.append(y)
            return result
        self._auto_test(f, (1, 2, 3, 4, 5))

    def test_mixed_nested_comprehension(self):
        def f(n):
            return [i + j for i in range(n) for j in range(n) if (i + j) % 2 == 0]
        self._auto_test(f, 4)

    def test_mixed_boolean_logic(self):
        def f(a, b, c):
            return (a or b) and (not c or a) and (b or c)
        self._auto_test(f, True, False, True)

    # ===================================================================
    # 14. Edge cases (6 tests)
    # ===================================================================

    def test_edge_empty_function(self):
        def f():
            pass
        result = self._auto_test(f)
        self.assertIsNone(result)

    def test_edge_return_none(self):
        def f(x):
            if x > 100:
                return x
            return None
        self._auto_test(f, 5)

    def test_edge_deep_nesting(self):
        def f(x):
            if x > 0:
                if x > 1:
                    if x > 2:
                        if x > 3:
                            return "deep"
                        return "3"
                    return "2"
                return "1"
            return "0"
        self._auto_test(f, 5)

    def test_edge_many_locals(self):
        def f(x):
            a = x + 1
            b = x + 2
            c = x + 3
            d = x + 4
            e = x + 5
            f_ = x + 6
            g = x + 7
            h = x + 8
            return a + b + c + d + e + f_ + g + h
        self._auto_test(f, 1)

    def test_edge_chained_methods(self):
        def f(s):
            return s.strip().lower().replace("a", "b")
        self._auto_test(f, "  Hello World  ")

    def test_edge_multiple_assignment(self):
        def f(x):
            a = b = c = x
            return a + b + c
        self._auto_test(f, 7)


if __name__ == "__main__":
    unittest.main()
