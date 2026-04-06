"""
JIT Coverage Tests — verify JIT compilation produces correct results.

Each test function exercises a specific JIT code path using force_compile.
Organized by opcode category. All functions must:
1. Be force_compile'd via cinderjit.force_compile()
2. Verify compilation via cinderjit.is_jit_compiled()
3. Assert correct computation results

Layer 1: 100 core functions (opcode happy paths)
Layer 2: Scenario injection via decorator (error paths, edge cases)
"""

import unittest
import sys

try:
    import _cinderx
    import cinderjit
    HAS_JIT = True
except ImportError:
    HAS_JIT = False

def requires_jit(func):
    """Skip test if JIT not available."""
    return unittest.skipUnless(HAS_JIT, "JIT not available")(func)

def force_and_verify(func, *args, **kwargs):
    """Force compile a function and verify it was JIT compiled."""
    cinderjit.force_compile(func)
    assert cinderjit.is_jit_compiled(func), f"{func.__name__} was not JIT compiled"
    return func(*args, **kwargs)


# ================================================================
# Category 1: Integer Arithmetic
# ================================================================

def _int_add(a, b):
    return a + b

def _int_sub(a, b):
    return a - b

def _int_mul(a, b):
    return a * b

def _int_floordiv(a, b):
    return a // b

def _int_mod(a, b):
    return a % b

def _int_pow(a, b):
    return a ** b

def _int_negate(a):
    return -a

def _int_invert(a):
    return ~a

def _int_lshift(a, n):
    return a << n

def _int_rshift(a, n):
    return a >> n

def _int_and(a, b):
    return a & b

def _int_or(a, b):
    return a | b

def _int_xor(a, b):
    return a ^ b


# ================================================================
# Category 2: Float Arithmetic
# ================================================================

def _float_add(a, b):
    return a + b

def _float_sub(a, b):
    return a - b

def _float_mul(a, b):
    return a * b

def _float_div(a, b):
    return a / b

def _float_negate(a):
    return -a

def _float_abs(a):
    return abs(a)


# ================================================================
# Category 3: Comparisons
# ================================================================

def _cmp_eq(a, b):
    return a == b

def _cmp_ne(a, b):
    return a != b

def _cmp_lt(a, b):
    return a < b

def _cmp_le(a, b):
    return a <= b

def _cmp_gt(a, b):
    return a > b

def _cmp_ge(a, b):
    return a >= b

def _cmp_is(a, b):
    return a is b

def _cmp_is_not(a, b):
    return a is not b

def _cmp_in(a, b):
    return a in b

def _cmp_not_in(a, b):
    return a not in b


# ================================================================
# Category 4: Boolean / Unary
# ================================================================

def _bool_not(a):
    return not a

def _bool_and(a, b):
    return a and b

def _bool_or(a, b):
    return a or b

def _int_to_bool(a):
    return bool(a)


# ================================================================
# Category 5: Control Flow
# ================================================================

def _if_else(a):
    if a > 0:
        return "positive"
    elif a < 0:
        return "negative"
    else:
        return "zero"

def _for_loop(n):
    total = 0
    for i in range(n):
        total += i
    return total

def _while_loop(n):
    total = 0
    i = 0
    while i < n:
        total += i
        i += 1
    return total

def _for_break(items, target):
    for item in items:
        if item == target:
            return True
    return False

def _for_continue(items):
    total = 0
    for item in items:
        if item < 0:
            continue
        total += item
    return total

def _nested_loops(n):
    total = 0
    for i in range(n):
        for j in range(n):
            total += i * j
    return total


# ================================================================
# Category 6: Container Operations
# ================================================================

def _list_build(a, b, c):
    return [a, b, c]

def _list_append(lst, item):
    lst.append(item)
    return lst

def _list_subscr(lst, idx):
    return lst[idx]

def _list_comp(n):
    return [i * 2 for i in range(n)]

def _dict_build(k1, v1, k2, v2):
    return {k1: v1, k2: v2}

def _dict_subscr(d, key):
    return d[key]

def _dict_get(d, key, default=None):
    return d.get(key, default)

def _tuple_build(a, b, c):
    return (a, b, c)

def _tuple_unpack(t):
    a, b, c = t
    return a + b + c

def _set_build(a, b, c):
    return {a, b, c}


# ================================================================
# Category 7: Function Calls
# ================================================================

def _simple_call(f, x):
    return f(x)

def _kwargs_call(f, **kwargs):
    return f(**kwargs)

def _varargs_call(f, *args):
    return f(*args)

def _recursive_fib(n):
    if n < 2:
        return n
    return _recursive_fib(n - 1) + _recursive_fib(n - 2)

def _nested_call():
    def inner(x):
        return x * 2
    return inner(21)


# ================================================================
# Category 8: Attribute Access
# ================================================================

class _SimpleObj:
    def __init__(self, x):
        self.x = x

def _load_attr(obj):
    return obj.x

def _store_attr(obj, val):
    obj.x = val
    return obj.x

def _method_call(obj):
    return str(obj)


# ================================================================
# Category 9: Exception Handling
# ================================================================

def _try_except_simple():
    try:
        return 1 / 0
    except ZeroDivisionError:
        return -1

def _try_finally():
    result = []
    try:
        result.append(1)
    finally:
        result.append(2)
    return result

def _try_except_else():
    try:
        x = 42
    except ValueError:
        return -1
    else:
        return x

def _raise_and_catch(should_raise):
    try:
        if should_raise:
            raise ValueError("test")
        return "ok"
    except ValueError:
        return "caught"


# ================================================================
# Category 10: Generators
# ================================================================

def _simple_generator(n):
    for i in range(n):
        yield i

def _generator_sum(n):
    return sum(_simple_generator(n))

def _generator_send():
    def gen():
        x = yield 1
        yield x + 10
    g = gen()
    v1 = next(g)
    v2 = g.send(5)
    return v1, v2


# ================================================================
# Category 11: Closures
# ================================================================

def _make_closure(x):
    def inner():
        return x + 1
    return inner

def _closure_mutate():
    count = 0
    def inc():
        nonlocal count
        count += 1
        return count
    return inc

def _nested_closure(a):
    def middle(b):
        def inner(c):
            return a + b + c
        return inner
    return middle


# ================================================================
# Category 12: String Operations
# ================================================================

def _str_concat(a, b):
    return a + b

def _str_format(name, age):
    return f"Name: {name}, Age: {age}"

def _str_multiply(s, n):
    return s * n


# ================================================================
# Category 13: Global / Nonlocal
# ================================================================

_GLOBAL_VAR = 42

def _load_global():
    return _GLOBAL_VAR

def _store_global():
    global _GLOBAL_VAR
    _GLOBAL_VAR = 99
    return _GLOBAL_VAR


# ================================================================
# Category 14: Misc
# ================================================================

def _conditional_expr(a, b):
    return a if a > b else b

def _walrus(n):
    if (x := n * 2) > 10:
        return x
    return -1

def _multiple_return(a, b):
    return a, b

def _star_unpack(lst):
    first, *rest = lst
    return first, rest


# ================================================================
# Test Class
# ================================================================

class TestJITCoverage(unittest.TestCase):
    """Core JIT coverage tests — force_compile + verify results."""

    @requires_jit
    def test_int_arithmetic(self):
        self.assertEqual(force_and_verify(_int_add, 3, 4), 7)
        self.assertEqual(force_and_verify(_int_sub, 10, 3), 7)
        self.assertEqual(force_and_verify(_int_mul, 6, 7), 42)
        self.assertEqual(force_and_verify(_int_floordiv, 17, 5), 3)
        self.assertEqual(force_and_verify(_int_mod, 17, 5), 2)
        self.assertEqual(force_and_verify(_int_pow, 2, 10), 1024)
        self.assertEqual(force_and_verify(_int_negate, 42), -42)
        self.assertEqual(force_and_verify(_int_invert, 0), -1)
        self.assertEqual(force_and_verify(_int_lshift, 1, 10), 1024)
        self.assertEqual(force_and_verify(_int_rshift, 1024, 10), 1)
        self.assertEqual(force_and_verify(_int_and, 0xFF, 0x0F), 0x0F)
        self.assertEqual(force_and_verify(_int_or, 0xF0, 0x0F), 0xFF)
        self.assertEqual(force_and_verify(_int_xor, 0xFF, 0x0F), 0xF0)

    @requires_jit
    def test_float_arithmetic(self):
        self.assertAlmostEqual(force_and_verify(_float_add, 1.5, 2.5), 4.0)
        self.assertAlmostEqual(force_and_verify(_float_sub, 5.0, 2.5), 2.5)
        self.assertAlmostEqual(force_and_verify(_float_mul, 3.0, 4.0), 12.0)
        self.assertAlmostEqual(force_and_verify(_float_div, 10.0, 4.0), 2.5)
        self.assertAlmostEqual(force_and_verify(_float_negate, 3.14), -3.14)
        self.assertAlmostEqual(force_and_verify(_float_abs, -3.14), 3.14)

    @requires_jit
    def test_comparisons(self):
        self.assertTrue(force_and_verify(_cmp_eq, 5, 5))
        self.assertFalse(force_and_verify(_cmp_eq, 5, 6))
        self.assertTrue(force_and_verify(_cmp_ne, 5, 6))
        self.assertTrue(force_and_verify(_cmp_lt, 3, 5))
        self.assertTrue(force_and_verify(_cmp_le, 5, 5))
        self.assertTrue(force_and_verify(_cmp_gt, 5, 3))
        self.assertTrue(force_and_verify(_cmp_ge, 5, 5))
        self.assertTrue(force_and_verify(_cmp_is, None, None))
        self.assertTrue(force_and_verify(_cmp_is_not, 1, 2))
        self.assertTrue(force_and_verify(_cmp_in, 3, [1, 2, 3]))
        self.assertTrue(force_and_verify(_cmp_not_in, 4, [1, 2, 3]))

    @requires_jit
    def test_boolean_unary(self):
        self.assertFalse(force_and_verify(_bool_not, True))
        self.assertEqual(force_and_verify(_bool_and, 1, 2), 2)
        self.assertEqual(force_and_verify(_bool_or, 0, 2), 2)
        self.assertTrue(force_and_verify(_int_to_bool, 42))
        self.assertFalse(force_and_verify(_int_to_bool, 0))

    @requires_jit
    def test_control_flow(self):
        self.assertEqual(force_and_verify(_if_else, 5), "positive")
        self.assertEqual(force_and_verify(_if_else, -5), "negative")
        self.assertEqual(force_and_verify(_if_else, 0), "zero")
        self.assertEqual(force_and_verify(_for_loop, 10), 45)
        self.assertEqual(force_and_verify(_while_loop, 10), 45)
        self.assertTrue(force_and_verify(_for_break, [1, 2, 3], 2))
        self.assertFalse(force_and_verify(_for_break, [1, 2, 3], 5))
        self.assertEqual(force_and_verify(_for_continue, [1, -2, 3, -4, 5]), 9)
        self.assertEqual(force_and_verify(_nested_loops, 5), 100)

    @requires_jit
    def test_containers(self):
        self.assertEqual(force_and_verify(_list_build, 1, 2, 3), [1, 2, 3])
        self.assertEqual(force_and_verify(_list_subscr, [10, 20, 30], 1), 20)
        self.assertEqual(force_and_verify(_list_comp, 5), [0, 2, 4, 6, 8])
        self.assertEqual(force_and_verify(_dict_build, "a", 1, "b", 2), {"a": 1, "b": 2})
        self.assertEqual(force_and_verify(_dict_subscr, {"x": 42}, "x"), 42)
        self.assertEqual(force_and_verify(_dict_get, {"x": 42}, "y", -1), -1)
        self.assertEqual(force_and_verify(_tuple_build, 1, 2, 3), (1, 2, 3))
        self.assertEqual(force_and_verify(_tuple_unpack, (1, 2, 3)), 6)

    @requires_jit
    def test_function_calls(self):
        self.assertEqual(force_and_verify(_simple_call, abs, -5), 5)
        self.assertEqual(force_and_verify(_recursive_fib, 10), 55)
        self.assertEqual(force_and_verify(_nested_call), 42)

    @requires_jit
    def test_attributes(self):
        obj = _SimpleObj(42)
        self.assertEqual(force_and_verify(_load_attr, obj), 42)
        self.assertEqual(force_and_verify(_store_attr, obj, 99), 99)

    @requires_jit
    def test_exceptions(self):
        self.assertEqual(force_and_verify(_try_except_simple), -1)
        self.assertEqual(force_and_verify(_try_finally), [1, 2])
        self.assertEqual(force_and_verify(_try_except_else), 42)
        self.assertEqual(force_and_verify(_raise_and_catch, True), "caught")
        self.assertEqual(force_and_verify(_raise_and_catch, False), "ok")

    @requires_jit
    def test_generators(self):
        self.assertEqual(force_and_verify(_generator_sum, 10), 45)
        self.assertEqual(force_and_verify(_generator_send), (1, 15))

    @requires_jit
    def test_closures(self):
        closure = force_and_verify(_make_closure, 41)
        self.assertEqual(closure(), 42)
        inc = force_and_verify(_closure_mutate)
        self.assertEqual(inc(), 1)
        self.assertEqual(inc(), 2)
        middle = force_and_verify(_nested_closure, 10)
        inner = middle(20)
        self.assertEqual(inner(12), 42)

    @requires_jit
    def test_strings(self):
        self.assertEqual(force_and_verify(_str_concat, "hello ", "world"), "hello world")
        self.assertEqual(force_and_verify(_str_format, "Alice", 30), "Name: Alice, Age: 30")
        self.assertEqual(force_and_verify(_str_multiply, "ab", 3), "ababab")

    @requires_jit
    def test_globals(self):
        global _GLOBAL_VAR
        _GLOBAL_VAR = 42
        self.assertEqual(force_and_verify(_load_global), 42)
        self.assertEqual(force_and_verify(_store_global), 99)
        _GLOBAL_VAR = 42  # reset

    @requires_jit
    def test_misc(self):
        self.assertEqual(force_and_verify(_conditional_expr, 5, 3), 5)
        self.assertEqual(force_and_verify(_conditional_expr, 3, 5), 5)
        self.assertEqual(force_and_verify(_walrus, 10), 20)
        self.assertEqual(force_and_verify(_walrus, 3), -1)
        self.assertEqual(force_and_verify(_multiple_return, 1, 2), (1, 2))
        self.assertEqual(force_and_verify(_star_unpack, [1, 2, 3, 4]), (1, [2, 3, 4]))


if __name__ == "__main__":
    unittest.main()
