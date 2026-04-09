"""Tests for Phoenix JIT comparison, boolean, and type-checking operations.

Every test:
1. Defines a Python function exercising specific opcode(s)
2. Runs it under the interpreter first to get the expected result
3. Force-compiles via cinderjit.force_compile()
4. Runs under JIT and asserts the result matches

This catches silent codegen bugs where comparison/boolean logic in JIT-compiled
code diverges from interpreter semantics.

Run with: ./python -m test test_phoenix_jit_comparisons
"""

import sys
import unittest

try:
    import _cinderx
    import cinderjit
    HAS_JIT = True
except ImportError:
    HAS_JIT = False


# =====================================================================
# Helper classes used by multiple tests
# =====================================================================

class AlwaysEqual:
    """Object whose __eq__ always returns True."""
    def __eq__(self, other):
        return True
    def __ne__(self, other):
        return False

class NeverEqual:
    """Object whose __eq__ always returns False."""
    def __eq__(self, other):
        return False
    def __ne__(self, other):
        return True

class NotImplementedEq:
    """Object whose __eq__ returns NotImplemented."""
    def __eq__(self, other):
        return NotImplemented

class CustomBool:
    """Object with custom __bool__."""
    def __init__(self, val):
        self._val = val
    def __bool__(self):
        return self._val

class CustomContains:
    """Object with custom __contains__."""
    def __init__(self, items):
        self._items = items
    def __contains__(self, item):
        return item in self._items

class RichCompare:
    """Object with all rich comparison methods."""
    def __init__(self, val):
        self.val = val
    def __lt__(self, other):
        if isinstance(other, RichCompare):
            return self.val < other.val
        return NotImplemented
    def __le__(self, other):
        if isinstance(other, RichCompare):
            return self.val <= other.val
        return NotImplemented
    def __gt__(self, other):
        if isinstance(other, RichCompare):
            return self.val > other.val
        return NotImplemented
    def __ge__(self, other):
        if isinstance(other, RichCompare):
            return self.val >= other.val
        return NotImplemented
    def __eq__(self, other):
        if isinstance(other, RichCompare):
            return self.val == other.val
        return NotImplemented
    def __ne__(self, other):
        if isinstance(other, RichCompare):
            return self.val != other.val
        return NotImplemented

class Animal:
    pass

class Dog(Animal):
    pass

class Cat(Animal):
    pass

class GoldenRetriever(Dog):
    pass


# =====================================================================
# Test class
# =====================================================================

@unittest.skipUnless(HAS_JIT, "Phoenix JIT not available")
class TestJitComparisons(unittest.TestCase):
    """Test comparison, boolean, and type-checking operations under JIT."""

    def _jit_test(self, func, *args):
        """Run func under interpreter, force-compile, run under JIT, compare."""
        interp_result = func(*args)
        cinderjit.force_compile(func)
        self.assertTrue(
            cinderjit.is_jit_compiled(func),
            f"{func.__name__} did not compile"
        )
        jit_result = func(*args)
        self.assertEqual(
            interp_result, jit_result,
            f"{func.__name__}{args}: interpreter={interp_result!r}, JIT={jit_result!r}"
        )
        return jit_result

    # -----------------------------------------------------------------
    # 1. COMPARE_OP: ==, !=, <, <=, >, >= for int, float, str, tuple, list
    # -----------------------------------------------------------------

    def test_compare_int_eq(self):
        def f(a, b):
            return a == b
        self._jit_test(f, 5, 5)
        self._jit_test(f, 5, 6)
        self._jit_test(f, -1, -1)
        self._jit_test(f, 0, 0)

    def test_compare_int_ne(self):
        def f(a, b):
            return a != b
        self._jit_test(f, 5, 5)
        self._jit_test(f, 5, 6)

    def test_compare_int_lt(self):
        def f(a, b):
            return a < b
        self._jit_test(f, 3, 5)
        self._jit_test(f, 5, 3)
        self._jit_test(f, 5, 5)
        self._jit_test(f, -10, 10)

    def test_compare_int_le(self):
        def f(a, b):
            return a <= b
        self._jit_test(f, 3, 5)
        self._jit_test(f, 5, 5)
        self._jit_test(f, 6, 5)

    def test_compare_int_gt(self):
        def f(a, b):
            return a > b
        self._jit_test(f, 10, 5)
        self._jit_test(f, 5, 5)
        self._jit_test(f, 3, 5)

    def test_compare_int_ge(self):
        def f(a, b):
            return a >= b
        self._jit_test(f, 10, 5)
        self._jit_test(f, 5, 5)
        self._jit_test(f, 3, 5)

    def test_compare_float_all_ops(self):
        def f(a, b):
            return (a == b, a != b, a < b, a <= b, a > b, a >= b)
        self._jit_test(f, 1.5, 2.5)
        self._jit_test(f, 2.5, 2.5)
        self._jit_test(f, 3.5, 2.5)
        self._jit_test(f, -0.0, 0.0)
        self._jit_test(f, float('inf'), 1e308)
        self._jit_test(f, float('-inf'), float('inf'))

    def test_compare_float_nan(self):
        """NaN is not equal to anything, including itself."""
        def f(a, b):
            return (a == b, a != b, a < b, a <= b, a > b, a >= b)
        nan = float('nan')
        self._jit_test(f, nan, nan)
        self._jit_test(f, nan, 0.0)
        self._jit_test(f, 0.0, nan)

    def test_compare_string(self):
        def f(a, b):
            return (a == b, a != b, a < b, a <= b, a > b, a >= b)
        self._jit_test(f, "hello", "hello")
        self._jit_test(f, "abc", "abd")
        self._jit_test(f, "xyz", "abc")
        self._jit_test(f, "", "a")
        self._jit_test(f, "a", "")

    def test_compare_tuple(self):
        def f(a, b):
            return (a == b, a != b, a < b, a <= b, a > b, a >= b)
        self._jit_test(f, (1, 2, 3), (1, 2, 3))
        self._jit_test(f, (1, 2), (1, 3))
        self._jit_test(f, (1, 2, 3), (1, 2))
        self._jit_test(f, (), ())
        self._jit_test(f, (1,), (2,))

    def test_compare_list(self):
        def f(a, b):
            return (a == b, a != b, a < b, a <= b, a > b, a >= b)
        self._jit_test(f, [1, 2, 3], [1, 2, 3])
        self._jit_test(f, [1, 2], [1, 3])
        self._jit_test(f, [1], [1, 2])
        self._jit_test(f, [], [])

    # -----------------------------------------------------------------
    # 2. IS / IS NOT
    # -----------------------------------------------------------------

    def test_is_identity(self):
        def f(a, b):
            return a is b
        sentinel = object()
        self._jit_test(f, sentinel, sentinel)
        self._jit_test(f, sentinel, object())

    def test_is_not_identity(self):
        def f(a, b):
            return a is not b
        sentinel = object()
        self._jit_test(f, sentinel, sentinel)
        self._jit_test(f, sentinel, object())

    def test_is_none(self):
        def f(x):
            return x is None
        self._jit_test(f, None)
        self._jit_test(f, 0)
        self._jit_test(f, "")
        self._jit_test(f, False)

    def test_is_not_none(self):
        def f(x):
            return x is not None
        self._jit_test(f, None)
        self._jit_test(f, 42)
        self._jit_test(f, "hello")

    # -----------------------------------------------------------------
    # 3. IN / NOT IN
    # -----------------------------------------------------------------

    def test_in_list(self):
        def f(x, lst):
            return x in lst
        self._jit_test(f, 3, [1, 2, 3, 4, 5])
        self._jit_test(f, 99, [1, 2, 3])
        self._jit_test(f, "a", ["a", "b"])

    def test_not_in_list(self):
        def f(x, lst):
            return x not in lst
        self._jit_test(f, 3, [1, 2, 3])
        self._jit_test(f, 99, [1, 2, 3])

    def test_in_tuple(self):
        def f(x, t):
            return x in t
        self._jit_test(f, 2, (1, 2, 3))
        self._jit_test(f, 99, (1, 2, 3))

    def test_in_dict(self):
        def f(x, d):
            return x in d
        self._jit_test(f, "a", {"a": 1, "b": 2})
        self._jit_test(f, "c", {"a": 1, "b": 2})

    def test_in_set(self):
        def f(x, s):
            return x in s
        self._jit_test(f, 3, {1, 2, 3, 4})
        self._jit_test(f, 99, {1, 2, 3})

    def test_in_string(self):
        def f(sub, s):
            return sub in s
        self._jit_test(f, "ell", "hello")
        self._jit_test(f, "xyz", "hello")
        self._jit_test(f, "", "hello")

    def test_in_range(self):
        def f(x, r):
            return x in r
        self._jit_test(f, 5, range(10))
        self._jit_test(f, 10, range(10))
        self._jit_test(f, -1, range(10))
        self._jit_test(f, 6, range(0, 20, 3))

    # -----------------------------------------------------------------
    # 4. Chained comparisons
    # -----------------------------------------------------------------

    def test_chained_lt(self):
        def f(a, b, c):
            return a < b < c
        self._jit_test(f, 1, 2, 3)
        self._jit_test(f, 1, 3, 2)
        self._jit_test(f, 3, 2, 1)
        self._jit_test(f, 1, 1, 2)

    def test_chained_le_ge(self):
        def f(a, b, c):
            return a <= b >= c
        self._jit_test(f, 1, 3, 2)
        self._jit_test(f, 3, 3, 3)
        self._jit_test(f, 4, 3, 2)

    def test_chained_eq(self):
        def f(a, b, c):
            return a == b == c
        self._jit_test(f, 1, 1, 1)
        self._jit_test(f, 1, 1, 2)
        self._jit_test(f, 1, 2, 1)

    def test_chained_four_way(self):
        def f(a, b, c, d):
            return a < b <= c < d
        self._jit_test(f, 1, 2, 3, 4)
        self._jit_test(f, 1, 2, 2, 3)
        self._jit_test(f, 1, 3, 2, 4)

    def test_chained_mixed_types(self):
        """Chained comparison with int and float."""
        def f(a, b, c):
            return a < b < c
        self._jit_test(f, 1, 2.5, 4)
        self._jit_test(f, 1.0, 2, 3.0)

    # -----------------------------------------------------------------
    # 5. Boolean operations: and, or, not
    # -----------------------------------------------------------------

    def test_bool_and(self):
        def f(a, b):
            return a and b
        self._jit_test(f, True, True)
        self._jit_test(f, True, False)
        self._jit_test(f, False, True)
        self._jit_test(f, False, False)

    def test_bool_or(self):
        def f(a, b):
            return a or b
        self._jit_test(f, True, True)
        self._jit_test(f, True, False)
        self._jit_test(f, False, True)
        self._jit_test(f, False, False)

    def test_bool_not(self):
        def f(x):
            return not x
        self._jit_test(f, True)
        self._jit_test(f, False)
        self._jit_test(f, 0)
        self._jit_test(f, 1)
        self._jit_test(f, "")
        self._jit_test(f, "hi")
        self._jit_test(f, None)
        self._jit_test(f, [])
        self._jit_test(f, [1])

    def test_and_returns_value(self):
        """'and' returns the first falsy or the last truthy operand."""
        def f(a, b):
            return a and b
        self._jit_test(f, 0, 42)
        self._jit_test(f, "hello", "world")
        self._jit_test(f, "", "world")
        self._jit_test(f, None, 5)
        self._jit_test(f, 1, 0)

    def test_or_returns_value(self):
        """'or' returns the first truthy or the last falsy operand."""
        def f(a, b):
            return a or b
        self._jit_test(f, 0, 42)
        self._jit_test(f, "hello", "world")
        self._jit_test(f, "", "world")
        self._jit_test(f, None, 0)
        self._jit_test(f, 0, "")

    def test_compound_boolean(self):
        def f(a, b, c):
            return (a or b) and c
        self._jit_test(f, False, True, True)
        self._jit_test(f, False, False, True)
        self._jit_test(f, True, False, False)

    def test_nested_boolean(self):
        def f(a, b, c, d):
            return (a and b) or (c and d)
        self._jit_test(f, 1, 2, 3, 4)
        self._jit_test(f, 0, 2, 3, 4)
        self._jit_test(f, 0, 0, 0, 4)
        self._jit_test(f, 0, 0, 0, 0)

    # -----------------------------------------------------------------
    # 6. Short-circuit evaluation
    # -----------------------------------------------------------------

    def test_short_circuit_and(self):
        """'and' must not evaluate RHS when LHS is falsy."""
        def f(log):
            def side_effect():
                log.append("evaluated")
                return True
            result = False and side_effect()
            return (result, list(log))
        self._jit_test(f, [])

    def test_short_circuit_or(self):
        """'or' must not evaluate RHS when LHS is truthy."""
        def f(log):
            def side_effect():
                log.append("evaluated")
                return False
            result = True or side_effect()
            return (result, list(log))
        self._jit_test(f, [])

    def test_short_circuit_and_evaluates_rhs(self):
        """'and' evaluates RHS when LHS is truthy."""
        def f():
            log = []
            def side_effect():
                log.append("evaluated")
                return 42
            result = True and side_effect()
            return (result, log)
        self._jit_test(f)

    def test_short_circuit_or_evaluates_rhs(self):
        """'or' evaluates RHS when LHS is falsy."""
        def f():
            log = []
            def side_effect():
                log.append("evaluated")
                return 99
            result = False or side_effect()
            return (result, log)
        self._jit_test(f)

    def test_short_circuit_chained_comparison(self):
        """Chained comparison a < b < c must not evaluate c if a < b is False."""
        def f():
            log = []
            class Logged:
                def __init__(self, val):
                    self.val = val
                def __lt__(self, other):
                    log.append(f"{self.val}<{other.val}")
                    return self.val < other.val
            a, b, c = Logged(5), Logged(3), Logged(10)
            result = a < b < c
            return (result, log)
        self._jit_test(f)

    # -----------------------------------------------------------------
    # 7. Truthiness
    # -----------------------------------------------------------------

    def test_truthiness(self):
        def f(x):
            return bool(x)
        for val in [0, 1, -1, 0.0, 1.5, "", "hi", [], [1], {}, {1: 2},
                    set(), {1}, None, True, False, (), (0,)]:
            self._jit_test(f, val)

    def test_truthiness_in_if(self):
        """Truthiness via if statement rather than bool()."""
        def f(x):
            if x:
                return "truthy"
            return "falsy"
        for val in [0, 1, "", "hi", [], [1], None, True, False]:
            self._jit_test(f, val)

    # -----------------------------------------------------------------
    # 8. isinstance()
    # -----------------------------------------------------------------

    def test_isinstance_single(self):
        def f(x, t):
            return isinstance(x, t)
        self._jit_test(f, 42, int)
        self._jit_test(f, "hi", int)
        self._jit_test(f, "hi", str)
        self._jit_test(f, 3.14, float)
        self._jit_test(f, [], list)
        self._jit_test(f, (), tuple)
        self._jit_test(f, True, bool)
        self._jit_test(f, True, int)  # bool is subclass of int

    def test_isinstance_tuple_of_types(self):
        def f(x, types):
            return isinstance(x, types)
        self._jit_test(f, 42, (int, str))
        self._jit_test(f, "hi", (int, str))
        self._jit_test(f, 3.14, (int, str))
        self._jit_test(f, [], (list, tuple))

    def test_isinstance_inheritance(self):
        def f(obj):
            return (
                isinstance(obj, Animal),
                isinstance(obj, Dog),
                isinstance(obj, Cat),
                isinstance(obj, GoldenRetriever),
            )
        self._jit_test(f, Animal())
        self._jit_test(f, Dog())
        self._jit_test(f, Cat())
        self._jit_test(f, GoldenRetriever())

    # -----------------------------------------------------------------
    # 9. issubclass()
    # -----------------------------------------------------------------

    def test_issubclass_single(self):
        def f(cls, parent):
            return issubclass(cls, parent)
        self._jit_test(f, Dog, Animal)
        self._jit_test(f, Cat, Animal)
        self._jit_test(f, GoldenRetriever, Dog)
        self._jit_test(f, GoldenRetriever, Animal)
        self._jit_test(f, Cat, Dog)
        self._jit_test(f, Animal, Animal)

    def test_issubclass_tuple(self):
        def f(cls, parents):
            return issubclass(cls, parents)
        self._jit_test(f, Dog, (Animal, Cat))
        self._jit_test(f, Cat, (Dog, int))
        self._jit_test(f, bool, (int, float))

    # -----------------------------------------------------------------
    # 10. type() comparison
    # -----------------------------------------------------------------

    def test_type_is(self):
        def f(x):
            return type(x) is int
        self._jit_test(f, 42)
        self._jit_test(f, "hi")
        self._jit_test(f, True)  # type(True) is bool, not int

    def test_type_eq(self):
        def f(x):
            return type(x) == str
        self._jit_test(f, "hello")
        self._jit_test(f, 42)
        self._jit_test(f, b"bytes")

    def test_type_comparison_subclass(self):
        """type() is X vs isinstance(x, X) for subclasses."""
        def f(x):
            return (type(x) is int, isinstance(x, int))
        self._jit_test(f, 42)
        self._jit_test(f, True)  # True is instance of int but type is bool

    # -----------------------------------------------------------------
    # 11. Comparison with None
    # -----------------------------------------------------------------

    def test_none_identity_in_branch(self):
        def f(x):
            if x is None:
                return "none"
            elif x is not None:
                return "something"
        self._jit_test(f, None)
        self._jit_test(f, 0)
        self._jit_test(f, "")
        self._jit_test(f, False)

    def test_none_equality(self):
        """== with None (not recommended, but must work)."""
        def f(x):
            return x == None  # noqa: E711
        self._jit_test(f, None)
        self._jit_test(f, 0)
        self._jit_test(f, "")

    # -----------------------------------------------------------------
    # 12. String comparison (lexicographic)
    # -----------------------------------------------------------------

    def test_string_lexicographic(self):
        def f(a, b):
            return (a < b, a == b, a > b)
        self._jit_test(f, "apple", "banana")
        self._jit_test(f, "banana", "apple")
        self._jit_test(f, "apple", "apple")
        self._jit_test(f, "abc", "abcd")
        self._jit_test(f, "abcd", "abc")
        self._jit_test(f, "", "")
        self._jit_test(f, "A", "a")  # uppercase < lowercase in ASCII

    # -----------------------------------------------------------------
    # 13. Tuple comparison (element-by-element)
    # -----------------------------------------------------------------

    def test_tuple_elementwise(self):
        def f(a, b):
            return (a < b, a == b, a > b, a <= b, a >= b)
        self._jit_test(f, (1, 2, 3), (1, 2, 4))
        self._jit_test(f, (1, 2, 3), (1, 2, 3))
        self._jit_test(f, (1, 3), (1, 2))
        self._jit_test(f, (1,), (1, 2))
        self._jit_test(f, (), ())

    # -----------------------------------------------------------------
    # 14. List comparison (element-by-element)
    # -----------------------------------------------------------------

    def test_list_elementwise(self):
        def f(a, b):
            return (a < b, a == b, a > b, a <= b, a >= b)
        self._jit_test(f, [1, 2, 3], [1, 2, 4])
        self._jit_test(f, [1, 2, 3], [1, 2, 3])
        self._jit_test(f, [1, 3], [1, 2])
        self._jit_test(f, [1], [1, 2])

    # -----------------------------------------------------------------
    # 15. Mixed type comparison
    # -----------------------------------------------------------------

    def test_mixed_int_float(self):
        def f(a, b):
            return (a == b, a != b, a < b, a > b)
        self._jit_test(f, 1, 1.0)
        self._jit_test(f, 2, 1.5)
        self._jit_test(f, 0, 0.0)
        self._jit_test(f, -1, -1.0)

    def test_mixed_int_bool(self):
        def f(a, b):
            return (a == b, a != b)
        self._jit_test(f, 1, True)
        self._jit_test(f, 0, False)
        self._jit_test(f, 2, True)

    # -----------------------------------------------------------------
    # 16. NotImplemented handling
    # -----------------------------------------------------------------

    def test_not_implemented_eq(self):
        """When __eq__ returns NotImplemented, Python falls back to identity."""
        def f():
            a = NotImplementedEq()
            b = NotImplementedEq()
            return (a == b, a == a, a != b, a != a)
        self._jit_test(f)

    def test_not_implemented_with_normal(self):
        """NotImplemented on one side, normal on the other."""
        def f():
            a = NotImplementedEq()
            return (a == 42, 42 == a)
        self._jit_test(f)

    # -----------------------------------------------------------------
    # 17. Rich comparison methods
    # -----------------------------------------------------------------

    def test_rich_compare_lt(self):
        def f():
            a, b = RichCompare(3), RichCompare(5)
            return (a < b, b < a, a < a)
        self._jit_test(f)

    def test_rich_compare_le(self):
        def f():
            a, b = RichCompare(3), RichCompare(5)
            return (a <= b, b <= a, a <= a)
        self._jit_test(f)

    def test_rich_compare_gt(self):
        def f():
            a, b = RichCompare(3), RichCompare(5)
            return (a > b, b > a, a > a)
        self._jit_test(f)

    def test_rich_compare_ge(self):
        def f():
            a, b = RichCompare(3), RichCompare(5)
            return (a >= b, b >= a, a >= a)
        self._jit_test(f)

    def test_rich_compare_eq_ne(self):
        def f():
            a, b = RichCompare(3), RichCompare(5)
            c = RichCompare(3)
            return (a == b, a == c, a != b, a != c)
        self._jit_test(f)

    # -----------------------------------------------------------------
    # 18. __bool__ method
    # -----------------------------------------------------------------

    def test_custom_bool_true(self):
        def f():
            obj = CustomBool(True)
            if obj:
                return "truthy"
            return "falsy"
        self._jit_test(f)

    def test_custom_bool_false(self):
        def f():
            obj = CustomBool(False)
            if obj:
                return "truthy"
            return "falsy"
        self._jit_test(f)

    def test_custom_bool_in_and_or(self):
        def f():
            t = CustomBool(True)
            fl = CustomBool(False)
            return (bool(t and fl), bool(t or fl), bool(fl and t), bool(fl or t))
        self._jit_test(f)

    def test_custom_bool_not(self):
        def f():
            return (not CustomBool(True), not CustomBool(False))
        self._jit_test(f)

    # -----------------------------------------------------------------
    # 19. __contains__ method
    # -----------------------------------------------------------------

    def test_custom_contains(self):
        def f():
            c = CustomContains([1, 2, 3])
            return (1 in c, 4 in c, 2 not in c, 5 not in c)
        self._jit_test(f)

    def test_custom_contains_string_keys(self):
        def f():
            c = CustomContains(["hello", "world"])
            return ("hello" in c, "foo" in c)
        self._jit_test(f)

    # -----------------------------------------------------------------
    # 20. all() and any()
    # -----------------------------------------------------------------

    def test_all_basic(self):
        def f(lst):
            return all(lst)
        self._jit_test(f, [True, True, True])
        self._jit_test(f, [True, False, True])
        self._jit_test(f, [])
        self._jit_test(f, [1, 2, 3])
        self._jit_test(f, [1, 0, 3])

    def test_any_basic(self):
        def f(lst):
            return any(lst)
        self._jit_test(f, [False, False, False])
        self._jit_test(f, [False, True, False])
        self._jit_test(f, [])
        self._jit_test(f, [0, 0, 0])
        self._jit_test(f, [0, 1, 0])

    def test_all_with_generator(self):
        def f(lst):
            return all(x > 0 for x in lst)
        self._jit_test(f, [1, 2, 3])
        self._jit_test(f, [1, -1, 3])
        self._jit_test(f, [])

    def test_any_with_generator(self):
        def f(lst):
            return any(x < 0 for x in lst)
        self._jit_test(f, [1, 2, 3])
        self._jit_test(f, [1, -1, 3])
        self._jit_test(f, [])

    # -----------------------------------------------------------------
    # 21. min() and max()
    # -----------------------------------------------------------------

    def test_min_basic(self):
        def f(lst):
            return min(lst)
        self._jit_test(f, [3, 1, 4, 1, 5])
        self._jit_test(f, [-10, 0, 10])
        self._jit_test(f, [42])

    def test_max_basic(self):
        def f(lst):
            return max(lst)
        self._jit_test(f, [3, 1, 4, 1, 5])
        self._jit_test(f, [-10, 0, 10])
        self._jit_test(f, [42])

    def test_min_max_with_key(self):
        def f(lst):
            return (min(lst, key=abs), max(lst, key=abs))
        self._jit_test(f, [3, -1, -4, 1, -5])
        self._jit_test(f, [-10, 5, -2])

    def test_min_max_strings(self):
        def f(lst):
            return (min(lst), max(lst))
        self._jit_test(f, ["banana", "apple", "cherry"])

    # -----------------------------------------------------------------
    # 22. sorted() with key function
    # -----------------------------------------------------------------

    def test_sorted_basic(self):
        def f(lst):
            return sorted(lst)
        self._jit_test(f, [3, 1, 4, 1, 5, 9])
        self._jit_test(f, [])
        self._jit_test(f, [42])

    def test_sorted_reverse(self):
        def f(lst):
            return sorted(lst, reverse=True)
        self._jit_test(f, [3, 1, 4, 1, 5, 9])

    def test_sorted_with_key(self):
        def f(lst):
            return sorted(lst, key=lambda x: -x)
        self._jit_test(f, [3, 1, 4, 1, 5, 9])

    def test_sorted_strings_by_length(self):
        def f(lst):
            return sorted(lst, key=len)
        self._jit_test(f, ["banana", "pie", "a", "cherry"])

    def test_sorted_tuples(self):
        def f(lst):
            return sorted(lst, key=lambda x: x[1])
        self._jit_test(f, [(1, "b"), (2, "a"), (3, "c")])

    # -----------------------------------------------------------------
    # 23. Conditional expression chains (ternary)
    # -----------------------------------------------------------------

    def test_ternary_basic(self):
        def f(x):
            return "pos" if x > 0 else "non-pos"
        self._jit_test(f, 5)
        self._jit_test(f, 0)
        self._jit_test(f, -3)

    def test_nested_ternary(self):
        def f(x):
            return "pos" if x > 0 else ("zero" if x == 0 else "neg")
        self._jit_test(f, 5)
        self._jit_test(f, 0)
        self._jit_test(f, -3)

    def test_ternary_chain(self):
        def f(x):
            return (
                "A" if x >= 90 else
                "B" if x >= 80 else
                "C" if x >= 70 else
                "D" if x >= 60 else
                "F"
            )
        for score in [95, 85, 75, 65, 55, 100, 0]:
            self._jit_test(f, score)

    def test_ternary_with_comparison_result(self):
        def f(a, b):
            return a if a > b else b
        self._jit_test(f, 10, 20)
        self._jit_test(f, 20, 10)
        self._jit_test(f, 5, 5)

    # -----------------------------------------------------------------
    # 24. Match/case with guards
    # -----------------------------------------------------------------

    def test_match_with_guard(self):
        def f(x):
            match x:
                case n if n > 100:
                    return "big"
                case n if n > 0:
                    return "small"
                case 0:
                    return "zero"
                case _:
                    return "negative"
        self._jit_test(f, 200)
        self._jit_test(f, 50)
        self._jit_test(f, 0)
        self._jit_test(f, -5)

    def test_match_with_type_guard(self):
        def f(x):
            match x:
                case int() if x > 0:
                    return "positive int"
                case int():
                    return "non-positive int"
                case str() if len(x) > 3:
                    return "long string"
                case str():
                    return "short string"
                case _:
                    return "other"
        self._jit_test(f, 42)
        self._jit_test(f, -5)
        self._jit_test(f, 0)
        self._jit_test(f, "hello")
        self._jit_test(f, "hi")
        self._jit_test(f, 3.14)

    def test_match_sequence_guard(self):
        def f(x):
            match x:
                case [a, b] if a < b:
                    return "ascending pair"
                case [a, b]:
                    return "descending pair"
                case [single]:
                    return "single"
                case _:
                    return "other"
        self._jit_test(f, [1, 2])
        self._jit_test(f, [3, 1])
        self._jit_test(f, [5])
        self._jit_test(f, [1, 2, 3])

    # -----------------------------------------------------------------
    # 25. Assert with comparison
    # -----------------------------------------------------------------

    def test_assert_passes(self):
        def f(x):
            assert x > 0, "must be positive"
            assert x < 1000, "must be < 1000"
            return x * 2
        self._jit_test(f, 5)
        self._jit_test(f, 999)

    def test_assert_fails(self):
        def f(x):
            assert x > 0, f"expected positive, got {x}"
            return x
        # Verify both interpreter and JIT raise the same error
        def check_assert(x):
            try:
                f(x)
                return None
            except AssertionError as e:
                return str(e)
        self._jit_test(check_assert, -5)
        self._jit_test(check_assert, 0)

    # -----------------------------------------------------------------
    # Additional edge cases
    # -----------------------------------------------------------------

    def test_compare_large_ints(self):
        """Large integers that don't fit in a machine word."""
        def f(a, b):
            return (a == b, a < b, a > b)
        big1 = 10**100
        big2 = 10**100 + 1
        self._jit_test(f, big1, big2)
        self._jit_test(f, big1, big1)
        self._jit_test(f, big2, big1)

    def test_compare_negative_zero_float(self):
        def f(a, b):
            return (a == b, a < b, a > b, a <= b, a >= b)
        self._jit_test(f, -0.0, 0.0)
        self._jit_test(f, 0.0, -0.0)

    def test_always_equal_object(self):
        def f():
            a = AlwaysEqual()
            return (a == 1, a == "anything", a == None)
        self._jit_test(f)

    def test_never_equal_object(self):
        def f():
            a = NeverEqual()
            return (a == 1, a == a, a != 1, a != a)
        self._jit_test(f)

    def test_comparison_in_loop(self):
        """Comparison inside a loop (hot path)."""
        def f(n):
            count = 0
            for i in range(n):
                if i % 2 == 0 and i > 10:
                    count += 1
            return count
        self._jit_test(f, 100)
        self._jit_test(f, 0)
        self._jit_test(f, 1)

    def test_comparison_as_dict_key_guard(self):
        """Typical pattern: check key existence before access."""
        def f(d, key):
            if key in d:
                return d[key]
            return "missing"
        self._jit_test(f, {"a": 1, "b": 2}, "a")
        self._jit_test(f, {"a": 1, "b": 2}, "c")

    def test_filter_with_comparison(self):
        """list comprehension with comparison filter."""
        def f(lst, threshold):
            return [x for x in lst if x > threshold]
        self._jit_test(f, [1, 5, 3, 8, 2, 9], 4)
        self._jit_test(f, [], 0)
        self._jit_test(f, [1, 2, 3], 10)

    def test_bool_accumulation(self):
        """Accumulate boolean results across iterations."""
        def f(lst):
            found_pos = False
            found_neg = False
            for x in lst:
                if x > 0:
                    found_pos = True
                if x < 0:
                    found_neg = True
            return (found_pos, found_neg)
        self._jit_test(f, [1, -2, 3, -4])
        self._jit_test(f, [1, 2, 3])
        self._jit_test(f, [-1, -2])
        self._jit_test(f, [])

    def test_comparison_return_used_as_int(self):
        """Bool result of comparison used in arithmetic."""
        def f(lst):
            return sum(x > 0 for x in lst)
        self._jit_test(f, [1, -2, 3, 0, -4, 5])
        self._jit_test(f, [])
        self._jit_test(f, [-1, -2, -3])

    def test_walrus_in_condition(self):
        """Walrus operator (:=) inside a comparison."""
        def f(values):
            results = []
            for v in values:
                if (n := v * 2) > 5:
                    results.append(n)
            return results
        self._jit_test(f, [1, 2, 3, 4, 5])

    def test_multi_type_isinstance_chain(self):
        """isinstance used in if/elif chain for dispatch."""
        def f(x):
            if isinstance(x, bool):
                return "bool"
            elif isinstance(x, int):
                return "int"
            elif isinstance(x, float):
                return "float"
            elif isinstance(x, str):
                return "str"
            elif isinstance(x, (list, tuple)):
                return "sequence"
            else:
                return "unknown"
        self._jit_test(f, True)
        self._jit_test(f, 42)
        self._jit_test(f, 3.14)
        self._jit_test(f, "hello")
        self._jit_test(f, [1, 2])
        self._jit_test(f, (1, 2))
        self._jit_test(f, {"a": 1})

    def test_none_coalesce_pattern(self):
        """Common pattern: x if x is not None else default."""
        def f(x, default):
            return x if x is not None else default
        self._jit_test(f, None, 42)
        self._jit_test(f, 0, 42)
        self._jit_test(f, "", 42)
        self._jit_test(f, False, 42)
        self._jit_test(f, "value", 42)

    def test_comparison_exception_propagation(self):
        """Comparison that raises TypeError must propagate identically."""
        def f(a, b):
            try:
                return a < b
            except TypeError as e:
                return f"TypeError: {e}"
        self._jit_test(f, 1, "two")
        self._jit_test(f, [1], {2})

    def test_eq_ne_symmetry(self):
        """a == b should be the inverse of a != b for well-behaved types."""
        def f(a, b):
            return ((a == b) != (a != b),)
        self._jit_test(f, 1, 1)
        self._jit_test(f, 1, 2)
        self._jit_test(f, "a", "a")
        self._jit_test(f, "a", "b")

    def test_chained_is_none(self):
        """Multiple None checks in one expression."""
        def f(a, b, c):
            return (a is None or b is None or c is None,
                    a is not None and b is not None and c is not None)
        self._jit_test(f, None, 1, 2)
        self._jit_test(f, 1, None, 2)
        self._jit_test(f, 1, 2, None)
        self._jit_test(f, 1, 2, 3)
        self._jit_test(f, None, None, None)

    def test_complex_predicate(self):
        """Complex boolean predicate combining multiple patterns."""
        def f(x, lo, hi, exclude):
            return (lo <= x <= hi) and (x not in exclude) and (x % 2 == 0)
        self._jit_test(f, 4, 1, 10, {3, 6, 9})
        self._jit_test(f, 6, 1, 10, {3, 6, 9})
        self._jit_test(f, 3, 1, 10, {3, 6, 9})
        self._jit_test(f, 11, 1, 10, {3, 6, 9})
        self._jit_test(f, 5, 1, 10, {3, 6, 9})

    def test_bisect_style_comparison(self):
        """Binary search pattern with comparisons."""
        def f(arr, target):
            lo, hi = 0, len(arr)
            while lo < hi:
                mid = (lo + hi) // 2
                if arr[mid] < target:
                    lo = mid + 1
                elif arr[mid] > target:
                    hi = mid
                else:
                    return mid
            return -1
        self._jit_test(f, [1, 3, 5, 7, 9, 11, 13], 7)
        self._jit_test(f, [1, 3, 5, 7, 9, 11, 13], 4)
        self._jit_test(f, [1, 3, 5, 7, 9, 11, 13], 1)
        self._jit_test(f, [1, 3, 5, 7, 9, 11, 13], 13)
        self._jit_test(f, [], 5)


if __name__ == '__main__':
    unittest.main()
